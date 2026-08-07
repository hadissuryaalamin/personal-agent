"""Which layer should the probe read?

PLAN.md M2: capture h_L per turn, write the seed dataset, run the layer sweep
across all 36 layers. Exit criterion: docs/eval.md has a layer-vs-accuracy
table and a chosen L.

    python scripts\\sweep_layers.py              sweep, print, write docs/eval.md
    python scripts\\sweep_layers.py --print-only
    python scripts\\sweep_layers.py --refresh    recompute hidden states

Hidden states for the whole dataset are computed once and cached in
`data/hidden_cache.npz`, because the sweep itself is cheap and the forward
passes are not. The cache is keyed by the dataset contents, so editing
`data/probe/*.jsonl` invalidates it automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dataset  # noqa: E402
from src.llm import prompts  # noqa: E402

CACHE = ROOT / "data" / "hidden_cache.npz"
RESULTS = ROOT / "data" / "layer_sweep.json"

TZ = ZoneInfo("Australia/Sydney")
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)

#: Regularisation for the linear probe. PLAN.md section 4: linear first, reach
#: for an MLP only if the sweep shows the linear probe is the bottleneck.
C_GRID = (0.01, 0.1, 1.0, 10.0)


def fingerprint(examples: list[dataset.Example]) -> str:
    """Identifies the cached hidden states.

    Keyed on the dataset *and* the system prompt: the cache holds activations
    from a prefix that begins with the prompt, so editing the prompt invalidates
    every vector in it just as surely as editing an utterance does. Keying on
    the dataset alone silently reuses activations from a prompt that no longer
    exists.
    """
    digest = hashlib.sha256()
    digest.update(prompts.system_fingerprint().encode())
    for example in examples:
        digest.update(example.text.encode("utf-8"))
        digest.update(str(example.label).encode())
        digest.update(json.dumps(example.history).encode())
    return digest.hexdigest()[:16]


def compute_hidden(examples: list[dataset.Example], engine):
    """One prefill per example, keeping every layer's last-token state."""
    import numpy as np

    total = len(examples)
    stack = np.zeros(
        (total, engine.info["layers"] + 1, engine.info["hidden_size"]), dtype=np.float16
    )
    started = time.perf_counter()
    for index, example in enumerate(examples):
        messages = prompts.build_messages(
            NOW, "Australia/Sydney", example.text, example.messages()
        )
        prefill = engine.prefill(engine.prefix_text(messages))
        stack[index] = prefill.hidden.astype(np.float16)
        if (index + 1) % 50 == 0 or index + 1 == total:
            rate = (time.perf_counter() - started) / (index + 1)
            print(
                f"  {index + 1}/{total}  {rate * 1000:.0f} ms/example, "
                f"{(total - index - 1) * rate:.0f}s left",
                flush=True,
            )
    return stack


def load_or_compute(examples: list[dataset.Example], refresh: bool = False):
    import numpy as np

    key = fingerprint(examples)
    if CACHE.exists() and not refresh:
        with np.load(CACHE, allow_pickle=False) as data:
            if str(data["key"]) == key:
                print(f"using cached hidden states ({CACHE.name})")
                return data["hidden"], None
        print("dataset changed since the cache was written; recomputing")

    from src.llm.engine import Engine

    engine = Engine().load()
    print(f"computing hidden states for {len(examples)} examples")
    stack = compute_hidden(examples, engine)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, hidden=stack, key=key)
    print(f"cached to {CACHE}")
    return stack, engine.info


def evaluate_layer(train_x, train_y, test_x, test_y, seed: int = 0) -> dict:
    """Standardise, then logistic regression. Linear first, as PLAN.md says.

    C is chosen on a validation slice carved out of *train*, never on the
    held-out set. Picking the regularisation that happens to score best on the
    test data inflates every layer's number and turns the held-out set into a
    second training set -- the reported accuracy would then be the maximum over
    four models rather than an estimate of one.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def pipeline(c):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced"),
        )

    fit_x, val_x, fit_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=seed, stratify=train_y
    )

    best_c, best_val = None, -1.0
    for c in C_GRID:
        model = pipeline(c).fit(fit_x, fit_y)
        accuracy = float((model.predict(val_x) == val_y).mean())
        if accuracy > best_val:
            best_c, best_val = c, accuracy

    model = pipeline(best_c).fit(train_x, train_y)
    scores = model.predict_proba(test_x)[:, 1]
    predicted = (scores >= 0.5).astype(int)

    chat, tool = test_y == 0, test_y == 1
    return {
        "C": best_c,
        "val_accuracy": best_val,
        "accuracy": float((predicted == test_y).mean()),
        "false_call": float(predicted[chat].mean()) if chat.any() else 0.0,
        "false_skip": float(1 - predicted[tool].mean()) if tool.any() else 0.0,
        "scores": scores.tolist(),
    }


def sweep(stack, examples: list[dataset.Example], split: dataset.Split) -> list[dict]:
    import numpy as np

    index_of = {example.key: i for i, example in enumerate(examples)}
    train_idx = np.array([index_of[e.key] for e in split.train if e.key in index_of])
    test_idx = np.array([index_of[e.key] for e in split.test])
    train_y = np.array([e.label for e in split.train if e.key in index_of])
    test_y = np.array([e.label for e in split.test])

    rows = []
    n_layers = stack.shape[1]
    for layer in range(n_layers):
        train_x = stack[train_idx, layer].astype(np.float32)
        test_x = stack[test_idx, layer].astype(np.float32)
        result = evaluate_layer(train_x, train_y, test_x, test_y)
        result["layer"] = layer
        rows.append(result)
        print(
            f"  layer {layer:>2}  acc {result['accuracy']:.1%}  "
            f"false-call {result['false_call']:.1%}  C={result['C']}"
        )
    return rows


#: Words that give the answer away without any understanding of intent. If a
#: bag-of-words model can separate the classes on these alone, the dataset
#: cannot tell us whether hidden states carry intent -- see lexical_control.
_DOMAIN_WORDS = (
    "assignment", "class", "due", "deadline", "comp", "engn", "lecture",
    "tutorial", "schedule", "timetable", "remind", "reminder", "undo",
    "percent", "hours", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "today", "tomorrow", "week", "term",
)


def lexical_control(split: dataset.Split) -> dict:
    """Can a bag of words do just as well?

    This is the control the layer sweep needs. The premise of the project is
    that hidden states carry the *intent* to call a tool. If TF-IDF over the
    raw characters scores the same, then the dataset is separable by surface
    vocabulary and the sweep has demonstrated nothing about intent -- the probe
    would just be an expensive keyword detector.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit([e.text for e in split.train], [e.label for e in split.train])
    predicted = model.predict([e.text for e in split.test])
    actual = [e.label for e in split.test]

    correct = sum(p == a for p, a in zip(predicted, actual))
    chat = [(p, a) for p, a in zip(predicted, actual) if a == 0]
    tool = [(p, a) for p, a in zip(predicted, actual) if a == 1]
    return {
        "accuracy": correct / len(actual),
        "false_call": sum(p for p, _ in chat) / max(len(chat), 1),
        "false_skip": sum(1 - p for p, _ in tool) / max(len(tool), 1),
    }


def subset_breakdown(scores, split: dataset.Split) -> dict:
    """How the chosen layer does on the cases that are meant to be hard.

    Three buckets worth separating:
      needs_context -- only a tool call given the previous turns
      lexical_trap  -- chatter that is full of schedule vocabulary
      plain         -- everything else
    """
    buckets: dict[str, list[tuple[int, float]]] = {
        "needs_context": [], "lexical_trap": [], "plain": []
    }
    for example, score in zip(split.test, scores):
        lowered = example.text.lower()
        if example.history:
            name = "needs_context"
        elif example.label == 0 and any(w in lowered for w in _DOMAIN_WORDS):
            name = "lexical_trap"
        else:
            name = "plain"
        buckets[name].append((example.label, score))

    out = {}
    for name, rows in buckets.items():
        if not rows:
            out[name] = {"n": 0, "accuracy": None}
            continue
        correct = sum((s >= 0.5) == bool(label) for label, s in rows)
        out[name] = {"n": len(rows), "accuracy": correct / len(rows)}
    return out


def choose(rows: list[dict]) -> dict:
    """Best accuracy; ties broken by the lower false-call rate.

    A false call writes wrong data, a false no-call is only unhelpful, so when
    two layers are indistinguishable on accuracy the safer one wins.
    """
    return sorted(rows, key=lambda r: (-r["accuracy"], r["false_call"], r["layer"]))[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sweep the probe across all layers")
    parser.add_argument("--refresh", action="store_true", help="recompute hidden states")
    parser.add_argument("--print-only", action="store_true", help="do not write results")
    args = parser.parse_args(argv)

    examples = dataset.load_seed()
    split = dataset.split(examples)
    print(f"dataset: {split.summary()}")

    stack, info = load_or_compute(examples, refresh=args.refresh)
    print(f"hidden states: {stack.shape} (examples, layers+1, hidden)")

    rows = sweep(stack, examples, split)
    best = choose(rows)
    print(
        f"\nbest layer {best['layer']} — {best['accuracy']:.1%} accuracy, "
        f"{best['false_call']:.1%} false calls (C={best['C']})"
    )

    control = lexical_control(split)
    print(
        f"lexical control (TF-IDF, no model): {control['accuracy']:.1%} accuracy, "
        f"{control['false_call']:.1%} false calls"
    )
    if control["accuracy"] >= best["accuracy"] - 0.02:
        print(
            "  WARNING: a bag of words does as well as the probe. The dataset is\n"
            "  separable by surface vocabulary, so this sweep says nothing about\n"
            "  whether hidden states carry intent. Harder negatives are needed."
        )

    breakdown = subset_breakdown(best["scores"], split)
    print("\nchosen layer, by difficulty:")
    for name, stats in breakdown.items():
        if stats["n"]:
            print(f"  {name:<14} {stats['accuracy']:.1%}  (n={stats['n']})")

    if not args.print_only:
        payload = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": split.summary(),
            "layers": [{k: v for k, v in r.items() if k != "scores"} for r in rows],
            "chosen": {k: v for k, v in best.items() if k != "scores"},
            "control": control,
            "breakdown": breakdown,
            "engine": info,
        }
        RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {RESULTS}")
        print("now run: python scripts\\eval_gate.py --with-decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())

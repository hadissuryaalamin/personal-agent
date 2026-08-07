"""Train the probe and tune its thresholds.

PLAN.md M3: train the probe, tune τ_lo/τ_hi, wire the uncertainty band to the
clarification branch, put it behind `config.gate`.

    python scripts\\train_probe.py
    python scripts\\train_probe.py --layer 12      override the swept layer
    python scripts\\train_probe.py --print-only

Three splits, not two. The 80/20 held-out split from `src.dataset` is
untouched; the thresholds are tuned on a validation slice carved out of the
*training* 80%. Tuning τ on the held-out set and then reporting the false-call
rate on that same set would report a number that has already been fitted, and
PLAN.md section 4 sets a target of under 2% -- a target is worth nothing if it
is measured on the data it was chosen against.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dataset  # noqa: E402
from src.llm import prompts  # noqa: E402
from sweep_layers import CACHE, fingerprint, load_or_compute  # noqa: E402

ARTIFACT = ROOT / "data" / "probe.joblib"
SWEEP_RESULTS = ROOT / "data" / "layer_sweep.json"
PROBE_EVAL = ROOT / "data" / "probe_eval.json"

#: PLAN.md section 4: "Target false-call rate < 2% on the held-out set; let the
#: uncertainty band absorb the rest." A false call writes wrong data; a false
#: no-call is only unhelpful, so the two budgets are deliberately different.
MAX_FALSE_CALL = 0.02
MAX_FALSE_SKIP = 0.10

VALIDATION_FRACTION = 0.2


def chosen_layer(override: int | None = None) -> int:
    if override is not None:
        return override
    if not SWEEP_RESULTS.exists():
        raise SystemExit(
            "No layer sweep on disk. Run: python scripts\\sweep_layers.py"
        )
    return int(json.loads(SWEEP_RESULTS.read_text(encoding="utf-8"))["chosen"]["layer"])


def tune_thresholds(scores, labels, max_false_call=MAX_FALSE_CALL,
                    max_false_skip=MAX_FALSE_SKIP) -> tuple[float, float]:
    """Pick τ_hi then τ_lo from a score distribution.

    τ_hi is the lowest threshold whose false-call rate is inside budget: call a
    tool only when the probe is confident enough that it is rarely wrong.
    τ_lo is the highest threshold whose false-skip rate is inside budget.
    Anything between them is the uncertainty band, which asks.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    chat, tool = scores[labels == 0], scores[labels == 1]

    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))

    tau_hi = 1.0
    for t in candidates:
        if chat.size == 0 or float((chat >= t).mean()) <= max_false_call:
            tau_hi = float(t)
            break

    tau_lo = 0.0
    for t in reversed(candidates):
        if tool.size == 0 or float((tool < t).mean()) <= max_false_skip:
            tau_lo = float(t)
            break

    # Perfect separation puts τ_lo above τ_hi: every real request already
    # scores above the point where chatter stops. There is then no band to
    # tune, and pretending otherwise would invent one.
    if tau_lo >= tau_hi:
        tau_lo = tau_hi
    return tau_lo, tau_hi


def rates(scores, labels, tau_lo: float, tau_hi: float) -> dict:
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    chat, tool = scores[labels == 0], scores[labels == 1]

    hard = (scores >= 0.5).astype(int)
    banded = (scores >= tau_lo) & (scores < tau_hi)

    return {
        "n": int(scores.size),
        "accuracy": float((hard == labels).mean()),
        "false_call": float((chat >= 0.5).mean()) if chat.size else 0.0,
        "false_skip": float((tool < 0.5).mean()) if tool.size else 0.0,
        "false_call_at_tau": float((chat >= tau_hi).mean()) if chat.size else 0.0,
        "false_skip_at_tau": float((tool < tau_lo).mean()) if tool.size else 0.0,
        "banded_rate": float(banded.mean()),
        "banded_chat": float(banded[labels == 0].mean()) if chat.size else 0.0,
        "banded_tool": float(banded[labels == 1].mean()) if tool.size else 0.0,
    }


def build_pipeline(c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced"),
    )


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="train the probe")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--C", type=float, default=None, dest="c")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    examples = dataset.load_seed()
    split = dataset.split(examples)
    layer = chosen_layer(args.layer)
    print(f"layer {layer}, {split.summary()}")

    stack, info = load_or_compute(examples)
    index_of = {e.key: i for i, e in enumerate(examples)}
    train_x = np.array([stack[index_of[e.key], layer] for e in split.train], dtype=np.float32)
    train_y = np.array([e.label for e in split.train])
    test_x = np.array([stack[index_of[e.key], layer] for e in split.test], dtype=np.float32)
    test_y = np.array([e.label for e in split.test])

    # Carve validation out of train. The held-out set stays untouched.
    fit_x, val_x, fit_y, val_y = train_test_split(
        train_x, train_y, test_size=VALIDATION_FRACTION,
        random_state=dataset.SPLIT_SEED, stratify=train_y,
    )

    c = args.c
    if c is None:
        best = None
        for candidate in (0.01, 0.1, 1.0, 10.0):
            model = build_pipeline(candidate).fit(fit_x, fit_y)
            accuracy = float((model.predict(val_x) == val_y).mean())
            if best is None or accuracy > best[1]:
                best = (candidate, accuracy)
        c = best[0]
        print(f"C={c} (validation accuracy {best[1]:.1%})")

    # Thresholds come from validation scores only.
    model = build_pipeline(c).fit(fit_x, fit_y)
    val_scores = model.predict_proba(val_x)[:, 1]
    tau_lo, tau_hi = tune_thresholds(val_scores, val_y)
    print(f"thresholds from validation: τ_lo {tau_lo:.3f}, τ_hi {tau_hi:.3f}")
    if tau_lo == tau_hi:
        print(
            "  NOTE: the classes separate cleanly on validation, so there is no\n"
            "  uncertainty band. The clarification branch will never fire from the\n"
            "  probe until real turns make the score distribution messier."
        )

    # Refit on all of train, with the C and thresholds already decided.
    final = build_pipeline(c).fit(train_x, train_y)
    test_scores = final.predict_proba(test_x)[:, 1]
    held_out = rates(test_scores, test_y, tau_lo, tau_hi)
    validation = rates(val_scores, val_y, tau_lo, tau_hi)

    print(
        f"\nheld-out: {held_out['accuracy']:.1%} accuracy, "
        f"{held_out['false_call']:.1%} false calls, "
        f"{held_out['false_skip']:.1%} false skips"
    )
    print(
        f"  at the tuned thresholds: {held_out['false_call_at_tau']:.1%} false calls, "
        f"{held_out['banded_rate']:.1%} land in the band"
    )
    if held_out["false_call_at_tau"] > MAX_FALSE_CALL:
        print(
            f"  WARNING: above the {MAX_FALSE_CALL:.0%} target from PLAN.md "
            "section 4."
        )

    if args.print_only:
        return 0

    import joblib

    payload = {
        "pipeline": final,
        "layer": layer,
        "C": c,
        "tau_lo": tau_lo,
        "tau_hi": tau_hi,
        "hidden_size": int(train_x.shape[1]),
        "n_layers": int(stack.shape[1]) - 1,
        "dataset_fingerprint": fingerprint(examples),
        # Which system prompt these activations came from. ProbeGate refuses to
        # run if the prompt has moved since -- see prompts.system_fingerprint.
        "prompt_fingerprint": prompts.system_fingerprint(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, ARTIFACT)
    print(f"\nwrote {ARTIFACT}")

    PROBE_EVAL.write_text(
        json.dumps(
            {
                "layer": layer, "C": c, "tau_lo": tau_lo, "tau_hi": tau_hi,
                "held_out": held_out, "validation": validation,
                "dataset": split.summary(),
                "trained_at": payload["trained_at"],
                "scores": test_scores.tolist(),
                "labels": test_y.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {PROBE_EVAL}")
    print("now run: python scripts\\eval_gate.py --with-decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())

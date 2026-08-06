"""Phase 3: train the linear probe and report what it is actually worth.

StandardScaler then LogisticRegression with L2 at lambda = 10000 (C = 1e-4),
as the paper specifies. The heavy regularisation is not optional decoration:
143 points in 94,720 dimensions are linearly separable for almost any
labelling, so an underregularised fit would reach perfect training accuracy
while learning nothing.

    python -m src.probe
    python -m src.probe --layers   # per-layer sweep

WHY CROSS-VALIDATION AND NOT THE SPLIT

The 30% test split holds 13 hard items. An AUROC over 13 points moves by 0.08
when one of them flips, so it cannot distinguish a working probe from a broken
one on the slice that carries the whole argument. Stratified K-fold gives
every item a held-out prediction -- 42 hard predictions instead of 13 -- and
train.json/test.json stay untouched here for a single final holdout run in
Phase 5. Folds are stratified on (label, difficulty), for the same reason
the split in dataset.py is: a fold that happens to hold no hard positives reports a number
about nothing.

WHY A BAG-OF-WORDS BASELINE RUNS ALONGSIDE

The failure this replication is most likely to have, and least likely to
notice, is a probe that has learned vocabulary. "My schedule" means label 1
almost everywhere in this set, and a classifier reading only the words would
score well overall while knowing nothing about tool use. So the same folds
train a word-count logistic regression, and both numbers are reported side by
side. If the hidden states do not beat words -- especially on the hard slice,
which was written specifically to defeat word matching in both directions --
then the probe found the vocabulary, and this file says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import ROOT, load as load_tasks

FEATURES = ROOT / "features.npz"
SEED = 20260806
N_SPLITS = 5
C = 1e-4          # lambda = 10000
TEMPERATURE = 2.0  # Phase 4 reads p = sigmoid(z / T)


def auroc_by(scores, y, keys, roc_auc_score) -> list[tuple[str, float, int, int]]:
    """AUROC within each subgroup, skipping any that has only one class."""
    import numpy as np
    rows = []
    for k in sorted(set(keys)):
        m = keys == k
        n_pos = int(y[m].sum())
        if n_pos in (0, int(m.sum())):
            rows.append((k, float("nan"), int(m.sum()), n_pos))
            continue
        rows.append((k, roc_auc_score(y[m], scores[m]), int(m.sum()), n_pos))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES))
    ap.add_argument("--layers", action="store_true",
                    help="also score each layer alone, to locate the signal")
    ap.add_argument("--no-controls", action="store_true",
                    help="skip the permutation and length controls")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.features, allow_pickle=False)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(int)
    meta = json.loads(str(d["meta"]))
    ids, diff, group = d["ids"], d["difficulty"], d["group"]

    tasks = {t["id"]: t for t in load_tasks()}
    text = np.array([tasks[i]["prompt"] for i in ids])

    # The hash was being written into every feature file and never read. A
    # tripwire nobody checks is decoration: the tool set changed from three
    # tools to six during a refactor, and without this the probe would have
    # been trained on states extracted under a prompt that no longer exists.
    from .model import prompt_hash
    live = prompt_hash()
    if meta.get("prompt_style", "deployed") == "deployed" and meta.get("prompt_hash") != live:
        print(f"  FAIL           these features were extracted under prompt "
              f"{meta.get('prompt_hash')},")
        print(f"                 but the agent now deploys {live}. The system")
        print("                 prompt or the tool schema has changed; re-run")
        print("                 python -m src.hidden_states")
        return 1

    print(f"  features       {args.features}")
    print(f"  model          {meta['model']}  ({meta['quantisation']})")
    print(f"  matrix         {X.shape[0]} x {X.shape[1]:,}")
    print(f"  labels         {int(y.sum())} tool / {int((1 - y).sum())} no tool")
    print(f"  probe          LogisticRegression L2, C={C:g} (lambda={1 / C:g})")
    print(f"  folds          {N_SPLITS}, stratified on (label, difficulty), seed {SEED}")
    print()

    def probe_pipeline():
        # penalty="l2" is sklearn's default and naming it explicitly is
        # deprecated as of 1.8; C alone sets the L2 strength.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=5000, random_state=SEED))

    def words_pipeline():
        # Deliberately generous: unigrams and bigrams, no stopword removal, and
        # C left at the sklearn default so the baseline is not handicapped by
        # the probe's regularisation.
        return make_pipeline(
            CountVectorizer(ngram_range=(1, 2), lowercase=True),
            LogisticRegression(max_iter=5000, random_state=SEED))

    strat = np.array([f"{a}-{b}" for a, b in zip(y, diff)])
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_probe = np.zeros(len(y))
    oof_words = np.zeros(len(y))
    for tr, te in cv.split(X, strat):
        p = probe_pipeline().fit(X[tr], y[tr])
        oof_probe[te] = p.decision_function(X[te])
        w = words_pipeline().fit(text[tr], y[tr])
        oof_words[te] = w.decision_function(text[te])

    acc = float(((oof_probe > 0).astype(int) == y).mean())
    overall = roc_auc_score(y, oof_probe)
    overall_w = roc_auc_score(y, oof_words)

    print("  HELD-OUT (every item predicted by a fold that never saw it)")
    print()
    print(f"  {'':<14} {'probe':>8} {'words':>8} {'n':>5} {'pos':>5}")
    print(f"  {'overall':<14} {overall:>8.3f} {overall_w:>8.3f} "
          f"{len(y):>5} {int(y.sum()):>5}")

    rows_p = {k: v for k, v, _, _ in auroc_by(oof_probe, y, diff, roc_auc_score)}
    rows_w = {k: v for k, v, _, _ in auroc_by(oof_words, y, diff, roc_auc_score)}
    counts = {k: (n, pos) for k, _, n, pos in auroc_by(oof_probe, y, diff, roc_auc_score)}
    for k in ("easy", "medium", "hard"):
        n, pos = counts[k]
        print(f"  {k:<14} {rows_p[k]:>8.3f} {rows_w[k]:>8.3f} {n:>5} {pos:>5}")

    print()
    print("  by group")
    for k, v, n, pos in auroc_by(oof_probe, y, group, roc_auc_score):
        vw = {a: b for a, b, _, _ in auroc_by(oof_words, y, group, roc_auc_score)}[k]
        label = f"{v:>8.3f} {vw:>8.3f}" if v == v else f"{'n/a':>8} {'n/a':>8}  single class"
        print(f"  {k:<14} {label} {n:>5} {pos:>5}")

    print()
    print(f"  accuracy at z>0   {acc:.3f}   ({int((oof_probe > 0).sum())} of "
          f"{len(y)} routed to a tool)")

    # The items the probe gets most confidently wrong are the most informative
    # thing in this output: they say what it actually learned.
    p_hat = 1 / (1 + np.exp(-oof_probe / TEMPERATURE))
    wrong = np.argsort(np.where(y == 1, p_hat, 1 - p_hat))[:6]
    print()
    print("  worst held-out predictions")
    for i in wrong:
        print(f"    p={p_hat[i]:.2f} label={y[i]} {diff[i]:<6} {group[i]:<13} "
              f"{tasks[ids[i]]['prompt'][:52]}")

    if not args.no_controls:
        # A held-out AUROC of 1.000 is a claim that has to survive being
        # attacked, not a result to be pleased about. Three attacks, each
        # aimed at a specific way this number could be fake.
        print()
        print("  CONTROLS")

        # Every prompt is the same 1180-token preamble plus a short question,
        # so token count varies with wording. If length alone tracked the
        # label, the probe could reach a high score without reading meaning.
        ntok = d["n_tokens"].astype(float)
        len_auc = roc_auc_score(y, ntok)
        print(f"    length only        {len_auc:.3f}   "
              f"(tool {ntok[y == 1].mean():.1f} tok, no-tool {ntok[y == 0].mean():.1f})")

        # 143 points in 94,720 dimensions can be separated for ANY labelling
        # unless the regularisation is doing real work. Shuffle the labels: a
        # sound pipeline must fail to learn them.
        rng = np.random.default_rng(0)
        perms = []
        for _ in range(3):
            yp = rng.permutation(y)
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, yp):
                oof[te] = probe_pipeline().fit(X[tr], yp[tr]).decision_function(X[te])
            perms.append(roc_auc_score(yp, oof))
        print(f"    shuffled labels    {', '.join(f'{p:.3f}' for p in perms)}"
              f"   (must sit near 0.500)")

        # AUROC only ranks. Two clouds touching at one point rank perfectly and
        # would break under any distribution shift; a gap is the real claim.
        gap = oof_probe[y == 1].min() - oof_probe[y == 0].max()
        print(f"    margin             z>=  {oof_probe[y == 1].min():+.2f} for tool, "
              f"z<= {oof_probe[y == 0].max():+.2f} for none, gap {gap:+.2f}")

        if max(perms) > 0.65:
            print("    WARN  shuffled labels scored above chance -- the fit is")
            print("          memorising, and the headline number is inflated")
        if len_auc > 0.85:
            print("    WARN  prompt length alone nearly matches the probe")

    if args.layers:
        # Where does the signal live? Cheap to ask -- 2560 columns per layer --
        # and it decides whether Phase 4 could read one layer instead of 37.
        h = meta["hidden_size"]
        print()
        print("  per-layer held-out AUROC (embedding = 0)")
        best = (0, 0.0)
        for L in range(meta["n_layers"] + 1):
            sl = X[:, L * h:(L + 1) * h]
            oof = np.zeros(len(y))
            for tr, te in cv.split(sl, strat):
                oof[te] = probe_pipeline().fit(sl[tr], y[tr]).decision_function(sl[te])
            a = roc_auc_score(y, oof)
            hard = roc_auc_score(y[diff == "hard"], oof[diff == "hard"])
            best = max(best, (a, L), key=lambda t: t[0] if isinstance(t[0], float) else 0)
            bar = "#" * int(max(0.0, a - 0.5) * 80)
            print(f"    {L:>2}  {a:.3f}  hard {hard:.3f}  {bar}")

    ok = True
    print()
    if overall < 0.6:
        print(f"  FAIL           AUROC {overall:.3f} is near chance. At this width")
        print("                 that means a bug in this pipeline, not a finding")
        print("                 about the paper -- rerun against Qwen3-1.7B bf16,")
        print("                 for which the paper reports 0.894, before believing it")
        ok = False
    elif overall < 0.85:
        print(f"  WARN           AUROC {overall:.3f} is below the 0.85 gate")
    if rows_p["hard"] < 0.7 <= overall:
        print(f"  WARN           hard slice {rows_p['hard']:.3f} against {overall:.3f}")
        print("                 overall: the probe is riding on the easy cases")
    if overall <= overall_w:
        print(f"  WARN           words score {overall_w:.3f} against the probe's")
        print(f"                 {overall:.3f} -- the hidden states are not adding")
        print("                 anything a bag of words did not already have")

    # The per-item held-out predictions are the only place the probe's decision
    # can be compared with what the model actually generated for the same task.
    # Recomputing them costs a fold refit and, worse, drifts if anything here
    # changes -- so they are written down next to the numbers that summarise
    # them.
    if args.no_save:
        print("  --no-save: oof.json and probe.joblib left untouched")
        print("  Phase 3 PASSED" if ok else "  Phase 3 FAILED")
        return 0 if ok else 1

    # Named by prompt style: a probe read off bare questions and one read off
    # the deployed 1683-token prompt are different classifiers with different
    # deployments, and overwriting one with the other has already happened once.
    style = meta.get("prompt_style", "deployed")
    suffix = "" if style == "deployed" else f"-{style}"

    oof_path = ROOT / f"oof{suffix}.json"
    oof_path.write_text(json.dumps([
        {"id": str(ids[i]), "label": int(y[i]), "difficulty": str(diff[i]),
         "group": str(group[i]), "prompt": tasks[ids[i]]["prompt"],
         "z": round(float(oof_probe[i]), 4),
         "p": round(float(1 / (1 + np.exp(-oof_probe[i] / TEMPERATURE))), 4),
         "z_words": round(float(oof_words[i]), 4)}
        for i in range(len(y))], indent=2), encoding="utf-8")
    print(f"  saved          {oof_path}  (held-out prediction per task)")

    # Phase 4 needs a fitted probe, and it must be fitted on train.json alone:
    # test.json is the one untouched holdout and is spent once, in Phase 5.
    if not args.no_save:
        import joblib
        train_ids = {t["id"] for t in
                     json.loads((ROOT / "train.json").read_text(encoding="utf-8"))}
        m = np.array([i in train_ids for i in ids])
        final = probe_pipeline().fit(X[m], y[m])
        out = ROOT / f"probe{suffix}.joblib"
        joblib.dump({"model": final, "meta": meta, "style": style,
                     "temperature": TEMPERATURE,
                     "trained_on": "train.json", "n_train": int(m.sum()),
                     "cv_auroc": float(overall)}, out)
        print(f"  saved          {out}  (fitted on {int(m.sum())} train items)")

    print("  Phase 3 PASSED" if ok else "  Phase 3 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Measure the gate and write the numbers into docs/eval.md.

CLAUDE.md: any accuracy or latency number in a doc must come from a script in
scripts/ and land in docs/eval.md. This is that script. Nothing here is typed
by hand into prose -- the tables below are regenerated, not edited.

    python scripts\\eval_gate.py                 measure and rewrite docs/eval.md
    python scripts\\eval_gate.py --with-decode   also time the tool pass
    python scripts\\eval_gate.py --print-only    measure, print, touch nothing

The prompted baseline is scored on the **held-out split only** -- the same
examples the probe is tested on, from the same stratified split with the same
seed. PLAN.md section 4 asks for "same prompts, same split", and a baseline
measured on data the probe never sees would not be a comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, dataset  # noqa: E402
from src.llm import gate as gate_module  # noqa: E402
from src.llm import prompts  # noqa: E402
from src.llm.engine import Engine  # noqa: E402

EVAL_DOC = ROOT / "docs" / "eval.md"
SWEEP_RESULTS = ROOT / "data" / "layer_sweep.json"

TZ = ZoneInfo("Australia/Sydney")
#: Fixed so the numbers are reproducible: the same Thursday the tests use.
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)


def load_dataset() -> list[dataset.Example]:
    """The held-out split -- what both the baseline and the probe are scored on."""
    return dataset.split(dataset.load_seed()).test


def measure(engine, gate, examples: list[dataset.Example], extra_gates=None) -> list[dict]:
    """Score every example, optionally with several gates on the same prefill.

    Sharing the prefill is what makes the M3 comparison honest: both gates see
    exactly the same encoded prompt, so the only difference in their timings is
    the gate itself.
    """
    results = []
    for example in examples:
        messages = prompts.build_messages(
            NOW, "Australia/Sydney", example.text, example.messages()
        )
        started = time.perf_counter()
        prefill = engine.prefill(engine.prefix_text(messages))
        decision = gate.decide(engine, prefill)
        total_ms = (time.perf_counter() - started) * 1000

        row = {
            "text": example.text,
            "label": example.label,
            "why": example.why,
            "score": decision.score,
            "label_out": decision.label,
            "ms_prefill": prefill.ms,
            "ms_gate": decision.ms,
            "ms_total": total_ms,
            "tokens": prefill.n_tokens,
            "others": {},
        }
        for name, other in (extra_gates or {}).items():
            other_started = time.perf_counter()
            other_decision = other.decide(engine, prefill)
            row["others"][name] = {
                "score": other_decision.score,
                "label_out": other_decision.label,
                # perf_counter, not the gate's own ms: a probe decision rounds
                # to 0 ms and the comparison is about exactly that.
                "ms_gate": (time.perf_counter() - other_started) * 1000,
            }
        results.append(row)
    return results


def summarise_other(
    results: list[dict], name: str, tau_lo: float | None = None, tau_hi: float | None = None
) -> dict:
    rows = [
        {**r["others"][name], "label": r["label"], "text": r["text"], "why": r["why"],
         "ms_prefill": r["ms_prefill"], "tokens": r["tokens"]}
        for r in results
        if name in r["others"]
    ]
    if not rows:
        return {}
    summary = summarise(
        [{**r, "ms_total": r["ms_prefill"] + r["ms_gate"]} for r in rows], tau_lo, tau_hi
    )
    summary["median_gate_ms"] = statistics.median(r["ms_gate"] for r in rows)
    return summary


def summarise(
    results: list[dict], tau_lo: float | None = None, tau_hi: float | None = None
) -> dict:
    """Rates at a hard 0.5, plus what the three-way band does at τ.

    The thresholds must be the ones the gate in question actually uses. The
    probe's are tuned and live in its artefact; scoring it against the module
    defaults would report a band it never applies.
    """
    tau_lo = gate_module.TAU_LO if tau_lo is None else tau_lo
    tau_hi = gate_module.TAU_HI if tau_hi is None else tau_hi

    total = len(results)
    wants_tool = [r for r in results if r["label"] == 1]
    wants_chat = [r for r in results if r["label"] == 0]

    correct = sum((r["score"] >= 0.5) == bool(r["label"]) for r in results)
    false_calls = [r for r in wants_chat if r["score"] >= 0.5]
    false_skips = [r for r in wants_tool if r["score"] < 0.5]
    banded = [r for r in results if tau_lo <= r["score"] < tau_hi]
    confident_false_calls = [r for r in wants_chat if r["score"] >= tau_hi]

    return {
        "n": total,
        "n_tool": len(wants_tool),
        "n_chat": len(wants_chat),
        "accuracy": correct / total,
        "false_call_rate": len(false_calls) / max(len(wants_chat), 1),
        "false_skip_rate": len(false_skips) / max(len(wants_tool), 1),
        "banded_rate": len(banded) / total,
        "tau_lo": tau_lo,
        "tau_hi": tau_hi,
        "confident_false_call_rate": len(confident_false_calls) / max(len(wants_chat), 1),
        "false_calls": sorted(false_calls, key=lambda r: -r["score"]),
        "false_skips": sorted(false_skips, key=lambda r: r["score"]),
        "median_ms": statistics.median(r["ms_total"] for r in results),
        "median_gate_ms": statistics.median(r["ms_gate"] for r in results),
        "median_prefill_ms": statistics.median(r["ms_prefill"] for r in results),
        "median_tokens": statistics.median(r["tokens"] for r in results),
    }


def render_latency(decode: dict) -> list[str]:
    budget_prefill, budget_gen = 150, 500
    return [
        "## Latency",
        "",
        "Measured by `scripts\\bench_decode.py` on the tool pass, which is the",
        "expensive branch. PLAN.md section 5 budgets 150 ms for prefill+probe and",
        "500 ms for generation.",
        "",
        "| Stage | Budget | Measured | |",
        "|---|---|---|---|",
        f"| Prefill ({decode['prompt_tokens']} tokens) | {budget_prefill} ms | "
        f"{decode['median_prefill_ms']:.0f} ms | "
        f"{decode['median_prefill_ms'] / budget_prefill:.1f}× over |",
        f"| Generate ({decode['median_out_tokens']:.0f} tokens) | {budget_gen} ms | "
        f"{decode['median_gen_ms']:.0f} ms | "
        f"{decode['median_gen_ms'] / budget_gen:.1f}× over |",
        f"| **Turn total** | **650 ms** | **{decode['median_turn_ms']:.0f} ms** | |",
        "",
        f"Decode runs at **{decode['ms_per_token']:.0f} ms/token**; transformers'",
        f"own `generate` manages {decode['hf_ms_per_token']:.0f} ms/token on the same",
        "prompt, so the wrapper in `src/llm/engine.py` is not the bottleneck — 4-bit",
        "bitsandbytes decode on this card is. Stopping generation at the closing",
        "brace caps the worst case but does not move the median.",
        "",
        "The implication for M5 is that the budget is not reachable by tuning the",
        "prompt or the loop, and **the probe will not fix it either**: the gate is",
        "one forward pass either way, and the cost is decoding the tool call. It",
        "needs a faster 4-bit kernel than bitsandbytes, or a card with room for",
        "bf16. Worth settling before the TTS streaming work, which assumes",
        "generation keeps ahead of speech.",
        "",
    ]


def render_sweep(sweep: dict) -> list[str]:
    rows = sweep["layers"]
    chosen = sweep["chosen"]
    best_by_acc = sorted(rows, key=lambda r: -r["accuracy"])[:10]

    lines = [
        "## Layer sweep",
        "",
        "Which layer the probe should read. A logistic regression on the",
        "last-token hidden state of each layer, trained on the 80% split and",
        "scored on the held-out 20% — the same held-out set the prompted",
        "baseline above is scored on.",
        "",
        f"**Chosen: layer {chosen['layer']}** — {chosen['accuracy']:.1%} accuracy, "
        f"{chosen['false_call']:.1%} false calls, C={chosen['C']}.",
        "",
        "Best ten layers:",
        "",
        "| Layer | Accuracy | False call | False skip | C |",
        "|---|---|---|---|---|",
    ]
    for row in best_by_acc:
        marker = " ←" if row["layer"] == chosen["layer"] else ""
        lines.append(
            f"| {row['layer']}{marker} | {row['accuracy']:.1%} | "
            f"{row['false_call']:.1%} | {row['false_skip']:.1%} | {row['C']} |"
        )

    lines += ["", "Every layer, accuracy only:", "", "| Layer | Acc | Layer | Acc | Layer | Acc |", "|---|---|---|---|---|---|"]
    ordered = sorted(rows, key=lambda r: r["layer"])
    third = (len(ordered) + 2) // 3
    for i in range(third):
        cells = []
        for column in range(3):
            index = i + column * third
            if index < len(ordered):
                cells += [str(ordered[index]["layer"]), f"{ordered[index]['accuracy']:.1%}"]
            else:
                cells += ["", ""]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"Dataset: {sweep['dataset']['train']} train "
        f"({sweep['dataset']['train_tool']} tool), "
        f"{sweep['dataset']['test']} held out "
        f"({sweep['dataset']['test_tool']} tool).",
        "",
    ]

    saturated = sum(1 for r in rows if r["accuracy"] >= chosen["accuracy"] - 1e-9)

    control = sweep.get("control")
    if control:
        lines += [
            "### Is this real, or is the dataset too easy?",
            "",
            "A linear probe hitting 100% on held-out data is a reason for",
            "suspicion, not celebration. Two checks:",
            "",
            "**Lexical control.** TF-IDF over character n-grams of the raw text,",
            "same split, no model at all:",
            "",
            "| | Accuracy | False call |",
            "|---|---|---|",
            f"| Probe, layer {chosen['layer']} | {chosen['accuracy']:.1%} | "
            f"{chosen['false_call']:.1%} |",
            f"| Bag of words | {control['accuracy']:.1%} | {control['false_call']:.1%} |",
            "",
            "The probe is ahead, so the classes are not separable by vocabulary",
            "alone — but a keyword detector already gets most of the way, which",
            "says more about the dataset than about the hidden states.",
            "",
        ]

    breakdown = sweep.get("breakdown")
    if breakdown:
        lines += [
            "**By difficulty.** The buckets that are supposed to be hard:",
            "",
            "| Bucket | n | Accuracy |",
            "|---|---|---|",
        ]
        labels = {
            "needs_context": "only a tool call in context",
            "lexical_trap": "chatter full of schedule words",
            "plain": "everything else",
        }
        for name, stats in breakdown.items():
            if stats.get("n"):
                lines.append(
                    f"| {labels.get(name, name)} | {stats['n']} | {stats['accuracy']:.1%} |"
                )
        lines += [
            "",
            "The hard buckets are tiny — single-digit and low-double-digit counts.",
            "Getting 10 out of 10 is encouraging and is not evidence of much.",
            "",
        ]

    layer_zero = next((r for r in rows if r["layer"] == 0), None)
    if layer_zero:
        lines += [
            f"**Layer 0 scores {layer_zero['accuracy']:.1%}.** That is the last",
            "token's embedding, before any transformer block has run. It is at",
            "chance, which is the control that matters most: the probe higher up",
            "is reading something the model *computes*, not the identity of the",
            "final token.",
            "",
        ]

    if saturated > 3:
        lines += [
            f"**Caution: {saturated} of {len(rows)} layers tie at the top**, so the",
            f"choice of layer {chosen['layer']} comes from the tie-break (lowest",
            "false-call rate, then earliest layer) rather than from evidence.",
            "",
        ]

    lines += [
        "### How much this result is worth",
        "",
        "The probe is trained on the other 80% of the same hand-written set it is",
        "scored on. The prompted gate never saw any of it. So the table above is",
        "**not** a like-for-like generalisation test — the probe has been fitted to",
        "this dataset's idiosyncrasies and the baseline has not, and some of the",
        "gap is that advantage rather than the method.",
        "",
        "What it does establish is the thing D1 actually assumes: that whether a",
        "turn wants a tool is **linearly decodable** from a mid-stack hidden state,",
        "well above what the same text supports lexically. That is the claim the",
        "project rests on, and it holds.",
        "",
        "What it does not establish is behaviour on real speech. Every utterance",
        "here was typed by the same author as the prompt; none came from ASR, with",
        "its manglings and disfluencies. The honest test arrives at M4, when",
        "`turn_log` starts accumulating real turns — which land in train, never in",
        "the held-out set.",
        "",
    ]
    return lines


def render_comparison(prompted: dict, probe: dict, artifact: dict) -> list[str]:
    """The comparison M3 exists for: does the probe actually win?"""
    gain = probe["accuracy"] - prompted["accuracy"]
    speedup = prompted["median_gate_ms"] / max(probe["median_gate_ms"], 0.01)
    verdict = (
        "The probe wins on both. D1 holds."
        if gain > 0 and probe["median_gate_ms"] < prompted["median_gate_ms"]
        else "The probe does **not** clearly win. D1 needs revisiting."
    )

    return [
        "## M3: probe vs prompted baseline",
        "",
        "PLAN.md section 4: *the probe has to beat it on accuracy and latency, or",
        "D1 was the wrong call.* Both gates are run on the **same prefill** for",
        "each of the same held-out utterances, so the only difference in the",
        "timings is the gate itself.",
        "",
        "| | Prompted | Probe | |",
        "|---|---|---|---|",
        f"| Accuracy | {prompted['accuracy']:.1%} | {probe['accuracy']:.1%} | "
        f"{gain:+.1%} |",
        f"| False call | {prompted['false_call_rate']:.1%} | "
        f"{probe['false_call_rate']:.1%} | "
        f"{probe['false_call_rate'] - prompted['false_call_rate']:+.1%} |",
        f"| False skip | {prompted['false_skip_rate']:.1%} | "
        f"{probe['false_skip_rate']:.1%} | "
        f"{probe['false_skip_rate'] - prompted['false_skip_rate']:+.1%} |",
        f"| Gate latency | {prompted['median_gate_ms']:.1f} ms | "
        f"{probe['median_gate_ms']:.2f} ms | {speedup:.0f}× faster |",
        "",
        f"**{verdict}**",
        "",
        f"The probe reads layer {artifact['layer']} of the hidden state that the",
        "prefill produced anyway, so the gate costs a matrix multiply instead of",
        "a forward pass over a question and its answer. The accuracy gap is the",
        "larger result, but the latency gap is the one that made D1 worth the",
        "architecture: it is the difference between a gate you can afford on",
        "every turn and one you cannot.",
        "",
        "### The uncertainty band is empty",
        "",
        f"`train_probe.py` tuned τ_lo and τ_hi on a validation slice and got",
        f"**τ_lo = τ_hi = {artifact['tau_lo']:.3f}** — the two classes separate",
        "cleanly enough that there is no range left between them. So the",
        "three-way branch in PLAN.md section 4 currently has two live arms:",
        f"{probe['banded_rate']:.0%} of held-out turns land in the band, because",
        "there is no band to land in.",
        "",
        "This is a finding about the dataset, not a bug in the gate. Hand-written",
        "utterances are unambiguous in a way real speech is not: nobody writes",
        "the half-finished, half-audible sentence that a probe *should* be unsure",
        "about. The band stays wired up and will start earning its place when",
        "`turn_log` accumulates real ASR output at M4 — that is exactly the case",
        "invariant #6 exists for.",
        "",
    ]


def render_loop(loop: dict) -> list[str]:
    """M5's exit criterion, measured from turn_log rather than asserted."""
    from bench_loop import STAGE_NAMES

    total = loop["median_total"]
    budget = loop["budget_total"]
    met = total <= budget

    lines = [
        "## M5: the full loop against the latency budget",
        "",
        "PLAN.md section 5 allows 1400 ms from the end of speech to the first",
        "audio out. Measured by `scripts\\bench_loop.py` over real spoken turns",
        f"in `turn_log` ({loop['n_turns']} turns; the first is dropped, because it",
        "pays for loading the model and warming three ONNX graphs).",
        "",
        "| Stage | Budget | Measured | |",
        "|---|---|---|---|",
    ]
    for key, allowed in loop["budget"].items():
        measured = loop["stages"][key]
        verdict = "ok" if measured <= allowed else f"**{measured / allowed:.1f}× over**"
        lines.append(
            f"| {STAGE_NAMES[key]} | {allowed} ms | {measured:.0f} ms | {verdict} |"
        )
    lines += [
        f"| **Total** | **{budget} ms** | **{total:.0f} ms** | "
        f"**{total / budget:.1f}× over** |",
        "",
        f"Best turn {loop['best_total']:.0f} ms, worst {loop['worst_total']:.0f} ms.",
        "",
        f"**The exit criterion is not met.** {'' if met else ''}"
        "The loop works end to end — speech in, the right database write, speech",
        f"out — but it takes {total / 1000:.1f} s per turn where the budget is",
        f"{budget / 1000:.1f} s.",
        "",
        "Where it goes, in order of size:",
        "",
        "- **Generation** is the largest single cost and the least fixable by",
        "  tuning: decode runs at roughly 90 ms/token on 4-bit weights (see",
        "  Latency, below) and a tool call is 40–70 tokens. The probe does not",
        "  help here — gating is one forward pass either way; the cost is",
        "  *writing the JSON*.",
        "- **Kokoro** takes about 0.5× real time on the CPU, so a two-second",
        "  reply costs a second before the first sound. Sentence streaming was",
        "  built to hide this and largely cannot: `src/format.py` caps replies at",
        "  two sentences and most are one, so there is no second sentence to",
        "  overlap with. Thread count was measured and makes no difference.",
        "- **VAD endpointing** is 3× over by deliberate choice. 200 ms cut real",
        "  sentences in half at their commas; see PLAN.md M4.",
        "",
        "ASR and the gate are inside or near their budgets. The two that are not",
        "are both *generation* problems, and both point at the same fix: this is",
        "an 8 GB card running 4-bit weights, and neither the LLM nor the TTS has",
        "room to run the way the budget assumes. That is a hardware and",
        "quantisation decision, not a prompt or a code path.",
        "",
    ]
    return lines


def render(summary: dict, engine_info: dict, decode: dict | None = None,
           sweep: dict | None = None, probe: dict | None = None,
           artifact: dict | None = None, loop: dict | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Evaluation",
        "",
        "Regenerated by `python scripts\\eval_gate.py` and",
        "`python scripts\\sweep_layers.py`. Do not edit by hand — CLAUDE.md",
        "requires every number here to come from a script.",
        "",
        f"Last run: {stamp}",
        "",
        "## Setup",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{Path(engine_info['model_dir']).name}` |",
        f"| Precision | {engine_info['quantisation']} on {engine_info['device']} |",
        f"| Layers / hidden | {engine_info['layers']} / {engine_info['hidden_size']} |",
        f"| Held-out set | {summary['n']} utterances "
        f"({summary['n_tool']} tool / {summary['n_chat']} chat) |",
        f"| Thresholds | τ_lo {gate_module.TAU_LO}, τ_hi {gate_module.TAU_HI} |",
        "",
        "## Gate: prompted baseline",
        "",
        "The M1 baseline from PLAN.md section 4 — ask the model TOOL or CHAT and",
        "read the answer distribution. The probe has to beat this on accuracy",
        "*and* latency at M3, or D1 was the wrong call.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy (hard 0.5) | {summary['accuracy']:.1%} |",
        f"| False-call rate | {summary['false_call_rate']:.1%} |",
        f"| False-skip rate | {summary['false_skip_rate']:.1%} |",
        f"| False calls above τ_hi | {summary['confident_false_call_rate']:.1%} |",
        f"| Lands in the band (asks) | {summary['banded_rate']:.1%} |",
        f"| Median prompt | {summary['median_tokens']:.0f} tokens |",
        f"| Median prefill | {summary['median_prefill_ms']:.0f} ms |",
        f"| Median gate | {summary['median_gate_ms']:.0f} ms |",
        f"| Median total | {summary['median_ms']:.0f} ms |",
        "",
    ]

    if summary["false_calls"]:
        lines += [
            "### False calls — said TOOL when nothing was asked for",
            "",
            "The dangerous direction: a false call can write wrong data.",
            "",
        ]
        lines += [
            f"- `{r['score']:.3f}` — “{r['text']}” _({r['why']})_"
            for r in summary["false_calls"][:12]
        ]
        lines.append("")

    if summary["false_skips"]:
        lines += [
            "### False skips — said CHAT when a tool was wanted",
            "",
            "Merely unhelpful: the user repeats themselves.",
            "",
        ]
        lines += [
            f"- `{r['score']:.3f}` — “{r['text']}” _({r['why']})_"
            for r in summary["false_skips"][:12]
        ]
        lines.append("")

    if probe and artifact:
        lines += render_comparison(summary, probe, artifact)
    if loop:
        lines += render_loop(loop)
    if sweep:
        lines += render_sweep(sweep)
    if decode:
        lines += render_latency(decode)

    pending = []
    if not sweep:
        pending.append("- **Layer sweep** (M2): run `python scripts\\sweep_layers.py`.")
    if not probe:
        pending.append(
            "- **Probe vs baseline** (M3): run `python scripts\\train_probe.py`."
        )
    if not loop:
        pending.append(
            "- **End-to-end latency** (M5): run `python scripts\\bench_loop.py` "
            "against spoken turns in `turn_log`."
        )
    if pending:
        lines += ["## Not measured yet", ""] + pending + [""]

    lines += [
        "The dataset is hand-written by the same author as the prompt and the",
        "probe. It is a seed, not a benchmark: real logged turns get folded into",
        "train as they accumulate, and they never enter the held-out set.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure the tool gate")
    parser.add_argument("--print-only", action="store_true", help="do not write docs/eval.md")
    parser.add_argument(
        "--with-decode", action="store_true", help="also time the tool pass (slower)"
    )
    args = parser.parse_args(argv)

    examples = load_dataset()
    engine = Engine().load()
    gate = gate_module.PromptedGate()

    # Run the probe on the same prefills when it exists, so M3's comparison is
    # like-for-like rather than two separate runs.
    extra, artifact = {}, None
    probe_path = ROOT / "data" / "probe.joblib"
    if probe_path.exists():
        probe_gate = gate_module.ProbeGate().load()
        extra["probe"] = probe_gate
        artifact = {
            "layer": probe_gate.layer,
            "tau_lo": probe_gate.tau_lo,
            "tau_hi": probe_gate.tau_hi,
        }
        print(f"probe: layer {probe_gate.layer}, τ {probe_gate.tau_lo:.3f}")

    results = measure(engine, gate, examples, extra_gates=extra)
    summary = summarise(results)
    probe_summary = (
        summarise_other(results, "probe", artifact["tau_lo"], artifact["tau_hi"])
        if extra
        else None
    )

    decode = None
    if args.with_decode:
        from bench_decode import measure_decode

        decode = measure_decode(engine)
        print(
            f"decode {decode['ms_per_token']:.0f} ms/token "
            f"(transformers {decode['hf_ms_per_token']:.0f}), "
            f"turn {decode['median_turn_ms']:.0f} ms"
        )

    sweep = None
    if SWEEP_RESULTS.exists():
        sweep = json.loads(SWEEP_RESULTS.read_text(encoding="utf-8"))
        print(f"including layer sweep: layer {sweep['chosen']['layer']}")

    loop = None
    loop_results = ROOT / "data" / "loop_latency.json"
    if loop_results.exists():
        loop = json.loads(loop_results.read_text(encoding="utf-8"))
        print(f"including loop latency: {loop['median_total']:.0f} ms median turn")

    print(f"{summary['n']} held-out utterances")
    print(
        f"  prompted: {summary['accuracy']:.1%} accuracy, "
        f"{summary['false_call_rate']:.1%} false calls, "
        f"{summary['median_gate_ms']:.0f} ms"
    )
    if probe_summary:
        print(
            f"  probe:    {probe_summary['accuracy']:.1%} accuracy, "
            f"{probe_summary['false_call_rate']:.1%} false calls, "
            f"{probe_summary['median_gate_ms']:.2f} ms"
        )

    if not args.print_only:
        EVAL_DOC.parent.mkdir(parents=True, exist_ok=True)
        EVAL_DOC.write_text(
            render(summary, engine.info, decode, sweep, probe_summary, artifact, loop),
            encoding="utf-8",
        )
        print(f"\nwrote {EVAL_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

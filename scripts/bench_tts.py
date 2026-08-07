"""Kokoro's share of the latency budget, measured on its own.

PLAN.md section 5 budgets 250 ms to the first audio chunk, and docs/eval.md
measures 1929 ms. `bench_loop.py` gets that number from whole spoken turns,
which needs the GPU, a microphone and a person; this measures the synthesiser
by itself, so a change to the TTS can be checked in seconds rather than by
holding a conversation with it.

What matters here is **time to the first chunk**, not total synthesis. The
speaker starts on piece one while piece two is still being made, so the user
waits for the first piece alone -- which is why `--first-max-chars` is worth
measuring separately from the rest.

    python scripts\\bench_tts.py
    python scripts\\bench_tts.py --provider CPUExecutionProvider

Writes data/tts_latency.json so docs/eval.md can render it. Run it twice --
once before a change and once after -- and compare the two numbers rather
than trusting a claim about which is faster.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tts.kokoro import FIRST_MAX_CHARS, Kokoro, split_sentences  # noqa: E402

RESULTS = ROOT / "data" / "tts_latency.json"

#: Replies shaped like the ones src/format.py actually produces: short, spoken,
#: one or two sentences. Benchmarking on paragraphs would measure a case the
#: response rules forbid.
REPLIES = [
    "Added, due Friday the fourteenth.",
    "Nothing on tomorrow.",
    "It is nine in the morning, Thursday.",
    "COMP4020 is at nine, in Copland G30.",
    "Three things due this week — the closest is the data structures assignment, Friday.",
    "Marked sixty percent done, about two and a half hours left.",
    "Deleted the poster session. Say undo if that was wrong.",
    "Two classes today — intelligent systems at ten, then the studio at two.",
]

#: The budget this is measured against.
BUDGET_MS = 250


def measure(
    kokoro: Kokoro,
    replies: list[str],
    repeats: int = 3,
    first_max_chars: int | None = FIRST_MAX_CHARS,
) -> dict:
    """Time to first chunk, and the whole reply, for each utterance."""
    rows = []

    # One throwaway pass: the first call builds the ONNX graph and loads the
    # voice, which no later turn pays for. bench_loop.py drops its first turn
    # for the same reason.
    kokoro.synthesise("Warming up.")

    for text in replies:
        # Exactly what Kokoro.stream would produce, so this measures the code
        # path the session runs rather than a tidier one.
        pieces = split_sentences(text, first_max_chars=first_max_chars)
        firsts, totals, audio = [], [], 0.0
        for _ in range(repeats):
            started = time.perf_counter()
            first_ms = None
            seconds = 0.0
            for index, piece in enumerate(pieces):
                chunk = kokoro.synthesise(piece)
                if index == 0:
                    first_ms = (time.perf_counter() - started) * 1000
                seconds += chunk.seconds
            totals.append((time.perf_counter() - started) * 1000)
            firsts.append(first_ms)
            audio = seconds
        rows.append(
            {
                "text": text,
                "pieces": len(pieces),
                "first_chars": len(pieces[0]) if pieces else 0,
                "ms_first": statistics.median(firsts),
                "ms_total": statistics.median(totals),
                "audio_seconds": audio,
                # < 1.0 means synthesis is slower than speaking it.
                "realtime_factor": (audio * 1000) / statistics.median(totals),
            }
        )
    return {
        "rows": rows,
        "median_first_ms": statistics.median(r["ms_first"] for r in rows),
        "worst_first_ms": max(r["ms_first"] for r in rows),
        "median_realtime_factor": statistics.median(r["realtime_factor"] for r in rows),
        "budget_ms": BUDGET_MS,
        "first_max_chars": first_max_chars,
    }


def report(summary: dict, provider: str) -> None:
    print(f"provider: {provider}\n")
    print(f"{'first':>8} {'total':>8} {'audio':>7} {'xRT':>6}  reply")
    print("-" * 72)
    for row in summary["rows"]:
        print(
            f"{row['ms_first']:>7.0f}m {row['ms_total']:>7.0f}m "
            f"{row['audio_seconds']:>6.1f}s {row['realtime_factor']:>5.2f}  "
            f"{row['text'][:34]}"
        )
    print("-" * 72)
    median = summary["median_first_ms"]
    print(
        f"median first chunk {median:.0f} ms against a {BUDGET_MS} ms budget "
        f"({median / BUDGET_MS:.1f}x)"
    )
    print(f"worst first chunk  {summary['worst_first_ms']:.0f} ms")
    print(f"median realtime    {summary['median_realtime_factor']:.2f}x")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="time Kokoro's first chunk")
    parser.add_argument("--provider", default=None, help="force an ONNX provider")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--print-only", action="store_true", help="do not write the json")
    parser.add_argument(
        "--no-first-cap",
        action="store_true",
        help="synthesise whole sentences, the pre-M5.1 behaviour, for comparison",
    )
    args = parser.parse_args(argv)

    kokoro = Kokoro(provider=args.provider).load()
    summary = measure(
        kokoro,
        REPLIES,
        repeats=args.repeats,
        first_max_chars=None if args.no_first_cap else FIRST_MAX_CHARS,
    )
    summary["provider"] = kokoro.provider_in_use
    report(summary, kokoro.provider_in_use)

    if not args.print_only:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Measure the whole voice loop from turn_log.

PLAN.md M5 exits on "full loop runs inside the latency budget, measured from
`turn_log`". This is that measurement. It reads real turns rather than timing a
synthetic call, because the budget is about what the user waits through.

    python scripts\\bench_loop.py                     the configured database
    python scripts\\bench_loop.py --db path\\to.db
    python scripts\\bench_loop.py --session <id>      one session only

Writes data/loop_latency.json, which scripts/eval_gate.py renders into
docs/eval.md. Nothing here is typed into prose by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.store.db import connect  # noqa: E402

RESULTS = ROOT / "data" / "loop_latency.json"

#: PLAN.md section 5, in milliseconds. VAD endpointing is not in turn_log --
#: it is a property of the segmenter, not of a turn -- so it is carried here
#: from src/audio/vad.py to keep the total honest.
BUDGET = {
    "vad": 200,
    "ms_asr": 300,
    "ms_prefill": 150,
    "ms_gen": 500,
    "ms_tts": 250,
}
STAGE_NAMES = {
    "vad": "VAD endpoint",
    "ms_asr": "Parakeet",
    "ms_prefill": "Prefill + gate",
    "ms_gen": "Generate",
    "ms_tts": "Kokoro first chunk",
}
TOTAL_BUDGET = sum(BUDGET.values())


def load_turns(conn, session: str | None = None, only_spoken: bool = True) -> list[dict]:
    sql = "SELECT * FROM turn_log WHERE 1=1"
    params: list = []
    if session:
        sql += " AND session_id = ?"
        params.append(session)
    if only_spoken:
        # A turn with no ASR time was typed, and typing is not what the budget
        # is about.
        sql += " AND ms_asr IS NOT NULL"
    sql += " ORDER BY id"
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def summarise(turns: list[dict], drop_first: bool = True) -> dict:
    """Median per stage. The first turn of a session is excluded by default.

    The first turn pays for lazily loading the model, warming the ONNX graphs
    and filling CUDA's caches. Including it measures startup, not a turn.
    """
    usable = turns[1:] if drop_first and len(turns) > 1 else turns
    if not usable:
        raise SystemExit("no spoken turns to measure — run src.session first")

    from src.audio import vad

    stages: dict[str, float] = {"vad": vad.MIN_SILENCE * 1000}
    for column in ("ms_asr", "ms_prefill", "ms_gen", "ms_tts"):
        values = [t[column] for t in usable if t[column] is not None]
        stages[column] = statistics.median(values) if values else 0.0

    totals = []
    for turn in usable:
        totals.append(
            vad.MIN_SILENCE * 1000
            + sum(turn[c] or 0 for c in ("ms_asr", "ms_prefill", "ms_gen", "ms_tts"))
        )

    return {
        "n_turns": len(usable),
        "n_dropped": len(turns) - len(usable),
        "stages": stages,
        "median_total": statistics.median(totals),
        "best_total": min(totals),
        "worst_total": max(totals),
        "budget": BUDGET,
        "budget_total": TOTAL_BUDGET,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def report(summary: dict) -> None:
    print(f"{summary['n_turns']} spoken turns (first {summary['n_dropped']} dropped as warm-up)\n")
    print(f"  {'stage':<20} {'budget':>8} {'measured':>10}   {'':>6}")
    for key, budget in BUDGET.items():
        measured = summary["stages"][key]
        ratio = measured / budget if budget else 0
        flag = "ok" if measured <= budget else f"{ratio:.1f}x over"
        print(f"  {STAGE_NAMES[key]:<20} {budget:>6} ms {measured:>8.0f} ms   {flag:>10}")
    print(
        f"  {'TOTAL':<20} {TOTAL_BUDGET:>6} ms "
        f"{summary['median_total']:>8.0f} ms   "
        f"{summary['median_total'] / TOTAL_BUDGET:.1f}x over"
    )
    print(f"\n  best turn {summary['best_total']:.0f} ms, worst {summary['worst_total']:.0f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure the voice loop")
    parser.add_argument("--db", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--keep-first", action="store_true", help="include the cold turn")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    conn = connect(args.db or config.load().db_path)
    turns = load_turns(conn, args.session)
    summary = summarise(turns, drop_first=not args.keep_first)
    report(summary)

    if not args.print_only:
        RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {RESULTS}")
        print("now run: python scripts\\eval_gate.py")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

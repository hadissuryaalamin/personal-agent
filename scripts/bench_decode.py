"""Where the time in a turn actually goes.

PLAN.md section 5 budgets 1400 ms from end of speech to first audio, of which
150 ms is prefill+probe and 500 ms is generation. This measures the two that
exist at M1, and separates "the wrapper is slow" from "the weights are slow"
by timing the same work three ways.

    python scripts\\bench_decode.py

Imported by scripts\\eval_gate.py --with-decode, which is what writes the
numbers into docs/eval.md. Nothing here is typed into prose by hand.
"""

from __future__ import annotations

import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm import prompts  # noqa: E402
from src.tools import registry  # noqa: E402

TZ = ZoneInfo("Australia/Sydney")
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)

#: A command that produces a full tool call, so the token count is realistic.
SAMPLE = "put comp4020 on thursdays from nine am to eleven am"


def tool_suffix() -> str:
    return (
        f"<|im_start|>user\n{prompts.tool_instruction(registry.schemas())}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def measure_decode(engine, repeats: int = 3) -> dict:
    """Time a real tool pass, then isolate the wrapper's share of the cost."""
    import torch

    messages = prompts.build_messages(NOW, "Australia/Sydney", SAMPLE)
    prefix = engine.prefix_text(messages)
    suffix = tool_suffix()

    # Warm the kernels: the first call through bitsandbytes pays for itself
    # several times over and would otherwise dominate a three-run median.
    warm = engine.prefill(prefix)
    engine.continue_from(warm, suffix, max_new_tokens=8)

    prefill_ms, gen_ms, token_counts = [], [], []
    for _ in range(repeats):
        prefill = engine.prefill(prefix)
        text, ms = engine.continue_from(
            prefill, suffix, max_new_tokens=160,
            stop_when=lambda s: prompts.extract_json(s) is not None,
        )
        n = len(engine.tokenizer(text, add_special_tokens=False).input_ids)
        prefill_ms.append(prefill.ms)
        gen_ms.append(ms)
        token_counts.append(n)

    # The same generation through transformers' own loop. If this is much
    # faster than ours, the wrapper is the problem; if not, the weights are.
    full = prefix + suffix
    ids = engine.tokenizer(full, return_tensors="pt").input_ids.to(engine.model.device)
    want = int(statistics.median(token_counts)) or 40
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        out = engine.model.generate(
            ids, max_new_tokens=want, do_sample=False,
            pad_token_id=engine.tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - started) * 1000
    baseline_tokens = int(out.shape[1] - ids.shape[1])

    median_gen = statistics.median(gen_ms)
    median_tokens = statistics.median(token_counts)

    return {
        "prompt_tokens": int(ids.shape[1]),
        "median_prefill_ms": statistics.median(prefill_ms),
        "median_gen_ms": median_gen,
        "median_out_tokens": median_tokens,
        "ms_per_token": median_gen / max(median_tokens, 1),
        "hf_ms_per_token": baseline_ms / max(baseline_tokens, 1),
        "median_turn_ms": statistics.median(prefill_ms) + median_gen,
    }


def main() -> int:
    from src.llm.engine import Engine

    engine = Engine().load()
    result = measure_decode(engine)
    for key, value in result.items():
        print(f"  {key:<20} {value:.1f}" if isinstance(value, float) else f"  {key:<20} {value}")
    print(
        f"\nours {result['ms_per_token']:.0f} ms/token vs transformers "
        f"{result['hf_ms_per_token']:.0f} ms/token"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

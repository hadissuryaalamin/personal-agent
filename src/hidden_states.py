"""Phase 2: one forward pass per task, keep the last token's hidden state.

The feature vector is the hidden state at the position generation is about to
start, concatenated across every layer plus the embedding output -- 37 x 2560
= 94,720 floats for Qwen3-4B. That tensor already exists during the prefill
the model performs anyway, which is why reading it costs microseconds and why
the method can claim sub-millisecond overhead. Nothing here generates a token.

    python -m src.hidden_states --limit 4    # smoke test
    python -m src.hidden_states              # all 143, about 134 s

Loading, prompt building and the last-token slice all come from model.py, so
there is exactly one definition of "the prompt the agent deploys" and this
file cannot drift away from it.

STORING float16, BUT CHECKING BEFORE IT DOES

fp16 tops out at 65504 and transformer activations carry outliers; an overflow
would become inf silently and poison the probe. Every vector is checked finite
in float32 AND after the cast, and the largest magnitude seen is reported, so
the margin is a number rather than a hope. It measured 418 on this model --
a 156x margin -- which is a fact about this configuration, not a guarantee
about the next one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import model as M
from .dataset import ROOT, load as load_tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=M.DEFAULT_MODEL)
    ap.add_argument("--quant", default="int4", choices=("int4", "int8", "bf16"))
    ap.add_argument("--style", default="deployed", choices=("deployed", "bare"),
                    help="bare = question only; valid when the probe model is "
                         "not the model that answers")
    ap.add_argument("--limit", type=int, default=0,
                    help="extract only the first N tasks, for a smoke test")
    ap.add_argument("--out", default="", help="output .npz")
    args = ap.parse_args()

    import numpy as np
    import torch
    import transformers

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[:args.limit]

    print(f"  model          {args.model}")
    print(f"  quantisation   {args.quant}")
    print(f"  tasks          {len(tasks)}")
    print()

    t0 = time.perf_counter()
    tok, model = M.load(args.model, args.quant)
    print(f"  loaded in      {time.perf_counter() - t0:.1f}s")

    n_layers, hidden, width = M.geometry(model)
    print(f"  geometry       {n_layers + 1} x {hidden} = {width:,} features")
    print(f"  prompt hash    {M.prompt_hash()}")
    print()

    X = np.zeros((len(tasks), width), dtype=np.float16)
    n_tokens = np.zeros(len(tasks), dtype=np.int32)
    biggest = 0.0
    bad: list[str] = []
    times: list[float] = []

    for i, task in enumerate(tasks):
        inputs = tok(M.build_prompt(tok, task["prompt"], args.style),
                     return_tensors="pt").to("cuda")
        n_tokens[i] = inputs.input_ids.shape[1]

        if i == 0:
            # The probe reads the generation-start position. On Qwen3 that is
            # the newline after <|im_start|>assistant; if the template ever
            # changes, this is the line that says so.
            last = tok.decode(inputs.input_ids[0, -1])
            note = "" if last == "\n" else "   <-- not the expected newline"
            print(f"  last token     {last!r}{note}")
            print()

        t = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        vec = M.last_token_state(out)
        del out
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)

        finite = bool(torch.isfinite(vec).all())
        peak = float(vec.abs().max())
        biggest = max(biggest, peak)
        row = vec.to(torch.float16).cpu().numpy()
        if not finite or not np.isfinite(row).all():
            bad.append(f"{task['id']} (finite in fp32: {finite}, max |x| {peak:.1f})")
        X[i] = row
        del vec

        if (i + 1) % 20 == 0 or i + 1 == len(tasks):
            done = sum(times)
            eta = done / (i + 1) * (len(tasks) - i - 1)
            print(f"  {i + 1:>3}/{len(tasks)}   {done:5.1f}s elapsed"
                  f"   {eta:5.1f}s left   {times[-1] * 1000:.0f} ms/pass")

    print()
    warm = sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else times[0]
    print(f"  forward pass   {warm * 1000:.0f} ms median warm"
          f"  ({times[0] * 1000:.0f} ms cold)")
    print(f"  total          {sum(times):.0f} s")
    print(f"  prompt tokens  {n_tokens.min()}-{n_tokens.max()}"
          f"  (gate measured {M.EXPECTED_TOKENS})")
    print(f"  largest |x|    {biggest:.1f}   (fp16 overflows at 65504)")

    meta = {
        "model": args.model,
        "quantisation": args.quant,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "width": width,
        "prompt_hash": M.prompt_hash() if args.style == "deployed" else f"bare-{M.prompt_hash()[:8]}",
        "prompt_style": args.style,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": len(tasks),
        "median_ms": round(warm * 1000, 1),
        "max_abs": round(biggest, 2),
    }

    ok = True
    if bad:
        ok = False
        print()
        for b in bad[:10]:
            print(f"  FAIL           non-finite features: {b}")
    if X.shape[1] != width:
        ok = False
        print(f"  FAIL           width {X.shape[1]}, expected {width}")
    if args.style == "deployed" and abs(int(n_tokens.max()) - M.EXPECTED_TOKENS) > 60:
        print(f"  WARN           prompts are {n_tokens.max()} tokens against"
              f" {M.EXPECTED_TOKENS} at the gate -- the system prompt or the")
        print("                 tool schema has changed since it was measured")

    if not ok:
        print("\n  extraction FAILED, nothing written")
        return 1

    tag = ("" if args.style == "deployed" else f"-{args.style}") + (f"-{args.limit}" if args.limit else "")
    path = Path(args.out) if args.out else ROOT / f"features{tag}.npz"
    np.savez_compressed(
        path,
        X=X,
        y=np.array([t["label"] for t in tasks], dtype=np.int8),
        ids=np.array([t["id"] for t in tasks]),
        group=np.array([t["group"] for t in tasks]),
        difficulty=np.array([t["difficulty"] for t in tasks]),
        source=np.array([t["source"] for t in tasks]),
        n_tokens=n_tokens,
        meta=np.array(json.dumps(meta)),
    )
    print(f"\n  written        {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print("  extraction PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

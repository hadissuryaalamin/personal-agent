"""Phase 4: put words in the model's mouth before it starts talking.

A chat prompt ends with `<|im_start|>assistant\\n` and the model continues from
there. Prefilling means appending our own text to that turn first, so the
model is no longer choosing how to begin -- it is continuing something already
begun. Two strengths:

    soft   a sentence it may ignore
           "I need to use a tool for this question."
           "I can solve this directly without using a tool."

    hard   the opening of the structure itself, which it cannot ignore
           <tool_call>\\n{"name":

THE PREFIX IS NOT THE PAPER'S

The paper prefills `{"name":`. Qwen3 does not emit a bare JSON object -- its
tool calls are wrapped:

    <tool_call>
    {"name": ..., "arguments": ...}
    </tool_call>

so prefilling `{"name":` would open a JSON object inside a tag the model has
been trained to write first, and the result parses as nothing. The prefix here
is `<tool_call>\\n{"name":`, which was checked against the template rather than
assumed.

WHY HARD PREFILL IS WORTH TESTING ON ITS OWN

It needs no probe. It is a fix for a bug this agent actually has: asked
"is it a good time to take a break?", the model answered "...Let me check
what's coming up next." and stopped -- prose where a tool call belonged, in
direct violation of a system prompt that forbids exactly that sentence. The
paper saw the same failure in Llama, where tool calls collapsed to zero and
accuracy fell 83.1% -> 47.9%, and hard prefill recovered it.

So `--mode hard` on the label-1 tasks answers a question that has nothing to
do with the probe: can the format be forced? The probe's job comes after, and
is different -- deciding WHICH turns should get the hard prefill at all, since
forcing it on "hi" would be worse than the disease.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import model as M
from .dataset import ROOT, load as load_tasks

HARD_PREFIX = '<tool_call>\n{"name":'
SOFT_TOOL = "I need to use a tool for this question."
SOFT_DIRECT = "I can solve this directly without using a tool."

TEMPERATURE = 2.0   # p = sigmoid(z / T), as the probe reports it
MAX_NEW = 32        # enough to see whether a tool call starts; not a full answer

# The prose bug: an announcement that a tool is coming, with no tool.
PROSE = re.compile(
    r"\b(let me (check|look|have a look|see)|i'?ll (check|look|take a look)|"
    r"checking your|let'?s (check|see)|i'?m going to check|one (sec|moment))",
    re.I)


def called_tool(text: str) -> bool:
    return "<tool_call>" in text


def build(tok, question: str, prefill: str = "") -> str:
    """The deployed prompt, plus whatever we are putting in the model's mouth.

    Concatenated as text rather than as another message: a message would be
    closed by the template with <|im_end|>, and the model would answer *after*
    our sentence instead of continuing from it.
    """
    return M.build_prompt(tok, question) + prefill


def generate(model, tok, text: str, max_new: int = MAX_NEW) -> str:
    import torch

    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)


def probe_score(model, tok, question: str, clf, style: str = "deployed") -> float:
    """p that this question needs a tool, from the fitted probe.

    A forward pass of its own, which is honest about the cost but not what a
    deployment would do: the states the probe reads are produced by the prefill
    the model performs anyway, so a real integration reads them from that pass
    and adds microseconds. Measuring it separately here keeps the code simple
    and the overhead figure pessimistic.
    """
    import numpy as np
    import torch

    inputs = tok(M.build_prompt(tok, question, style), return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    vec = M.last_token_state(out).to(torch.float16).cpu().numpy()
    z = float(clf.decision_function(np.asarray([vec], dtype=np.float32))[0])
    return 1 / (1 + np.exp(-z / TEMPERATURE))


def ask_ollama(question: str, prefill: str = "", num_predict: int = 64) -> dict:
    """One turn through the deployed stack, optionally with words already in
    the model's mouth.

    Ollama continues from a trailing assistant message -- checked, not assumed.
    The catch is that it does NOT parse the continuation into `tool_calls`,
    because the opening <tool_call> tag came from us rather than from the
    model. So the caller gets the reassembled text and has to read it, which is
    the one change llm.py would need to run this for real.
    """
    import requests

    from . import config
    from .tools import SCHEMA

    msgs = [{"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question}]
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})

    body = {"model": config.OLLAMA_MODEL, "messages": msgs, "tools": SCHEMA,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0}}
    t0 = time.perf_counter()
    r = requests.post(config.OLLAMA_CHAT_URL, json=body,
                      timeout=config.OLLAMA_TIMEOUT).json()
    ms = (time.perf_counter() - t0) * 1000

    m = r.get("message", {})
    content = prefill + (m.get("content") or "")
    native = bool(m.get("tool_calls"))
    return {"content": content, "tool_call": native or called_tool(content),
            "ms": ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="none",
                    choices=("none", "hard", "soft", "probe-hard", "probe-soft"),
                    help="none = the prompt-only baseline")
    ap.add_argument("--backend", default="hf",
                    choices=("hf", "hf-cached", "ollama"),
                    help="ollama = Qwen3-4B probes, qwen2.5:7b answers; "
                         "hf-cached = Qwen3-4B does both, prefix cached")
    ap.add_argument("--style", default="deployed", choices=("deployed", "bare"),
                    help="prompt the PROBE reads; bare is 22x cheaper and only "
                         "valid when the probe is not the model answering")
    ap.add_argument("--model", default=M.DEFAULT_MODEL)
    ap.add_argument("--quant", default="int4", choices=("int4", "int8", "bf16"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--group", default="", help="only this group")
    ap.add_argument("--label", type=int, default=-1, help="only label 0 or 1")
    ap.add_argument("--difficulty", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.group:
        tasks = [t for t in tasks if t["group"] == args.group]
    if args.label in (0, 1):
        tasks = [t for t in tasks if t["label"] == args.label]
    if args.difficulty:
        tasks = [t for t in tasks if t["difficulty"] == args.difficulty]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        print("  no tasks match those filters")
        return 1

    clf = None
    if args.mode.startswith("probe"):
        import joblib
        suffix = "" if args.style == "deployed" else f"-{args.style}"
        blob = joblib.load(ROOT / f"probe{suffix}.joblib")
        clf = blob["model"]
        if args.style == "deployed" and blob["meta"].get("prompt_hash") != M.prompt_hash():
            print("  FAIL           the saved probe was fitted under prompt "
                  f"{blob['meta'].get('prompt_hash')},")
            print(f"                 the agent now deploys {M.prompt_hash()}")
            return 1

    print(f"  mode           {args.mode}"
          + (f"   tau {args.tau}" if clf else ""))
    print(f"  tasks          {len(tasks)}")
    print(f"  max new        {args.max_new} tokens")
    print()

    tok = model = prefix = None
    if clf is not None or args.backend.startswith("hf"):
        tok, model = M.load(args.model, args.quant)
    if args.backend == "hf-cached":
        t0 = time.perf_counter()
        prefix = M.PrefixCache(model, tok, args.style)
        print(f"  prefix cache   {prefix.length} tokens in "
              f"{(time.perf_counter() - t0) * 1000:.0f} ms")

    rows = []
    t0 = time.perf_counter()
    for i, task in enumerate(tasks):
        p = probe_ms = gen_ms = None
        cache = None
        if args.backend == "hf-cached":
            # One pass over the short tail gives the probe its features AND
            # leaves the KV the answer will continue from. This is the paper's
            # actual claim -- the probe rides a computation that had to happen.
            t_p = time.perf_counter()
            cache = prefix.fork()
            with __import__("torch").inference_mode():
                out = model(input_ids=prefix.tail_ids(task["prompt"]),
                            past_key_values=cache, output_hidden_states=True,
                            use_cache=True)
            if clf is not None:
                import numpy as np
                import torch as _t
                vec = M.last_token_state(out).to(_t.float16).cpu().numpy()
                z = float(clf.decision_function(np.asarray([vec], dtype=np.float32))[0])
                p = 1 / (1 + np.exp(-z / TEMPERATURE))
            del out
            cache.crop(prefix.length)
            probe_ms = (time.perf_counter() - t_p) * 1000
            wants_tool = p is None or p >= args.tau
        elif clf is not None:
            t_p = time.perf_counter()
            p = probe_score(model, tok, task["prompt"], clf, args.style)
            probe_ms = (time.perf_counter() - t_p) * 1000
            wants_tool = p >= args.tau
        else:
            wants_tool = True   # unconditional prefill; the probe is not gating

        prefill = ""
        if args.mode in ("hard", "probe-hard") and wants_tool:
            prefill = HARD_PREFIX
        elif args.mode in ("soft", "probe-soft"):
            prefill = SOFT_TOOL if wants_tool else SOFT_DIRECT

        if args.backend == "hf-cached":
            import torch as _t
            seq = prefix.full_ids(task["prompt"])
            if prefill:
                seq = _t.cat([seq, tok(prefill, return_tensors="pt",
                                       add_special_tokens=False).input_ids.to("cuda")], dim=1)
            t_g = time.perf_counter()
            with _t.inference_mode():
                o = model.generate(input_ids=seq, past_key_values=cache,
                                   max_new_tokens=args.max_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            gen = tok.decode(o[0, seq.shape[1]:], skip_special_tokens=False)
            gen_ms = (time.perf_counter() - t_g) * 1000
            full = prefill + gen
            tool = called_tool(full)
        elif args.backend == "ollama":
            res = ask_ollama(task["prompt"], prefill, args.max_new)
            full, tool, gen_ms = res["content"], res["tool_call"], res["ms"]
            gen = full[len(prefill):]
        else:
            gen = generate(model, tok, build(tok, task["prompt"], prefill),
                           args.max_new)
            full = prefill + gen      # the prefix is part of what was produced
            tool = called_tool(full)
            gen_ms = None
        prose = bool(PROSE.search(gen)) and not tool

        rows.append({**{k: task[k] for k in ("id", "prompt", "label", "group",
                                             "difficulty")},
                     "p": None if p is None else round(p, 4),
                     "probe_ms": None if p is None else round(probe_ms, 1),
                     "gen_ms": None if gen_ms is None else round(gen_ms, 1),
                     "prefill": prefill,
                     "tool_call": tool,
                     "prose_bug": prose,
                     "correct": tool == bool(task["label"]),
                     "output": full.replace("<|im_end|>", "").strip()[:300]})

        if (i + 1) % 20 == 0 or i + 1 == len(tasks):
            done = time.perf_counter() - t0
            print(f"  {i + 1:>3}/{len(tasks)}   {done:5.1f}s"
                  f"   {done / (i + 1):.1f}s/task")

    n = len(rows)
    tool_calls = sum(r["tool_call"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    prose = sum(r["prose_bug"] for r in rows)

    print()
    print(f"  tool calls     {tool_calls}/{n}  ({tool_calls / n:.0%})")
    print(f"  correct        {correct}/{n}  ({correct / n:.0%})")
    print(f"  prose bug      {prose}/{n}"
          + ("   <-- announced a tool and did not call one" if prose else ""))

    lat = [r for r in rows if r["probe_ms"] is not None]
    if lat:
        pm = sorted(x["probe_ms"] for x in lat)[len(lat) // 2]
        print(f"  probe          {pm:.0f} ms median")
    lat = [r for r in rows if r["gen_ms"] is not None]
    if lat:
        gm = sorted(x["gen_ms"] for x in lat)[len(lat) // 2]
        print(f"  answer         {gm:.0f} ms median")

    for diff in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["difficulty"] == diff]
        if sub:
            c = sum(r["correct"] for r in sub)
            print(f"    {diff:<7} {c:>3}/{len(sub):<3} correct")

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print()
        print(f"  wrong ({len(wrong)})")
        for r in wrong[:12]:
            got = "called a tool" if r["tool_call"] else "answered directly"
            print(f"    label={r['label']} {r['group']:<13} {got:<17} "
                  f"{r['prompt'][:44]}")

    path = (Path(args.out) if args.out
            else ROOT / f"prefill-{args.backend}-{args.mode}.json")
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  written        {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

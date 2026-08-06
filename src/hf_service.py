"""Qwen3-4B answering as well as probing, with no Ollama at all.

    .\.venv-probe\Scripts\python.exe -m src.hf_service

Then set `LLM_BACKEND=hf` and restart the agent.

WHY THIS EXISTS AS A SERVICE AND NOT AS A BACKEND CLASS

Same reason as probe_service.py: torch, transformers and bitsandbytes are
about 3 GB of wheels that `.venv-agent` deliberately does not have, so that a
research dependency cannot destabilise a working voice assistant. The agent
talks over 127.0.0.1 instead.

It answers in Ollama's dialect -- POST /api/chat, newline-delimited JSON, the
same `message.content` and `message.tool_calls` shape -- so `llm.py` reuses
its streaming loop, its tool loop and its stall guard unchanged. The agent
cannot tell the difference, which is the point.

WHAT THIS BUYS OVER THE SPLIT SETUP

The probe stops costing anything. In the Ollama arrangement it needs its own
forward pass because the model that reads is not the model that answers. Here
they are the same model, so the hidden state the probe wants is produced by
the prefill the answer needs anyway. Reading it is a tensor slice and a
94,720-wide dot product.

And one model sits in VRAM instead of two. Measured over the 132 hard tasks:

    Ollama, no probe          69% routing   1451 ms/turn   1 model
    Ollama + probe service   100% routing   2892 ms/turn   2 models, 8.6 GB
    this                      98% routing   1508 ms/turn   1 model, 3.6 GB

THE PREFIX CACHE IS WHAT MAKES IT COMPETITIVE

1668 of the 1683 prompt tokens are the system prompt and the tool schema,
identical every turn. Recomputing them is 2086 ms; keeping them is 101 ms.
Ollama was never doing less work -- it was doing the same work once. See
model.PrefixCache, including why the cache must be cropped before generating.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 11501

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_tool_calls(text: str) -> list[dict]:
    """Ollama's shape, so llm.py's tool loop needs no special case."""
    calls = []
    for blob in TOOL_CALL.findall(text):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        calls.append({"function": {"name": obj.get("name", ""),
                                   "arguments": obj.get("arguments", {})}})
    return calls


def serve(model_id: str = "", quant: str = "int4", port: int = PORT,
          tau: float = 0.5, probe_style: str = "bare") -> int:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import joblib
    import numpy as np
    import torch
    from transformers import TextIteratorStreamer

    from . import config, model as M
    from .dataset import ROOT
    from .tools import SCHEMA

    model_id = model_id or M.DEFAULT_MODEL
    suffix = "" if probe_style == "deployed" else f"-{probe_style}"
    blob = joblib.load(ROOT / f"probe{suffix}.joblib")
    clf, temp = blob["model"], blob.get("temperature", 2.0)

    print(f"  model          {model_id}  ({quant})")
    print(f"  probe          probe{suffix}.joblib, CV AUROC "
          f"{blob.get('cv_auroc', float('nan')):.3f}, tau {tau}")
    t0 = time.perf_counter()
    tok, model = M.load(model_id, quant)
    print(f"  loaded in      {time.perf_counter() - t0:.1f}s")

    # Built on demand from whatever head the first request actually presents,
    # not from config.SYSTEM_PROMPT. The agent appends a word-limit rule to the
    # system prompt at runtime, so a cache built from the constant alone never
    # matched and every turn silently fell back to the slow path -- with the
    # probe switched off, because the probe rides that same pass.
    caches: dict[str, tuple] = {}

    def prefix_for(head: str):
        if head not in caches:
            t = time.perf_counter()
            ids = tok(head, return_tensors="pt").input_ids.to("cuda")
            with torch.inference_mode():
                base = model(input_ids=ids, use_cache=True).past_key_values
            caches.clear()          # one system prompt at a time; 220 MB each
            caches[head] = (base, ids, ids.shape[1])
            print(f"  prefix cache   {ids.shape[1]} tokens in "
                  f"{(time.perf_counter() - t) * 1000:.0f} ms")
        return caches[head]

    lock = threading.Lock()   # one GPU, one request at a time

    def build_text(messages: list[dict]) -> str:
        return tok.apply_chat_template(messages, tools=SCHEMA,
                                       add_generation_prompt=True, tokenize=False)

    def handle(messages: list[dict], max_new: int):
        """Yields Ollama-shaped chunks."""
        last_user = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
        fresh = messages and messages[-1].get("role") == "user"

        text = build_text(messages)

        # The constant head is the system message and the tool schema, and
        # nothing after them. Derived from the system message the REQUEST
        # carries, not from config.SYSTEM_PROMPT: the agent appends a word
        # limit rule at runtime, so a cache built from the constant alone never
        # matched and every turn fell back to the slow path in silence.
        #
        # Every prompt begins with this, tool-result rounds included, so one
        # cache serves them all.
        system = next((m for m in messages if m.get("role") == "system"), None)
        marker = "⁣MARKER⁣"
        head = build_text(([system] if system else [])
                          + [{"role": "user", "content": marker}]).split(marker)[0]
        base, head_ids, head_len = prefix_for(head)

        import copy
        cache = copy.deepcopy(base)

        p = None
        t0 = time.perf_counter()
        if fresh:
            # A separate 15-token pass, deliberately. Riding the answer's pass
            # would be free, but that probe reads the 1683-token prompt it was
            # trained on -- and the agent's runtime system prompt is not that
            # prompt. The bare probe reads the question alone, so it cannot
            # drift when the system prompt changes, and it scores better on the
            # hard slice besides (0.988 against 0.977). 76 ms is a fair price
            # for a number that stays true.
            bare = tok(M.build_prompt(tok, last_user, "bare"),
                       return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model(**bare, output_hidden_states=True)
            vec = M.last_token_state(out).to(torch.float16).cpu().numpy()
            z = float(clf.decision_function(np.asarray([vec], dtype=np.float32))[0])
            p = 1 / (1 + np.exp(-z / temp))
            del out

        probe_ms = (time.perf_counter() - t0) * 1000

        prefill = ""
        if p is not None:
            prefill = (config.PREFILL_HARD if p >= tau else config.PREFILL_SOFT)
            log.info("probe %.2f in %.0f ms -> %s: %s", p, probe_ms,
                     "tool" if p >= tau else "direct", last_user[:50])

        ids = tok(text + prefill, return_tensors="pt").input_ids.to("cuda")
        forcing_tool = prefill == config.PREFILL_HARD

        if forcing_tool:
            # The output is JSON, not speech. Generate it whole, parse it, and
            # hand back tool_calls -- exactly what Ollama would have sent.
            with torch.inference_mode():
                o = model.generate(input_ids=ids, past_key_values=cache,
                                   max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            raw = prefill + tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
            calls = _parse_tool_calls(raw)
            if calls:
                yield {"message": {"role": "assistant", "content": "",
                                   "tool_calls": calls}}
                return
            log.warning("prefilled reply did not parse: %r", raw[:120])
            yield {"message": {"role": "assistant", "content": raw}}
            return

        streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                        skip_special_tokens=True)
        kw = dict(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                  temperature=None, top_p=None, top_k=None,
                  pad_token_id=tok.eos_token_id, streamer=streamer)
        if cache is not None:
            kw["past_key_values"] = cache
        threading.Thread(target=lambda: model.generate(**kw), daemon=True).start()

        buf = ""
        for piece in streamer:
            buf += piece
            yield {"message": {"role": "assistant", "content": piece}}

        # A tool call can still appear unprefilled, when the probe said no but
        # the model disagreed. Report it the same way.
        calls = _parse_tool_calls(buf)
        if calls:
            yield {"message": {"role": "assistant", "content": "",
                               "tool_calls": calls}}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))) or b"{}")
            messages = body.get("messages", [])
            max_new = int((body.get("options") or {}).get("num_predict", 160))

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write(obj):
                data = (json.dumps(obj) + "\n").encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

            try:
                with lock:
                    for chunk in handle(messages, max_new):
                        write(chunk)
                write({"message": {"content": ""}, "done": True})
            except Exception as e:
                log.exception("generation failed")
                write({"message": {"content": ""}, "done": True, "error": str(e)})
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *a):
            pass

    srv = HTTPServer((HOST, port), Handler)
    srv.daemon_threads = True
    print(f"\n  listening      http://{HOST}:{port}/api/chat")
    print("  set LLM_BACKEND=hf in .env, then restart the agent")
    print("  Ctrl+C to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--quant", default="int4", choices=("int4", "int8", "bf16"))
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--probe-style", default="bare",
                    choices=("deployed", "bare"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return serve(args.model, args.quant, args.port, args.tau, args.probe_style)


if __name__ == "__main__":
    sys.exit(main())

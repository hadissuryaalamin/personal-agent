"""The probe, as a service the agent can call.

    .\.venv-probe\Scripts\python.exe -m src.probe_service        # start it
    curl 127.0.0.1:11500/probe -d "{\"question\":\"am I busy?\"}"

WHY A SERVICE AND NOT AN IMPORT

The probe needs torch, transformers and bitsandbytes -- about 3 GB of wheels
that `.venv-agent` deliberately does not have. That separation is not an
accident of setup: it exists so a research dependency cannot destabilise a
voice assistant that works. Importing the probe into the agent would erase it.

So the agent talks to this over 127.0.0.1 instead, and the hop is free next to
what it is buying. What is NOT free is the model: 3.6 GB of VRAM held
alongside whatever answers, and about 25 s to load. Start it before the agent
if you want the first question probed.

If this is not running, `ask()` returns None and the agent behaves exactly as
it did before -- no probe, no prefill, no error. That is the whole point of
the fallback: a research component may not be able to take the assistant down.

WHY THE BARE PROMPT

The probe reads a 15-token prompt, not the agent's 1683-token one, because it
is not riding the agent's forward pass -- it has its own. Measured: 76 ms
against 2098 ms, and the bare probe is slightly MORE accurate on the hard
slice (0.988 against 0.977 AUROC). The tool schema is identical across every
question, so it carries nothing that separates them.

This is only legitimate because these labels are about ACCESS. Whether a
question needs the user's calendar is a property of the question, not of the
model answering it -- which is also why a probe reading Qwen3-4B can gate
qwen2.5:7b at all.
"""

from __future__ import annotations

import json
import logging
import sys

log = logging.getLogger(__name__)

HOST = "127.0.0.1"          # never "localhost": see config.OLLAMA_URL
PORT = 11500
_warned = False


# --- The client half. No torch here, so agent code can import this file. -----


def ask(question: str, url: str = "", timeout: float = 5.0) -> float | None:
    """p that this question needs a tool, or None if the service is not up."""
    global _warned
    import requests

    try:
        r = requests.post(f"{url or f'http://{HOST}:{PORT}'}/probe",
                          json={"question": question}, timeout=timeout)
        r.raise_for_status()
        return float(r.json()["p"])
    except Exception as e:
        if not _warned:
            _warned = True
            log.info("probe service unavailable (%s); routing tools by prompt "
                     "alone, as before", type(e).__name__)
        return None


# --- The server half --------------------------------------------------------


def serve(model_id: str = "", quant: str = "int4", style: str = "bare",
          port: int = PORT) -> int:
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import joblib
    import numpy as np
    import torch

    from . import model as M
    from .dataset import ROOT

    model_id = model_id or M.DEFAULT_MODEL
    suffix = "" if style == "deployed" else f"-{style}"
    blob = joblib.load(ROOT / f"probe{suffix}.joblib")
    clf = blob["model"]
    temperature = blob.get("temperature", 2.0)

    print(f"  model          {model_id}  ({quant})")
    print(f"  probe          probe{suffix}.joblib, fitted on "
          f"{blob.get('n_train', '?')} items, CV AUROC "
          f"{blob.get('cv_auroc', float('nan')):.3f}")
    t0 = time.perf_counter()
    tok, model = M.load(model_id, quant)
    print(f"  loaded in      {time.perf_counter() - t0:.1f}s")

    def score(question: str) -> tuple[float, float]:
        t = time.perf_counter()
        inputs = tok(M.build_prompt(tok, question, style),
                     return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        vec = M.last_token_state(out).to(torch.float16).cpu().numpy()
        z = float(clf.decision_function(np.asarray([vec], dtype=np.float32))[0])
        return 1 / (1 + np.exp(-z / temperature)), (time.perf_counter() - t) * 1000

    # Warm it. The first pass costs ~1.3 s of kernel autotuning, and paying
    # that during the user's first question would look exactly like a hang.
    p, ms = score("warming up")
    print(f"  first pass     {ms:.0f} ms (cold)")
    p, ms = score("what is my next class")
    print(f"  warm pass      {ms:.0f} ms   p={p:.2f} for a question that needs a tool")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                q = json.loads(body or b"{}").get("question", "")
                p, ms = score(q)
                payload = {"p": round(p, 4), "ms": round(ms, 1)}
                log.info("probe %.2f in %.0f ms: %s", p, ms, q[:60])
            except Exception as e:  # never take the agent down from here
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                return
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):    # the handler logs its own line already
            pass

    srv = HTTPServer((HOST, port), Handler)
    print(f"\n  listening      http://{HOST}:{port}/probe")
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
    ap.add_argument("--style", default="bare", choices=("bare", "deployed"))
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return serve(args.model, args.quant, args.style, args.port)


if __name__ == "__main__":
    sys.exit(main())

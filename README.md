# personal-agent

A local, voice-driven scheduling assistant. You hold a hotkey, say *"add the
data structures assignment, due next Friday, about six hours of work"*, and it
lands in a database on your own machine. Ask *"what's on tomorrow?"* and it
answers out loud. Nothing leaves the computer.

The interesting part is not the CRUD — it's how the agent decides **whether to
call a tool at all**. Instead of asking the language model to announce its own
intent in text, we read the model's internal activations mid-forward-pass and
classify them directly.

## Pipeline

```
 mic ─► VAD ─► Parakeet ASR ─► Qwen3-4B-Instruct (prefill)
                                        │
                                        ├─► hidden state h_L ─► probe
                                        │                        │
                    ┌───────────────────┴────────────┬───────────┘
                    │                                │
              score < τ_lo                     score ≥ τ_hi
              "just talking"                   "wants a tool"
                    │                                │
                    │                        2nd pass, same KV cache
                    │                        ► {"tool": …, "args": …}
                    │                                │
                    │                        schedule tools ─► SQLite
                    │                                │
                    └────────────► reply text ◄──────┘
                                        │
                          τ_lo ≤ score < τ_hi ─► ask a clarifying question
                                        │
                                  Kokoro TTS ─► speaker
```

One model load, one prefill. The probe reads the hidden state that the forward
pass already produced, so the gate costs a matrix multiply rather than a second
inference.

## Stack

| Stage | Component | Notes |
|---|---|---|
| Voice activity | Silero VAD (ONNX) | segments turns inside a session |
| ASR | `nvidia/parakeet-tdt-0.6b-v2` | English only; sherpa-onnx runtime on Windows |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` | HF transformers + torch, bf16 |
| Tool gate | logistic probe on layer *L* hidden state | scikit-learn, trained offline |
| Store | SQLite | soft deletes + audit log |
| TTS | Kokoro 82M (ONNX) | streamed sentence by sentence |

**Why transformers and not Ollama:** Ollama and llama.cpp do not expose
per-layer hidden states. The entire premise of this project depends on reading
them, so the model is loaded directly through HF transformers.

## What it schedules

- **Classes** — recurring weekly within a term, with per-date exceptions
  (cancelled, moved, room change).
- **Assignments** — due date, estimated working hours, progress percentage.
  Hours remaining is derived, never stored.
- **Reminders** — one-off, read back on request. They do not fire on their own
  in v1; a proactive daemon is a later milestone.

The agent is always told the current date and time, and every temporal
expression you speak ("next Friday", "in two weeks") is resolved in Python
against that timestamp. The model never does date arithmetic.

## Requirements

- Windows 11, Python 3.12 (3.13+ has no torch wheel for this CUDA build yet)
- NVIDIA GPU. 8 GB VRAM is enough: the loader falls back to 4-bit NF4 when
  bf16 will not fit, which is still one HF load and still exposes hidden
  states. bf16 needs ~9 GB free and is picked automatically when it fits.
- ~8 GB of disk for the weights, in `models/` (not tracked in git). If that
  volume is tight, `models/` can be a directory junction somewhere roomier —
  nothing in the code cares.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# torch first: an RTX 50-series card needs the CUDA 12.8 build, which is not
# the one on PyPI.
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt

python scripts\fetch_models.py        # one-time, ~8 GB
python scripts\fetch_models.py --check
python -m src.store.db --init         # creates data\agent.db
python -m src.llm.engine --info       # confirms the weights load
```

## Run

```powershell
# text mode — no audio devices needed, fastest way to work on tools
python -m src.cli

# same, but never loads the weights: typed tool calls only
python -m src.cli --no-model

# voice
python -m src.session
python -m src.session --devices          # which microphone?
python -m src.session --wav command.wav  # same pipeline, no microphone
```

In text mode you can either say what you want (`what have I got due this
week`) or type the tool call directly (`list_assignments due_before="this
week"`). Both take the same path through the registry, so confirmations and
the audit log behave identically; the typed form is what keeps the tools
testable without the model.

In voice mode the session opens immediately and the VAD handles each turn after
that. Say goodbye, or press Ctrl+C, to close. (A global hotkey is still an open
question in PLAN.md — it needs a dependency that can grab keys outside the
terminal, and nothing else is blocked on it.)

`--wav` runs the identical pipeline over a recording, which is how the voice
path is tested without a microphone in the room:

```powershell
python scripts\make_test_audio.py        # synthesises commands with Windows TTS
python -m src.session --wav data\test_audio\conversation.wav
```

## Repo layout

```
src/
  audio/
    capture.py  microphone, wav read/write
    playback.py speaker queue, interruptible mid-word
    vad.py      Silero turn segmentation
  asr/parakeet.py   speech to text
  tts/kokoro.py     text to speech, a sentence at a time
  llm/
    engine.py   the single model load, prefill + continuation
    prompts.py  system prompt, gate question, tool schemas
    gate.py     prompted gate now, probe gate at M3
    agent.py    one turn: prefill, gate, extract a call
  tools/        registry, schedule tools, fuzzy matching
  store/        SQLite schema, audit log, undo
  timeparse.py  ALL relative-date resolution lives here
  format.py     tool result -> something worth saying out loud
  turnlog.py    every turn writes a row, no exceptions
  dataset.py    the probe's seed data and its 80/20 split
  hidden.py     per-turn hidden states on disk
  session.py    the voice turn loop                          (M4)
  cli.py        text REPL (same tools, no audio)
scripts/
  fetch_models.py  one-time download; the only network access
  eval_gate.py     measures both gates, rewrites docs/eval.md
  sweep_layers.py  which layer the probe should read
  train_probe.py   trains it and tunes τ_lo/τ_hi on validation
  bench_decode.py  where the time in a turn goes
  bench_loop.py    the whole loop, measured from turn_log
  make_test_audio.py  synthesises spoken commands for testing
data/
  probe/*.jsonl    the seed dataset (tracked)
  layer_sweep.json the chosen layer and why (tracked)
  agent.db         (gitignored)
  probe.joblib     the trained probe (gitignored — tied to the weights)
  hidden/          per-turn hidden states (gitignored)
  hidden_cache.npz dataset hidden states, rebuilt on demand (gitignored)
models/         model weights (gitignored)
docs/eval.md    generated — do not edit by hand
tests/          test_hardware.py needs weights; the rest do not
```

## Status

**M0 through M4 complete.** Speech goes in, the probe gates it, and the write
lands:

```powershell
pytest                            # 362 tests, no GPU, no audio, no network
pytest -m hardware                # 30 more, needs the weights

python scripts\sweep_layers.py    # which layer to read
python scripts\train_probe.py     # train it, tune the thresholds
python scripts\eval_gate.py       # regenerates docs/eval.md

$env:AGENT_GATE = "probe"         # use it
python -m src.session
```

All thirteen v1 tools are implemented, dates resolve in `src/timeparse.py`,
every turn writes a `turn_log` row with its hidden states on disk, and every
write is undoable. The model is loaded once through transformers and prefilled
once per turn; the probe reads layer 18 of the hidden state that prefill
already produced.

**D1 holds.** Both gates run on the *same* prefill for the same 127 held-out
utterances, so only the gate differs:

| | Prompted | Probe |
|---|---|---|
| Accuracy | 60.6% | **98.4%** |
| False call | 45.3% | **3.1%** |
| Gate latency | 142 ms | **0.96 ms** |

A TF-IDF bag of words on the same split gets 85.8%, and layer 0 — the raw
embedding — sits at chance, so the probe is reading something the model
*computes* rather than vocabulary or token identity.

Three things worth knowing before trusting any of it:

- **The uncertainty band is empty.** Tuned on validation, τ_lo and τ_hi land on
  the same value: hand-written utterances separate too cleanly to leave a range
  to be unsure in. The three-way branch is wired and tested but has two live
  arms today. Real ASR output at M4 is what should change that.
- **The probe was fitted to the same distribution it is scored on**, and no
  utterance in the dataset has been through ASR. [docs/eval.md](docs/eval.md)
  says what this does and does not prove.
- **The latency budget is missed by 5×.** PLAN.md section 5 allows 1400 ms from
  end of speech to first audio; a real turn takes about 7 s. ASR and the gate
  are fine; generation and TTS are each ~7.7× over. The probe does not fix this
  — the cost is *generating* the tool call and *synthesising* the reply, not
  gating. See [docs/eval.md](docs/eval.md) for the stage-by-stage table.
- **Tool selection is now the weak link, not the gate.** On mangled ASR the
  probe gates correctly and the second pass then picks the wrong tool. A write
  that does not happen is reported as a read that did.
- **Barge-in needs headphones.** The microphone stays open while the agent
  speaks, and without echo cancellation it will hear itself through speakers
  and cut itself off. `--no-barge-in` disables it.

**Editing `prompts.SYSTEM` invalidates the trained probe** — it reads
activations produced by a prefix that starts with that message. The artefact
records a fingerprint of the prompt and refuses to load against a different
one, and the hidden-state cache is keyed on it too. Changing a wording costs a
re-sweep and a retrain; this is not optional, and it moved the chosen layer
from 18 to 17 the one time it happened.

`AGENT_GATE` defaults to `prompted`, because the probe artefact is tied to
specific weights and is not in the repo. Train it once, then switch.

Not built yet: speech *out* — Kokoro, sentence streaming and barge-in arrive at
M5, so today the agent answers in text.

See [PLAN.md](PLAN.md) for the spec, data model, tool contracts, and
milestones; [CLAUDE.md](CLAUDE.md) for the rules any agent working in this repo
must follow.

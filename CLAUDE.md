# CLAUDE.md

A local, offline voice assistant for Windows: hotkey → mic → VAD → STT → LLM →
TTS → speaker. English only, no network.

| Document | What it answers |
|---|---|
| [README.md](README.md) | How to install, run, and configure it |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Why each decision was made, with the measurement |
| [PROBE.md](PROBE.md) | The PROBE&PREFILL replication, phase by phase |
| [PLAN.md](PLAN.md) | **What is being built right now.** Read this before writing code |

## The repo is mid-rewrite

The runtime is being rebuilt from zero on the all-HF architecture. `PLAN.md`
holds the build order, the acceptance criteria, and the list of measured facts
the rewrite is not allowed to lose. Anything in `src/` is v1 unless `PLAN.md`
says that phase has landed.

**Nothing in `src/` has been committed.** The `agent/` → `src/` rename is
staged but not committed; `tray.py`, both services, all six research modules,
`PROBE.md` and `data/` are untracked. Treat destructive git commands here as
data loss until Phase 0 of `PLAN.md` is done.

## Environments

Three virtualenvs. Picking the wrong one is the most common way to waste ten
minutes here.

| venv | For | Has torch? |
|---|---|---|
| `.venv-agent` | The agent runtime. `pip install -e .` | no, deliberately |
| `.venv-probe` | Research and the GPU services. `requirements-probe.txt` | yes, ~3 GB |
| `.venv` | Leftover Jupyter environment. Not used | — |

`.venv-agent` stays free of torch so a research dependency cannot destabilise
a working voice assistant. **Never add torch, transformers, or bitsandbytes to
`pyproject.toml`.** The agent reaches the GPU models over `127.0.0.1` instead.

torch must come from the cu128 index — this card is Blackwell (sm_120) and the
default PyPI wheel reports `cuda.is_available()` as `True` and then fails on
the first real operation. See the header of `requirements-probe.txt`.

## Commands

```powershell
# The agent
.\.venv-agent\Scripts\python.exe -m src.agent

# The GPU services (start before the agent; neither is part of its process)
.\.venv-probe\Scripts\python.exe -m src.probe_service   # probe only, port 11500
.\.venv-probe\Scripts\python.exe -m src.hf_service      # brain + probe, port 11501

# Every module runs on its own
.\.venv-agent\Scripts\python.exe -m src.vad
.\.venv-agent\Scripts\python.exe -m src.stt
.\.venv-agent\Scripts\python.exe -m src.tts "hello"
.\.venv-agent\Scripts\python.exe -m src.llm "hi"
.\.venv-agent\Scripts\python.exe -m src.audio
.\.venv-agent\Scripts\python.exe -m src.tools
.\.venv-agent\Scripts\python.exe -m src.offline_check

powershell -File scripts\status.ps1                     # is it running?
Get-Content logs\agent.log -Tail 20 -Wait               # watch it live
```

## Rules that come from real failures

**Stop the scheduled task before running the agent by hand.** It holds a global
hotkey and refuses to start twice. Two agents fight over the mic and the
symptom reads as "the feature is broken", not "there are two agents".

```powershell
Stop-ScheduledTask -TaskName PersonalAgent
```

**Editing a file does not touch a running process.** Python loads source at
start. The second line of the log names the running commit; check it before
concluding a fix did not work.

**`127.0.0.1`, never `localhost`.** Windows resolves `localhost` to `::1`
first and these services listen on IPv4 only, so every call pays the IPv6
timeout — measured at 2819 ms against 764 ms.

**`memory/events.json` is real user data**, gitignored and not backed up.
Never clear, overwrite, or re-import over it. Work against a copy.

**Give the model a decision, never a calculation.** Choosing a tool and mapping
"the next two weeks" to `days=14` it does well. Comparing timestamps, filtering
a list, keeping a count it does badly — every one of those belongs in Python.
This is why `get_next` is its own tool rather than a flag on `get_schedule`.

**Measure the value, not its existence.** Twice in this project a test passed
while the answer was wrong, because the criterion was only "there is an
answer".

## Writing code here

- English throughout: code, comments, docstrings, commit messages, documents.
- `from __future__ import annotations` at the top of every module.
- A module docstring says what the file is for **and the failure that shaped
  it**. See `src/tools.py` and `src/hf_service.py` for the register.
- Comments carry the number. "Measured here: 119 ms goes missing from a
  2-second tone" earns its place; "pad the audio" does not.
- Never claim a measurement that was not run. If it is a guess, say so.
- Commit subjects are a sentence naming the symptom or the behaviour, not a
  conventional-commits prefix: *"The stall guard leaked whenever a second
  sentence followed"*, *"status.ps1: false alarm when the launcher crosses a
  second boundary"*.
- Do not commit `data/features*.npz` or `data/probe*.joblib` — they are
  rebuilt by `src/hidden_states.py` in about 134 s.

## Working with the user

- Reply in Indonesian. Everything written into the repo stays English.
- Hand over manual verification **one step at a time** and wait for the result
  before giving the next. Do not dump a checklist.
- Any background or long-running process comes with its `Get-Content -Wait`
  command in the same message, unasked.

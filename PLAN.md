# personal-agent — Build Plan (v2)

The runtime is being rewritten from zero. This document is the spec: build in
the order below, and where a decision is ambiguous take the default written
here rather than stopping to ask.

What is being rewritten is the **agent runtime**. The probe research and its
measurements are not — they are the asset this project actually accumulated,
and Phase 0 exists to make sure a rewrite cannot take them with it.

## Why v2 exists

Two things came out of the research in [PROBE.md](PROBE.md) that v1 cannot
reach by patching:

**The brain and the probe should be one model.** v1 runs qwen2.5:7b in Ollama
and, when the probe is on, a second Qwen3-4B beside it — 8.6 GB on an 8 GB
card, and the probe pays for a forward pass the answer was going to make
anyway. Measured end to end over the 132 hard tasks:

| | routing | turn | models in VRAM |
|---|---:|---:|---|
| Ollama, no probe | 69% | 1451 ms | 1 — qwen2.5:7b, ~5 GB |
| Ollama + probe service | 100% | 2892 ms | 2 — **8.6 GB** |
| **All-HF, prefix cached** | **98%** | **1508 ms** | 1 — Qwen3-4B, 3.6 GB |

98% against 69% for 57 ms a turn. The all-HF path exists in the working tree
(`src/hf_service.py`) but sits behind `LLM_BACKEND=hf` as an experiment, while
every default, the system prompt, and the whole tool loop are still shaped
around Ollama. v2 makes it the architecture rather than an option.

**The store can be read and added to, but not corrected.** `events.py` has no
update operation at all. The v1 working tree closed part of this — `mark_done`,
`log_hours` and `remove_entry` are already tools — so the real remaining gap is
narrow and specific: *"push that deadline to Friday"* has no path, and
inventing one while reorganising files is how a data store quietly loses an
entry.

A third reason, unglamorous: **there are no tests.** Not one, in 41 commits.
ARCHITECTURE.md §8 records that twice a test passed while the answer was wrong
— those tests were ad-hoc scripts that no longer exist.

## Target

```
                      .venv-agent, no torch          .venv-probe, torch cu128
   ┌──────────────────────────────────────┐      ┌──────────────────────────┐
   │ src/  the agent                      │      │ serve/  GPU services     │
   │                                      │      │                          │
   │  hotkey → mic → VAD → Parakeet ──────┼─────▶│  Qwen3-4B, int4, 3.6 GB  │
   │                                      │ 127. │   ├ prefix cache         │
   │  speaker ◀── Kokoro ◀── llm.py ◀─────┼─0.0.1│   ├ bare probe, 76 ms    │
   │                          │           │      │   └ prefill, then answer │
   │                       tools.py       │      └──────────────────────────┘
   │                          │           │
   │                      events.json     │      research/ — not imported by
   └──────────────────────────────────────┘      src/ or serve/ at runtime
```

```
src/          the agent runtime. .venv-agent. never gains torch.
serve/        model loading, prefix cache, the two HTTP services. .venv-probe.
research/     the replication: dataset, labelling, features, probe, prefill.
tests/        new.
```

The split is not cosmetic. Today the lower half of `src/` is research code that
happens to share a folder, and both README and ARCHITECTURE have to apologise
for it. `serve/` imports `src.config` and `src.tools` **on purpose** — a probe
built against anything other than the prompt the agent actually deploys is
measuring a different system — and that import direction is the only one
allowed.

---

## Phase 0 — Safeguard. Nothing is deleted before this is done.

No new code. This phase exists because the entire research pipeline, both
services, `tray.py`, `PROBE.md`, and the `agent/` → `src/` rename are sitting
in the working tree with **no commit behind them**.

1. Commit the working tree in honest, separable commits — the rename, the
   research pipeline, the services, the tray, the docs.
2. Tag it `v1`.
3. Fix `requirements-probe.txt`: its header tells you to install
   `probe\requirements.txt`, which does not exist. `probe/` holds one stale
   `data/oof.json` and should go.
4. Add the four keys `config.py` reads but `.env.example` never mentions:
   `HF_SERVICE_URL`, `HF_MODEL`, `PREFILL_HARD`, `PREFILL_SOFT`.

**Done when** `git status` is clean, `git show v1 --stat` accounts for all 20
modules in `src/` — 12 runtime, 6 research, 2 services — and
`python -m src.offline_check` still passes on the tagged tree. Only then does
Phase 1 start, on a branch.

## Phase 1 — Foundation: config, events, tools

Pure Python. No models, no audio, no GPU, no network. This is the only layer
that can be tested without hardware, so it is also where the test suite starts.

- `config.py` — every constant, `.env` overrides, offline enforcement at import
  rather than at a single entry point.
- `events.py` — the store. Carry the v1 schema over unchanged: one row per
  occurrence, `start` as either a date or a date-and-time, ids derived from
  `kind-title-start`, atomic write through a temp file.
- **New: a real update operation.** `events.update(id, **fields)` that rewrites
  a row in place, keeps the id stable when `start` moves, and refuses silently
  to create a row that did not exist. Nothing else in this project may edit a
  row by remove-then-add.
- `tools.py` — one registry, two doors (model schema, argparse). Seven tools:
  the six from v1 plus `reschedule`.
- `text.py` — the normaliser.

**Done when**

- `pytest` passes with the store exercised against a temp file, never
  `memory/events.json`.
- Overrun, negative logs, and the zero floor are covered by tests, not by hand
  (`11h of 8h, 3h over`; overrun contributes 0 to the outstanding total).
- `_resolve` ambiguity is a test: two matching titles must list both and change
  nothing.
- Every tool is reachable from both doors, and a test asserts the two cannot
  drift — the schema and the CLI are generated from the same registry.

## Phase 2 — serve: the model, the cache, the probe

- `serve/model.py` — load Qwen3-4B int4, build prompts, `PrefixCache`.
- `serve/hf_service.py` — POST `/api/chat`, newline-delimited JSON, Ollama's
  `message.content` / `message.tool_calls` shape so the agent cannot tell the
  difference.
- `serve/probe_service.py` — the probe alone, kept for the Ollama arrangement
  and for measuring one against the other.

Three things from PROBE.md that are easy to lose in a rewrite:

- **`cache.crop(prefix_len)` before generating.** Hand `generate()` a sequence
  already cached in full and it emits garbage rather than failing:
  `A111331411111 AI1113142140`.
- **The cache head comes from the request, not from `config.SYSTEM_PROMPT`.**
  The agent appends a word-limit rule at runtime; a cache built from the
  constant never matches, and every turn falls back to the slow path in
  silence, with the probe off, because the probe rides that same pass.
- **The bare probe, not the deployed one.** 0.988 against 0.977 on the hard
  slice, for a separate 76 ms pass, and it cannot drift when the system prompt
  changes.

**Done when** the service scores **≥ 95% routing on the 132 hard tasks**, at
**≤ 1600 ms a turn**, holding **≤ 4 GB of VRAM**, with the prefix cache
confirmed live in the log (101 ms, not 2086 ms).

## Phase 3 — llm.py: one brain interface

- `hf` is the default. `ollama` and `claude` stay behind the same `chat()` /
  `chat_stream()`; a backend without streaming yields one whole chunk so the
  caller never knows.
- The tool loop, bounded at `TOOL_MAX_ROUNDS`.
- Sentence-by-sentence streaming, with the splitter that knows `3 p.m.` is not
  a sentence end — that is precisely the form Parakeet produces.
- The stall guard: hold the first sentence back until it is known whether a
  tool call came with it, and drop it unsaid if it reads as a stall or as code.
  v1's version leaked whenever a second sentence followed (`e5dc6e5`); the
  rewrite gets a test for that case specifically.
- **Soft prefill has never been run.** Seven prose bugs survive at τ = 0.5, all
  of them questions routed to "answer directly" where the model announced a
  check anyway. Measure it here; if it does not help, delete `PREFILL_SOFT`
  rather than leaving an untested branch in the tree.

**Done when** the hard set produces **0 prose bugs**, malformed tool calls
recover **100%**, and a deletion is never executed without the model having
first said which entry it means.

## Phase 4 — Voice I/O

`audio.py`, `vad.py`, `stt.py`, `tts.py`. Ports of v1, which works. Do not
redesign these; the numbers in the table below are what they are because of
choices that took measurement to find.

**Done when** the VAD renders a sentence with a pause in the middle as **one**
line and a cough as none, playback loses **< 20 ms** off a 2-second tone, and
Parakeet holds ~0.4 s a sentence at 0 VRAM.

## Phase 5 — The shell

`agent.py`: hotkey backends, toggle and session mode, the pipeline thread, the
OS file lock, rotating logs, the `build:` line. `tray.py`.

**Done when** first sound on *"what should I work on today?"* is **≤ 4 s**, a
second instance refuses to start and names the holder's PID, and killing the
agent with `-9` does not block the next start.

## Phase 6 — Ops and documentation

`scripts/`, `offline_check.py`, autostart, and rewriting README.md and
ARCHITECTURE.md against what v2 actually is. ARCHITECTURE's module map is
already stale — it still lists `main.py` at 672 lines and knows nothing of the
tray, the probe, or the services.

**Done when** `offline_check` runs 7/7 with **zero outbound connection
attempts** under a blocked-socket test, and no document names a file that does
not exist.

---

## Facts the rewrite is not allowed to lose

Every line here cost a measurement or a real failure. If v2 contradicts one,
that is a regression, not a simplification.

| | |
|---|---|
| Hotkey must be a **single key** | A modifier combination makes Windows send spurious UP/DOWN pairs — 18 in 3 s. The recording shatters |
| `pynput`, not `keyboard` | `keyboard`'s dispatch never fires on this machine, and it needs admin |
| `127.0.0.1`, never `localhost` | IPv6 first, IPv4-only listener: 2819 ms against 764 ms |
| Pad playback 0.2 s at both ends | 119 ms goes missing from a 2-second tone without it |
| Pad **once** per utterance, not per sentence | Per-sentence padding cost 0.83 s of silence on a three-sentence reply |
| Stream by sentence | First sound 17.4 s → 8.2 s, and 11.8 s → 3.4 s. The total barely moves; the silence is what goes |
| `REPLY_MAX_WORDS`, not a token cap | 41 words → 17.8. A token cap adds nothing (17.0 vs 17.0) and cuts mid-word |
| Parakeet on **CPU** | GPU is 0.2 s faster and takes VRAM the brain needs |
| Release idle models | Weights are idle data. Reload costs 2–7 s warm, 13–35 s cold; say *"one moment"* rather than going quiet |
| Mic closed while speaking | The alternative is echo cancellation or mandatory headphones |
| VAD: three stop reasons, not one | Collapse them and a single cough closes the session |
| Silence deadline is **not** reset by short sounds | A noisy room would hold models in memory all day |
| `get_next` is its own tool | Marked rows were not enough: at 19:25 it still offered the 15:30 tutorial |
| The tool `description` **is** the prompt | 30/30 called when they should, 0/10 when they should not |
| Prefill is `<tool_call>\n{"name":` | The paper's `{"name":` opens JSON inside a tag Qwen writes first, and parses as nothing |
| τ = 0.5 is not tuned | It is the classifier's own boundary; the sweep confirmed the default |
| Hard prefill **unconditionally** is worse than nothing | 70% → 42%. It fixes the broken half and destroys the healthy half |
| Ids derived from `kind-title-start` | Re-importing replaces rather than duplicates |
| One row per occurrence | Expanding a recurrence rule is where calendar bugs live |
| Log hours **spent**, never deduct the estimate | The original estimate is the number worth keeping |
| OS file lock, not a PID file | A crashed agent must not leave a file that blocks every future start |
| No conversation history on disk | The model trusts its own earlier words over the system prompt |

## Out of scope for v2

Wanted, but not while the runtime is being rebuilt. Revisit once Phase 6 lands:

- Google Calendar sync, over OAuth
- Cross-session memory that survives a restart
- Deterministic answers without the LLM — clock, date, and schedule computed in
  Python. Measured 5.7x faster to first sound and 6/6 correct on the clock,
  against 4/6–6/6 wrong through the model
- Wake word, barge-in, proactive notifications
- Any language other than English. `LANGUAGE` stays, and stays `en`

Also unmeasured and deliberately left so: the easy and medium task slices end
to end, and whether Qwen3-4B answers as well as qwen2.5:7b **once it has the
tool result**. Routing is not the whole job, and the system prompt was tuned
against qwen2.5. If Phase 2 clears its bar but the answers get worse, that is
the finding, and `LLM_BACKEND=ollama` is why the fallback stays.

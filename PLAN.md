# PLAN — spec and milestones

## 1. Decisions locked in

| # | Decision | Rationale |
|---|---|---|
| D1 | Probe is a **binary gate**; tool name and arguments come from a second constrained pass over the same KV cache | Adding a tool needs no probe retraining |
| D2 | Model served via **HF transformers**, bf16, single load | Only route that exposes per-layer hidden states |
| D3 | Reminders are **on-demand only** in v1 | No daemon, no audio-ownership problem, no quiet hours |
| D4 | **English** speech in and out | Parakeet TDT and Kokoro both cover it natively |
| D5 | Storage is **SQLite** with soft deletes and an audit log | Needs range queries and atomic writes; JSON export via script |
| D6 | **All date math in Python** (`src/timeparse.py`) | LLM date arithmetic is unreliable and untestable |
| D7 | Latency budget: end of speech → first audio **≤ 1500 ms** | Forces streaming TTS and a single prefill |

Reversible if wrong: D5 (swap the store layer), D3 (add a daemon).
Not reversible without redesign: D1, D2.

## 2. Data model

All timestamps stored as UTC ISO-8601. Display timezone from `config.tz`
(IANA name). Every table carries `created_at`, `updated_at`, `deleted_at`.

### `course`
`id`, `code` (e.g. COMP4020), `title`, `instructor`, `location`,
`weekday` 0–6, `start_time`, `end_time`, `term_start`, `term_end`, `notes`

### `course_exception`
`id`, `course_id`, `date`, `kind` ∈ {cancelled, moved, room_change},
`new_start`, `new_end`, `new_location`, `note`

### `assignment`
`id`, `course_id` (nullable), `title`, `due_at`, `est_hours` REAL,
`progress_pct` INTEGER 0–100, `status` ∈ {todo, in_progress, done, dropped},
`notes`

Derived, never stored: `hours_left = est_hours × (1 − progress_pct/100)`.
Setting `progress_pct` to 100 sets `status = done`.

### `reminder`
`id`, `title`, `remind_at`, `related_type` ∈ {course, assignment, null},
`related_id`, `notes`

### `turn_log`
`id`, `session_id`, `ts`, `transcript`, `asr_conf`, `probe_score`,
`probe_label`, `tool_name`, `tool_args_json`, `tool_result_json`,
`reply_text`, `ms_asr`, `ms_prefill`, `ms_gen`, `ms_tts`, `hidden_state_path`

This table is the probe's training set and the only debugging surface for a
system where the input is sound. Every turn writes a row, including failures.

### `audit_log`
`id`, `ts`, `table_name`, `row_id`, `op`, `before_json`, `after_json`, `turn_id`

## 3. Tool contracts

Tools take **natural-language time expressions**, not ISO strings. The resolver
converts them against the injected `now`. If an expression is ambiguous the tool
returns `{"needs": "clarification", "question": …}` rather than guessing.

| Tool | Args | Writes |
|---|---|---|
| `get_now` | — | no |
| `list_schedule` | `when` ("today", "next week") | no |
| `list_assignments` | `status?`, `due_before?`, `course?` | no |
| `add_class` | `code`, `title?`, `weekday`, `start`, `end`, `location?`, `term?` | yes |
| `update_class` | `course`, `fields` | yes |
| `cancel_class` | `course`, `date`, `kind`, `details?` | yes |
| `delete_class` | `course` | yes, confirm |
| `add_assignment` | `title`, `course?`, `due`, `est_hours?` | yes |
| `update_assignment` | `assignment`, `fields` | yes |
| `set_progress` | `assignment`, `percent` | yes |
| `delete_assignment` | `assignment` | yes, confirm |
| `add_reminder` | `title`, `when`, `related?` | yes |
| `undo_last_write` | — | yes |

`course` and `assignment` accept an id **or** a spoken name, fuzzy-matched
against existing rows. Ambiguous match (≥ 2 candidates above threshold) returns
a clarification instead of picking one — ASR mangles course codes, and a
confident wrong match silently corrupts data.

Stretch tools, not in v1: `find_free_time`, `plan_work_sessions`
(schedule `hours_left` into gaps before `due_at`), `sync_google_calendar`.

## 4. The gate

Prefill the chat-formatted prompt once with `output_hidden_states=True` and
`use_cache=True`. Take `hidden_states[L][0, -1]` — the last prompt token at
layer *L*. Qwen3-4B is 36 layers, hidden size 2560 (confirm against
`config.json` at load; assert the shape rather than trusting this doc).

Probe: standardise, then logistic regression. Linear first — reach for an MLP
only if the layer sweep shows the linear probe is the bottleneck.

Three-way outcome from one score:

| Score | Branch |
|---|---|
| `< τ_lo` | chat reply, no tool |
| `≥ τ_hi` | second pass emits `{"tool", "args"}`, execute, verbalise result |
| between | ask one clarifying question, re-gate on the answer |

Thresholds are tuned asymmetrically: a false *call* writes wrong data, a false
*no-call* only produces an unhelpful reply. Target false-call rate < 2% on the
held-out set; let the uncertainty band absorb the rest.

The second pass reuses `past_key_values` from the prefill and appends the tool
schema suffix, so the prompt is encoded once per turn.

### Dataset

- ~600 seed utterances, balanced call / no-call, covering every tool plus
  chatter, greetings, follow-ups ("make it 60 instead"), and refusals.
- Follow-up handling depends on context: the gate sees the **last 3 turns**, not
  just the current utterance. Dataset must include multi-turn examples.
- 80/20 split, stratified. Real logged turns from `turn_log` are folded in as
  they accumulate, and always land in train, never in the held-out set.

### Baseline

A prompted gate — ask the model to reply `TOOL` or `CHAT` and read the token.
Same prompts, same split. The probe has to beat it on accuracy *and* latency, or
D1 was the wrong call. Numbers go in `docs/eval.md`, produced by a script, never
typed by hand.

## 5. Latency budget

| Stage | Budget |
|---|---|
| VAD endpoint detection | 200 ms |
| Parakeet | 300 ms |
| Prefill + probe | 150 ms |
| Generate to end of first sentence | 500 ms |
| Kokoro first chunk | 250 ms |
| **Total to first audio** | **1400 ms** |

TTS streams per sentence; the speaker starts before generation finishes.

## 6. Milestones

Each milestone ends with something demonstrable. No milestone is "done" until
its exit criterion is met by a command someone else can run.

### M0 — Store and tools, no model
SQLite schema, all tools as plain Python functions, `timeparse.py` with a real
test suite, text REPL that dispatches typed tool calls.
**Exit:** full CRUD by typing, `pytest` green, no GPU involved.

### M1 — LLM in the loop, prompted gate
Load Qwen3-4B, system prompt with injected `now`, prompted TOOL/CHAT gate, JSON
second pass, tool execution, spoken-style replies.
**Exit:** a typed conversation performs real CRUD end to end. This is also the
baseline the probe is measured against.

Verbalising a tool *result* is not the model's job: `src/format.py` does it
from templates, deterministically, with the response-style rules covered by
tests. The model decides, extracts arguments, and chats — the three things a
template cannot do.

### M2 — Hidden states and dataset
Capture `h_L` per turn into `turn_log` + `.npz`, write the seed dataset, run the
layer sweep across all 36 layers.
**Exit:** `docs/eval.md` has a layer-vs-accuracy table and a chosen *L*.

**The first seed set was too easy.** The linear probe hit 100% at fifteen
different layers, so the sweep could not separate them and *L* fell out of a
tie-break rather than evidence. Adding 75 hard cases fixed it — chatter that is
full of schedule vocabulary but wants no tool ("the comp4020 assignment was due
last friday and I got it in"), and requests with almost no vocabulary at all
("what else", "clear that one"). The sweep now has a real curve, a single best
layer, and a wider margin over the bag-of-words control.

Two controls make the result worth believing, and both belong in any future
sweep:

- **Layer 0 scores at chance.** The last token's embedding, before any
  transformer block, carries no intent. So the probe is reading something the
  model computes, not token identity.
- **A TF-IDF bag of words is the floor.** If it matches the probe, the dataset
  is separable by vocabulary and the sweep says nothing about hidden states.
  `scripts/sweep_layers.py` prints a warning when the margin closes.

### M3 — Probe
Train the probe, tune τ_lo/τ_hi, wire the uncertainty band to the clarification
branch, put it behind `config.gate = probe | prompted`.
**Exit:** probe beats the prompted baseline on the held-out set, with the
comparison in `docs/eval.md`.

**The uncertainty band came out empty.** Tuned on a validation slice, τ_lo and
τ_hi land on the same value: the two classes separate cleanly enough that there
is no range between them to be unsure in. The three-way branch is wired and
tested, but only two arms fire today. This is a property of hand-written data —
nobody writes the half-audible sentence a probe *should* hesitate over — and is
expected to change once `turn_log` carries real ASR output at M4. The band is
not removed: invariant #6 exists for exactly the case that is missing here.

**Model selection must not touch the held-out set.** The first sweep picked `C`
per layer by held-out accuracy, which inflated every layer and changed which
layer won (15 instead of 18). Both `sweep_layers.py` and `train_probe.py` now
select on a validation slice carved out of train. If a future script reports a
number, check what it was tuned against first.

**Editing the system prompt invalidates the probe.** The probe reads a hidden
state produced by a prefix that begins with `prompts.SYSTEM`. Change one line of
it and every activation downstream moves — but a stale probe does not fail, it
keeps returning confident scores for a prompt it never saw. Three things guard
this now, and none of them should be removed:

- the probe artefact records `prompt_fingerprint`, and `ProbeGate` refuses to
  load against a prompt that does not match — including an artefact that has no
  fingerprint at all, because unverifiable is the dangerous case;
- `data/hidden_cache.npz` is keyed on the prompt as well as the dataset, so the
  sweep cannot silently reuse activations from a prompt that no longer exists;
- changing the prompt therefore costs a re-sweep and a retrain. That is the
  real price of the architecture, and it is worth knowing before touching a
  wording.

### M4 — Voice in
Silero VAD, Parakeet, hotkey session loop, fuzzy entity matching against course
names for ASR errors.
**Exit:** speaking a command produces the correct database write.

**Endpointing costs more than the budget allows.** PLAN's 200 ms for VAD
endpoint detection is not reachable on real sentences: at 350 ms the segmenter
cut "add the data structures assignment due next friday, about six hours of
work" in half at the comma, and answered the orphaned fragment as a new
command. `MIN_SILENCE` is 0.6 s, which spends 400 ms of the budget to stop
cutting people off. Section 5 should be read as aspirational until measured.

**The gate is no longer the weak link; tool selection is.** On mangled ASR
("the data structures 1 as 60% done", from "mark the data structures one as
sixty percent done") the probe correctly gated `tool` at 0.95 — and the second
pass then chose `list_assignments` and invented a `due_before` argument. The
write silently did not happen. Improving that is a second-pass problem, not a
gate problem, and it is the next thing worth working on.

### M5 — Voice out
Kokoro, sentence streaming, speech-shaped response rules, barge-in (speaking
interrupts playback).
**Exit:** full loop runs inside the latency budget, measured from `turn_log`.

**Built, and the exit criterion is not met.** The loop runs end to end — speech
in, the right write, speech out — at about 7 s a turn against a 1400 ms budget.
The numbers are in `docs/eval.md`, from `scripts/bench_loop.py`. ASR and the
gate are at or near budget; generation is 7.7× over and Kokoro's first chunk
was 7.7× over.

The TTS half of that is now fixed — see below — which leaves **generation** as
the one stage standing between the loop and section 5. It is not fixable by
tuning: decode is ~105 ms/token on 4-bit bitsandbytes weights and a tool call
is 40–70 tokens.

That is also not a hardware verdict, which is what this milestone first
concluded. Measured on the card: 240 GB/s of memory bandwidth and 24 TFLOP/s
bf16, with native `sm_120` kernels. At 2.2 GB of NF4 weights the roofline for
decode is about **9 ms/token**, so the measured 105 is roughly 11× off what
the hardware allows. The card is not the limit; the 4-bit kernel and the
per-token framework overhead are, and both are software.

**Sentence streaming cannot buy what section 5 assumed it would.** The design
has the speaker start on sentence one while sentence two is synthesised — but
the response rules cap replies at two sentences and most are one, so there is
usually no second sentence to overlap with. Streaming is still there and still
correct; it just cannot hide a slow synthesiser behind a one-sentence answer.
Thread count was measured across 1/4/8/16 and makes no difference.

**Fixed, and the diagnosis above was wrong twice.** `scripts/bench_tts.py` now
measures Kokoro on its own, and the first chunk is **157 ms against a 250 ms
budget** — the first stage of the loop to land inside section 5. Two things
were wrong in what is written above:

- *"Kokoro takes about 0.5× real time on the CPU"* was never measured. It is
  3.3× real time on the CPU and 12× on CUDA. The synthesiser was not slow;
  `onnxruntime` was the CPU wheel, so it never touched the GPU at all.
- *"there is no second sentence to overlap with"* is true and beside the point.
  What matters is not how many sentences there are but how long the **first
  piece** is, since that is the entire wait. Breaking it at the first natural
  pause past 15 characters took the median from 726 ms to 504 ms on the CPU
  alone, before any of the GPU work.

Two traps found on the way, both worth keeping in mind elsewhere:
`onnxruntime` advertises a provider it cannot actually build and then falls
back to the CPU with a log line, so `src/tts/kokoro.py` reads the provider back
off the built session instead of trusting the one it asked for. And on Windows
`os.add_dll_directory` does not cover a native DLL's own dependencies — the
CUDA libraries have to go on `PATH` before the session is built.

**This does not mean the loop got 1.7 s faster.** These are warm, standalone
numbers with nothing else running; the 1929 ms came from real turns with the
model resident. Re-run `bench_loop.py` over spoken turns before believing the
end-to-end figure moved.

**Barge-in needs headphones.** The microphone stays open while the agent talks,
so with speakers it hears itself and interrupts itself. There is no acoustic
echo cancellation and adding one is out of scope; `--no-barge-in` turns it off.

Section 5 should be rewritten against measurements rather than intentions. The
honest version of that table is in `docs/eval.md`.

### M6 — Hardening
`undo_last_write`, class/assignment conflict detection, confirmation prompts on
destructive writes, DB export and backup script.
**Exit:** a misheard destructive command can be reversed by voice.

### Later
Proactive reminder daemon (D3 revisited), `find_free_time` and work-session
planning from `hours_left`, Google Calendar sync, non-English support.

## 7. Open questions

- Which term/semester dates apply, and does the agent need to understand
  "week 5" style academic references?
- ~~Should completed assignments be archived out of the default query window,
  or stay visible?~~ **Decided at M0:** `list_assignments` returns `todo` and
  `in_progress` only, unless `status` is passed explicitly (`done`, `dropped`,
  or `all`). Finished work is noise in a spoken answer.
- Hotkey choice, and whether the session should auto-close on silence.
- ~~How much conversation history persists across sessions — none, last
  session, or a rolling summary?~~ **Decided at M1:** none across sessions.
  Within a session the gate and the tool pass see the last three turns
  (`prompts.HISTORY_TURNS`), which is what follow-ups like "make it 60
  instead" need. `turn_log` keeps everything on disk regardless, so a rolling
  summary can be added later without changing the schema.

# personal-agent architecture

This document explains **how** the system works and **why** it was built this
way. For how to use and configure it, see [README.md](README.md).

Most decisions here came from measurement rather than preference. The numbers
are included so they can be revisited when circumstances change.

---

## 1. The shape of it

A voice assistant that sits quietly in the Windows background. Press a hotkey,
speak, press again — it answers through the speakers. English, fully offline.

```
                    ┌──────────────────────────────────────┐
   press hotkey ───▶│  main.py — listener & orchestrator   │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   ┌─────────┐   ┌─────┐      ┌──────────┐                ┌──────────┐
   │ audio   │──▶│ vad │─────▶│   stt    │ speech→text ──▶│   llm    │
   │ mic     │   │edges│      │ Parakeet │                │  qwen2.5 │
   └─────────┘   └─────┘      └──────────┘                └────┬─────┘
        ▲                                                      │
        │                     ┌──────────┐                     │
        └──── speaker ────────│   tts    │◀── text→speech ─────┘
                              │  Kokoro  │
                              └──────────┘
```

Four models for four jobs: **Silero VAD knows when you have finished speaking**
(CPU, 0 VRAM), **Parakeet hears** (CPU, 0 VRAM), **qwen2.5:7b thinks** (GPU,
~5 GB), **Kokoro speaks** (CPU, 0 VRAM). Only one occupies the GPU — see §4.7.

Alongside them sits an **event store** (§4.12) — classes, tasks, and reminders
in one JSON file — which the model reaches through **tools** (§4.15), so
answers about your schedule come from your data rather than from its
imagination.

Nothing else. A calendar integration, a separate task list, cross-session
memory, and a deterministic answerer all existed once and were **deliberately
removed** during a restart; the git history is intact if any of it is wanted
back.

### Two layers, not one

The only thing running continuously is the **hotkey listener** — 28 MB of RAM,
0.02% CPU, zero VRAM. Everything heavy (STT, LLM) loads on use and is released
when idle. That is deliberate: the agent is alive 24 hours a day but used for a
few minutes, so holding models in memory all day is pure waste.

---

## 2. Module map

| Module | Lines | Responsibility |
|---|---:|---|
| `main.py` | 672 | Hotkey listener, session mode, pipeline orchestration, logging |
| `llm.py` | 288 | Two brain backends (Ollama/Claude), sentence streaming |
| `stt.py` | 275 | Parakeet / Whisper: load, transcribe, release when idle |
| `audio.py` | 271 | Mic capture, gapless playback, beeps |
| `config.py` | 257 | Every constant from `.env` + offline-mode enforcement |
| `vad.py` | 226 | Per-frame speech detection (Silero) |
| `tts.py` | 142 | Kokoro / Piper: text -> WAV |
| `offline_check.py` | 128 | Verifies readiness to run with no network |
| `events.py` | 425 | Class / task / reminder store, and its CLI |
| `tools.py` | 293 | What the model may call, and the code that runs it |
| `text.py` | 55 | Text normaliser |

About 3,281 lines in total.

### Dependency direction

```
main  ──▶ audio, vad, stt, llm, tts, text
llm   ──▶ tools
tools ──▶ events
everything ──▶ config
```

One direction, no cycles. `config` depends on nothing; nothing imports `main`.
Each model module (`vad`, `stt`, `llm`, `tts`) stands alone and can be run on
its own with `python -m agent.<module>`.

---

## 3. One conversation, step by step

```
1. Hotkey pressed
   └─ if the STT model was released, start loading it NOW (in parallel)
   └─ open the mic stream, then play the beep (non-blocking)

2. Hotkey pressed again
   └─ recording stops

3. Transcribe (Parakeet)
   └─ if the model is not ready yet: say "One moment, just getting ready."

4. Send to the LLM
5. Answer → Kokoro → speaker, SENTENCE BY SENTENCE (§4.11)
```

In **session mode** (`SESSION_MODE=true`), steps 1–2 are replaced: the hotkey
opens a session once, and each utterance boundary comes from the VAD. Steps 3–5
are identical, shared through `_route_and_reply()`.

```
1. Hotkey pressed → session opens
2. Repeat until it closes:
   └─ wait for speech, record until 800 ms of silence
   └─ steps 3-5 above
   └─ the mic is CLOSED while the agent speaks (§4.9)
3. Closes on: hotkey pressed again, 30 s of silence, or "goodbye"
```

`_route_and_reply()` now only forwards to the LLM. It stays a separate function
because **that is where intent routing will attach** when features return — and
the order of those checks has already proved easy to get wrong: *"add a task to
finish the assignment"* also matches the *"add ..."* pattern for creating an
event, so tasks had to be checked before events.

---

## 4. Design decisions, with the reasoning

Every decision below was either measured or came out of a real failure.

### 4.1 The hotkey must be a single key

A key held **together with a modifier** (`Ctrl+Space`) makes Windows send
spurious UP/DOWN pairs continuously — measured at 18 in 3 seconds, some only
0.01 s apart. The recording shatters; what survives is sentence fragments.

A key held **on its own** only repeats DOWN, never UP. Hence the default of
`right ctrl`.

`RELEASE_GRACE_SECONDS` (0.6 s) remains as a safety net: recording only stops
once the key has read as released for that long.

### 4.2 The `pynput` backend, not `keyboard`

The `keyboard` library received no events at all on the test machine — its
low-level hook worked (proven by calling `_winkeyboard.listen` directly), but
`add_hotkey`/`hook` dispatch stayed completely silent. `pynput` works without
admin.

### 4.3 The pipeline runs on its own thread

The listener callback runs on the same thread that processes keyboard events.
Block it and the release event queues up behind it, leaving `is_held()` stuck at
`True` forever.

### 4.4 The start beep is non-blocking

Opening the mic stream takes ~0.5 s, so the beep plays **after** the stream is
ready. But the beep itself must not block: the mic buffer is flushed once
`on_ready` returns, and the user starts speaking the instant they hear the tone
— a blocking beep would take their first word with it.

### 4.5 Playback is padded with silence

Measured: **119 ms lost** from a 2-second tone without padding. The output
device needs time to open (the start is cut) and its internal buffer is still
playing when `sd.wait()` considers it finished (the end is cut).
`PLAYBACK_PAD_SECONDS=0.2` covers both.

### 4.6 Models are released when idle

Model weights are **idle data**, not a running process: Whisper held ~1.9 GB of
VRAM for the whole life of the agent while GPU utilisation sat at 1%. On a
personal GPU used for other things, that is a waste.

The cost is real and is not an average: a reload takes **2–7 seconds** from a
warm disk cache but **13–35 seconds** after hours of idling — and a long idle is
exactly what triggers the release. Three things soften it:

1. Loading starts the moment the hotkey is pressed (in parallel with speaking)
2. The agent says *"One moment"* so the silence does not read as death
3. A second press is answered with two short taps

### 4.7 Only one model occupies the GPU

The GPU here is 8 GB. qwen2.5:7b alone is ~5 GB. Put the STT model on the GPU as
well and only ~1 GB is left, at which point qwen starts spilling into RAM — far
more expensive than the time saved.

| Component | Where | VRAM | RAM | Speed |
|---|---|---:|---:|---|
| Silero VAD | CPU | 0 | ~50 MB | 190x realtime |
| Parakeet TDT 0.6B | CPU | 0 | ~2.2 GB | ~0.40 s/sentence |
| qwen2.5:7b | GPU | ~5 GB | — | ~2–4 s |
| Kokoro-82M | CPU | 0 | ~0.4 GB | ~4x realtime |

Parakeet on GPU is only ~0.2 s faster. Trading 0.2 s for VRAM pressure on qwen
is a losing deal, so Parakeet is deliberately pinned to the CPU.

**A warm Ollama is faster than Claude.** The first measurement concluded Ollama
was "2x slower"; what was actually being measured was **model loading**, not
inference. Ollama uses a 5-minute keep-alive. Warm, it runs 1.8–2.2 s against
~2.8 s for Claude, which has to cross the network.

### 4.8 Offline mode is enforced at startup, in `config`

`OFFLINE_MODE=true` rejects backends that need the network and the agent exits
with code 2.

Two small decisions here matter:

**Enforced at startup, not on use.** The agent runs with no window. A network
failure mid-conversation just sounds like the agent went mute; a failure at
startup is written plainly to the log.

**Placed in `config.py`, not `main.py`.** A check that exists at a single entry
point leaks through scripts and tests. `config.require_offline()` can be called
by anyone.

The proof is runnable: `python -m agent.offline_check` verifies the settings,
the weights on disk, that the models genuinely load, and that Ollama answers.
Tested with every non-localhost socket blocked, the full chain ran 7/7 steps
with **zero** outbound connection attempts.

### 4.9 Session mode: the mic closes while the agent speaks

In session mode, utterance boundaries come from the VAD (Silero, through
`onnx_asr` — the same package as Parakeet, so no new dependency). The
consequence: the mic is live continuously, and a live mic while the speaker is
playing will **hear the agent itself** and transcribe that as user speech. The
agent then talks to itself until the session is forced shut.

Three ways out; the simplest was chosen:

| | Can interrupt? | Headphones required? | Risk |
|---|---|---|---|
| **Mic closed while speaking** | no | no | near zero |
| Mic always live | yes | **yes** | speakers = chaos |
| Echo cancellation | yes | no | most complex, most fragile |

The first row won. The price is real: a long reply cannot be interrupted, and
you are stuck listening to the end. That is exactly why `REPLY_MAX_WORDS` is
mandatory rather than decorative — see §4.10.

The VAD also distinguishes **three** reasons for stopping, not one. "The user
closed the session", "the session timed out in silence", and "that sound was too
short" must stay separate: collapse them and a single cough closes the session.
Sounds that are too short are swallowed inside the VAD and never reach the
caller.

The silence deadline is measured from entry and is **not** reset by short
sounds. Reset it, and a noisy room could hold a session open indefinitely
without you saying a word — with the models held in memory the whole time.

### 4.10 Reply length: a word limit, not a token limit

Measured on qwen2.5:7b:

| | Words | Audio |
|---|---:|---:|
| Unconstrained | 41.0 | 18.4 s |
| 25-word limit in the prompt | 17.8 | 9.1 s |
| + ask for short sentences | **13.0** | **7.1 s** |
| 25-word limit + 60-token cap | 17.0 | 9.5 s |

A token cap (`num_predict`) **adds nothing** on top of the word limit — 17.0
against 17.0 — and it cuts mid-word, which sounds like the agent being choked
off. So the word limit is what does the work; `num_predict` stays only as a
safety net for genuine rambling.

**Sentence** length is controlled separately from **reply** length, and that is
not duplication: the TTS cannot start until one sentence is complete, so a
single 36-word sentence delays the first sound just as much as not streaming at
all.

### 4.11 Streaming cuts the silence, not the total

The answer is split by sentence and synthesised immediately, while the model is
still writing the next one.

| Question | First sound, wait for all | First sound, streaming |
|---|---:|---:|
| "What classes do I have today?" | 17.4 s | 8.2 s |
| "What should I work on today?" | 11.8 s | 3.4 s |

Down 53–71%. The total barely changes — what disappears is the silence, and
that is the difference between a conversation feeling alive and feeling laggy.

The splitter has to know that `3 p.m.` is not a sentence end. Without it the TTS
says "three pee." and pauses before "em." — and `3 p.m.` is precisely the form
Parakeet produces.

Playback goes through `audio.Speaker` rather than repeated `play_wav()` calls.
`play_wav()` pads silence onto **both** ends of every chunk, so each sentence
join costs 0.4 s of silence — measured at 0.83 s of excess silence on a
three-sentence reply. `Speaker` pads once at the start and once at the end of
the whole utterance.

---

## 5. Storage

Almost none, deliberately.

| File | Contents | Lifetime |
|---|---|---|
| `memory/events.json` | Classes, tasks, reminders | Until you delete them |
| `memory/agent.lock` | Single-instance lock | Released by the OS on exit |
| `logs/agent.log` | Log | Rotating, 1 MB x 4 |

**Conversation history is not written to disk.** It used to be, and repeatedly
led the model to invent things about data that had since changed — up to
claiming it had moved a deadline whose event had already been deleted. History
is useful for continuing a conversation, but it is not a source of fact. A
restart starts clean.

Large model weights live in the HuggingFace cache (`~/.cache/huggingface`),
not the repo: Parakeet and Silero VAD are fetched by `onnx-asr` by model name
rather than by path.

### 4.12 One event store, not three

Classes, tasks, and reminders share one file and one schema rather than living
in a calendar file, a task file, and a reminder file.

The previous design had exactly that split — an `.ics` for the calendar and a
`tasks.json` for the task list — and the cost showed up in the question asked
most often: *"what's on today?"* That needed two lookups, two formats, and two
sets of date handling, and the two disagreed about what a date even was (ICS
`DTSTART` versus an ISO string).

What makes one schema honest is `start`. Everything on a timeline has a moment
it belongs to; the kinds differ only in precision and in whether they occupy
time:

| Kind | Fields | Meaning |
|---|---|---|
| `class` | `start` + `end` | a slot |
| `task` | `start` | a deadline, no slot |
| `reminder` | `start` | a point in time, no slot |

`start` is a date (`2026-08-14`) or a date and time (`2026-08-14T17:00`). That
one rule replaces what would otherwise be an `all_day` flag — and a flag that
can disagree with the value beside it is a bug waiting to happen.

**One row per occurrence, not a recurrence rule.** A weekly class is 12 rows.
The 79 imported rows collapse to 8 weekly patterns, so a rule-based store would
be about a tenth the size — but then nothing you read in the file is what the
code actually sees, because a pattern has to be expanded into dates first. The
expansion is where recurring-calendar bugs live. The cost accepted instead:
moving a weekly class means editing every row.

**Ids are derived, not random**: `kind-title-start`. Importing the same source
twice replaces rather than duplicates, which is what makes the ICS import safe
to re-run.

### 4.13 Effort is logged as spent, not deducted from the estimate

A task carries two numbers: `estimate_hours` (what you thought) and
`spent_hours` (what you have actually put in). Logging work adds to the second
rather than subtracting from the first.

The tempting shortcut — decrement the estimate from 8 to 5 — destroys the
original figure. That figure is the more valuable of the two: comparing what a
task was estimated at against what it really took is the only way to find out
how wrong your estimates run, and that is knowledge no other field can recover
once it is overwritten.

Both numbers are shown together (*5 of 8, 3 left*) rather than just the
remainder. *3h left* says nothing about whether you are nearly finished or have
barely begun.

Two edges are handled deliberately rather than clamped away:

- **Overrun is reported, not hidden.** 11 hours against an 8-hour estimate reads
  *11h of 8h, 3h over*. Silently capping it at 8 would erase exactly the signal
  worth having.
- **Overrun contributes 0 to the outstanding total.** That total answers "how
  much work is left", and a task past its estimate has no meaningful remainder
  to add. Letting it go negative would quietly cancel out other tasks' hours.

A negative log corrects a mis-entry, but the total is floored at zero: below
that reads as a bug in the log rather than as work undone.

### 4.14 JSON, not a database

A few hundred entries do not justify SQLite. What plain JSON buys:

- You can open it in Notepad and fix a typo
- A git diff shows what changed in words, not as a binary blob
- No schema migration when a field is added

Written through a temp file then renamed, so a process that dies mid-write
leaves the old file intact rather than half of a new one.

This stops being the right call somewhere around tens of thousands of entries,
or when queries need more than a linear scan. Neither is close.

### 4.15 Tools: the model asks, the code decides

The model never opens a file. It returns a request — a name and arguments — and
`agent/tools.py` runs it. The model decides *what it needs*; the code decides
*what it is allowed to do*.

Its only job in the first round is **translation**. For *"how many assignments
in this two weeks"* it emits `get_schedule({"kind": "task", "days": 14})`: the
user never said 14. Then the result is appended to the conversation and the
model writes the sentence.

**The description is the prompt.** A tool is chosen by matching the question
against its `description`, so the description has to name the triggers rather
than the function. Measured on qwen2.5:7b, 5 repetitions each: 30/30 questions
that should call a tool did, 0/10 that should not did, and the right tool was
picked every time.

Two jobs had to be taken away from the model, both for the same reason.

**Working out what has already happened.** Asked for the next class at 19:25,
with today's 15:30–17:00 tutorial in the list, it answered *"at 3:30 PM"* — a
class finished two and a half hours earlier. Marking rows `already_finished`
was not enough: it read the marker for "what's on today" and ignored it for
"what's next". So `get_next` is a separate tool that filters in Python.
Choosing between two tools is a far easier decision than filtering a list, and
unlike the filtering, the choice is measurable.

**Recovering from its own malformed tool calls.** About 20% of the time the
call arrived as text instead of structure:

    CallCheck for your next class time and location.
    .GetOrdinal("nextlecture")
    Let me check your schedule.

Spoken aloud that is worse than a wrong answer: it sounds like work is
happening when nothing is. The first sentence of a reply is now held back until
we know whether a tool call came with it; if it reads as a stall or as code and
no tool was called, it is dropped unsaid and the request retried. 80% to 100%,
at no latency cost — a sentence is only spoken once complete anyway.

The pattern underneath all three: **give the model a decision, never a
calculation.** Choosing a tool, translating a phrase into an argument — those
it does well. Comparing timestamps, filtering a list, keeping a count — those
it does badly, and every one of them moved into Python.

---

## 6. Swappable parts

Four points are designed to be changed through `.env` without touching code:

**Brain** — `LLM_BACKEND=ollama|claude`. Both sit behind the same `chat()` and
`chat_stream()`. A backend without streaming support still works: the default
`chat_stream()` yields one whole chunk, so the caller never has to know.

**Ears** — `STT_BACKEND=parakeet|whisper`. Both sit behind the same
`get_model()` / `transcribe()` / `unload_model()`. Parakeet is English-only and
accepts no vocabulary prompt; Whisper is multilingual but wants VRAM on GPU.

**Voice** — `TTS_BACKEND=kokoro|piper`, both behind `speak(text) -> WAV`. The
sample rates differ (24 kHz vs 22.05 kHz) but that causes no trouble: `_to_wav()`
writes a WAV header and the player reads the rate from the header, not from a
constant.

**Hotkey backend** — `pynput` (default) or `keyboard`, in `toggle` or `hold`
mode.

**Interaction style** — `SESSION_MODE=false` (press every turn) or `true` (press
once, keep talking). Only the handler changes; transcript through answer is
shared via `_route_and_reply()`.

---

## 7. Not here yet

Everything below **existed** and was removed during the restart. The git history
is complete — this is not a wishlist, it is a list of things waiting to be
picked back up:

- **Editing and deleting by voice** — the model can read the store and add
  tasks, but has no tool to change or remove an entry
- **Google Calendar sync** — two-way, over OAuth
- **Cross-session memory** — history and facts surviving a restart
- **Deterministic answers without the LLM** — clock, date, and schedule computed
  in Python. Measured 5.7x faster to first sound, and 6/6 correct on the clock
  against 4/6–6/6 wrong through the model
- **Deterministic time parsing** — `time_en` (33/33) and an Indonesian parser

Never built:

- **Wake word** — entering a session without touching a key
- **Interrupting the agent** — needs echo cancellation, see §4.9
- **Proactive notifications** — the agent speaking first

### Known limits of qwen2.5:7b

Two measured facts, not guesses.

**A wrong answer poisons the following turns.** Asked where a class was with an
empty history: 0 wrong out of 8. But once a single incorrect answer entered the
history, the next turn copied it — the model trusts its own earlier words over
the data in the system prompt. This is why history is no longer written to disk
(§5), and why `OLLAMA_NUM_CTX` was raised to 8192.

Its most expensive form: the model **claims to have done something it did not
do**. Asked *"can you delay that?"*, it answered *"the deadline is now set for
September 1st"* — while no code path existed that could modify an event at all.
The system prompt now forbids this explicitly: the agent has no tools, and must
say so. A wrong answer still gets caught when the user checks; a false claim of
having acted makes them stop checking.

**Replies run longer than asked.** See §4.10.

---

## 8. How to re-measure

The numbers here go stale when the hardware or the models change. What you need
to know to measure them yourself:

- **STT accuracy** — turn on `SAVE_RECORDINGS=true`, use it for a few days, then
  compare the transcripts against what you actually said
- **VAD** — `python -m agent.vad`. Say a sentence with a pause in the middle: it
  must come out as ONE line, not two. A cough must not appear at all
- **Perceived latency** — what matters is time to **first sound**, not total
  time. Streaming barely changes the total but cuts the silence by 53–71%
- **VRAM** — `nvidia-smi`, or `scripts/status.ps1` for the summary
- **Offline** — `python -m agent.offline_check`. For harder proof, block every
  non-localhost socket and run the full chain; what counts is not that it "runs"
  but that there are **zero outbound connection attempts**

One lesson that keeps recurring: **a test that is too lenient hides real
failures.** Twice in this project a test passed while the answer was wrong,
because the criterion was only "there is an answer". Measure the value, not its
existence.

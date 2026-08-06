# personal-agent

A local voice assistant for Windows. Runs quietly in the background with no
window. Press a hotkey, speak, and it answers through the speakers.
**English, fully offline** — not a byte leaves this machine.

```
hotkey → mic → Silero VAD → Parakeet TDT 0.6B → qwen2.5:7b → Kokoro-82M → speaker
```

Four models, four jobs: **the VAD knows when you have finished speaking**,
**Parakeet hears**, **qwen thinks**, **Kokoro answers**.

It also keeps an **event store** — your class schedule, tasks, and reminders in
one JSON file — and the model **reads it through tools**, so answers about your
schedule come from your data rather than from its imagination. See
[The event store](#the-event-store) and [How tools work](#how-tools-work).

Conversation history lives only as long as the process.

> **Status: the runtime is being rewritten.** What is described below is v1 and
> it works. v2 rebuilds it on the all-HF architecture — one model in VRAM
> answering *and* probing — and adds rescheduling by voice. The build order and
> the acceptance criteria are in **[PLAN.md](PLAN.md)**; if you are working in
> this repo, start at **[CLAUDE.md](CLAUDE.md)**.

## Requirements

- Windows 10/11
- [Ollama](https://ollama.com/download) installed and running
- Python 3.11 (via [mise](https://mise.jdx.dev/); a fallback is below)
- A microphone and speakers

## Install

```powershell
git clone <repo> personal-agent
cd personal-agent

# 1. Python 3.11 + venv
mise trust
mise install
mise where python@3.11              # note the path
& "<path-from-above>\python.exe" -m venv .venv-agent

# 2. Dependencies
.\.venv-agent\Scripts\python.exe -m pip install -e .

# 3. Every model weight (~5.7 GB)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 4. Configuration
copy .env.example .env

# 5. Confirm it really is offline-ready
.\.venv-agent\Scripts\python.exe -m src.offline_check
```

<details>
<summary>Without mise</summary>

```powershell
py -3.11 -m venv .venv-agent
.\.venv-agent\Scripts\python.exe -m pip install -e .
```

Python 3.13+ is not used yet because `ctranslate2` (the engine behind
faster-whisper) has no stable wheel there. faster-whisper is only used when
`STT_BACKEND=whisper`, but its dependencies install regardless.
</details>

`scripts\setup.ps1` fetches every weight up front. Skip it and each model
downloads itself on first use — which means the first use needs the network, and
that is exactly what makes offline mode look broken.

## Running it

```powershell
.\.venv-agent\Scripts\python.exe -m src.agent
```

Press the hotkey **once** to start, speak, press **again** to stop.
Don't want to press every turn? See [Session mode](#session-mode).

Since there is no window, the feedback is audible:

| Sound | Meaning |
|---|---|
| high beep (880 Hz) | recording started |
| mid beep (560 Hz) | recording stopped, thinking |
| two short taps (420 Hz) | heard you, but busy |
| long low beep (240 Hz) | something failed — check `logs/agent.log` |

### The tray icon

A dot appears in the notification area, because started-from-a-scheduled-task
under `pythonw.exe` otherwise leaves no evidence the agent exists at all.

| Colour | |
|---|---|
| grey | starting — loading models, not ready for the hotkey yet |
| green | idle, listening for the hotkey |
| red | recording |
| amber | transcribing or thinking |
| blue | speaking |

Grey answers the question the beeps cannot: on a cold disk cache warmup takes
up to 35 s, and until now "still loading" and "dead" looked identical.

Right-click for **Open log** and **Quit**. Quit uses the same shutdown path as
Ctrl+C, with a 3-second watchdog behind it for the case where the hotkey
listener is parked in native code and cannot be interrupted.

`TRAY_ENABLED=false` turns it off. The icon is also strictly optional at
runtime: [src/tray.py](src/tray.py) no-ops if `pystray` is missing or the
notification area refuses it, so a cosmetic feature can never take down a
working voice assistant.

```powershell
.\.venv-agent\Scripts\python.exe -m src.tray    # cycle every state, 2 s each
```

## The probe

The agent decides whether a question needs a tool by reading the model's own
hidden state before it speaks, not by hoping the system prompt lands.

Measured on the 132 hardest questions in the task set, through the real stack:

| | tool calls | routing correct | prose bugs |
|---|---:|---:|---:|
| prompt only | 15/132 | **69%** | 34 |
| probe + prefill | 56/132 | **100%** | 0 |

The failure it removes is specific and was easy to reproduce. Asked *"is it a
good time to take a break?"* the model would say *"Let me check what's coming
up next."* — and stop. No tool call, no answer, and the system prompt forbids
that exact sentence in as many words. Telling a model what to do has a limit;
this works by removing the option. When the probe is confident, the assistant
turn is opened for it:

```
<|im_start|>assistant
<tool_call>
{"name":
```

and there is no longer a way to begin with a sentence. Note the opening tag:
the paper prefills `{"name":`, which on Qwen opens a JSON object inside a tag
the model writes first, and parses as nothing.

**Start the service before the agent** — it is not part of the agent process:

```powershell
.\.venv-probe\Scripts\python.exe -m src.probe_service
```

It holds Qwen3-4B in 4-bit (3.6 GB of VRAM, ~23 s to load) and answers in
about 90 ms. It lives in `.venv-probe` on purpose: the probe needs ~3 GB of
torch wheels that `.venv-agent` deliberately does not have, so that a research
dependency cannot destabilise a working voice assistant.

**If it is not running, nothing breaks.** The agent logs one line and routes
tools the way it always did. Same if it times out, or the reply does not
parse.

| setting | | |
|---|---|---|
| `PROBE_ENABLED` | `true` | off returns the agent to prompt-only routing |
| `PROBE_URL` | `http://127.0.0.1:11500` | `127.0.0.1`, never `localhost` |
| `PROBE_TAU` | `0.5` | p above this means "needs a tool" |
| `PROBE_TIMEOUT` | `5` | seconds before giving up and carrying on |

τ = 0.5 is not tuned. It is p = 0.5, the classifier's own boundary; a sweep
from 0.1 to 0.9 on the training split picked it out anyway. Lower it and the
agent reaches for tools more readily, raise it and it answers more from its
own head.

### One model instead of two

The arrangement above holds **two** models in VRAM — qwen2.5:7b answering,
Qwen3-4B probing — which is 8.6 GB on an 8 GB card, and the probe pays for a
forward pass the answer was going to make anyway.

`src/hf_service.py` collapses them: Qwen3-4B answers *and* probes, and the
probe reads the pass that already exists.

```powershell
.\.venv-probe\Scripts\python.exe -m src.hf_service   # then LLM_BACKEND=hf
```

| | routing | turn | VRAM |
|---|---:|---:|---:|
| Ollama, no probe | 69% | 1451 ms | ~5 GB |
| Ollama + probe service | 100% | 2892 ms | 8.6 GB |
| **All-HF, prefix cached** | **98%** | **1508 ms** | **3.6 GB** |

The prefix cache is what makes it competitive: 1668 of the 1683 prompt tokens
are the system prompt and the tool schema, identical every turn. Recomputing
them is 2086 ms, keeping them is 101 ms. Ollama was never doing less work — it
was doing the same work once.

It speaks Ollama's dialect on purpose, so `src/llm.py` reuses its streaming
loop, its tool loop and its stall guard unchanged. `HF_SERVICE_URL` and
`HF_MODEL` are read from [src/config.py](src/config.py); they are not yet in
`.env.example`.

**This is the architecture v2 is built on** — see [PLAN.md](PLAN.md).

Full method, measurements and failure modes: **[PROBE.md](PROBE.md)**.

## Session mode

Press the hotkey **once** to enter a session, then talk back and forth without
pressing again. Utterance boundaries come from your voice, not the key.

```
SESSION_MODE=true
```

A session closes when you press the hotkey again, go quiet for 30 seconds, or
say *"goodbye"* / *"that's all"* / *"stop listening"*.

The silence limit is **a safeguard, not a convenience**: without it, forgetting
to close means the models sit in memory all day.

**The mic is closed while the agent speaks.** So you cannot cut it off
mid-sentence — but it also can never hear itself, which means no echo
cancellation and ordinary speakers are fine (headphones are not required).

Detection is Silero VAD through `onnx_asr` — the same package already pulling
Parakeet, so it adds **zero new pip packages**. Measured:

| | Mean probability | Frames above threshold |
|---|---:|---:|
| Silence | 0.004 | 0/62 |
| Loud noise | 0.011 | 0/62 |
| Real speech | 0.580 | 95/166 |

0.17 ms per 32 ms frame — **190x realtime**, so the CPU cost is effectively nil.

Sentences getting cut in half? Raise `VAD_SILENCE_MS`. Agent answering the air
conditioning? Raise `VAD_THRESHOLD`.

## The event store

Everything that sits on a timeline lives in one file, `memory/events.json`:
classes, tasks, and reminders together.

```powershell
.\.venv-agent\Scripts\python.exe -m src.tools                  # every tool, then the agenda
.\.venv-agent\Scripts\python.exe -m src.tools list all 60      # next 60 days
.\.venv-agent\Scripts\python.exe -m src.tools add task "Assignment 1" 2026-08-14 --course ENGN4122 --estimate-hours 8
.\.venv-agent\Scripts\python.exe -m src.tools add reminder "Pay rego" 2026-08-19T09:00
.\.venv-agent\Scripts\python.exe -m src.tools log "Assignment 1" 3         # worked 3 hours
.\.venv-agent\Scripts\python.exe -m src.tools done "Assignment 1"
.\.venv-agent\Scripts\python.exe -m src.tools rm "Pay rego"
```

### Tracking effort

`log` records time worked, and the countdown follows:

```
$ python -m src.tools log "Assignment 1" 3
Assignment 1: 3h of 8h, 5h left

$ python -m src.tools log "Assignment 1" 2
Assignment 1: 5h of 8h, 3h left
```

```
=== OPEN TASKS (1) ===

   Assignment 1  [9d left]  —  5h of 8h, 3h left

   3 hours of work outstanding
```

It **adds to hours spent** rather than subtracting from the estimate. Subtracting
would leave 5 where 8 used to be, and that original 8 is the more useful number:
comparing it against what the task actually took is the only way to learn how
wrong your estimates run.

Both figures are shown, not just the remainder — *5 of 8* tells you how far in
you are, which a bare *3h left* does not.

Log a negative number to correct a mistake (`log "Assignment 1" -1`). It will not
go below zero. Overrunning the estimate is reported honestly rather than clamped:

```
Assignment 1: 11h of 8h, 3h over
```

Overrun tasks contribute 0 to "hours of work outstanding" — the total answers
"how much is left", and a task past its estimate has no meaningful remainder.

`done` and `rm` match on any distinctive fragment of the title, so you never
type a full id. If the fragment matches more than one entry it lists them and
does nothing — deleting the wrong entry is not worth a guess.

**What unifies the three kinds is `start`.** Everything has a moment it belongs
to; they differ in precision and in whether they occupy time:

| Kind | Fields | Meaning |
|---|---|---|
| `class` | `start` + `end` | a slot on the calendar |
| `task` | `start` | a deadline, no slot |
| `reminder` | `start` | a point in time, no slot |

`start` is either a date (`2026-08-14`, meaning sometime that day) or a date and
time (`2026-08-14T17:00`, meaning exactly then). That single rule removes any
need for an "all day" flag.

```json
{
  "id": "class-intelligent-autonomous-systems-2026-07-27T0900",
  "kind": "class",
  "title": "Intelligent Autonomous Systems",
  "start": "2026-07-27T09:00",
  "end": "2026-07-27T11:00",
  "course": "ENGN4122",
  "session": "lecture",
  "location": "Fulton Muir, Rm 2.04"
}
```

Plain JSON, so you can open it in Notepad and fix a typo. Every occurrence is
stored on its own — a weekly class is 12 rows, not one recurrence rule. Nothing
has to expand a pattern into dates, so what you read is exactly what the code
sees; the cost is that moving a weekly class means editing every row.

Ids are derived from kind + title + start, so importing the same source twice
replaces rather than duplicates.

**Importing an .ics** (one-off, then the .ics is no longer needed):

```powershell
.\.venv-agent\Scripts\python.exe scripts\import_ics.py --dry-run
.\.venv-agent\Scripts\python.exe scripts\import_ics.py
```

`memory/` is gitignored, so this file is **not** backed up by git. Copy it
somewhere if the contents matter.

## How tools work

The model does not read `events.json`. It returns a *request*; `src/tools.py`
runs it. That boundary is the point: the model decides **what it needs**, the
code decides **what it is allowed to do**.

For *"How many assignments do I have in this two weeks?"*:

```
1. we POST     question + tool descriptions
2. model says  tool_calls: get_schedule({"kind": "task", "days": 14})
               content: ""          <- it has not answered anything yet
3. we run      events.between(2026-08-05, 2026-08-19, kinds=("task",))
4. we POST     the same conversation + the result appended
5. model says  "You have 1 assignment due in the next two weeks."
```

Step 2 is the whole trick. The model's only job there is **translation**:
*"assignments"* becomes `kind: "task"`, *"this two weeks"* becomes `days: 14`.
You never said the number 14.

**How it knows to call at all**: it matches your question against the tool's
`description`. That text is a prompt, not documentation — a lazy `"gets
schedule"` misses far more often than one that names the triggers.

Measured on qwen2.5:7b, 5 repetitions of each question:

| | Rate |
|---|---|
| Called a tool when it should | **30/30** |
| Called a tool when it should not | **0/10** |
| Picked `get_next` for "what's next" questions | 10/10 |

### Two things had to be taken away from the model

**Deciding what has already happened.** Asked *"where is my next class?"* at
19:25, with today's 15:30–17:00 tutorial in the list, it answered *"at 3:30
PM"* — a class that had finished two and a half hours earlier. It had every
timestamp it needed. Marking rows `already_finished` was not enough either; it
read the marker for "what's on today" and ignored it for "what's next".

So `get_next` is a separate tool that filters in Python. Choosing between two
tools turns out to be a far easier decision than filtering a list.

**Recovering from its own malformed tool calls.** About 20% of the time the
call came out as text instead:

```
CallCheck for your next class time and location.
.GetOrdinal("nextlecture")
Let me check your schedule.
```

Spoken aloud, that is worse than a wrong answer — it sounds like the agent is
working while nothing happens. The first sentence of every reply is now held
back until we know whether a tool call came with it; if it looks like a stall or
like code and no tool was called, it is dropped unsaid and the request is
retried. That took 80% to 100%, and costs nothing: a sentence is only spoken
once complete anyway.

Turn it all off with `TOOLS_ENABLED=false`.

## Starting automatically at login

```powershell
# PowerShell as Administrator
powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
```

```powershell
powershell -File scripts\status.ps1                       # running or not?
Start-ScheduledTask -TaskName PersonalAgent               # start
Stop-ScheduledTask -TaskName PersonalAgent                # stop
Get-Content logs\agent.log -Tail 20 -Wait                 # watch live
powershell -File scripts\install-startup.ps1 -Uninstall   # remove the task
```

### Only one agent may run

The agent grabs a **global hotkey**. Two agents means every press is caught by
both and both fight over the mic — and the symptom misleads: what you see is not
"there are two agents" but "the feature is broken", because the older agent is
the one answering first.

A second agent now refuses to start on its own:

```
Another agent is already running (PID 41696). This one is stopping — two
agents would fight over the 'right ctrl' hotkey.
Check: powershell -File scripts\status.ps1
```

It uses an OS file lock rather than a PID file: the OS releases the lock when
the process dies, so a crashed agent does not leave behind a file that blocks
every future start (tested, including `kill -9`).

**Testing from a terminal while autostart is on?** Stop the task first:

```powershell
Stop-ScheduledTask -TaskName PersonalAgent
.\.venv-agent\Scripts\python.exe -m src.agent
```

### If something looks wrong, check the build first

The second line of the log names the running commit:

```
build: 9a00404 05/08 08:31 (newest file 05/08 08:31)
```

Python loads source when the process starts — **editing a file does not touch a
running process.** This is the single most common cause of "the fix didn't work".

## Configuration

Defaults live in [src/config.py](src/config.py) and can be overridden in
`.env` (see [.env.example](.env.example) — 57 keys, every one of them actually
read by the code).

| Variable | Default | Notes |
|---|---|---|
| `HOTKEY` | `ctrl+space` | **Use a single key** — see below |
| `SESSION_MODE` | `false` | `true` = continuous conversation |
| `OFFLINE_MODE` | `true` | Reject network backends at startup |
| `LLM_BACKEND` | `ollama` | `ollama`, `hf` (one model, see above), or `claude` (needs `OFFLINE_MODE=false`) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Any model you have `ollama pull`ed |
| `REPLY_MAX_WORDS` | `25` | Reply length cap — see below |
| `STT_BACKEND` | `parakeet` | `parakeet` (English, 0 VRAM) or `whisper` |
| `TTS_BACKEND` | `kokoro` | `kokoro` (24 kHz, 54 voices) or `piper` |
| `VAD_SILENCE_MS` | `800` | Silence this long ends your sentence |

### Use a single key

A key held **together with a modifier** (`Ctrl+Space`) makes Windows send
spurious UP/DOWN pairs — measured at 18 in 3 seconds. The recording shatters
into fragments.

A key held **on its own** only repeats DOWN, never UP. Hence the local default
of `right ctrl`. Other good choices: `f8`, `right shift`.

### Reply length

qwen does not obey "one or two sentences". What **does** work is an explicit
word limit, not a token limit:

| | Words | Audio |
|---|---:|---:|
| Unconstrained | 41.0 | 18.4 s |
| `REPLY_MAX_WORDS=25` | 17.8 | 9.1 s |
| + ask for short sentences | **13.0** | **7.1 s** |

A token cap adds nothing on top (17.0 vs 17.0) and cuts mid-word, so
`OLLAMA_NUM_PREDICT` is only a safety net for genuine rambling.

It matters most in session mode: without barge-in, a long reply cannot be cut
short.

## Resource notes (8 GB VRAM)

Only **one model occupies the GPU**:

| Component | Where | VRAM | RAM | Speed |
|---|---|---:|---:|---|
| Silero VAD | CPU | 0 | ~50 MB | 190x realtime |
| Parakeet TDT 0.6B | CPU | 0 | ~2.2 GB | ~0.4 s/sentence |
| qwen2.5:7b | GPU | ~5 GB | — | ~2–4 s |
| Kokoro-82M | CPU | 0 | ~0.4 GB | ~4x realtime |

Parakeet is on CPU **deliberately**: on GPU it is only ~0.2 s faster, but it
takes VRAM away from qwen — and pushing qwen into RAM costs far more than 0.2 s.

Ollama uses a 5-minute keep-alive, so the first question after a long idle
stretch pays a reload (~7 s). `OLLAMA_KEEP_ALIVE=-1` makes it instant, at the
cost of the model sitting in VRAM permanently.

## Layout

```
src/
  agent.py           hotkey, session mode, pipeline orchestration, logging
  config.py          every constant from .env + offline-mode enforcement
  audio.py           mic capture, gapless playback, beeps
  vad.py             per-frame speech detection (Silero)
  stt.py             Parakeet / Whisper — loads and releases automatically
  llm.py             Ollama / Claude, sentence-by-sentence streaming
  tts.py             Kokoro / Piper
  events.py          class / task / reminder store
  tools.py           EVERY capability, once — the model's schema and the CLI
  text.py            text normaliser
  tray.py            notification-area icon: alive, and what it is doing
  offline_check.py   verifies readiness to run with no network

  probe_service.py   the probe alone, over 127.0.0.1          :11500
  hf_service.py      Qwen3-4B answering AND probing, no Ollama :11501
  model.py           research model: loading, prefix cache, the hardware gate
  hidden_states.py   one forward pass per task -> data/features.npz
  dataset.py         the 143 prompts and the train/test split
  labeling.py        what a label means, and the checks that keep it honest
  probe.py           trains the linear probe, reports held-out AUROC
  prefill.py         the three architectures, measured end to end

data/                tasks, splits, features, fitted probe
models/              Kokoro & Piper weights (gitignored)
memory/              events.json, agent.lock (gitignored)
scripts/             setup, autostart, status
logs/                agent.log (rotating, 1 MB x 4)
```

**One file answers "what can this thing do?" — [src/tools.py](src/tools.py).**
Each tool is declared once and both doors are derived from it: the JSON schema
the model is given, and the argparse command a terminal gets. They cannot
drift apart. Adding a capability means adding one entry.

The lower half of `src/` is the PROBE&PREFILL replication
([PROBE.md](PROBE.md)). It runs in its own virtualenv, `.venv-probe`, and the
agent never imports it — but it does import `config` and `tools` from here, on
purpose: a probe built against anything other than the prompt the agent
actually deploys is measuring a different system.

**[ARCHITECTURE.md](ARCHITECTURE.md)** explains the reasoning behind each
decision, with the measurements behind it. Its module map is stale as of the
`agent/` → `src/` rename and is rewritten in Phase 6 of [PLAN.md](PLAN.md).

Every module can be exercised on its own:

```powershell
.\.venv-agent\Scripts\python.exe -m src.vad            # speech detection
.\.venv-agent\Scripts\python.exe -m src.stt            # transcription
.\.venv-agent\Scripts\python.exe -m src.tts "hello"    # voice
.\.venv-agent\Scripts\python.exe -m src.llm "hi"       # brain
.\.venv-agent\Scripts\python.exe -m src.audio          # mic & speakers
.\.venv-agent\Scripts\python.exe -m src.text           # normaliser
.\.venv-agent\Scripts\python.exe -m src.tools            # tools + event store
```

## Offline proof

```powershell
.\.venv-agent\Scripts\python.exe -m src.offline_check
```

Checks the settings, the weights on disk, that the models genuinely load, and
that Ollama answers. Tested with **every non-localhost socket blocked**: the
full chain ran 7/7 steps with **zero** outbound connection attempts.

## Troubleshooting

| Symptom | Check |
|---|---|
| Hotkey does nothing | `scripts\status.ps1` — two agents? old build? |
| No sound at all | Output device moved. `python -m src.audio` |
| Recording comes out fragmented | Use a single key, not a combination |
| Sentences cut short in session mode | Raise `VAD_SILENCE_MS` |
| Answers are very slow | The first one after idle really is ~7 s (Ollama reload) |
| A fix appears not to work | Check the `build:` line — an old process holds old code |

## Not here yet

**Rescheduling.** The model can read the store, add to it, mark things done,
log hours, and delete — but it cannot *move* an entry. `events.py` has no
update operation at all, and remove-then-add is how a store quietly loses a
row. It is the next tool to build, not one to smuggle in.

Everything below was removed during the restart, and the git history is intact
if you want any of it back:

- **Google Calendar sync** — two-way, over OAuth
- **Cross-session memory** — history and facts that survive a restart
- **Deterministic answers without the LLM** — clock, date, schedule computed in
  Python. Measured 5.7x faster to first sound, and 6/6 correct on the clock
  against 4/6–6/6 wrong through the model
- **Deterministic time parsing** — `time_en` (33/33) and an Indonesian parser

Never built:

- **Wake word** — enter a session without touching a key
- **Interrupting the agent** — needs echo cancellation
- **Proactive notifications** — the agent speaking first

Which of these v2 picks up, and in what order, is decided in
**[PLAN.md](PLAN.md)** — everything here is out of scope until Phase 6 lands.

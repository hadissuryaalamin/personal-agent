# personal-agent

A local voice assistant for Windows. Runs quietly in the background with no
window. Press a hotkey, speak, and it answers through the speakers.
**English, fully offline** — not a byte leaves this machine.

```
hotkey → mic → Silero VAD → Parakeet TDT 0.6B → qwen2.5:7b → Kokoro-82M → speaker
```

Four models, four jobs: **the VAD knows when you have finished speaking**,
**Parakeet hears**, **qwen thinks**, **Kokoro answers**.

That is all it does. No calendar, no task list, no cross-session memory, no
integrations — deliberately emptied out so it can be rebuilt from a clean base.
Conversation history lives only as long as the process.

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
.\.venv-agent\Scripts\python.exe -m agent.offline_check
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
.\.venv-agent\Scripts\python.exe -m agent.main
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
.\.venv-agent\Scripts\python.exe -m agent.main
```

### If something looks wrong, check the build first

The second line of the log names the running commit:

```
build: 9a00404 05/08 08:31 (newest file 05/08 08:31)
```

Python loads source when the process starts — **editing a file does not touch a
running process.** This is the single most common cause of "the fix didn't work".

## Configuration

Defaults live in [agent/config.py](agent/config.py) and can be overridden in
`.env` (see [.env.example](.env.example) — 54 keys, every one of them actually
read by the code).

| Variable | Default | Notes |
|---|---|---|
| `HOTKEY` | `ctrl+space` | **Use a single key** — see below |
| `SESSION_MODE` | `false` | `true` = continuous conversation |
| `OFFLINE_MODE` | `true` | Reject network backends at startup |
| `LLM_BACKEND` | `ollama` | `ollama` or `claude` (needs `OFFLINE_MODE=false`) |
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
agent/
  main.py            hotkey, session mode, pipeline orchestration, logging
  config.py          every constant from .env + offline-mode enforcement
  audio.py           mic capture, gapless playback, beeps
  vad.py             per-frame speech detection (Silero)
  stt.py             Parakeet / Whisper — loads and releases automatically
  llm.py             Ollama / Claude, sentence-by-sentence streaming
  tts.py             Kokoro / Piper
  text.py            text normaliser
  offline_check.py   verifies readiness to run with no network
models/              Kokoro & Piper weights (gitignored)
memory/              agent.lock (gitignored)
scripts/             setup, autostart, status
logs/                agent.log (rotating, 1 MB x 4)
```

~2,300 lines. **[ARCHITECTURE.md](ARCHITECTURE.md)** explains the reasoning
behind each decision, with the measurements behind it.

Every module can be exercised on its own:

```powershell
.\.venv-agent\Scripts\python.exe -m agent.vad            # speech detection
.\.venv-agent\Scripts\python.exe -m agent.stt            # transcription
.\.venv-agent\Scripts\python.exe -m agent.tts "hello"    # voice
.\.venv-agent\Scripts\python.exe -m agent.llm "hi"       # brain
.\.venv-agent\Scripts\python.exe -m agent.audio          # mic & speakers
.\.venv-agent\Scripts\python.exe -m agent.text           # normaliser
```

## Offline proof

```powershell
.\.venv-agent\Scripts\python.exe -m agent.offline_check
```

Checks the settings, the weights on disk, that the models genuinely load, and
that Ollama answers. Tested with **every non-localhost socket blocked**: the
full chain ran 7/7 steps with **zero** outbound connection attempts.

## Troubleshooting

| Symptom | Check |
|---|---|
| Hotkey does nothing | `scripts\status.ps1` — two agents? old build? |
| No sound at all | Output device moved. `python -m agent.audio` |
| Recording comes out fragmented | Use a single key, not a combination |
| Sentences cut short in session mode | Raise `VAD_SILENCE_MS` |
| Answers are very slow | The first one after idle really is ~7 s (Ollama reload) |
| A fix appears not to work | Check the `build:` line — an old process holds old code |

## Not here yet

Everything below was removed during the restart, and the git history is intact
if you want any of it back:

- **Calendar** — read a schedule, create events (local ICS & Google Calendar)
- **Task list** — capture, mark done, choose what to work on
- **Cross-session memory** — history and facts that survive a restart
- **Deterministic answers without the LLM** — clock, date, schedule computed in
  Python. Measured 5.7x faster to first sound, and 6/6 correct on the clock
  against 4/6–6/6 wrong through the model
- **Deterministic time parsing** — `time_en` (33/33) and an Indonesian parser

Never built:

- **Wake word** — enter a session without touching a key
- **Interrupting the agent** — needs echo cancellation
- **Proactive notifications** — the agent speaking first

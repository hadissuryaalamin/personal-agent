# MIGRATE_PLAN

## Goal

Migrate the current personal voice assistant to a fully local, offline English voice stack:

```text
Microphone
    ↓
Parakeet TDT 0.6B v2
    ↓
Qwen2.5 7B
    ↓
Kokoro-82M
    ↓
Speaker
```

The application must continue to work without an internet connection after all required model files have been downloaded.

For now, optimize only for **English**.

---

## 1. Current Architecture

The existing application is a Windows background personal voice assistant.

The current interaction flow is:

```text
Hotkey pressed
    ↓
Record microphone
    ↓
Speech-to-Text
    ↓
Intent routing
    ↓
LLM when required
    ↓
Text-to-Speech
    ↓
Speaker
```

Current major modules:

```text
main.py       → listener, orchestration, intent routing
audio.py      → microphone recording, playback, beep
stt.py        → current Whisper STT backend
llm.py        → current LLM backends
tts.py        → current Piper TTS backend

calendar.py
memory.py
tugas.py
jadwal_baru.py
waktu_id.py
gcal.py
kalender_lokal.py
config.py
```

Do not redesign unrelated features unless necessary.

Preserve the existing:

- hotkey interaction
- recording behavior
- intent routing
- calendar logic
- memory
- task handling
- confirmation flow
- background listener
- modular architecture

The migration should focus primarily on replacing the model stack and converting Indonesian-specific behavior to English.

---

## 2. Target Architecture

```text
HOTKEY
  ↓
MICROPHONE
  ↓
PARAKEET TDT 0.6B v2
  ↓
INTENT ROUTER
  ├──── deterministic/local actions
  │
  └──── QWEN2.5 7B
             ↓
         KOKORO-82M
             ↓
          SPEAKER
```

Architectural separation must remain:

```text
main
 ├── audio
 ├── stt
 ├── llm
 ├── tts
 ├── calendar
 ├── memory
 └── task / intent modules
```

Do not combine STT, LLM, and TTS into one large module.

Each model must remain independently replaceable.

---

# 3. STT Migration

## Target

Replace Whisper with:

```text
nvidia/parakeet-tdt-0.6b-v2
```

Parakeet becomes the English Speech-to-Text backend.

Target flow:

```text
Microphone
    ↓
audio.py
    ↓
stt.py
    ↓
Parakeet
    ↓
English text
```

## Requirements

`stt.py` should remain the abstraction boundary.

Prefer an interface conceptually similar to:

```python
load()
transcribe(audio)
unload()
```

or preserve the existing public interface if it already differs.

`main.py` should not contain Parakeet-specific code.

The rest of the application should not need to know whether STT is implemented by Whisper or Parakeet.

## Investigate

Before implementation, determine:

- recommended Parakeet runtime
- NVIDIA NeMo dependency requirements
- whether a lighter inference path exists
- Windows compatibility
- WSL compatibility
- GPU support
- required sample rate
- input audio format
- preprocessing requirements
- VRAM usage
- cold-start latency
- warm inference latency
- model unloading behavior

## Error Handling

Handle:

- model files missing
- model load failure
- GPU unavailable
- GPU out-of-memory
- microphone input invalid
- empty audio
- transcription returning empty text

---

# 4. LLM Migration

## Target

Use:

```text
Qwen2.5 7B Instruct
```

Prefer a quantized local model:

```text
Qwen2.5 7B Instruct
4-bit quantization
```

The model must run locally.

## Existing Architecture

Preserve:

```text
main.py
   ↓
llm.py
   ↓
local LLM runtime
```

Do not make `main.py` depend directly on:

- Ollama
- llama.cpp
- vLLM
- Transformers

`llm.py` must remain responsible for the backend abstraction.

## Runtime

Inspect the existing repository before changing the runtime.

If the existing Ollama integration already supports Qwen2.5 7B reliably, prefer reusing it.

Avoid introducing another inference server unless there is a clear technical reason.

## Preserve Existing Logic

Do not change the business logic around:

- calendar context
- conversation history
- memory
- tasks
- intent routing
- deterministic commands

unless required for the English migration.

---

# 5. TTS Migration

## Target

Replace Piper with:

```text
Kokoro-82M
```

Target flow:

```text
LLM / local action
    ↓
text
    ↓
tts.py
    ↓
Kokoro-82M
    ↓
audio waveform
    ↓
audio.py
    ↓
speaker
```

## Requirements

`tts.py` must remain the abstraction boundary.

`main.py` should only need to do something conceptually similar to:

```python
tts.speak(response)
```

Do not place Kokoro-specific implementation details in `main.py`.

## Investigate

Determine:

- recommended local Kokoro runtime
- model loading method
- English voice selection
- sample rate
- output waveform format
- CPU inference performance
- GPU inference performance
- model initialization latency
- whether Kokoro should stay resident in memory
- sentence-level synthesis support
- future streaming possibilities

For the initial migration, prioritize reliability over streaming.

---

# 6. Offline Requirement

After setup and model download, the following must work with network access disabled:

```text
record speech
transcribe speech
process intents
generate LLM response
generate speech
play response
read/write local memory
read/write local calendar
```

Cloud services must not be required for the core voice pipeline.

Google Calendar may remain an optional external integration.

The core agent must continue to function if Google Calendar or internet access is unavailable.

If Claude or other cloud backends currently exist, they may remain optional, but offline mode must not depend on them.

Consider configuration similar to:

```env
OFFLINE_MODE=true

STT_BACKEND=parakeet
LLM_BACKEND=ollama
TTS_BACKEND=kokoro
```

Use the project's existing configuration naming conventions when possible.

---

# 7. English Migration

The current system was designed around Bahasa Indonesia.

The new version should target English.

Inspect the repository for Indonesian-specific behavior in:

- system prompts
- confirmation phrases
- intent keywords
- task commands
- calendar commands
- date/time parsing
- status messages
- STT fallback messages
- TTS responses

Examples:

```text
"catat tugas ..."
→
"add a task ..."

"buat acara ..."
→
"create an event ..."

"lupakan semua"
→
"forget everything"
```

Do not simply remove existing functionality.

Convert the existing behavior cleanly.

---

# 8. Date and Time Parsing

Pay special attention to:

```text
waktu_id.py
```

This module is currently Indonesian-specific.

Evaluate three options:

1. rewrite it into an English deterministic parser
2. create a new `time_en.py`
3. generalize date parsing into a language-independent interface

Prefer deterministic parsing over asking the LLM to calculate dates.

Target concept:

```text
"tomorrow at 3 PM"
    ↓
deterministic parser
    ↓
exact datetime
```

The LLM should handle semantic understanding where useful, but exact date computation should remain deterministic whenever practical.

---

# 9. Model Lifecycle

The existing application does not keep large models loaded permanently.

Preserve this design goal where practical, but evaluate each new model independently.

Possible strategy:

```text
Parakeet → unload after idle
Qwen     → unload after idle
Kokoro   → keep loaded
```

Do not assume the same lifecycle strategy should apply to all models.

Measure:

- VRAM while loaded
- RAM while loaded
- cold load time
- warm load time
- inference latency
- idle cost

Kokoro is small enough that keeping it resident may significantly improve perceived latency.

Parakeet and Qwen may benefit more from idle unloading.

---

# 10. Hotkey and Audio Behavior

Preserve the existing interaction model.

Target:

```text
press hotkey
    ↓
start microphone
    ↓
user speaks
    ↓
press/release according to current mode
    ↓
stop recording
    ↓
STT
```

Preserve current design decisions unless the migration makes them impossible:

- single hotkey interaction
- `pynput` backend
- processing pipeline on a separate thread
- non-blocking start beep
- playback padding
- background listener
- model preload while user is speaking if practical

Do not rewrite stable hotkey/audio logic only for stylistic reasons.

---

# 11. Intent Routing

Preserve the existing rule that deterministic/local actions are checked before falling back to the LLM.

Target:

```text
Transcription
    ↓
Intent Router
    ├── forget memory
    ├── confirmation
    ├── complete task
    ├── add task
    ├── create event
    └── general LLM request
```

Convert matching rules from Indonesian to English.

Ordering must remain deliberate because commands can overlap.

Do not route everything through Qwen.

Use deterministic code for closed, rule-based operations whenever practical.

---

# 12. Calendar Safety

Preserve confirmation before calendar writes.

Target example:

```text
User:
"Create an event tomorrow at 3 PM called project meeting."

Agent:
"Create 'Project Meeting' tomorrow from 3 PM ...?"

User:
"Yes."

Agent:
event saved
```

If date or time parsing is uncertain:

```text
reject / ask for clarification
```

Do not silently guess and write an event.

---

# 13. Memory

Preserve the existing distinction between:

- conversation history
- persistent user facts
- tasks
- calendar data

Do not copy calendar or task data into persistent fact memory.

Keep bounded conversation history.

The migration should not expand memory scope unless explicitly required.

---

# 14. Configuration

Centralize model configuration in:

```text
config.py
.env
```

Possible configuration:

```env
OFFLINE_MODE=true

STT_BACKEND=parakeet
STT_MODEL=nvidia/parakeet-tdt-0.6b-v2

LLM_BACKEND=ollama
LLM_MODEL=qwen2.5:7b

TTS_BACKEND=kokoro
TTS_VOICE=<english_voice>

MODEL_IDLE_TIMEOUT=<seconds>
```

These names are examples.

Reuse existing configuration conventions when possible.

Avoid hardcoding model names or device settings throughout the codebase.

---

# 15. Hardware Strategy

This application is intended for a consumer NVIDIA laptop GPU.

Do not assume all three models must stay on the GPU simultaneously.

Evaluate:

```text
Parakeet → GPU
Qwen     → GPU
Kokoro   → CPU or GPU
```

Possible runtime flow:

```text
Idle
 ↓
minimal GPU usage

Hotkey
 ↓
load/wake Parakeet

Speech finished
 ↓
STT inference

Parakeet no longer needed
 ↓
Qwen inference

Qwen response
 ↓
Kokoro synthesis

Idle timeout
 ↓
release expensive models
```

Measure combined peak memory.

Estimate:

- Parakeet VRAM
- Qwen 4-bit VRAM
- Kokoro RAM/VRAM
- framework overhead
- CUDA overhead
- combined peak VRAM

Avoid GPU OOM by design rather than relying only on exception handling.

---

# 16. Latency Measurement

Measure each stage separately.

Conceptually:

```text
T_total =
    T_STT
  + T_LLM_first_token
  + T_LLM_generation
  + T_TTS_start
  + T_playback
```

Add timing logs for:

```text
recording duration
STT cold load
STT inference
LLM cold load
LLM first token
LLM generation
TTS cold load
TTS synthesis
time to first audible response
total pipeline latency
```

Do not optimize blindly.

Use measured results.

---

# 17. Streaming

Do not implement complex full streaming during the first migration.

Initial target:

```text
record complete utterance
    ↓
transcribe
    ↓
generate complete or mostly complete response
    ↓
synthesize
    ↓
play
```

However, avoid architectural decisions that would block future:

```text
streaming STT
    ↓
streaming LLM
    ↓
sentence/chunk TTS
    ↓
speaker
```

Future streaming should be possible without rewriting the entire application.

---

# 18. Error Handling

The final system should gracefully handle:

## Audio

- microphone missing
- device unavailable
- recording failure
- empty recording

## STT

- Parakeet files missing
- Parakeet load failure
- invalid audio
- empty transcription
- CUDA failure
- GPU OOM

## LLM

- Ollama/local runtime unavailable
- Qwen model missing
- generation failure
- model load failure
- GPU OOM

## TTS

- Kokoro model missing
- voice unavailable
- synthesis failure
- invalid output waveform
- playback failure

## Offline Mode

If:

```text
OFFLINE_MODE=true
```

and a cloud-only backend is selected, fail clearly at startup rather than attempting network access silently.

---

# 19. Testing

## STT Tests

Use recordings such as:

```text
"What is my schedule tomorrow?"

"Add a task to finish the assignment on Friday."

"Create an event tomorrow at three PM."
```

Measure:

```text
transcription accuracy
latency
cold start
warm start
VRAM
```

---

## LLM Tests

Verify:

```text
general questions
calendar questions
task queries
conversation context
memory use
short answers
long answers
```

Confirm existing context-building behavior still works.

---

## TTS Tests

Test:

```text
short responses
long responses
numbers
times
dates
punctuation
questions
```

Measure:

```text
time to first audio
total synthesis time
naturalness
audio clipping
playback completeness
```

---

## Intent Tests

Verify English equivalents for:

```text
forget everything
add a task
mark a task complete
create an event
confirm event
cancel event
general question
```

Ensure routing precedence remains correct.

---

## End-to-End Tests

Test:

```text
hotkey
→ speech
→ transcription
→ intent routing
→ response
→ TTS
→ speaker
```

Test both:

```text
deterministic action path
```

and:

```text
LLM response path
```

---

## Offline Validation

After all model files are available locally:

1. disconnect network access
2. start the assistant
3. run STT
4. ask Qwen a general question
5. create local TTS output
6. access local memory
7. access local calendar
8. create a local task
9. restart the application
10. verify persisted state

Core functionality must continue to work.

---

# 20. Migration Milestones

## Milestone 1 — Architecture Audit

Inspect the repository and document:

- current `stt.py` interface
- current `llm.py` interface
- current `tts.py` interface
- model lifecycle
- device handling
- configuration
- Indonesian-specific dependencies
- current tests
- current startup flow

### Deliverable

A concrete file-by-file implementation plan.

Do not make major code changes yet.

---

## Milestone 2 — Parakeet Migration

Replace Whisper with Parakeet while keeping:

```text
current LLM
current Piper TTS
```

### Goals

- Parakeet loads locally
- recorded audio is accepted
- English transcription works
- current intent router receives text unchanged
- no Parakeet-specific code leaks into `main.py`

### Acceptance Criteria

```text
Mic
→ Parakeet
→ correct English text
→ existing pipeline
```

works reliably.

---

## Milestone 3 — Offline Qwen

Make Qwen2.5 7B the default local LLM.

Prefer existing Ollama integration if suitable.

### Goals

- no cloud API required
- Qwen receives existing system/context prompt
- calendar context still works
- conversation history still works
- memory still works
- tasks still work

### Acceptance Criteria

With network disabled:

```text
text
→ Qwen2.5 7B
→ valid response
```

works.

---

## Milestone 4 — Kokoro Migration

Replace Piper with Kokoro.

### Goals

- English voice configured
- local synthesis
- output passed through existing playback layer
- no Kokoro-specific code in `main.py`

### Acceptance Criteria

```text
text
→ Kokoro
→ WAV/audio waveform
→ speaker
```

works reliably.

---

## Milestone 5 — English Migration

Convert Indonesian-specific behavior to English.

Inspect and migrate:

```text
system prompts
intent patterns
task commands
event commands
confirmation phrases
date parsing
status messages
error messages
```

### Acceptance Criteria

All main voice commands work naturally in English without depending on Indonesian patterns.

---

## Milestone 6 — Resource Management

Benchmark:

```text
Parakeet
Qwen
Kokoro
```

independently.

Measure:

- RAM
- VRAM
- cold startup
- warm startup
- inference time
- unload time

Determine the best lifecycle for each model.

### Possible Target

```text
Parakeet → lazy load + idle unload
Qwen     → lazy load + idle unload
Kokoro   → persistent
```

Only adopt this if measurements support it.

---

## Milestone 7 — End-to-End Offline Validation

Run the entire assistant with internet disabled.

Validate:

```text
Hotkey
→ microphone
→ Parakeet
→ intent router
→ Qwen/local action
→ Kokoro
→ speaker
```

Also verify:

- local tasks
- local calendar
- memory
- history
- restart persistence
- error recovery

---

# 21. What to Preserve

Do not rewrite these unless technically necessary:

```text
main orchestration structure
audio recording logic
hotkey handling
threading model
playback padding
intent precedence
calendar confirmation flow
memory separation
task storage
local calendar storage
bounded conversation history
config-based backend selection
```

Prefer changing implementation behind existing interfaces.

---

# 22. Expected Final Architecture

```text
                    ┌────────────────────────────┐
                    │          main.py           │
                    │ listener + orchestration   │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │          audio.py          │
                    │ mic / beep / playback      │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │           stt.py           │
                    │ Parakeet TDT 0.6B v2       │
                    └─────────────┬──────────────┘
                                  │
                               text
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │       Intent Router        │
                    └────────┬───────────┬───────┘
                             │           │
                    local action         │
                             │           ▼
                             │   ┌─────────────────┐
                             │   │     llm.py      │
                             │   │ Qwen2.5 7B      │
                             │   └────────┬────────┘
                             │            │
                             └──────┬─────┘
                                    │
                                  text
                                    │
                                    ▼
                         ┌────────────────────┐
                         │       tts.py       │
                         │    Kokoro-82M      │
                         └─────────┬──────────┘
                                   │
                                 audio
                                   │
                                   ▼
                               Speaker
```

---

# 23. Instructions for Implementation Agent

Before implementing anything:

1. inspect the existing repository carefully
2. understand the current interfaces
3. identify exactly which files need modification
4. identify which files can remain unchanged
5. inspect current dependency management
6. inspect current environment/config handling
7. inspect existing tests
8. identify Indonesian-specific behavior
9. identify model lifecycle code
10. identify Windows/WSL assumptions

Then produce a concrete implementation plan.

For every milestone include:

1. files that need to change
2. specific functions/classes affected
3. new dependencies
4. architecture changes
5. risks
6. tests
7. acceptance criteria

Do not implement the migration yet.

Do not propose a full rewrite unless there is a strong technical reason.

Prefer incremental changes behind the existing abstractions.

The final target core stack is:

```text
Parakeet TDT 0.6B v2
        ↓
Intent Router
        ↓
Qwen2.5 7B
        ↓
Kokoro-82M
```

with all core inference running locally and offline.

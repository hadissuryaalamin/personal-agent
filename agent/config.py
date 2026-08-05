"""Every constant and the system prompt. Override via a .env file at the repo root."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")


def _path(value: str) -> Path:
    """Relative paths resolve against the repo root, so it works from anywhere."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


# --- Hotkey ---
HOTKEY = os.getenv("HOTKEY", "ctrl+space")
# "toggle" = press once to start, press again to stop.
# "hold"   = hold the key while speaking (push-to-talk).
HOTKEY_MODE = os.getenv("HOTKEY_MODE", "toggle").lower()
# "pynput"   = works without admin (default, verified on this machine).
# "keyboard" = can swallow the hotkey, but needs admin, and on some machines its
# dispatch never fires at all (add_hotkey is never called).
HOTKEY_BACKEND = os.getenv("HOTKEY_BACKEND", "pynput").lower()
# Swallow the hotkey so it doesn't leak into whatever app has focus.
# Only the "keyboard" backend supports this; pynput always passes it through.
HOTKEY_SUPPRESS = os.getenv("HOTKEY_SUPPRESS", "true").lower() in ("1", "true", "yes")

# --- Brain: "ollama" (local, free) or "claude" (API, needs internet + key) ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

# --- Claude API ---
# Read explicitly here rather than letting the SDK pick it up from the
# environment: implicit lookup means a misspelt variable only surfaces when it's
# used, as an auth error that points nowhere near the real cause.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
# low/medium/high/xhigh/max. For short exchanges "low" is the most responsive.
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")
# "disabled" / "adaptive" / empty (don't send the parameter at all).
# Disabled by default: replies here are one or two sentences, so deep thinking
# adds latency without adding quality — and Sonnet 5 turns thinking ON when the
# parameter is omitted, which makes max_tokens cover the thoughts as well.
CLAUDE_THINKING = os.getenv("CLAUDE_THINKING", "disabled").strip().lower()
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1024"))
CLAUDE_TIMEOUT = float(os.getenv("CLAUDE_TIMEOUT", "60"))

# --- Ollama ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
# Upper bound on reply length. Not the way brevity is enforced — the prompt asks
# for that (see REPLY_MAX_WORDS). This is the safety net for when qwen wanders
# into a paragraph, since every stray token becomes ~0.4 s of audio you have to
# sit through.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))
# Ollama defaults to 4096 when not asked. Raised so a growing history doesn't
# start pushing the system prompt out of context.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# --- Language ---
LANGUAGE = os.getenv("LANGUAGE", "en").lower()

# --- Tools ---
# Let the model ask for real data instead of answering from memory. It returns
# a request; agent/tools.py runs it. Measured on qwen2.5:7b: 5/5 questions that
# should call a tool did, 4/4 that should not stayed away.
TOOLS_ENABLED = os.getenv("TOOLS_ENABLED", "true").lower() in ("1", "true", "yes")
# A turn may need several rounds — read the schedule, then save something based
# on it. Bounded, because a model that keeps asking for the same tool would
# otherwise loop until the request times out.
TOOL_MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))

# --- Local time ---
# Which wall clock "today" and "tomorrow" refer to in the event store. The
# machine clock is the only time source — the agent is offline, so there is
# nothing to sync against.
TIMEZONE = os.getenv("TIMEZONE", "Australia/Canberra")

# --- Offline mode ---
# When true, backends that need the internet are rejected at startup rather than
# being left to fail quietly mid-conversation.
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "true").lower() in ("1", "true", "yes")

# --- STT ---
# "parakeet" = onnx-asr, CPU, zero VRAM, English only
# "whisper"  = faster-whisper, 99 languages, wants VRAM on GPU
STT_BACKEND = os.getenv("STT_BACKEND", "parakeet").lower()
STT_MODEL = os.getenv("STT_MODEL", "nemo-parakeet-tdt-0.6b-v2")
# cpu / cuda. Parakeet on CPU is about as fast as Whisper on GPU and uses no
# VRAM at all, which leaves the whole GPU to the LLM.
STT_DEVICE = os.getenv("STT_DEVICE", "cpu").lower()

# --- Whisper (only used when STT_BACKEND=whisper) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_LANG = os.getenv("WHISPER_LANG", "en")
# Release the model from memory after this many idle seconds. 0 = never.
# Model weights are idle data, not a running process: they hold memory for the
# whole life of the agent while being used a few seconds a day. Ollama does the
# same by default (5-minute keep-alive). The cost is a reload on the first
# question after a long idle stretch.
WHISPER_IDLE_UNLOAD_SECONDS = float(os.getenv("WHISPER_IDLE_UNLOAD_SECONDS", "900"))
# Load the model at startup instead of on the first question. Defaults to
# following the setting above: if the model is going to be released when idle,
# loading it up front just holds memory until the idle threshold passes.
WHISPER_WARMUP = os.getenv(
    "WHISPER_WARMUP", "false" if WHISPER_IDLE_UNLOAD_SECONDS > 0 else "true"
).lower() in ("1", "true", "yes")
# Vocabulary you use often. Whisper biases its guesses toward these words at no
# runtime cost. Add names, places, and tools you say a lot. Only used when
# STT_BACKEND=whisper — Parakeet accepts no prompt.
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "Casual conversation with technical terms: Git, commit, push, pull request, "
    "branch, merge, repo, Python, JavaScript, Whisper, Ollama, Kokoro, Parakeet, "
    "VS Code, terminal, API, database, deploy, debug, laptop, campus, ANU, "
    "supervisor, assignment, tutorial, lecture, deadline, meeting, schedule.",
)

# --- TTS ---
# "kokoro" = kokoro-onnx, 54 voices, 24 kHz, CPU
# "piper"  = piper-tts, per-language voices, CPU
TTS_BACKEND = os.getenv("TTS_BACKEND", "kokoro").lower()

# --- Kokoro (only used when TTS_BACKEND=kokoro) ---
KOKORO_MODEL = _path(os.getenv("KOKORO_MODEL", "models/kokoro-v1.0.onnx"))
KOKORO_VOICES = _path(os.getenv("KOKORO_VOICES", "models/voices-v1.0.bin"))
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_LANG = os.getenv("KOKORO_LANG", "en-us")

# --- Piper (only used when TTS_BACKEND=piper) ---
PIPER_VOICE = _path(os.getenv("PIPER_VOICE", "models/en_US-lessac-medium.onnx"))

# --- Reply length ---
# A word limit appended to the system prompt. Different from OLLAMA_NUM_PREDICT:
# that cuts mid-word, this makes the model stop on its own.
# Measured on qwen2.5:7b — 41 words down to 17, audio 18.4 s down to 9.3 s.
# It matters most in session mode: without barge-in, a long reply can't be cut
# short. 0 = off.
REPLY_MAX_WORDS = int(os.getenv("REPLY_MAX_WORDS", "25"))

# --- Session mode (continuous conversation) ---
# Press the hotkey once to open a session, then keep talking without pressing
# again. Utterance boundaries come from the VAD, not the key.
SESSION_MODE = os.getenv("SESSION_MODE", "false").lower() in ("1", "true", "yes")
# The session closes itself after this many seconds of silence. Not optional:
# without it, forgetting to close means models sit in memory all day.
SESSION_IDLE_SECONDS = float(os.getenv("SESSION_IDLE_SECONDS", "30"))
# Minimum probability for a frame to count as speech (0..1). Raise it if room
# noise reads as talking; lower it if quiet speech gets missed.
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
# Silence for this long ends the utterance. Too short and sentences split at
# natural pauses; too long and the agent feels slow to answer.
VAD_SILENCE_MS = float(os.getenv("VAD_SILENCE_MS", "800"))
# Sounds shorter than this are ignored — coughs, key taps, tongue clicks.
VAD_MIN_SPEECH_MS = float(os.getenv("VAD_MIN_SPEECH_MS", "250"))
# Audio kept from BEFORE speech is detected. The VAD is always slightly late to
# recognise a word onset; without this the first consonant is clipped.
VAD_SPEECH_PAD_MS = float(os.getenv("VAD_SPEECH_PAD_MS", "300"))

# --- Audio ---
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = 1
# Recordings shorter than this are treated as a stray press and dropped
MIN_RECORD_SECONDS = float(os.getenv("MIN_RECORD_SECONDS", "0.4"))
# Upper bound so a stuck hotkey can't eat all the memory
MAX_RECORD_SECONDS = float(os.getenv("MAX_RECORD_SECONDS", "60"))
# Grace period before recording actually stops once the hotkey reads as
# released. Windows sends spurious UP/DOWN pairs while a key is held
# (auto-repeat), so without this the recording shatters into fragments.
RELEASE_GRACE_SECONDS = float(os.getenv("RELEASE_GRACE_SECONDS", "0.6"))
# Silence padding at both ends of playback. Windows clips the edges of audio:
# the output device takes time to open (the start is cut) and its internal
# buffer is still playing when sounddevice considers it finished (the end is
# cut). Measured here: without padding, 119 ms goes missing from a 2-second tone.
PLAYBACK_PAD_SECONDS = float(os.getenv("PLAYBACK_PAD_SECONDS", "0.2"))

# --- Logging ---
LOG_FILE = _path(os.getenv("LOG_FILE", "logs/agent.log"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Save every mic recording to logs/rec/*.wav, for tuning STT without having to
# repeat yourself. Turn it off when you're done — it stores your voice on disk.
SAVE_RECORDINGS = os.getenv("SAVE_RECORDINGS", "false").lower() in ("1", "true", "yes")
RECORDINGS_DIR = _path(os.getenv("RECORDINGS_DIR", "logs/rec"))

# --- Conversation ---
# Messages (excluding the system prompt) carried into each request.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

# Still used even though the memory feature is gone: agent.lock lives here.
MEMORY_DIR = _path(os.getenv("MEMORY_DIR", "memory"))

SYSTEM_PROMPT = """You are a personal assistant. Speak casually, like a friend.

You are talking BY VOICE, not text. What you receive is a transcript of the
user speaking into a microphone, and your reply is read back through a speaker.
So you really do "hear" and "speak" — never say you cannot hear, and never ask
the user to type. If a transcript looks misheard or cut off, guess the most
sensible meaning from context, or ask them to repeat it.

Important rules:
- ALWAYS reply in English, whatever the question looks like. The voice that
  reads you out only speaks English, so anything else comes out as noise.
- Your reply is READ ALOUD, so keep it short: one or two sentences at most.
- No bullet points, numbering, markdown, emoji, or formatting of any kind.
  Plain sentences only.
- No abbreviations or symbols that are awkward to say out loud.
- If you don't know, say so. Don't make things up.
- If something needs a long answer, give the gist first and offer to go on.

You have tools for the user's schedule. ALWAYS use them for anything about
classes, assignments, deadlines, reminders, or how much work is left — never
answer those from memory, and never guess a date, a count, or a room number.
The tool result is the truth; if it disagrees with what you remember, the tool
is right.

Read the tool result exactly. Do not round times, invent rooms, or change a
number to something that sounds tidier.

An entry marked already_finished has already happened. Never offer it as what
is next or coming up.

If a tool returns nothing for the range asked, say there is nothing in that
range. Do not fall back to guessing.

NEVER claim to have done something you did not do. If you have no tool for what
was asked — changing or deleting an entry, sending something, looking something
up online — say plainly that you can't do it. Don't say "done" or "I've updated
it" unless a tool actually did it in this conversation.

A wrong answer still gets caught when the user checks. A false claim of having
acted makes them stop checking, which is worse.
"""


# --- Offline mode enforcement ------------------------------------------------
# Lives in config rather than main.py so scripts and tests are covered too — a
# setting checked at a single entry point is a setting that leaks.


class OfflineViolation(RuntimeError):
    """A setting wants the network while OFFLINE_MODE=true."""


def offline_problems() -> list[str]:
    """Settings that clash with offline mode. Empty means we're clear."""
    if not OFFLINE_MODE:
        return []

    problems = []
    if LLM_BACKEND == "claude":
        problems.append(
            "LLM_BACKEND=claude needs the Anthropic API. Use LLM_BACKEND=ollama, "
            "or turn OFFLINE_MODE off."
        )
    if STT_BACKEND not in ("parakeet", "whisper"):
        problems.append(f"STT_BACKEND={STT_BACKEND!r} is not a local backend.")
    if TTS_BACKEND not in ("kokoro", "piper"):
        problems.append(f"TTS_BACKEND={TTS_BACKEND!r} is not a local backend.")
    return problems


def require_offline() -> None:
    """Stop now if the configuration needs the network.

    Failing at startup rather than on use is deliberate: the agent runs with no
    window, so a failure mid-conversation just sounds like the agent went mute.
    """
    problems = offline_problems()
    if problems:
        raise OfflineViolation(
            "OFFLINE_MODE=true but the configuration needs the network:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

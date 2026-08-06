"""Entry point: hotkey listener and the STT -> LLM -> TTS pipeline."""

from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib
import sys
import threading
import time
from typing import Callable

from . import audio, config, llm, stt, tray, tts, vad
from . import text as text_utils

log = logging.getLogger("agent")


def _build_version() -> str:
    """The running commit, plus whether any file is newer than the process.

    Written to the startup log because it caught us out twice: the code was
    fixed, but what got tested was an old process still holding the old code —
    Python loads source at start, and editing a file does not touch a running
    process. The symptom misleads: it looks like the fix did not work.
    """
    import subprocess

    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad", "--date=format:%d/%m %H:%M"],
            cwd=pathlib.Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        commit = r.stdout.strip() or "?"
    except Exception:
        commit = "?"

    # A .py newer than the commit means uncommitted changes; harmless, but
    # worth flagging when comparing logs.
    try:
        newest = max(
            f.stat().st_mtime for f in pathlib.Path(__file__).parent.glob("*.py")
        )
        from datetime import datetime

        return f"{commit} (newest file {datetime.fromtimestamp(newest):%d/%m %H:%M})"
    except Exception:
        return commit


# --- Single instance ---------------------------------------------------------


_lock_handle = None  # held for the life of the process; must not be collected


def claim_single_instance() -> int | None:
    """Ensure only ONE agent runs. Returns the holder's PID on failure.

    Why this is enforced in code rather than left to habit: the agent grabs a
    global hotkey. Two agents means every press is caught by both, and both
    fight over the microphone. This actually happened — autostart launched one
    at login, another was started by hand for testing, and session mode looked
    broken because the old agent was the one answering.

    Uses an OS file lock rather than a plain PID file: the OS releases the lock
    when the process dies, so a crashed agent does not leave behind a file that
    blocks every future start.
    """
    global _lock_handle
    path = config.MEMORY_DIR / "agent.lock"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import msvcrt
    except ImportError:  # not Windows — skip rather than refuse to start
        log.debug("single-instance lock skipped: msvcrt unavailable")
        return None

    f = open(path, "a+")
    try:
        f.seek(0)
        # Lock byte 0 only. The PID is written from byte 1 onward, so writing
        # it never touches the locked byte.
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.seek(1)
        held_by = f.read().strip()
        f.close()
        try:
            return int(held_by)
        except ValueError:
            return 0  # someone holds it, but the PID is unreadable

    f.seek(1)
    f.truncate(1)
    f.write(str(os.getpid()))
    f.flush()
    _lock_handle = f  # held deliberately: closing the file releases the lock
    return None


# --- Logging ----------------------------------------------------------------


def setup_logging() -> None:
    """Log to a file (required: runs in the background with no console), plus
    the console when one exists."""
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        config.LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # pythonw.exe has no stdout — only attach a console handler if one exists
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        root.addHandler(console)

    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Exceptions escaping any thread still reach the log
    sys.excepthook = lambda *exc: log.critical("uncaught exception", exc_info=exc)
    threading.excepthook = lambda args: log.critical(
        "uncaught exception in thread %s",
        args.thread.name if args.thread else "?",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


# --- Hotkey backends --------------------------------------------------------


class _BackendBase:
    """Everything the hotkey backends share.

    Two modes:
    - toggle: first press starts recording, second press stops
    - hold:   record while the key is held (push-to-talk)
    """

    name = "?"
    # Presses closer together than this count as one. Guards against the
    # spurious UP/DOWN events Windows sends while a key is held.
    EDGE_DEBOUNCE = 0.3

    def __init__(self, combo: str, on_activate: Callable[[Callable[[], bool]], None]):
        self.combo_label = combo
        self.on_activate = on_activate
        self._busy = False  # pipeline is running
        self._recording = False  # toggle mode: currently recording
        self._last_edge = 0.0
        self._is_held: Callable[[], bool] = lambda: False

    def _on_edge(self) -> None:
        """The hotkey combo just became complete (fires once per press)."""
        now = time.monotonic()
        if now - self._last_edge < self.EDGE_DEBOUNCE:
            return
        self._last_edge = now

        if config.HOTKEY_MODE == "hold":
            self._launch(self._is_held)
            return

        if self._recording:
            self._recording = False  # second press: stop recording
            log.debug("toggle: stop")
        elif not self._busy:
            self._recording = True  # first press: start recording
            log.debug("toggle: start")
            self._launch(lambda: self._recording)
        else:
            log.info("still working on the previous turn, hotkey ignored")
            # Do not stay silent: the user needs to know they were heard
            threading.Thread(target=audio.beep_busy, daemon=True).start()

    def _launch(self, is_recording: Callable[[], bool]) -> None:
        """Run the pipeline on its own thread.

        This MUST be separate: the backend callback runs on the same thread
        that processes keyboard events. Block it and the second press (toggle)
        or the release event (hold) is never seen.
        """
        if self._busy:
            return
        self._busy = True

        def _run():
            try:
                self.on_activate(is_recording)
            finally:
                self._busy = False
                self._recording = False

        threading.Thread(target=_run, name="utterance", daemon=True).start()


class KeyboardBackend(_BackendBase):
    """The `keyboard` backend. Most accurate, but needs admin on Windows."""

    name = "keyboard"

    def run(self) -> None:
        import keyboard as kb

        self._is_held = lambda: kb.is_pressed(self.combo_label)

        kb.add_hotkey(
            self.combo_label,
            self._on_edge,
            suppress=config.HOTKEY_SUPPRESS,
        )
        log.info(
            "Hotkey '%s' active, %s mode (keyboard backend). Ctrl+C to stop.",
            self.combo_label,
            "session" if config.SESSION_MODE else config.HOTKEY_MODE,
        )
        kb.wait()


class PynputBackend(_BackendBase):
    """The `pynput` backend. Works without admin; select via HOTKEY_BACKEND=pynput.

    Note: pynput cannot suppress the hotkey, so `Ctrl+Space` still reaches
    whichever application has focus.
    """

    name = "pynput"

    _ALIASES = {
        "control": "ctrl",
        "win": "cmd",
        "super": "cmd",
        "meta": "cmd",
        "esc": "escape",
        "return": "enter",
        "rctrl": "right ctrl",
        "lctrl": "left ctrl",
        "ctrl_r": "right ctrl",
        "ctrl_l": "left ctrl",
        "ralt": "right alt",
        "lalt": "left alt",
        "rshift": "right shift",
        "lshift": "left shift",
    }

    # Suffix pynput -> kata sisi
    _SIDES = {"_l": "left", "_r": "right", "_gr": "right"}

    def __init__(self, combo: str, on_activate: Callable[[Callable[[], bool]], None]):
        super().__init__(combo, on_activate)
        self.combo = {
            self._ALIASES.get(p.strip().lower(), p.strip().lower())
            for p in combo.split("+")
            if p.strip()
        }
        # Store the raw keys held down; names are resolved on demand, so
        # holding left and right ctrl together cannot corrupt the state.
        self._pressed: set[str] = set()
        self._is_held = self._combo_complete
        self._combo_was_complete = False

    def _raw_name(self, key) -> str | None:
        from pynput import keyboard as pk

        if isinstance(key, pk.Key):
            return key.name
        if isinstance(key, pk.KeyCode):
            if key.char:
                return key.char.lower()
            # While a modifier is held the char becomes a control character; use vk.
            if key.vk is not None and 65 <= key.vk <= 90:
                return chr(key.vk).lower()
        return None

    def _names(self, raw: str) -> set[str]:
        """Every name a single key satisfies.

        ctrl_r matches both 'ctrl' and 'right ctrl', so you can be side-specific
        or not, as you prefer.
        """
        for suffix, side in self._SIDES.items():
            if raw.endswith(suffix):
                base = raw[: -len(suffix)]
                base = self._ALIASES.get(base, base)
                return {base, f"{side} {base}"}
        return {self._ALIASES.get(raw, raw)}

    def _combo_complete(self) -> bool:
        active: set[str] = set()
        for raw in tuple(self._pressed):
            active |= self._names(raw)
        return self.combo <= active

    def _on_press(self, key) -> None:
        raw = self._raw_name(key)
        if raw is None:
            return
        self._pressed.add(raw)
        # Fire only when the combo becomes complete, not on every auto-repeat
        if self._combo_complete() and not self._combo_was_complete:
            self._combo_was_complete = True
            self._on_edge()

    def _on_release(self, key) -> None:
        raw = self._raw_name(key)
        if raw is not None:
            self._pressed.discard(raw)
        if not self._combo_complete():
            self._combo_was_complete = False

    def run(self) -> None:
        from pynput import keyboard as pk

        log.info(
            "Hotkey '%s' active, %s mode (pynput backend). Ctrl+C to stop.",
            self.combo_label,
            "session" if config.SESSION_MODE else config.HOTKEY_MODE,
        )
        with pk.Listener(on_press=self._on_press, on_release=self._on_release) as ln:
            ln.join()


def make_backend(on_activate):
    if config.HOTKEY_BACKEND == "keyboard":
        return KeyboardBackend(config.HOTKEY, on_activate)
    if config.HOTKEY_BACKEND != "pynput":
        log.warning(
            "HOTKEY_BACKEND '%s' not recognised, falling back to 'pynput'",
            config.HOTKEY_BACKEND,
        )
    return PynputBackend(config.HOTKEY, on_activate)


# --- Pipeline ---------------------------------------------------------------

# Memory-wipe phrases. Matched locally, not through the LLM: an irreversible
# command must not hinge on a model's guess.
def handle_utterance(is_recording: Callable[[], bool]) -> None:
    """One turn: record -> transcribe -> LLM -> speak."""
    try:
        _handle_utterance(is_recording)
    finally:
        # Whatever happened, the icon goes back to idle. A tray stuck on
        # "listening" after a failed turn would be a worse lie than no tray.
        tray.state("idle")


def _handle_utterance(is_recording: Callable[[], bool]) -> None:
    # 0. If the model has been released, start loading it NOW — in parallel
    #    with the user speaking. The time they spend talking (typically 3-5 s)
    #    is spent loading instead of idling. get_model() has its own lock, so
    #    the transcribe below waits politely if loading is not finished.
    if not stt.is_loaded():
        threading.Thread(target=stt.warmup, name="preload", daemon=True).start()

    # 1. Record
    tray.state("recording")
    try:
        # the beep plays from inside, once the mic is genuinely ready
        clip = audio.record_until_release(
            is_recording,
            on_ready=audio.beep_start,
            # Toggle mode stops explicitly, so no anti-flicker grace is needed
            release_grace=0.0 if config.HOTKEY_MODE == "toggle" else None,
        )
        audio.beep_stop()
    except Exception:
        log.exception("failed to record from the mic")
        audio.beep_error()
        return

    if clip.size == 0:
        log.info("no usable audio, skipping")
        return

    # 2. STT
    tray.state("thinking")
    # Loading can take 10-35 s from a cold disk cache. With no word at all,
    # that silence is indistinguishable from a dead agent — the user presses
    # again and gets more confused. TTS is already warm, so this speaks instantly.
    if not stt.is_loaded():
        log.info("model still loading, telling the user first")
        _say_safely("One moment, just getting ready.")

    try:
        text = stt.transcribe(clip)
    except Exception:
        log.exception("transcription failed")
        audio.beep_error()
        return

    if not text:
        log.info("empty transcript, skipping")
        audio.beep_error()
        return
    log.info("User: %s", text)
    _route_and_reply(text)


def _route_and_reply(text: str) -> None:
    """One entry point shared by press mode and session mode.

    It only forwards to the LLM now. It stays a separate function because this
    is where intent routing will attach when features come back — and the order
    of those checks has already proved easy to get wrong.
    """
    _speak_streaming(text)


def _speak_streaming(text: str) -> None:
    """Speak while answering: the first sentence plays while the model is still
    writing the next one.

    Measured: time to first sound drops 53-71% against waiting for the whole
    answer. What gets cut is not the total time but the silence — and that is
    what makes a conversation feel alive rather than laggy.
    """
    conv = llm.get_conversation()
    sp = audio.Speaker()
    spoke_any = False
    try:
        for sentence in conv.chat_stream(text):
            sp.add(tts.speak(sentence))
            if not spoke_any:
                # Only on the first sentence: the icon should flip when sound
                # actually starts, not when the model starts writing.
                tray.state("speaking")
            spoke_any = True
        sp.finish()
    except Exception:
        # Let whatever is already playing finish; never cut mid-word.
        try:
            sp.finish()
        except Exception:
            log.debug("failed to close playback", exc_info=True)
        log.exception("failed to answer (%s)", config.LLM_BACKEND)
        if not spoke_any:
            _say_safely("Sorry, my brain isn't reachable right now.")


def handle_session(is_active: Callable[[], bool]) -> None:
    """Session mode: press the hotkey once, then keep talking without pressing.

    Utterance boundaries come from the VAD, not the key. The session closes when
    you press the hotkey again OR go quiet for SESSION_IDLE_SECONDS.

    The mic is closed while the agent speaks — the deliberate consequence of
    having no barge-in. The upside: the agent can never hear itself, so no echo
    cancellation is needed and ordinary speakers are fine.
    """
    if not stt.is_loaded():
        threading.Thread(target=stt.warmup, name="muat-duluan", daemon=True).start()

    audio.beep_start()
    log.info("session opened (close: hotkey again, or %.0f s of silence)",
             config.SESSION_IDLE_SECONDS)
    turn = 0
    started_at = time.monotonic()

    try:
        while is_active():
            try:
                clip, reason = vad.record_utterance(should_stop=lambda: not is_active())
            except Exception:
                log.exception("failed to record from the mic")
                audio.beep_error()
                return

            if clip is None:
                log.info("session closed: %s", reason)
                break

            if not stt.is_loaded():
                log.info("model still loading, telling the user first")
                _say_safely("One moment, just getting ready.")

            try:
                text = stt.transcribe(clip)
            except Exception:
                log.exception("transcription failed")
                audio.beep_error()
                continue  # one failed utterance must not close the session

            if not text:
                log.info("empty transcript, still listening")
                continue

            turn += 1
            log.info("User (turn %d): %s", turn, text)

            if _wants_end_session(text):
                log.info("session closed by voice")
                _say_safely("Okay, talk to you later.")
                break

            try:
                _route_and_reply(text)
            except Exception:
                # One failed turn must not kill the session — the user can just
                # speak again. Only a mic that will not open is fatal.
                log.exception("turn %d failed", turn)
                audio.beep_error()
    finally:
        audio.beep_stop()
        log.info("session done: %d turns, %.0f s", turn,
                 time.monotonic() - started_at)


# Session-ending phrases. Matched locally, not through the LLM: a control
# command must not hinge on a model's guess.
#
# Two tiers, and the split matters.
#
# STRONG = cannot mean anything else, so it may match anywhere in the sentence.
_END_STRONG = (
    "goodbye", "good bye", "bye bye", "talk to you later", "see you later",
    "stop listening", "end session", "end the session",
)
# WEAK = only closes when it sits at the END of the utterance. "I'm done" ends
# the session, but "I'm done with assignment one" is a report about work — match
# it anywhere and that sentence could never reach anything else.
_END_WEAK = (
    "bye", "stop", "thanks", "thank you", "see you",
    "thats all", "thats it", "were done", "im done", "we are done", "i am done",
)


def _wants_end_session(text: str) -> bool:
    t = text_utils.normalize(text)
    if not t:
        return False
    if any(f in t for f in _END_STRONG):
        return True
    return any(t == f or t.endswith(" " + f) for f in _END_WEAK)


def _speak_sentences(text: str) -> None:
    """Speak sentence by sentence, starting as soon as the first one is ready.

    Same reason as the streaming path: the TTS cannot start until a sentence is
    complete, so synthesising the whole answer first makes the silence too long.
    """
    from .llm import _split_sentence

    sp = audio.Speaker()
    try:
        # The trailing space is required: _split_sentence() looks for
        # punctuation FOLLOWED by a space, so the final sentence would never be
        # detected without it.
        rest = text.rstrip() + " "
        spoke_any = False
        while True:
            sentence_out, rest = _split_sentence(rest)
            if sentence_out is None:
                break
            sp.add(tts.speak(sentence_out))
            spoke_any = True
        tail = rest.strip()
        if tail:
            sp.add(tts.speak(tail))
            spoke_any = True
        if not spoke_any:
            sp.add(tts.speak(text))
        sp.finish()
    except Exception:
        try:
            sp.finish()
        except Exception:
            log.debug("failed to close playback", exc_info=True)
        log.exception("failed to speak")
        audio.beep_error()


def _say_safely(text: str) -> None:
    """Speak without raising a new error (used to report failures)."""
    try:
        audio.play_wav(tts.speak(text))
    except Exception:
        log.exception("failed to speak the error message")
        audio.beep_error()


def warmup() -> None:
    """Load models in the background so the first press is not slow.

    TTS always loads: CPU-only, 0 VRAM, only ~1.5-2.7 s. STT follows
    WHISPER_WARMUP — if the model is going to be released when idle, loading it
    up front just holds memory until the idle threshold passes.

    The VAD loads only in session mode, and always: 0 VRAM, ~1 s, but if it
    loaded on the first key press your first word would be clipped.
    """

    def _run():
        try:
            tts.get_voice()
            if config.SESSION_MODE:
                vad._model()
            if config.WHISPER_WARMUP:
                stt.warmup()
            else:
                log.info(
                    "STT (%s) not warmed up; loads on the first question",
                    config.STT_BACKEND,
                )
            log.info("Warmup done, ready.")
        except Exception:
            log.exception("warmup failed (models will load on demand)")

    threading.Thread(target=_run, name="warmup", daemon=True).start()


def main() -> int:
    setup_logging()
    log.info("=" * 60)
    log.info("build: %s", _build_version())
    otak = (
        config.OLLAMA_MODEL if config.LLM_BACKEND == "ollama" else config.CLAUDE_MODEL
    )
    log.info(
        "personal-agent start | lang=%s offline=%s | hotkey=%s (%s, %s) | "
        "stt=%s/%s | llm=%s/%s | tts=%s",
        config.LANGUAGE,
        config.OFFLINE_MODE,
        config.HOTKEY,
        "SESSION" if config.SESSION_MODE else config.HOTKEY_MODE,
        config.HOTKEY_BACKEND,
        config.STT_BACKEND,
        config.STT_DEVICE,
        config.LLM_BACKEND,
        otak,
        config.TTS_BACKEND,
    )

    # Before anything else: make sure no other agent is running. Checked FIRST
    # because a second agent that has already loaded models would waste memory
    # before giving up.
    other_pid = claim_single_instance()
    if other_pid is not None:
        log.error(
            "Another agent is already running (PID %s). This one is stopping — "
            "two agents would fight over the '%s' hotkey. "
            "Check: powershell -File scripts\\status.ps1",
            other_pid or "?",
            config.HOTKEY,
        )
        return 3

    # Settings that clash with offline mode must surface in the startup log,
    # not once the user has already asked something.
    try:
        config.require_offline()
    except config.OfflineViolation:
        log.exception("settings clash with OFFLINE_MODE, agent not starting")
        return 2

    # Before warmup, not after: warmup can take 35 s from a cold disk cache,
    # and "is it loading or is it dead?" is exactly the question the icon
    # exists to answer.
    tray.start()
    tray.state("starting")
    warmup()
    tray.state("idle")

    # Session mode changes the whole interaction style, not just a setting:
    # the hotkey opens a session instead of being a record button.
    backend = make_backend(handle_session if config.SESSION_MODE else handle_utterance)

    try:
        backend.run()
    except KeyboardInterrupt:
        log.info("stopped with Ctrl+C")
    except ImportError:
        log.exception("hotkey backend '%s' could not be imported", backend.name)
        return 1
    except Exception:
        log.exception("hotkey listener died")
        return 1
    finally:
        tray.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

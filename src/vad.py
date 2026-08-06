"""Voice activity detection: knowing when you start and stop speaking.

Uses Silero VAD through onnx_asr — the same package already pulling Parakeet, so
this adds **zero new pip packages**. Same reasoning as choosing Parakeet: taking
on a heavy dependency for one small job is expensive on a machine that has to
run offline.

`onnx_asr` ships its own VAD, but it segments a finished recording in one batch.
What session mode needs is different: a decision **per frame, as audio arrives**,
because it has to know your sentence ended without you pressing anything. So the
ONNX session is driven directly, frame by frame.

One frame = 512 samples @ 16 kHz = 32 ms.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from . import config

log = logging.getLogger(__name__)

# Silero v5 at 16 kHz. These come from the model, they are not free choices.
HOP = 512
CONTEXT = 64
FRAME_MS = HOP * 1000 // 16000  # 32

_session = None
_lock = threading.Lock()


def _model():
    """Load once. Kept separate from stt.get_model() because their lifetimes
    differ: the VAD is held for the whole session, while the STT model may be
    released between turns."""
    global _session
    with _lock:
        if _session is None:
            import onnx_asr

            t0 = time.perf_counter()
            _session = onnx_asr.load_vad("silero")._model
            log.info("Silero VAD ready (%.1f s)", time.perf_counter() - t0)
        return _session


def is_loaded() -> bool:
    return _session is not None


def unload() -> bool:
    global _session
    with _lock:
        was_loaded, _session = _session is not None, None
    return was_loaded


class Detector:
    """Speech probability per frame, carrying state between frames.

    The state matters: Silero is recurrent. The same frame can score differently
    depending on what was just heard, and that is exactly what keeps it from
    being fooled by a fan or a keyboard tap.
    """

    def __init__(self) -> None:
        self._m = _model()
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._tail = np.zeros(CONTEXT, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        """`frame` is HOP float32 samples. Returns a probability in 0..1."""
        if len(frame) < HOP:
            frame = np.pad(frame, (0, HOP - len(frame)))
        window = np.concatenate([self._tail, frame[:HOP]])[None, :].astype(np.float32)
        out, new_state = self._m.run(
            ["output", "stateN"],
            {"input": window, "state": self._state, "sr": np.array(16000, dtype=np.int64)},
        )
        self._state = new_state
        self._tail = frame[HOP - CONTEXT : HOP].copy()
        return float(out[0, 0])


def record_utterance(
    should_stop: Callable[[], bool],
    silence_limit: float | None = None,
    on_speech_start: Callable[[], None] | None = None,
) -> tuple[np.ndarray | None, str]:
    """Wait until you speak, record, and stop when you go quiet.

    Returns `(audio, reason)`:
      - `(array, "ok")`      : an utterance was captured
      - `(None, "stopped")`  : `should_stop()` became True
      - `(None, "silence")`  : nothing was said before `silence_limit` elapsed

    The reasons are kept separate on purpose: the caller must be able to tell
    "the user closed the session" from "the session timed out". A sound that is
    too short (a cough, a tap on the desk) is NOT a reason of its own — it is
    swallowed in here, because a cough must not close a conversation.

    `on_speech_start` fires once when speech is detected — for cancelling a
    session timer, not for playing anything.
    """
    if silence_limit is None:
        silence_limit = config.SESSION_IDLE_SECONDS

    det = Detector()
    frames: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, _n, _t, status):
        if status:
            log.debug("input stream status: %s", status)
        frames.put(indata[:, 0].copy())

    silence_needed = int(config.VAD_SILENCE_MS / FRAME_MS)
    speech_needed = int(config.VAD_MIN_SPEECH_MS / FRAME_MS)
    # Keep a little audio from BEFORE speech is detected. The VAD is always
    # slightly late to recognise a word onset, and without this padding the
    # first consonant is clipped off.
    pad = max(1, int(config.VAD_SPEECH_PAD_MS / FRAME_MS))

    before: list[np.ndarray] = []
    body: list[np.ndarray] = []
    speaking = False
    n_speech = 0
    n_silence = 0
    # The silence deadline is measured from entry and deliberately NOT reset by
    # short sounds. If it were, a noisy room could hold the session open forever
    # without you saying a word.
    deadline = time.monotonic() + silence_limit

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="float32",
        blocksize=HOP,
        callback=callback,
    ):
        while True:
            if should_stop():
                return None, "stopped"

            try:
                block = frames.get(timeout=0.1)
            except queue.Empty:
                if not speaking and time.monotonic() >= deadline:
                    return None, "silence"
                continue

            p = det.prob(block)

            if not speaking:
                before.append(block)
                if len(before) > pad:
                    before.pop(0)

                if p >= config.VAD_THRESHOLD:
                    n_speech += 1
                    if n_speech >= speech_needed:
                        speaking = True
                        body = list(before)
                        before.clear()
                        if on_speech_start:
                            on_speech_start()
                else:
                    n_speech = 0
                    if time.monotonic() >= deadline:
                        return None, "silence"
                continue

            body.append(block)
            # The release threshold is deliberately lower than the capture
            # threshold. With a single threshold, a natural mid-sentence pause
            # reads as the end and your sentence gets cut in two.
            if p < config.VAD_THRESHOLD - 0.15:
                n_silence += 1
                if n_silence < silence_needed:
                    continue
            else:
                n_silence = 0
                if len(body) * FRAME_MS / 1000 <= config.MAX_RECORD_SECONDS:
                    continue
                log.warning("utterance cut off at %.0f s", config.MAX_RECORD_SECONDS)

            audio = np.concatenate(body)
            seconds = len(audio) / config.SAMPLE_RATE
            if seconds >= config.MIN_RECORD_SECONDS:
                return audio, "ok"

            # Too short — a cough, a tap, a click. Go back to waiting; do NOT
            # close the session.
            log.debug("sound of %.2f s too short, still waiting", seconds)
            speaking = False
            n_speech = n_silence = 0
            body = []
            before.clear()
            det.reset()


if __name__ == "__main__":
    # Manual check: speak, and see whether the sentence boundaries land right.
    #
    #     .venv-agent\Scripts\python.exe -m agent.vad
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from . import audio as audio_io

    print("Say something. 30 seconds of silence ends it. Ctrl+C also works.")
    n = 0
    while True:
        t0 = time.monotonic()
        clip, reason = record_utterance(lambda: False)
        if clip is None:
            print(f"({reason} — done)")
            break
        n += 1
        print(f"  utterance {n}: {len(clip)/config.SAMPLE_RATE:.2f} s "
              f"(wait + record {time.monotonic()-t0:.1f} s)")
        audio_io.beep_stop()

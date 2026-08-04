"""Microphone capture and speaker playback."""

from __future__ import annotations

import io
import logging
import queue
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config

log = logging.getLogger(__name__)


# --- Playback ---------------------------------------------------------------


def _pad(data: np.ndarray, samplerate: int) -> np.ndarray:
    """Insert silence at both ends so the audio does not get clipped.

    Without it, Windows loses the start (the output device is still opening
    when the first samples arrive) and cuts the end (sd.wait() returns once the
    buffer has been fed, while the device is still playing the rest).
    """
    n = int(samplerate * config.PLAYBACK_PAD_SECONDS)
    if n <= 0:
        return data
    hening = np.zeros((n,) + data.shape[1:], dtype=data.dtype)
    return np.concatenate([hening, data, hening])


def _play_array(data: np.ndarray, samplerate: int, blocking: bool = True) -> None:
    sd.play(_pad(data, samplerate), samplerate)
    if blocking:
        sd.wait()


def play_wav(wav_bytes: bytes, blocking: bool = True) -> None:
    """Play a WAV (bytes) on the default output device."""
    data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    _play_array(data, samplerate, blocking)


def stop_playback() -> None:
    sd.stop()


class Speaker:
    """Play WAV chunks back to back with NO gap at the joins.

    Why not call play_wav() repeatedly: `_pad()` adds silence to both ends of
    every chunk, so each sentence boundary costs 0.4 s of silence and a
    four-sentence reply sounds broken up. Here the padding goes in once at the
    start and once at the end of the whole utterance.

    Used like this:
        sp = Speaker()
        for wav in chunks:
            sp.add(wav)      # the first one starts playing immediately
        sp.finish()          # waits until playback is really done
    """

    def __init__(self) -> None:
        self._stream: sd.OutputStream | None = None
        self._rate: int | None = None

    def add(self, wav_bytes: bytes) -> None:
        data, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]

        if self._stream is None:
            self._rate = rate
            self._stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
            self._stream.start()
            n = int(rate * config.PLAYBACK_PAD_SECONDS)
            if n > 0:
                self._stream.write(np.zeros(n, dtype=np.float32))
        elif rate != self._rate:
            # Cannot happen while one TTS backend serves a whole utterance,
            # but if it did, silently changing the rate would turn the voice
            # into a chipmunk — better to make the mistake visible.
            log.warning("sample rate changed mid-utterance: %s -> %s", self._rate, rate)

        self._stream.write(np.ascontiguousarray(data, dtype=np.float32))

    def finish(self) -> None:
        if self._stream is None:
            return
        n = int((self._rate or 0) * config.PLAYBACK_PAD_SECONDS)
        if n > 0:
            self._stream.write(np.zeros(n, dtype=np.float32))
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None


# --- Beep feedback ----------------------------------------------------------


def _tone(freq: float, duration: float, volume: float = 0.25) -> np.ndarray:
    """A short sine with fade in/out so it does not click."""
    sr = config.SAMPLE_RATE
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    wave = np.sin(2 * np.pi * freq * t, dtype=np.float32) * volume

    fade = max(1, int(sr * 0.01))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    wave[:fade] *= ramp
    wave[-fade:] *= ramp[::-1]
    return wave


def _beep(freq: float, duration: float = 0.09, blocking: bool = True) -> None:
    try:
        # The beep is very short, which makes it the easiest thing to lose
        # without padding
        _play_array(_tone(freq, duration), config.SAMPLE_RATE, blocking)
    except Exception:
        # A beep is only feedback; it must never kill the pipeline
        log.warning("failed to play beep", exc_info=True)


def beep_start() -> None:
    """Rising tone: recording started.

    Deliberately NON-BLOCKING. If we waited for the beep to finish, the mic
    buffer would only be flushed afterwards — but the user starts speaking the
    moment they hear the tone, so their first word would be flushed with it.
    The 880 Hz tone bleeding into the mic is harmless: the VAD does not read it
    as speech.
    """
    _beep(880.0, blocking=False)


def beep_stop() -> None:
    """Falling tone: recording finished."""
    _beep(560.0)


def beep_error() -> None:
    """Long low tone: something failed."""
    _beep(240.0, duration=0.25)


def beep_busy() -> None:
    """Two short taps: heard you, but busy — your press was ignored.

    Distinct from beep_error so "still working" is never mistaken for "failed".
    """
    _beep(420.0, duration=0.05)
    _beep(420.0, duration=0.05)


# --- Recording --------------------------------------------------------------


def record_until_release(
    is_held: Callable[[], bool],
    on_ready: Callable[[], None] | None = None,
    release_grace: float | None = None,
    poll_interval: float = 0.02,
) -> np.ndarray:
    """Record from the mic for as long as `is_held()` stays True.

    `release_grace` is how long to wait after `is_held()` goes False before
    actually stopping. Needed in hold mode (Windows sends spurious UP/DOWN
    pairs); in toggle mode the stop is explicit, so it is skipped (0).

    `on_ready` fires once the stream is genuinely open — used to play the beep,
    so the user does not start talking before the mic is live (opening a stream
    can take half a second).

    Returns 1-D mono float32 at `config.SAMPLE_RATE`. An empty array means the
    recording was too short to be anything but a stray press.
    """
    if release_grace is None:
        release_grace = config.RELEASE_GRACE_SECONDS

    frames: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, _frames, _time_info, status):
        if status:
            log.debug("input stream status: %s", status)
        frames.put(indata.copy())

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="float32",
        callback=callback,
    ):
        if on_ready is not None:
            on_ready()
        # Drop whatever was captured before and during the beep
        while not frames.empty():
            frames.get()

        started = time.monotonic()
        released_at: float | None = None
        while True:
            now = time.monotonic()

            if is_held():
                # Combo complete again — only a flicker, keep recording
                released_at = None
            elif released_at is None:
                released_at = now
            elif now - released_at >= release_grace:
                break

            if now - started > config.MAX_RECORD_SECONDS:
                log.warning(
                    "recording cut off at %.0f s (safety limit)",
                    config.MAX_RECORD_SECONDS,
                )
                break
            time.sleep(poll_interval)

    chunks = []
    while not frames.empty():
        chunks.append(frames.get())

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).reshape(-1)
    seconds = len(audio) / config.SAMPLE_RATE
    if seconds < config.MIN_RECORD_SECONDS:
        log.info("recording was only %.2f s, dropped", seconds)
        return np.zeros(0, dtype=np.float32)

    log.info("recorded %.2f s (%d samples)", seconds, len(audio))
    if config.SAVE_RECORDINGS:
        _save_recording(audio)
    return audio


def _save_recording(audio: np.ndarray) -> None:
    """Save the raw recording for debugging. Must never fail the pipeline."""
    try:
        config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + ".wav"
        path = config.RECORDINGS_DIR / name
        sf.write(path, audio, config.SAMPLE_RATE)
        log.info("recording saved: %s", path)
    except Exception:
        log.warning("failed to save recording", exc_info=True)


def list_devices() -> str:
    return str(sd.query_devices())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(list_devices())
    print("\nBeep start/stop:")
    beep_start()
    time.sleep(0.3)
    beep_stop()

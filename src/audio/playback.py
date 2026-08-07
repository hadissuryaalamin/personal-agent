"""Speaker output, interruptible.

Two things matter here beyond "make a sound":

**Barge-in.** Playback runs on a background thread and can be stopped mid-word,
because the session keeps listening while the agent talks. Interrupting has to
be immediate -- an assistant that finishes its sentence after you have started
talking over it is worse than one that never spoke.

**Saying which device it is using.** An agent that has gone quiet with a clean
log is almost always playing into a device that is no longer there -- a headset
that went to sleep, or a default that moved when something was plugged in. The
code cannot fix that, but it can stop it looking like a bug in the code, so the
session prints the output device by name at startup and `--devices` lists both
directions.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

SAMPLE_RATE = 24000  # Kokoro's native rate


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    default: bool


def list_output_devices() -> list[DeviceInfo]:
    import sounddevice as sd

    default_index = None
    try:
        default_index = sd.default.device[1]
    except Exception:  # noqa: BLE001 - nothing configured
        pass

    out = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_output_channels"] > 0:
            out.append(
                DeviceInfo(
                    index=index,
                    name=device["name"],
                    channels=int(device["max_output_channels"]),
                    default=index == default_index,
                )
            )
    return out


def describe_output(device=None) -> str:
    """What the speaker actually is, for the startup line."""
    try:
        import sounddevice as sd

        info = sd.query_devices(device, "output")
        return f"{info['name']} @ {int(info['default_samplerate'])} Hz"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"


class Speaker:
    """A queue of audio chunks, played in order, stoppable at any moment."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device=None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._playing = threading.Event()

    @property
    def playing(self) -> bool:
        return self._playing.is_set()

    def start(self) -> "Speaker":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def say(self, samples) -> None:
        """Queue one chunk -- typically one sentence."""
        self.start()
        self._queue.put(samples)

    def interrupt(self) -> int:
        """Stop now and drop anything queued. Returns how many chunks were dropped."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        self._stop.set()
        return dropped

    def wait(self, timeout: float | None = None) -> None:
        """Block until the queue drains. Used by the offline runner."""
        self._queue.join()
        if timeout:
            self._playing.wait(timeout=0)

    def close(self) -> None:
        self.interrupt()
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        import sounddevice as sd

        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                device=self.device,
            )
            stream.start()
            while True:
                chunk = self._queue.get()
                try:
                    if chunk is None:
                        return
                    self._stop.clear()
                    self._playing.set()
                    self._write_interruptibly(stream, chunk)
                finally:
                    self._playing.clear()
                    self._queue.task_done()
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

    def _write_interruptibly(self, stream, chunk, block: int = 2048) -> None:
        """Write in small blocks so interrupt() lands within ~85 ms."""
        import numpy as np

        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        for start in range(0, len(samples), block):
            if self._stop.is_set():
                return
            stream.write(samples[start : start + block])


class NullSpeaker:
    """Collects audio instead of playing it. For tests and --wav runs."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.chunks: list[Any] = []
        self.interrupted = 0

    @property
    def playing(self) -> bool:
        return False

    def start(self) -> "NullSpeaker":
        return self

    def say(self, samples) -> None:
        self.chunks.append(samples)

    def interrupt(self) -> int:
        self.interrupted += 1
        return 0

    def wait(self, timeout: float | None = None) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def seconds(self) -> float:
        return sum(len(c) for c in self.chunks) / self.sample_rate

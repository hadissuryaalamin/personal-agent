"""Microphone capture.

A thin wrapper over sounddevice that hands out float32 blocks at 16 kHz, which
is what both Silero and Parakeet want. The callback pushes into a queue and
returns immediately -- doing any work in a PortAudio callback drops samples,
and dropped samples are indistinguishable from the user pausing.

Opening a device is deliberately not done at import: the tests, `src.cli` and
`scripts/` all run on machines with no microphone.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any, Iterator

SAMPLE_RATE = 16000
#: 32 ms. Silero's window is 512 samples (32 ms at 16 kHz), so this is one
#: window per block and the VAD never has to buffer a partial one.
BLOCK_SAMPLES = 512


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    default: bool


def list_input_devices() -> list[DeviceInfo]:
    """Every input device, for when the agent goes quiet and nobody knows why."""
    import sounddevice as sd

    default_index = None
    try:
        default_index = sd.default.device[0]
    except Exception:  # noqa: BLE001 - no device configured at all
        pass

    out = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            out.append(
                DeviceInfo(
                    index=index,
                    name=device["name"],
                    channels=int(device["max_input_channels"]),
                    default=index == default_index,
                )
            )
    return out


class Microphone:
    """Context manager yielding float32 blocks.

        with Microphone() as mic:
            for block in mic.blocks():
                ...
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_samples: int = BLOCK_SAMPLES,
        device: int | str | None = None,
        max_queued_blocks: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_samples = block_samples
        self.device = device
        self._queue: queue.Queue = queue.Queue(maxsize=max_queued_blocks)
        self._stream: Any = None
        #: Counts blocks PortAudio reported as overflowed. A non-zero value
        #: here explains transcripts with holes in them.
        self.dropped = 0

    def __enter__(self) -> "Microphone":
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                self.dropped += 1
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                self.dropped += 1

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_samples,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def blocks(self, timeout: float = 1.0) -> Iterator:
        """Yield captured blocks until the stream is closed."""
        while self._stream is not None:
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                continue


def read_wav(path) -> tuple[Any, int]:
    """Load a mono float32 waveform. Used by tests and the offline runner."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def write_wav(path, samples, sample_rate: int = SAMPLE_RATE) -> None:
    import wave

    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

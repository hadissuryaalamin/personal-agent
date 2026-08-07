"""Parakeet TDT 0.6B v2, through sherpa-onnx.

English only (D4). Runs on the CPU: the GPU is holding Qwen, and an int8
transducer on a handful of threads comfortably beats the 300 ms budget in
PLAN.md section 5 for utterances of the length people actually speak.

The transcript this produces is the *only* thing downstream sees -- invariant
#3 exists because a system whose input is sound cannot be debugged any other
way. Every transcript is logged, including the empty ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.config import ROOT

DEFAULT_MODEL_DIR = ROOT / "models" / "parakeet-tdt-0.6b-v2-int8"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Transcript:
    text: str
    ms: int
    seconds_of_audio: float
    #: sherpa-onnx does not hand back a scalar confidence for a transducer, so
    #: this stays None rather than being invented. turn_log.asr_conf keeps the
    #: column for a model that does provide one.
    confidence: float | None = None

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    @property
    def real_time_factor(self) -> float:
        return (self.ms / 1000) / self.seconds_of_audio if self.seconds_of_audio else 0.0


class Parakeet:
    def __init__(self, model_dir: Path | str | None = None, num_threads: int = 4) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.num_threads = num_threads
        self._recognizer = None

    def _file(self, *names: str) -> str:
        for name in names:
            candidate = self.model_dir / name
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"{self.model_dir} has none of {names}. Run:\n"
            "    python scripts\\fetch_models.py --only asr"
        )

    def load(self) -> "Parakeet":
        if self._recognizer is not None:
            return self

        import sherpa_onnx

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"No ASR model at {self.model_dir}. Run:\n"
                "    python scripts\\fetch_models.py --only asr"
            )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self._file("encoder.int8.onnx", "encoder.onnx"),
            decoder=self._file("decoder.int8.onnx", "decoder.onnx"),
            joiner=self._file("joiner.int8.onnx", "joiner.onnx"),
            tokens=self._file("tokens.txt"),
            num_threads=self.num_threads,
            model_type="nemo_transducer",
        )
        return self

    def transcribe(self, samples, sample_rate: int = SAMPLE_RATE) -> Transcript:
        """One utterance in, one transcript out. Never raises on silence."""
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        seconds = len(audio) / sample_rate
        if seconds == 0:
            # Checked before load(): there is no reason to bring up a 600 MB
            # recogniser to transcribe nothing, and the empty case is the one
            # most likely to be hit on a machine that has no model yet.
            return Transcript(text="", ms=0, seconds_of_audio=0.0)

        self.load()
        started = time.perf_counter()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        elapsed = int((time.perf_counter() - started) * 1000)

        return Transcript(
            text=(stream.result.text or "").strip(),
            ms=elapsed,
            seconds_of_audio=seconds,
        )

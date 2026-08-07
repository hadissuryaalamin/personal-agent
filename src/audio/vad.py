"""Turn segmentation with Silero VAD.

A session is a stream of samples; a *turn* is the speech between two silences.
This wraps sherpa-onnx's ``VoiceActivityDetector`` so the session loop deals in
whole utterances rather than 32 ms blocks.

PLAN.md section 5 budgets 200 ms for endpoint detection. That budget is the
`min_silence` below: it is how long the speaker has to stop before the turn is
considered finished, and it trades directly against cutting people off
mid-sentence. Nothing else here is on the latency path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import ROOT

SAMPLE_RATE = 16000

DEFAULT_MODEL = ROOT / "models" / "silero-vad" / "silero_vad.onnx"

#: How confident Silero must be that a window contains speech.
THRESHOLD = 0.5
#: How long the speaker has to stop before the turn is treated as finished.
#:
#: PLAN.md section 5 budgets 200 ms for this. Measured against real sentences,
#: 200 ms is not achievable and 0.35 s was still too short: "add the data
#: structures assignment due next friday, about six hours of work" got cut in
#: half at the comma, and the orphaned second half was answered as if it were
#: a new command. People pause mid-sentence and the VAD cannot tell that pause
#: from the end of a turn.
#:
#: 0.6 s costs 400 ms against the budget and buys back whole sentences. That
#: is the right trade: being cut off mid-command is worse than a beat of
#: silence before the reply.
MIN_SILENCE = 0.6
#: Shorter than this is a cough, a door, or a keyboard.
MIN_SPEECH = 0.25
#: A hard ceiling so a stuck stream cannot buffer forever.
MAX_SPEECH = 20.0


@dataclass(frozen=True)
class Segment:
    """One detected utterance."""

    samples: Any
    start_sample: int
    sample_rate: int = SAMPLE_RATE

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate


class Segmenter:
    """Feed it samples, get whole utterances back."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = THRESHOLD,
        min_silence: float = MIN_SILENCE,
        min_speech: float = MIN_SPEECH,
        max_speech: float = MAX_SPEECH,
        buffer_seconds: float = 60.0,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence = min_silence
        self.min_speech = min_speech
        self.max_speech = max_speech
        self.buffer_seconds = buffer_seconds
        self._vad = None

    def load(self) -> "Segmenter":
        if self._vad is not None:
            return self

        import sherpa_onnx

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No VAD model at {self.model_path}. Run:\n"
                "    python scripts\\fetch_models.py --only vad"
            )

        silero = sherpa_onnx.SileroVadModelConfig(
            model=str(self.model_path),
            threshold=self.threshold,
            min_silence_duration=self.min_silence,
            min_speech_duration=self.min_speech,
            max_speech_duration=self.max_speech,
        )
        config = sherpa_onnx.VadModelConfig(
            silero_vad=silero, sample_rate=self.sample_rate, num_threads=1
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=self.buffer_seconds
        )
        return self

    @property
    def speaking(self) -> bool:
        """True while the speaker is mid-utterance. Used for barge-in at M5."""
        return bool(self._vad and self._vad.is_speech_detected())

    def push(self, samples) -> list[Segment]:
        """Accept a block of float32 samples; return any completed turns."""
        import numpy as np

        self.load()
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._vad.accept_waveform(block)
        return self._drain()

    def flush(self) -> list[Segment]:
        """End of session: emit whatever speech is still buffered."""
        self.load()
        self._vad.flush()
        return self._drain()

    def reset(self) -> None:
        if self._vad is not None:
            self._vad.reset()

    def _drain(self) -> list[Segment]:
        import numpy as np

        out: list[Segment] = []
        while not self._vad.empty():
            segment = self._vad.front
            out.append(
                Segment(
                    samples=np.asarray(segment.samples, dtype=np.float32),
                    start_sample=int(segment.start),
                    sample_rate=self.sample_rate,
                )
            )
            self._vad.pop()
        return out

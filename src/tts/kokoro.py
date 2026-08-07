"""Kokoro 82M, streamed a sentence at a time.

PLAN.md section 5 budgets 250 ms to the first audio chunk, and the way to hit
that is not a faster model -- it is to stop waiting for the whole reply. The
speaker starts on sentence one while sentence two is still being synthesised,
so what the user waits for is the first sentence, not the last.

That is also why `split_sentences` is careful about abbreviations: a reply cut
at "COMP4020 at 9 a." would be spoken as two fragments, and the first one would
start playing before anything noticed.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.config import ROOT

DEFAULT_MODEL = ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"
DEFAULT_VOICES = ROOT / "models" / "kokoro" / "voices-v1.0.bin"

#: D4: English in and out. Australian English, since the user is at ANU.
DEFAULT_VOICE = "bf_emma"
DEFAULT_LANG = "en-gb"
DEFAULT_SPEED = 1.05

SAMPLE_RATE = 24000

#: Things that end in a full stop but are not the end of a sentence.
_ABBREVIATIONS = (
    "a.m", "p.m", "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "st",
    "no", "approx", "dept", "prof",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])[\s]+")


@dataclass(frozen=True)
class Chunk:
    """One synthesised sentence."""

    text: str
    samples: object
    sample_rate: int
    ms: int
    #: True for the first chunk of a reply -- the one the budget is about.
    first: bool = False

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """Split a reply into speakable pieces.

    Not a general sentence splitter: it only has to be right about the strings
    src/format.py produces, which contain times ("9 a.m."), ordinals and dashes.
    """
    if not text or not text.strip():
        return []

    pieces: list[str] = []
    for candidate in _SENTENCE_END.split(text.strip()):
        if not candidate:
            continue
        if pieces:
            tail = pieces[-1].rstrip().rstrip(".").lower()
            last_word = tail.split()[-1] if tail.split() else ""
            if last_word in _ABBREVIATIONS:
                pieces[-1] = f"{pieces[-1]} {candidate}"
                continue
        pieces.append(candidate)

    # A very long sentence still has to start playing promptly, so break it at
    # a clause boundary rather than making the user wait for the whole thing.
    out: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(" — ", 0, max_chars)
            if cut == -1:
                cut = piece.rfind(", ", 0, max_chars)
            if cut == -1:
                cut = piece.rfind(" ", 0, max_chars)
            if cut == -1:
                break
            out.append(piece[:cut].strip())
            piece = piece[cut:].lstrip(" —,")
        if piece.strip():
            out.append(piece.strip())
    return out


class Kokoro:
    def __init__(
        self,
        model_path: Path | str | None = None,
        voices_path: Path | str | None = None,
        voice: str = DEFAULT_VOICE,
        lang: str = DEFAULT_LANG,
        speed: float = DEFAULT_SPEED,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL)
        self.voices_path = Path(voices_path or DEFAULT_VOICES)
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self._engine = None

    def load(self) -> "Kokoro":
        if self._engine is not None:
            return self

        from kokoro_onnx import Kokoro as _Kokoro

        for path in (self.model_path, self.voices_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"No TTS weights at {path}. Run:\n"
                    "    python scripts\\fetch_models.py --restore-kokoro"
                )

        self._engine = _Kokoro(str(self.model_path), str(self.voices_path))
        if self.voice not in self.voices():
            raise ValueError(
                f"No voice {self.voice!r}. Available: {', '.join(sorted(self.voices())[:8])}…"
            )
        return self

    def voices(self) -> list[str]:
        self.load()
        return list(self._engine.get_voices())

    def synthesise(self, text: str) -> Chunk:
        """One sentence in, one waveform out."""
        self.load()
        started = time.perf_counter()
        samples, rate = self._engine.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        return Chunk(
            text=text,
            samples=samples,
            sample_rate=int(rate),
            ms=int((time.perf_counter() - started) * 1000),
        )

    def stream(self, text: str) -> Iterator[Chunk]:
        """Yield one chunk per sentence, in order.

        The caller plays each chunk as it arrives, so the wait before the first
        sound is one sentence of synthesis rather than the whole reply.
        """
        for index, sentence in enumerate(split_sentences(text)):
            chunk = self.synthesise(sentence)
            yield Chunk(
                text=chunk.text, samples=chunk.samples, sample_rate=chunk.sample_rate,
                ms=chunk.ms, first=index == 0,
            )

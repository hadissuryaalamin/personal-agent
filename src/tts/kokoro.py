"""Kokoro 82M, streamed a sentence at a time.

PLAN.md section 5 budgets 250 ms to the first audio chunk. Two things buy it,
and M5 showed that only having one of them is not enough:

**Start speaking before the reply is finished.** The speaker starts on sentence
one while sentence two is still being synthesised, so what the user waits for
is the first sentence, not the last. M5 measured what this cannot do: the
response rules cap replies at two sentences and most are one, so there is
usually no second sentence to overlap with.

**So the first piece has to be small.** `FIRST_MAX_CHARS` breaks the opening
sentence at a clause boundary even when the sentence would fit, because time
to first audio is proportional to the length of that first piece and nothing
else. The remainder is synthesised while it plays. This costs nothing when the
reply is already short -- "Added, due Friday the fourteenth." is one piece
either way.

That is also why `split_sentences` is careful about abbreviations: a reply cut
at "COMP4020 at 9 a." would be spoken as two fragments, and the first one would
start playing before anything noticed.

**The provider is chosen, not assumed.** Kokoro is 82M parameters and runs on
the CPU by default, at roughly half real time -- a two-second reply costs a
second of silence first. `provider` moves it onto the GPU when the installed
onnxruntime has one, and falls back to the CPU when it does not, because an
agent that will not talk is worse than one that talks late.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src import config
from src.config import ROOT

DEFAULT_MODEL = ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"
DEFAULT_VOICES = ROOT / "models" / "kokoro" / "voices-v1.0.bin"

#: D4: English in and out. Australian English, since the user is at ANU.
DEFAULT_VOICE = "bf_emma"
DEFAULT_LANG = "en-gb"
DEFAULT_SPEED = 1.05

SAMPLE_RATE = 24000

#: How long the *first* spoken piece may be. Time to first audio tracks the
#: length of this piece and nothing else, so it is the one number the budget
#: is actually about.
#:
#: 45 was chosen against measurement, not taste: scripts/bench_tts.py shows
#: synthesis runs at about 3.3x real time, so a piece of this size is roughly
#: 250 ms of work -- the PLAN.md section 5 budget. Every reply src/format.py
#: produces that is shorter than this is still spoken as one piece, which is
#: most of them.
FIRST_MAX_CHARS = 45

#: Never cut the opening piece shorter than this. "Added" on its own, followed
#: by a pause, sounds like the agent lost its train of thought -- the point is
#: to start speaking sooner, not to sound broken.
FIRST_MIN_CHARS = 15

#: Everything after the first piece. Larger, because it is synthesised while
#: the previous piece is playing and only has to keep ahead of the speaker.
MAX_CHARS = 180

#: Providers worth trying, best first. Kokoro is small enough that CUDA is a
#: large win; DirectML is the fallback on a Windows box without the CUDA build.
_PREFERRED = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")

#: Things that end in a full stop but are not the end of a sentence.
_ABBREVIATIONS = (
    "a.m", "p.m", "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "st",
    "no", "approx", "dept", "prof",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])[\s]+")


@contextlib.contextmanager
def _env(name: str, value: str):
    """Set an environment variable for the duration of a block, then restore."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


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


def _break_at(piece: str, max_chars: int) -> list[str]:
    """Break one sentence at clause boundaries so no part exceeds ``max_chars``.

    Prefers a dash, then a comma, then any space -- the places a speaker would
    pause anyway. A piece with no break point at all is left long rather than
    cut mid-word.
    """
    out: list[str] = []
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


def _split_opening(
    piece: str,
    max_chars: int = FIRST_MAX_CHARS,
    min_chars: int = FIRST_MIN_CHARS,
) -> list[str]:
    """Take a short opening piece off the front, at a natural pause.

    Only breaks where a speaker would draw breath anyway -- a dash or a comma,
    never a bare space. A sentence with no such pause in reach is left whole:
    a fragment cut mid-clause is worse than waiting a moment longer, and this
    runs on every reply the agent speaks.

    Takes the *earliest* pause past ``min_chars`` rather than the latest, since
    the shorter the opening piece the sooner the speaker starts. The rest is
    synthesised while it plays.
    """
    if len(piece) <= max_chars:
        return [piece]

    best: int | None = None
    for marker in (" — ", ", "):
        cut = piece.find(marker, min_chars)
        if cut != -1 and cut <= max_chars and (best is None or cut < best):
            best = cut
    if best is None:
        return [piece]

    head = piece[:best].strip()
    tail = piece[best:].lstrip(" —,").strip()
    return [head, tail] if head and tail else [piece]


def split_sentences(
    text: str,
    max_chars: int = MAX_CHARS,
    first_max_chars: int | None = None,
) -> list[str]:
    """Split a reply into speakable pieces.

    Not a general sentence splitter: it only has to be right about the strings
    src/format.py produces, which contain times ("9 a.m."), ordinals and dashes.

    ``first_max_chars`` caps the opening piece more tightly than the rest. The
    user waits for that piece and no other, so it is the only length that is
    worth trading naturalness for. Leave it None to split evenly.
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
        out.extend(_break_at(piece, max_chars))

    # The opening piece is the whole of the latency budget, so it gets a
    # tighter limit. Only the first one -- the rest are made while it plays.
    if first_max_chars and out:
        out = _split_opening(out[0], first_max_chars) + out[1:]
    return out


def register_cuda_dll_dirs() -> list[str]:
    """Put the pip-installed CUDA runtime on the DLL search path.

    onnxruntime's CUDA provider loads `cublasLt64_13.dll` and friends by bare
    name, and Windows searches the process DLL directories -- not site-packages,
    where the `nvidia-*` wheels put them. Without this the provider is listed as
    available, fails to build, and onnxruntime quietly runs on the CPU instead.

    torch ships its own CUDA 12 libraries in `torch/lib`, which are the wrong
    major version for onnxruntime 1.28; the two sets coexist because the version
    is in the filename. Returns what it added, so a caller can say why CUDA is
    still missing.
    """
    if not hasattr(os, "add_dll_directory"):  # not Windows
        return []

    import site

    roots = [Path(p) for p in site.getsitepackages()]
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    # Every directory under nvidia/ that holds a DLL, rather than a fixed
    # layout: cudnn ships them in <pkg>/bin, the CUDA 13 wheels one level
    # deeper in <pkg>/bin/x86_64, and that has moved between releases before.
    added: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        for dll in (root / "nvidia").rglob("*.dll"):
            parent = dll.parent.resolve()
            if parent in seen:
                continue
            seen.add(parent)
            with contextlib.suppress(OSError):
                os.add_dll_directory(str(parent))
                added.append(str(parent))

    # add_dll_directory alone is not enough. It covers DLLs Python loads, but
    # onnxruntime loads onnxruntime_providers_cuda.dll from native code, and
    # Windows resolves *that* DLL's own dependencies -- cublasLt64_13.dll and
    # the rest -- by the ordinary search order, which reads PATH and does not
    # consult directories added for the Python process.
    if added:
        existing = os.environ.get("PATH", "")
        missing = [d for d in added if d not in existing]
        if missing:
            os.environ["PATH"] = os.pathsep.join([*missing, existing])
    return added


def available_providers() -> list[str]:
    """What the installed onnxruntime can actually run.

    A CPU-only wheel returns just the CPU provider, which is why asking for
    CUDA has to be able to fail informatively rather than silently.
    """
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime ships with kokoro-onnx
        return []
    return list(ort.get_available_providers())


def choose_provider(requested: str = "auto", available: list[str] | None = None) -> str:
    """Resolve a provider name against what is installed.

    "auto" takes the best available. An explicit name that is not installed is
    an error rather than a silent downgrade -- if someone pins CUDA and gets
    the CPU, the only symptom is that the agent is mysteriously slow, which is
    exactly the class of bug M5 spent its time on.
    """
    if available is None:
        available = available_providers()
    if requested and requested != "auto":
        if requested not in available:
            raise ValueError(
                f"onnxruntime has no {requested}. Available: "
                f"{', '.join(available) or 'none'}.\n"
                "For CUDA, install the GPU build:\n"
                "    pip uninstall -y onnxruntime\n"
                "    pip install onnxruntime-gpu"
            )
        return requested
    for name in _PREFERRED:
        if name in available:
            return name
    return available[0] if available else "CPUExecutionProvider"


class Kokoro:
    def __init__(
        self,
        model_path: Path | str | None = None,
        voices_path: Path | str | None = None,
        voice: str = DEFAULT_VOICE,
        lang: str = DEFAULT_LANG,
        speed: float = DEFAULT_SPEED,
        provider: str | None = None,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL)
        self.voices_path = Path(voices_path or DEFAULT_VOICES)
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.provider = provider or config.load().tts_provider
        #: Resolved at load. Report this, never `provider` -- "auto" tells the
        #: reader nothing about what actually ran.
        self.provider_in_use: str | None = None
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

        if "CUDA" in (self.provider or "") or self.provider == "auto":
            register_cuda_dll_dirs()

        chosen = choose_provider(self.provider)
        # kokoro-onnx builds its own InferenceSession and takes no provider
        # argument, but it reads ONNX_PROVIDER when it does. Set it around the
        # construction only, so this never leaks into another session.
        with _env("ONNX_PROVIDER", chosen):
            self._engine = _Kokoro(str(self.model_path), str(self.voices_path))

        # Being *offered* a provider is not the same as getting it. onnxruntime
        # lists CUDA whenever the GPU wheel is installed, then falls back to the
        # CPU at session build if the CUDA runtime DLLs are not there -- it logs
        # and carries on. Reporting the requested name here would mean the
        # benchmark cheerfully labels a CPU run "CUDAExecutionProvider", which
        # is how a performance regression hides for a week. Ask the session.
        self.provider_in_use = self._engine.sess.get_providers()[0]
        if self.provider != "auto" and self.provider_in_use != chosen:
            raise RuntimeError(
                f"asked for {chosen}, got {self.provider_in_use}. onnxruntime "
                "loaded but could not build that provider -- usually a missing "
                "CUDA/cuDNN runtime. The onnxruntime log above names the DLL."
            )

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

    def stream(self, text: str, first_max_chars: int | None = FIRST_MAX_CHARS) -> Iterator[Chunk]:
        """Yield one chunk per sentence, in order.

        The caller plays each chunk as it arrives, so the wait before the first
        sound is one *piece* of synthesis rather than the whole reply -- and the
        first piece is deliberately the shortest one.
        """
        pieces = split_sentences(text, first_max_chars=first_max_chars)
        for index, sentence in enumerate(pieces):
            chunk = self.synthesise(sentence)
            yield Chunk(
                text=chunk.text, samples=chunk.samples, sample_rate=chunk.sample_rate,
                ms=chunk.ms, first=index == 0,
            )

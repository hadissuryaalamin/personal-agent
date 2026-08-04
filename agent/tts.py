"""Text-to-speech. Two backends behind one interface: `speak(text) -> WAV`.

- **Kokoro** (default) — 82M parameters, 54 voices, 24 kHz, CPU. Measured at
  ~4.3x realtime on this machine.
- **Piper** — kept for languages Kokoro does not cover.

`main.py` only needs to know about `speak()`. Switching backends is one line in
.env.
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np

from . import config

log = logging.getLogger(__name__)

_backend = None


class _Kokoro:
    """kokoro-onnx. CPU-only and zero VRAM, so it deliberately stays resident."""

    name = "kokoro"

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro

        for f in (config.KOKORO_MODEL, config.KOKORO_VOICES):
            if not f.exists():
                raise FileNotFoundError(
                    f"Kokoro file not found: {f}. Run scripts\\setup.ps1"
                )

        log.info("Loading Kokoro: %s", config.KOKORO_MODEL.name)
        self._k = Kokoro(str(config.KOKORO_MODEL), str(config.KOKORO_VOICES))

        available = set(self._k.get_voices())
        self.voice = config.KOKORO_VOICE
        if self.voice not in available:
            fallback = sorted(available)[0]
            log.warning(
                "voice %r not found, using %r. Options: %s",
                self.voice, fallback, ", ".join(sorted(available)[:8]),
            )
            self.voice = fallback
        log.info("Kokoro ready (voice=%s, %d available)", self.voice, len(available))

    def speak(self, text: str) -> bytes:
        samples, sr = self._k.create(
            text,
            voice=self.voice,
            speed=config.KOKORO_SPEED,
            lang=config.KOKORO_LANG,
        )
        return _to_wav(samples, sr)


class _Piper:
    name = "piper"

    def __init__(self) -> None:
        from piper import PiperVoice

        model = config.PIPER_VOICE
        cfg = model.with_suffix(model.suffix + ".json")
        for f in (model, cfg):
            if not f.exists():
                raise FileNotFoundError(
                    f"Piper voice not found: {f}. Run scripts\\setup.ps1"
                )

        log.info("Loading Piper voice: %s", model.name)
        self._v = PiperVoice.load(model, config_path=cfg)
        log.info("Voice ready (sample_rate=%d Hz)", self._v.config.sample_rate)

    def speak(self, text: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self._v.synthesize_wav(text, w)
        return buf.getvalue()


def _to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1,1] -> 16-bit WAV. audio.play_wav() reads the sample rate from
    the header, so Kokoro at 24 kHz and Piper at 22 kHz both just work."""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def get_voice():
    """Load the TTS backend once (lazy singleton)."""
    global _backend
    if _backend is not None:
        return _backend

    if config.TTS_BACKEND == "piper":
        _backend = _Piper()
    else:
        if config.TTS_BACKEND != "kokoro":
            log.warning(
                "TTS_BACKEND %r not recognised, falling back to 'kokoro'",
                config.TTS_BACKEND,
            )
        _backend = _Kokoro()
    return _backend


def speak(text: str) -> bytes:
    """Turn `text` into WAV bytes, ready for audio.play_wav()."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")

    data = get_voice().speak(text)
    log.debug("TTS: %d chars -> %d bytes wav", len(text), len(data))
    return data


if __name__ == "__main__":
    import sys

    from .audio import play_wav

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    text = " ".join(sys.argv[1:]) or "Hello, this is a test of your personal assistant."
    print(f"> {text}")
    play_wav(speak(text))

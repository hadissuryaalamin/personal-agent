"""Text-to-speech. Dua backend di balik satu antarmuka: `speak(text) -> WAV`.

- **Kokoro** (default) — 82 juta parameter, 54 suara, 24 kHz, CPU. Terukur
  ~4,3x realtime di mesin ini.
- **Piper** — dipertahankan buat bahasa yang nggak didukung Kokoro.

`main.py` cuma perlu tahu `speak()`. Mengganti backend itu satu baris di .env.
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
    """kokoro-onnx. CPU-only, nol VRAM, jadi sengaja tetap residen."""

    nama = "kokoro"

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro

        for berkas in (config.KOKORO_MODEL, config.KOKORO_VOICES):
            if not berkas.exists():
                raise FileNotFoundError(
                    f"Berkas Kokoro nggak ketemu: {berkas}. Jalanin scripts\\setup.ps1"
                )

        log.info("Load Kokoro: %s", config.KOKORO_MODEL.name)
        self._k = Kokoro(str(config.KOKORO_MODEL), str(config.KOKORO_VOICES))

        tersedia = set(self._k.get_voices())
        self.voice = config.KOKORO_VOICE
        if self.voice not in tersedia:
            pengganti = sorted(tersedia)[0]
            log.warning(
                "suara %r nggak ada, pakai %r. Pilihan: %s",
                self.voice, pengganti, ", ".join(sorted(tersedia)[:8]),
            )
            self.voice = pengganti
        log.info("Kokoro siap (suara=%s, %d pilihan)", self.voice, len(tersedia))

    def speak(self, text: str) -> bytes:
        samples, sr = self._k.create(
            text,
            voice=self.voice,
            speed=config.KOKORO_SPEED,
            lang=config.KOKORO_LANG,
        )
        return _ke_wav(samples, sr)


class _Piper:
    nama = "piper"

    def __init__(self) -> None:
        from piper import PiperVoice

        model = config.PIPER_VOICE
        cfg = model.with_suffix(model.suffix + ".json")
        for berkas in (model, cfg):
            if not berkas.exists():
                raise FileNotFoundError(
                    f"Voice Piper nggak ketemu: {berkas}. Jalanin scripts\\setup.ps1"
                )

        log.info("Load voice Piper: %s", model.name)
        self._v = PiperVoice.load(model, config_path=cfg)
        log.info("Voice siap (sample_rate=%d Hz)", self._v.config.sample_rate)

    def speak(self, text: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self._v.synthesize_wav(text, w)
        return buf.getvalue()


def _ke_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1,1] -> WAV 16-bit. audio.play_wav() baca sample rate dari
    header, jadi 24 kHz Kokoro maupun 22 kHz Piper sama-sama jalan."""
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
    """Muat backend TTS sekali (lazy singleton).

    Nama fungsinya dipertahankan dari versi Piper supaya pemanggil lama
    (warmup di main.py) nggak perlu berubah.
    """
    global _backend
    if _backend is not None:
        return _backend

    if config.TTS_BACKEND == "piper":
        _backend = _Piper()
    else:
        if config.TTS_BACKEND != "kokoro":
            log.warning(
                "TTS_BACKEND %r nggak dikenal, pakai 'kokoro'", config.TTS_BACKEND
            )
        _backend = _Kokoro()
    return _backend


def speak(text: str) -> bytes:
    """Ubah `text` jadi WAV (bytes, siap dilempar ke audio.play_wav)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("teks kosong")

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

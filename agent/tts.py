"""Wrapper Piper TTS (CPU-only, Bahasa Indonesia)."""

from __future__ import annotations

import io
import logging
import wave

from . import config

log = logging.getLogger(__name__)

_voice = None  # piper.PiperVoice, di-load lazy


def _voice_config_path():
    """Piper butuh <voice>.onnx.json di samping <voice>.onnx."""
    return config.PIPER_VOICE.with_suffix(config.PIPER_VOICE.suffix + ".json")


def get_voice():
    """Load voice Piper sekali aja (lazy singleton)."""
    global _voice
    if _voice is not None:
        return _voice

    from piper import PiperVoice  # import lokal: berat, cuma kepake pas TTS

    model = config.PIPER_VOICE
    cfg = _voice_config_path()
    if not model.exists():
        raise FileNotFoundError(
            f"Voice Piper nggak ketemu: {model}. Jalanin dulu scripts\\setup.ps1"
        )
    if not cfg.exists():
        raise FileNotFoundError(
            f"Config voice nggak ketemu: {cfg}. Jalanin dulu scripts\\setup.ps1"
        )

    log.info("Load voice Piper: %s", model.name)
    _voice = PiperVoice.load(model, config_path=cfg)
    log.info("Voice siap (sample_rate=%d Hz)", _voice.config.sample_rate)
    return _voice


def speak(text: str) -> bytes:
    """Ubah `text` jadi WAV (bytes, siap dilempar ke audio.play_wav)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("teks kosong")

    voice = get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize_wav(text, wav)
    data = buf.getvalue()
    log.debug("TTS: %d chars -> %d bytes wav", len(text), len(data))
    return data


if __name__ == "__main__":
    import sys

    from .audio import play_wav

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    text = " ".join(sys.argv[1:]) or "Halo, ini tes suara dari asisten pribadi kamu."
    print(f"> {text}")
    play_wav(speak(text))

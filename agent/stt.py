"""Wrapper faster-whisper (speech-to-text)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

_model = None  # faster_whisper.WhisperModel, di-load lazy
_cuda_siap = False


def _daftarin_dll_cuda() -> None:
    """Bikin DLL CUDA dari paket pip kebaca (Windows).

    ctranslate2 nyari cublas64_12.dll / cudnn*.dll lewat LoadLibrary biasa, yang
    baca PATH — bukan lewat direktori add_dll_directory. Paket nvidia-cublas-cu12
    & nvidia-cudnn-cu12 naruhnya di site-packages/nvidia/*/bin yang nggak ada di
    PATH, jadi dua-duanya didaftarin manual (PATH-nya yang beneran nolong).
    """
    global _cuda_siap
    if _cuda_siap or os.name != "nt":
        return
    try:
        import nvidia

        # `nvidia` itu namespace package: __file__ = None, jadi pakai __path__
        folders = [
            str(f)
            for root in nvidia.__path__
            for f in sorted(Path(root).glob("*/bin"))
        ]
        for folder in folders:
            os.add_dll_directory(folder)
        if folders:
            os.environ["PATH"] = os.pathsep.join(folders) + os.pathsep + os.environ["PATH"]
        log.info("%d folder DLL CUDA didaftarin", len(folders))
        _cuda_siap = True
    except ImportError:
        log.warning(
            "paket CUDA dari pip nggak ketemu. Kalau WHISPER_DEVICE=cuda gagal, "
            "jalanin: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
        )


def get_model():
    """Load model Whisper sekali aja (lazy singleton).

    Download otomatis dari HuggingFace pas pertama kali dipanggil.
    """
    global _model
    if _model is not None:
        return _model

    if config.WHISPER_DEVICE.startswith("cuda"):
        _daftarin_dll_cuda()

    from faster_whisper import WhisperModel  # import lokal: berat

    log.info(
        "Load Whisper '%s' (device=%s, compute=%s)...",
        config.WHISPER_MODEL,
        config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE,
    )
    t0 = time.monotonic()
    _model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE,
    )
    log.info("Whisper siap dalam %.1f detik", time.monotonic() - t0)
    return _model


def warmup() -> None:
    """Panggil pas startup biar pertanyaan pertama nggak kena delay load model."""
    get_model()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio float32 mono 16 kHz jadi teks Bahasa Indonesia."""
    if audio is None or len(audio) == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    model = get_model()

    t0 = time.monotonic()
    segments, info = model.transcribe(
        audio,
        language=config.WHISPER_LANG,
        initial_prompt=config.WHISPER_PROMPT or None,
        beam_size=5,
        # Buang bagian hening biar halusinasi ("Terima kasih", dst) berkurang
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log.info(
        "STT %.1f detik audio -> %.1f detik proses: %r",
        info.duration,
        time.monotonic() - t0,
        text,
    )
    return text


if __name__ == "__main__":
    import sys

    from . import audio as audio_io

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) > 1:
        # decode_audio sekalian resample ke 16 kHz mono
        from faster_whisper import decode_audio

        print(transcribe(decode_audio(sys.argv[1], sampling_rate=config.SAMPLE_RATE)))
    else:
        seconds = 5
        warmup()
        print(f"Ngomong sekarang ({seconds} detik)...")
        audio_io.beep_start()
        deadline = time.monotonic() + seconds
        clip = audio_io.record_until_release(lambda: time.monotonic() < deadline)
        audio_io.beep_stop()
        print("Hasil:", transcribe(clip))

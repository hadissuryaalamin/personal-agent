"""Speech-to-text. Dua backend di balik antarmuka yang sama.

- **Parakeet** (default) — nvidia/parakeet-tdt-0.6b-v2 lewat onnx-asr.
  English saja. Terukur di mesin ini: 0,40 detik/kalimat di CPU, **nol VRAM**,
  ~2,2 GB RAM. Secepat Whisper di GPU tanpa memakan VRAM sama sekali, jadi
  VRAM-nya bisa dipakai penuh sama LLM.
- **Whisper** — faster-whisper, 99 bahasa. Dipertahankan buat non-Inggris.

Antarmuka publiknya dijaga tetap: get_model(), is_loaded(), unload_model(),
warmup(), transcribe(). main.py nggak tahu backend mana yang dipakai.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

_model = None
_cuda_siap = False

# Lindungi _model dari dilepas pas lagi dipakai transcribe(). Reentrant karena
# transcribe() manggil get_model() yang juga ngambil lock ini.
_lock = threading.RLock()
_terakhir_dipakai = 0.0
_pelepas_jalan = False


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
            "paket CUDA dari pip nggak ketemu. Kalau device=cuda gagal, "
            "jalanin: pip install -e \".[gpu]\""
        )


# --- Backend ---------------------------------------------------------------


class _Parakeet:
    nama = "parakeet"

    def __init__(self) -> None:
        import onnx_asr

        if config.STT_DEVICE.startswith("cuda"):
            _daftarin_dll_cuda()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        log.info(
            "Load %s (device=%s)...", config.STT_MODEL, config.STT_DEVICE
        )
        self._m = onnx_asr.load_model(config.STT_MODEL, providers=providers)

        # onnxruntime diam-diam jatuh ke CPU kalau provider yang diminta nggak
        # ada. Lebih baik dikabari daripada bingung kenapa lambat.
        if config.STT_DEVICE.startswith("cuda"):
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                log.warning(
                    "CUDA diminta tapi onnxruntime yang terpasang CPU-only — "
                    "jalan di CPU. Pasang onnxruntime-gpu kalau mau GPU."
                )

    def transcribe(self, audio: np.ndarray) -> str:
        return (self._m.recognize(audio, sample_rate=config.SAMPLE_RATE) or "").strip()


class _Whisper:
    nama = "whisper"

    def __init__(self) -> None:
        if config.WHISPER_DEVICE.startswith("cuda"):
            _daftarin_dll_cuda()

        from faster_whisper import WhisperModel

        log.info(
            "Load Whisper '%s' (device=%s, compute=%s)...",
            config.WHISPER_MODEL,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE,
        )
        self._m = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, info = self._m.transcribe(
            audio,
            language=config.WHISPER_LANG,
            initial_prompt=config.WHISPER_PROMPT or None,
            beam_size=5,
            # Buang bagian hening biar halusinasi berkurang
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
        )
        teks = " ".join(seg.text.strip() for seg in segments).strip()
        log.debug("Whisper: %.1f detik audio", info.duration)
        return teks


def _bikin_backend():
    if config.STT_BACKEND == "whisper":
        return _Whisper()
    if config.STT_BACKEND != "parakeet":
        log.warning(
            "STT_BACKEND %r nggak dikenal, pakai 'parakeet'", config.STT_BACKEND
        )
    return _Parakeet()


# --- Siklus hidup ----------------------------------------------------------


def get_model():
    """Muat model STT (lazy singleton)."""
    global _model, _terakhir_dipakai

    with _lock:
        _terakhir_dipakai = time.monotonic()
        if _model is not None:
            return _model

        t0 = time.monotonic()
        _model = _bikin_backend()
        log.info("%s siap dalam %.1f detik", _model.nama, time.monotonic() - t0)
        _mulai_pelepas_idle()
        return _model


def is_loaded() -> bool:
    """Model lagi siap di memori? Dipakai buat mutusin perlu ngabarin user apa nggak."""
    return _model is not None


def warmup() -> None:
    """Panggil pas startup biar pertanyaan pertama nggak kena delay muat."""
    get_model()


def unload_model() -> bool:
    """Lepas model dari memori. Balikin True kalau tadinya emang ke-load."""
    global _model
    with _lock:
        if _model is None:
            return False
        _model = None
        gc.collect()
        log.info("model STT dilepas dari memori (nganggur)")
        return True


def _pelepas_idle() -> None:
    batas = config.WHISPER_IDLE_UNLOAD_SECONDS
    jeda = max(1.0, min(30.0, batas / 10))
    while True:
        time.sleep(jeda)
        with _lock:
            if _model is None:
                continue
            nganggur = time.monotonic() - _terakhir_dipakai
        if nganggur >= batas:
            unload_model()


def _mulai_pelepas_idle() -> None:
    """Nyalain pengawas idle sekali aja, pas model pertama kali ke-load."""
    global _pelepas_jalan
    if _pelepas_jalan or config.WHISPER_IDLE_UNLOAD_SECONDS <= 0:
        return
    _pelepas_jalan = True
    threading.Thread(target=_pelepas_idle, name="stt-idle", daemon=True).start()
    batas = config.WHISPER_IDLE_UNLOAD_SECONDS
    log.info(
        "Model bakal dilepas kalau nganggur %s",
        f"{batas:.0f} detik" if batas < 60 else f"{batas / 60:.0f} menit",
    )


# --- Transkripsi -----------------------------------------------------------


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio float32 mono 16 kHz jadi teks."""
    if audio is None or len(audio) == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    # Lock ditahan selama transcribe biar pengawas idle nggak ngelepas model
    # di tengah jalan. get_model() reentrant, jadi aman.
    with _lock:
        return _transcribe_terkunci(audio)


def _transcribe_terkunci(audio: np.ndarray) -> str:
    global _terakhir_dipakai

    model = get_model()
    durasi = len(audio) / config.SAMPLE_RATE

    t0 = time.monotonic()
    text = model.transcribe(audio)
    log.info(
        "STT %.1f detik audio -> %.1f detik proses: %r",
        durasi,
        time.monotonic() - t0,
        text,
    )
    # Dicatat setelah selesai: hitungan nganggur mulai dari akhir pemakaian
    _terakhir_dipakai = time.monotonic()
    return text


if __name__ == "__main__":
    import sys

    from . import audio as audio_io

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) > 1:
        from faster_whisper import decode_audio

        print(transcribe(decode_audio(sys.argv[1], sampling_rate=config.SAMPLE_RATE)))
    else:
        seconds = 5
        warmup()
        print(f"Speak now ({seconds} seconds)...")
        audio_io.beep_start()
        deadline = time.monotonic() + seconds
        clip = audio_io.record_until_release(lambda: time.monotonic() < deadline)
        audio_io.beep_stop()
        print("Result:", transcribe(clip))

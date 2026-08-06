"""Speech-to-text. Two backends behind the same interface.

- **Parakeet** (default) — nvidia/parakeet-tdt-0.6b-v2 through onnx-asr.
  English only. Measured on this machine: 0.40 s per sentence on CPU, **zero
  VRAM**, ~2.2 GB RAM. As fast as Whisper on GPU without touching VRAM at all,
  which leaves the whole GPU to the LLM.
- **Whisper** — faster-whisper, 99 languages. Kept for non-English.

The public interface is deliberately stable: get_model(), is_loaded(),
unload_model(), warmup(), transcribe(). main.py never knows which backend is in
use.
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
_cuda_ready = False

# Protects _model from being released while transcribe() is using it.
# Reentrant because transcribe() calls get_model(), which takes the same lock.
_lock = threading.RLock()
_last_used = 0.0
_unloader_running = False


def _register_cuda_dlls() -> None:
    """Make CUDA DLLs from pip packages loadable (Windows).

    ctranslate2 looks for cublas64_12.dll / cudnn*.dll with a plain LoadLibrary,
    which reads PATH — not the add_dll_directory list. The nvidia-cublas-cu12
    and nvidia-cudnn-cu12 packages put them in site-packages/nvidia/*/bin, which
    is not on PATH, so both mechanisms are registered here (PATH is the one that
    actually helps).
    """
    global _cuda_ready
    if _cuda_ready or os.name != "nt":
        return
    try:
        import nvidia

        # `nvidia` is a namespace package: __file__ is None, so use __path__
        folders = [
            str(f)
            for root in nvidia.__path__
            for f in sorted(Path(root).glob("*/bin"))
        ]
        for folder in folders:
            os.add_dll_directory(folder)
        if folders:
            os.environ["PATH"] = os.pathsep.join(folders) + os.pathsep + os.environ["PATH"]
        log.info("registered %d CUDA DLL folders", len(folders))
        _cuda_ready = True
    except ImportError:
        log.warning(
            'CUDA pip packages not found. If device=cuda fails, run: '
            'pip install -e ".[gpu]"'
        )


# --- Backends ---------------------------------------------------------------


class _Parakeet:
    name = "parakeet"

    def __init__(self) -> None:
        import onnx_asr

        if config.STT_DEVICE.startswith("cuda"):
            _register_cuda_dlls()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        log.info("Loading %s (device=%s)...", config.STT_MODEL, config.STT_DEVICE)
        self._m = onnx_asr.load_model(config.STT_MODEL, providers=providers)

        # onnxruntime silently falls back to CPU when the requested provider is
        # missing. Better to say so than to leave you wondering why it is slow.
        if config.STT_DEVICE.startswith("cuda"):
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                log.warning(
                    "CUDA requested but the installed onnxruntime is CPU-only — "
                    "running on CPU. Install onnxruntime-gpu for GPU."
                )

    def transcribe(self, audio: np.ndarray) -> str:
        return (self._m.recognize(audio, sample_rate=config.SAMPLE_RATE) or "").strip()


class _Whisper:
    name = "whisper"

    def __init__(self) -> None:
        if config.WHISPER_DEVICE.startswith("cuda"):
            _register_cuda_dlls()

        from faster_whisper import WhisperModel

        log.info(
            "Loading Whisper '%s' (device=%s, compute=%s)...",
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
            # Drop silent stretches to cut down on hallucination
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.debug("Whisper: %.1f s of audio", info.duration)
        return text


def _make_backend():
    if config.STT_BACKEND == "whisper":
        return _Whisper()
    if config.STT_BACKEND != "parakeet":
        log.warning(
            "STT_BACKEND %r not recognised, falling back to 'parakeet'",
            config.STT_BACKEND,
        )
    return _Parakeet()


# --- Lifecycle --------------------------------------------------------------


def get_model():
    """Load the STT model (lazy singleton)."""
    global _model, _last_used

    with _lock:
        _last_used = time.monotonic()
        if _model is not None:
            return _model

        t0 = time.monotonic()
        _model = _make_backend()
        log.info("%s ready in %.1f s", _model.name, time.monotonic() - t0)
        _start_idle_unloader()
        return _model


def is_loaded() -> bool:
    """Is the model resident right now? Used to decide whether to warn the user."""
    return _model is not None


def warmup() -> None:
    """Call at startup so the first question doesn't pay the load time."""
    get_model()


def unload_model() -> bool:
    """Release the model from memory. True if it had actually been loaded."""
    global _model
    with _lock:
        if _model is None:
            return False
        _model = None
        gc.collect()
        log.info("STT model released from memory (idle)")
        return True


def _idle_unloader() -> None:
    limit = config.WHISPER_IDLE_UNLOAD_SECONDS
    interval = max(1.0, min(30.0, limit / 10))
    while True:
        time.sleep(interval)
        with _lock:
            if _model is None:
                continue
            idle = time.monotonic() - _last_used
        if idle >= limit:
            unload_model()


def _start_idle_unloader() -> None:
    """Start the idle watchdog once, when the model is first loaded."""
    global _unloader_running
    if _unloader_running or config.WHISPER_IDLE_UNLOAD_SECONDS <= 0:
        return
    _unloader_running = True
    threading.Thread(target=_idle_unloader, name="stt-idle", daemon=True).start()
    limit = config.WHISPER_IDLE_UNLOAD_SECONDS
    log.info(
        "Model will be released after %s idle",
        f"{limit:.0f} s" if limit < 60 else f"{limit / 60:.0f} min",
    )


# --- Transcription ----------------------------------------------------------


def transcribe(audio: np.ndarray) -> str:
    """Transcribe mono float32 16 kHz audio into text."""
    if audio is None or len(audio) == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    # The lock is held for the whole transcription so the idle watchdog cannot
    # pull the model out from under it. get_model() is reentrant, so this is safe.
    with _lock:
        return _transcribe_locked(audio)


def _transcribe_locked(audio: np.ndarray) -> str:
    global _last_used

    model = get_model()
    seconds = len(audio) / config.SAMPLE_RATE

    t0 = time.monotonic()
    text = model.transcribe(audio)
    log.info(
        "STT %.1f s audio -> %.1f s processing: %r",
        seconds,
        time.monotonic() - t0,
        text,
    )
    # Recorded after the work, so the idle countdown starts from end of use
    _last_used = time.monotonic()
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

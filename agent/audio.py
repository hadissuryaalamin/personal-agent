"""Capture mic & playback speaker."""

from __future__ import annotations

import io
import logging
import queue
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config

log = logging.getLogger(__name__)


# --- Playback ---------------------------------------------------------------


def _pad(data: np.ndarray, samplerate: int) -> np.ndarray:
    """Sisipin hening di kedua ujung biar suaranya nggak kepotong.

    Tanpa ini, di Windows awal suara hilang (device output masih kebuka pas
    sampel pertama dikirim) dan akhirnya kepotong (sd.wait() balik pas buffer
    habis disuapin, padahal device masih mainin sisanya).
    """
    n = int(samplerate * config.PLAYBACK_PAD_SECONDS)
    if n <= 0:
        return data
    hening = np.zeros((n,) + data.shape[1:], dtype=data.dtype)
    return np.concatenate([hening, data, hening])


def _play_array(data: np.ndarray, samplerate: int, blocking: bool = True) -> None:
    sd.play(_pad(data, samplerate), samplerate)
    if blocking:
        sd.wait()


def play_wav(wav_bytes: bytes, blocking: bool = True) -> None:
    """Mainin WAV (bytes) ke output device default."""
    data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    _play_array(data, samplerate, blocking)


def stop_playback() -> None:
    sd.stop()


# --- Beep feedback ----------------------------------------------------------


def _tone(freq: float, duration: float, volume: float = 0.25) -> np.ndarray:
    """Sine pendek dengan fade in/out biar nggak 'klik'."""
    sr = config.SAMPLE_RATE
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    wave = np.sin(2 * np.pi * freq * t, dtype=np.float32) * volume

    fade = max(1, int(sr * 0.01))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    wave[:fade] *= ramp
    wave[-fade:] *= ramp[::-1]
    return wave


def _beep(freq: float, duration: float = 0.09, blocking: bool = True) -> None:
    try:
        # Beep-nya pendek banget, jadi justru paling gampang ketelan tanpa bantalan
        _play_array(_tone(freq, duration), config.SAMPLE_RATE, blocking)
    except Exception:
        # Beep cuma feedback; jangan sampai bikin pipeline mati
        log.warning("gagal bunyiin beep", exc_info=True)


def beep_start() -> None:
    """Nada naik: mulai rekam.

    Sengaja NON-BLOCKING. Kalau nunggu beep selesai, buffer mic baru dibuang
    setelahnya — padahal user udah mulai ngomong begitu denger nadanya, jadi
    kata pertamanya ikut kebuang. Nada 880 Hz yang bocor ke mic nggak masalah:
    VAD filter Whisper nganggapnya bukan suara orang.
    """
    _beep(880.0, blocking=False)


def beep_stop() -> None:
    """Nada turun: selesai rekam."""
    _beep(560.0)


def beep_error() -> None:
    """Nada rendah panjang: ada yang error."""
    _beep(240.0, duration=0.25)


# --- Recording --------------------------------------------------------------


def record_until_release(
    is_held: Callable[[], bool],
    on_ready: Callable[[], None] | None = None,
    release_grace: float | None = None,
    poll_interval: float = 0.02,
) -> np.ndarray:
    """Rekam mic selama `is_held()` masih True.

    `release_grace` = tenggang sebelum berhenti pas `is_held()` jadi False.
    Perlu di mode tahan (Windows ngirim UP/DOWN palsu), tapi di mode toggle
    berhentinya eksplisit jadi dilewatin aja (0).

    `on_ready` dipanggil pas stream udah beneran kebuka — dipakai buat bunyiin
    beep, biar user nggak mulai ngomong sebelum mic-nya nyala (buka stream bisa
    makan setengah detik).

    Balikin float32 mono 1-D pada `config.SAMPLE_RATE`. Array kosong kalau
    rekamannya terlalu pendek (kepencet nggak sengaja).
    """
    if release_grace is None:
        release_grace = config.RELEASE_GRACE_SECONDS

    frames: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, _frames, _time_info, status):
        if status:
            log.debug("status input stream: %s", status)
        frames.put(indata.copy())

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="float32",
        callback=callback,
    ):
        if on_ready is not None:
            on_ready()
        # Buang apa pun yang kerekam sebelum/selama beep
        while not frames.empty():
            frames.get()

        started = time.monotonic()
        released_at: float | None = None
        while True:
            now = time.monotonic()

            if is_held():
                # Kombinasi utuh lagi — ternyata cuma kedipan, lanjut rekam
                released_at = None
            elif released_at is None:
                released_at = now
            elif now - released_at >= release_grace:
                break

            if now - started > config.MAX_RECORD_SECONDS:
                log.warning(
                    "rekaman dipotong di %.0f detik (batas aman)",
                    config.MAX_RECORD_SECONDS,
                )
                break
            time.sleep(poll_interval)

    chunks = []
    while not frames.empty():
        chunks.append(frames.get())

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).reshape(-1)
    seconds = len(audio) / config.SAMPLE_RATE
    if seconds < config.MIN_RECORD_SECONDS:
        log.info("rekaman cuma %.2f detik, dibuang", seconds)
        return np.zeros(0, dtype=np.float32)

    log.info("rekaman %.2f detik (%d sample)", seconds, len(audio))
    if config.SAVE_RECORDINGS:
        _save_recording(audio)
    return audio


def _save_recording(audio: np.ndarray) -> None:
    """Simpan rekaman mentah buat debugging. Nggak boleh bikin pipeline gagal."""
    try:
        config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + ".wav"
        path = config.RECORDINGS_DIR / name
        sf.write(path, audio, config.SAMPLE_RATE)
        log.info("rekaman disimpan: %s", path)
    except Exception:
        log.warning("gagal nyimpen rekaman", exc_info=True)


def list_devices() -> str:
    return str(sd.query_devices())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(list_devices())
    print("\nBeep start/stop:")
    beep_start()
    time.sleep(0.3)
    beep_stop()

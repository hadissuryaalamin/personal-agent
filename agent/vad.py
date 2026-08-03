"""Deteksi suara (VAD) buat mode sesi: tau kapan kamu mulai & berhenti ngomong.

Pakai Silero VAD lewat onnx_asr — model yang sama yang udah dipakai buat
Parakeet, jadi **nol paket pip baru**. Alasannya sama kayak waktu milih
Parakeet: nambah dependensi berat buat satu fungsi kecil itu mahal di mesin
yang harus jalan offline.

`onnx_asr` sendiri nyediain VAD untuk memotong rekaman utuh (batch). Yang
dibutuhin di sini beda: keputusan **per frame, sambil jalan**, karena mode sesi
harus tau kapan kalimatmu selesai tanpa nunggu kamu mencet apa pun. Jadi
sesi ONNX-nya dipakai langsung, frame demi frame.

Satu frame = 512 sampel @ 16 kHz = 32 ms.
"""

import logging
import queue
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from . import config

log = logging.getLogger(__name__)

# Silero v5 di 16 kHz. Angkanya dari model, bukan pilihan bebas.
HOP = 512
CONTEXT = 64
FRAME_MS = HOP * 1000 // 16000  # 32

_sesi = None
_lock = __import__("threading").Lock()


def _model():
    """Muat sekali. Dipisah dari stt.get_model() karena umurnya beda: VAD
    dipegang selama sesi ngobrol, model STT boleh dilepas di antara giliran."""
    global _sesi
    with _lock:
        if _sesi is None:
            import onnx_asr

            t0 = time.perf_counter()
            _sesi = onnx_asr.load_vad("silero")._model
            log.info("VAD Silero siap (%.1f detik)", time.perf_counter() - t0)
        return _sesi


def is_loaded() -> bool:
    return _sesi is not None


def unload() -> bool:
    global _sesi
    with _lock:
        ada, _sesi = _sesi is not None, None
    return ada


class Detector:
    """Ngasih peluang-ada-suara per frame, sambil bawa state antar frame.

    Statenya penting: Silero itu rekuren. Frame yang sama bisa dinilai beda
    tergantung apa yang barusan kedengeran, dan itu justru yang bikin dia
    nggak gampang ketipu suara kipas atau ketukan keyboard.
    """

    def __init__(self) -> None:
        self._m = _model()
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ekor = np.zeros(CONTEXT, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        """`frame` = HOP sampel float32. Balikin peluang 0..1."""
        if len(frame) < HOP:
            frame = np.pad(frame, (0, HOP - len(frame)))
        masuk = np.concatenate([self._ekor, frame[:HOP]])[None, :].astype(np.float32)
        keluar, state_baru = self._m.run(
            ["output", "stateN"],
            {"input": masuk, "state": self._state, "sr": np.array(16000, dtype=np.int64)},
        )
        self._state = state_baru
        self._ekor = frame[HOP - CONTEXT : HOP].copy()
        return float(keluar[0, 0])


def rekam_ucapan(
    berhenti: Callable[[], bool],
    batas_sepi: float | None = None,
    on_mulai_bicara: Callable[[], None] | None = None,
) -> tuple[np.ndarray | None, str]:
    """Tunggu sampai kamu ngomong, rekam, berhenti pas kamu diam.

    Balikin `(audio, alasan)`:
      - `(array, "ok")`   : ada ucapan, udah selesai
      - `(None, "stop")`  : `berhenti()` jadi True
      - `(None, "sepi")`  : nggak ada suara sampai `batas_sepi` lewat

    Alasannya dipisah dengan sengaja: pemanggil harus bisa bedain "user nutup
    sesi" dari "sesi mati sendiri". Suara yang kependekan (batuk, ketukan meja)
    NGGAK dibalikin sebagai alasan tersendiri — itu ditelan di dalam sini,
    karena batuk nggak boleh nutup sesi ngobrol.

    `on_mulai_bicara` dipanggil sekali begitu suara kedeteksi — dipakai buat
    matiin timer sesi, bukan buat bunyiin apa pun.
    """
    if batas_sepi is None:
        batas_sepi = config.SESSION_IDLE_SECONDS

    det = Detector()
    antre: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, _n, _t, status):
        if status:
            log.debug("status input stream: %s", status)
        antre.put(indata[:, 0].copy())

    diam_perlu = int(config.VAD_SILENCE_MS / FRAME_MS)
    min_bicara = int(config.VAD_MIN_SPEECH_MS / FRAME_MS)
    # Simpen sedikit audio SEBELUM suara kedeteksi. VAD selalu telat sedikit
    # ngenalin awal kata, dan tanpa bantalan ini konsonan pertama kepotong.
    pad = max(1, int(config.VAD_SPEECH_PAD_MS / FRAME_MS))

    sebelum: list[np.ndarray] = []
    isi: list[np.ndarray] = []
    bicara = False
    n_bicara = 0
    n_diam = 0
    # Tenggat sepi dihitung dari awal fungsi dan SENGAJA nggak di-reset sama
    # suara pendek. Kalau di-reset, ruangan berisik bisa nahan sesi kebuka
    # selamanya tanpa kamu ngomong sekali pun.
    tenggat = time.monotonic() + batas_sepi

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="float32",
        blocksize=HOP,
        callback=callback,
    ):
        while True:
            if berhenti():
                return None, "stop"

            try:
                blok = antre.get(timeout=0.1)
            except queue.Empty:
                if not bicara and time.monotonic() >= tenggat:
                    return None, "sepi"
                continue

            p = det.prob(blok)

            if not bicara:
                sebelum.append(blok)
                if len(sebelum) > pad:
                    sebelum.pop(0)

                if p >= config.VAD_THRESHOLD:
                    n_bicara += 1
                    if n_bicara >= min_bicara:
                        bicara = True
                        isi = list(sebelum)
                        sebelum.clear()
                        if on_mulai_bicara:
                            on_mulai_bicara()
                else:
                    n_bicara = 0
                    if time.monotonic() >= tenggat:
                        return None, "sepi"
                continue

            isi.append(blok)
            # Ambang lepas sengaja lebih rendah dari ambang tangkap. Dengan satu
            # ambang, jeda alami di tengah kalimat kebaca sebagai selesai dan
            # kalimatmu kepotong dua.
            if p < config.VAD_THRESHOLD - 0.15:
                n_diam += 1
                if n_diam < diam_perlu:
                    continue
            else:
                n_diam = 0
                if len(isi) * FRAME_MS / 1000 <= config.MAX_RECORD_SECONDS:
                    continue
                log.warning("ucapan dipotong di %.0f detik", config.MAX_RECORD_SECONDS)

            audio = np.concatenate(isi)
            durasi = len(audio) / config.SAMPLE_RATE
            if durasi >= config.MIN_RECORD_SECONDS:
                return audio, "ok"

            # Kependekan — batuk, ketukan meja, decak. Balik nunggu, JANGAN
            # nutup sesi.
            log.debug("suara %.2f dtk kependekan, lanjut nunggu", durasi)
            bicara = False
            n_bicara = n_diam = 0
            isi = []
            sebelum.clear()
            det.reset()


if __name__ == "__main__":
    # Uji manual: ngomong, lihat apa batas kalimatnya ketangkep bener.
    #
    #     .venv-agent\Scripts\python.exe -m agent.vad
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from . import audio as audio_io

    print("Ngomong sesuatu. Diam 30 detik buat berhenti. Ctrl+C juga bisa.")
    n = 0
    while True:
        t0 = time.monotonic()
        clip, alasan = rekam_ucapan(lambda: False)
        if clip is None:
            print(f"({alasan} — selesai)")
            break
        n += 1
        print(f"  ucapan {n}: {len(clip)/config.SAMPLE_RATE:.2f} dtk "
              f"(nunggu+rekam {time.monotonic()-t0:.1f} dtk)")
        audio_io.beep_stop()

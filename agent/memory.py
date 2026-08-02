"""Memori antar-sesi: riwayat obrolan + catatan fakta, disimpan ke disk.

Dua lapis dengan umur yang beda:

- **Riwayat** (`history.json`) — pesan mentah, buat nyambungin obrolan yang
  kepotong restart. Punya batas umur: riwayat basi bikin salah konteks.
- **Fakta** (`facts.md`) — hal yang layak diingat lama (nama, jurusan,
  preferensi). Teks polos biar kamu bisa baca dan sunting sendiri.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()


def _history_path() -> Path:
    return config.MEMORY_DIR / "history.json"


def _facts_path() -> Path:
    return config.MEMORY_DIR / "facts.md"


def _tulis_atomik(path: Path, isi: str) -> None:
    """Tulis lewat file sementara lalu ganti nama.

    Kalau proses mati di tengah penulisan, file lamanya tetap utuh — nggak
    kepotong separuh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(isi, encoding="utf-8")
    os.replace(tmp, path)


# --- Riwayat ---------------------------------------------------------------


def load_history() -> list[dict[str, str]]:
    """Baca riwayat dari disk. Balikin [] kalau mati, nggak ada, atau basi."""
    if not config.MEMORY_ENABLED:
        return []

    path = _history_path()
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pesan = data.get("messages", [])
        disimpan = float(data.get("saved_at", 0))
    except Exception:
        log.warning("riwayat nggak kebaca, mulai dari kosong", exc_info=True)
        return []

    umur_jam = (time.time() - disimpan) / 3600
    if umur_jam > config.HISTORY_MAX_AGE_HOURS:
        log.info(
            "riwayat udah %.1f jam (batas %.0f), nggak dimuat",
            umur_jam,
            config.HISTORY_MAX_AGE_HOURS,
        )
        return []

    log.info("riwayat dimuat: %d pesan, umur %.1f jam", len(pesan), umur_jam)
    return pesan


def save_history(messages: list[dict[str, str]]) -> None:
    if not config.MEMORY_ENABLED:
        return
    try:
        with _lock:
            _tulis_atomik(
                _history_path(),
                json.dumps(
                    {"saved_at": time.time(), "messages": messages},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
    except Exception:
        log.warning("gagal nyimpen riwayat", exc_info=True)


# --- Fakta -----------------------------------------------------------------


def read_facts() -> str:
    if not config.MEMORY_ENABLED:
        return ""
    path = _facts_path()
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        log.warning("fakta nggak kebaca", exc_info=True)
        return ""


def write_facts(teks: str) -> None:
    if not config.MEMORY_ENABLED:
        return
    baris = [b.strip() for b in teks.splitlines() if b.strip()]
    # Semua fakta ikut ke tiap permintaan, jadi jumlahnya dibatasi biar biaya
    # token nggak naik terus. Yang dibuang yang paling lama.
    if len(baris) > config.FACTS_MAX_ITEMS:
        baris = baris[-config.FACTS_MAX_ITEMS :]
    try:
        with _lock:
            _tulis_atomik(_facts_path(), "\n".join(baris) + "\n")
    except Exception:
        log.warning("gagal nyimpen fakta", exc_info=True)


def forget_all() -> None:
    """Hapus semua memori. Dipanggil pas user minta dilupakan."""
    with _lock:
        for path in (_history_path(), _facts_path()):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                log.warning("gagal hapus %s", path, exc_info=True)
    log.info("semua memori dihapus")

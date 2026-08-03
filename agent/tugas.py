"""Daftar tugas: simpan, tandai selesai, dan susun jadi teks buat agent.

Disimpan sebagai JSON di memory/tugas.json. Teks polos, boleh disunting manual.

Beda dari acara kalender: tugas nggak nempatin slot waktu, punya status
selesai/belum, dan bisa dicicil. Makanya disimpan terpisah, bukan jadi acara.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()

HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

# Kata yang nandain ini tugas, bukan acara kalender. Dicek DULUAN sebelum
# niat bikin acara, karena "catat tugas..." juga cocok sama pola "catat ...".
_KATA_TUGAS = ("tugas", "assignment", "pr ", "deadline", "tenggat", "quiz", "ujian",
               "laporan", "esai", "essay", "makalah")
_NIAT_TAMBAH = ("catat", "tambah", "ada", "inget", "ingat", "simpan")
_NIAT_SELESAI = ("selesai", "beres", "kelar", "udah ngerjain", "sudah ngerjain",
                 "done", "rampung")


def _path():
    return config.MEMORY_DIR / "tugas.json"


def _muat() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("tugas", [])
    except Exception:
        log.warning("daftar tugas nggak kebaca", exc_info=True)
        return []


def _simpan(daftar: list[dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"tugas": daftar}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, p)


def semua(termasuk_selesai: bool = False) -> list[dict]:
    daftar = _muat()
    return daftar if termasuk_selesai else [t for t in daftar if not t.get("selesai")]


def tambah(judul: str, tenggat: str = "", matkul: str = "", perkiraan_jam: float = 0) -> dict:
    """Tambah tugas. `tenggat` format YYYY-MM-DD, boleh kosong."""
    t = {
        "judul": judul.strip(),
        "tenggat": tenggat,
        "matkul": matkul.strip(),
        "perkiraan_jam": perkiraan_jam,
        "selesai": False,
        "dibuat": datetime.now().isoformat(timespec="seconds"),
    }
    with _lock:
        daftar = _muat()
        daftar.append(t)
        _simpan(daftar)
    log.info("tugas ditambah: %r", t)
    return t


def tandai(judul_kira: str, selesai: bool = True) -> dict | None:
    """Tandai tugas selesai/belum berdasarkan kecocokan kata.

    Balikin tugasnya kalau ketemu tepat satu, None kalau nggak ketemu atau
    ambigu — biar agent bisa minta perjelas, bukan nebak tugas mana.
    """
    kata = set(re.sub(r"[^\w\s]", " ", judul_kira.lower()).split())
    with _lock:
        daftar = _muat()
        skor = []
        for i, t in enumerate(daftar):
            if t.get("selesai") == selesai:
                continue
            teks = set(
                re.sub(r"[^\w\s]", " ", f"{t['judul']} {t.get('matkul','')}".lower()).split()
            )
            cocok = len(kata & teks)
            if cocok:
                skor.append((cocok, i))
        if not skor:
            return None
        skor.sort(reverse=True)
        # Ambigu kalau dua tugas sama-sama cocok terbanyak
        if len(skor) > 1 and skor[0][0] == skor[1][0]:
            log.info("tugas ambigu buat %r", judul_kira)
            return None
        idx = skor[0][1]
        daftar[idx]["selesai"] = selesai
        _simpan(daftar)
        log.info("tugas ditandai selesai=%s: %r", selesai, daftar[idx]["judul"])
        return daftar[idx]


def hapus_semua() -> None:
    with _lock:
        _simpan([])
    log.info("semua tugas dihapus")


# --- Deteksi niat ----------------------------------------------------------


def _mengandung_kata_tugas(teks: str) -> bool:
    t = " " + re.sub(r"[^\w\s]", " ", teks.lower()) + " "
    return any(k in t for k in _KATA_TUGAS)


def minta_tambah_tugas(teks: str) -> bool:
    if not _mengandung_kata_tugas(teks):
        return False
    t = re.sub(r"[^\w\s]", " ", teks.lower())
    if any(k in t for k in _NIAT_SELESAI):
        return False  # itu laporan selesai, bukan tugas baru
    return any(k in t for k in _NIAT_TAMBAH)


def minta_tandai_selesai(teks: str) -> bool:
    if not _mengandung_kata_tugas(teks):
        return False
    t = re.sub(r"[^\w\s]", " ", teks.lower())
    return any(k in t for k in _NIAT_SELESAI)


# --- Teks buat system prompt ----------------------------------------------


def _label_tenggat(tenggat: str, hari_ini: date) -> str:
    """Sisa hari dihitung di sini, bukan diserahin ke model — sama alasannya
    kayak KELAS BERIKUTNYA: hitung tanggal lintas baris itu rawan meleset."""
    try:
        d = datetime.strptime(tenggat, "%Y-%m-%d").date()
    except Exception:
        return ""
    sisa = (d - hari_ini).days
    tgl = f"{HARI[d.weekday()]} {d.day} {BULAN[d.month]}"
    if sisa < 0:
        return f"tenggat {tgl} (LEWAT {abs(sisa)} hari)"
    if sisa == 0:
        return f"tenggat {tgl} (HARI INI)"
    if sisa == 1:
        return f"tenggat {tgl} (BESOK)"
    return f"tenggat {tgl} ({sisa} hari lagi)"


def ringkasan() -> str:
    """Daftar tugas buat diselipin ke system prompt."""
    belum = semua()
    if not belum:
        return ""

    hari_ini = datetime.now(ZoneInfo(config.CALENDAR_TZ)).date()

    def kunci(t):
        # Yang punya tenggat duluan, urut dari yang paling dekat
        return (0, t["tenggat"]) if t.get("tenggat") else (1, "")

    baris = ["Daftar tugas user yang belum selesai:"]
    for t in sorted(belum, key=kunci):
        bagian = [t["judul"]]
        if t.get("matkul"):
            bagian.append(t["matkul"])
        label = _label_tenggat(t.get("tenggat", ""), hari_ini)
        bagian.append(label if label else "tanpa tenggat")
        if t.get("perkiraan_jam"):
            bagian.append(f"perkiraan {t['perkiraan_jam']:g} jam")
        baris.append("  - " + " | ".join(bagian))
    return "\n".join(baris)

"""Ubah ucapan jadi acara kalender, dengan konfirmasi sebelum disimpan.

Kenapa harus dikonfirmasi: STT punya angka salah dengar yang nyata (WER ~8%).
Buat pertanyaan, salah dengar cuma bikin jawaban ngawur dan langsung ketahuan.
Buat penulisan, salah dengar ninggalin acara palsu di kalender yang baru
ketahuan minggu depan. Jadi agent selalu bacain ulang dulu dan nunggu "ya".
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)

# Kata kunci niat bikin acara. Dicocokkan lokal buat mutusin apakah ucapan ini
# perlu diproses jadi acara — bukan buat nentuin isinya (itu tugas LLM).
_NIAT = (
    "catat", "catetin", "jadwalin", "jadwalkan", "tambahin acara", "tambah acara",
    "bikin acara", "buat acara", "masukin ke kalender", "masukkan ke kalender",
    "ingetin aku tanggal", "set jadwal", "atur jadwal",
)

# "ya" sengaja NGGAK di sini: dia sering nempel di akhir kalimat tanya
# ("hmm apa ya", "gimana ya") dan itu keraguan, bukan persetujuan. Dia cuma
# dihitung kalau jadi kata pertama — lihat jawaban_ya().
_YA = ("iya", "betul", "bener", "benar", "oke", "ok", "sip", "simpan", "lanjut", "gas", "yoi")
_TIDAK = ("nggak", "ngga", "gak", "tidak", "bukan", "batal", "salah", "jangan", "no")

SKEMA = {
    "type": "object",
    "properties": {
        "judul": {"type": "string", "description": "Nama acaranya, ringkas"},
        "tanggal": {"type": "string", "description": "Tanggal mulai, format YYYY-MM-DD"},
        "jam_mulai": {"type": "string", "description": "Jam mulai, format HH:MM 24 jam"},
        "durasi_menit": {"type": "integer", "description": "Lama acara dalam menit"},
        "lokasi": {"type": "string", "description": "Lokasi, boleh string kosong"},
        "yakin": {
            "type": "boolean",
            "description": "false kalau tanggal/jam-nya nggak jelas dari ucapan user",
        },
    },
    "required": ["judul", "tanggal", "jam_mulai", "durasi_menit", "lokasi", "yakin"],
    "additionalProperties": False,
}

PROMPT = """Ubah permintaan user jadi satu acara kalender.

Aturan:
- Pakai tanggal & jam sekarang buat ngartiin "besok", "Kamis depan", "lusa".
- Kalau user nggak nyebut durasi, pakai 60 menit.
- Kalau user nyebut jam tanpa keterangan pagi/siang/malam, pilih yang paling
  masuk akal buat mahasiswa (jam 3 = 15:00, jam 8 = 08:00).
- Set yakin=false kalau tanggal atau jamnya nggak bisa ditentukan dari ucapan.
- Teks user berasal dari pengenalan suara, jadi mungkin ada salah dengar.
  Tebak maksud yang paling masuk akal, tapi jangan ngarang detail yang
  nggak disebut sama sekali."""


def minta_bikin_acara(teks: str) -> bool:
    """Ucapan ini niatnya bikin acara?"""
    bersih = re.sub(r"[^\w\s]", " ", teks.lower())
    return any(k in bersih for k in _NIAT)


def jawaban_ya(teks: str) -> bool | None:
    """True=setuju, False=tolak, None=nggak jelas.

    Ragu-ragu sengaja dibaca None, bukan True. Salah tebak di sini artinya
    nulis acara yang user nggak minta — arah salah yang paling mahal.
    """
    kata = re.sub(r"[^\w\s]", " ", teks.lower()).split()
    if not kata:
        return None
    # Cek penolakan duluan: "ya nggak usah" itu penolakan, bukan persetujuan
    if any(k in _TIDAK for k in kata):
        return False
    if any(k in _YA for k in kata):
        return True
    # "ya" cuma sah sebagai kata pertama ("ya simpan"), bukan ekor kalimat
    # tanya ("hmm apa ya")
    if kata[0] == "ya":
        return True
    return None


def urai(teks: str, oneshot) -> dict | None:
    """Ucapan -> dict acara. `oneshot(system, user) -> str` dari backend LLM."""
    tz = ZoneInfo(config.CALENDAR_TZ)
    sekarang = datetime.now(tz)
    konteks = (
        f"Sekarang {sekarang.strftime('%A, %Y-%m-%d, %H:%M')} "
        f"({config.CALENDAR_TZ}).\n\nUcapan user: {teks}"
    )

    mentah = oneshot(PROMPT, konteks, SKEMA)
    try:
        data = json.loads(mentah)
    except Exception:
        log.warning("hasil urai bukan JSON: %r", mentah)
        return None

    try:
        mulai = datetime.strptime(
            f"{data['tanggal']} {data['jam_mulai']}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=tz)
    except Exception:
        log.warning("tanggal/jam nggak kebaca: %r", data)
        return None

    durasi = int(data.get("durasi_menit") or 60)
    return {
        "judul": (data.get("judul") or "Acara").strip(),
        "mulai": mulai,
        "selesai": mulai + timedelta(minutes=max(5, durasi)),
        "lokasi": (data.get("lokasi") or "").strip(),
        "yakin": bool(data.get("yakin", True)),
    }


SKEMA_TUGAS = {
    "type": "object",
    "properties": {
        "judul": {"type": "string", "description": "Nama tugasnya, ringkas"},
        "matkul": {
            "type": "string",
            "description": "Kode mata kuliah kalau disebut, contoh COMP4020. Boleh kosong.",
        },
        "tenggat": {
            "type": "string",
            "description": "Tenggat format YYYY-MM-DD. String kosong kalau nggak disebut.",
        },
        "perkiraan_jam": {
            "type": "number",
            "description": "Perkiraan lama ngerjain dalam jam. 0 kalau nggak disebut.",
        },
    },
    "required": ["judul", "matkul", "tenggat", "perkiraan_jam"],
    "additionalProperties": False,
}

PROMPT_TUGAS = """Ubah permintaan user jadi satu tugas kuliah.

Aturan:
- Pakai tanggal sekarang buat ngartiin "Jumat", "minggu depan", "besok".
- Kalau tenggatnya nggak disebut sama sekali, isi tenggat dengan string kosong.
  JANGAN ngarang tanggal.
- Judulnya ringkas, jangan sertakan kata "tugas" kalau nggak perlu.
- Teks user berasal dari pengenalan suara, jadi mungkin ada salah dengar."""


def urai_tugas(teks: str, oneshot) -> dict | None:
    """Ucapan -> dict tugas."""
    tz = ZoneInfo(config.CALENDAR_TZ)
    sekarang = datetime.now(tz)
    konteks = (
        f"Sekarang {sekarang.strftime('%A, %Y-%m-%d, %H:%M')} "
        f"({config.CALENDAR_TZ}).\n\nUcapan user: {teks}"
    )
    try:
        data = json.loads(oneshot(PROMPT_TUGAS, konteks, SKEMA_TUGAS))
    except Exception:
        log.warning("hasil urai tugas nggak kebaca", exc_info=True)
        return None

    tenggat = (data.get("tenggat") or "").strip()
    if tenggat:
        try:
            datetime.strptime(tenggat, "%Y-%m-%d")
        except ValueError:
            log.warning("tenggat nggak valid: %r", tenggat)
            tenggat = ""

    judul = (data.get("judul") or "").strip()
    if not judul:
        return None
    return {
        "judul": judul,
        "matkul": (data.get("matkul") or "").strip(),
        "tenggat": tenggat,
        "perkiraan_jam": float(data.get("perkiraan_jam") or 0),
    }


HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def kalimat_konfirmasi(acara: dict) -> str:
    """Bacaan ulang buat dikonfirmasi user.

    Sengaja nyebut hari DAN tanggal: kalau agent salah denger "Kamis" jadi
    "Kemis" lalu meleset harinya, user langsung denger ketidakcocokannya.
    """
    m = acara["mulai"]
    tanggal = f"{HARI[m.weekday()]} {m.day} {BULAN[m.month]}"
    # Ditulis kata, bukan "7:44" — Piper bakal bacain titik dua itu apa adanya.
    # Tetep pakai format 24 jam: "jam 2" ambigu siang/malam, "jam 14" nggak.
    jam = f"jam {m.hour}" + (f" lewat {m.minute}" if m.minute else "")
    teks = f"{acara['judul']}, {tanggal}, {jam}"
    if acara["lokasi"]:
        teks += f", di {acara['lokasi']}"
    return f"Aku catat: {teks}. Bener?"

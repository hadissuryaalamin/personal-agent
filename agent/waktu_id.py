"""Urai frasa tanggal & jam Bahasa Indonesia secara pasti, tanpa LLM.

Kenapa nggak diserahin ke model: frasa tanggal itu himpunan tertutup yang kecil
dan aturannya kaku, sementara model kecil terbukti nggak bisa diandalkan buat
aritmetika tanggal — qwen2.5:7b cuma benar 2 dari 10, bahkan setelah dikasih
tabel tanggal siap pakai. "besok" dijawab lusa, "hari Jumat" dijawab Senin.

Yang deterministik dikerjain di sini; model cukup ngurus judul dan lokasi.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

HARI = {
    "senin": 0, "selasa": 1, "rabu": 2, "kamis": 3,
    "jumat": 4, "jum'at": 4, "sabtu": 5, "minggu": 6, "ahad": 6,
}

BULAN = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11,
    "desember": 12,
}

ANGKA = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
    "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11,
    "duabelas": 12, "dua belas": 12,
}


def _bersih(teks: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s':]", " ", teks.lower())).strip()


def cari_tanggal(teks: str, hari_ini: date) -> date | None:
    """Frasa tanggal -> tanggal. None kalau nggak ada yang cocok."""
    t = _bersih(teks)

    if re.search(r"\bhari ini\b|\bsekarang\b", t):
        return hari_ini
    # "besok lusa" harus dicek sebelum "besok", biar nggak kebaca +1
    if re.search(r"\blusa\b|\bbesok lusa\b", t):
        return hari_ini + timedelta(days=2)
    if re.search(r"\bbesok\b|\besok\b", t):
        return hari_ini + timedelta(days=1)

    # "3 hari lagi", "dua minggu lagi"
    m = re.search(r"\b(\d+|\w+)\s+(hari|minggu)\s+lagi\b", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else ANGKA.get(m.group(1))
        if n:
            return hari_ini + timedelta(days=n * (7 if m.group(2) == "minggu" else 1))

    # "tanggal 14", "tanggal 14 Agustus"
    m = re.search(r"\btanggal\s+(\d{1,2})(?:\s+(\w+))?", t)
    if m:
        hari = int(m.group(1))
        bulan = BULAN.get(m.group(2) or "", hari_ini.month)
        tahun = hari_ini.year
        try:
            d = date(tahun, bulan, hari)
        except ValueError:
            return None
        # Tanggal yang udah lewat dianggap bulan/tahun berikutnya
        if d < hari_ini:
            d = date(tahun + 1, bulan, hari) if m.group(2) else (
                date(tahun + (bulan == 12), bulan % 12 + 1, hari)
            )
        return d

    # Nama hari: "hari Jumat", "Jumat depan", "minggu depan hari Senin"
    for nama, wd in HARI.items():
        if not re.search(rf"\b{re.escape(nama)}\b", t):
            continue
        maju = (wd - hari_ini.weekday()) % 7
        if maju == 0:
            maju = 7  # "hari Senin" pas hari Senin = Senin depan
        d = hari_ini + timedelta(days=maju)
        if re.search(r"\bminggu depan\b|\bdepan\b", t) and maju < 7:
            # "Jumat depan" pas belum Jumat = Jumat minggu berikutnya
            if re.search(r"\bminggu depan\b", t):
                d += timedelta(days=7)
        return d

    return None


def cari_jam(teks: str) -> tuple[int, int] | None:
    """Frasa jam -> (jam, menit) 24 jam. None kalau nggak ada."""
    t = _bersih(teks)

    keterangan = None
    if re.search(r"\bpagi\b", t):
        keterangan = "pagi"
    elif re.search(r"\bsiang\b", t):
        keterangan = "siang"
    elif re.search(r"\bsore\b|\bpetang\b", t):
        keterangan = "sore"
    elif re.search(r"\bmalam\b", t):
        keterangan = "malam"

    jam = menit = None

    # "setengah 3" = 02:30
    m = re.search(r"\bsetengah\s+(\d{1,2}|\w+)\b", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else ANGKA.get(m.group(1))
        if n:
            jam, menit = (n - 1) % 24, 30

    if jam is None:
        # "jam 14:30", "jam 9", "jam sembilan", "jam 9 lewat 15"
        m = re.search(r"\bjam\s+(\d{1,2})[:.](\d{2})\b", t)
        if m:
            jam, menit = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r"\bjam\s+(\d{1,2}|\w+)(?:\s+lewat\s+(\d{1,2}|\w+))?", t)
            if m:
                jam = int(m.group(1)) if m.group(1).isdigit() else ANGKA.get(m.group(1))
                if m.group(2):
                    menit = (
                        int(m.group(2)) if m.group(2).isdigit()
                        else ANGKA.get(m.group(2))
                    )

    if jam is None:
        return None
    menit = menit or 0

    # Keterangan waktu menang atas angka mentah: "jam 4 sore" = 16.
    # Jam 12 itu kasus khusus di dua arah: "12 siang" = 12, "12 malam" = 0.
    if keterangan in ("siang", "sore") and jam < 12:
        jam += 12
    elif keterangan == "malam":
        jam = 0 if jam == 12 else (jam + 12) % 24 if jam < 12 else jam
    elif keterangan == "pagi" and jam == 12:
        jam = 0

    if not (0 <= jam <= 23 and 0 <= menit <= 59):
        return None
    return jam, menit


def urai(teks: str, sekarang: datetime) -> datetime | None:
    """Frasa lengkap -> datetime. None kalau tanggal atau jamnya nggak ketemu."""
    tgl = cari_tanggal(teks, sekarang.date())
    jm = cari_jam(teks)
    if tgl is None or jm is None:
        return None
    return datetime(tgl.year, tgl.month, tgl.day, jm[0], jm[1], tzinfo=sekarang.tzinfo)

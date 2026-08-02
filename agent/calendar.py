"""Baca jadwal dari link ICS (read-only).

Sengaja lewat ICS, bukan API: nggak perlu OAuth, nggak perlu bikin project di
cloud console, dan agent nggak punya izin nulis ke kalendermu. Konsekuensinya
memang nggak bisa bikin acara — itu butuh API sungguhan.

Hasilnya diselipin ke system prompt sebagai teks agenda, jadi agent bisa jawab
"besok ada apa" tanpa perlu ronde tool-call tambahan (yang bakal nambah jeda).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: list[dict] | None = None
_cache_time = 0.0
_refresh_jalan = False

HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

# Kode jenis kelas di feed ANU: LecA, TutA, ComA, ...
JENIS = {
    "lec": "kuliah",
    "tut": "tutorial",
    "com": "lab komputer",
    "lab": "lab",
    "wor": "workshop",
    "sem": "seminar",
    "dro": "sesi konsultasi",
}


def _cache_path():
    return config.MEMORY_DIR / "calendar.ics"


def _bersihin_judul(summary: str) -> str:
    """'Advanced Topics in Human-Centr_Agentic Coding Studio (Class: 9056), LecA'
    -> 'Agentic Coding Studio'

    Format ANU: JudulUmum_NamaSpesifik. Yang umum ('Advanced Topics in AI')
    dibuang karena kode matkulnya udah nunjukin itu, dan ANU sering motong
    judulnya di tengah ('...Human-Centr'). Nomor kelas & kode jenis juga
    dibuang — nggak ada gunanya pas dibacakan.
    """
    teks = re.sub(r"\(Class:\s*\d+\)", "", summary)
    # Backslash-nya buat jaga-jaga: kalau teksnya belum lewat pelepas escape
    # ICS, koma pemisah masih ketulis '\,'
    teks = re.sub(r"\\?,\s*(Lec|Tut|Com|Lab|Wor|Sem|Dro)\w*\s*$", "", teks.strip())
    if "_" in teks:
        bagian = [b.strip() for b in teks.split("_") if b.strip()]
        if bagian:
            teks = bagian[-1]
    return re.sub(r"[\s,]+$", "", re.sub(r"\s+", " ", teks)).strip()


def _jenis_kelas(summary: str) -> str:
    m = re.search(r",\s*(Lec|Tut|Com|Lab|Wor|Sem|Dro)\w*\s*$", summary.strip())
    return JENIS.get(m.group(1).lower(), "") if m else ""


def _kode_matkul(description: str) -> str:
    m = re.match(r"\s*([A-Z]{4}\d{4})", description or "")
    return m.group(1) if m else ""


def _bersihin_lokasi(lokasi: str) -> str:
    """'Rm 2.02_Fulton Muir Bldg 95' -> 'Fulton Muir, Rm 2.02'

    Nama gedung didahulukan karena itu yang dicari orang duluan; nomor gedung
    dibuang karena nggak nolong pas didengar.
    """
    if not lokasi:
        return ""
    teks = re.sub(r"\s*Bldg\s*\d+\s*", "", lokasi).strip()
    bagian = [b.strip() for b in teks.split("_") if b.strip()]
    if len(bagian) == 2 and bagian[0].lower().startswith(("rm", "room")):
        bagian = [bagian[1], bagian[0]]
    return ", ".join(bagian)


def _ambil_ics() -> str:
    import requests

    resp = requests.get(config.CALENDAR_ICS_URL, timeout=config.CALENDAR_TIMEOUT)
    resp.raise_for_status()
    # Feed ANU nggak nyantumin charset, jadi requests nebak ISO-8859-1 dan
    # bikin apostrof jadi 'â€™'. ICS itu UTF-8 menurut RFC 5545.
    return resp.content.decode("utf-8")


def _parse(teks: str) -> list[dict]:
    from icalendar import Calendar

    tz = ZoneInfo(config.CALENDAR_TZ)
    hasil = []
    for komponen in Calendar.from_ical(teks).walk("VEVENT"):
        mulai = komponen.get("DTSTART")
        if mulai is None:
            continue
        mulai = mulai.dt
        selesai = komponen.get("DTEND")
        selesai = selesai.dt if selesai is not None else None

        # Acara seharian datang sebagai date, bukan datetime
        sepanjang_hari = not isinstance(mulai, datetime)
        if not sepanjang_hari:
            mulai = mulai.astimezone(tz)
            if isinstance(selesai, datetime):
                selesai = selesai.astimezone(tz)

        summary = str(komponen.get("SUMMARY", "")).strip()
        desc = str(komponen.get("DESCRIPTION", "")).strip()
        hasil.append(
            {
                "mulai": mulai,
                "selesai": selesai,
                "sepanjang_hari": sepanjang_hari,
                "judul": _bersihin_judul(summary),
                "jenis": _jenis_kelas(summary),
                "kode": _kode_matkul(desc),
                "lokasi": _bersihin_lokasi(str(komponen.get("LOCATION", "")).strip()),
            }
        )
    hasil.sort(key=lambda e: (e["mulai"].isoformat() if not e["sepanjang_hari"] else str(e["mulai"])))
    log.info("kalender: %d acara ke-parse", len(hasil))
    return hasil


def _muat_cache_disk() -> list[dict] | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        return _parse(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("cache kalender di disk rusak", exc_info=True)
        return None


def refresh(paksa: bool = False) -> None:
    """Tarik ulang dari server. Aman dipanggil dari thread mana pun."""
    global _cache, _cache_time
    if not config.CALENDAR_ICS_URL:
        return
    umur = time.monotonic() - _cache_time
    if not paksa and _cache is not None and umur < config.CALENDAR_CACHE_MINUTES * 60:
        return
    try:
        teks = _ambil_ics()
        acara = _parse(teks)
        with _lock:
            _cache = acara
            _cache_time = time.monotonic()
        # Simpan biar restart nggak perlu nunggu jaringan, dan tetep jalan offline
        config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(teks, encoding="utf-8")
    except Exception:
        log.warning("gagal ambil kalender (pakai cache lama kalau ada)", exc_info=True)


def _refresh_latar() -> None:
    """Segarkan di latar belakang — jangan bikin user nunggu jaringan."""
    global _refresh_jalan
    with _lock:
        if _refresh_jalan:
            return
        _refresh_jalan = True

    def _run():
        global _refresh_jalan
        try:
            refresh()
        finally:
            _refresh_jalan = False

    threading.Thread(target=_run, name="kalender", daemon=True).start()


def _acara() -> list[dict]:
    global _cache, _cache_time
    if _cache is None:
        with _lock:
            if _cache is None:
                dari_disk = _muat_cache_disk()
                if dari_disk is not None:
                    _cache = dari_disk
                    _cache_time = 0.0  # tetep picu penyegaran
    if _cache is None:
        refresh(paksa=True)  # belum ada apa-apa: terpaksa nunggu
    else:
        _refresh_latar()
    return _cache or []


def _label_hari(d: date, hari_ini: date) -> str:
    beda = (d - hari_ini).days
    nama = f"{HARI[d.weekday()]} {d.day} {BULAN[d.month]}"
    if beda == 0:
        return f"Hari ini ({nama})"
    if beda == 1:
        return f"Besok ({nama})"
    if beda == 2:
        return f"Lusa ({nama})"
    return nama


def agenda() -> str:
    """Teks jadwal buat diselipin ke system prompt. Kosong kalau fitur dimatiin."""
    if not config.CALENDAR_ICS_URL:
        return ""

    tz = ZoneInfo(config.CALENDAR_TZ)
    sekarang = datetime.now(tz)
    hari_ini = sekarang.date()
    batas = hari_ini + timedelta(days=config.CALENDAR_DAYS_AHEAD)

    per_hari: dict[date, list[dict]] = {}
    for e in _acara():
        d = e["mulai"].date() if not e["sepanjang_hari"] else e["mulai"]
        if hari_ini <= d < batas:
            per_hari.setdefault(d, []).append(e)

    if not per_hari:
        return (
            f"Jadwal kuliah user {config.CALENDAR_DAYS_AHEAD} hari ke depan: "
            "kosong, nggak ada kelas."
        )

    baris = [f"Jadwal kuliah user {config.CALENDAR_DAYS_AHEAD} hari ke depan."]

    # Hitung "kelas berikutnya" di sini, jangan diserahin ke model. Nyari acara
    # terdekat dari sekarang itu penalaran lintas-baris — persis hal yang bikin
    # model salah ambil data dari baris tetangga.
    berikutnya = None
    for e in _acara():
        if e["sepanjang_hari"]:
            continue
        if e["mulai"] > sekarang:
            berikutnya = e
            break
    if berikutnya is not None:
        baris.append("KELAS BERIKUTNYA: " + _satu_baris(berikutnya, hari_ini))

    baris.append("")
    baris.append("Format tiap baris: jam | kode & nama | jenis | lokasi")
    for d in sorted(per_hari):
        baris.append(_label_hari(d, hari_ini) + ":")
        for e in per_hari[d]:
            baris.append("  " + _satu_baris(e, hari_ini, dengan_hari=False))
    return "\n".join(baris)


def _satu_baris(e: dict, hari_ini: date, dengan_hari: bool = True) -> str:
    """Satu acara jadi satu baris berkolom.

    Kolomnya dipisah '|' dengan sengaja: tanpa batas yang jelas, model gampang
    ngambil nama matkul dari baris ini tapi lokasinya dari baris sebelahnya.
    """
    if e["sepanjang_hari"]:
        jam = "seharian"
    else:
        jam = e["mulai"].strftime("%H:%M")
        if isinstance(e["selesai"], datetime):
            jam += "-" + e["selesai"].strftime("%H:%M")

    nama = f"{e['kode']} {e['judul']}".strip() if e["kode"] else e["judul"]
    kolom = [jam, nama, e["jenis"] or "-", e["lokasi"] or "lokasi tidak tercantum"]
    teks = " | ".join(kolom)

    if dengan_hari:
        d = e["mulai"].date() if not e["sepanjang_hari"] else e["mulai"]
        teks = f"{_label_hari(d, hari_ini)} | {teks}"
    return teks


def konteks_waktu() -> str:
    """Tanggal & jam sekarang. Model nggak punya jam — tanpa ini dia nggak bisa
    ngartiin 'besok' atau 'minggu depan'."""
    tz = ZoneInfo(config.CALENDAR_TZ)
    n = datetime.now(tz)
    return (
        f"Sekarang {HARI[n.weekday()]}, {n.day} {BULAN[n.month]} {n.year}, "
        f"jam {n.strftime('%H:%M')} waktu {config.CALENDAR_TZ}."
    )

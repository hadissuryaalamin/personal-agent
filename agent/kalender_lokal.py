"""Kalender lokal: satu file .ics yang dibaca dan ditulisi agent.

Mandiri penuh — nol jaringan, nol akun, jalan walau internet mati. Bayarannya
jadwalnya cuma kelihatan lewat agent, dan kalau filenya kehapus jadwalnya
hilang. Filenya format .ics standar, jadi tetap bisa diimpor balik ke Google
atau Outlook kapan pun.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()


def aktif() -> bool:
    return bool(config.CALENDAR_ICS_FILE) and config.CALENDAR_ICS_FILE.exists()


def _baca_kalender():
    """Muat file jadi objek Calendar. Bikin baru kalau belum ada."""
    from icalendar import Calendar

    path = config.CALENDAR_ICS_FILE
    if path.exists():
        try:
            return Calendar.from_ical(path.read_bytes())
        except Exception:
            log.warning("kalender lokal rusak, dibikin baru", exc_info=True)

    kal = Calendar()
    kal.add("prodid", "-//personal-agent//kalender lokal//ID")
    kal.add("version", "2.0")
    return kal


def teks() -> str:
    """Isi file mentah, buat dilempar ke parser di calendar.py."""
    if not aktif():
        return ""
    try:
        return config.CALENDAR_ICS_FILE.read_bytes().decode("utf-8")
    except Exception:
        log.warning("kalender lokal nggak kebaca", exc_info=True)
        return ""


def bikin_acara(
    judul: str,
    mulai: datetime,
    selesai: datetime,
    lokasi: str = "",
    catatan: str = "",
) -> dict:
    """Tambah acara ke file. Ditulis atomik biar file lama tetep utuh kalau
    prosesnya mati di tengah nulis."""
    import uuid

    from icalendar import Event

    with _lock:
        kal = _baca_kalender()

        ev = Event()
        ev.add("summary", judul)
        ev.add("dtstart", mulai)
        ev.add("dtend", selesai)
        ev.add("dtstamp", datetime.now(mulai.tzinfo))
        uid = f"{uuid.uuid4()}@personal-agent"
        ev.add("uid", uid)
        if lokasi:
            ev.add("location", lokasi)
        if catatan:
            ev.add("description", catatan)
        kal.add_component(ev)

        path = config.CALENDAR_ICS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".ics.tmp")
        tmp.write_bytes(kal.to_ical())
        os.replace(tmp, path)

    log.info("acara ditulis ke kalender lokal: %s @ %s (uid=%s)", judul, mulai, uid)
    return {"id": uid, "judul": judul, "mulai": mulai}


def jumlah_acara() -> int:
    if not aktif():
        return 0
    try:
        from icalendar import Calendar

        return len(list(Calendar.from_ical(config.CALENDAR_ICS_FILE.read_bytes()).walk("VEVENT")))
    except Exception:
        return 0

"""Google Calendar: baca acara & bikin acara baru.

Scope-nya sengaja cuma `calendar.events` — agent boleh baca dan bikin acara,
tapi nggak boleh hapus kalender atau ubah setelan berbagi. Izin sesempit yang
cukup buat kerjanya.

Login sekali lewat browser, habis itu token disimpan dan diperbarui sendiri.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)

# Baca + bikin acara. BUKAN 'calendar' penuh yang juga ngasih izin hapus
# kalender dan ubah setelan berbagi.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

_service = None
_lock = threading.Lock()


def _token_path() -> Path:
    return config.MEMORY_DIR / "google_token.json"


def aktif() -> bool:
    """Siap dipakai TANPA login? Butuh file kredensial DAN token hasil login.

    Token ikut dicek dengan sengaja: agent jalan lewat pythonw tanpa jendela,
    jadi alur normal nggak boleh sampai memicu login browser. Kalau tokennya
    hilang, fitur kalender dimatiin dan user dikasih tau lewat log — bukan
    digantung nunggu browser yang mungkin nggak dia sadari.
    """
    return (
        bool(config.GOOGLE_CREDENTIALS_FILE)
        and Path(config.GOOGLE_CREDENTIALS_FILE).exists()
        and _token_path().exists()
    )


def _kredensial(boleh_login: bool):
    """Ambil kredensial yang valid.

    `boleh_login=False` (default di jalur normal) nggak akan pernah buka
    browser — mending gagal dengan pesan jelas daripada menggantung.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token = _token_path()
    creds = None
    if token.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token), SCOPES)
        except Exception:
            log.warning("token Google rusak, bakal minta login ulang", exc_info=True)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _simpan_token(creds)
            return creds
        except Exception:
            # Paling sering: consent screen masih status "Testing", yang bikin
            # refresh token mati tiap 7 hari. Perbaikannya di konsol Google,
            # bukan di sini — publish app-nya ke "In production".
            log.warning("gagal perbarui token Google, minta login ulang", exc_info=True)

    if not boleh_login:
        raise RuntimeError(
            "Belum login ke Google (atau tokennya kedaluwarsa). Jalanin: "
            "python scripts/login_google.py"
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        config.GOOGLE_CREDENTIALS_FILE, SCOPES
    )
    # Buka browser lokal. Cuma kejadian sekali (atau kalau token dicabut).
    creds = flow.run_local_server(port=0)
    _simpan_token(creds)
    return creds


def _simpan_token(creds) -> None:
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _token_path().write_text(creds.to_json(), encoding="utf-8")
    log.info("token Google disimpan")


def service(boleh_login: bool = False):
    global _service
    with _lock:
        if _service is None:
            from googleapiclient.discovery import build

            _service = build(
                "calendar",
                "v3",
                credentials=_kredensial(boleh_login),
                cache_discovery=False,
            )
            log.info("Google Calendar siap")
        return _service


# Jenis kelas yang ditempel di judul waktu jadwal ICS disalin ke Google.
_JENIS_DIKENAL = {"kuliah", "tutorial", "lab komputer", "lab", "workshop", "seminar", "sesi konsultasi"}


def _pisah_jenis(judul: str) -> tuple[str, str]:
    """'COMP4020 Agentic Coding Studio (kuliah)' -> ('COMP4020 ...', 'kuliah').

    Cuma memisah kalau isi kurungnya memang nama jenis kelas — biar acara
    pribadi kayak 'Makan siang (sama Budi)' nggak ikut terpotong.
    """
    import re

    m = re.match(r"^(.*)\s*\(([^)]+)\)\s*$", judul)
    if m and m.group(2).strip().lower() in _JENIS_DIKENAL:
        return m.group(1).strip(), m.group(2).strip().lower()
    return judul, ""


def ambil_acara(hari_ke_depan: int) -> list[dict]:
    """Acara dari sekarang sampai N hari ke depan, bentuknya sama kayak calendar.py."""
    tz = ZoneInfo(config.CALENDAR_TZ)
    sekarang = datetime.now(tz)
    mulai = sekarang.replace(hour=0, minute=0, second=0, microsecond=0)
    akhir = mulai + timedelta(days=hari_ke_depan)

    hasil = (
        service()
        .events()
        .list(
            calendarId=config.GOOGLE_CALENDAR_ID,
            timeMin=mulai.isoformat(),
            timeMax=akhir.isoformat(),
            singleEvents=True,  # jabarin acara berulang jadi satuan
            orderBy="startTime",
            maxResults=250,
        )
        .execute()
    )

    acara = []
    for item in hasil.get("items", []):
        s, e = item.get("start", {}), item.get("end", {})
        sepanjang_hari = "date" in s
        if sepanjang_hari:
            mulai_dt = datetime.fromisoformat(s["date"]).date()
            selesai_dt = None
        else:
            mulai_dt = datetime.fromisoformat(s["dateTime"]).astimezone(tz)
            selesai_dt = (
                datetime.fromisoformat(e["dateTime"]).astimezone(tz)
                if e.get("dateTime")
                else None
            )
        judul, jenis = _pisah_jenis((item.get("summary") or "(tanpa judul)").strip())
        acara.append(
            {
                "mulai": mulai_dt,
                "selesai": selesai_dt,
                "sepanjang_hari": sepanjang_hari,
                "judul": judul,
                "jenis": jenis,
                "kode": "",
                "lokasi": (item.get("location") or "").strip(),
            }
        )
    log.info("Google Calendar: %d acara", len(acara))
    return acara


def bikin_acara(
    judul: str,
    mulai: datetime,
    selesai: datetime,
    lokasi: str = "",
    catatan: str = "",
) -> dict:
    """Bikin acara. Balikin dict berisi id & htmlLink."""
    body = {
        "summary": judul,
        "start": {"dateTime": mulai.isoformat(), "timeZone": config.CALENDAR_TZ},
        "end": {"dateTime": selesai.isoformat(), "timeZone": config.CALENDAR_TZ},
    }
    if lokasi:
        body["location"] = lokasi
    if catatan:
        body["description"] = catatan

    hasil = (
        service()
        .events()
        .insert(calendarId=config.GOOGLE_CALENDAR_ID, body=body)
        .execute()
    )
    log.info("acara dibikin: %s @ %s (id=%s)", judul, mulai, hasil.get("id"))
    return hasil

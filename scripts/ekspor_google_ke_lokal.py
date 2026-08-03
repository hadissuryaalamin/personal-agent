"""Salin seluruh isi Google Calendar ke satu file .ics lokal.

Sekali jalan, buat pindah dari Google ke kalender lokal yang mandiri.

Pemakaian:
  python scripts/ekspor_google_ke_lokal.py
  python scripts/ekspor_google_ke_lokal.py --mundur 60 --maju 365
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("googleapiclient").setLevel(logging.WARNING)

from datetime import datetime, timedelta  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from agent import config, gcal  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mundur", type=int, default=180, help="berapa hari ke belakang")
p.add_argument("--maju", type=int, default=365, help="berapa hari ke depan")
p.add_argument("--keluar", default=None, help="file tujuan")
arg = p.parse_args()

if not gcal.aktif():
    raise SystemExit("Google belum tersambung — nggak ada yang bisa diekspor")

tujuan = Path(arg.keluar) if arg.keluar else config.CALENDAR_ICS_FILE
tz = ZoneInfo(config.CALENDAR_TZ)
sekarang = datetime.now(tz)
mulai = sekarang - timedelta(days=arg.mundur)
akhir = sekarang + timedelta(days=arg.maju)

svc = gcal.service()
items, token = [], None
while True:
    resp = (
        svc.events()
        .list(
            calendarId=config.GOOGLE_CALENDAR_ID,
            timeMin=mulai.isoformat(),
            timeMax=akhir.isoformat(),
            singleEvents=True,  # acara berulang dijabarin jadi satuan
            orderBy="startTime",
            maxResults=2500,
            pageToken=token,
        )
        .execute()
    )
    items += resp.get("items", [])
    token = resp.get("nextPageToken")
    if not token:
        break

print(f"\n{len(items)} acara diambil ({mulai:%d %b %Y} sampai {akhir:%d %b %Y})")

from icalendar import Calendar, Event  # noqa: E402

kal = Calendar()
kal.add("prodid", "-//personal-agent//kalender lokal//ID")
kal.add("version", "2.0")

dilewat = 0
for it in items:
    s, e = it.get("start", {}), it.get("end", {})
    ev = Event()
    ev.add("summary", it.get("summary") or "(tanpa judul)")
    if it.get("location"):
        ev.add("location", it["location"])
    if it.get("description"):
        ev.add("description", it["description"])
    try:
        if "date" in s:  # acara seharian
            ev.add("dtstart", datetime.fromisoformat(s["date"]).date())
            if e.get("date"):
                ev.add("dtend", datetime.fromisoformat(e["date"]).date())
        else:
            ev.add("dtstart", datetime.fromisoformat(s["dateTime"]).astimezone(tz))
            if e.get("dateTime"):
                ev.add("dtend", datetime.fromisoformat(e["dateTime"]).astimezone(tz))
    except Exception:
        dilewat += 1
        continue
    # UID dipertahankan biar impor ulang nggak bikin duplikat
    ev.add("uid", it.get("iCalUID") or it["id"])
    kal.add_component(ev)

tujuan.parent.mkdir(parents=True, exist_ok=True)
tujuan.write_bytes(kal.to_ical())

print(f"Ditulis ke: {tujuan}")
print(f"  {len(kal.subcomponents)} acara" + (f", {dilewat} dilewat" if dilewat else ""))
print(f"  ukuran: {tujuan.stat().st_size / 1024:.1f} KB")
print("\nLangkah berikutnya: kosongin GOOGLE_CREDENTIALS_FILE di .env")

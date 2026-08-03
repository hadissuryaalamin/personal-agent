"""Login sekali ke Google Calendar (buka browser).

Dipisah dari agent dengan sengaja: agent jalan lewat pythonw tanpa jendela,
jadi alur normalnya nggak boleh sampai memicu login browser — kalau tokennya
hilang di tengah pemakaian, agent bakal kelihatan menggantung tanpa sebab.

Jalanin ini kalau:
- baru pertama kali menyiapkan Google Calendar
- log agent bilang "Belum login ke Google"
- kamu mencabut akses lalu mau menyambungkan lagi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from agent import config, gcal  # noqa: E402

if not config.GOOGLE_CREDENTIALS_FILE:
    raise SystemExit("GOOGLE_CREDENTIALS_FILE belum diisi di .env")
if not Path(config.GOOGLE_CREDENTIALS_FILE).exists():
    raise SystemExit(f"File kredensial nggak ketemu: {config.GOOGLE_CREDENTIALS_FILE}")

print("\nBrowser akan terbuka. Selesaikan sekarang — tautannya cepat kedaluwarsa.")
print("Kalau muncul 'Google hasn't verified this app':")
print("  Advanced -> Go to (nama app)\n")

svc = gcal.service(boleh_login=True)
info = svc.calendarList().get(calendarId=config.GOOGLE_CALENDAR_ID).execute()

print(f"\nBerhasil. Tersambung ke kalender: {info.get('summary')}")
print(f"Token disimpan di: {config.MEMORY_DIR / 'google_token.json'}")

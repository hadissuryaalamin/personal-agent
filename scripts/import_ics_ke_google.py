"""Salin acara dari feed ICS ke Google Calendar.

Dipakai buat mindahin jadwal kuliah ANU ke Google, supaya Google jadi satu-
satunya sumber (dan jadwalnya kelihatan di aplikasi Google Calendar di HP).

Dua pengaman:

- **Idempoten.** Tiap acara dikirim pakai iCalUID tetap lewat events.import().
  Dijalanin dua kali nggak bikin duplikat — yang kedua cuma memperbarui.
- **Bisa dibatalkan.** Tiap acara ditandai extendedProperties private
  source=ics-import, jadi bisa dihapus massal pakai --hapus.

Pemakaian:
  python scripts/import_ics_ke_google.py            # salin
  python scripts/import_ics_ke_google.py --lihat    # lihat aja, nggak nulis
  python scripts/import_ics_ke_google.py --hapus    # hapus semua hasil salinan
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("googleapiclient").setLevel(logging.WARNING)

from agent import calendar, config, gcal  # noqa: E402

PENANDA = "ics-import"


def _uid(e: dict, urutan: int) -> str:
    """ID tetap per acara. Google minta huruf kecil & angka aja.

    Dibentuk dari waktu mulai + judul supaya acara yang sama selalu dapat ID
    yang sama, walau urutan di feed berubah.
    """
    import hashlib

    inti = f"{e['mulai'].isoformat()}|{e['judul']}|{e['lokasi']}"
    return hashlib.sha1(inti.encode()).hexdigest() + "@personal-agent"


def salin(lihat_saja: bool) -> None:
    acara = calendar._acara()
    print(f"\n{len(acara)} acara di feed ICS")
    if not acara:
        print("Kosong — cek CALENDAR_ICS_URL di .env")
        return

    svc = gcal.service()
    dibuat = 0
    for i, e in enumerate(acara):
        if e["sepanjang_hari"]:
            body_waktu = {
                "start": {"date": str(e["mulai"])},
                "end": {"date": str(e["mulai"])},
            }
        else:
            body_waktu = {
                "start": {
                    "dateTime": e["mulai"].isoformat(),
                    "timeZone": config.CALENDAR_TZ,
                },
                "end": {
                    "dateTime": (e["selesai"] or e["mulai"]).isoformat(),
                    "timeZone": config.CALENDAR_TZ,
                },
            }

        judul = f"{e['kode']} {e['judul']}".strip() if e["kode"] else e["judul"]
        if e["jenis"]:
            judul += f" ({e['jenis']})"

        body = {
            "iCalUID": _uid(e, i),
            "summary": judul,
            "location": e["lokasi"],
            "extendedProperties": {"private": {"source": PENANDA}},
            **body_waktu,
        }

        if lihat_saja:
            if i < 5 or i == len(acara) - 1:
                waktu = (
                    e["mulai"].strftime("%a %d %b %H:%M")
                    if not e["sepanjang_hari"]
                    else str(e["mulai"])
                )
                print(f"  akan dibuat: {waktu}  {judul}  @ {e['lokasi']}")
            elif i == 5:
                print(f"  ... ({len(acara) - 6} lainnya)")
            continue

        svc.events().import_(
            calendarId=config.GOOGLE_CALENDAR_ID, body=body
        ).execute()
        dibuat += 1
        if dibuat % 20 == 0:
            print(f"  {dibuat}/{len(acara)}...")

    if lihat_saja:
        print("\n(--lihat: nggak ada yang ditulis)")
    else:
        print(f"\nSelesai: {dibuat} acara masuk ke Google Calendar")
        print("Batalin dengan: python scripts/import_ics_ke_google.py --hapus")


def hapus() -> None:
    """Hapus semua acara yang berasal dari skrip ini."""
    svc = gcal.service()
    total, token = 0, None
    while True:
        resp = (
            svc.events()
            .list(
                calendarId=config.GOOGLE_CALENDAR_ID,
                privateExtendedProperty=f"source={PENANDA}",
                maxResults=250,
                pageToken=token,
                showDeleted=False,
            )
            .execute()
        )
        for item in resp.get("items", []):
            svc.events().delete(
                calendarId=config.GOOGLE_CALENDAR_ID, eventId=item["id"]
            ).execute()
            total += 1
        token = resp.get("nextPageToken")
        if not token:
            break
    print(f"{total} acara hasil salinan dihapus")


if __name__ == "__main__":
    if not gcal.aktif():
        raise SystemExit("GOOGLE_CREDENTIALS_FILE belum diisi di .env")
    if "--hapus" in sys.argv:
        hapus()
    else:
        salin(lihat_saja="--lihat" in sys.argv)

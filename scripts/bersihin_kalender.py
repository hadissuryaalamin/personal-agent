"""Buang acara non-kuliah dari .ics lokal, sisain jadwal kuliah aja.

Acara dianggap kuliah kalau punya **kode matkul** (COMP4620, ENGN4122, ...)
atau **jenis kelas** (lecture, tutorial, computer lab, ...). Klasifikasinya
minjam fungsi yang sama yang dipakai agent buat baca kalender, jadi nggak ada
aturan kembar yang bisa beda hasil.

Selalu bikin cadangan. Selalu tulis atomik — kalau mati di tengah, .ics-nya
nggak jadi separuh.

    python scripts\\bersihin_kalender.py --lihat     # lihat aja, nggak nulis
    python scripts\\bersihin_kalender.py             # beneran hapus
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from agent.calendar import _jenis_kelas, _kode_matkul  # noqa: E402


def kuliah(komponen) -> bool:
    summary = str(komponen.get("SUMMARY", "")).strip()
    desc = str(komponen.get("DESCRIPTION", "")).strip()
    return bool(_kode_matkul(desc)) or bool(_jenis_kelas(summary))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lihat", action="store_true", help="tampilin aja, jangan nulis")
    args = ap.parse_args()

    from icalendar import Calendar

    path = config.CALENDAR_ICS_FILE
    if not path.exists():
        print(f"nggak ada: {path}")
        return 1

    kal = Calendar.from_ical(path.read_text(encoding="utf-8"))
    acara = kal.walk("VEVENT")
    simpan = [e for e in acara if kuliah(e)]
    buang = [e for e in acara if not kuliah(e)]

    print(f"{len(acara)} acara: {len(simpan)} kuliah disimpan, {len(buang)} dihapus\n")
    for e in sorted(buang, key=lambda e: str(e.get("DTSTART").dt)):
        print(f"  hapus  {str(e.get('DTSTART').dt)[:16]}  {e.get('SUMMARY')}")

    if args.lihat:
        print("\n(--lihat: nggak ada yang ditulis)")
        return 0
    if not buang:
        print("\nnggak ada yang perlu dihapus")
        return 0

    cadangan = path.with_suffix(".ics.sebelum-bersih")
    shutil.copy2(path, cadangan)

    # Bikin kalender baru dengan properti tingkat atas yang sama, isinya cuma
    # yang disimpan. Nyalin propertinya penting — TZID & PRODID ikut di situ.
    baru = Calendar()
    for k, v in kal.items():
        baru.add(k, v)
    for sub in kal.subcomponents:
        if sub.name != "VEVENT" or kuliah(sub):
            baru.add_component(sub)

    tmp = path.with_suffix(".ics.tmp")
    tmp.write_bytes(baru.to_ical())
    tmp.replace(path)

    print(f"\ncadangan: {cadangan.name}")
    print(f"sisa    : {len(simpan)} acara kuliah")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

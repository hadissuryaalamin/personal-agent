"""Ganti label jenis kelas berbahasa Indonesia di .ics lokal jadi Inggris.

Cuma menyentuh label yang dulu ditempel skrip ekspor — '(kuliah)',
'(lab komputer)', dst. Judul acara buatan user sendiri nggak disentuh sama
sekali, termasuk yang berbahasa Indonesia: itu isi, bukan label sistem.

Jalanin sekali habis pindah ke Inggris:
    .venv-agent\\Scripts\\python.exe scripts\\migrasi_jenis_ics.py
"""

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config
from agent.calendar import LEGACY_KINDS


def migrasi(teks: str) -> tuple[str, dict[str, int]]:
    hitung: dict[str, int] = {}

    def ganti(m: re.Match) -> str:
        awal, isi = m.group(1), m.group(2).strip()
        baru = LEGACY_KINDS.get(isi.lower())
        # Yang bukan nama jenis dibiarkan: 'Makan siang (sama Budi)' bukan
        # label sistem, dan bahasa Inggris yang udah bener juga nggak perlu
        # disentuh dua kali.
        if not baru or baru == isi:
            return m.group(0)
        hitung[f"{isi} -> {baru}"] = hitung.get(f"{isi} -> {baru}", 0) + 1
        return f"{awal}({baru})"

    return re.sub(r"(?m)^(SUMMARY:.*?)\(([^)\r\n]+)\)\s*$", ganti, teks), hitung


def main() -> int:
    path = config.MEMORY_DIR / "kalender.ics"
    if not path.exists():
        print(f"nggak ada: {path}")
        return 1

    asli = path.read_text(encoding="utf-8")
    baru, hitung = migrasi(asli)

    if not hitung:
        print("nggak ada label lama, file udah bersih")
        return 0

    cadangan = path.with_suffix(".ics.sebelum-inggris")
    shutil.copy2(path, cadangan)

    # Tulis lewat file sementara: kalau mati di tengah, .ics-nya nggak jadi
    # separuh — sama alasannya kayak penulisan di kalender_lokal.py.
    tmp = path.with_suffix(".ics.tmp")
    tmp.write_text(baru, encoding="utf-8")
    tmp.replace(path)

    print(f"cadangan: {cadangan.name}")
    for k, n in sorted(hitung.items()):
        print(f"  {n:3d}x  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

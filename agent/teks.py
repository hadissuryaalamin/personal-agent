"""Penormal teks buat semua pencocokan niat.

Ada di satu tempat dengan sengaja. Waktu tiap modul nulis normalisasinya
sendiri, apostrofnya diganti spasi — dan `"don't save it"` pecah jadi
`don | t | save | it`, nggak ada yang cocok sama daftar penolakan, terus kata
`save` ketangkep sebagai persetujuan. Jadi agent **nyimpen acara padahal user
bilang jangan**, persis kebalikan dari yang diminta.

Aturannya: apostrof DIBUANG (nyambung), tanda baca lain jadi SPASI.

    "don't save it"  -> "dont save it"
    "I'm done"       -> "im done"
    "that's all"     -> "thats all"
    "3 p.m., okay?"  -> "3 p m okay"
"""

import re

_APOSTROF = re.compile(r"['’ʼ]")
_LAINNYA = re.compile(r"[^\w\s]")
_SPASI = re.compile(r"\s+")


def normal(text: str) -> str:
    """Huruf kecil, tanpa tanda baca, spasi tunggal, apostrof disambung."""
    t = _APOSTROF.sub("", text.lower())
    t = _LAINNYA.sub(" ", t)
    return _SPASI.sub(" ", t).strip()


def kata(text: str) -> list[str]:
    """`normal()` yang udah dipecah jadi daftar kata."""
    t = normal(text)
    return t.split() if t else []


if __name__ == "__main__":
    KASUS = [
        ("don't save it", "dont save it"),
        ("I'm done", "im done"),
        ("that's all", "thats all"),
        ("Hello,  world!", "hello world"),
        ("3 p.m., okay?", "3 p m okay"),
        ("I’ve got a quiz", "ive got a quiz"),
        ("", ""),
        ("   ", ""),
    ]
    gagal = 0
    for masuk, mau in KASUS:
        dapat = normal(masuk)
        ok = dapat == mau
        gagal += not ok
        print(f"  {'ok   ' if ok else 'GAGAL'}  {masuk!r} -> {dapat!r} (mau {mau!r})")
    print(f"\n{len(KASUS) - gagal}/{len(KASUS)} lolos")
    raise SystemExit(1 if gagal else 0)

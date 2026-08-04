"""Jawaban yang dihitung Python, bukan dikarang model.

Pertanyaan jam, tanggal, dan jadwal itu **himpunan tertutup**: datanya sudah
pasti dan jawabannya cuma perlu dirakit. Nyerahin itu ke qwen bukan cuma
nambah 3-12 detik, tapi nambah cara buat salah. Terukur di proyek ini:

- Jam ditanya, dijawab model: 4/6 sampai 6/6 salah. Dia nggak baca terus
  lapor — dia parafrase jadi ucapan yang "enak" lalu **membulatkan**. Jam 20:12
  jadi "quarter past eight", "half past eight", "eight fifteen". Ngasih format
  yang lebih ramah ucapan malah bikin lebih parah (6/6), karena justru
  ngundang pembulatan.
- Label hari sesekali meleset: kelas hari ini disebut "tomorrow", padahal
  barisnya jelas ketulis TODAY.
- Balasannya 40 kata satu kalimat, dan karena TTS baru mulai setelah kalimat
  utuh, bunyi pertama ketunda ~12 detik.

Prinsipnya sama persis kayak `time_en.py`, yang udah lebih dulu ngambil alih
penguraian tanggal dari model (2/10 -> 5/5).

ATURAN PALING PENTING: kalau ragu, balikin None. Pertanyaan yang nggak
dikenali harus jatuh ke model, bukan dijawab asal. Salah rute di sini artinya
pertanyaan wajar dijawab kaku dan melenceng — lebih buruk daripada lambat.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from . import calendar, config, teks, tugas

log = logging.getLogger(__name__)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
SATUAN = [
    "twelve", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven",
]


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _jam_ucap(dt: datetime) -> str:
    """Jam -> kalimat yang enak diucapkan, TANPA pembulatan.

    Menitnya dieja apa adanya. Ini justru yang nggak bisa dilakuin model: dia
    selalu tergoda ngebulatin ke 'quarter past' atau 'half past'.
    """
    jam12 = SATUAN[dt.hour % 12]
    bagian = "in the morning" if dt.hour < 12 else (
        "in the afternoon" if dt.hour < 18 else "in the evening"
    )
    if dt.minute == 0:
        return f"{jam12} o'clock {bagian}"
    if dt.minute == 15:
        return f"quarter past {jam12} {bagian}"
    if dt.minute == 30:
        return f"half past {jam12} {bagian}"
    if dt.minute == 45:
        return f"quarter to {SATUAN[(dt.hour + 1) % 12]} {bagian}"
    return f"{jam12} {dt.minute:02d} {bagian}"


def _jam_acara(dt: datetime) -> str:
    """Jam acara, lebih ringkas — dipakai di tengah kalimat jadwal."""
    if dt.minute == 0:
        return SATUAN[dt.hour % 12]
    return f"{SATUAN[dt.hour % 12]} {dt.minute:02d}"


def _tanggal_ucap(d: date) -> str:
    return f"{DAYS[d.weekday()]} the {_ordinal(d.day)} of {MONTHS[d.month]}"


def _sekarang() -> datetime:
    return datetime.now(ZoneInfo(config.CALENDAR_TZ))


# --- Pengenalan niat --------------------------------------------------------
#
# Semuanya dicocokin sebagai FRASA UTUH di dalam kalimat, bukan per kata. Kata
# telanjang kayak "time" atau "class" muncul di terlalu banyak kalimat lain.

_JAM = (
    "what time is it", "whats the time", "what is the time", "time is it",
    "do you know what time", "got the time", "current time",
)
_PAGI_SORE = ("is it morning", "is it evening", "is it afternoon", "is it night")
_TANGGAL = (
    "what is the date", "whats the date", "what date is it", "todays date",
    "what is todays date", "what day is it", "what day is today",
)
# Jadwal dikenali dari DUA bagian, bukan hafalan frasa utuh: kata jadwal +
# kata hari. Daftar frasa utuh kelewat kaku — "what classes do I have today"
# nggak ngandung "classes today" karena ada "do i have" nyelip di tengah, dan
# ngelistin tiap susunan kalimat itu pertarungan yang nggak bakal menang.
_JADWAL_KATA = (
    "class", "classes", "lecture", "lectures", "tutorial", "tutorials",
    "schedule", "timetable", "lab",
)
# Pola "apa yang aku punya" tanpa nyebut kelas. Sengaja spesifik: kata
# telanjang kayak "have" atau "on" muncul di kelewat banyak kalimat lain.
_PUNYA = (
    "what do i have", "what have i got", "whats on", "what is on",
    "anything on", "whats happening", "what is happening", "anything happening",
)
_BERIKUTNYA = (
    "next class", "next lecture", "next tutorial", "coming up next",
    "whats next", "what is next", "next thing",
)
_TUGAS = (
    "what should i work on", "what do i need to do", "whats due",
    "what is due", "anything due", "what tasks", "my tasks", "todo list",
    "what should i do today",
)


def _cocok(t: str, frasa: tuple[str, ...]) -> bool:
    return any(f in t for f in frasa)


# --- Perakit jawaban --------------------------------------------------------


def _acara_pada(d: date) -> list[dict]:
    """Acara berjadwal (bukan sepanjang hari) di tanggal itu, terurut."""
    keluar = [
        e for e in calendar._acara()
        if not e.get("sepanjang_hari") and _tanggal_mulai(e) == d
    ]
    return sorted(keluar, key=lambda e: e["mulai"])


def _tanggal_mulai(e: dict) -> date:
    m = e["mulai"]
    return m.date() if isinstance(m, datetime) else m


def _sebut_acara(e: dict, dengan_lokasi: bool = True) -> str:
    nama = e.get("kode") or ""
    judul = e.get("judul") or "an event"
    label = f"{nama} {judul}".strip() if nama and nama not in judul else judul
    mulai, selesai = e["mulai"], e.get("selesai")

    bagian = label
    if isinstance(mulai, datetime):
        bagian += f" from {_jam_acara(mulai)}"
        if isinstance(selesai, datetime):
            bagian += f" to {_jam_acara(selesai)}"
    if dengan_lokasi and e.get("lokasi"):
        # Lokasi dipotong: sebagian acara pribadi punya alamat sepanjang paragraf,
        # dan itu nggak enak dibacakan.
        lok = " ".join(str(e["lokasi"]).split())
        if len(lok) <= 60:
            bagian += f", in {lok}"
    return bagian


def _rangkai(hal: list[str]) -> str:
    if len(hal) == 1:
        return hal[0]
    return ", ".join(hal[:-1]) + ", and " + hal[-1]


def _jawab_hari(d: date, label: str) -> str:
    """Tiap acara jadi KALIMAT sendiri, bukan satu kalimat panjang berkoma.

    Dua alasan, dua-duanya terukur. TTS baru mulai bunyi setelah satu kalimat
    utuh, jadi satu kalimat panjang nunda bunyi pertama — jawaban pasti sempat
    LEBIH LAMBAT dari model (6,26 vs 5,58 detik) justru gara-gara ini.

    Lokasi juga dibuang pas ndaftar lebih dari satu. Nyebutin ruangan tiap
    acara bikin jawabannya panjang banget padahal jarang dibutuhin sekaligus —
    kalau perlu, tinggal nanya "where is my next class".
    """
    isi = _acara_pada(d)
    if not isi:
        return f"Nothing scheduled {label}."
    if len(isi) == 1:
        return f"One thing {label}. {_sebut_acara(isi[0])}."

    bagian = [f"{len(isi)} things {label}."]
    for i, e in enumerate(isi):
        awal = "First, " if i == 0 else ("Then " if i < len(isi) - 1 else "Last, ")
        bagian.append(f"{awal}{_sebut_acara(e, dengan_lokasi=False)}.")
    return " ".join(bagian)


def _jawab_berikutnya(now: datetime) -> str:
    calon = [
        e for e in calendar._acara()
        if not e.get("sepanjang_hari")
        and isinstance(e["mulai"], datetime)
        and e["mulai"] > now
    ]
    if not calon:
        return "Nothing else scheduled."
    e = min(calon, key=lambda x: x["mulai"])
    d = _tanggal_mulai(e)
    beda = (d - now.date()).days
    kapan = "Today" if beda == 0 else ("Tomorrow" if beda == 1 else DAYS[d.weekday()])
    return f"{kapan}, {_sebut_acara(e)}."


def _jawab_tugas() -> str | None:
    isi = tugas.all_tasks()
    if not isi:
        return "Nothing on your task list."
    # Lebih dari itu diserahin ke model: milih mana yang paling mendesak butuh
    # nimbang tenggat, lama ngerjain, dan jadwal sekaligus — itu penilaian,
    # bukan perakitan.
    if len(isi) > 3:
        return None
    hari_ini = _sekarang().date()
    bagian = []
    for t in isi:
        s = t.get("title") or "a task"
        due = t.get("due")
        if due:
            try:
                d = datetime.strptime(due, "%Y-%m-%d").date()
                sisa = (d - hari_ini).days
                s += (
                    " due today" if sisa == 0
                    else " due tomorrow" if sisa == 1
                    else f" overdue by {abs(sisa)} days" if sisa < 0
                    else f" due in {sisa} days"
                )
            except ValueError:
                pass
        bagian.append(s)
    return _rangkai(bagian) + "."


# --- Pintu masuk ------------------------------------------------------------


def answer(text: str) -> str | None:
    """Jawaban siap ucap, atau None kalau pertanyaannya harus lewat model."""
    t = teks.normal(text)
    if not t:
        return None

    try:
        now = _sekarang()

        if _cocok(t, _JAM):
            return f"It's {_jam_ucap(now)}."

        if _cocok(t, _PAGI_SORE):
            return f"It's {_jam_ucap(now)}."

        if _cocok(t, _TANGGAL):
            return f"It's {_tanggal_ucap(now.date())}, {now.year}."

        # Tugas dicek SEBELUM jadwal. "what should I work on today" itu soal
        # tugas, tapi gampang ketangkep pola jadwal harian — dan yang lebih
        # spesifik harus menang. Urutan yang sama alasannya kayak di
        # _route_and_reply(): tugas selalu duluan.
        if _cocok(t, _TUGAS):
            return _jawab_tugas()

        soal_jadwal = _cocok(t, _JADWAL_KATA) or _cocok(t, _PUNYA)
        if soal_jadwal:
            # Besok sebelum hari ini: "anything today or tomorrow" ngandung
            # dua-duanya, dan yang lebih jauh ke depan lebih informatif.
            if "tomorrow" in t:
                return _jawab_hari(now.date() + timedelta(days=1), "tomorrow")
            if "today" in t:
                return _jawab_hari(now.date(), "today")

        if _cocok(t, _BERIKUTNYA):
            return _jawab_berikutnya(now)

    except Exception:
        # Jawaban pasti yang error jangan ngebunuh giliran — jatuhin ke model,
        # yang setidaknya bakal ngomong sesuatu.
        log.exception("gagal nyusun jawaban pasti, dilempar ke model")
        return None

    return None


if __name__ == "__main__":
    #     .venv-agent\Scripts\python.exe -m agent.jawab_pasti
    logging.basicConfig(level=logging.WARNING)

    print("=== jam & tanggal ===")
    for q in ("what time is it", "what's the time", "do you know what time it is",
              "is it morning or evening", "what's the date today", "what day is it"):
        print(f"  {q!r:38s} -> {answer(q)}")

    print("\n=== jadwal ===")
    for q in ("what classes do I have today", "what's on tomorrow",
              "when is my next class", "where is my next class",
              "what's my schedule today"):
        print(f"  {q!r:38s} -> {answer(q)}")

    print("\n=== tugas ===")
    for q in ("what should I work on today", "what's due"):
        print(f"  {q!r:38s} -> {answer(q)}")

    print("\n=== HARUS kejawab (bukan None) ===")
    HARUS = (
        "what time is it", "whats the time", "what's the date",
        "what day is it", "what classes do I have today",
        "what do I have tomorrow", "what's my schedule today",
        "any lectures tomorrow", "when is my next class",
        "what's on today", "is it morning or evening",
    )
    for q in HARUS:
        h = answer(q)
        print(f"  {'ok   ' if h else 'SALAH'}  {q!r:36s} -> {(h or '')[:60]}")

    print("\n=== HARUS None (lempar ke model) ===")
    for q in ("why is the sky blue", "tell me a joke",
              "how many hours until midnight", "explain my COMP4620 course",
              "should I study tonight", "what's the weather",
              "add a task to finish the essay", "how was your day",
              "remind me to call mum", "is the library open today",
              "what do you think about my schedule", "book a room for tomorrow"):
        h = answer(q)
        print(f"  {'ok   ' if h is None else 'SALAH'}  {q!r:36s} -> {h}")

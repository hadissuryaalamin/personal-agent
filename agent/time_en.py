"""Parse English date & time phrases deterministically, without an LLM.

Same reasoning as `waktu_id.py`: date phrases are a small closed set with rigid
rules, while small models proved unreliable at date arithmetic — qwen2.5:7b got
only 2 of 10 right even when handed a ready-made date table.

One thing this parser must handle that the Indonesian one didn't: Parakeet
normalizes spoken numbers, so "three PM" arrives as "3 p.m." and "the
fourteenth" as "the 14th". Both word and digit forms are accepted.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

DAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
    "a": 1, "an": 1, "couple": 2, "few": 3,
}


def _clean(text: str) -> str:
    """Lowercase, drop punctuation except the colon in times.

    'p.m.' becomes 'p m' after this, so the meridiem patterns below accept both
    that and the plain 'pm'.
    """
    t = re.sub(r"[^\w\s:]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def _number(word: str) -> int | None:
    if word.isdigit():
        return int(word)
    return NUMBERS.get(word)


def find_date(text: str, today: date) -> date | None:
    """Date phrase -> date. None when nothing matches."""
    t = _clean(text)

    if re.search(r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b", t):
        return today
    # 'day after tomorrow' must be checked before 'tomorrow'
    if re.search(r"\bday after tomorrow\b|\bovermorrow\b", t):
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b|\btmr\b", t):
        return today + timedelta(days=1)
    if re.search(r"\byesterday\b", t):
        return today - timedelta(days=1)

    # "in 3 days", "in a week", "in two weeks"
    m = re.search(r"\bin (\d+|\w+) (day|days|week|weeks)\b", t)
    if m:
        n = _number(m.group(1))
        if n:
            return today + timedelta(days=n * (7 if m.group(2).startswith("week") else 1))

    # "on the 14th", "on august 14", "on 14 august"
    m = re.search(r"\b(?:on )?(?:the )?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(\w+)\b", t)
    if m and m.group(2) in MONTHS:
        return _make(int(m.group(1)), MONTHS[m.group(2)], today)
    m = re.search(r"\b(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m and m.group(1) in MONTHS:
        return _make(int(m.group(2)), MONTHS[m.group(1)], today)
    m = re.search(r"\bon the (\d{1,2})(?:st|nd|rd|th)\b", t)
    if m:
        return _make(int(m.group(1)), today.month, today, bulan_disebut=False)

    # Weekday names: "on friday", "next friday", "this friday"
    for name, wd in DAYS.items():
        if not re.search(rf"\b{name}\b", t):
            continue
        ahead = (wd - today.weekday()) % 7
        if ahead == 0:
            ahead = 7  # "on monday" said on a Monday means next Monday
        d = today + timedelta(days=ahead)
        if re.search(rf"\bnext (?:week )?{name}\b|\b{name} next week\b", t):
            # "next friday" when today is before Friday means the following week
            if ahead < 7:
                d += timedelta(days=7)
        return d

    return None


def _make(day: int, month: int, today: date, bulan_disebut: bool = True) -> date | None:
    """Build a date, rolling forward when it has already passed."""
    try:
        d = date(today.year, month, day)
    except ValueError:
        return None
    if d < today:
        if bulan_disebut:
            try:
                d = date(today.year + 1, month, day)
            except ValueError:
                return None
        else:
            month_next = month % 12 + 1
            year = today.year + (month == 12)
            try:
                d = date(year, month_next, day)
            except ValueError:
                return None
    return d


def find_time(text: str) -> tuple[int, int] | None:
    """Time phrase -> (hour, minute) in 24h. None when nothing matches."""
    t = _clean(text)

    if re.search(r"\bnoon\b|\bmidday\b", t):
        return 12, 0
    if re.search(r"\bmidnight\b", t):
        return 0, 0

    # '_clean' turns 'p.m.' into 'p m', so both spellings are accepted
    meridiem = None
    if re.search(r"\bp\s?m\b", t):
        meridiem = "pm"
    elif re.search(r"\ba\s?m\b", t):
        meridiem = "am"
    elif re.search(r"\bevening\b|\bnight\b|\bafternoon\b", t):
        meridiem = "pm"
    elif re.search(r"\bmorning\b", t):
        meridiem = "am"

    hour = minute = None

    # "half past two", "quarter past three", "quarter to five"
    m = re.search(r"\b(half|quarter) (past|to) (\d{1,2}|\w+)\b", t)
    if m:
        n = _number(m.group(3))
        if n:
            if m.group(2) == "past":
                hour, minute = n, 30 if m.group(1) == "half" else 15
            else:
                hour, minute = (n - 1) % 24, 30 if m.group(1) == "half" else 45

    if hour is None:
        # "at 14:30", "at 3", "at three", "at 3 30"
        m = re.search(r"\bat (\d{1,2}):(\d{2})\b", t) or re.search(r"\b(\d{1,2}):(\d{2})\b", t)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r"\bat (\d{1,2}|\w+)(?:\s+(\d{2}))?\b", t)
            if m and _number(m.group(1)) is not None:
                hour = _number(m.group(1))
                minute = int(m.group(2)) if m.group(2) else None

    if hour is None:
        return None
    minute = minute or 0

    # 12 is the special case in both directions: 12 am = 0, 12 pm = 12
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def parse(text: str, now: datetime) -> datetime | None:
    """Full phrase -> datetime. None when either date or time is missing."""
    d = find_date(text, now.date())
    hm = find_time(text)
    if d is None or hm is None:
        return None
    return datetime(d.year, d.month, d.day, hm[0], hm[1], tzinfo=now.tzinfo)


if __name__ == "__main__":
    # Tes mandiri. Frasa waktu itu himpunan tertutup, jadi cukup diperiksa
    # dengan tabel — dan tabelnya perlu ikut ter-commit, bukan nyangkut di
    # skrip sekali pakai.
    #
    #     .venv-agent\Scripts\python.exe -m agent.time_en
    from datetime import timezone

    # Senin 3 Agustus 2026, 10:00
    NOW = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)

    TANGGAL = [
        ("today", "2026-08-03"),
        ("tomorrow", "2026-08-04"),
        ("day after tomorrow", "2026-08-05"),
        ("on friday", "2026-08-07"),
        ("next monday", "2026-08-10"),
        ("this wednesday", "2026-08-05"),
        ("in 3 days", "2026-08-06"),
        ("in two weeks", "2026-08-17"),
        ("on the 14th", "2026-08-14"),
        ("on september 14", "2026-09-14"),
        ("on 2 september", "2026-09-02"),
        ("no date here", None),
    ]

    JAM = [
        ("at 3 pm", (15, 0)),
        ("at 3 p.m.", (15, 0)),        # bentuk ternormalisasi dari Parakeet
        ("at 9 am", (9, 0)),
        ("at 9", (9, 0)),
        ("at 14:30", (14, 30)),
        ("at half past two", (2, 30)),
        ("at quarter to five", (4, 45)),
        ("at quarter past six", (6, 15)),
        ("at noon", (12, 0)),
        ("at midnight", (0, 0)),
        ("at 12 am", (0, 0)),          # tengah malam, bukan siang
        ("at 12 pm", (12, 0)),         # tengah hari, bukan malam
        ("at 3:05 pm", (15, 5)),
        ("no time here", None),
    ]

    gagal = 0
    for teks, mau in TANGGAL:
        dapat = find_date(teks, NOW.date())
        dapat_s = dapat.isoformat() if dapat else None
        ok = dapat_s == mau
        gagal += not ok
        print(f"  {'ok  ' if ok else 'GAGAL'}  date {teks!r} -> {dapat_s} (mau {mau})")

    for teks, mau in JAM:
        dapat = find_time(teks)
        ok = dapat == mau
        gagal += not ok
        print(f"  {'ok  ' if ok else 'GAGAL'}  time {teks!r} -> {dapat} (mau {mau})")

    # Gabungan: tanggal DAN jam harus dua-duanya ketemu.
    for teks, mau in [
        ("tomorrow at 3 pm", "2026-08-04T15:00"),
        # Tanpa am/pm, jam kecil dibaca apa adanya: "half past two" -> 02:30,
        # bukan 14:30. Sengaja TIDAK ditebak jadi sore — yang nangkep kekeliruan
        # ini adalah kalimat konfirmasi, yang membacakan jam dalam format 24 jam
        # ("two thirty" vs "fourteen thirty"), jadi user bisa bilang tidak.
        ("on friday at half past two", "2026-08-07T02:30"),
        ("tomorrow", None),            # jam nggak ada -> None, bukan tebakan
        ("at 3 pm", None),             # tanggal nggak ada -> None
    ]:
        dapat = parse(teks, NOW)
        dapat_s = dapat.strftime("%Y-%m-%dT%H:%M") if dapat else None
        ok = dapat_s == mau
        gagal += not ok
        print(f"  {'ok  ' if ok else 'GAGAL'}  parse {teks!r} -> {dapat_s} (mau {mau})")

    total = len(TANGGAL) + len(JAM) + 4
    print(f"\n{total - gagal}/{total} lolos")
    raise SystemExit(1 if gagal else 0)

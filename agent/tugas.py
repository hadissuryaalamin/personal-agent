"""Task list: store, mark done, and render for the prompt.

Kept in memory/tasks.json. Plain JSON, editable by hand.

Deliberately separate from the calendar: a task doesn't occupy a time slot, it
has a done/not-done state, and it can be worked on in pieces. Calendars have no
concept of any of that.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import date, datetime
from zoneinfo import ZoneInfo

from . import config, teks

log = logging.getLogger(__name__)

_lock = threading.Lock()

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Words marking this as a task rather than a calendar event. Checked FIRST,
# because "add a task ..." also matches the "add ..." pattern for events.
_TASK_WORDS = (
    "task", "assignment", "homework", "deadline", "due", "quiz", "exam",
    "report", "essay", "paper", "lab report", "problem set", "readings",
)
# Kata perintah nyatet. "write" sempat nggak ada, dan "Write an assignment on
# 14 August" jatuh ke model — yang bukannya nyatet, malah ngelaporin isi daftar
# tugas. Nggak ada satu kata baku buat ini; orang bilang add/write/note/jot,
# jadi daftarnya emang harus lebar.
_ADD_INTENT = (
    "add", "create", "new", "remind me", "note", "put", "log", "track",
    "write", "jot", "record", "save a", "save the", "enter",
)
# Spoken declarations carry no add verb — "I have an assignment due Monday" is
# how a new task actually arrives, far more often than "add a task ...".
# Bentuknya ngikutin teks.normal(): apostrof disambung, jadi "I've" -> "ive".
# Sempat ketulis "i ve got" (apostrof jadi spasi) dan nggak pernah cocok.
_ADD_DECLARE = (
    "i have", "ive got", "i got", "i need to", "i have to", "i must",
    "theres a", "there is a", "gotta", "coming up", "handed out",
)
_DONE_INTENT = (
    "done", "finished", "complete", "completed", "submitted", "handed in",
    "turned in", "wrapped up",
)


def _path():
    return config.MEMORY_DIR / "tasks.json"


def _load() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("tasks", [])
    except Exception:
        log.warning("task list unreadable", exc_info=True)
        return []


def _save(items: list[dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"tasks": items}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, p)


def all_tasks(include_done: bool = False) -> list[dict]:
    items = _load()
    return items if include_done else [t for t in items if not t.get("done")]


def add(title: str, due: str = "", course: str = "", estimate_hours: float = 0) -> dict:
    """Add a task. `due` is YYYY-MM-DD, may be empty."""
    t = {
        "title": title.strip(),
        "due": due,
        "course": course.strip(),
        "estimate_hours": estimate_hours,
        "done": False,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with _lock:
        items = _load()
        items.append(t)
        _save(items)
    log.info("task added: %r", t)
    return t


def mark(rough_title: str, done: bool = True) -> dict | None:
    """Mark a task done/undone by word overlap.

    Returns the task when exactly one matches, None when nothing matches or the
    match is ambiguous — so the agent can ask instead of guessing which one.
    """
    words = set(teks.kata(rough_title))
    with _lock:
        items = _load()
        scored = []
        for i, t in enumerate(items):
            if t.get("done") == done:
                continue
            text = set(
                re.sub(r"[^\w\s]", " ", f"{t['title']} {t.get('course','')}".lower()).split()
            )
            overlap = len(words & text)
            if overlap:
                scored.append((overlap, i))
        if not scored:
            return None
        scored.sort(reverse=True)
        # Ambiguous when two tasks tie for the best match
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            log.info("ambiguous task for %r", rough_title)
            return None
        idx = scored[0][1]
        items[idx]["done"] = done
        _save(items)
        log.info("task marked done=%s: %r", done, items[idx]["title"])
        return items[idx]


# --- Kelengkapan & pengisian bertahap ---------------------------------------
#
# Tugas nggak disimpen kalau belum lengkap. Terukur dari kejadian nyata:
# "can you add some assignment for uh fourteen deadline?" nyimpen
#   {'title': 'Assignment', 'due': '2026-08-19', 'course': '', 'estimate_hours': 0}
# Judulnya generik, tenggatnya SALAH (dibilang 14, disimpen 19), matkul & lama
# ngerjain kosong. Entri kayak gitu nggak bisa dipakai buat mutusin apa pun —
# dan diem-diem kesimpen tanpa dibacain ulang.
#
# Tiga field ini wajib karena itu yang dipakai buat mutusin "hari ini ngerjain
# apa": tenggat (mendesak apa nggak), matkul (punya siapa), perkiraan jam
# (muat nggak di sela kuliah).
WAJIB = ("due", "course", "estimate_hours")

_LABEL = {
    "due": "the deadline",
    "course": "which course it's for",
    "estimate_hours": "roughly how many hours it'll take",
}

_ANGKA_KATA = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "half": 0.5, "an": 1, "a": 1,
}

_KODE = re.compile(r"\b([a-z]{4})\s?(\d{4})\b", re.I)
_JAM = re.compile(
    r"\b(\d+(?:\.\d+)?|" + "|".join(_ANGKA_KATA) + r")\s*(?:and a half\s*)?h(?:ou)?rs?\b",
    re.I,
)


def kurang(t: dict) -> list[str]:
    """Field wajib yang masih kosong."""
    return [k for k in WAJIB if not t.get(k)]


def kalimat_kurang(hilang: list[str]) -> str:
    """'I still need the deadline and which course it's for.'"""
    label = [_LABEL[k] for k in hilang]
    if len(label) == 1:
        isi = label[0]
    elif len(label) == 2:
        isi = f"{label[0]} and {label[1]}"
    else:
        isi = ", ".join(label[:-1]) + f", and {label[-1]}"
    return f"I still need {isi}."


def ekstrak(text: str, hari_ini: date | None = None) -> dict:
    """Ambil potongan yang kebaca dari jawaban pendek. Yang nggak ada, nggak
    dimasukin — jadi aman buat nambal dict yang udah ada."""
    from . import time_en

    keluar: dict = {}

    m = _KODE.search(text)
    if m:
        keluar["course"] = f"{m.group(1).upper()}{m.group(2)}"

    m = _JAM.search(text)
    if m:
        angka = m.group(1).lower()
        jam = float(angka) if angka.replace(".", "", 1).isdigit() else _ANGKA_KATA.get(angka, 0)
        if "and a half" in text.lower():
            jam += 0.5
        if jam:
            keluar["estimate_hours"] = jam

    d = time_en.find_date(text, hari_ini or datetime.now(ZoneInfo(config.CALENDAR_TZ)).date())
    if d:
        keluar["due"] = d.isoformat()

    return keluar


def clear_all() -> None:
    with _lock:
        _save([])
    log.info("all tasks cleared")


# --- Intent detection ------------------------------------------------------


def _mentions_task(text: str) -> bool:
    t = " " + teks.normal(text) + " "
    return any(f" {k} " in t or k in t for k in _TASK_WORDS)


def _flat(text: str) -> str:
    return teks.normal(text)


# Penanda pertanyaan, dicek DI MANA PUN dalam kalimat.
#
# Dulu cuma dicek di awal kalimat, dan itu bocor tiga kali:
#   "I need to know my assignments this month"
#   "Can you show me uh how many assignments do I have in this month?"
# Dua-duanya nanya, dua-duanya ketulis jadi tugas — karena kata tanyanya
# ada di TENGAH, dan "i have"/"i need to" cocok sama pemicu deklarasi.
#
# Arahnya sengaja nggak simetris: gagal ngenalin tugas cuma bikin kamu
# ngulang sekali, sedangkan salah nulis ninggalin entri palsu yang baru
# ketahuan berminggu-minggu kemudian.
#
# Pakai batas kata (\b) — tanpa itu "how" nyangkut di "show", dan tiap
# kalimat yang ngandung "show" bakal dianggap pertanyaan.
_TANYA = re.compile(
    r"\b("
    r"what|when|where|which|who|why"
    r"|how many|how much|how long|how do|how is"
    r"|do i have|do i need|did i|have i"
    r"|is there|are there|am i"
    r"|tell me|show me|read me|list my|list the"
    r"|need to know|want to know|like to know|have to know"
    r"|wondering|curious|remind me of|remind me what"
    r"|anything|should i"
    r")\b"
)


def wants_add_task(text: str) -> bool:
    if not _mentions_task(text):
        return False
    t = _flat(text)
    if _TANYA.search(t):
        return False
    if any(k in t for k in _DONE_INTENT):
        return False  # that's a completion report, not a new task
    return any(k in t for k in _ADD_INTENT) or any(k in t for k in _ADD_DECLARE)


def wants_mark_done(text: str) -> bool:
    if not _mentions_task(text):
        return False
    t = _flat(text)
    if _TANYA.search(t):
        return False
    return any(k in t for k in _DONE_INTENT)


# --- Prompt rendering ------------------------------------------------------


def _due_label(due: str, today: date) -> str:
    """Days remaining computed here, not left to the model — same reason as
    NEXT CLASS: cross-row date arithmetic is where models slip."""
    try:
        d = datetime.strptime(due, "%Y-%m-%d").date()
    except Exception:
        return ""
    left = (d - today).days
    when = f"{DAYS[d.weekday()]} {MONTHS[d.month]} {d.day}"
    if left < 0:
        return f"due {when} (OVERDUE by {abs(left)} days)"
    if left == 0:
        return f"due {when} (TODAY)"
    if left == 1:
        return f"due {when} (TOMORROW)"
    return f"due {when} ({left} days left)"


def summary() -> str:
    """Task list for the system prompt."""
    pending = all_tasks()
    if not pending:
        return ""

    today = datetime.now(ZoneInfo(config.CALENDAR_TZ)).date()

    def key(t):
        # Dated tasks first, soonest deadline first
        return (0, t["due"]) if t.get("due") else (1, "")

    lines = ["User's outstanding tasks:"]
    for t in sorted(pending, key=key):
        parts = [t["title"]]
        if t.get("course"):
            parts.append(t["course"])
        label = _due_label(t.get("due", ""), today)
        parts.append(label if label else "no deadline")
        if t.get("estimate_hours"):
            parts.append(f"about {t['estimate_hours']:g} hours")
        lines.append("  - " + " | ".join(parts))
    return "\n".join(lines)

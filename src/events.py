"""One store for everything that sits on a timeline: classes, tasks, reminders.

Kept in `memory/events.json` — plain JSON, editable by hand, readable in a git
diff. No database: a few hundred entries do not justify one, and being able to
open the file in Notepad and fix a typo is worth more than query speed here.

WHAT UNIFIES THEM is `start`. Everything has a moment it belongs to; the kinds
differ in how precise that moment is and whether it occupies time:

    class     start + end     a slot on the calendar
    task      start           a deadline, no slot
    reminder  start           a point in time, no slot

`start` is either a date (`2026-08-14`) meaning "sometime that day", or a date
and time (`2026-08-14T17:00`) meaning exactly then. That one rule removes the
need for a separate "all day" flag.

`done` only means something for tasks and reminders. A class cannot be finished,
only attended.

Every occurrence is stored on its own — a recurring class is 12 rows, not one
rule. That was a deliberate choice: nothing has to expand a pattern into dates,
so what you read in the file is exactly what the agent sees. The cost is that
moving a weekly class means editing every row.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()

KINDS = ("class", "task", "reminder")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def path():
    return config.MEMORY_DIR / "events.json"


# --- Reading and writing ----------------------------------------------------


def load() -> list[dict]:
    p = path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("events.json is unreadable", exc_info=True)
        return []
    items = data.get("events", [])
    return sorted(items, key=lambda e: str(e.get("start", "")))


def save(items: list[dict]) -> None:
    """Write through a temp file, then rename.

    If the process dies mid-write, the old file survives intact. A half-written
    events.json would be worse than a stale one.
    """
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(items, key=lambda e: str(e.get("start", "")))
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"events": ordered}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, p)


# --- Time helpers -----------------------------------------------------------


def when(e: dict) -> datetime | None:
    """`start` as a datetime. A date-only entry lands at 00:00 that day."""
    raw = str(e.get("start") or "")
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def day_of(e: dict) -> date | None:
    dt = when(e)
    return dt.date() if dt else None


def has_time(e: dict) -> bool:
    """True when `start` names an hour, not just a day."""
    return "T" in str(e.get("start") or "") or " " in str(e.get("start") or "").strip()


def today() -> date:
    return datetime.now(ZoneInfo(config.TIMEZONE)).date()


def now() -> datetime:
    return datetime.now(ZoneInfo(config.TIMEZONE)).replace(tzinfo=None)


# --- Queries ----------------------------------------------------------------


def on_day(d: date, kinds: tuple[str, ...] | None = None) -> list[dict]:
    return [
        e for e in load()
        if day_of(e) == d and (kinds is None or e.get("kind") in kinds)
    ]


def between(start: date, end: date, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """Entries from `start` up to but not including `end`."""
    out = []
    for e in load():
        d = day_of(e)
        if d is None or not (start <= d < end):
            continue
        if kinds is None or e.get("kind") in kinds:
            out.append(e)
    return out


def upcoming(limit: int = 1, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """The next entries still ahead of us, soonest first."""
    n = now()
    ahead = [
        e for e in load()
        if (dt := when(e)) is not None and dt > n
        and not e.get("done")
        and (kinds is None or e.get("kind") in kinds)
    ]
    return sorted(ahead, key=lambda e: when(e))[:limit]


def open_tasks() -> list[dict]:
    """Unfinished tasks, nearest deadline first."""
    items = [e for e in load() if e.get("kind") == "task" and not e.get("done")]
    return sorted(items, key=lambda e: str(e.get("start") or "9999"))


# --- Writing ----------------------------------------------------------------


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40] or "event"


def make_id(kind: str, title: str, start: str) -> str:
    """Stable and readable: kind-title-date. Re-importing the same source event
    produces the same id, so it updates in place rather than duplicating."""
    return f"{kind}-{_slug(title)}-{str(start)[:16].replace(':', '')}"


def add(
    kind: str,
    title: str,
    start: str,
    end: str = "",
    course: str = "",
    session: str = "",
    location: str = "",
    estimate_hours: float = 0,
    notes: str = "",
) -> dict:
    """Add one entry. `start` is YYYY-MM-DD or YYYY-MM-DDTHH:MM."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    e = {
        "id": make_id(kind, title, start),
        "kind": kind,
        "title": title.strip(),
        "start": start,
    }
    # Empty fields are left out entirely rather than stored as "". A file full
    # of empty strings is far harder to read by eye.
    for key, value in (
        ("end", end), ("course", course.strip()), ("session", session.strip()),
        ("location", location.strip()), ("notes", notes.strip()),
    ):
        if value:
            e[key] = value
    if estimate_hours:
        e["estimate_hours"] = estimate_hours
    if kind in ("task", "reminder"):
        e["done"] = False
    e["created"] = datetime.now().isoformat(timespec="seconds")

    with _lock:
        items = load()
        # Same id = same thing. Replace instead of piling up duplicates, which
        # is what happens when a source is imported twice.
        items = [x for x in items if x.get("id") != e["id"]]
        items.append(e)
        save(items)
    log.info("event added: %s", e["id"])
    return e


def log_hours(event_id: str, hours: float) -> dict | None:
    """Record time worked on a task. Adds to `spent_hours`.

    Adding to what was spent, rather than subtracting from the estimate, is
    deliberate. Subtracting would leave 5 where 8 used to be, and the original
    8 is the more useful number: comparing it against what the task actually
    took is the only way to find out how wrong your estimates run.

    Negative hours are allowed, for correcting a mis-log.
    """
    with _lock:
        items = load()
        for e in items:
            if e.get("id") != event_id:
                continue
            spent = round(float(e.get("spent_hours", 0)) + hours, 2)
            # Never let a correction push it below zero — that reads as a bug
            # in the log rather than as work undone.
            e["spent_hours"] = max(0.0, spent)
            save(items)
            log.info("event %s spent_hours=%s", event_id, e["spent_hours"])
            return e
    return None


def remaining(e: dict) -> float | None:
    """Hours left, or None when there is no estimate to count down from."""
    est = e.get("estimate_hours")
    if not est:
        return None
    return round(float(est) - float(e.get("spent_hours", 0)), 2)


def mark_done(event_id: str, done: bool = True) -> dict | None:
    with _lock:
        items = load()
        for e in items:
            if e.get("id") == event_id:
                e["done"] = done
                save(items)
                log.info("event %s done=%s", event_id, done)
                return e
    return None


def remove(event_id: str) -> bool:
    with _lock:
        items = load()
        left = [e for e in items if e.get("id") != event_id]
        if len(left) == len(items):
            return False
        save(left)
    log.info("event removed: %s", event_id)
    return True


def clear(kind: str | None = None) -> int:
    """Delete everything, or everything of one kind. Returns how many went."""
    with _lock:
        items = load()
        left = [e for e in items if kind is not None and e.get("kind") != kind]
        save(left)
    n = len(items) - len(left)
    log.info("cleared %d events (kind=%s)", n, kind or "all")
    return n


# --- Rendering --------------------------------------------------------------


def time_label(e: dict) -> str:
    dt = when(e)
    if dt is None:
        return "?"
    if not has_time(e):
        return "all day"
    out = f"{dt:%H:%M}"
    if e.get("end"):
        end = str(e["end"])
        out += f"-{end[11:16] if 'T' in end else end}"
    return out


def day_label(d: date, ref: date | None = None) -> str:
    ref = ref or today()
    diff = (d - ref).days
    name = f"{DAYS[d.weekday()]} {d.day} {MONTHS[d.month]}"
    if diff == 0:
        return f"Today ({name})"
    if diff == 1:
        return f"Tomorrow ({name})"
    if diff == -1:
        return f"Yesterday ({name})"
    return name


def one_line(e: dict) -> str:
    bits = [f"{time_label(e):>11}"]
    if e.get("course"):
        bits.append(e["course"])
    bits.append(e.get("title", "?"))
    if e.get("session"):
        bits.append(f"({e['session']})")
    if e.get("location"):
        bits.append(f"@ {' '.join(str(e['location']).split())}")
    bits.append(effort_label(e))
    if e.get("done"):
        bits.append("[done]")
    return "  ".join(b for b in bits if b)


def effort_label(e: dict) -> str:
    """'~8h', or '3h of 8h, 5h left' once work has been logged.

    Both numbers are shown rather than just the remainder: seeing 3 of 8 tells
    you how far in you are, which a bare '5h left' does not.
    """
    est = e.get("estimate_hours")
    spent = float(e.get("spent_hours", 0) or 0)
    if not est and not spent:
        return ""
    if not est:
        return f"{spent:g}h done"
    left = remaining(e)
    if not spent:
        return f"~{float(est):g}h"
    if left is not None and left < 0:
        return f"{spent:g}h of {float(est):g}h, {-left:g}h over"
    return f"{spent:g}h of {float(est):g}h, {left:g}h left"


def agenda(days: int = 14) -> str:
    """Human-readable listing, for `python -m src.tools`."""
    start = today()
    end = start + timedelta(days=days)
    items = between(start, end)

    out = [f"=== NEXT {days} DAYS ({start} to {end}) ===", ""]
    if not items:
        out.append("  (nothing scheduled)")
    by_day: dict = {}
    for e in items:
        by_day.setdefault(day_of(e), []).append(e)

    for d in sorted(by_day):
        out.append(day_label(d))
        for e in sorted(by_day[d], key=lambda x: str(x.get("start"))):
            out.append("   " + one_line(e))
        out.append("")

    tasks = open_tasks()
    out.append(f"=== OPEN TASKS ({len(tasks)}) ===")
    out.append("")
    if not tasks:
        out.append("  (none)")
    for t in tasks:
        d = day_of(t)
        days_left = (d - start).days if d else None
        due = ""
        if days_left is not None:
            due = (
                "  [due today]" if days_left == 0
                else "  [due tomorrow]" if days_left == 1
                else f"  [OVERDUE by {-days_left}d]" if days_left < 0
                else f"  [{days_left}d left]"
            )
        effort = effort_label(t)
        out.append(f"   {t.get('title', '?')}{due}" + (f"  —  {effort}" if effort else ""))

    hours_left = sum(
        r for t in tasks if (r := remaining(t)) is not None and r > 0
    )
    if hours_left:
        out.append("")
        out.append(f"   {hours_left:g} hours of work outstanding")
    return "\n".join(out)


# Everything arrives through Parakeet, which writes numbers as words. The
# store writes them as digits, because that is how a timetable import spells
# them. So "I've finished assignment one" found nothing while "Assignment 1"
# sat right there -- measured, not imagined.
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "first": "1", "second": "2", "third": "3",
    "fourth": "4", "fifth": "5",
}


def _normalise(text: str) -> str:
    words = re.split(r"(\W+)", (text or "").lower())
    return "".join(_NUMBER_WORDS.get(w, w) for w in words)


def find(needle: str) -> list[dict]:
    """Entries whose id or title contains `needle`, case-insensitively.

    Returns every match rather than picking one. Both doors resolve a fragment
    through here, and ambiguity has to reach the caller intact -- guessing
    which "assignment" was meant is how the wrong entry gets deleted.

    Matched twice: once literally, once with number words turned into digits.
    The second pass is what makes a spoken "assignment one" reach a stored
    "Assignment 1".
    """
    needle = (needle or "").strip().lower()
    if not needle:
        return []
    norm = _normalise(needle)

    hits = []
    for e in load():
        hay = f"{e.get('id', '')} {e.get('title', '')}".lower()
        if needle in hay or norm in _normalise(hay):
            hits.append(e)
    return hits


if __name__ == "__main__":
    print("The store has no command line of its own -- the tools do:")
    print()
    print("    python -m src.tools                 list everything")
    print("    python -m src.tools add task \"Assignment 1\" 2026-08-14 --course ENGN4122")
    print("    python -m src.tools log \"Assignment 1\" 3")
    print("    python -m src.tools done \"Assignment 1\"")
    print()
    print(f"file: {path()}")

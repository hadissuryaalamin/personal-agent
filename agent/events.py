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
    """Human-readable listing, for `python -m agent.events`."""
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


def _cli() -> int:
    """Read and edit the store from a terminal.

    Exists so the store is usable before anything is wired into the voice
    agent — and so there is always a way in that does not depend on the
    microphone, the models, or Ollama being up.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="python -m agent.events")
    sub = ap.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="show the agenda (default)")
    p_list.add_argument("days", nargs="?", type=int, default=14)

    p_add = sub.add_parser("add", help="add a task, reminder, or class")
    p_add.add_argument("kind", choices=KINDS)
    p_add.add_argument("title")
    p_add.add_argument("start", help="YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p_add.add_argument("--end", default="")
    p_add.add_argument("--course", default="")
    p_add.add_argument("--session", default="")
    p_add.add_argument("--location", default="")
    p_add.add_argument("--hours", type=float, default=0, help="estimated effort")
    p_add.add_argument("--notes", default="")

    p_log = sub.add_parser("log", help="record hours worked on a task")
    p_log.add_argument("id_part", help="any distinctive part of the id or title")
    p_log.add_argument("hours", type=float, help="negative to correct a mis-log")

    p_done = sub.add_parser("done", help="mark a task or reminder finished")
    p_done.add_argument("id_part", help="any distinctive part of the id or title")

    p_rm = sub.add_parser("rm", help="delete an entry")
    p_rm.add_argument("id_part")

    args = ap.parse_args()
    cmd = args.cmd or "list"

    if cmd == "add":
        e = add(
            args.kind, args.title, args.start, end=args.end, course=args.course,
            session=args.session, location=args.location,
            estimate_hours=args.hours, notes=args.notes,
        )
        print(f"added: {e['id']}\n  {one_line(e)}")
        return 0

    if cmd in ("log", "done", "rm"):
        # Match on a fragment so you never have to type a full id. Ambiguity is
        # reported rather than resolved by guessing — picking the wrong entry
        # here deletes the wrong thing.
        needle = args.id_part.lower()
        hits = [
            e for e in load()
            if needle in e.get("id", "").lower() or needle in e.get("title", "").lower()
        ]
        if not hits:
            print(f"nothing matches {args.id_part!r}")
            return 1
        if len(hits) > 1:
            print(f"{len(hits)} entries match {args.id_part!r} — be more specific:")
            for e in hits[:10]:
                print(f"  {e['id']}")
            return 1
        e = hits[0]
        if cmd == "log":
            updated = log_hours(e["id"], args.hours)
            print(f"{updated['title']}: {effort_label(updated)}")
            left = remaining(updated)
            if left is not None and left <= 0:
                print("  (estimate used up — mark it done when it really is:")
                print(f"   python -m agent.events done \"{updated['title']}\")")
        elif cmd == "done":
            mark_done(e["id"])
            print(f"done: {e['title']}")
        else:
            remove(e["id"])
            print(f"removed: {e['title']}")
        return 0

    items = load()
    counts = {k: sum(1 for e in items if e.get("kind") == k) for k in KINDS}
    print(f"\n{path()}  —  {len(items)} entries "
          f"({', '.join(f'{v} {k}' for k, v in counts.items())})\n")
    print(agenda(args.days))
    return 0


if __name__ == "__main__":
    #   python -m agent.events                                    # next 14 days
    #   python -m agent.events list 60                            # next 60 days
    #   python -m agent.events add task "Assignment 1" 2026-08-14 --course ENGN4122 --hours 8
    #   python -m agent.events add reminder "Pay rego" 2026-08-19T09:00
    #   python -m agent.events log "Assignment 1" 3     # worked 3 hours
    #   python -m agent.events log "Assignment 1" -1    # correct a mis-log
    #   python -m agent.events done "Assignment 1"
    #   python -m agent.events rm "Pay rego"
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(_cli())

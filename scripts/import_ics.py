"""One-time migration: an .ics calendar into memory/events.json.

Every VEVENT becomes one `class` entry — one row per occurrence, no recurrence
rules. Run it once and the .ics is no longer needed.

    python scripts\\import_ics.py --dry-run     # show what would happen
    python scripts\\import_ics.py               # actually write

Safe to re-run: ids are derived from title + start, so a second run replaces
rather than duplicates.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import events  # noqa: E402

# ANU session codes, as they appear at the end of a SUMMARY: "..., LecA"
SESSIONS = {
    "lec": "lecture", "tut": "tutorial", "com": "computer lab",
    "lab": "lab", "wor": "workshop", "sem": "seminar", "dro": "drop-in",
}
# Titles written by the earlier export carry the session in brackets instead.
BRACKETED = set(SESSIONS.values()) | {
    "kuliah", "tutorial", "lab komputer", "lab", "workshop", "seminar",
}
TRANSLATE = {"kuliah": "lecture", "lab komputer": "computer lab"}


def clean_title(summary: str) -> str:
    """'Advanced Topics_Agentic Coding Studio (Class: 9056), LecA'
    -> 'Agentic Coding Studio'

    The ANU format is GeneralTitle_SpecificName. The general half is dropped:
    the course code already says it, and ANU often truncates it mid-word.
    """
    text = re.sub(r"\(Class:\s*\d+\)", "", summary)
    text = re.sub(r"\\?,\s*(Lec|Tut|Com|Lab|Wor|Sem|Dro)\w*\s*$", "", text.strip())
    m = re.match(r"^(.*)\(([^)]+)\)\s*$", text)
    if m and m.group(2).strip().lower() in BRACKETED:
        text = m.group(1)
    if "_" in text:
        parts = [p.strip() for p in text.split("_") if p.strip()]
        if parts:
            text = parts[-1]
    return re.sub(r"[\s,]+$", "", re.sub(r"\s+", " ", text)).strip()


def session_of(summary: str) -> str:
    m = re.search(r"\\?,\s*(Lec|Tut|Com|Lab|Wor|Sem|Dro)\w*\s*$", summary.strip())
    if m:
        return SESSIONS.get(m.group(1).lower(), "")
    m = re.match(r"^.*\(([^)]+)\)\s*$", summary.strip())
    if m:
        inner = m.group(1).strip().lower()
        if inner in BRACKETED:
            return TRANSLATE.get(inner, inner)
    return ""


_COURSE = re.compile(r"\b([A-Z]{4}\d{4})\b")


def course_of(summary: str, description: str = "") -> str:
    """Course code from wherever it happens to live.

    The ANU feed puts it in DESCRIPTION. Events re-exported by an earlier
    version of this project have no DESCRIPTION at all and carry the code at the
    front of the title instead — so both are checked.
    """
    for source in (description, summary):
        m = _COURSE.search(source or "")
        if m:
            return m.group(1)
    return ""


def strip_course(title: str, course: str) -> str:
    """Drop the code from the title once it has its own field.

    Kept separate so it can be filtered on ("everything for COMP4620") without
    string matching, and so the renderer decides how to join them for speech.
    """
    if not course:
        return title
    return re.sub(rf"^\s*{re.escape(course)}\s*[-:]?\s*", "", title).strip() or title


def clean_location(location: str) -> str:
    """'Rm 2.02_Fulton Muir Bldg 95' -> 'Fulton Muir, Rm 2.02'

    Building name first, because that is what you look for; the building number
    is dropped because it does not help when spoken aloud.
    """
    if not location:
        return ""
    text = re.sub(r"\s*Bldg\s*\d+\s*", "", location).strip()
    parts = [p.strip() for p in text.split("_") if p.strip()]
    if len(parts) == 2 and parts[0].lower().startswith(("rm", "room")):
        parts = [parts[1], parts[0]]
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ics", nargs="?", default="memory/kalender.ics")
    ap.add_argument("--dry-run", action="store_true", help="show, do not write")
    args = ap.parse_args()

    from datetime import datetime

    from icalendar import Calendar

    src = Path(args.ics)
    if not src.is_absolute():
        src = Path(__file__).resolve().parent.parent / src
    if not src.exists():
        print(f"not found: {src}")
        return 1

    cal = Calendar.from_ical(src.read_text(encoding="utf-8"))
    vevents = cal.walk("VEVENT")
    print(f"{src.name}: {len(vevents)} VEVENT\n")

    rows = []
    for c in vevents:
        dtstart = c.get("DTSTART")
        if dtstart is None:
            continue
        start = dtstart.dt
        dtend = c.get("DTEND")
        end = dtend.dt if dtend is not None else None

        # A date (not a datetime) means an all-day entry: store the day only.
        if isinstance(start, datetime):
            start_s = start.strftime("%Y-%m-%dT%H:%M")
            end_s = end.strftime("%Y-%m-%dT%H:%M") if isinstance(end, datetime) else ""
        else:
            start_s = start.isoformat()
            end_s = ""

        summary = str(c.get("SUMMARY", "")).strip()
        course = course_of(summary, str(c.get("DESCRIPTION", "")).strip())
        rows.append({
            "title": strip_course(clean_title(summary), course),
            "start": start_s,
            "end": end_s,
            "course": course,
            "session": session_of(summary),
            "location": clean_location(str(c.get("LOCATION", "")).strip()),
        })

    rows.sort(key=lambda r: r["start"])
    for r in rows[:5]:
        print(f"  {r['start']}  {r['course']} {r['title']} ({r['session']}) @ {r['location']}")
    if len(rows) > 5:
        print(f"  ... and {len(rows) - 5} more")

    if args.dry_run:
        print(f"\n(--dry-run: nothing written; would import {len(rows)} entries)")
        return 0

    for r in rows:
        events.add(kind="class", **r)

    after = events.load()
    print(f"\nwrote {len(rows)} entries -> {events.path()}")
    print(f"file now holds {len(after)} entries "
          f"({sum(1 for e in after if e.get('kind') == 'class')} class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

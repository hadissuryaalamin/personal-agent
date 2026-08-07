"""Turning a tool result into something worth saying out loud.

CLAUDE.md, "Response style": two sentences or about 320 characters; lists cap
at three items plus a count; writes are confirmed by restating the *resolved*
value, not the raw input, because that is how the user catches a misheard date.
No filler openers -- answer, then stop.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

MAX_CHARS = 320
LIST_CAP = 3

_ONES = [
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth",
]

_COUNT_WORDS = {
    0: "Nothing", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
}


def ordinal_word(day: int) -> str:
    if 1 <= day <= 20:
        return _ONES[day]
    if day == 30:
        return "thirtieth"
    if 21 <= day <= 29:
        return f"twenty-{_ONES[day - 20]}"
    if day == 31:
        return "thirty-first"
    return str(day)


def spoken_time(value: str | None) -> str:
    """"09:00" -> "9am"; "13:30" -> "1:30pm"."""
    if not value:
        return ""
    hour, minute = (int(part) for part in value.split(":"))
    suffix = "am" if hour < 12 else "pm"
    display = hour % 12 or 12
    return f"{display}{suffix}" if minute == 0 else f"{display}:{minute:02d}{suffix}"


def spoken_day(when: date | datetime, today: date) -> str:
    """"today", "tomorrow", or "Friday the fourteenth"."""
    day = when.date() if isinstance(when, datetime) else when
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    if 2 <= delta <= 6:
        return day.strftime("%A")
    return f"{day.strftime('%A')} the {ordinal_word(day.day)}"


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value).date()


def _trim(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_CHARS:
        return text
    return text[: MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"


def _count_word(number: int) -> str:
    return _COUNT_WORDS.get(number, str(number))


def reply_for(result: dict[str, Any], today: date) -> str:
    """One spoken-shaped sentence or two for any tool result."""
    if not result:
        return "Nothing came back."

    if result.get("needs") in ("clarification", "confirmation"):
        return _trim(result["question"])
    if "error" in result:
        return _trim(str(result["error"]))

    tool = result.get("tool")

    if tool == "get_now":
        return _trim(f"It is {spoken_time(result['time'])} on {result['weekday']}.")

    if tool == "list_schedule":
        return _trim(_schedule_reply(result, today))

    if tool == "list_assignments":
        return _trim(_assignments_reply(result, today))

    if result.get("created") == "course":
        where = f" in {result['location']}" if result.get("location") else ""
        return _trim(
            f"Added {result['code']}, {result['weekday']}s "
            f"{spoken_time(result['start'])} to {spoken_time(result['end'])}{where}."
        )

    if result.get("created") == "course_exception":
        day = spoken_day(_as_date(result["date"]), today)
        if result["kind"] == "cancelled":
            return _trim(f"{result['code']} is off {day}.")
        if result["kind"] == "moved":
            return _trim(
                f"{result['code']} moves to {spoken_time(result['start'])} {day}."
            )
        return _trim(f"{result['code']} is in {result['location']} {day}.")

    if result.get("created") == "assignment":
        when = spoken_day(_as_date(result["due_date"]), today)
        at = f" at {spoken_time(result['due'][11:16])}" if result.get("explicit_time") else ""
        tail = ""
        if result.get("overdue"):
            tail = " That is already past."
        elif result.get("unlinked_course"):
            tail = f" I have no class called {result['unlinked_course']}, so it is not linked."
        return _trim(f"Added, due {when}{at}.{tail}")

    if result.get("created") == "reminder":
        when = spoken_day(_as_date(result["date"]), today)
        at = f" at {spoken_time(result['when'][11:16])}" if result.get("explicit_time") else ""
        return _trim(f"I will remind you {when}{at}.")

    if result.get("updated") == "course":
        return _trim(
            f"{result['code']} is now {result['weekday']}s "
            f"{spoken_time(result['start'])} to {spoken_time(result['end'])}."
        )

    if result.get("updated") == "assignment":
        if "progress_pct" in result and result.get("changed", ["progress_pct"]) == ["progress_pct"]:
            if result.get("status") == "done":
                return _trim(f"{result['title']} is done.")
            left = result.get("hours_left")
            tail = f", about {left:g} hours left" if left is not None else ""
            return _trim(f"{result['title']} is at {result['progress_pct']} percent{tail}.")
        if result.get("due_date"):
            return _trim(
                f"{result['title']} is now due {spoken_day(_as_date(result['due_date']), today)}."
            )
        return _trim(f"Updated {result['title']}.")

    if result.get("deleted") == "course":
        return _trim(f"{result['code']} is gone. Say undo if that was wrong.")
    if result.get("deleted") == "assignment":
        return _trim(f"{result['title']} is gone. Say undo if that was wrong.")

    if "undone" in result:
        return _trim(f"Done — {result['description']}.")

    return _trim(str(result))


def _schedule_reply(result: dict[str, Any], today: date) -> str:
    items = result["items"]
    when = result["when"]
    if not items:
        return f"Nothing on {when}."

    live = [i for i in items if i["status"] != "cancelled"]
    off = [i for i in items if i["status"] == "cancelled"]

    if not live and off:
        names = ", ".join(i["code"] for i in off[:LIST_CAP])
        return f"Everything is cancelled {when} — {names}."

    head = ", then ".join(
        f"{i['code']} at {spoken_time(i['start'])}" for i in live[:LIST_CAP]
    )
    lead = f"{_count_word(len(live))} class{'es' if len(live) != 1 else ''} {when}"
    sentence = f"{lead} — {head}."
    if len(live) > LIST_CAP:
        sentence = f"{lead}; the first {LIST_CAP} are {head}."
    if off:
        sentence += f" {off[0]['code']} is cancelled."
    return sentence


def _assignments_reply(result: dict[str, Any], today: date) -> str:
    items = result["items"]
    if not items:
        return "Nothing outstanding."

    dated = [i for i in items if i["due_date"]]
    first = dated[0] if dated else items[0]
    noun = "thing" if len(items) == 1 else "things"
    lead = f"{_count_word(len(items))} {noun} outstanding"

    if first.get("due_date"):
        closest = f"the closest is {first['title']}, {spoken_day(_as_date(first['due_date']), today)}"
    else:
        closest = f"the closest is {first['title']}"

    if len(items) == 1:
        return f"{lead} — {first['title']}, {spoken_day(_as_date(first['due_date']), today)}." \
            if first.get("due_date") else f"{lead} — {first['title']}."
    return f"{lead} — {closest}."

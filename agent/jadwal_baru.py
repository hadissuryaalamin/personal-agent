"""Turn speech into a calendar event or task, with confirmation before saving.

Why confirmation: STT has a real error rate. For a *question*, a misheard word
just produces a wrong answer you notice immediately. For a *write*, it leaves a
bogus event in your calendar that you only discover next week. So the agent
always reads the parsed event back and waits for a yes.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import config, teks, time_en

log = logging.getLogger(__name__)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Intent keywords for creating an event. Matched locally just to decide whether
# this utterance needs event parsing — the content itself is the LLM's job.
# Bare "schedule" and "book" are deliberately absent: they are nouns at least
# as often as verbs ("what's my schedule", "read a book"), and "what is my
# schedule tomorrow" matching here would try to write an event for a question.
_INTENT = (
    "create an event", "create event", "add an event", "add event",
    "schedule a", "schedule an", "schedule my", "schedule it", "schedule that",
    "book a", "book an", "book the", "book me",
    "put in my calendar", "put it in my calendar", "put that in my calendar",
    "add to my calendar", "add it to my calendar", "new event",
    "set up a meeting", "make an event", "set a reminder for",
)

# Openers that make the utterance a question about the calendar, not a request
# to write to it. Checked before _INTENT so "when should I schedule a break"
# doesn't silently create an event.
_QUESTION = (
    "what", "when", "where", "which", "who", "how", "do i", "did i",
    "is there", "are there", "am i", "tell me", "read me", "anything",
)

# Single words, matched against the word list. "yeah"/"yes" are handled
# separately below — see answer_yes().
_YES = (
    "yep", "yup", "correct", "right", "sure", "ok", "okay", "confirm",
    "save", "perfect", "exactly",
)
_NO = (
    "no", "nope", "cancel", "wrong", "dont", "nevermind", "stop",
    "incorrect", "nah",
)
# Multi-word forms, matched against the whole utterance. Kept apart because a
# per-word check can never match them — that silently killed "go ahead" and
# "never mind" once already.
_YES_PHRASES = ("please do", "go ahead", "sounds good", "do it", "that works")
_NO_PHRASES = ("never mind", "forget it", "do not", "hold on", "not right")

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short name for the event"},
        "date": {"type": "string", "description": "Start date, format YYYY-MM-DD"},
        "start_time": {"type": "string", "description": "Start time, 24h HH:MM"},
        "duration_minutes": {"type": "integer", "description": "Length in minutes"},
        "location": {"type": "string", "description": "Location, empty string if none"},
        "confident": {
            "type": "boolean",
            "description": "false if the date or time is unclear from the utterance",
        },
    },
    "required": ["title", "date", "start_time", "duration_minutes", "location", "confident"],
    "additionalProperties": False,
}

PROMPT = """Turn the user's request into a single calendar event.

Rules:
- Use the current date and time to resolve "tomorrow", "next Friday", "in two days".
- If no duration is given, use 60 minutes.
- If an hour is given with no am/pm, pick what makes sense for a student
  (3 = 15:00, 8 = 08:00).
- Set confident=false if the date or time cannot be determined from the request.
- The text came from speech recognition, so it may contain mishearings. Guess the
  most plausible meaning, but do not invent details that were never mentioned."""


def wants_event(text: str) -> bool:
    """Does this utterance ask to create an event?"""
    t = teks.normal(text)
    if any(t.startswith(q) for q in _QUESTION):
        return False
    return any(k in t for k in _INTENT)


def answer_yes(text: str) -> bool | None:
    """True=agree, False=refuse, None=unclear.

    Unclear deliberately reads as None, not True. Guessing wrong here means
    writing an event the user never asked for — the expensive direction.
    """
    flat = teks.normal(text)
    words = flat.split()
    if not words:
        return None
    # Check refusal first: "yeah no, cancel that" is a refusal
    if any(w in _NO for w in words) or any(p in flat for p in _NO_PHRASES):
        return False
    if any(w in _YES for w in words) or any(p in flat for p in _YES_PHRASES):
        return True
    # Bare "yes"/"yeah" only counts as the opening word — it trails questions
    # too often ("that's right, yeah?") to trust anywhere else.
    if words[0] in ("yes", "yeah", "yea", "aye"):
        return True
    return None


def _build_time(data: dict, tz) -> datetime | None:
    """Combine date + time, tolerating format drift.

    Small models often return a full ISO string in the date field, or a time
    with seconds and an offset. The content is right, only the shape differs —
    rejecting outright just fails for no reason.
    """
    raw_date = str(data.get("date") or "").strip()
    raw_time = str(data.get("start_time") or "").strip()
    if not raw_date:
        return None

    day = raw_date.split("T")[0].split(" ")[0]
    try:
        d = datetime.strptime(day, "%Y-%m-%d").date()
    except ValueError:
        return None

    clean = re.split(r"[+Z]", raw_time)[0].strip()
    parts = clean.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=tz)


def parse_event(text: str, oneshot) -> dict | None:
    """Utterance -> event dict. `oneshot(system, user, schema) -> str`."""
    tz = ZoneInfo(config.CALENDAR_TZ)
    now = datetime.now(tz)
    context = (
        f"Current date and time: {now.strftime('%A, %Y-%m-%d, %H:%M')} "
        f"({config.CALENDAR_TZ}).\n\nUser said: {text}"
    )

    raw = oneshot(PROMPT, context, SCHEMA)
    try:
        data = json.loads(raw)
    except Exception:
        log.warning("event parse was not JSON: %r", raw)
        return None

    # Date and time are parsed here and OVERRIDE the model. Time phrases are a
    # closed set with rigid rules, and small models proved unreliable at date
    # arithmetic (2 of 10 correct even with a date table). The model only needs
    # to handle the title and location.
    exact = time_en.parse(text, now)
    start = exact or _build_time(data, tz)
    if start is None:
        log.warning("could not read date/time: %r", data)
        return None
    if exact is not None and _build_time(data, tz) != exact:
        log.info("model's date/time corrected to %s", exact)

    minutes = int(data.get("duration_minutes") or 60)
    return {
        "title": (data.get("title") or "Event").strip(),
        "start": start,
        "end": start + timedelta(minutes=max(5, minutes)),
        "location": (data.get("location") or "").strip(),
        "confident": bool(data.get("confident", True)),
    }


TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short name for the task"},
        "course": {
            "type": "string",
            "description": "Course code if mentioned, e.g. COMP4020. May be empty.",
        },
        "due": {
            "type": "string",
            "description": "Due date, YYYY-MM-DD. Empty string if not mentioned.",
        },
        "estimate_hours": {
            "type": "number",
            "description": "Rough hours needed. 0 if not mentioned.",
        },
    },
    "required": ["title", "course", "due", "estimate_hours"],
    "additionalProperties": False,
}

TASK_PROMPT = """Turn the user's request into a single coursework task.

Rules:
- Use the current date to resolve "Friday", "next week", "tomorrow".
- If no due date is mentioned at all, set due to an empty string.
  DO NOT invent a date.
- Keep the title short; drop the word "task" unless it belongs.
- The text came from speech recognition, so it may contain mishearings."""


def parse_task(text: str, oneshot) -> dict | None:
    """Utterance -> task dict."""
    tz = ZoneInfo(config.CALENDAR_TZ)
    now = datetime.now(tz)
    context = (
        f"Current date and time: {now.strftime('%A, %Y-%m-%d, %H:%M')} "
        f"({config.CALENDAR_TZ}).\n\nUser said: {text}"
    )
    try:
        data = json.loads(oneshot(TASK_PROMPT, context, TASK_SCHEMA))
    except Exception:
        log.warning("task parse failed", exc_info=True)
        return None

    # Same as events: parse the date ourselves, don't trust the model
    exact = time_en.find_date(text, now.date())
    if exact is not None:
        due = exact.isoformat()
    else:
        due = (data.get("due") or "").strip().split("T")[0]
        if due:
            try:
                datetime.strptime(due, "%Y-%m-%d")
            except ValueError:
                log.warning("invalid due date: %r", due)
                due = ""

    title = (data.get("title") or "").strip()
    if not title:
        return None
    return {
        "title": title,
        "course": (data.get("course") or "").strip(),
        "due": due,
        "estimate_hours": float(data.get("estimate_hours") or 0),
    }


def confirmation_line(event: dict) -> str:
    """Read-back for the user to confirm.

    Says the weekday AND the date on purpose: if the recogniser misheard
    "Thursday" as "Tuesday", the mismatch is audible immediately.
    """
    s = event["start"]
    when = f"{DAYS[s.weekday()]} {MONTHS[s.month]} {s.day}"
    # Spelled out rather than "15:30" — the synthesiser reads the colon aloud.
    # 24-hour form, because "at 2" is ambiguous between day and night.
    time_part = f"at {s.hour}" + (f" {s.minute}" if s.minute else "")
    text = f"{event['title']}, {when}, {time_part}"
    if event["location"]:
        text += f", at {event['location']}"
    return f"I'll save: {text}. Is that right?"

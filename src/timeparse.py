"""Every relative time expression in the system is resolved here.

Invariant #1: the LLM never does date arithmetic. Anything the user says about
time -- "next Friday", "in two weeks", "the 14th" -- arrives here as text and
leaves as a concrete datetime, resolved against an injected ``now`` so the whole
module is testable without a clock.

Conventions, all deliberate and all covered by tests:

* Weeks start Monday (ISO). "next week" is the following Monday-to-Sunday
  block, never "the next seven days".
* A bare weekday ("friday") is the next occurrence on or after today, so on a
  Friday it means today.
* "next friday" is the Friday of *next* week -- see
  ``NEXT_WEEKDAY_MEANS_FOLLOWING_WEEK``. This is the one genuinely contested
  reading in the language; it is resolved by rule rather than by asking,
  because the caller restates the resolved date back to the user ("due Friday
  the fourteenth"), which is where a wrong reading gets caught.
* A bare hour with no am/pm marker raises rather than guessing when it could
  plausibly be either (1-11 o'clock). 13-23 is a 24-hour clock and is taken at
  face value; 12 is noon, 0 is midnight.
* Purely numeric dates are day/month, not month/day (Australian convention).

Nothing here touches the database, the clock, or the model. Callers pass
``now`` in; everything is a pure function of the text and that timestamp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

__all__ = [
    "TimeParseError",
    "AmbiguousTime",
    "UnknownTime",
    "Instant",
    "Interval",
    "resolve_instant",
    "resolve_interval",
    "resolve_time_of_day",
    "parse_weekday",
    "weekday_name",
    "number_from_words",
    "END_OF_DAY",
    "MORNING",
]

#: See the module docstring. Flip to False to make "next friday" mean the very
#: next Friday instead of the Friday of the following week.
NEXT_WEEKDAY_MEANS_FOLLOWING_WEEK = True

#: Default clock time for a date with no time attached. A due date is the end
#: of that day; a reminder is the morning of it.
END_OF_DAY = time(23, 59)
MORNING = time(9, 0)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class TimeParseError(Exception):
    """Base class. Carries the question to put back to the user.

    Invariant #6: a time expression we cannot pin down produces one clarifying
    question, never a best guess.
    """

    def __init__(self, question: str, text: str = "") -> None:
        super().__init__(question)
        self.question = question
        self.text = text


class AmbiguousTime(TimeParseError):
    """Parsed, but with more than one plausible reading."""


class UnknownTime(TimeParseError):
    """Not recognised as a time expression at all."""


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Instant:
    """A single resolved point in time, timezone-aware."""

    dt: datetime
    #: False when the time of day was defaulted rather than spoken. Callers use
    #: this to decide whether to say the time back to the user.
    explicit_time: bool

    @property
    def utc_iso(self) -> str:
        return self.dt.astimezone(ZoneInfo("UTC")).isoformat()


@dataclass(frozen=True)
class Interval:
    """A half-open span [start, end), timezone-aware."""

    start: datetime
    end: datetime
    label: str

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    @property
    def utc_bounds(self) -> tuple[str, str]:
        utc = ZoneInfo("UTC")
        return self.start.astimezone(utc).isoformat(), self.end.astimezone(utc).isoformat()


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

_WEEKDAYS = {
    "monday": 0, "mon": 0, "mondays": 0,
    "tuesday": 1, "tue": 1, "tues": 1, "tuesdays": 1,
    "wednesday": 2, "wed": 2, "weds": 2, "wednesdays": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "thursdays": 3,
    "friday": 4, "fri": 4, "fridays": 4,
    "saturday": 5, "sat": 5, "saturdays": 5,
    "sunday": 6, "sun": 6, "sundays": 6,
}

_WEEKDAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_CARDINALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}

_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "twenty first": 21, "twenty second": 22,
    "twenty third": 23, "twenty fourth": 24, "twenty fifth": 25,
    "twenty sixth": 26, "twenty seventh": 27, "twenty eighth": 28,
    "twenty ninth": 29, "thirtieth": 30, "thirty first": 31,
}

#: Expressions with no defensible resolution. These ask rather than pick a day.
_VAGUE = (
    "soon", "later", "sometime", "some time", "whenever", "eventually",
    "at some point", "in a bit", "in a while", "one of these days",
    "a few days", "a couple of days", "in a few", "asap",
    "as soon as possible", "when i get a chance", "before long",
)

_UNIT_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "fortnight": 14, "fortnights": 14,
}


# --------------------------------------------------------------------------
# normalisation and small parsers
# --------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lowercase, fold ASR punctuation, collapse whitespace.

    Colons, slashes and hyphens survive because they carry date and time
    meaning; sentence punctuation does not.
    """
    t = (text or "").strip().lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("a.m.", "am").replace("p.m.", "pm")
    t = t.replace("o'clock", "oclock").replace("o clock", "oclock")
    t = re.sub(r"[.,!?;\"]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _number_from_words(phrase: str) -> int | None:
    """"two" -> 2, "twenty one" -> 21, "a" -> 1, "3" -> 3."""
    phrase = phrase.strip().replace("-", " ")
    phrase = re.sub(r"\s+", " ", phrase)
    if not phrase:
        return None
    if phrase.isdigit():
        return int(phrase)
    if phrase in ("a", "an"):
        return 1
    if phrase in ("couple", "couple of", "a couple", "a couple of"):
        return 2
    if phrase in ("a hundred", "one hundred", "hundred"):
        return 100
    parts = phrase.split()
    if len(parts) == 1:
        return _CARDINALS.get(parts[0])
    if len(parts) == 2 and parts[0] in _TENS:
        unit = _CARDINALS.get(parts[1])
        if unit is not None and 1 <= unit <= 9:
            return _TENS[parts[0]] + unit
    return None


#: Public alias -- tools parse spoken quantities ("sixty percent", "six hours")
#: with the same vocabulary the dates use.
number_from_words = _number_from_words


def _day_of_month_from_words(phrase: str) -> int | None:
    """"14th" -> 14, "third" -> 3, "twenty first" -> 21."""
    phrase = phrase.strip().replace("-", " ")
    phrase = re.sub(r"\s+", " ", phrase)
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", phrase)
    if m:
        value = int(m.group(1))
        return value if 1 <= value <= 31 else None
    return _ORDINALS.get(phrase)


def _add_months(d: date, months: int) -> date:
    """Calendar-month arithmetic, clamping to the end of a short month."""
    total = (d.year * 12 + d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _combine(d: date, t: time, tz: ZoneInfo) -> datetime:
    return datetime.combine(d, t, tzinfo=tz)


def _as_local(now: datetime, tz: ZoneInfo) -> datetime:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware; a naive clock is a bug waiting to happen")
    return now.astimezone(tz)


# --------------------------------------------------------------------------
# time of day
# --------------------------------------------------------------------------

_TIME_PATTERNS = (
    # "at 3:30 pm", "at 15:00", "at 9am"
    re.compile(r"\bat\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>am|pm)?\b"),
    # "3:30pm", "15:00"
    re.compile(r"(?<!\d)(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ap>am|pm)?\b"),
    # "3pm", "9 am"
    re.compile(r"(?<!\d)(?P<h>\d{1,2})\s*(?P<ap>am|pm)\b"),
    # "3 oclock"
    re.compile(r"(?<!\d)(?P<h>\d{1,2})\s*oclock\b"),
)

#: A spoken hour with a marker: "nine am", "at eleven pm", "three oclock".
#: The marker is required, which is what keeps this from matching "in two
#: weeks" or the word before any other noun.
_WORD_TIME_MARKED = re.compile(
    r"(?:\bat\s+)?(?<![a-z])(?P<word>[a-z]+)\s*(?P<ap>am|pm|oclock)\b"
)

#: A spoken hour with no marker at all: "at three". Resolves to a question.
_WORD_TIME_BARE = re.compile(r"\bat\s+(?P<word>[a-z]+(?:\s+[a-z]+)?)\b")


def _resolve_hour(hour: int, minute: int, meridiem: str | None, source: str) -> time:
    if meridiem == "am":
        if not 1 <= hour <= 12:
            raise UnknownTime(f"I did not follow the time in “{source}”. What time?", source)
        return time(0 if hour == 12 else hour, minute)
    if meridiem == "pm":
        if not 1 <= hour <= 12:
            raise UnknownTime(f"I did not follow the time in “{source}”. What time?", source)
        return time(hour if hour == 12 else hour + 12, minute)

    # No am/pm marker.
    if hour == 0:
        return time(0, minute)
    if hour == 12:
        return time(12, minute)
    if 13 <= hour <= 23:
        return time(hour, minute)
    if 1 <= hour <= 11:
        raise AmbiguousTime(
            f"Do you mean {hour} in the morning or {hour} in the afternoon?", source
        )
    raise UnknownTime(f"There is no {hour} o'clock. What time did you mean?", source)


def _extract_time(text: str) -> tuple[time | None, str]:
    """Pull a time of day out of ``text``, returning it and what is left."""
    if re.search(r"\b(noon|midday)\b", text):
        return time(12, 0), re.sub(r"\b(at\s+)?(noon|midday)\b", " ", text)
    if re.search(r"\bmidnight\b", text):
        return time(0, 0), re.sub(r"\b(at\s+)?midnight\b", " ", text)

    for pattern in _TIME_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        groups = m.groupdict()
        hour = int(groups["h"])
        minute = int(groups.get("m") or 0)
        if minute > 59:
            raise UnknownTime(f"I did not follow the time in “{text}”. What time?", text)
        resolved = _resolve_hour(hour, minute, groups.get("ap"), text)
        return resolved, text[: m.start()] + " " + text[m.end() :]

    # Spoken hours. "half past nine" is deliberately not supported: it would
    # need its own grammar, and asking is better than half-parsing it.
    for pattern in (_WORD_TIME_MARKED, _WORD_TIME_BARE):
        m = pattern.search(text)
        if not m:
            continue
        value = _number_from_words(m.group("word"))
        if value is None or not 0 <= value <= 23:
            continue
        meridiem = m.groupdict().get("ap")
        meridiem = None if meridiem == "oclock" else meridiem
        resolved = _resolve_hour(value, 0, meridiem, text)
        return resolved, text[: m.start()] + " " + text[m.end() :]

    return None, text


def parse_weekday(text: str) -> int:
    """"Tuesday" / "tues" -> 1. Raises rather than defaulting to Monday."""
    normalised = normalise(text)
    if normalised in _WEEKDAYS:
        return _WEEKDAYS[normalised]
    m = re.search(rf"\b({'|'.join(sorted(_WEEKDAYS, key=len, reverse=True))})\b", normalised)
    if m:
        return _WEEKDAYS[m.group(1)]
    raise UnknownTime(f"Which day of the week is “{text}”?", text)


def weekday_name(index: int) -> str:
    return _WEEKDAY_NAMES[index]


def resolve_time_of_day(text: str) -> time:
    """Parse a bare clock time, for class start and end times.

    Raises rather than guessing, same as everything else here.
    """
    normalised = normalise(text)
    if not normalised:
        raise UnknownTime("What time?", text)
    found, _ = _extract_time(normalised)
    if found is None:
        # Bare "9" with no "at" still reads as a time in this context.
        m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", normalised)
        if m:
            return _resolve_hour(int(m.group(1)), int(m.group(2) or 0), None, text)
        value = _number_from_words(normalised)
        if value is not None:
            return _resolve_hour(value, 0, None, text)
        raise UnknownTime(f"I did not catch a time in “{text}”. What time?", text)
    return found


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------


def _reject_vague(text: str, original: str) -> None:
    for phrase in _VAGUE:
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            raise AmbiguousTime(
                f"“{phrase}” could be any day — which day did you have in mind?",
                original,
            )


def _weekday_date(target: int, today: date, qualifier: str | None) -> date:
    """Resolve a weekday name against ``today``.

    ``qualifier`` is "next", "this", "last", or None. See the module docstring
    for what each one means.
    """
    if qualifier == "next":
        if NEXT_WEEKDAY_MEANS_FOLLOWING_WEEK:
            return _monday_of(today) + timedelta(days=7 + target)
        ahead = (target - today.weekday()) % 7
        return today + timedelta(days=ahead or 7)

    if qualifier == "this":
        candidate = _monday_of(today) + timedelta(days=target)
        if candidate < today:
            raise AmbiguousTime(
                f"{_WEEKDAY_NAMES[target]} of this week has already gone — "
                f"do you mean the coming {_WEEKDAY_NAMES[target]}?"
            )
        return candidate

    if qualifier == "last":
        behind = (today.weekday() - target) % 7
        return today - timedelta(days=behind or 7)

    # Bare weekday: the next one, today included.
    return today + timedelta(days=(target - today.weekday()) % 7)


def _extract_date(text: str, today: date, original: str) -> date | None:
    """Find a calendar date in ``text``. Returns None if there isn't one."""

    # ISO, unambiguous, so it goes first.
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)), original)

    # Numeric day/month, Australian order.
    m = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?!\d)", text)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        if year < 100:
            year += 2000
        return _safe_date(year, month, day, original)

    # "14 august", "august 14", "14th of august"
    month_names = "|".join(_MONTHS)
    ordinal_words = "|".join(_ORDINALS)
    m = re.search(
        rf"\b(?P<day>\d{{1,2}}(?:st|nd|rd|th)?|{ordinal_words})\s+(?:of\s+)?"
        rf"(?P<month>{month_names})\b(?:\s+(?P<year>\d{{4}}))?",
        text,
    )
    if not m:
        m = re.search(
            rf"\b(?P<month>{month_names})\s+(?P<day>\d{{1,2}}(?:st|nd|rd|th)?|{ordinal_words})"
            rf"\b(?:\s+(?P<year>\d{{4}}))?",
            text,
        )
    if m:
        day = _day_of_month_from_words(m.group("day"))
        month = _MONTHS[m.group("month")]
        if day is None:
            raise UnknownTime(f"Which day in {m.group('month').title()}?", original)
        year = int(m.group("year")) if m.group("year") else today.year
        resolved = _safe_date(year, month, day, original)
        # A bare month/day that has already passed means next year.
        if m.group("year") is None and resolved < today:
            resolved = _safe_date(year + 1, month, day, original)
        return resolved

    # Named days.
    if re.search(r"\bday after tomorrow\b", text):
        return today + timedelta(days=2)
    if re.search(r"\bday before yesterday\b", text):
        return today - timedelta(days=2)
    if re.search(r"\b(today|tonight|this evening|this afternoon|this morning)\b", text):
        return today
    if re.search(r"\btomorrow\b", text):
        return today + timedelta(days=1)
    if re.search(r"\byesterday\b", text):
        return today - timedelta(days=1)

    # "in two weeks", "in 3 days", "in a fortnight"
    units = "|".join(_UNIT_DAYS)
    m = re.search(rf"\bin\s+(?P<n>[\w\s]+?)\s+(?P<unit>{units})\b", text)
    if m:
        count = _number_from_words(m.group("n"))
        if count is None:
            raise AmbiguousTime(
                f"How many {m.group('unit')} did you mean?", original
            )
        return today + timedelta(days=count * _UNIT_DAYS[m.group("unit")])

    # "in two months"
    m = re.search(r"\bin\s+(?P<n>[\w\s]+?)\s+months?\b", text)
    if m:
        count = _number_from_words(m.group("n"))
        if count is None:
            raise AmbiguousTime("How many months did you mean?", original)
        return _add_months(today, count)

    # "three days from now", "two weeks from today"
    m = re.search(rf"\b(?P<n>[\w\s]+?)\s+(?P<unit>{units})\s+from\s+(?:now|today)\b", text)
    if m:
        count = _number_from_words(m.group("n"))
        if count is not None:
            return today + timedelta(days=count * _UNIT_DAYS[m.group("unit")])

    # "end of the week" / "end of the month"
    if re.search(r"\bend of (?:the )?week\b", text):
        return _monday_of(today) + timedelta(days=6)
    if re.search(r"\bend of (?:the )?month\b", text):
        return date(today.year, today.month, _days_in_month(today.year, today.month))

    # Qualified and bare weekdays.
    weekday_names = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
    m = re.search(rf"\b(?:(?P<q>next|this coming|this|last)\s+)?(?P<wd>{weekday_names})\b", text)
    if m:
        qualifier = m.group("q")
        if qualifier == "this coming":
            qualifier = None
        return _weekday_date(_WEEKDAYS[m.group("wd")], today, qualifier)

    # A span used where a single day is required.
    m = re.search(r"\b(next|this|last)\s+(week|month|term|semester|year)\b", text)
    if m:
        raise AmbiguousTime(
            f"Which day {m.group(1)} {m.group(2)}?", original
        )

    # "the 14th", "on the third"
    m = re.search(
        rf"\bon\s+the\s+(?P<day>\d{{1,2}}(?:st|nd|rd|th)?|{ordinal_words})\b", text
    )
    if not m:
        m = re.search(rf"\bthe\s+(?P<day>\d{{1,2}}(?:st|nd|rd|th)|{ordinal_words})\b", text)
    if m:
        day = _day_of_month_from_words(m.group("day"))
        if day is not None:
            if day >= today.day:
                return _safe_date(today.year, today.month, day, original)
            following = _add_months(date(today.year, today.month, 1), 1)
            return _safe_date(following.year, following.month, day, original)

    return None


def _safe_date(year: int, month: int, day: int, original: str) -> date:
    try:
        return date(year, month, day)
    except ValueError:
        raise UnknownTime(
            f"There is no such date as {day}/{month}/{year}. What date did you mean?",
            original,
        ) from None


def _sub_day_offset(text: str, now: datetime) -> datetime | None:
    """"in two hours", "in 30 minutes" -- offsets that keep a clock time."""
    m = re.search(r"\bin\s+(?P<n>[\w\s]+?)\s+(?P<unit>hours?|hrs?|minutes?|mins?)\b", text)
    if not m:
        return None
    count = _number_from_words(m.group("n"))
    if count is None:
        raise AmbiguousTime(f"How many {m.group('unit')}?", text)
    if m.group("unit").startswith(("hour", "hr")):
        return now + timedelta(hours=count)
    return now + timedelta(minutes=count)


# --------------------------------------------------------------------------
# public resolution
# --------------------------------------------------------------------------


def resolve_instant(
    text: str,
    now: datetime,
    tz: ZoneInfo | None = None,
    default_time: time = END_OF_DAY,
) -> Instant:
    """Resolve ``text`` to a single point in time relative to ``now``.

    ``default_time`` fills in the clock when the user gave only a date.
    """
    tz = tz or (now.tzinfo if isinstance(now.tzinfo, ZoneInfo) else ZoneInfo("UTC"))
    local_now = _as_local(now, tz)

    normalised = normalise(text)
    if not normalised:
        raise UnknownTime("When?", text)
    _reject_vague(normalised, text)

    offset = _sub_day_offset(normalised, local_now)
    if offset is not None:
        return Instant(offset, explicit_time=True)

    clock, remainder = _extract_time(normalised)
    day = _extract_date(remainder, local_now.date(), text)

    if day is None:
        if clock is None:
            raise UnknownTime(
                f"I could not work out a date from “{text}”. When did you mean?", text
            )
        # A time with no date means today, or tomorrow if it has already gone.
        candidate = _combine(local_now.date(), clock, tz)
        if candidate <= local_now:
            candidate = _combine(local_now.date() + timedelta(days=1), clock, tz)
        return Instant(candidate, explicit_time=True)

    return Instant(_combine(day, clock or default_time, tz), explicit_time=clock is not None)


def resolve_interval(text: str | None, now: datetime, tz: ZoneInfo | None = None) -> Interval:
    """Resolve ``text`` to a half-open span. Empty text means today."""
    tz = tz or (now.tzinfo if isinstance(now.tzinfo, ZoneInfo) else ZoneInfo("UTC"))
    local_now = _as_local(now, tz)
    today = local_now.date()

    normalised = normalise(text or "")
    if not normalised:
        normalised = "today"
    _reject_vague(normalised, text or "")

    def span(start: date, days: int, label: str) -> Interval:
        return Interval(
            _combine(start, time(0, 0), tz),
            _combine(start + timedelta(days=days), time(0, 0), tz),
            label,
        )

    if re.search(r"\btoday\b", normalised):
        return span(today, 1, "today")
    if re.search(r"\btonight\b|\bthis evening\b", normalised):
        return Interval(
            _combine(today, time(17, 0), tz), _combine(today + timedelta(days=1), time(0, 0), tz),
            "tonight",
        )
    if re.search(r"\btomorrow\b", normalised):
        return span(today + timedelta(days=1), 1, "tomorrow")
    if re.search(r"\byesterday\b", normalised):
        return span(today - timedelta(days=1), 1, "yesterday")

    if re.search(r"\bthis week\b", normalised):
        return span(_monday_of(today), 7, "this week")
    if re.search(r"\bnext week\b", normalised):
        return span(_monday_of(today) + timedelta(days=7), 7, "next week")
    if re.search(r"\blast week\b", normalised):
        return span(_monday_of(today) - timedelta(days=7), 7, "last week")

    if re.search(r"\b(this |the )?weekend\b", normalised):
        saturday = _monday_of(today) + timedelta(days=5)
        if re.search(r"\bnext weekend\b", normalised):
            saturday += timedelta(days=7)
        return span(saturday, 2, "the weekend")

    if re.search(r"\bthis month\b", normalised):
        first = date(today.year, today.month, 1)
        return span(first, _days_in_month(today.year, today.month), "this month")
    if re.search(r"\bnext month\b", normalised):
        first = _add_months(date(today.year, today.month, 1), 1)
        return span(first, _days_in_month(first.year, first.month), "next month")

    # "the next three days", "in the next 2 weeks"
    units = "|".join(_UNIT_DAYS)
    m = re.search(rf"\b(?:next|coming)\s+(?P<n>[\w\s]+?)\s+(?P<unit>{units})\b", normalised)
    if m:
        count = _number_from_words(m.group("n"))
        if count is not None:
            days = count * _UNIT_DAYS[m.group("unit")]
            return span(today, days, f"the next {count} {m.group('unit')}")

    if re.search(r"\b(this |the )?(term|semester)\b", normalised):
        raise AmbiguousTime(
            "I do not know the term dates yet — which weeks does the term cover?",
            text or "",
        )

    # Anything else that resolves to a day is that whole day.
    instant = resolve_instant(normalised, now, tz)
    day = instant.dt.date()
    return span(day, 1, day.strftime("%A %d %B").replace(" 0", " "))

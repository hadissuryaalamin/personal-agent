"""Invariant #1 lives or dies here.

Every case is anchored to Thursday 6 August 2026, 10:00 Sydney.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from src import timeparse
from src.timeparse import AmbiguousTime, UnknownTime, resolve_instant, resolve_interval

TZ = ZoneInfo("Australia/Sydney")
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)  # a Thursday


def day_of(text: str, **kwargs) -> date:
    return resolve_instant(text, NOW, TZ, **kwargs).dt.date()


# -- relative days ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("today", date(2026, 8, 6)),
        ("tonight", date(2026, 8, 6)),
        ("tomorrow", date(2026, 8, 7)),
        ("yesterday", date(2026, 8, 5)),
        ("the day after tomorrow", date(2026, 8, 8)),
        ("in three days", date(2026, 8, 9)),
        ("in 3 days", date(2026, 8, 9)),
        ("in a week", date(2026, 8, 13)),
        ("in two weeks", date(2026, 8, 20)),
        ("in a fortnight", date(2026, 8, 20)),
        ("two weeks from now", date(2026, 8, 20)),
        ("in a month", date(2026, 9, 6)),
        ("in two months", date(2026, 10, 6)),
    ],
)
def test_relative_days(text, expected):
    assert day_of(text) == expected


# -- weekdays --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Today is Thursday, so the next Friday is tomorrow.
        ("friday", date(2026, 8, 7)),
        ("this friday", date(2026, 8, 7)),
        # "next friday" is the Friday of next week -- the documented rule, and
        # the one the README quotes as "due Friday the fourteenth".
        ("next friday", date(2026, 8, 14)),
        ("next monday", date(2026, 8, 10)),
        ("last friday", date(2026, 7, 31)),
        ("thursday", date(2026, 8, 6)),
        ("next thursday", date(2026, 8, 13)),
        ("tues", date(2026, 8, 11)),
    ],
)
def test_weekdays(text, expected):
    assert day_of(text) == expected


def test_this_weekday_already_gone_asks():
    # Sunday 9 August: "this thursday" would point backwards.
    sunday = datetime(2026, 8, 9, 10, 0, tzinfo=TZ)
    with pytest.raises(AmbiguousTime) as excinfo:
        resolve_instant("this thursday", sunday, TZ)
    assert "coming Thursday" in excinfo.value.question


# -- calendar dates --------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-12-25", date(2026, 12, 25)),
        ("25/12", date(2026, 12, 25)),
        ("25/12/2026", date(2026, 12, 25)),
        ("14 august", date(2026, 8, 14)),
        ("august 14", date(2026, 8, 14)),
        ("the 14th of august", date(2026, 8, 14)),
        ("14th august", date(2026, 8, 14)),
        ("the fourteenth of august", date(2026, 8, 14)),
        ("sept 3", date(2026, 9, 3)),
        # Day-of-month only: this month if still ahead, otherwise next.
        ("the 14th", date(2026, 8, 14)),
        ("on the 3rd", date(2026, 9, 3)),
        ("the twenty first", date(2026, 8, 21)),
    ],
)
def test_calendar_dates(text, expected):
    assert day_of(text) == expected


def test_numeric_dates_are_day_month():
    assert day_of("3/9") == date(2026, 9, 3)


def test_bare_month_day_already_past_rolls_to_next_year():
    assert day_of("1 march") == date(2027, 3, 1)


def test_impossible_date_asks():
    with pytest.raises(UnknownTime):
        resolve_instant("31 february", NOW, TZ)


def test_end_of_week_and_month():
    assert day_of("end of the week") == date(2026, 8, 9)
    assert day_of("end of the month") == date(2026, 8, 31)


# -- times of day ----------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("tomorrow at 3pm", time(15, 0)),
        ("tomorrow at 9am", time(9, 0)),
        ("tomorrow at 15:00", time(15, 0)),
        ("tomorrow at 3:30pm", time(15, 30)),
        ("tomorrow at noon", time(12, 0)),
        ("tomorrow at midnight", time(0, 0)),
        ("tomorrow 9am", time(9, 0)),
        ("tomorrow at 12pm", time(12, 0)),
        ("tomorrow at 12am", time(0, 0)),
    ],
)
def test_times(text, expected):
    assert resolve_instant(text, NOW, TZ).dt.time() == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("nine am", time(9, 0)),
        ("eleven am", time(11, 0)),
        ("at nine am", time(9, 0)),
        ("two pm", time(14, 0)),
        ("twelve pm", time(12, 0)),
        ("noon", time(12, 0)),
        ("9am", time(9, 0)),
        ("15:30", time(15, 30)),
    ],
)
def test_spoken_clock_times(text, expected):
    """ASR gives words, not digits: "nine am", not "9am"."""
    assert timeparse.resolve_time_of_day(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("thursdays from nine am to eleven am", time(9, 0)),
        ("tomorrow at nine am", time(9, 0)),
        ("friday at eight pm", time(20, 0)),
    ],
)
def test_spoken_times_inside_a_sentence(text, expected):
    assert resolve_instant(text, NOW, TZ).dt.time() == expected


def test_a_spoken_hour_with_no_marker_still_asks():
    with pytest.raises(AmbiguousTime):
        timeparse.resolve_time_of_day("nine")


@pytest.mark.parametrize("text", ["tomorrow at 3", "tomorrow at three", "friday at 9 oclock"])
def test_bare_hour_is_ambiguous(text):
    """Invariant #6: 3 could be either end of the day, so ask."""
    with pytest.raises(AmbiguousTime) as excinfo:
        resolve_instant(text, NOW, TZ)
    assert "morning" in excinfo.value.question


def test_time_without_date_is_today_or_tomorrow():
    assert resolve_instant("at 3pm", NOW, TZ).dt == datetime(2026, 8, 6, 15, 0, tzinfo=TZ)
    # 9am has already gone at 10:00, so it means tomorrow.
    assert resolve_instant("at 9am", NOW, TZ).dt == datetime(2026, 8, 7, 9, 0, tzinfo=TZ)


def test_default_time_fills_in_and_is_flagged():
    resolved = resolve_instant("next friday", NOW, TZ)
    assert resolved.dt.time() == timeparse.END_OF_DAY
    assert resolved.explicit_time is False

    spoken = resolve_instant("next friday at 5pm", NOW, TZ)
    assert spoken.dt.time() == time(17, 0)
    assert spoken.explicit_time is True


def test_reminder_default_is_the_morning():
    resolved = resolve_instant("tomorrow", NOW, TZ, default_time=timeparse.MORNING)
    assert resolved.dt.time() == time(9, 0)


def test_sub_day_offsets():
    assert resolve_instant("in two hours", NOW, TZ).dt == datetime(2026, 8, 6, 12, 0, tzinfo=TZ)
    assert resolve_instant("in 30 minutes", NOW, TZ).dt == datetime(2026, 8, 6, 10, 30, tzinfo=TZ)


def test_resolve_time_of_day():
    assert timeparse.resolve_time_of_day("9am") == time(9, 0)
    assert timeparse.resolve_time_of_day("15:30") == time(15, 30)
    assert timeparse.resolve_time_of_day("noon") == time(12, 0)
    with pytest.raises(AmbiguousTime):
        timeparse.resolve_time_of_day("9")


# -- refusing to guess -----------------------------------------------------


@pytest.mark.parametrize("text", ["soon", "later", "sometime", "whenever", "at some point"])
def test_vague_expressions_ask(text):
    with pytest.raises(AmbiguousTime) as excinfo:
        resolve_instant(text, NOW, TZ)
    assert excinfo.value.question.endswith("?")


@pytest.mark.parametrize("text", ["next week", "this month", "next semester"])
def test_a_span_used_as_a_day_asks(text):
    with pytest.raises(AmbiguousTime) as excinfo:
        resolve_instant(text, NOW, TZ)
    assert "Which day" in excinfo.value.question


def test_nonsense_asks_rather_than_crashing():
    with pytest.raises(UnknownTime):
        resolve_instant("the fnord of blimp", NOW, TZ)
    with pytest.raises(UnknownTime):
        resolve_instant("", NOW, TZ)


def test_naive_now_is_rejected():
    with pytest.raises(ValueError):
        resolve_instant("tomorrow", datetime(2026, 8, 6, 10, 0), TZ)


# -- intervals -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,start,end,label",
    [
        ("today", date(2026, 8, 6), date(2026, 8, 7), "today"),
        ("tomorrow", date(2026, 8, 7), date(2026, 8, 8), "tomorrow"),
        ("yesterday", date(2026, 8, 5), date(2026, 8, 6), "yesterday"),
        # Weeks start Monday: this week is 3-9 August.
        ("this week", date(2026, 8, 3), date(2026, 8, 10), "this week"),
        ("next week", date(2026, 8, 10), date(2026, 8, 17), "next week"),
        ("last week", date(2026, 7, 27), date(2026, 8, 3), "last week"),
        ("the weekend", date(2026, 8, 8), date(2026, 8, 10), "the weekend"),
        ("this month", date(2026, 8, 1), date(2026, 9, 1), "this month"),
        ("next month", date(2026, 9, 1), date(2026, 10, 1), "next month"),
    ],
)
def test_intervals(text, start, end, label):
    interval = resolve_interval(text, NOW, TZ)
    assert interval.start.date() == start
    assert interval.end.date() == end
    assert interval.label == label


def test_interval_defaults_to_today():
    assert resolve_interval(None, NOW, TZ).label == "today"
    assert resolve_interval("", NOW, TZ).label == "today"


def test_interval_next_n_days():
    interval = resolve_interval("the next three days", NOW, TZ)
    assert interval.start.date() == date(2026, 8, 6)
    assert interval.end.date() == date(2026, 8, 9)


def test_interval_of_a_single_named_day():
    interval = resolve_interval("next friday", NOW, TZ)
    assert interval.start.date() == date(2026, 8, 14)
    assert interval.end.date() == date(2026, 8, 15)


def test_interval_is_half_open():
    interval = resolve_interval("today", NOW, TZ)
    assert interval.contains(NOW)
    assert not interval.contains(interval.end)


def test_term_dates_are_an_open_question_not_a_guess():
    with pytest.raises(AmbiguousTime) as excinfo:
        resolve_interval("this semester", NOW, TZ)
    assert "term dates" in excinfo.value.question


# -- storage round trip ----------------------------------------------------


def test_utc_iso_round_trip():
    resolved = resolve_instant("next friday at 5pm", NOW, TZ)
    assert resolved.utc_iso.startswith("2026-08-14T07:00")  # Sydney is UTC+10 in August


def test_parse_weekday():
    assert timeparse.parse_weekday("Tuesday") == 1
    assert timeparse.parse_weekday("thurs") == 3
    assert timeparse.weekday_name(3) == "Thursday"
    with pytest.raises(UnknownTime):
        timeparse.parse_weekday("someday")

"""The thirteen v1 tools from PLAN.md section 3.

Every one of these takes natural-language time expressions rather than ISO
strings, resolves them through :mod:`src.timeparse` against the injected
``now``, and either returns a result or raises for a clarification. None of
them reads the clock, and none of them writes to SQLite except through
:mod:`src.store.audit`.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from src import timeparse
from src.store import audit
from src.tools.context import ToolContext
from src.tools.errors import NeedsClarification, NeedsConfirmation, ToolError
from src.tools.match import resolve_assignment, resolve_course

_ACTIVE_STATUSES = ("todo", "in_progress")
_ALL_STATUSES = ("todo", "in_progress", "done", "dropped")

_CLASS_TIME_FIELDS = ("weekday", "start_time", "end_time")

#: Loose field names a user might say, mapped onto columns.
_COURSE_FIELD_ALIASES = {
    "start": "start_time", "start_time": "start_time",
    "end": "end_time", "end_time": "end_time",
    "day": "weekday", "weekday": "weekday",
    "room": "location", "location": "location", "place": "location",
    "title": "title", "name": "title", "code": "code",
    "instructor": "instructor", "lecturer": "instructor", "teacher": "instructor",
    "notes": "notes", "note": "notes",
    "term_start": "term_start", "term_end": "term_end",
}

_ASSIGNMENT_FIELD_ALIASES = {
    "title": "title", "name": "title",
    "due": "due_at", "due_at": "due_at", "deadline": "due_at",
    "est_hours": "est_hours", "hours": "est_hours", "estimate": "est_hours",
    "progress": "progress_pct", "progress_pct": "progress_pct", "percent": "progress_pct",
    "status": "status",
    "notes": "notes", "note": "notes",
    "course": "course_id", "class": "course_id", "course_id": "course_id",
}


# --------------------------------------------------------------------------
# storage helpers
# --------------------------------------------------------------------------


def _to_utc_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _from_utc_iso(value: str, tz) -> datetime:
    return datetime.fromisoformat(value).astimezone(tz)


def _hhmm(value: time) -> str:
    return value.strftime("%H:%M")


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _iso_date(value: date) -> str:
    return value.isoformat()


def _percent(value: Any) -> int:
    if isinstance(value, (int, float)):
        number = int(value)
    else:
        text = str(value).strip().lower().replace("%", "").replace("percent", "").strip()
        try:
            number = int(float(text))
        except ValueError:
            parsed = timeparse.number_from_words(text)
            if parsed is None:
                raise NeedsClarification(f"What percentage is “{value}”?") from None
            number = parsed
    if not 0 <= number <= 100:
        raise NeedsClarification("Progress has to be between nought and a hundred — what is it?")
    return number


def _hours(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    text = text.replace("hours", "").replace("hour", "").replace("hrs", "").replace("hr", "")
    text = text.replace("about", "").replace("around", "").strip()
    try:
        return float(text)
    except ValueError:
        parsed = timeparse.number_from_words(text)
        if parsed is None:
            raise NeedsClarification(f"How many hours is “{value}”?") from None
        return float(parsed)


def _hours_left(row: dict[str, Any]) -> float | None:
    """Derived, never stored -- PLAN.md section 2."""
    if row.get("est_hours") is None:
        return None
    return round(float(row["est_hours"]) * (1 - (row.get("progress_pct") or 0) / 100), 2)


def _course_label(row: dict[str, Any]) -> str:
    return str(row["code"]) if not row.get("title") else f"{row['code']} {row['title']}"


def _parse_term(term: str, ctx: ToolContext) -> tuple[str | None, str | None]:
    """"18 august to 27 october" -> two local dates."""
    lowered = timeparse.normalise(term)
    for separator in (" to ", " until ", " till ", " through ", " - "):
        if separator in lowered:
            first, second = lowered.split(separator, 1)
            start = timeparse.resolve_instant(first, ctx.now, ctx.tz)
            end = timeparse.resolve_instant(second, ctx.now, ctx.tz)
            return _iso_date(start.dt.date()), _iso_date(end.dt.date())
    raise NeedsClarification(
        f"I could not read “{term}” as a term — when does it start and finish?"
    )


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------


def get_now(ctx: ToolContext) -> dict[str, Any]:
    local = ctx.now.astimezone(ctx.tz)
    return {
        "iso": local.isoformat(),
        "date": _iso_date(local.date()),
        "time": _hhmm(local.time()),
        "weekday": timeparse.weekday_name(local.weekday()),
        "timezone": str(ctx.tz),
    }


def list_schedule(ctx: ToolContext, when: str | None = "today") -> dict[str, Any]:
    """Expand recurring classes across a span, applying per-date exceptions."""
    interval = timeparse.resolve_interval(when, ctx.now, ctx.tz)
    courses = ctx.conn.execute("SELECT * FROM course WHERE deleted_at IS NULL").fetchall()

    items: list[dict[str, Any]] = []
    day = interval.start.date()
    last = (interval.end - timedelta(seconds=1)).date()
    while day <= last:
        for course in courses:
            if course["weekday"] != day.weekday():
                continue
            if course["term_start"] and _iso_date(day) < course["term_start"]:
                continue
            if course["term_end"] and _iso_date(day) > course["term_end"]:
                continue

            exception = ctx.conn.execute(
                "SELECT * FROM course_exception"
                " WHERE course_id = ? AND date = ? AND deleted_at IS NULL"
                " ORDER BY id DESC LIMIT 1",
                (course["id"], _iso_date(day)),
            ).fetchone()

            item = {
                "course_id": course["id"],
                "code": course["code"],
                "title": course["title"],
                "date": _iso_date(day),
                "weekday": timeparse.weekday_name(day.weekday()),
                "start": course["start_time"],
                "end": course["end_time"],
                "location": course["location"],
                "status": "scheduled",
                "note": None,
            }
            if exception:
                item["status"] = exception["kind"]
                item["note"] = exception["note"]
                if exception["new_start"]:
                    item["start"] = exception["new_start"]
                if exception["new_end"]:
                    item["end"] = exception["new_end"]
                if exception["new_location"]:
                    item["location"] = exception["new_location"]
            items.append(item)
        day += timedelta(days=1)

    items.sort(key=lambda entry: (entry["date"], entry["start"] or ""))
    return {"when": interval.label, "count": len(items), "items": items}


def list_assignments(
    ctx: ToolContext,
    status: str | None = None,
    due_before: str | None = None,
    course: str | int | None = None,
) -> dict[str, Any]:
    """Open assignments by default.

    PLAN.md leaves "should completed assignments stay visible" open; the answer
    here is that they do not, unless asked for by status. Finished work is
    noise in a spoken answer.
    """
    sql = "SELECT * FROM assignment WHERE deleted_at IS NULL"
    params: list[Any] = []

    if status in (None, "", "open", "active"):
        sql += f" AND status IN ({','.join('?' * len(_ACTIVE_STATUSES))})"
        params.extend(_ACTIVE_STATUSES)
    elif status not in ("all", "any"):
        if status not in _ALL_STATUSES:
            raise NeedsClarification(
                f"“{status}” is not a status I keep — todo, in progress, done or dropped?"
            )
        sql += " AND status = ?"
        params.append(status)

    resolved_course = None
    if course not in (None, ""):
        resolved_course = resolve_course(ctx.conn, course)
        sql += " AND course_id = ?"
        params.append(resolved_course["id"])

    cutoff_label = None
    if due_before not in (None, ""):
        # A span is the normal way to say this -- "due before this week" means
        # by the end of it. resolve_interval handles both, falling back to the
        # whole day for anything that names a single date.
        cutoff = timeparse.resolve_interval(due_before, ctx.now, ctx.tz)
        cutoff_label = cutoff.label
        sql += " AND due_at IS NOT NULL AND due_at <= ?"
        params.append(_to_utc_iso(cutoff.end))

    sql += " ORDER BY due_at IS NULL, due_at"
    rows = [dict(row) for row in ctx.conn.execute(sql, params).fetchall()]

    items = []
    for row in rows:
        local_due = _from_utc_iso(row["due_at"], ctx.tz) if row["due_at"] else None
        items.append(
            {
                "id": row["id"],
                "title": row["title"],
                "course_id": row["course_id"],
                "due": local_due.isoformat() if local_due else None,
                "due_date": _iso_date(local_due.date()) if local_due else None,
                "status": row["status"],
                "progress_pct": row["progress_pct"],
                "est_hours": row["est_hours"],
                "hours_left": _hours_left(row),
            }
        )

    return {
        "count": len(items),
        "filter": {
            "status": status or "open",
            "due_before": cutoff_label,
            "course": _course_label(resolved_course) if resolved_course else None,
        },
        "items": items,
    }


# --------------------------------------------------------------------------
# class writes
# --------------------------------------------------------------------------


def add_class(
    ctx: ToolContext,
    code: str,
    weekday: str,
    start: str,
    end: str,
    title: str | None = None,
    location: str | None = None,
    term: str | None = None,
    instructor: str | None = None,
) -> dict[str, Any]:
    if not (code or "").strip():
        raise NeedsClarification("Which class is it — what is the course code?")

    day_index = timeparse.parse_weekday(weekday)
    start_time = timeparse.resolve_time_of_day(start)
    end_time = timeparse.resolve_time_of_day(end)
    if end_time <= start_time:
        raise NeedsClarification(
            f"You said {_hhmm(start_time)} to {_hhmm(end_time)}, which runs backwards. "
            "When does it finish?"
        )

    existing = ctx.conn.execute(
        "SELECT * FROM course WHERE deleted_at IS NULL AND lower(code) = lower(?)",
        (code.strip(),),
    ).fetchall()
    if existing and not ctx.confirmed:
        current = existing[0]
        raise NeedsClarification(
            f"I already have {current['code']} on "
            f"{timeparse.weekday_name(current['weekday'])} at {current['start_time']}. "
            "Is this a second session, or should I change that one?"
        )

    term_start = term_end = None
    if term:
        term_start, term_end = _parse_term(term, ctx)

    row_id = audit.insert(
        ctx.conn,
        "course",
        {
            "code": code.strip(),
            "title": title,
            "instructor": instructor,
            "location": location,
            "weekday": day_index,
            "start_time": _hhmm(start_time),
            "end_time": _hhmm(end_time),
            "term_start": term_start,
            "term_end": term_end,
            "notes": None,
        },
        ctx.turn_id,
    )
    return {
        "created": "course",
        "id": row_id,
        "code": code.strip(),
        "weekday": timeparse.weekday_name(day_index),
        "start": _hhmm(start_time),
        "end": _hhmm(end_time),
        "location": location,
    }


def update_class(
    ctx: ToolContext, course: str | int, fields: dict[str, Any] | None = None, **loose: Any
) -> dict[str, Any]:
    target = resolve_course(ctx.conn, course)
    requested = {**(fields or {}), **loose}
    if not requested:
        raise NeedsClarification(f"What should I change about {_course_label(target)}?")

    changes: dict[str, Any] = {}
    for key, value in requested.items():
        column = _COURSE_FIELD_ALIASES.get(key.lower())
        if column is None:
            raise NeedsClarification(f"I do not keep a “{key}” for a class. What should I change?")
        if column == "weekday":
            changes[column] = timeparse.parse_weekday(str(value))
        elif column in ("start_time", "end_time"):
            changes[column] = _hhmm(timeparse.resolve_time_of_day(str(value)))
        elif column in ("term_start", "term_end"):
            changes[column] = _iso_date(
                timeparse.resolve_instant(str(value), ctx.now, ctx.tz).dt.date()
            )
        else:
            changes[column] = value

    merged = {**target, **changes}
    if _parse_hhmm(merged["end_time"]) <= _parse_hhmm(merged["start_time"]):
        raise NeedsClarification(
            f"That would put {_course_label(target)} finishing before it starts. "
            "What are the times?"
        )

    # Invariant #5: overwriting a class time is destructive; confirm it first.
    overwrites = [
        column
        for column in _CLASS_TIME_FIELDS
        if column in changes and target[column] is not None and changes[column] != target[column]
    ]
    if overwrites and not ctx.confirmed:
        raise NeedsConfirmation(
            f"{_course_label(target)} is currently "
            f"{timeparse.weekday_name(target['weekday'])} "
            f"{target['start_time']} to {target['end_time']}. Change it?",
            {"tool": "update_class", "args": {"course": target["id"], "fields": requested}},
        )

    after = audit.update(ctx.conn, "course", target["id"], changes, ctx.turn_id)
    return {
        "updated": "course",
        "id": target["id"],
        "code": after.get("code"),
        "changed": sorted(changes),
        "weekday": timeparse.weekday_name(after["weekday"]),
        "start": after.get("start_time"),
        "end": after.get("end_time"),
        "location": after.get("location"),
    }


def cancel_class(
    ctx: ToolContext,
    course: str | int,
    date: str,
    kind: str = "cancelled",
    details: str | None = None,
    new_start: str | None = None,
    new_end: str | None = None,
    new_location: str | None = None,
) -> dict[str, Any]:
    target = resolve_course(ctx.conn, course)

    kind = (kind or "cancelled").strip().lower().replace(" ", "_")
    if kind in ("cancel", "cancelled", "canceled", "off"):
        kind = "cancelled"
    elif kind in ("moved", "move", "rescheduled"):
        kind = "moved"
    elif kind in ("room_change", "room", "relocated"):
        kind = "room_change"
    else:
        raise NeedsClarification(
            f"Is {_course_label(target)} cancelled, moved, or just in a different room?"
        )

    when = timeparse.resolve_instant(date, ctx.now, ctx.tz)
    on_date = when.dt.date()
    if on_date.weekday() != target["weekday"] and kind != "moved":
        raise NeedsClarification(
            f"{_course_label(target)} does not meet on a "
            f"{timeparse.weekday_name(on_date.weekday())}. Which date did you mean?"
        )

    start_value = end_value = None
    if kind == "moved":
        if not new_start:
            raise NeedsClarification(f"What time does {_course_label(target)} move to?")
        start_value = _hhmm(timeparse.resolve_time_of_day(new_start))
        if new_end:
            end_value = _hhmm(timeparse.resolve_time_of_day(new_end))
        else:
            # Keep the original duration rather than inventing an end time.
            original = datetime.combine(on_date, _parse_hhmm(target["start_time"]))
            length = datetime.combine(on_date, _parse_hhmm(target["end_time"])) - original
            end_value = _hhmm((datetime.combine(on_date, _parse_hhmm(start_value)) + length).time())

    if kind == "room_change" and not new_location:
        raise NeedsClarification(f"Which room is {_course_label(target)} in?")

    row_id = audit.insert(
        ctx.conn,
        "course_exception",
        {
            "course_id": target["id"],
            "date": _iso_date(on_date),
            "kind": kind,
            "new_start": start_value,
            "new_end": end_value,
            "new_location": new_location,
            "note": details,
        },
        ctx.turn_id,
    )
    return {
        "created": "course_exception",
        "id": row_id,
        "code": target["code"],
        "kind": kind,
        "date": _iso_date(on_date),
        "weekday": timeparse.weekday_name(on_date.weekday()),
        "start": start_value,
        "end": end_value,
        "location": new_location,
    }


def delete_class(ctx: ToolContext, course: str | int) -> dict[str, Any]:
    target = resolve_course(ctx.conn, course)
    if not ctx.confirmed:
        raise NeedsConfirmation(
            f"Delete {_course_label(target)}, "
            f"{timeparse.weekday_name(target['weekday'])} at {target['start_time']}?",
            {"tool": "delete_class", "args": {"course": target["id"]}},
        )
    audit.soft_delete(ctx.conn, "course", target["id"], ctx.turn_id)
    return {"deleted": "course", "id": target["id"], "code": target["code"]}


# --------------------------------------------------------------------------
# assignment writes
# --------------------------------------------------------------------------


def add_assignment(
    ctx: ToolContext,
    title: str,
    due: str,
    course: str | int | None = None,
    est_hours: Any = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if not (title or "").strip():
        raise NeedsClarification("What is the assignment called?")

    when = timeparse.resolve_instant(due, ctx.now, ctx.tz, default_time=timeparse.END_OF_DAY)

    # The course is optional, so a name that matches nothing must not sink the
    # write. The reply says the link was not made; the user can correct it.
    resolved_course = (
        resolve_course(ctx.conn, course, required=False) if course not in (None, "") else None
    )
    unlinked = course not in (None, "") and resolved_course is None

    row_id = audit.insert(
        ctx.conn,
        "assignment",
        {
            "course_id": resolved_course["id"] if resolved_course else None,
            "title": title.strip(),
            "due_at": _to_utc_iso(when.dt),
            "est_hours": _hours(est_hours) if est_hours not in (None, "") else None,
            "progress_pct": 0,
            "status": "todo",
            "notes": notes,
        },
        ctx.turn_id,
    )
    return {
        "created": "assignment",
        "id": row_id,
        "title": title.strip(),
        "due": when.dt.isoformat(),
        "due_date": _iso_date(when.dt.date()),
        "explicit_time": when.explicit_time,
        "overdue": when.dt < ctx.now.astimezone(ctx.tz),
        "course": _course_label(resolved_course) if resolved_course else None,
        "unlinked_course": str(course) if unlinked else None,
        "est_hours": _hours(est_hours) if est_hours not in (None, "") else None,
    }


def update_assignment(
    ctx: ToolContext, assignment: str | int, fields: dict[str, Any] | None = None, **loose: Any
) -> dict[str, Any]:
    target = resolve_assignment(ctx.conn, assignment)
    requested = {**(fields or {}), **loose}
    if not requested:
        raise NeedsClarification(f"What should I change about {target['title']}?")

    changes: dict[str, Any] = {}
    for key, value in requested.items():
        column = _ASSIGNMENT_FIELD_ALIASES.get(key.lower())
        if column is None:
            raise NeedsClarification(
                f"I do not keep a “{key}” for an assignment. What should I change?"
            )
        if column == "due_at":
            resolved = timeparse.resolve_instant(
                str(value), ctx.now, ctx.tz, default_time=timeparse.END_OF_DAY
            )
            changes[column] = _to_utc_iso(resolved.dt)
        elif column == "est_hours":
            changes[column] = _hours(value)
        elif column == "progress_pct":
            changes[column] = _percent(value)
        elif column == "course_id":
            changes[column] = resolve_course(ctx.conn, value)["id"]
        elif column == "status":
            if str(value) not in _ALL_STATUSES:
                raise NeedsClarification(
                    f"“{value}” is not a status — todo, in progress, done or dropped?"
                )
            changes[column] = str(value)
        else:
            changes[column] = value

    if changes.get("progress_pct") == 100:
        changes["status"] = "done"

    # Invariant #5: overwriting a due date that already exists is destructive.
    if (
        "due_at" in changes
        and target["due_at"]
        and changes["due_at"] != target["due_at"]
        and not ctx.confirmed
    ):
        current = _from_utc_iso(target["due_at"], ctx.tz)
        raise NeedsConfirmation(
            f"{target['title']} is currently due {current.strftime('%A %d %B')}. Move it?",
            {
                "tool": "update_assignment",
                "args": {"assignment": target["id"], "fields": requested},
            },
        )

    after = audit.update(ctx.conn, "assignment", target["id"], changes, ctx.turn_id)
    due_local = _from_utc_iso(after["due_at"], ctx.tz) if after.get("due_at") else None
    return {
        "updated": "assignment",
        "id": target["id"],
        "title": after.get("title"),
        "changed": sorted(changes),
        "due": due_local.isoformat() if due_local else None,
        "due_date": _iso_date(due_local.date()) if due_local else None,
        "status": after.get("status"),
        "progress_pct": after.get("progress_pct"),
        "hours_left": _hours_left(after),
    }


def set_progress(ctx: ToolContext, assignment: str | int, percent: Any) -> dict[str, Any]:
    target = resolve_assignment(ctx.conn, assignment)
    value = _percent(percent)

    changes: dict[str, Any] = {"progress_pct": value}
    if value == 100:
        changes["status"] = "done"
    elif value > 0 and target["status"] == "todo":
        changes["status"] = "in_progress"

    after = audit.update(ctx.conn, "assignment", target["id"], changes, ctx.turn_id)
    return {
        "updated": "assignment",
        "id": target["id"],
        "title": after.get("title"),
        "progress_pct": after.get("progress_pct"),
        "status": after.get("status"),
        "hours_left": _hours_left(after),
    }


def delete_assignment(ctx: ToolContext, assignment: str | int) -> dict[str, Any]:
    target = resolve_assignment(ctx.conn, assignment)
    if not ctx.confirmed:
        raise NeedsConfirmation(
            f"Delete {target['title']}?",
            {"tool": "delete_assignment", "args": {"assignment": target["id"]}},
        )
    audit.soft_delete(ctx.conn, "assignment", target["id"], ctx.turn_id)
    return {"deleted": "assignment", "id": target["id"], "title": target["title"]}


# --------------------------------------------------------------------------
# reminders and meta
# --------------------------------------------------------------------------


def add_reminder(
    ctx: ToolContext,
    title: str,
    when: str,
    related: str | int | None = None,
    related_type: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if not (title or "").strip():
        raise NeedsClarification("What should I remind you about?")

    moment = timeparse.resolve_instant(when, ctx.now, ctx.tz, default_time=timeparse.MORNING)

    related_id = None
    kind = None
    if related not in (None, ""):
        if related_type == "course":
            related_id, kind = resolve_course(ctx.conn, related)["id"], "course"
        elif related_type == "assignment":
            related_id, kind = resolve_assignment(ctx.conn, related)["id"], "assignment"
        else:
            try:
                related_id, kind = resolve_assignment(ctx.conn, related)["id"], "assignment"
            except NeedsClarification:
                related_id, kind = resolve_course(ctx.conn, related)["id"], "course"

    row_id = audit.insert(
        ctx.conn,
        "reminder",
        {
            "title": title.strip(),
            "remind_at": _to_utc_iso(moment.dt),
            "related_type": kind,
            "related_id": related_id,
            "notes": notes,
        },
        ctx.turn_id,
    )
    return {
        "created": "reminder",
        "id": row_id,
        "title": title.strip(),
        "when": moment.dt.isoformat(),
        "date": _iso_date(moment.dt.date()),
        "explicit_time": moment.explicit_time,
    }


def undo_last_write(ctx: ToolContext) -> dict[str, Any]:
    try:
        result = audit.undo_last(ctx.conn, ctx.turn_id)
    except audit.UndoError as exc:
        raise ToolError(str(exc)) from None
    return {"undone": result["reversed"], "description": result["description"]}

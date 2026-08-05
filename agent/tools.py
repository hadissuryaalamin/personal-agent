"""Tools the model may call, and the code that runs them.

The model never touches a file. It returns a request — a name and arguments —
and everything here runs on our side. That boundary is the whole point: the
model decides *what* it needs, the code decides *what it is allowed to do*.

THE DESCRIPTION IS THE PROMPT. A tool is chosen by matching the question
against its `description`, so the description has to name the triggers, not
just the function. Measured on qwen2.5:7b with the wording below: 5/5 questions
that should call it did, and 4/4 that should not stayed away. A lazy
`"gets schedule"` misses far more often.

Keeping every definition in one place also keeps them honest — a tool that
exists here but has no runner, or the reverse, fails loudly at import rather
than at the moment you ask for it.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta

from . import config, events

log = logging.getLogger(__name__)

# How far ahead to look when the model does not say.
DEFAULT_DAYS = 14
# Rows sent back per call. A voice answer cannot use fifty of them, and every
# row costs context.
MAX_ROWS = 30


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_schedule",
            "description": (
                "Read the user's real schedule: classes, tasks with deadlines, "
                "and reminders. Call this for any question about what they have "
                "on, what is due, when a class is, where a class is, or how much "
                "work is left. Never answer those from memory. Entries marked "
                "already_finished have already happened today - skip them when "
                "asked what is NEXT."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["class", "task", "reminder", "all"],
                        "description": (
                            "class = lectures and tutorials, task = assignments "
                            "with deadlines, reminder = one-off things to "
                            "remember, all = everything"
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": (
                            "How many days ahead to look. 1 = today only, "
                            "7 = this week, 14 = the next two weeks."
                        ),
                    },
                },
                "required": ["kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_next",
            "description": (
                "What is coming up NEXT, from right now onwards. Call this for "
                "'what's next', 'when is my next class', 'where is my next "
                "lecture', 'what's coming up'. Anything already finished today "
                "is excluded, so you never have to work that out yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["class", "task", "reminder", "all"],
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many upcoming entries, default 1",
                    },
                },
                "required": ["kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": (
                "Save a new assignment or piece of work with a deadline. Call "
                "this when the user says they have something due, or asks you "
                "to note a task down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short name"},
                    "due": {"type": "string", "description": "Deadline, YYYY-MM-DD"},
                    "course": {"type": "string", "description": "Course code, e.g. ENGN4122"},
                    "estimate_hours": {
                        "type": "number",
                        "description": "Rough hours of work it will take",
                    },
                },
                "required": ["title", "due"],
            },
        },
    },
]


# --- Runners ----------------------------------------------------------------


def _is_past(e: dict, now) -> bool:
    """Has this already finished? Uses `end` when there is one.

    Computed here rather than left to the model. Asked "where is my next
    class?" at 19:25, with today's 15:30-17:00 tutorial in the list and no
    marker on it, qwen answered "your next class is at 3:30 PM" — a class that
    had ended two and a half hours earlier. It had every timestamp it needed;
    comparing them is simply not something to trust it with.

    A boolean it only has to read is a different proposition from arithmetic it
    has to perform.
    """
    finish = e.get("end") or e.get("start")
    when = events.when({"start": finish})
    if when is None:
        return False
    # A date with no time covers the whole day, so it is only past once the day
    # is over — an assignment due today is not past at 09:00.
    if not events.has_time({"start": finish}):
        return when.date() < now.date()
    return when < now


def _row(e: dict, now) -> dict:
    """One entry, trimmed to what a spoken answer can use.

    Internal fields (`id`, `created`) are left out on purpose: they are noise
    in the context window, and a model that sees an id will eventually recite
    one out loud.
    """
    out = {
        "kind": e.get("kind"),
        "title": e.get("title"),
        "when": e.get("start"),
    }
    for key in ("end", "course", "session", "location"):
        if e.get(key):
            out[key] = e[key]
    left = events.remaining(e)
    if left is not None:
        out["hours_left"] = left
    if e.get("done"):
        out["done"] = True
    if _is_past(e, now):
        out["already_finished"] = True
    return out


def get_schedule(kind: str = "all", days: int | None = None) -> str:
    days = int(days or DEFAULT_DAYS)
    now = events.now()
    start = events.today()
    end = start + timedelta(days=max(1, days))
    kinds = None if kind in (None, "", "all") else (kind,)

    rows = events.between(start, end, kinds=kinds)
    rows = [e for e in rows if not e.get("done")][:MAX_ROWS]

    # `now` and the date range travel with the rows. Without the range the model
    # cannot tell "nothing this week" from "nothing at all", and it guesses.
    # Without `now` it cannot tell which of today's entries have already been
    # and gone — see _is_past().
    return json.dumps({
        "now": now.strftime("%Y-%m-%dT%H:%M"),
        "today": str(start),
        "range_end": str(end),
        "count": len(rows),
        "events": [_row(e, now) for e in rows],
    })


def get_next(kind: str = "all", count: int | None = None) -> str:
    """Strictly what is ahead. A separate tool rather than a flag on
    get_schedule, because the filtering happens HERE.

    Marking rows `already_finished` and trusting the model to skip them was
    tried and failed: asked for the next class at 19:25 it still offered the
    15:30 tutorial, even with the marker present and the prompt telling it to
    skip such rows. Reading a marker is easy; acting on it while composing a
    sentence is evidently not.

    Choosing between two tools is a far easier decision than filtering a list,
    and it is a decision we can measure.
    """
    now = events.now()
    kinds = None if kind in (None, "", "all") else (kind,)
    rows = events.upcoming(limit=max(1, int(count or 1)), kinds=kinds)
    return json.dumps({
        "now": now.strftime("%Y-%m-%dT%H:%M"),
        "count": len(rows),
        "events": [_row(e, now) for e in rows],
    })


def add_task(
    title: str, due: str, course: str = "", estimate_hours: float = 0
) -> str:
    e = events.add(
        "task", title, due, course=course or "", estimate_hours=estimate_hours or 0
    )
    return json.dumps({"saved": True, "title": e["title"], "due": e["start"]})


RUNNERS = {
    "get_schedule": get_schedule,
    "get_next": get_next,
    "add_task": add_task,
}

# A definition with no runner would fail only when the model happened to pick
# it — which is to say, in front of the user.
_declared = {t["function"]["name"] for t in SCHEMA}
assert _declared == set(RUNNERS), (
    f"tool definitions and runners disagree: {_declared ^ set(RUNNERS)}"
)


def enabled() -> bool:
    return config.TOOLS_ENABLED


def run(name: str, arguments) -> str:
    """Execute one tool call. Never raises — the model gets the error as data.

    An exception here would kill the turn. Handing the failure back as a tool
    result lets the model say "I could not read your schedule", which is far
    better than silence.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
    arguments = arguments or {}

    fn = RUNNERS.get(name)
    if fn is None:
        log.warning("model asked for an unknown tool: %r", name)
        return json.dumps({"error": f"no such tool: {name}"})

    try:
        result = fn(**arguments)
        log.info("tool %s(%s) -> %d chars", name, arguments, len(result))
        return result
    except TypeError as e:
        # Wrong or missing arguments. Telling the model exactly what was wrong
        # lets it retry with the right shape.
        log.warning("tool %s got bad arguments %r: %s", name, arguments, e)
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
    except Exception as e:
        log.exception("tool %s failed", name)
        return json.dumps({"error": f"{name} failed: {e}"})


if __name__ == "__main__":
    #     .venv-agent\Scripts\python.exe -m agent.tools
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"tools declared: {sorted(_declared)}\n")
    for kind in ("all", "class", "task", "reminder"):
        out = json.loads(get_schedule(kind, 14))
        print(f"  get_schedule({kind!r}, 14) -> {out['count']} rows")
    print()
    print("  sample row:")
    sample = json.loads(get_schedule("all", 14))["events"]
    if sample:
        print("   ", json.dumps(sample[0]))

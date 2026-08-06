"""Every tool the agent has, defined once, reachable two ways.

This is the file to open when the question is "what can this thing do?".
Nothing else declares a capability.

    python -m src.tools                       # list them, then show the agenda
    python -m src.tools next class
    python -m src.tools add task "Assignment 1" 2026-08-14 --course ENGN4122
    python -m src.tools log "Assignment 1" 3
    python -m src.tools done "Assignment 1"
    python -m src.tools rm "Pay rego"

TWO DOORS, ONE REGISTRY

The list below produces both the JSON schema the model is given and the
argparse commands a terminal gets. They cannot drift apart, because there is
only one description of each tool and one runner behind it.

Before this, the store could do more than the voice could ask for: mark_done,
log_hours and remove existed in events.py and were wired to the keyboard only.
The system prompt carried a paragraph apologising for it. That gap is closed
here -- not by building anything new, but by hanging the missing four on the
door that already had a handle for the other three.

THE DESCRIPTION IS THE PROMPT. A tool is chosen by matching the question
against its `description`, so the description names the triggers, not just the
function. Measured on qwen2.5:7b with the get_schedule wording below: 5/5
questions that should call it did, and 4/4 that should not stayed away. A lazy
"gets schedule" misses far more often.

WHAT IS DELIBERATELY NOT HERE

An update/reschedule tool. events.py has no update operation -- only remove
plus add -- and inventing one while reorganising files is how a data store
quietly loses an entry. It is the next tool to build, not one to smuggle in.

WHY THERE IS NO free_time OR am_i_behind TOOL

Standing rule of this project: give the model a decision, never a calculation.
Choosing a tool and mapping "the next two weeks" to days=14 it does well.
Comparing timestamps, filtering a list, keeping a count it does badly. Every
one of those lives in Python -- which is why get_next exists as a tool of its
own rather than a flag on get_schedule.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable

from . import config, events

log = logging.getLogger(__name__)

# How far ahead to look when the model does not say.
DEFAULT_DAYS = 14
# Rows sent back per call. A voice answer cannot use fifty of them, and every
# row costs context.
MAX_ROWS = 30

KIND_HELP = ("class = lectures and tutorials, task = assignments with "
             "deadlines, reminder = one-off things to remember")


@dataclass(frozen=True)
class Param:
    name: str
    type: str                       # JSON schema type
    help: str
    enum: tuple[str, ...] | None = None
    required: bool = False
    positional: bool = False        # how the CLI takes it
    default: object = None


@dataclass(frozen=True)
class Tool:
    name: str                       # what the model calls
    command: str                    # what the terminal calls
    description: str                # THE PROMPT -- see the module docstring
    run: Callable[..., str]
    params: tuple[Param, ...] = ()
    help: str = ""                  # one line, for the CLI listing

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p.name: ({"type": p.type, "description": p.help}
                                 | ({"enum": list(p.enum)} if p.enum else {}))
                        for p in self.params
                    },
                    "required": [p.name for p in self.params if p.required],
                },
            },
        }


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

    Internal fields (`created`) are left out on purpose: they are noise in the
    context window. The id stays, because without it the model cannot name an
    entry to mark_done or remove_entry — but the system prompt tells it never
    to read one aloud.
    """
    out = {
        "id": e.get("id"),
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


def add_entry(kind: str, title: str, when: str, course: str = "",
              location: str = "", estimate_hours: float = 0) -> str:
    e = events.add(kind, title, when, course=course or "",
                   location=location or "", estimate_hours=estimate_hours or 0)
    return json.dumps({"saved": True, "id": e["id"], "kind": e["kind"],
                       "title": e["title"], "when": e["start"]})


def _resolve(what: str) -> tuple[dict | None, str | None]:
    """One entry from a fragment of its id or title.

    Ambiguity is reported, never guessed. This is shared by the three
    write tools because picking the wrong entry deletes the wrong thing, and
    the voice door is the one where the fragment arrives via speech-to-text.
    """
    hits = events.find(what)
    if not hits:
        return None, json.dumps({"error": f"nothing matches {what!r}"})
    if len(hits) > 1:
        return None, json.dumps({
            "error": f"{len(hits)} entries match {what!r}; be more specific",
            "candidates": [e["title"] for e in hits[:8]],
        })
    return hits[0], None


def mark_done(what: str) -> str:
    e, err = _resolve(what)
    if err:
        return err
    events.mark_done(e["id"])
    return json.dumps({"done": True, "title": e["title"]})


def log_hours(what: str, hours: float) -> str:
    e, err = _resolve(what)
    if err:
        return err
    updated = events.log_hours(e["id"], float(hours))
    return json.dumps({"logged": float(hours), "title": updated["title"],
                       "hours_left": events.remaining(updated)})


def remove_entry(what: str) -> str:
    e, err = _resolve(what)
    if err:
        return err
    events.remove(e["id"])
    return json.dumps({"removed": True, "title": e["title"]})


# --- The registry -----------------------------------------------------------


TOOLS: tuple[Tool, ...] = (
    Tool(
        name="get_schedule", command="list", help="read the schedule",
        description=(
            "Read the user's real schedule: classes, tasks with deadlines, "
            "and reminders. Call this for any question about what they have "
            "on, what is due, when a class is, where a class is, or how much "
            "work is left. Never answer those from memory. Entries marked "
            "already_finished have already happened today - skip them when "
            "asked what is NEXT. Call the tool immediately; never reply "
            "that you are about to check, or that you will look it up."
        ),
        run=get_schedule,
        params=(
            Param("kind", "string", KIND_HELP + ", all = everything",
                  enum=("class", "task", "reminder", "all"),
                  required=True, positional=True, default="all"),
            Param("days", "integer",
                  "How many days ahead to look. ALWAYS give a number. "
                  "1 = today, 2 = tomorrow, 7 = this week, 14 = the next two "
                  "weeks, 30 = this month, 90 = this semester. If unsure, use 30.",
                  positional=True, default=DEFAULT_DAYS),
        ),
    ),
    Tool(
        name="get_next", command="next", help="what is coming up next",
        description=(
            "What is coming up NEXT, from right now onwards. Call this for "
            "'what's next', 'when is my next class', 'where is my next "
            "lecture', 'what's coming up'. Anything already finished today "
            "is excluded, so you never have to work that out yourself."
        ),
        run=get_next,
        params=(
            Param("kind", "string", KIND_HELP + ", all = everything",
                  enum=("class", "task", "reminder", "all"),
                  required=True, positional=True, default="all"),
            Param("count", "integer", "How many upcoming entries, default 1",
                  positional=True, default=1),
        ),
    ),
    Tool(
        name="add_entry", command="add", help="save a class, task, or reminder",
        description=(
            "Save something new to the schedule: an assignment with a "
            "deadline, a class, or a reminder. Call this when the user says "
            "they have something due, that a class is on at some time, or "
            "asks you to note anything down."
        ),
        run=add_entry,
        params=(
            Param("kind", "string", KIND_HELP,
                  enum=("class", "task", "reminder"),
                  required=True, positional=True),
            Param("title", "string", "Short name", required=True, positional=True),
            Param("when", "string",
                  "YYYY-MM-DD, or YYYY-MM-DDTHH:MM when there is a time",
                  required=True, positional=True),
            Param("course", "string", "Course code, e.g. ENGN4122"),
            Param("location", "string", "Room or building, if known"),
            Param("estimate_hours", "number", "Rough hours of work it will take"),
        ),
    ),
    Tool(
        name="mark_done", command="done", help="mark an entry finished",
        description=(
            "Mark a task or reminder as finished. Call this when the user "
            "says they have done, finished, submitted or completed something. "
            "Identify it by any distinctive part of its title."
        ),
        run=mark_done,
        params=(
            Param("what", "string", "Part of the title, e.g. 'assignment one'",
                  required=True, positional=True),
        ),
    ),
    Tool(
        name="log_hours", command="log", help="record hours worked",
        description=(
            "Record hours the user has spent working on a task, so the hours "
            "remaining stay accurate. Call this when they say how long they "
            "worked on something. Use a negative number to correct a mistake."
        ),
        run=log_hours,
        params=(
            Param("what", "string", "Part of the title of the task",
                  required=True, positional=True),
            Param("hours", "number", "Hours worked; negative corrects a mis-log",
                  required=True, positional=True),
        ),
    ),
    Tool(
        name="remove_entry", command="rm", help="delete an entry",
        description=(
            "Delete an entry from the schedule entirely. This cannot be "
            "undone, so ALWAYS repeat back which entry you are about to "
            "delete and wait for the user to confirm before calling this. If "
            "they only want to mark something finished, use mark_done instead."
        ),
        run=remove_entry,
        params=(
            Param("what", "string", "Part of the title of the entry",
                  required=True, positional=True),
        ),
    ),
)

BY_NAME = {t.name: t for t in TOOLS}
SCHEMA = [t.schema() for t in TOOLS]


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

    tool = BY_NAME.get(name)
    if tool is None:
        log.warning("model asked for an unknown tool: %r", name)
        return json.dumps({"error": f"no such tool: {name}"})

    try:
        result = tool.run(**arguments)
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


# --- The keyboard door ------------------------------------------------------


def _cli() -> int:
    """The same tools, driven from a terminal.

    Exists so the store is reachable without the microphone, the models, or
    Ollama being up — and so a tool can be exercised directly when the
    question is whether the tool is broken or the model is.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="python -m src.tools")
    sub = ap.add_subparsers(dest="cmd")
    for tool in TOOLS:
        p = sub.add_parser(tool.command, help=tool.help)
        for param in tool.params:
            kw = {"help": param.help.split(".")[0]}
            if param.type == "integer":
                kw["type"] = int
            elif param.type == "number":
                kw["type"] = float
            if param.enum:
                kw["choices"] = list(param.enum)
            if param.positional:
                # Optional positionals need a default, or argparse demands them.
                if not param.required or param.default is not None:
                    kw["nargs"] = "?"
                    kw["default"] = param.default
                p.add_argument(param.name, **kw)
            else:
                kw["default"] = param.default if param.default is not None else ""
                p.add_argument(f"--{param.name.replace('_', '-')}", **kw)

    args = ap.parse_args()
    if args.cmd is None:
        print(f"\n  {len(TOOLS)} tools\n")
        for tool in TOOLS:
            names = " ".join(
                (p.name if p.required else f"[{p.name}]") for p in tool.params)
            print(f"  {tool.command:<6} {tool.name:<13} {names}")
        print(f"\n{events.path()}\n")
        print(events.agenda(DEFAULT_DAYS))
        return 0

    tool = next(t for t in TOOLS if t.command == args.cmd)
    kwargs = {p.name: getattr(args, p.name) for p in tool.params}
    kwargs = {k: v for k, v in kwargs.items() if v not in ("", None)}

    out = json.loads(tool.run(**kwargs))
    if "error" in out:
        print(f"  {out['error']}")
        for c in out.get("candidates", []):
            print(f"    {c}")
        return 1

    # Read tools return rows; write tools return a receipt. Print each in the
    # shape a human reads, not the shape the model gets.
    if "events" in out:
        print(f"\n  now {out['now']}   {out['count']} entries\n")
        for e in out["events"]:
            entry = events.find(e["id"])
            print("  " + (events.one_line(entry[0]) if entry else e["title"]))
        if not out["events"]:
            print("  nothing in that range")
    else:
        print("  " + json.dumps(out))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(_cli())

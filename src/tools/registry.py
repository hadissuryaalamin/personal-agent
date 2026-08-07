"""Tool schemas and the single dispatch point.

The schemas here are what the model will be shown at M1; the dispatcher is what
turns a tool's refusal to guess into the wire format from PLAN.md section 3, so
no individual tool has to remember it.

A tool returns either a result or ``{"needs": "clarification" | "confirmation"}``
-- never a guess, and never a bare exception for a foreseeable input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src import timeparse
from src.tools import schedule
from src.tools.context import ToolContext
from src.tools.errors import NeedsClarification, NeedsConfirmation, ToolError


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    required: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable[..., dict[str, Any]]
    description: str
    params: tuple[Param, ...] = ()
    writes: bool = False
    #: Destructive writes confirm first (invariant #5). Reads never do.
    destructive: bool = False

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)

    def json_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {"type": p.type, "description": p.description} for p in self.params
                },
                "required": list(self.required),
            },
        }


_WHEN = "natural language, e.g. “next Friday”, “tomorrow at 3pm” — never an ISO string"

TOOLS: dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> None:
    TOOLS[spec.name] = spec


_register(ToolSpec("get_now", schedule.get_now, "The current date and time."))

_register(
    ToolSpec(
        "list_schedule",
        schedule.list_schedule,
        "Classes on a day or across a span, with cancellations and room changes applied.",
        (Param("when", "string", f"which day or span — {_WHEN}"),),
    )
)

_register(
    ToolSpec(
        "list_assignments",
        schedule.list_assignments,
        "Assignments, open ones by default.",
        (
            Param("status", "string", "todo, in_progress, done, dropped, or all"),
            Param("due_before", "string", f"only those due by then — {_WHEN}"),
            Param("course", "string", "course code or name"),
        ),
    )
)

_register(
    ToolSpec(
        "add_class",
        schedule.add_class,
        "Add a weekly class.",
        (
            Param("code", "string", "course code, e.g. COMP4020", required=True),
            Param("weekday", "string", "day of the week", required=True),
            Param("start", "string", "start time, e.g. “9am”", required=True),
            Param("end", "string", "end time", required=True),
            Param("title", "string", "course title"),
            Param("location", "string", "room or building"),
            Param("term", "string", "term span, e.g. “18 August to 27 October”"),
            Param("instructor", "string", "who teaches it"),
        ),
        writes=True,
    )
)

_register(
    ToolSpec(
        "update_class",
        schedule.update_class,
        "Change a class's day, time, room, or details.",
        (
            Param("course", "string", "course code or name", required=True),
            Param("fields", "object", "the fields to change"),
        ),
        writes=True,
    )
)

_register(
    ToolSpec(
        "cancel_class",
        schedule.cancel_class,
        "Record a one-off cancellation, move, or room change for a date.",
        (
            Param("course", "string", "course code or name", required=True),
            Param("date", "string", f"which date — {_WHEN}", required=True),
            Param("kind", "string", "cancelled, moved, or room_change"),
            Param("details", "string", "free-text note"),
            Param("new_start", "string", "new start time, if moved"),
            Param("new_end", "string", "new end time, if moved"),
            Param("new_location", "string", "new room, if changed"),
        ),
        writes=True,
    )
)

_register(
    ToolSpec(
        "delete_class",
        schedule.delete_class,
        "Remove a class entirely.",
        (Param("course", "string", "course code or name", required=True),),
        writes=True,
        destructive=True,
    )
)

_register(
    ToolSpec(
        "add_assignment",
        schedule.add_assignment,
        "Add an assignment with a due date.",
        (
            Param("title", "string", "what it is called", required=True),
            Param("due", "string", f"when it is due — {_WHEN}", required=True),
            Param("course", "string", "course code or name"),
            Param("est_hours", "number", "estimated hours of work"),
            Param("notes", "string", "free-text note"),
        ),
        writes=True,
    )
)

_register(
    ToolSpec(
        "update_assignment",
        schedule.update_assignment,
        "Change an assignment's title, due date, estimate, status, or course.",
        (
            Param("assignment", "string", "which assignment", required=True),
            Param("fields", "object", "the fields to change"),
        ),
        writes=True,
    )
)

_register(
    ToolSpec(
        "set_progress",
        schedule.set_progress,
        "Set how far through an assignment you are. 100 marks it done.",
        (
            Param("assignment", "string", "which assignment", required=True),
            Param("percent", "number", "0 to 100", required=True),
        ),
        writes=True,
    )
)

_register(
    ToolSpec(
        "delete_assignment",
        schedule.delete_assignment,
        "Remove an assignment entirely.",
        (Param("assignment", "string", "which assignment", required=True),),
        writes=True,
        destructive=True,
    )
)

_register(
    ToolSpec(
        "add_reminder",
        schedule.add_reminder,
        "Add a one-off reminder. Reminders are read back on request, not fired.",
        (
            Param("title", "string", "what to remember", required=True),
            Param("when", "string", f"when — {_WHEN}", required=True),
            Param("related", "string", "a class or assignment it belongs to"),
            Param("related_type", "string", "course or assignment"),
            Param("notes", "string", "free-text note"),
        ),
        writes=True,
    )
)

_register(
    ToolSpec(
        "undo_last_write",
        schedule.undo_last_write,
        "Reverse the most recent change. Works for any write.",
        writes=True,
    )
)


def schemas() -> list[dict[str, Any]]:
    return [spec.json_schema() for spec in TOOLS.values()]


def call(name: str, args: dict[str, Any] | None, ctx: ToolContext) -> dict[str, Any]:
    """Run a tool, converting every foreseeable failure into a reply payload."""
    spec = TOOLS.get(name)
    if spec is None:
        return {
            "needs": "clarification",
            "question": f"I do not have a “{name}” tool. What would you like me to do?",
        }

    args = dict(args or {})
    args.pop("_confirmed", None)

    missing = [p for p in spec.required if args.get(p) in (None, "")]
    if missing:
        return {
            "needs": "clarification",
            "question": _missing_question(spec, missing),
            "tool": name,
        }

    try:
        result = spec.fn(ctx, **args)
    except NeedsConfirmation as exc:
        action = dict(exc.action)
        action.setdefault("tool", name)
        return {"needs": "confirmation", "question": exc.question, "resume": action}
    except NeedsClarification as exc:
        payload = {"needs": "clarification", "question": exc.question, "tool": name}
        if exc.options:
            payload["options"] = exc.options
        return payload
    except timeparse.TimeParseError as exc:
        # Invariant #6: a date we cannot pin down asks, it does not guess.
        return {"needs": "clarification", "question": exc.question, "tool": name}
    except ToolError as exc:
        return {"error": str(exc), "tool": name}
    except TypeError as exc:
        if "argument" not in str(exc):
            raise
        return {
            "needs": "clarification",
            "question": f"I did not understand the details for {name}. Could you say that again?",
            "tool": name,
        }

    result.setdefault("tool", name)
    return result


def _missing_question(spec: ToolSpec, missing: list[str]) -> str:
    lookup = {p.name: p for p in spec.params}
    if len(missing) == 1:
        param = lookup[missing[0]]
        if param.name in ("due", "when", "date"):
            return "When?"
        if param.name in ("course", "assignment"):
            return f"Which {param.name}?"
        return f"What is the {param.name.replace('_', ' ')}?"
    names = [m.replace("_", " ") for m in missing]
    return f"I need the {' and '.join((', '.join(names[:-1]), names[-1]))} — what are they?"

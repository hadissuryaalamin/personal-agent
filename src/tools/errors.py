"""How a tool declines to guess.

Invariant #6: two plausible course matches, a vague date, or a probe score in
the uncertainty band all produce one clarifying question, not a best guess.
Invariant #5: destructive writes are confirmed first; reads never are.

Tools raise these; the registry turns them into the ``{"needs": ...}`` payloads
described in PLAN.md section 3, so no tool has to remember the wire format.
"""

from __future__ import annotations

from typing import Any


class NeedsClarification(Exception):
    def __init__(self, question: str, options: list[str] | None = None) -> None:
        super().__init__(question)
        self.question = question
        self.options = options or []


class NeedsConfirmation(Exception):
    """Raised before a destructive write, with everything needed to resume."""

    def __init__(self, question: str, action: dict[str, Any] | None = None) -> None:
        super().__init__(question)
        self.question = question
        self.action = action or {}


class ToolError(Exception):
    """A foreseeable failure that is not the user's fault to fix by rephrasing."""

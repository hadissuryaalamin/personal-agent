"""What a tool needs to know besides its own arguments."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    #: Injected, never read from the clock inside a tool -- that is what makes
    #: every one of these testable.
    now: datetime
    tz: ZoneInfo
    #: The turn_log row this call belongs to, so audit entries can be grouped
    #: and undone together.
    turn_id: int | None = None
    #: Set when the user has answered a confirmation question with yes.
    confirmed: bool = False

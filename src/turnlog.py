"""Invariant #3: every turn writes a ``turn_log`` row.

Including errors, empty transcripts, and turns that ended in a clarification.
That table is the probe's training data and the only way to debug a system
whose input is sound.

The row is opened *before* the tool runs so that its id exists to group the
audit entries under, and so a crash mid-tool still leaves a record of what was
said. Never add a code path that answers the user without calling both halves.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from src.store.db import utc_now_iso

_FINISHABLE = (
    "transcript", "asr_conf", "probe_score", "probe_label", "tool_name",
    "tool_args_json", "tool_result_json", "reply_text", "ms_asr",
    "ms_prefill", "ms_gen", "ms_tts", "hidden_state_path",
)


def start_turn(
    conn: sqlite3.Connection,
    session_id: str,
    transcript: str | None,
    asr_conf: float | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO turn_log (session_id, ts, transcript, asr_conf) VALUES (?, ?, ?, ?)",
        (session_id, utc_now_iso(), transcript, asr_conf),
    )
    return int(cursor.lastrowid)


def record_timing(conn: sqlite3.Connection, turn_id: int, **timings: Any) -> None:
    """Update only the named timing columns, leaving everything else alone.

    ``finish_turn`` writes the whole row, so it cannot be used twice -- the
    second call would null out the tool and the reply. Speech-out finishes
    after the turn is already logged, so it needs this instead.
    """
    if not timings:
        return
    for key in timings:
        if key not in _FINISHABLE:
            raise ValueError(f"{key!r} is not a turn_log column")
    assignments = ", ".join(f"{column} = ?" for column in timings)
    conn.execute(
        f"UPDATE turn_log SET {assignments} WHERE id = ?", (*timings.values(), turn_id)
    )


def finish_turn(
    conn: sqlite3.Connection,
    turn_id: int,
    *,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    tool_result: dict[str, Any] | None = None,
    reply_text: str | None = None,
    **timings: Any,
) -> None:
    fields: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_args_json": json.dumps(tool_args) if tool_args is not None else None,
        "tool_result_json": json.dumps(tool_result, default=str)
        if tool_result is not None
        else None,
        "reply_text": reply_text,
    }
    for key, value in timings.items():
        if key not in _FINISHABLE:
            raise ValueError(f"{key!r} is not a turn_log column")
        fields[key] = value

    assignments = ", ".join(f"{column} = ?" for column in fields)
    conn.execute(
        f"UPDATE turn_log SET {assignments} WHERE id = ?", (*fields.values(), turn_id)
    )

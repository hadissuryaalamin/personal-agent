"""Every mutation in the system goes through this module.

Invariant #4: writes go through the audit log and deletes are soft, so
``undo_last_write`` can reverse any mutation. A misheard command must be
reversible, which means no tool is allowed to issue a bare INSERT, UPDATE or
DELETE of its own.

Undo works a turn at a time, not a row at a time: one spoken command can touch
several rows, and "undo that" means all of them. Reversing does not write new
audit rows -- it stamps ``undone_at`` on the rows it reversed -- so undoing
twice walks further back rather than redoing.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from src.store.db import utc_now_iso

#: Tables a tool may mutate. Table names are interpolated into SQL, so this
#: whitelist is load-bearing, not decorative.
MUTABLE_TABLES = ("course", "course_exception", "assignment", "reminder")

_SOFT_DELETE_LABEL = {
    "course": "class",
    "course_exception": "class change",
    "assignment": "assignment",
    "reminder": "reminder",
}


class UndoError(Exception):
    """Nothing left to undo."""


def _check_table(table: str) -> None:
    if table not in MUTABLE_TABLES:
        raise ValueError(f"{table!r} is not a mutable table")


def row_as_dict(conn: sqlite3.Connection, table: str, row_id: int) -> dict[str, Any] | None:
    _check_table(table)
    row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
    return dict(row) if row else None


def _record(
    conn: sqlite3.Connection,
    table: str,
    row_id: int,
    op: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    turn_id: int | None,
) -> None:
    conn.execute(
        "INSERT INTO audit_log (ts, table_name, row_id, op, before_json, after_json, turn_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            utc_now_iso(),
            table,
            row_id,
            op,
            json.dumps(before) if before is not None else None,
            json.dumps(after) if after is not None else None,
            turn_id,
        ),
    )


def insert(
    conn: sqlite3.Connection,
    table: str,
    values: dict[str, Any],
    turn_id: int | None = None,
) -> int:
    _check_table(table)
    now = utc_now_iso()
    payload = {**values, "created_at": now, "updated_at": now, "deleted_at": None}
    columns = ", ".join(payload)
    placeholders = ", ".join("?" for _ in payload)
    cursor = conn.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(payload.values())
    )
    row_id = int(cursor.lastrowid)
    _record(conn, table, row_id, "insert", None, row_as_dict(conn, table, row_id), turn_id)
    return row_id


def update(
    conn: sqlite3.Connection,
    table: str,
    row_id: int,
    changes: dict[str, Any],
    turn_id: int | None = None,
) -> dict[str, Any]:
    _check_table(table)
    before = row_as_dict(conn, table, row_id)
    if before is None:
        raise ValueError(f"no {table} row with id {row_id}")
    payload = {**changes, "updated_at": utc_now_iso()}
    assignments = ", ".join(f"{column} = ?" for column in payload)
    conn.execute(
        f"UPDATE {table} SET {assignments} WHERE id = ?", (*payload.values(), row_id)
    )
    after = row_as_dict(conn, table, row_id)
    _record(conn, table, row_id, "update", before, after, turn_id)
    return after or {}


def soft_delete(
    conn: sqlite3.Connection,
    table: str,
    row_id: int,
    turn_id: int | None = None,
) -> dict[str, Any]:
    _check_table(table)
    before = row_as_dict(conn, table, row_id)
    if before is None:
        raise ValueError(f"no {table} row with id {row_id}")
    now = utc_now_iso()
    conn.execute(
        f"UPDATE {table} SET deleted_at = ?, updated_at = ? WHERE id = ?", (now, now, row_id)
    )
    after = row_as_dict(conn, table, row_id)
    _record(conn, table, row_id, "delete", before, after, turn_id)
    return before


def _reverse(conn: sqlite3.Connection, entry: sqlite3.Row) -> str:
    table = entry["table_name"]
    row_id = entry["row_id"]
    op = entry["op"]
    before = json.loads(entry["before_json"]) if entry["before_json"] else None
    label = _SOFT_DELETE_LABEL.get(table, table)

    if op == "insert":
        # Undoing a creation hides the row; deletes stay soft even here, so the
        # audit trail keeps pointing at something real.
        conn.execute(
            f"UPDATE {table} SET deleted_at = ?, updated_at = ? WHERE id = ?",
            (utc_now_iso(), utc_now_iso(), row_id),
        )
        return f"removed the {label} I just added"

    if before is None:
        raise UndoError("that change has no recorded previous state")

    restorable = {k: v for k, v in before.items() if k != "id"}
    assignments = ", ".join(f"{column} = ?" for column in restorable)
    conn.execute(
        f"UPDATE {table} SET {assignments} WHERE id = ?", (*restorable.values(), row_id)
    )
    return f"restored the {label}" if op == "delete" else f"put the {label} back as it was"


def undo_last(conn: sqlite3.Connection, turn_id: int | None = None) -> dict[str, Any]:
    """Reverse every write from the most recent turn that still has one."""
    latest = conn.execute(
        "SELECT turn_id, MAX(id) AS last_id FROM audit_log"
        " WHERE undone_at IS NULL"
        " GROUP BY turn_id ORDER BY last_id DESC LIMIT 1"
    ).fetchone()
    if latest is None:
        raise UndoError("there is nothing to undo")

    if latest["turn_id"] is None:
        entries = conn.execute(
            "SELECT * FROM audit_log WHERE undone_at IS NULL AND turn_id IS NULL"
            " ORDER BY id DESC LIMIT 1"
        ).fetchall()
    else:
        entries = conn.execute(
            "SELECT * FROM audit_log WHERE undone_at IS NULL AND turn_id = ?"
            " ORDER BY id DESC",
            (latest["turn_id"],),
        ).fetchall()

    descriptions = []
    now = utc_now_iso()
    for entry in entries:
        descriptions.append(_reverse(conn, entry))
        conn.execute("UPDATE audit_log SET undone_at = ? WHERE id = ?", (now, entry["id"]))

    return {
        "reversed": len(entries),
        "description": descriptions[0] if descriptions else "nothing",
        "turn_id": latest["turn_id"],
    }

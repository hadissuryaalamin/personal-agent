"""Invariant #4: writes go through the audit log and deletes are soft."""

from __future__ import annotations

import json

import pytest

from src.store import audit
from src.turnlog import finish_turn, start_turn


def _add_course(conn, turn_id=None, code="COMP4020"):
    return audit.insert(
        conn,
        "course",
        {
            "code": code, "title": "Agentic Coding Studio", "instructor": None,
            "location": "HN 1.21", "weekday": 3, "start_time": "09:00",
            "end_time": "11:00", "term_start": None, "term_end": None, "notes": None,
        },
        turn_id,
    )


def test_schema_has_every_table(conn):
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "course", "course_exception", "assignment", "reminder", "turn_log", "audit_log"
    } <= names


def test_every_table_carries_the_timestamp_columns(conn):
    for table in ("course", "course_exception", "assignment", "reminder"):
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert {"created_at", "updated_at", "deleted_at"} <= columns, table


def test_insert_writes_an_audit_row(conn):
    row_id = _add_course(conn)
    entry = conn.execute("SELECT * FROM audit_log").fetchone()
    assert entry["op"] == "insert"
    assert entry["table_name"] == "course"
    assert entry["row_id"] == row_id
    assert entry["before_json"] is None
    assert json.loads(entry["after_json"])["code"] == "COMP4020"


def test_update_records_before_and_after(conn):
    row_id = _add_course(conn)
    audit.update(conn, "course", row_id, {"start_time": "10:00"})
    entry = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    assert json.loads(entry["before_json"])["start_time"] == "09:00"
    assert json.loads(entry["after_json"])["start_time"] == "10:00"


def test_delete_is_soft(conn):
    row_id = _add_course(conn)
    audit.soft_delete(conn, "course", row_id)
    row = conn.execute("SELECT * FROM course WHERE id = ?", (row_id,)).fetchone()
    assert row is not None
    assert row["deleted_at"] is not None


def test_undo_reverses_an_insert(conn):
    row_id = _add_course(conn)
    audit.undo_last(conn)
    row = conn.execute("SELECT * FROM course WHERE id = ?", (row_id,)).fetchone()
    assert row["deleted_at"] is not None


def test_undo_reverses_an_update(conn):
    row_id = _add_course(conn)
    audit.update(conn, "course", row_id, {"start_time": "14:00"})
    audit.undo_last(conn)
    row = conn.execute("SELECT * FROM course WHERE id = ?", (row_id,)).fetchone()
    assert row["start_time"] == "09:00"


def test_undo_restores_a_soft_delete(conn):
    row_id = _add_course(conn)
    audit.soft_delete(conn, "course", row_id)
    audit.undo_last(conn)
    row = conn.execute("SELECT * FROM course WHERE id = ?", (row_id,)).fetchone()
    assert row["deleted_at"] is None


def test_undo_walks_backwards_rather_than_redoing(conn):
    row_id = _add_course(conn)
    audit.update(conn, "course", row_id, {"start_time": "12:00"})
    audit.update(conn, "course", row_id, {"start_time": "14:00"})

    audit.undo_last(conn)
    assert conn.execute("SELECT start_time FROM course WHERE id = ?", (row_id,)).fetchone()[0] == "12:00"
    audit.undo_last(conn)
    assert conn.execute("SELECT start_time FROM course WHERE id = ?", (row_id,)).fetchone()[0] == "09:00"
    audit.undo_last(conn)
    assert conn.execute("SELECT deleted_at FROM course WHERE id = ?", (row_id,)).fetchone()[0] is not None


def test_undo_takes_a_whole_turn_at_once(conn):
    turn = start_turn(conn, "s1", "add both classes")
    first = _add_course(conn, turn, "COMP4020")
    second = _add_course(conn, turn, "COMP3500")
    finish_turn(conn, turn, tool_name="add_class")

    result = audit.undo_last(conn, turn_id=None)
    assert result["reversed"] == 2
    live = conn.execute("SELECT COUNT(*) FROM course WHERE deleted_at IS NULL").fetchone()[0]
    assert live == 0
    assert first != second


def test_undo_with_nothing_to_undo(conn):
    with pytest.raises(audit.UndoError):
        audit.undo_last(conn)


def test_undone_entries_are_not_reused(conn):
    _add_course(conn)
    audit.undo_last(conn)
    with pytest.raises(audit.UndoError):
        audit.undo_last(conn)


def test_only_whitelisted_tables_are_mutable(conn):
    with pytest.raises(ValueError):
        audit.insert(conn, "turn_log; DROP TABLE course", {"a": 1})
    with pytest.raises(ValueError):
        audit.soft_delete(conn, "audit_log", 1)


def test_turn_log_round_trip(conn):
    turn = start_turn(conn, "s1", "what is on tomorrow")
    finish_turn(
        conn, turn,
        tool_name="list_schedule",
        tool_args={"when": "tomorrow"},
        tool_result={"count": 0, "items": []},
        reply_text="Nothing on tomorrow.",
        ms_gen=12,
    )
    row = conn.execute("SELECT * FROM turn_log WHERE id = ?", (turn,)).fetchone()
    assert row["transcript"] == "what is on tomorrow"
    assert json.loads(row["tool_args_json"]) == {"when": "tomorrow"}
    assert row["reply_text"] == "Nothing on tomorrow."
    assert row["ms_gen"] == 12


def test_turn_log_rejects_unknown_columns(conn):
    turn = start_turn(conn, "s1", "hello")
    with pytest.raises(ValueError):
        finish_turn(conn, turn, ms_nonsense=1)

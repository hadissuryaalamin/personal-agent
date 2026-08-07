"""The text REPL, and invariant #3: every turn writes a turn_log row."""

from __future__ import annotations

import pytest

from src.cli import Repl, looks_like_tool_call, parse_command
from tests.conftest import NOW


@pytest.fixture
def repl(conn, cfg):
    """Typed-call mode: no weights, so these stay pure M0 behaviour."""
    return Repl(conn, cfg, session_id="test", now_fn=lambda: NOW, use_model=False)


def turns(conn):
    return conn.execute("SELECT * FROM turn_log ORDER BY id").fetchall()


# -- parsing ---------------------------------------------------------------


def test_parse_key_values():
    name, args = parse_command('add_assignment title="data structures" due="next friday"')
    assert name == "add_assignment"
    assert args == {"title": "data structures", "due": "next friday"}


def test_parse_alias():
    assert parse_command("undo")[0] == "undo_last_write"
    assert parse_command("schedule when=tomorrow")[0] == "list_schedule"


def test_parse_json_form():
    name, args = parse_command('{"tool": "list_schedule", "args": {"when": "tomorrow"}}')
    assert (name, args) == ("list_schedule", {"when": "tomorrow"})


def test_parse_rejects_bare_words():
    with pytest.raises(ValueError):
        parse_command("add_assignment essay")


# -- logging ---------------------------------------------------------------


def test_a_successful_turn_is_logged(repl, conn):
    repl.handle("add_assignment title=essay due=tomorrow")
    row = turns(conn)[0]
    assert row["transcript"] == "add_assignment title=essay due=tomorrow"
    assert row["tool_name"] == "add_assignment"
    assert "essay" in row["tool_args_json"]
    assert row["reply_text"].startswith("Added, due tomorrow")


def test_an_empty_turn_is_still_logged(repl, conn):
    reply = repl.handle("")
    assert reply == "I did not catch that."
    assert len(turns(conn)) == 1


def test_a_clarification_turn_is_logged(repl, conn):
    repl.handle("add_assignment title=essay due=sometime")
    row = turns(conn)[0]
    assert "clarification" in row["tool_result_json"]
    assert row["reply_text"].endswith("?")


def test_an_unparseable_turn_is_logged(repl, conn):
    repl.handle("this is not a tool call at all")
    assert len(turns(conn)) == 1


def test_a_crashing_turn_is_still_logged(repl, conn, monkeypatch):
    monkeypatch.setattr(
        "src.cli.registry.call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    reply = repl.handle("list_schedule when=today")
    assert reply == "Something went wrong with that one."
    row = turns(conn)[0]
    assert "boom" in row["tool_result_json"]


def test_meta_commands_are_logged_too(repl, conn):
    repl.handle(".tools")
    assert len(turns(conn)) == 1


def test_every_turn_of_a_conversation_lands(repl, conn):
    for line in ["", ".help", "add_assignment title=essay due=tomorrow", "garbage", "undo"]:
        repl.handle(line)
    assert len(turns(conn)) == 5


# -- confirmation flow -----------------------------------------------------


def test_confirmation_requires_a_yes(repl, conn):
    repl.handle("add_assignment title=essay due=tomorrow")
    reply = repl.handle("delete_assignment assignment=essay")
    assert reply.endswith("(yes/no)")
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is None

    repl.handle("yes")
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is not None


def test_saying_no_leaves_it_alone(repl, conn):
    repl.handle("add_assignment title=essay due=tomorrow")
    repl.handle("delete_assignment assignment=essay")
    assert repl.handle("no") == "Left it as it was."
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is None


def test_a_new_instruction_lapses_the_confirmation(repl, conn):
    repl.handle("add_assignment title=essay due=tomorrow")
    repl.handle("delete_assignment assignment=essay")
    repl.handle("list_schedule when=today")
    assert repl.pending is None
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is None


# -- writes are undoable end to end ----------------------------------------


def test_undo_by_typing(repl, conn):
    repl.handle("add_assignment title=essay due=tomorrow")
    reply = repl.handle("undo")
    assert "Done" in reply
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is not None


def test_audit_rows_are_grouped_by_turn(repl, conn):
    repl.handle("add_assignment title=essay due=tomorrow")
    entry = conn.execute("SELECT * FROM audit_log").fetchone()
    turn = conn.execute("SELECT id FROM turn_log ORDER BY id LIMIT 1").fetchone()[0]
    assert entry["turn_id"] == turn


# -- spoken replies --------------------------------------------------------


def test_write_is_confirmed_by_restating_the_resolved_date(repl):
    reply = repl.handle('add_assignment title="data structures" due="next friday"')
    assert reply == "Added, due Friday the fourteenth."


def test_reply_stays_short(repl):
    repl.handle("add_class code=COMP4020 weekday=thursday start=9am end=11am")
    repl.handle("add_class code=COMP3500 weekday=thursday start=2pm end=4pm")
    repl.handle("add_class code=ENGN4122 weekday=thursday start=5pm end=6pm")
    reply = repl.handle("schedule when=today")
    assert len(reply) <= 320
    assert reply.startswith("Three classes today")


def test_a_byte_order_mark_does_not_become_part_of_the_tool_name(repl):
    assert repl.handle("﻿schedule when=tomorrow") == "Nothing on tomorrow."


def test_empty_schedule_says_so(repl):
    assert repl.handle("schedule when=tomorrow") == "Nothing on tomorrow."


# -- typed call or sentence? -----------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "undo",
        "get_now",
        'add_assignment title="essay" due="next friday"',
        '{"tool": "get_now", "args": {}}',
        "schedule when=tomorrow",
    ],
)
def test_recognised_as_a_typed_call(text):
    assert looks_like_tool_call(text)


@pytest.mark.parametrize(
    "text",
    [
        "undo that last thing",
        "what is on tomorrow",
        "add the data structures assignment due next friday",
        "how are you",
        "get_now please",
        "",
    ],
)
def test_recognised_as_a_sentence(text):
    assert not looks_like_tool_call(text)


# -- the model path --------------------------------------------------------


class FakePlan:
    def __init__(self, label, tool=None, args=None, reply=None, score=None):
        self.label = label
        self.tool = tool
        self.args = args or {}
        self.reply = reply
        self.score = score
        self.gate_source = "fake"
        self.raw = None
        self.hidden = None
        self.ms_prefill = 30
        self.ms_gate = 4
        self.ms_gen = 90
        self.prompt_tokens = 42


class FakeAgent:
    def __init__(self, plan):
        self._plan = plan
        self.remembered = []

    def plan(self, transcript, now):
        return self._plan

    def remember(self, user, assistant):
        self.remembered.append((user, assistant))


def spoken_repl(conn, cfg, plan):
    return Repl(
        conn, cfg, session_id="test", now_fn=lambda: NOW, agent=FakeAgent(plan)
    )


def test_a_spoken_command_performs_the_write(conn, cfg):
    plan = FakePlan("tool", "add_assignment", {"title": "essay", "due": "next friday"}, score=0.94)
    repl = spoken_repl(conn, cfg, plan)

    assert repl.handle("add an essay due next friday") == "Added, due Friday the fourteenth."
    assert conn.execute("SELECT title FROM assignment").fetchone()[0] == "essay"


def test_a_spoken_turn_logs_the_gate_score(conn, cfg):
    plan = FakePlan("tool", "get_now", {}, score=0.88)
    repl = spoken_repl(conn, cfg, plan)
    repl.handle("what time is it")

    row = turns(conn)[0]
    assert row["probe_score"] == pytest.approx(0.88)
    assert row["probe_label"] == "tool"
    assert row["tool_name"] == "get_now"
    assert row["ms_prefill"] == 34
    assert row["ms_gen"] == 90


def test_a_chat_turn_writes_nothing_but_still_logs(conn, cfg):
    plan = FakePlan("chat", reply="Not much.", score=0.02)
    repl = spoken_repl(conn, cfg, plan)

    assert repl.handle("how are you") == "Not much."
    row = turns(conn)[0]
    assert row["tool_name"] is None
    assert row["probe_label"] == "chat"
    assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0


def test_an_unsure_turn_asks_and_writes_nothing(conn, cfg):
    plan = FakePlan("unsure", reply="Did you want me to check your schedule?", score=0.5)
    repl = spoken_repl(conn, cfg, plan)

    assert repl.handle("hmm friday").endswith("?")
    assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0


def test_a_spoken_destructive_command_still_confirms(conn, cfg):
    repl = spoken_repl(conn, cfg, FakePlan("tool", "add_assignment", {"title": "essay", "due": "tomorrow"}))
    repl.handle("add an essay due tomorrow")

    repl._agent = FakeAgent(FakePlan("tool", "delete_assignment", {"assignment": "essay"}))
    assert repl.handle("delete the essay").endswith("(yes/no)")
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is None

    repl.handle("yes")
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is not None


def test_the_reply_is_remembered_for_follow_ups(conn, cfg):
    plan = FakePlan("tool", "add_assignment", {"title": "essay", "due": "next friday"})
    repl = spoken_repl(conn, cfg, plan)
    repl.handle("add an essay")

    assert repl._agent.remembered == [("add an essay", "Added, due Friday the fourteenth.")]


def test_without_the_model_a_sentence_says_so(repl, conn):
    reply = repl.handle("what is on tomorrow")
    assert "typed tool calls" in reply
    assert len(turns(conn)) == 1

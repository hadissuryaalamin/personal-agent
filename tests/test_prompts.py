"""Prompt construction and the parsing of what comes back.

None of this needs weights: it is string handling, and it is where the M1
failure modes actually live (fenced JSON, a chatty gate answer, a model that
helpfully converts "next Friday" into a date it made up).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.llm import prompts
from src.tools import registry

TZ = ZoneInfo("Australia/Sydney")
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)


def test_system_prompt_injects_now():
    text = prompts.system_prompt(NOW, "Australia/Sydney")
    assert "Thursday 06 August 2026, 10:00" in text
    assert "Australia/Sydney" in text


def test_system_prompt_forbids_date_arithmetic():
    """Invariant #1, expressed as a prompt rule."""
    text = prompts.system_prompt(NOW, "Australia/Sydney")
    assert "Never work out a date" in text
    assert "copy that phrase through unchanged" in text


def test_messages_carry_the_last_three_turns_only():
    history = []
    for i in range(10):
        history += [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    messages = prompts.build_messages(NOW, "Australia/Sydney", "now this", history)

    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "now this"}
    # system + 3 turns * 2 + the current message
    assert len(messages) == 1 + prompts.HISTORY_TURNS * 2 + 1
    assert messages[1]["content"] == "u7"


def test_messages_work_with_no_history():
    messages = prompts.build_messages(NOW, "Australia/Sydney", "hello", None)
    assert len(messages) == 2


def test_compact_schemas_marks_required_arguments():
    rendered = prompts.compact_schemas(registry.schemas())
    assert "add_assignment(title*, due*, course, est_hours, notes)" in rendered
    assert "get_now()" in rendered
    # Every registered tool has to reach the model, or it can never be called.
    for name in registry.TOOLS:
        assert f"- {name}(" in rendered


def test_tool_instruction_tells_the_model_to_pass_time_through():
    text = prompts.tool_instruction(registry.schemas())
    assert "exactly as the student said them" in text


# -- reading the model's answer --------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"tool": "get_now", "args": {}}', {"tool": "get_now", "args": {}}),
        ('```json\n{"tool": "get_now", "args": {}}\n```', {"tool": "get_now", "args": {}}),
        ('```\n{"tool": "get_now", "args": {}}\n```', {"tool": "get_now", "args": {}}),
        ('Sure! {"tool": "get_now", "args": {}}', {"tool": "get_now", "args": {}}),
        ('{"tool": "get_now", "args": {}} and that is it', {"tool": "get_now", "args": {}}),
    ],
)
def test_extract_json_survives_the_usual_mess(raw, expected):
    assert prompts.extract_json(raw) == expected


def test_extract_json_handles_nested_objects_and_braces_in_strings():
    raw = '{"tool": "add_assignment", "args": {"title": "the {weird} one", "due": "friday"}}'
    parsed = prompts.extract_json(raw)
    assert parsed["args"]["title"] == "the {weird} one"
    assert parsed["args"]["due"] == "friday"


@pytest.mark.parametrize("raw", ["", "no json here", "{not valid}", "{", None])
def test_extract_json_gives_up_cleanly(raw):
    assert prompts.extract_json(raw) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("TOOL", "tool"),
        ("CHAT", "chat"),
        ("  TOOL  ", "tool"),
        ("TOOL.", "tool"),
        ('"CHAT"', "chat"),
        ("**TOOL**", "tool"),
        ("TOOL - they are asking about a class", "tool"),
        ("tool", "tool"),
    ],
)
def test_read_gate_token(raw, expected):
    assert prompts.read_gate_token(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "I think so", "MAYBE"])
def test_read_gate_token_refuses_to_guess(raw):
    assert prompts.read_gate_token(raw) is None

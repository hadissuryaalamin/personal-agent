"""The turn planner, against a fake engine.

The agent decides; it does not execute. That split is what lets the whole M1
control flow -- gate, second pass, malformed JSON, the uncertainty band -- be
tested with no weights on disk, which is invariant #7.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src import config
from src.llm import gate as gate_module
from src.llm.agent import UNSURE_REPLY, Agent
from src.llm.engine import Prefill

TZ = ZoneInfo("Australia/Sydney")
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)


class FakeEngine:
    """Scripted stand-in. Records what it was asked, returns what it was told."""

    def __init__(self, tool_text: str = "", chat_text: str = "") -> None:
        self.tool_text = tool_text
        self.chat_text = chat_text
        self.prefills = 0
        self.suffixes: list[str] = []
        self.info = {"layers": 36, "hidden_size": 2560}

    def prefix_text(self, messages):
        self.messages = messages
        return "PREFIX"

    def prefill(self, text):
        self.prefills += 1
        return Prefill(n_tokens=11, hidden=np.zeros((37, 2560), dtype=np.float32), ms=5)

    def continue_from(self, prefill, suffix, max_new_tokens=96, stop_on=None, stop_when=None):
        self.suffixes.append(suffix)
        self.stop_when = stop_when
        if "These tools are available" in suffix:
            return self.tool_text, 7
        return self.chat_text, 7


class FakeGate:
    source = "fake"

    def __init__(self, label: str, score: float | None = None) -> None:
        self.label = label
        self.score = score

    def decide(self, engine, prefill):
        return gate_module.Decision(self.label, self.score, self.source, ms=1)


@pytest.fixture
def cfg(tmp_path):
    return config.Config(
        tz=TZ, db_path=tmp_path / "t.db", gate="prompted",
        model_dir=tmp_path / "model", quantise="auto", probe_layer=None,
    )


def build(cfg, label, tool_text="", chat_text="", score=None):
    engine = FakeEngine(tool_text=tool_text, chat_text=chat_text)
    return Agent(engine, cfg, gate=FakeGate(label, score)), engine


# -- the tool branch -------------------------------------------------------


def test_tool_turn_extracts_name_and_arguments(cfg):
    agent, engine = build(
        cfg, "tool",
        tool_text='{"tool": "add_assignment", "args": {"title": "essay", "due": "next friday"}}',
    )
    plan = agent.plan("add an essay due next friday", NOW)

    assert plan.label == "tool"
    assert plan.tool == "add_assignment"
    assert plan.args == {"title": "essay", "due": "next friday"}


def test_the_prompt_is_encoded_once_per_turn(cfg):
    """PLAN.md section 4: one prefill, then suffixes onto the same cache."""
    agent, engine = build(cfg, "tool", tool_text='{"tool": "get_now", "args": {}}')
    agent.plan("what time is it", NOW)
    assert engine.prefills == 1


def test_the_tool_pass_stops_as_soon_as_the_json_closes(cfg):
    """Decode is the whole cost of a turn; do not pay for trailing chatter."""
    agent, engine = build(cfg, "tool", tool_text='{"tool": "get_now", "args": {}}')
    agent.plan("what time is it", NOW)

    assert engine.stop_when is not None
    assert not engine.stop_when('{"tool": "get_now"')
    assert engine.stop_when('{"tool": "get_now", "args": {}}')


def test_the_second_pass_carries_the_tool_schemas(cfg):
    agent, engine = build(cfg, "tool", tool_text='{"tool": "get_now", "args": {}}')
    agent.plan("what is on today", NOW)
    assert any("These tools are available" in s for s in engine.suffixes)
    assert any("add_assignment" in s for s in engine.suffixes)


def test_empty_arguments_are_dropped_rather_than_passed_as_blanks(cfg):
    agent, _ = build(
        cfg, "tool",
        tool_text='{"tool": "add_assignment", "args": {"title": "essay", '
                  '"due": "friday", "course": null, "est_hours": ""}}',
    )
    plan = agent.plan("add an essay", NOW)
    assert plan.args == {"title": "essay", "due": "friday"}


@pytest.mark.parametrize(
    "raw",
    [
        "I'll add that for you!",
        "",
        "{}",
        '{"args": {"title": "essay"}}',
        '{"tool": "add_assignment", "args": "essay next friday"}',
    ],
)
def test_unusable_json_asks_rather_than_calling_something_arbitrary(cfg, raw):
    agent, _ = build(cfg, "tool", tool_text=raw)
    plan = agent.plan("add an essay", NOW)
    assert plan.label == "unsure"
    assert plan.tool is None
    assert plan.reply.endswith("?")


@pytest.mark.parametrize(
    "named", ["get_now", "get_now()", "`get_now`", "get_now(args)", " get_now "]
)
def test_the_tool_name_survives_the_models_punctuation(cfg, named):
    """The schema listing shows get_now(...), and the model copies the parens."""
    agent, _ = build(cfg, "tool", tool_text='{"tool": "%s", "args": {}}' % named)
    plan = agent.plan("what time is it", NOW)
    assert plan.tool == "get_now"


def test_fenced_json_still_works(cfg):
    agent, _ = build(
        cfg, "tool",
        tool_text='```json\n{"tool": "list_schedule", "args": {"when": "tomorrow"}}\n```',
    )
    plan = agent.plan("what is on tomorrow", NOW)
    assert plan.tool == "list_schedule"


# -- the chat branch -------------------------------------------------------


def test_chat_turn_never_reaches_the_tool_pass(cfg):
    agent, engine = build(cfg, "chat", chat_text="Not much, you?")
    plan = agent.plan("how are you", NOW)

    assert plan.label == "chat"
    assert plan.tool is None
    assert plan.reply == "Not much, you?"
    assert not any("These tools are available" in s for s in engine.suffixes)


def test_an_empty_chat_reply_still_says_something(cfg):
    agent, _ = build(cfg, "chat", chat_text="")
    assert agent.plan("hello", NOW).reply


# -- the uncertainty band --------------------------------------------------


def test_the_band_asks_one_question(cfg):
    """Invariant #6: a score between the thresholds does not become a guess."""
    agent, engine = build(cfg, "unsure", score=0.5)
    plan = agent.plan("maybe something about friday", NOW)

    assert plan.reply == UNSURE_REPLY
    assert plan.tool is None
    assert engine.suffixes == [], "an unsure turn should not run a second pass"


# -- history ---------------------------------------------------------------


def test_history_is_remembered_and_capped(cfg):
    agent, engine = build(cfg, "chat", chat_text="ok")
    for i in range(8):
        agent.plan(f"turn {i}", NOW)
        agent.remember(f"turn {i}", "ok")

    from src.llm import prompts

    assert len(agent.history) == prompts.HISTORY_TURNS * 2
    assert agent.history[0]["content"] == "turn 5"


def test_history_reaches_the_prompt(cfg):
    agent, engine = build(cfg, "chat", chat_text="ok")
    agent.remember("add the essay", "Added, due Friday the fourteenth.")
    agent.plan("make it 60 instead", NOW)

    contents = [m["content"] for m in engine.messages]
    assert "add the essay" in contents
    assert "make it 60 instead" == contents[-1]


# -- timings and the probe columns -----------------------------------------


def test_plan_reports_what_turn_log_needs(cfg):
    agent, _ = build(cfg, "tool", tool_text='{"tool": "get_now", "args": {}}', score=0.9)
    plan = agent.plan("what time is it", NOW)

    assert plan.score == 0.9
    assert plan.prompt_tokens == 11
    assert plan.ms_prefill == 5
    assert plan.hidden.shape == (37, 2560)


# -- the gate's own arithmetic ---------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [(0.0, "chat"), (0.34, "chat"), (0.35, "unsure"), (0.5, "unsure"),
     (0.64, "unsure"), (0.65, "tool"), (1.0, "tool")],
)
def test_classify_is_a_three_way_split(score, expected):
    assert gate_module.classify(score) == expected


def test_thresholds_are_asymmetric_about_the_middle():
    """A false call writes wrong data; a false no-call is merely unhelpful."""
    assert gate_module.TAU_LO < 0.5 < gate_module.TAU_HI


def test_the_gate_is_swappable_without_touching_the_agent(cfg):
    """PLAN.md M3: the probe goes behind config.gate, and nothing else changes."""
    from dataclasses import replace

    assert isinstance(gate_module.build(cfg), gate_module.PromptedGate)
    assert isinstance(
        gate_module.build(replace(cfg, gate="probe")), gate_module.ProbeGate
    )


def test_the_agent_does_not_care_which_gate_it_has(cfg):
    """Both gates return a Decision; the branch in Agent.plan reads only that."""
    agent, engine = build(cfg, "tool", tool_text='{"tool": "get_now", "args": {}}')
    agent.gate = FakeGate("tool", score=0.99)
    assert agent.plan("what time is it", NOW).tool == "get_now"

    agent.gate = FakeGate("chat", score=0.01)
    agent.engine.chat_text = "Hello."
    assert agent.plan("hello", NOW).tool is None


def test_unknown_gate_is_rejected(cfg):
    from dataclasses import replace

    with pytest.raises(ValueError):
        gate_module.build(replace(cfg, gate="vibes"))

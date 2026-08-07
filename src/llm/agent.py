"""One turn: prefill once, gate, then either extract a tool call or chat.

The agent *plans*; it does not execute. It returns which tool to call with what
arguments, and the caller runs it through ``src.tools.registry``. Keeping
execution out of here means the confirmation and clarification flow works
identically whether a turn came from the model or was typed by hand, and it
means the interesting part -- gate plus argument extraction -- can be tested
against a fake engine with no weights on disk.

Verbalising a tool *result* is deliberately not the model's job: ``src.format``
already turns results into speech-shaped replies, deterministically, with the
length and restatement rules from CLAUDE.md baked in and covered by tests. The
model is asked only for things a template cannot do -- deciding, extracting
arguments, and chatting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.llm import gate as gate_module
from src.llm import prompts
from src.tools import registry


@dataclass
class Plan:
    """What the model decided, before anything has been executed."""

    label: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    #: Set when the turn is chat, or when the gate landed in the uncertainty
    #: band and one question has to go back to the user.
    reply: str | None = None
    score: float | None = None
    gate_source: str = "prompted"
    raw: str | None = None
    hidden: Any = field(default=None, repr=False)
    ms_prefill: int = 0
    ms_gate: int = 0
    ms_gen: int = 0
    prompt_tokens: int = 0


#: What to say when the gate cannot tell. Invariant #6: one clarifying
#: question, not a best guess.
UNSURE_REPLY = "Did you want me to check your schedule, or were you just saying?"

_BAD_JSON_REPLY = "I did not follow that — could you say it again?"


class Agent:
    def __init__(self, engine, cfg, gate=None) -> None:
        self.engine = engine
        self.cfg = cfg
        self.gate = gate or gate_module.build(cfg)
        self.history: list[dict[str, str]] = []

    def remember(self, user: str, assistant: str) -> None:
        """Keep the last few turns so follow-ups resolve (PLAN.md section 4)."""
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        keep = prompts.HISTORY_TURNS * 2
        if len(self.history) > keep:
            del self.history[:-keep]

    def plan(self, transcript: str, now: datetime) -> Plan:
        messages = prompts.build_messages(
            now, self.cfg.tz_name, transcript, self.history
        )
        prefix = self.engine.prefix_text(messages)
        prefill = self.engine.prefill(prefix)

        decision = self.gate.decide(self.engine, prefill)

        base = Plan(
            label=decision.label,
            score=decision.score,
            gate_source=decision.source,
            hidden=prefill.hidden,
            ms_prefill=prefill.ms,
            ms_gate=decision.ms,
            prompt_tokens=prefill.n_tokens,
        )

        if decision.label == "unsure":
            base.reply = UNSURE_REPLY
            return base

        if decision.label == "chat":
            text, ms = self.engine.continue_from(
                prefill, "<|im_start|>assistant\n", max_new_tokens=96
            )
            base.reply = text or "Sorry, I have nothing for that."
            base.ms_gen = ms
            return base

        return self._extract_call(prefill, base)

    def _extract_call(self, prefill, base: Plan) -> Plan:
        """The second pass: same cache, tool schemas appended."""
        suffix = (
            f"<|im_start|>user\n{prompts.tool_instruction(registry.schemas())}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        # Stop the instant the JSON object closes. This is a cap on the worst
        # case, not a speed-up: the model usually stops itself near the closing
        # brace, so the median turn does not move. It is worth keeping because
        # decode costs ~100 ms/token here (scripts/bench_decode.py), so the
        # difference between stopping at 67 tokens and running to the 160-token
        # limit is about ten seconds.
        started = time.perf_counter()
        text, ms = self.engine.continue_from(
            prefill,
            suffix,
            max_new_tokens=160,
            stop_when=lambda so_far: prompts.extract_json(so_far) is not None,
        )
        base.raw = text
        base.ms_gen = ms or int((time.perf_counter() - started) * 1000)

        payload = prompts.extract_json(text)
        if not payload or not payload.get("tool"):
            # The gate said tool but the model produced nothing usable. Asking
            # beats calling something arbitrary.
            base.label = "unsure"
            base.reply = _BAD_JSON_REPLY
            return base

        args = payload.get("args") or {}
        if not isinstance(args, dict):
            base.label = "unsure"
            base.reply = _BAD_JSON_REPLY
            return base

        # The schema listing shows "get_now(...)", and the model copies the
        # parentheses back about a third of the time. Strip them rather than
        # failing a call that named the right tool.
        base.tool = str(payload["tool"]).split("(")[0].strip().strip("`\"' ")
        base.args = {k: v for k, v in args.items() if v not in (None, "")}
        return base

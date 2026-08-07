"""Text REPL -- the development surface (CLAUDE.md, "Working style").

Same tools, same store, same clarification and confirmation behaviour as the
voice loop, with the audio stack left out. Get behaviour right here before
touching audio.

    python -m src.cli                 natural language, model loaded on demand
    python -m src.cli --no-model      typed tool calls only, no weights needed

Both forms of input end up in the same place. Type a tool call directly:

    add_assignment title="data structures" due="next friday" est_hours=6
    set_progress assignment="data structures" percent=60
    undo

or just say it:

    add the data structures assignment, due next friday, about six hours

The typed form is what makes tools testable without the model; the spoken form
is what M1 adds. Execution, confirmation and logging are identical for both --
only the way the tool name and arguments are arrived at differs.
"""

from __future__ import annotations

import json
import shlex
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src import config, format, hidden
from src.store.db import connect, init_db
from src.tools import registry
from src.tools.context import ToolContext
from src.turnlog import finish_turn, start_turn

#: Short names for the things you type most while working on tools.
ALIASES = {
    "now": "get_now",
    "schedule": "list_schedule",
    "classes": "list_schedule",
    "assignments": "list_assignments",
    "todo": "list_assignments",
    "undo": "undo_last_write",
}

_YES = {"y", "yes", "yeah", "yep", "go ahead", "do it", "confirm", "sure"}
_NO = {"n", "no", "nope", "cancel", "stop", "leave it", "never mind", "nevermind"}

_NO_MODEL_REPLY = (
    "Running without the model, so I only take typed tool calls. "
    "Try '.help', or restart without --no-model."
)

HELP = """\
Say what you want, or type a tool call directly:

  <tool> key=value key="value with spaces"
  {"tool": "list_schedule", "args": {"when": "tomorrow"}}

  .tools    list tool names
  .schema   full tool schemas as JSON
  .last     the raw result of the last turn
  .gate     how the gate scored the last turn
  .load     load the model now rather than on first sentence
  .help     this
  .quit     leave

Aliases: now, schedule, assignments, undo
"""


@dataclass
class TurnRecord:
    """Everything one turn needs to write to turn_log."""

    reply: str
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    #: probe_score, probe_label, ms_prefill, ms_gen -- real turn_log columns.
    extra: dict[str, Any] = field(default_factory=dict)


class Repl:
    """One session. Holds the pending confirmation and the model between turns."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        cfg: config.Config | None = None,
        session_id: str | None = None,
        now_fn: Callable[[], datetime] | None = None,
        use_model: bool = True,
        agent: Any = None,
    ) -> None:
        self.conn = conn
        self.cfg = cfg or config.load()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.use_model = use_model
        self.pending: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_plan: Any = None
        #: So the voice loop can add ms_tts once it has finished speaking.
        self.last_turn_id: int | None = None
        self._agent = agent

    # -- the model, loaded only when a sentence actually needs it ---------

    @property
    def agent(self):
        if self._agent is None:
            from src.llm.agent import Agent
            from src.llm.engine import Engine

            print(f"loading {self.cfg.model_dir.name}…", flush=True)
            engine = Engine().load()
            print(
                f"  {engine.info['layers']} layers, hidden {engine.info['hidden_size']}, "
                f"{engine.info['quantisation']} on {engine.info['device']} "
                f"({engine.info['load_ms'] / 1000:.1f}s)",
                flush=True,
            )
            self._agent = Agent(engine, self.cfg)
        return self._agent

    # -- turn handling ----------------------------------------------------

    def handle(
        self,
        line: str,
        *,
        spoken: bool = False,
        ms_asr: int | None = None,
        asr_conf: float | None = None,
    ) -> str:
        """Run one turn. Always writes a turn_log row -- invariant #3.

        ``spoken`` marks a transcript from the microphone. It never takes the
        typed-tool-call shortcut: "undo" said out loud has to go through the
        gate like anything else, or the voice loop would be running a different
        control flow from the one the probe was measured on.
        """
        # A byte-order mark or zero-width space arrives whenever input is piped
        # in on Windows, and would otherwise become part of the tool name.
        line = (line or "").lstrip("﻿​").rstrip("﻿​")
        turn_id = start_turn(self.conn, self.session_id, line, asr_conf)
        self.last_turn_id = turn_id
        record = TurnRecord(reply="Something went wrong with that one.")

        try:
            record = self._dispatch(line, turn_id, spoken=spoken)
        except Exception as exc:  # noqa: BLE001 - the log row matters more than the traceback
            record.result = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            if ms_asr is not None:
                record.extra["ms_asr"] = ms_asr
            finish_turn(
                self.conn,
                turn_id,
                tool_name=record.tool_name,
                tool_args=record.args,
                tool_result=record.result,
                reply_text=record.reply,
                **record.extra,
            )

        self.last_result = record.result
        return record.reply

    def _dispatch(self, line: str, turn_id: int, spoken: bool = False) -> TurnRecord:
        text = (line or "").strip()

        if not text:
            return TurnRecord(reply="I did not catch that.", result={"empty": True})

        if not spoken and text.startswith("."):
            return TurnRecord(reply=self._meta(text), result={"meta": text})

        if self.pending is not None:
            answer = text.lower().strip("!.? ")
            if answer in _YES:
                resume = self.pending
                self.pending = None
                return self._run(resume["tool"], resume.get("args", {}), turn_id, True)
            if answer in _NO:
                self.pending = None
                return TurnRecord(reply="Left it as it was.", result={"cancelled": True})
            # Anything else is a new instruction; the confirmation lapses.
            self.pending = None

        if not spoken and looks_like_tool_call(text):
            try:
                tool_name, args = parse_command(text)
            except ValueError as exc:
                return TurnRecord(reply=str(exc), result={"error": str(exc)})
            return self._run(tool_name, args, turn_id)

        return self._converse(text, turn_id)

    def _converse(self, text: str, turn_id: int) -> TurnRecord:
        """The M1 path: gate the sentence, then extract a call or just talk."""
        if not self.use_model:
            return TurnRecord(reply=_NO_MODEL_REPLY, result={"no_model": True})

        now = self.now_fn()
        plan = self.agent.plan(text, now)
        self.last_plan = plan

        # M2: the hidden state this turn produced is the probe's training data.
        saved = hidden.save_turn(plan.hidden, turn_id, self.session_id)

        extra = {
            "probe_score": plan.score,
            "probe_label": plan.label,
            "ms_prefill": plan.ms_prefill + plan.ms_gate,
            "ms_gen": plan.ms_gen,
            "hidden_state_path": hidden.relative_to_root(saved),
        }

        if plan.label != "tool" or not plan.tool:
            self.agent.remember(text, plan.reply or "")
            return TurnRecord(
                reply=plan.reply or "",
                result={"gate": plan.label, "score": plan.score, "raw": plan.raw},
                extra=extra,
            )

        record = self._run(plan.tool, plan.args, turn_id)
        record.extra.update(extra)
        self.agent.remember(text, record.reply)
        return record

    def _run(
        self, tool_name: str, args: dict[str, Any], turn_id: int, confirmed: bool = False
    ) -> TurnRecord:
        ctx = ToolContext(
            conn=self.conn,
            now=self.now_fn(),
            tz=self.cfg.tz,
            turn_id=turn_id,
            confirmed=confirmed,
        )
        result = registry.call(tool_name, args, ctx)

        if result.get("needs") == "confirmation":
            self.pending = result["resume"]

        today = ctx.now.astimezone(self.cfg.tz).date()
        reply = format.reply_for(result, today)
        if result.get("needs") == "confirmation":
            reply = f"{reply} (yes/no)"
        return TurnRecord(reply=reply, tool_name=tool_name, args=args, result=result)

    # -- meta commands ----------------------------------------------------

    def _meta(self, text: str) -> str:
        command = text.split()[0].lower()
        if command in (".help", ".h", ".?"):
            return HELP
        if command == ".tools":
            return "\n".join(sorted(registry.TOOLS))
        if command == ".schema":
            return json.dumps(registry.schemas(), indent=2)
        if command == ".last":
            return json.dumps(self.last_result, indent=2, default=str)
        if command == ".gate":
            plan = self.last_plan
            if plan is None:
                return "No sentence has been through the gate yet."
            score = f" (score {plan.score:.3f})" if plan.score is not None else ""
            lines = [
                f"{plan.gate_source} gate: {plan.label}{score}",
                f"  prompt {plan.prompt_tokens} tokens, prefill {plan.ms_prefill} ms, "
                f"gate {plan.ms_gate} ms, generate {plan.ms_gen} ms",
            ]
            if plan.raw:
                lines.append(f"  raw: {plan.raw!r}")
            return "\n".join(lines)
        if command == ".load":
            return f"model ready: {self.agent.engine.info}"
        return f"No such command: {command}. Try .help"


def looks_like_tool_call(text: str) -> bool:
    """A typed call, or a sentence?

    Deliberately strict: the first word has to be a real tool name and every
    remaining word has to be key=value. "undo" is a tool call; "undo that last
    thing" is a sentence.
    """
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return True
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return False
    if not parts:
        return False
    name = ALIASES.get(parts[0].lower(), parts[0])
    if name not in registry.TOOLS:
        return False
    return all("=" in part for part in parts[1:])


def parse_command(text: str) -> tuple[str, dict[str, Any]]:
    """``add_assignment title="essay" due="next friday"`` or a JSON object."""
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"That is not valid JSON: {exc.msg}") from None
        name = payload.get("tool") or payload.get("name")
        if not name:
            raise ValueError('JSON needs a "tool" key.')
        return name, dict(payload.get("args") or {})

    try:
        parts = shlex.split(text)
    except ValueError:
        raise ValueError("Unbalanced quotes in that command.") from None
    if not parts:
        raise ValueError("Nothing to run.")

    name = ALIASES.get(parts[0].lower(), parts[0])
    args: dict[str, Any] = {}
    for token in parts[1:]:
        if "=" not in token:
            raise ValueError(f"Arguments look like key=value; “{token}” does not.")
        key, value = token.split("=", 1)
        args[key.strip()] = value
    return name, args


def ensure_schema(conn: sqlite3.Connection) -> None:
    present = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='turn_log'"
    ).fetchone()
    if present is None:
        init_db(conn)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="personal-agent text mode")
    parser.add_argument(
        "--no-model", action="store_true", help="typed tool calls only; never load weights"
    )
    parser.add_argument("--preload", action="store_true", help="load the model at startup")
    args = parser.parse_args(argv)

    cfg = config.load()
    conn = connect(cfg.db_path)
    ensure_schema(conn)
    repl = Repl(conn, cfg, use_model=not args.no_model)

    now = datetime.now(timezone.utc).astimezone(cfg.tz)
    print(f"personal-agent — text mode. {now:%A %d %B %Y, %H:%M} {cfg.tz_name}")
    if args.no_model:
        print("no model: typed tool calls only.")
    print("'.help' for commands, '.quit' to leave.\n")

    if args.preload:
        repl.agent  # noqa: B018 - loading is the point

    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in (".quit", ".q", ".exit"):
            break
        print(repl.handle(line))
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

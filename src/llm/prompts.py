"""Prompt construction.

The shape here is dictated by PLAN.md section 4: one prefill per turn, with the
gate and the tool schemas appended as *suffixes* to the same KV cache. That
means the shared prefix has to end at the user's message -- anything that comes
after it (the TOOL/CHAT question, the tool schemas) is a separate continuation
from the same cached prefix.

    <prefix>  system + last three turns + this user message   <- prefilled once,
                                                                 h_L read here
    <suffix>  either "TOOL or CHAT?" or the tool schema block

Invariant #1 shows up here as a prompt rule: the model is told never to compute
a date. It copies "next Friday" through verbatim and src/timeparse.py resolves
it. Any prompt that asks the model to work out a date is a bug in this file.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

#: PLAN.md section 4: the gate sees the last three turns, not just the current
#: utterance, because follow-ups ("make it 60 instead") only make sense in
#: context.
HISTORY_TURNS = 3

SYSTEM = """\
You are a scheduling assistant for a university student. You run on their own \
machine and you speak out loud, so answers are short.

The current date and time is {now}. The timezone is {tz}.

You cannot see the student's schedule yourself. The only way to know anything \
about their classes, assignments or reminders is to call a tool, and you \
remember nothing about them between sessions.

Rules you must follow:
- Never work out a date or a time yourself. When the user says "next Friday", \
"tomorrow", "in two weeks" or "the 14th", copy that phrase through unchanged \
as the argument. Something else resolves it. Do not convert it to a calendar \
date, and never write an ISO date unless the user said the digits.
- Do not guess which class or assignment is meant. Pass on what the user \
called it.
- Keep replies to two sentences at most. No preamble, no "sure thing", no \
restating the question. Answer, then stop.
- Never say you have added, changed, moved or removed anything. You cannot: \
something else does that and reports it. If you are replying in words rather \
than calling a tool, then nothing has been recorded, and saying otherwise \
would be a lie the student acts on.
"""

#: Wording chosen by measurement, not taste -- see scripts/eval_gate.py and
#: docs/eval.md. Telling the model it cannot answer from memory is what moves
#: read queries ("what have I got due this week") off CHAT; a few-shot version
#: of this was measured and was *worse*, so it is not used.
GATE_QUESTION = """\
You cannot answer anything about their schedule from memory.

Reply with one word:
TOOL - answering that message requires reading or changing their saved \
classes, assignments or reminders.
CHAT - it is small talk, thanks, a greeting, or a question about you.\
"""

TOOL_INSTRUCTION = """\
These tools are available:

{schemas}

Reply with one JSON object and nothing else, in the form
{{"tool": "<name>", "args": {{...}}}}

Put time expressions in exactly as the student said them. Leave out any \
argument they did not mention.\
"""


def system_prompt(now: datetime, tz: str) -> str:
    """The system message, with ``now`` injected -- never inferred."""
    return SYSTEM.format(now=now.strftime("%A %d %B %Y, %H:%M"), tz=tz)


def system_fingerprint() -> str:
    """Identifies the prompt the probe's hidden states were computed from.

    The probe reads a hidden state produced by a prefix that *begins* with this
    system message. Edit the message and every activation downstream of it
    moves, which makes a probe trained before the edit quietly wrong -- it will
    still return confident scores, just for a prompt it never saw. The artefact
    records this fingerprint and `ProbeGate` refuses to run against a prompt it
    does not match.

    Hashes the template, not the formatted message: `now` changes every minute
    and is not what the probe is sensitive to.
    """
    import hashlib

    return hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:16]


def build_messages(
    now: datetime,
    tz: str,
    transcript: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    recent = (history or [])[-HISTORY_TURNS * 2 :]
    return [
        {"role": "system", "content": system_prompt(now, tz)},
        *recent,
        {"role": "user", "content": transcript},
    ]


def compact_schemas(schemas: list[dict[str, Any]]) -> str:
    """One line per tool. Full JSON Schema costs tokens and reads worse."""
    lines = []
    for schema in schemas:
        params = schema["parameters"]["properties"]
        required = set(schema["parameters"].get("required", []))
        rendered = ", ".join(
            f"{name}*" if name in required else name for name in params
        )
        lines.append(f"- {schema['name']}({rendered}) — {schema['description']}")
    return "\n".join(lines)


def tool_instruction(schemas: list[dict[str, Any]]) -> str:
    return TOOL_INSTRUCTION.format(schemas=compact_schemas(schemas))


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a reply, fences and all.

    Small models wrap JSON in ```json blocks, add a sentence in front, or emit
    a trailing comma. None of that is worth a retry if it can be salvaged.
    """
    if not text:
        return None

    cleaned = text.strip()
    if "```" in cleaned:
        chunks = cleaned.split("```")
        for chunk in chunks[1:]:
            chunk = chunk.removeprefix("json").strip()
            if chunk.startswith("{"):
                cleaned = chunk
                break

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def read_gate_token(text: str) -> str | None:
    """Map the gate's first word onto "tool" or "chat"."""
    if not text:
        return None
    word = text.strip().split()[0].strip('".,:*').upper() if text.strip() else ""
    if word.startswith("TOOL"):
        return "tool"
    if word.startswith("CHAT"):
        return "chat"
    return None

"""Fuzzy matching of spoken names to rows.

ASR mangles course codes -- "COMP4020" comes back as "comp four thousand and
twenty", "camp 40 20", "comp for zero to zero". A confident wrong match
silently corrupts data, so this module resolves to exactly one row or asks.

The rule from PLAN.md section 3: an exact match wins outright; otherwise two or
more candidates above the threshold produce a clarification rather than a pick.
"""

from __future__ import annotations

import re
import sqlite3
from difflib import SequenceMatcher
from typing import Any

from src.tools.errors import NeedsClarification

#: Below this, a candidate is not in the running at all.
THRESHOLD = 0.72

#: ASR homophones for digits, applied only when comparing course codes.
_SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "won": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3", "tree": "3",
    "four": "4", "for": "4", "fore": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9",
}

_MULTIPLIERS = {"hundred": 100, "thousand": 1000}


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _spoken_number_to_digits(tokens: list[str]) -> list[str]:
    """"four thousand twenty" -> "4020"; "four zero two zero" -> "4020"."""
    out: list[str] = []
    buffer: list[str] = []
    running = 0
    current = 0

    def flush() -> None:
        nonlocal running, current
        if buffer:
            out.append("".join(buffer))
            buffer.clear()
        if running or current:
            out.append(str(running + current))
            running = current = 0

    for token in tokens:
        if token in _MULTIPLIERS and (current or buffer):
            if buffer:
                current = int("".join(buffer))
                buffer.clear()
            current *= _MULTIPLIERS[token]
            running += current
            current = 0
            continue
        if token in _SPOKEN_DIGITS:
            buffer.append(_SPOKEN_DIGITS[token])
            continue
        if token.isdigit():
            buffer.append(token)
            continue
        if token == "and" and (buffer or running):
            continue
        flush()
        out.append(token)
    flush()
    return out


def normalise_code(text: str) -> str:
    """Fold a spoken course code down to something comparable."""
    lowered = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    tokens = [t for t in lowered.split() if t]
    return "".join(_spoken_number_to_digits(tokens))


def _score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    a, b = _squash(query), _squash(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    # A spoken query is often a fragment of the stored title.
    if a in b or b in a:
        ratio = max(ratio, 0.85)
    return ratio


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _code_score(query: str, candidate: str) -> float:
    """Score a course code, where the number is the identity.

    COMP4020 and COMP4620 are 87% similar as strings and completely different
    as courses. A digit mismatch is therefore a veto, not a penalty: better to
    ask than to hang an assignment off the wrong course. A query with no digits
    at all ("comp", "the agentic one") is treated as a fragment and still
    matches, which is what produces the two-candidate clarification.
    """
    a, b = normalise_code(query), normalise_code(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    query_digits, candidate_digits = _digits(a), _digits(b)
    if query_digits and candidate_digits and query_digits != candidate_digits:
        return 0.0

    ratio = SequenceMatcher(None, a, b).ratio()
    if not query_digits and a in b:
        ratio = max(ratio, 0.85)
    return ratio


def score_row(query: str, row: sqlite3.Row, fields: tuple[str, ...]) -> float:
    best = 0.0
    for field in fields:
        value = row[field] if field in row.keys() else None
        if not value:
            continue
        if field == "code":
            best = max(best, _code_score(query, str(value)))
        else:
            best = max(best, _score(query, str(value)))
    return best


def _describe_course(row: sqlite3.Row) -> str:
    return f"{row['code']}{' ' + row['title'] if row['title'] else ''}".strip()


def _describe_assignment(row: sqlite3.Row) -> str:
    return str(row["title"])


def resolve_one(
    conn: sqlite3.Connection,
    table: str,
    query: str | int,
    *,
    fields: tuple[str, ...],
    describe,
    noun: str,
    plural: str,
    required: bool = True,
) -> dict[str, Any] | None:
    """Resolve a spoken name or an id to exactly one live row.

    ``required=False`` is for optional links, like the course an assignment
    belongs to. A name that matches *nothing* then returns None so the caller
    can go ahead without the link and say so, rather than refusing the whole
    write over an argument the user never had to give. Two plausible matches
    still ask, whether the field was required or not -- that is invariant #6,
    and it is about ambiguity, not about absence.
    """
    if isinstance(query, int) or (isinstance(query, str) and str(query).strip().isdigit()):
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND deleted_at IS NULL", (int(query),)
        ).fetchone()
        if row is None:
            if not required:
                return None
            raise NeedsClarification(f"I have no {noun} with id {query}. Which one did you mean?")
        return dict(row)

    text = (query or "").strip()
    if not text:
        if not required:
            return None
        raise NeedsClarification(f"Which {noun}?")

    rows = conn.execute(f"SELECT * FROM {table} WHERE deleted_at IS NULL").fetchall()
    if not rows:
        if not required:
            return None
        raise NeedsClarification(f"I do not have any {plural} yet. What should I add?")

    scored = sorted(
        ((score_row(text, row, fields), row) for row in rows),
        key=lambda pair: pair[0],
        reverse=True,
    )

    exact = [row for value, row in scored if value >= 0.999]
    if len(exact) == 1:
        return dict(exact[0])
    if len(exact) > 1:
        raise NeedsClarification(
            f"I have {len(exact)} that match “{text}” — which one?",
            [describe(row) for row in exact[:3]],
        )

    above = [(value, row) for value, row in scored if value >= THRESHOLD]
    if not above:
        if not required:
            return None
        raise NeedsClarification(f"I could not find a {noun} matching “{text}”. What is it called?")
    if len(above) == 1:
        return dict(above[0][1])

    names = [describe(row) for _, row in above[:3]]
    raise NeedsClarification(
        f"Did you mean {' or '.join(names)}?" if len(names) == 2
        else f"Which one — {', '.join(names)}?",
        names,
    )


def resolve_course(
    conn: sqlite3.Connection, query: str | int, required: bool = True
) -> dict[str, Any] | None:
    return resolve_one(
        conn, "course", query, fields=("code", "title"), describe=_describe_course,
        noun="class", plural="classes", required=required,
    )


def resolve_assignment(
    conn: sqlite3.Connection, query: str | int, required: bool = True
) -> dict[str, Any] | None:
    return resolve_one(
        conn, "assignment", query, fields=("title",), describe=_describe_assignment,
        noun="assignment", plural="assignments", required=required,
    )

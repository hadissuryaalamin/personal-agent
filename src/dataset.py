"""The probe's training data.

PLAN.md section 4: ~600 seed utterances, balanced call / no-call, covering
every tool plus chatter, greetings, follow-ups and refusals; 80/20 stratified
split; real logged turns folded in as they accumulate, and always landing in
train, never in the held-out set.

The seed lives in ``data/probe/*.jsonl``, one JSON object per line:

    {"text": "...", "label": 1, "why": "add_assignment",
     "history": [["earlier user turn", "earlier reply"]]}

``history`` is optional and matters: the gate sees the last three turns, so
"make it sixty instead" is only a tool call in context. An utterance appearing
both with and without history is two different examples, not a duplicate.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ROOT

PROBE_DIR = ROOT / "data" / "probe"

#: Fixed so the held-out set is the same set every time anyone runs the sweep.
SPLIT_SEED = 20260806
TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Example:
    text: str
    label: int
    why: str = ""
    history: tuple[tuple[str, str], ...] = ()
    #: "seed" or "turn_log". Logged turns never enter the held-out set.
    source: str = "seed"

    @property
    def key(self) -> tuple:
        return (self.text.strip().lower(), self.history)

    def messages(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for user, assistant in self.history:
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": assistant})
        return out


@dataclass
class Split:
    train: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "train_tool": sum(e.label for e in self.train),
            "test": len(self.test),
            "test_tool": sum(e.label for e in self.test),
            "logged": sum(1 for e in self.train if e.source == "turn_log"),
        }


def _example_from(row: dict, source: str = "seed") -> Example:
    history = tuple(
        (pair[0], pair[1]) for pair in (row.get("history") or []) if len(pair) == 2
    )
    return Example(
        text=row["text"],
        label=int(row["label"]),
        why=row.get("why", ""),
        history=history,
        source=source,
    )


def load_seed(directory: Path = PROBE_DIR) -> list[Example]:
    """Every ``*.jsonl`` in ``data/probe``, deduplicated on text + history."""
    if not directory.exists():
        raise FileNotFoundError(f"no dataset at {directory}")

    examples: list[Example] = []
    seen: set[tuple] = set()
    for path in sorted(directory.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: {exc.msg}") from None
            if "text" not in row or "label" not in row:
                raise ValueError(f"{path.name}:{number}: needs 'text' and 'label'")
            example = _example_from(row)
            if example.key in seen:
                continue
            seen.add(example.key)
            examples.append(example)

    if not examples:
        raise ValueError(f"{directory} has no usable examples")
    return examples


def load_logged(conn: sqlite3.Connection, limit: int = 2000) -> list[Example]:
    """Real turns from ``turn_log``, as positive examples only.

    A turn that ran a tool and got a real result back is evidence that a tool
    was wanted. There is no equivalent evidence for the negative class -- a
    turn the gate called "chat" is just the gate's own opinion, and training on
    that would teach the probe to agree with the baseline it is supposed to
    beat. So this returns label-1 examples only, it is off by default in the
    sweep, and anything it returns lands in train.
    """
    rows = conn.execute(
        "SELECT transcript, tool_name, tool_result_json FROM turn_log"
        " WHERE tool_name IS NOT NULL AND transcript IS NOT NULL AND transcript != ''"
        " ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()

    examples: list[Example] = []
    seen: set[tuple] = set()
    for row in rows:
        result = json.loads(row["tool_result_json"]) if row["tool_result_json"] else {}
        if result.get("needs") or result.get("error"):
            continue  # a clarification is not evidence the tool was right
        example = Example(
            text=row["transcript"], label=1, why=f"logged {row['tool_name']}",
            source="turn_log",
        )
        if example.key in seen:
            continue
        seen.add(example.key)
        examples.append(example)
    return examples


def split(
    examples: list[Example],
    logged: list[Example] | None = None,
    test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> Split:
    """Stratified 80/20. Logged turns are appended to train, never to test."""
    from sklearn.model_selection import train_test_split

    labels = [e.label for e in examples]
    train, test = train_test_split(
        examples,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )

    seed_keys = {e.key for e in examples}
    for example in logged or []:
        # Never let a logged turn duplicate something already held out.
        if example.key not in seed_keys:
            train.append(example)

    return Split(train=list(train), test=list(test))

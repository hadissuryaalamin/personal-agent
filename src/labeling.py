"""What a label means here, and the checks that keep it honest.

Kept apart from dataset.py on purpose. That file is 143 sentences; this one is
the rule those sentences are written against. When a label looks wrong, the
argument belongs here, not buried in a list of prompts.

WHY THE RULE DIFFERS FROM THE PAPER

PROBE&PREFILL derives labels by measurement: force the model to answer with no
tool, and label the task tool-necessary exactly when it gets it wrong. That
works because whether a tool helps is a question about the MODEL's competence
-- it can do 12 + 7, it cannot do C(80, 40) -- and the boundary is genuinely
fuzzy, which is why they need a medium difficulty band at all.

Here it is a question about ACCESS. No model of any size can know when this
user's tutorial starts, and no tool can make small talk go better. So the
label is not something to measure; it is a property of the question, and
measuring it would be an expensive way to confirm a certainty.

    label 1   the answer lives in memory/events.json, and one of the tools
              can reach it
    label 0   no tool this agent has could change the right action

THE LABELS ARE A FACT ABOUT TODAY'S TOOL SET, NOT ABOUT THE QUESTIONS

This is the part that will age. The five `clock` tasks are label 1 only
because the agent has no clock -- the current time reaches it solely as the
`now` field inside a get_schedule result. Add a clock and all five flip to 0.
Add web search and most of `knowledge` and `general-hard` flip to 1.

So a feature file is only valid for the tool set it was extracted under, which
is why model.py hashes the system prompt together with the schema and stores
the hash beside the features.

WHY GENERAL KNOWLEDGE IS NOT MEASURED

An earlier version of this project promised to measure the 18 knowledge labels
against the model before training, on the grounds that whether it knows the
capital of Australia is a fact about the model. That was dropped deliberately.

The pipeline decides one thing: call a tool, or answer directly. No tool this
agent has can answer the capital of Australia, so if the model gets it wrong
the right action is still to answer directly and be wrong. The label is 0
either way, and measuring would report on the model's knowledge rather than on
the routing decision being replicated. Those tasks keep source="measured" as a
marker that the ANSWER may be wrong even though the ROUTING is not in doubt.

WHY DIFFICULTY IS TAGGED

If tool-needed were always "what's my next class" and no-tool always "why is
the sky blue", the two clouds would separate on vocabulary alone and the probe
would score near 1.0 while proving nothing. So each class carries cases that
sit on the boundary:

    hard, needs a tool     no schedule words at all
                           "Can I go to the gym at four?"
    hard, needs no tool    schedule words everywhere, no personal data
                           "How many weeks is a typical semester?"

A probe that beats prompting only on the easy cases has earned nothing. The
hard slice is the one to read.
"""

from __future__ import annotations

# group -> (label, difficulty, source). The single place these three travel
# together; dataset.py builds the task file from this table, so a group cannot
# acquire a label in one place and a difficulty in another.
GROUPS: dict[str, tuple[int, str, str]] = {
    "read-easy":    (1, "easy",   "definitional"),
    "read-medium":  (1, "medium", "definitional"),
    "read-hard":    (1, "hard",   "definitional"),
    "write":        (1, "medium", "definitional"),
    "write-done":   (1, "medium", "definitional"),
    "write-log":    (1, "medium", "definitional"),
    "write-remove": (1, "medium", "definitional"),
    "write-hard":   (1, "hard",   "definitional"),
    "clock":        (1, "hard",   "definitional"),
    "social":       (0, "easy",   "definitional"),
    "knowledge":    (0, "medium", "measured"),
    "general-hard": (0, "hard",   "definitional"),
    "about-self":   (0, "easy",   "definitional"),
    "write-trap":   (0, "hard",   "definitional"),
    "chat-idiom":   (0, "hard",   "definitional"),
}

DIFFICULTIES = ("easy", "medium", "hard")


def check(items: list[dict]) -> list[str]:
    """Everything that would make a downstream number meaningless.

    Returns problems rather than raising, so a caller can print all of them at
    once. A duplicate prompt is the dangerous one: with cross-validation it
    puts the same sentence in a training fold and a test fold, and the held-out
    score stops being held-out.
    """
    problems = []

    seen: dict[str, str] = {}
    for i in items:
        key = i["prompt"].strip().lower().rstrip("?.!,")
        if key in seen:
            problems.append(f"duplicate prompt: {i['prompt']!r} (also {seen[key]})")
        seen[key] = i["id"]

    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")

    for i in items:
        if i["group"] not in GROUPS:
            problems.append(f"{i['id']}: unknown group {i['group']!r}")
            continue
        label, difficulty, source = GROUPS[i["group"]]
        if (i["label"], i["difficulty"], i["source"]) != (label, difficulty, source):
            problems.append(
                f"{i['id']}: {i['group']} should be "
                f"({label}, {difficulty}, {source}), got "
                f"({i['label']}, {i['difficulty']}, {i['source']})")
        if source == "measured" and not i["expect"]:
            problems.append(f"{i['id']} is measured but has no expected answer")
        if source == "definitional" and i["expect"]:
            problems.append(f"{i['id']} is definitional but carries an answer")

    return problems

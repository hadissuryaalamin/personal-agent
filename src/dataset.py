"""The 143 prompts, and the train/test split over them.

What a label MEANS, and why general knowledge is not measured, lives in
labeling.py. This file is the sentences themselves plus the mechanics of
turning them into data/tasks.json.

    python -m src.dataset              # build the task file
    python -m src.dataset --split      # build, then write train/test.json

WHY THE WORDING IS SPOKEN, NOT WRITTEN

Every one of these arrives through Parakeet, so the probe must be trained on
what STT actually emits: no final punctuation on some, filler words, false
starts, and numbers spelled out the way they are said. A set written in tidy
prose would be a different distribution from the one at deployment.

WHY THE SPLIT IS STRATIFIED ON (label, difficulty)

A plain random split of 143 items can hand most of the hard cases to one side.
That is the worst available outcome, because the hard slice is the only part
carrying information -- a probe TESTED mostly on easy cases reports a number
that means nothing. The seed is fixed so a probe trained today and one trained
next week are comparable.

Note that Phase 3 does not use this split for its headline number: 30% of 143
leaves 13 hard items, too few to measure. It reports K-fold over the whole set
and keeps train/test.json as an untouched final holdout, spent once.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from .labeling import GROUPS, check

ROOT = Path(__file__).resolve().parent.parent / "data"
DATA = ROOT / "tasks.json"

SEED = 20260806
TEST_FRACTION = 0.3

# --------------------------------------------------------------------------
# label 1 -- the answer is in memory/events.json and nowhere else
# --------------------------------------------------------------------------

NEEDS_TOOL_EASY = [
    "What's my next class?",
    "What classes do I have tomorrow?",
    "How many assignments do I have?",
    "When is my next assignment due?",
    "What's on my schedule today?",
    "Do I have any classes today?",
    "What's my schedule for this week?",
    "When is assignment one due?",
    "How many assignments do I have this month?",
    "What have I got on tomorrow?",
    "Show me my schedule for the next two weeks",
    "What's my next lecture?",
    "Do I have any tutorials tomorrow?",
    "What reminders do I have?",
    "List my tasks",
    "What's due this week?",
    "When's my next tutorial?",
    "What's coming up this semester?",
    "Do I have anything on Friday?",
    "What are my deadlines?",
    # the disfluent forms Parakeet actually produces
    "Can you show me uh how many assignments I have in this month?",
    "So what do I have on um tomorrow?",
    "I need to know my assignments this month",
    "Tell me what's my next class again",
    "What's on today, like, the whole day?",
]

NEEDS_TOOL_MEDIUM = [
    "Am I free tomorrow afternoon?",
    "Have I got anything before lunch today?",
    "How much work have I got left on assignment one?",
    "Is there anything I'm forgetting this week?",
    "What's the busiest day this week?",
    "How many hours of study have I got left?",
    "Anything urgent coming up?",
    "Is my Thursday clear?",
    "What room is my next class in?",
    "Do I need to be anywhere in the next hour?",
    "How long until my next class?",
    "Have I got time for lunch before my tutorial?",
    "What did I have on yesterday?",
    "Is assignment one finished yet?",
    "Which course has the most work due?",
    "What's left to do before Friday?",
    "Have I got two things on at once anywhere this week?",
    "When does my last class of the day finish?",
]

# No schedule vocabulary at all. Answering these still requires reading the
# store, and that is the whole point -- a keyword rule cannot catch them.
NEEDS_TOOL_HARD = [
    "Should I start working tonight?",
    "Do I have time to watch a movie this evening?",
    "Can I go to the gym at four?",
    "Am I going to be busy this week?",
    "Is it a good time to take a break?",
    "Should I be worried about anything right now?",
    "Can I sleep in tomorrow?",
    "How's my week looking?",
    "Do I need to leave soon?",
    "Is there anything I should be doing right now?",
    "Can I take Saturday off?",
    "How far behind am I?",
    "Am I on track?",
    "Is tonight going to be a late one?",
    "Have I got a spare couple of hours anywhere?",
    "Would it be silly to go out on Thursday?",
    "Is there enough time left or should I panic?",
    "Is now a bad time to call mum?",
    "Have I got room to take on something else?",
    "Should I say yes to Friday drinks?",
    "Am I going to regret staying up?",
    "Is there a gap somewhere I can use?",
    "Can I afford an afternoon off?",
    "How's tomorrow shaping up?",
    "Is the rest of today quiet?",
    "Do I have breathing room this week?",
    "Should I be somewhere right now?",
    "Is there anything urgent left?",
    "Am I overcommitted?",
    "Have I got anything clashing?",
    "Is my morning free enough for a run?",
    "Can I fit a haircut in this week?",
    "Would tomorrow be a better day for this?",
]

NEEDS_TOOL_WRITE = [
    "Add a task to read chapter four by Friday",
    "Remind me to submit the lab report on Tuesday",
    "New assignment, data analysis, due next Monday, about six hours",
    "Put down a task to finish the essay by the tenth",
    "I need to add a reading task for next week",
    "Can you note down that the quiz is on the twentieth",
    "New task, revise lecture notes, due Thursday",
    "Save this, group presentation due in two weeks, ten hours",
]

# --------------------------------------------------------------------------
# label 0 -- no tool exists that would make the answer better
# --------------------------------------------------------------------------

NO_TOOL_SOCIAL = [
    "Hi",
    "Hey there",
    "Good morning",
    "How are you?",
    "How's it going?",
    "Thanks",
    "Thank you so much",
    "That's great",
    "Cool",
    "Never mind",
    "Goodbye",
    "See you later",
    "Nice one",
    "Sorry, what?",
    "You're a legend",
    "Ha, fair enough",
    "I'm pretty tired today",
    "It's freezing in here",
    "I had a good weekend",
    "This weather is miserable",
    "I'm a bit stressed out",
    "Talk to me for a sec",
    "Morning",
    "How's your day?",
    "Cheers",
    "No worries",
    "All good",
    "Alright then",
    "Catch you later",
    "That's hilarious",
    "Yeah nah",
    "Fair call",
    "I'm back",
    "Hello again",
]

# The one group where the paper's measurement procedure applies unchanged:
# whether the model knows these is a fact about the model.
NO_TOOL_KNOWLEDGE = [
    ("What's the capital of Australia?", "canberra"),
    ("Who wrote Hamlet?", "shakespeare"),
    ("What's twelve plus seven?", "19"),
    ("How many days are in February in a leap year?", "29"),
    ("What's the boiling point of water in Celsius?", "100"),
    ("Why is the sky blue?", "scatter"),
    ("What language do they speak in Brazil?", "portuguese"),
    ("How many continents are there?", "seven"),
    ("What's the largest planet in the solar system?", "jupiter"),
    ("Who painted the Mona Lisa?", "vinci"),
    ("What does CPU stand for?", "central processing"),
    ("How many minutes are there in a day?", "1440"),
    ("What's the square root of one hundred and forty four?", "12"),
    ("What's photosynthesis?", "light"),
    ("What's the tallest mountain in the world?", "everest"),
    ("How many sides does a hexagon have?", "six"),
    ("What year did the Second World War end?", "1945"),
    ("What's the chemical symbol for gold?", "au"),
]

# The trap. Every one of these is soaked in schedule vocabulary -- semester,
# deadline, lecture, hours, week -- and not one of them touches the store.
# A prompt rule built on those words fires on all of them.
NO_TOOL_HARD = [
    "How many weeks is a typical university semester?",
    "What time zone is Canberra in?",
    "What does ENGN stand for?",
    "How long should I spend on a two thousand word essay?",
    "What's a good way to manage deadlines?",
    "How many hours a week should a full time student study?",
    "When do most Australian universities start semester two?",
    "Is it better to study in the morning or at night?",
    "How do I stop procrastinating on assignments?",
    "What's the difference between a lecture and a tutorial?",
    "How many units is a normal course load?",
    "What's the best way to plan out a week?",
    "Do universities usually run classes on Fridays?",
    "How long is a typical lecture?",
    "What does it mean when an assignment is worth thirty percent?",
    "Any tips for writing a good report?",
    "Is it normal to feel behind at university?",
    "What's the Pomodoro technique?",
    "How early should you start on a big assignment?",
    "What's a reasonable number of contact hours?",
    "How many assignments does a normal semester have?",
    "Is a two hour tutorial common?",
    "What time do most lectures start?",
    "How far ahead do universities publish timetables?",
    "Do tutorials usually have attendance marks?",
    "What's a credit point?",
    "How many students are in a typical tutorial?",
    "Is it normal to have no classes on a Wednesday?",
    "How long is the mid semester break?",
    "What's a good weekly study routine?",
    "Do most people work while studying full time?",
    "How many hours ahead should you plan your week?",
    "Is it better to have classes in a block or spread out?",
    "What does a course outline usually include?",
    "How much reading is normal per week?",
    "When are exams usually held?",
    "Is a group assignment worth more than an individual one?",
    "How do you work out how long something will take?",
]

# I first wrote these as label 0, assuming the clock was injected into the
# system prompt each turn. It is not. src/config.py SYSTEM_PROMPT carries no
# date and no time; the only place the current moment appears anywhere is the
# "now" field that get_schedule and get_next put in their RESULT payload
# (src/tools.py). So the model cannot answer any of these without a call,
# and label 1 is correct.
#
# This is uncomfortable and worth leaving visible rather than tidying away.
# There is no clock tool -- the time arrives as a side effect of asking for a
# schedule. The label is therefore a fact about how the agent is configured
# today, not a fact about the question, and it flips the moment either a clock
# tool is added or the time is put back in the prompt.
#
# Kept because they are genuinely hard in the useful direction: no schedule
# vocabulary at all, and a tool is still required.
NEEDS_TOOL_CLOCK = [
    "What time is it?",
    "What's today's date?",
    "What day is it today?",
    "Is it morning or afternoon?",
    "What month are we in?",
]

NO_TOOL_ABOUT_SELF = [
    "What can you do?",
    "Can you hear me?",
    "Are you there?",
    "What's your name?",
    "Repeat that",
    "Say that again",
    "Can you speak up a bit?",
    "Are you an AI?",
    "How do I stop you listening?",
    "Say that more slowly",
    "Can you understand me okay?",
    "Are you listening right now?",
    "How do I make you stop talking?",
    "Do you remember what I said before?",
    "Are you recording me?",
    "Speak a bit louder please",
    "What can't you do?",
    "Are you always on?",
    "Who made you?",
    "Can you hear me properly now?",
]

# --------------------------------------------------------------------------
# The three tools the set never covered
#
# mark_done, log_hours and remove_entry existed in the store for months and
# were reachable only from the keyboard, so no prompt here ever needed them.
# A probe that has never seen "I've finished assignment one" has no reason to
# know it needs a tool -- that is not an untested capability, it is an absent
# one.
# --------------------------------------------------------------------------

NEEDS_TOOL_DONE = [
    "I've finished assignment one",
    "Mark the lab report as done",
    "I submitted the essay this morning",
    "Tick off the reading task",
    "That quiz is done now",
    "I handed in the data analysis",
    "Can you mark the group presentation finished",
    "Done with the revision task",
    "I finished chapter four",
    "Cross off the rego reminder",
    "The lab report's in, mark it off",
    "I've done the essay, take it off my list",
    "Assignment one is complete",
    "Just submitted the report",
    "You can mark the tutorial prep as done",
    "Finished the presentation slides",
    "Um I've done that reading task now",
    "Take the quiz off, I sat it yesterday",
]

NEEDS_TOOL_LOG = [
    "I worked three hours on the essay",
    "Log two hours on assignment one",
    "Just did an hour of study on the report",
    "Put down four hours for the data analysis",
    "I spent about ninety minutes on the reading",
    "Add three hours to the presentation",
    "That was two hours on the lab report",
    "I did five hours on assignment one today",
    "Log half an hour on the revision",
    "Spent the whole morning on the essay, about three hours",
    "Take an hour off the essay, I logged too much",
    "I put in six hours on the group project",
    "Record two and a half hours for the report",
    "I worked on assignment one for two hours this arvo",
    "Chuck in an hour for the reading task",
]

NEEDS_TOOL_REMOVE = [
    "Delete the rego reminder",
    "Remove the quiz from my list",
    "Get rid of that reading task",
    "Take the group presentation off my schedule",
    "Cancel the reminder about the lab report",
    "Delete assignment one, it got withdrawn",
    "That task shouldn't be there, remove it",
    "Scrap the revision task",
    "Can you delete the essay entry",
    "Bin the reminder about the tutorial",
    "Remove the data analysis task please",
    "Wipe the quiz entry, it's not happening",
]

# Writing intent with the verb left out. The three lists above can all be
# caught by a keyword rule -- "mark", "log", "delete" are right there. These
# cannot, and they are how people actually speak once they trust the thing.
NEEDS_TOOL_WRITE_HARD = [
    "Actually I already did that one",
    "Scratch that, it's finished",
    "That one's off, they cancelled it",
    "I got through about three hours of it today",
    "Never mind the quiz, it's not happening anymore",
    "Chapter four, sorted",
    "That's out of the way now",
    "Two hours down on the essay",
    "It's not due anymore",
    "I'm through with the lab report",
    "Knock the presentation off the list",
    "Put me down for another hour on that",
    "The essay can go, I already handed it in",
    "That's one down, the essay's in",
    "Add another two hours to what I've done on it",
    "Right, the reading's done and dusted",
    "So um the quiz got cancelled, take it out",
    "I've been at the report all morning, about four hours",
]

# --------------------------------------------------------------------------
# The traps that face the other way
#
# Every one of these is soaked in the vocabulary of the three new tools --
# finished, submitted, hours, delete, mark off -- and not one of them touches
# the store. Without these the new groups would separate on their verbs and
# prove nothing at all.
# --------------------------------------------------------------------------

NO_TOOL_WRITE_TRAP = [
    "How many hours should a three thousand word essay take?",
    "Is it better to log study hours daily or weekly?",
    "What does it mean when an assignment is marked as submitted late?",
    "How do you know when an essay is actually finished?",
    "Should you delete old notes at the end of semester?",
    "What happens if you hand in an assignment after the deadline?",
    "How many hours a day is too much study?",
    "Is it worth tracking how long tasks take you?",
    "What's a good way to break a big task into smaller ones?",
    "Can you delete things from my schedule?",
    "Can you mark things off for me?",
    "What's the point of estimating hours before you start?",
    "Is it normal for an essay to take twice as long as planned?",
    "How do people remember to remove things from their to do lists?",
    "What's the difference between finished and submitted?",
    "Should I finish the easy tasks first or the hard ones?",
    "How many hours of work is a thirty percent assignment worth?",
    "Is deleting a task the same as giving up on it?",
    "What's a realistic number of tasks to finish in a day?",
    "Do people usually log their hours as they go or at the end?",
]

# "I'm done" is the same two words in "I'm done with assignment one" and "I'm
# so done with this week". One is a tool call and one is a complaint, and the
# difference is not in the vocabulary.
NO_TOOL_CHAT_IDIOM = [
    "I'm so done with this week",
    "That lecture finished me off",
    "I'm absolutely wrecked",
    "Well that's me done for today",
    "My brain is fried",
    "I could delete this whole week from my memory",
    "That was three hours of my life I won't get back",
    "I'm finished, I need a nap",
    "What a day, completely done in",
    "I've had it",
    "That's it, I'm out",
    "I'm running on empty today",
    "Honestly I'm cooked",
    "That took forever",
    "I'm counting down the hours",
    "Two more hours of this and I'm free",
    "I've finished my coffee, that's the real achievement",
    "Never again",
]


def build() -> list[dict]:
    """Assemble the set. IDs are stable so a split can be reproduced."""
    items: list[dict] = []

    def add(prompts, group):
        # label, difficulty and source come from labeling.GROUPS, so they
        # cannot drift away from the rule they are supposed to follow.
        label, difficulty, source = GROUPS[group]
        for i, p in enumerate(prompts):
            expect = None
            if source == "measured":
                p, expect = p
            items.append({
                "id": f"{group}-{i:03d}",
                "prompt": p,
                "label": label,          # 1 = a tool is needed
                "group": group,
                "difficulty": difficulty,
                "source": source,
                "expect": expect,        # substring grader, measured group only
            })

    add(NEEDS_TOOL_EASY, "read-easy")
    add(NEEDS_TOOL_MEDIUM, "read-medium")
    add(NEEDS_TOOL_HARD, "read-hard")
    add(NEEDS_TOOL_WRITE, "write")
    add(NEEDS_TOOL_CLOCK, "clock")

    add(NEEDS_TOOL_DONE, "write-done")
    add(NEEDS_TOOL_LOG, "write-log")
    add(NEEDS_TOOL_REMOVE, "write-remove")
    add(NEEDS_TOOL_WRITE_HARD, "write-hard")

    add(NO_TOOL_SOCIAL, "social")
    add(NO_TOOL_KNOWLEDGE, "knowledge")
    add(NO_TOOL_HARD, "general-hard")
    add(NO_TOOL_ABOUT_SELF, "about-self")
    add(NO_TOOL_WRITE_TRAP, "write-trap")
    add(NO_TOOL_CHAT_IDIOM, "chat-idiom")

    return items


def load() -> list[dict]:
    """The built task set. Everything downstream reads through here."""
    return json.loads(DATA.read_text(encoding="utf-8"))


def split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    buckets: dict[tuple, list[dict]] = collections.defaultdict(list)
    for i in items:
        buckets[(i["label"], i["difficulty"])].append(i)

    train, test = [], []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda x: x["id"])
        rng.shuffle(group)
        n_test = round(len(group) * TEST_FRACTION)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def report(name: str, rows: list[dict]) -> None:
    pos = sum(r["label"] for r in rows)
    print(f"  {name:<6} {len(rows):>4}   tool {pos:>3}  /  no tool {len(rows) - pos:>3}")
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["difficulty"] == diff]
        if not sub:
            continue
        p = sum(r["label"] for r in sub)
        print(f"           {diff:<7} {len(sub):>3}   {p:>3} / {len(sub) - p:>3}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true",
                    help="also write train.json and test.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the split without writing anything")
    args = ap.parse_args()

    items = build()
    problems = check(items)
    if problems:
        for p in problems:
            print(f"  PROBLEM  {p}")
        print("\n  task set rejected, nothing written")
        return 1

    if not args.dry_run:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(items, indent=2), encoding="utf-8")

    pos = sum(i["label"] for i in items)
    print(f"  {len(items)} tasks -> {DATA}")
    print(f"  needs a tool {pos}   no tool {len(items) - pos}")
    print()
    counts: dict[str, int] = collections.Counter(i["group"] for i in items)
    for group, (label, diff, source) in GROUPS.items():
        print(f"  {group:<14} label {label}  {diff:<7} {source:<13} {counts[group]:>3}")
    print()
    for diff in ("easy", "medium", "hard"):
        sub = [i for i in items if i["difficulty"] == diff]
        p = sum(i["label"] for i in sub)
        print(f"  {diff:<7} {len(sub):>4}   tool {p:>3}  /  no tool {len(sub) - p:>3}")

    if not (args.split or args.dry_run):
        return 0

    print()
    train, test = split(items)
    assert not ({i["id"] for i in train} & {i["id"] for i in test}), "leak"
    assert len(train) + len(test) == len(items)
    report("train", train)
    report("test", test)

    if args.dry_run:
        print("\n  dry run, nothing written")
        return 0

    (ROOT / "train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (ROOT / "test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(f"\n  written to {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

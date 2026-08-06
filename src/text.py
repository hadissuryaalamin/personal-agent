"""Text normaliser for every intent match.

Deliberately in one place. When each module wrote its own normaliser, they all
replaced apostrophes with a space — so `"don't save it"` split into
`don | t | save | it`, nothing matched the refusal list, and the word `save` was
read as agreement. The agent saved an event the user had just refused.

The rule: apostrophes are DROPPED (joining the word), every other punctuation
mark becomes a space.

    "don't save it"  -> "dont save it"
    "I'm done"       -> "im done"
    "that's all"     -> "thats all"
    "3 p.m., okay?"  -> "3 p m okay"
"""

import re

_APOSTROPHE = re.compile(r"['’ʼ]")
_OTHER = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, no punctuation, single spaces, apostrophes joined."""
    t = _APOSTROPHE.sub("", text.lower())
    t = _OTHER.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def words(text: str) -> list[str]:
    """`normalize()` split into a word list."""
    t = normalize(text)
    return t.split() if t else []


if __name__ == "__main__":
    CASES = [
        ("don't save it", "dont save it"),
        ("I'm done", "im done"),
        ("that's all", "thats all"),
        ("Hello,  world!", "hello world"),
        ("3 p.m., okay?", "3 p m okay"),
        ("I’ve got a quiz", "ive got a quiz"),
        ("", ""),
        ("   ", ""),
    ]
    failed = 0
    for given, want in CASES:
        got = normalize(given)
        ok = got == want
        failed += not ok
        print(f"  {'ok   ' if ok else 'FAIL '}  {given!r} -> {got!r} (want {want!r})")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    raise SystemExit(1 if failed else 0)

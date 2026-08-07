# CLAUDE.md

Rules for any agent working in this repo. Read [PLAN.md](PLAN.md) for the spec
before changing behaviour; read this file before changing code.

## Hard invariants

These are not preferences. Breaking one is a bug even if the tests pass.

1. **The LLM never does date arithmetic.** Every relative expression is resolved
   in `src/timeparse.py` against the injected `now`. If you find yourself
   prompting the model to compute a date, you are in the wrong file.
2. **One model load, through HF transformers.** No Ollama, no llama.cpp, no
   second copy of Qwen for the probe. Hidden states are the point of the
   project; anything that hides them is disqualified.
3. **Every turn writes a `turn_log` row** — including errors, empty transcripts,
   and turns that ended in a clarification. That table is the probe's training
   data and the only way to debug a system whose input is sound. Never add a
   code path that answers the user without logging.
4. **Writes go through the audit log; deletes are soft.** `undo_last_write` must
   work for any mutation. A misheard command must be reversible.
5. **Confirm before destructive writes** (delete, or overwriting an existing due
   date / class time). Never confirm a read — it makes the agent tiresome.
6. **Ambiguity is not resolved by guessing.** Two plausible course matches, a
   vague date, or a probe score inside the uncertainty band all produce one
   clarifying question, not a best guess.
7. **`pytest` runs with no GPU, no audio device, and no network.** Tests that
   need real hardware or weights are marked `@pytest.mark.hardware` and are
   excluded from the default run.
8. **Nothing leaves the machine at runtime.** Models load from `models/`. No
   API calls, no telemetry.

## Working style

- Text mode (`python -m src.cli`) is the development surface. Get behaviour
  right there before touching audio — the voice loop is for verifying the parts
  text mode cannot exercise.
- Measure before claiming. Any accuracy or latency number in a doc must come
  from a script in `scripts/` and land in `docs/eval.md`. Do not paste numbers
  from memory or estimate them in prose.
- Assert model shapes at load (layer count, hidden size) rather than trusting
  the values written in PLAN.md. Docs go stale; assertions do not.
- Prefer a boring implementation that can be tested at the text level over a
  clever one that can only be checked by talking to it.

## Adding a tool

A new tool is not done until all four exist:

1. Schema registered in `src/tools/registry.py`
2. Unit test covering the happy path and one ambiguity case
3. At least 20 utterances added to the probe dataset
4. A row in the tool table in PLAN.md

Tools accept natural-language time expressions and fuzzy entity names. They
return either a result or `{"needs": "clarification", "question": …}` — never a
guess, never a bare exception for a foreseeable input.

## Response style (what the agent says out loud)

- Two sentences or ~320 characters, unless the user asked for a list.
- Lists cap at three items plus a count: *"Three things due this week — the
  closest is the data structures assignment, Friday."*
- Confirm writes by restating the resolved value, not the raw input: *"Added,
  due Friday the fourteenth."* That is how the user catches a misheard date.
- No filler openers. Answer, then stop.

## Environment

- Windows 11, PowerShell. Do not assume bash, `make`, or POSIX paths in scripts.
- Virtualenv at `.venv`; activate with `.\.venv\Scripts\Activate.ps1`.
- Long-running commands (model downloads, probe training, layer sweeps) run in
  the background — and whenever you start one, hand over the command to follow
  it: `Get-Content -Wait logs\<name>.log`.
- Never commit: `models/`, `data/*.db`, `data/*.npz`, `data/*.joblib`, `.venv/`,
  `.env`, `logs/`.

## Before destructive operations

Check what is actually irreplaceable first — gitignored files (`data/`,
`models/`, `.env`) exist in no commit and no remote. Do not trust a status
claim in a doc over the state on disk.

## Verification

When asked to verify something manually, give **one step at a time** and wait
for the result before moving to the next. Do not hand over a checklist of six
things to run at once.

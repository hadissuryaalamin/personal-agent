# Working prompt — PROBE&PREFILL replication

Paste this to resume the work in a fresh session. It carries the measured
facts, so that nothing here has to be rediscovered; everything below with a
number attached was observed on this machine, not estimated.

---

## The task

Replicate PROBE&PREFILL (arXiv 2605.09252) on a personal voice assistant that
already exists at `e:\personal-agent`.

The paper's claim: a linear probe reading the last-token hidden state before
generation predicts whether a tool call is needed (AUROC 0.89–0.96), and
prefilling a steering sentence into the assistant turn acts on that
prediction — 48% fewer tool calls at 1.7% accuracy loss, under 1 ms overhead.

The agent today runs on Ollama with qwen2.5:7b and decides tool use the way
the paper's *Prompt-only* baseline does: a tool description plus system-prompt
instructions. The replication runs Qwen3-4B through HuggingFace Transformers
instead, because Ollama does not expose hidden states.

Two goals, and they pull in different directions — be clear which one is being
served by any given decision:

1. **Replicate the method** and see whether it works on this domain.
2. **Fix a real bug**: roughly 20% of tool calls used to arrive as prose
   (`Let me check your schedule.`) instead of structure. Hard prefill attacks
   this directly and needs no probe at all. The paper found the same failure
   in Llama — tool calls dropped to zero, accuracy 83.1% → 47.9% — and hard
   prefill recovered it.

## What makes this domain different from the paper's

**Their labels are about model competence. Ours are about access.**

The paper labels a task tool-necessary by measurement: force an answer with no
tool, and label it 1 when the model gets it wrong. That works because whether
a tool helps is a fact about the model — it can do 12 + 7, it cannot do
C(80, 40) — and the boundary is genuinely fuzzy, which is why they have a
*medium* difficulty band.

Here, no model of any size can know when this user's tutorial starts. The
boundary is sharp: *is this about the user's data?* That is why the current
prompt-only agent already scores 30/30 and 0/10 on tool routing where the
paper's models struggle. **The easy version of this problem is already
solved.** Anything this work produces has to be judged on the hard slice.

## Measured environment — do not re-derive

```
GPU        RTX 5050 Laptop, compute 12.0 (Blackwell, sm_120)
           7.96 GB usable; the Windows desktop holds ~1.06 GB at idle
driver     591.91
RAM        15.6 GB total, ~5 GB free in practice
disk       C: ~31 GB free after clearing a 22.5 GB openwebtext cache
env        .venv-probe (separate from .venv-agent, which runs the live agent)
           torch 2.11.0+cu128, transformers 5.14.1, bitsandbytes 0.50.0
```

**torch must come from `--index-url https://download.pytorch.org/whl/cu128`.**
The default PyPI wheel carries no sm_120 kernels, reports
`cuda.is_available() == True` anyway, and dies on the first real op.

### Model configuration, chosen by measurement

`Qwen/Qwen3-4B-Instruct-2507`, **4-bit NF4** via bitsandbytes, compute dtype
bf16, double quant on.

| | VRAM held | Forward, warm | 143 tasks* | Spare |
|---|---:|---:|---:|---:|
| **4-bit NF4** | 3.61 GB | **919 ms** | 131 s | **2.38 GB** |
| 8-bit | 5.34 GB | 928 ms | 133 s | 0.39 GB |
| bf16 | 7.96 GB | 48626 ms | ~2 hours | 0.00 GB |

4-bit is no slower than 8-bit and leaves six times the headroom.

Those three timings were taken on the 1183-token, three-tool prompt. At 1683
tokens the 4-bit pass is 2098 ms and the full extraction 301 s; the ranking
between the configurations is unchanged, and 4-bit is still the answer.

**bf16 does not raise OutOfMemory — it gets 53x slower.** Windows pages
oversubscribed VRAM to system RAM over PCIe. Any check that only asks "was
the output finite?" will pass that configuration. `src/model.py` now fails
anything above 5 s per pass.

Geometry: **36 layers, hidden 2560, so 37 x 2560 = 94,720 probe features.**
That matches the paper's figure for Qwen3-4B exactly — independent
confirmation the extraction is right. Under 4-bit, `sum(p.numel())` reports
2.21 B instead of 4.02 B because quantised weights are packed; the parameter
count is an artefact, not a different model.

### Four facts about the prompt, each found by checking rather than assuming

**The prompt is 1683 tokens.** `apply_chat_template` inserts the tool block
and nothing else — the agent's system prompt is a separate message that
`src/llm.py` adds itself. It was 1183 while the agent had three tools; going
to six added 500 tokens and doubled the forward pass, 929 ms → 2098 ms. Never
build a prompt by hand — call `src/model.py`'s `build_prompt`, which is the
single definition of this:

```python
msgs = [{"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user",   "content": task["prompt"]}]
tok.apply_chat_template(msgs, tools=SCHEMA, add_generation_prompt=True)
```

Anything less is a different distribution from the deployed one.

**The agent has no clock.** `SYSTEM_PROMPT` carries no date and no time. The
current moment appears only as the `now` field inside `get_schedule` and
`get_next` *results*. So *"What time is it?"* requires a tool call, and those
five tasks are label 1. That label is a fact about today's configuration, not
about the question — it flips if a clock tool is added.

**Qwen3 does not emit a bare JSON object.** Its format is

```
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

so the hard-prefill prefix is `<tool_call>\n{"name":`, **not** the paper's
`{"name":`. Using the paper's string opens a tag the model has been made to
skip.

**The last prompt token is `\n`, following `<|im_start|>assistant`.** That is
the position the probe reads, and it is the generation-start position the
paper uses.

## What already exists

```
src/labeling.py        what a label means; group table; the honesty checks
src/dataset.py         300 hand-written tasks + the split, seed 20260806
src/model.py           loading, prompt building, prompt hash, Phase 0 gate
src/hidden_states.py   Phase 2: last-token states -> data/features.npz
src/probe.py           Phase 3: K-fold probe + controls -> data/probe.joblib
                       and data/oof.json, one held-out prediction per task
src/tools.py           the tool schema the prompt is built from
src/config.py          SYSTEM_PROMPT
requirements-probe.txt
PROBE.md               findings and rationale
```

The task set: 300 tasks, 152 needing a tool, 148 not; 79 easy / 89 medium /
**132 hard**. Groups: `read-easy`, `read-medium`, `read-hard`, `write`,
`write-done`, `write-log`, `write-remove`, `write-hard`, `clock` (label 1);
`social`, `knowledge`, `general-hard`, `about-self`, `write-trap`,
`chat-idiom` (label 0).

The probe scores 0.994 overall and 0.977 on the hard slice, with 7 errors --
3 of them in `chat-idiom`, where "I'm so done with this week" reads as a
mark_done command. The bag-of-words baseline makes 49 errors, and only 2 of
the probe's 7 overlap with them.

Two properties of it matter more than the count.

**Labels are `definitional` except for `knowledge`, and none of them are
measured.** Where the answer lives in `memory/events.json`, or where no tool
could possibly help, measuring the label would be an expensive way to confirm
a certainty. The 18 general-knowledge tasks keep `source="measured"` and an
`expect` substring, but measuring them was dropped deliberately: no tool this
agent has can answer the capital of Australia, so if the model gets it wrong
the right action is still to answer directly and be wrong. The label is 0
either way. The full argument is in `src/labeling.py`.

**The hard slice is the only part that proves anything.** It deliberately
contains cases that defeat vocabulary matching in both directions: tasks
needing a tool with no schedule words at all (*"Can I go to the gym at
four?"*, *"How far behind am I?"*) and tasks needing none that are saturated
with them (*"How many weeks is a typical semester?"*). A probe that beats
prompting only on easy cases has earned nothing.

Wording is spoken, not written — filler words, missing punctuation, numbers
said aloud — because every question arrives through Parakeet at deployment.

## Status

| | | |
|---|---|---|
| 0 | Hardware and model gate | **passed** |
| 1 | Task set built and split | **done** |
| 2 | Extract last-token hidden states | **done** — 300 x 94,720 in 567 s |
| 3 | Train the probe, report AUROC | **done** — held-out 0.994, hard 0.977 |
| 4 | Prefill at inference | **done** |
| 5 | Sweep the threshold, compare to baselines | **done** — all-HF cached 98% vs 69% baseline, +57 ms/turn |

---

## Phase 2 — extraction

For each of the 300 tasks, run one forward pass with `output_hidden_states=True`
and keep `torch.cat([h[0, -1, :] for h in hidden_states])` — 94,720 floats.

- Build the prompt exactly as specified above. This is the single most
  likely place to silently diverge from deployment.
- Slice the last token immediately. The full `hidden_states` for one
  1683-token prompt is 319 MB; holding several will exhaust the spare.
- Store as float16 to disk (300 x 94,720 x 2 B ≈ 57 MB), cast to float32 for
  training.
- Save alongside: task id, label, group, difficulty, prompt token count, and
  the model + quantisation used. A feature file whose provenance is unknown
  is worthless.

**Gate:** 300 vectors, all finite, width exactly 94,720, ~570 s total.

## Phase 3 — the probe

`StandardScaler` then `LogisticRegression` with L2, λ = 10000 (`C = 1/10000`),
as the paper specifies.

**Report stratified K-fold cross-validation over the whole set, not the single
split.** The 30% test split leaves only 13 hard items, and an AUROC over 13
points is noise. K-fold gives every item a held-out prediction — 42 hard
predictions instead of 13. Keep `train.json`/`test.json` as a final untouched
holdout, used once.

Report AUROC overall **and broken down by difficulty**. The headline number
will be flattered by the easy cases; the hard number is the finding.

**Gates:**
- overall AUROC > 0.85 → proceed
- around 0.5 → a bug in this code, not a result about the paper
- hard-slice AUROC near chance while overall is high → the probe learned
  vocabulary, and the honest report says so

**If the numbers disappoint, there is no reference point.** The paper reports
0.894 for Qwen3-**1.7B** and nothing for a 4-bit 4B, so a poor result cannot
be attributed between method, quantisation, and bug. Qwen3-1.7B in bf16
(3.44 GB, 173 ms/pass) is already in the cache; running the pipeline once
against it costs about three minutes and buys a known-good comparison. Do
this the moment anything looks wrong.

## Phase 4 — prefill

`p = sigmoid(z / T)`, T = 2.0.

**Soft** (a suggestion the model may decline), appended to the assistant turn:

- `p < τ` → `I can solve this directly without using a tool.`
- `p ≥ τ` → `I need to use a tool for this question.`

**Hard** (format forced, no room to deviate):

- `p ≥ τ` → begin the assistant turn with `<tool_call>\n{"name":`
- `p < τ` → let it answer directly

Prefill tokens go after the generation prompt; `generate()` continues from
there. Hard prefill is the half that addresses the agent's actual bug, and it
needs no probe — it can be tested on its own first.

## Phase 5 — the comparison

Sweep τ from 0.1 to 0.9, recording accuracy and tool-call count at each point.
Compare against the five fixed prompt-only baselines (Force / Default /
Necessary / Sparse / No-tool).

**Gate:** the probe's curve sits above the baseline points, not on them. And
check the hard slice separately — the whole argument for this work is that
prompting controls tool use too bluntly, and that claim is only tested where
the decision is actually difficult.

## Two standing rules from this project

**Give the model a decision, never a calculation.** Choosing a tool and
mapping "this two weeks" to `days=14` it does well. Comparing timestamps,
filtering a list, keeping a count it does badly — every one of those was moved
into Python, and `get_next` exists as a separate tool for exactly this reason.

**Almost every "bug" reported here turned out to be an old process running old
code.** Check `build:` in the startup log before investigating behaviour.

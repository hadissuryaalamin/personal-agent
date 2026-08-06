# probe

Replicating PROBE&PREFILL (arXiv 2605.09252) on this agent's own domain:
read the hidden state before generation, decide with a linear classifier
whether a tool is needed, and steer the model by prefilling its turn.

Kept separate from `src/` on purpose. The running agent stays on Ollama and
is not touched by anything here; `.venv-probe` is its own environment so a
3 GB torch install cannot destabilise a voice assistant that currently works.

## Phases

| | | Status |
|---|---|---|
| 0 | Prove the GPU runs the model at all | **passed** |
| 1 | Build and split the task set | **done** |
| 2 | Extract last-token hidden states | **done** |
| 3 | Train the linear probe, report AUROC | **done** — held-out 0.994 |
| 4 | Prefill at inference, sweep the threshold | **done** |
| 5 | Compare architectures end to end | **done** — 98% vs 69% |

## Phase 0 — the hardware gate

The card is an RTX 5050 Laptop, compute capability **12.0** (Blackwell,
`sm_120`). This is the part of the plan most likely to fail outright, so it
runs first.

`torch.cuda.is_available()` is not a sufficient test. It answers whether a
driver and a CUDA build are present, not whether the installed wheel carries
kernels compiled for this architecture. A cu124 build reports `True` on this
card and then fails on the first real operation with *no kernel image is
available for execution on the device*. So `gpu_check.py` runs an actual fp32
matmul and an actual bf16 matmul, and prints the wheel's architecture list so
that a failure is legible rather than mysterious.

```
.\.venv-probe\Scripts\python.exe -m src.model
```

Result on 2026-08-06, torch 2.11.0+cu128, driver 591.91:

```
wheel kernels  sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
PASS           fp32 matmul on GPU
PASS           bf16 matmul on GPU
```

`src/model.py` is the second half of the gate, because a matmul does not
prove a model fits. It loads the weights, runs one forward pass with hidden
states on, and reports measured VRAM at each step.

```
after weights   5.17 GB used,  2.79 GB free
parameters      1.72 B      layers 28      hidden 2048
probe features  59,392  (29 x 2048)
prompt tokens   661         forward  173 ms warm / 734 ms cold
headroom left   2.78 GB
```

Two of those numbers are worth keeping.

**661 tokens, not 17.** The tool schema is most of the prompt. The probe must
build prompts with the same schema the agent deploys — a probe trained on
bare questions would be reading a different distribution from the one it
meets at inference.

**173 ms warm.** This is not overhead the probe adds. That forward pass is
the prefill the model performs anyway; the probe only reads a tensor that
already exists, which is why the paper can claim sub-millisecond cost. All
143 tasks extract in about 25 seconds.

**VRAM is the binding constraint,** and the desktop takes ~1.06 GB before
anything loads. All three variants of Qwen3-4B-Instruct-2507 measured on the
real 1183-token prompt:

| | VRAM held | Forward, warm | 143 tasks | Spare |
|---|---:|---:|---:|---:|
| **4-bit NF4** | 3.61 GB | **919 ms** | 131 s | **2.38 GB** |
| 8-bit | 5.34 GB | 928 ms | 133 s | 0.39 GB |
| bf16 | 7.96 GB | **48626 ms** | ~2 hours | 0.00 GB |

**4-bit is the configuration.** It is no slower than 8-bit — the dequantise
cost is hidden behind the same memory-bound matmuls — and leaves six times
the headroom. 8-bit's 0.39 GB spare is not a margin: the hidden states alone
for one 1183-token prompt are 224 MB, and any longer question eats the rest.

**bf16 did not fail. It got 53x slower.** Windows does not refuse to
oversubscribe VRAM; the driver pages the excess to system RAM over PCIe, so a
model that does not fit degrades instead of raising `OutOfMemory`. This is
worse than a clean failure, because without timing it you would conclude that
4B in bf16 "works".

`src/model.py` originally printed PASSED for that run — it checked that the
feature vector had the right shape and was finite, which was all true. It now
fails any configuration above 5 s per pass and warns below 0.5 GB spare.

Qwen3-1.7B in bf16 (3.44 GB, 173 ms) remains available as a reference point:
the paper reports AUROC 0.894 for it, so a pipeline that reproduces roughly
that number is a pipeline known to be wired correctly. Nothing else in this
setup provides that check.

Two notes on reading the output. Under 4-bit, `sum(p.numel())` reports 2.21 B
rather than 4.02 B because quantised weights are packed — the parameter count
is an artefact, not a different model. And the 94,720-wide feature vector
(37 x 2560) matches the paper's figure for Qwen3-4B exactly, which is
independent confirmation that the extraction geometry is right.

## Phase 1 — the task set

300 hand-written tasks in `src/dataset.py`, built to `data/tasks.json`.

```
needs a tool  152    no tool  148
easy  79      medium  89      hard  132
```

It was 143 until the agent gained three tools it had never been tested on;
see [The set at 300](#the-set-at-300) at the end for what was added and why.
The numbers quoted through the middle of this file are the 143-task ones, kept
because the argument they support is the same and the reasoning is the record.

Two decisions in there matter more than the count.

**The labelling rule is not the paper's, and could not be.** They label by
measurement: force an answer with no tool, and call the task tool-necessary
when the model gets it wrong. That works because whether a tool helps is a
question about the model's *competence*. Here it is a question about
*access* — no model of any size knows when this user's tutorial starts. For
those, measuring would be an expensive way to confirm a certainty. Only the
general-knowledge group carries `source="measured"`, and measuring even those
was dropped on purpose: no tool this agent has can answer the capital of
Australia, so the label is 0 whether the model knows it or not. The full
argument is in `src/labeling.py`.

**The hard slice is the only part that proves anything.** If tool-needed were
always "what's my next class" and no-tool always "hi", the classes would
separate on vocabulary and the probe would score near 1.0 while demonstrating
nothing. So each class carries cases sitting on the boundary:

- needs a tool, no schedule words at all — *"Can I go to the gym at four?"*
- needs no tool, schedule words throughout — *"How many weeks is a typical
  semester?"*

A probe that beats prompting only on the easy cases has earned nothing. Read
the hard slice.

Wording is spoken, not written — no final punctuation on some, filler words,
numbers said aloud — because every one of these arrives through Parakeet at
deployment and a tidy-prose set would be a different distribution.

### Three things found while checking the prompt, not assumed

**The agent has no clock.** `SYSTEM_PROMPT` in `src/config.py` carries no
date and no time. The only place the current moment appears anywhere is the
`now` field that `get_schedule` and `get_next` put in their *result* payload.
So *"What time is it?"* cannot be answered without a call, and those five
tasks moved from label 0 to label 1.

That label is a fact about how the agent is configured today, not about the
question, and it flips if a clock tool is added or the time is put back in
the prompt. It also points at a real gap: the time arrives only as a side
effect of asking for a schedule.

**The prompt is 1183 tokens, not 661.** The chat template inserts the tool
block but nothing else; the agent's system prompt is a separate message it
adds itself. A probe built from `apply_chat_template(msgs, tools=...)` alone
would be reading a prompt 522 tokens shorter than the deployed one. Both must
go in:

```python
msgs = [{"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user",   "content": task["prompt"]}]
tok.apply_chat_template(msgs, tools=SCHEMA, add_generation_prompt=True)
```

**Qwen3 does not emit a bare JSON object.** Its tool call format is

```
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

which fixes the hard-prefill string for Phase 4. The paper's `{"name":` is
not the right prefix here — it has to be `<tool_call>\n{"name":`, or the
model opens a tag it has already been made to skip. This was flagged earlier
as an unverified assumption; it is now checked.

The last prompt token is `\n`, following `<|im_start|>assistant`. That is the
position the probe reads, and it is the generation-start position the paper
uses.

## Splitting

```
.\.venv-probe\Scripts\python.exe -m src.dataset --dry-run
```

Stratified on `(label, difficulty)`, seed fixed at 20260806. A plain random
split of 143 items can hand most of the hard cases to one side, which is the
worst available outcome given that the hard slice carries the signal.

**Known limitation:** the 30% test split leaves only 13 hard items, and an
AUROC over 13 points is noisy. Phase 3 should report stratified K-fold
cross-validation over the whole set — every item gets a held-out prediction,
giving 42 hard predictions instead of 13 — and keep this fixed split as an
untouched final holdout. Expanding the hard groups is the single most
valuable addition to the set.

## Phase 2 — extraction

```
.\.venv-probe\Scripts\python.exe -m src.hidden_states --limit 4   # smoke test
.\.venv-probe\Scripts\python.exe -m src.hidden_states             # all 143
```

One forward pass per task, no tokens generated, the last token's state kept
from all 37 layers. Run of 2026-08-06:

```
143/143   133.6 s      929 ms median warm  (1704 ms cold)
prompt tokens  1175-1188      largest |x|  418      features.npz  21.0 MB
```

Everything the Phase 0 gate predicted held: 929 ms against 919 ms measured,
134 s against 131 s, and the last prompt token is `'\n'` as expected. All 143
vectors are finite and 94,720 wide.

**Storage is float16, and that needed checking rather than assuming.** fp16
overflows at 65504; transformer activations carry outliers, and an overflow
would become `inf` silently and poison the probe. The largest magnitude across
all 143 x 94,720 values is 418 — a 156x margin — so the cast is safe *for this
model at this quantisation*, which is why `src/hidden_states.py` measures it every run
instead of trusting this paragraph.

The file carries its own provenance: model, quantisation, torch and
transformers versions, GPU, layer geometry, timestamp, and a hash of
`SYSTEM_PROMPT` + tool schema. If the agent's prompt changes, a feature file
built before it stops matching, visibly.

### A number to distrust

A mean-difference direction — no training at all, just the difference of the
two class means, scored on the same points that defined them — separates the
set at AUROC 0.991 overall and 0.984 on the hard slice.

**That is not a result.** With 143 points in 94,720 dimensions, in-sample
separation is close to geometrically guaranteed; any labelling would score
well. It rules out the failure that matters at this stage — a slicing bug
handing every task the same vector, which would have shown up here as 0.5 —
and nothing else. The held-out K-fold in Phase 3 is the first number worth
reading.

Independent checks on the file: ids and labels match `tasks.json` in order, no
two rows are duplicates (max off-diagonal cosine 0.9994, min 0.3293), and row
norms sit in 864–1066. The high mean cosine of 0.71 is expected — 1180 of the
~1183 prompt tokens are shared by every task.

## Phase 3 — the probe

*(Numbers in this section are from the 143-task set. The current set is 300;
jump to [The set at 300](#the-set-at-300) for what replaced them.)*

```
.\.venv-probe\Scripts\python.exe -m src.probe
.\.venv-probe\Scripts\python.exe -m src.probe --layers
```

`StandardScaler` then `LogisticRegression`, L2 at λ = 10000, five folds
stratified on `(label, difficulty)`. Every item is predicted by a fold that
never saw it. `train.json`/`test.json` are not touched here; the probe saved
for Phase 4 is fitted on `train.json` alone, and `test.json` stays unspent for
a single final run in Phase 5.

```
                  probe    words     n   pos
overall           1.000    0.930   143    73
easy              1.000    0.995    57    25
medium            1.000    0.964    44    26
hard              1.000    0.705    42    22
```

**The hard column is the finding, and it is not the 1.000.** It is the 0.705
next to it. The hard slice was written to defeat word matching in both
directions — tool-needed questions with no schedule vocabulary (*"Can I go to
the gym at four?"*), no-tool questions saturated with it (*"How many weeks is
a typical semester?"*) — and it did defeat word matching, which fell from
0.995 on easy to 0.705. The hidden states separated the same items completely.
Whatever the model is representing before it speaks, it is not the words.

### Why a perfect score is a problem, not a triumph

The paper reports 0.894 for Qwen3-1.7B. Beating it by this margin is not a
sign of a better method; it is a sign of an easier question. Their labels ask
whether the model is *competent* to answer, which is genuinely fuzzy — it can
do 12 + 7 and cannot do C(80, 40), with a real gradient in between. These
labels ask whether the answer is in the user's data, which is not fuzzy at
all. `src/dataset.py` predicted this and it is what happened.

So the honest reading: **the probe is not the limiting instrument any more,
the task set is.** At a ceiling, this set cannot tell a good probe from a
better one, and cannot detect a regression. Every future number from these 143
items is bounded above by 1.000 and therefore uninformative. Growing the hard
slice until something is misclassified is now the highest-value work available.

### Four controls, because 1.000 demands them

| | | |
|---|---|---|
| prompt length alone | **0.655** | a mild confound exists; it is not the answer |
| shuffled labels | **0.500, 0.541, 0.463** | the fit cannot memorise noise |
| margin | **gap 1.90** | z ≥ +0.38 for tool, z ≤ −1.52 for none, no overlap |
| layer 0 | **0.492** | see below |

The permutation control is the one that matters most. With 143 points in
94,720 dimensions, *any* labelling is linearly separable unless the
regularisation is doing real work — λ = 10000 is not decoration, and shuffled
labels collapsing to chance is the proof that it holds.

The margin control matters because AUROC only ranks. Two clouds touching at a
single point rank perfectly and would shatter under any distribution shift.
A gap of 1.90 between the closest positive and the closest negative is a
different and much stronger claim than perfect ordering.

**Layer 0 is the sharpest control of the four, and it was free.** The
last-token embedding is the embedding of `'\n'` — literally the same 2560
numbers for all 143 tasks, verified identical, one unique row. Feeding a
pipeline 143 identical vectors with mixed labels must produce chance, and it
produced 0.492. A cross-validation with any leak in it could not have done
that.

(Why 0.492 rather than exactly 0.500: with constant features the decision
value is just the intercept, which differs slightly per fold because each
fold's training set has a slightly different class balance. Pooling
out-of-fold scores then compares numbers from five different fits and invents
an ordering out of nothing. It is invisible here against a gap of 1.90, but it
is a real artefact of pooling and worth knowing about.)

### Where the decision is formed

```
layer   0  0.492   <- embedding of '\n', identical for every task
layer   4  0.922
layer  11  0.989
layer  25  1.000   <- and flat through 34
layer  36  0.997
```

The routing decision is essentially settled by layer 11 of 36, a third of the
way in, and the last two layers are very slightly *worse* than the middle —
consistent with late layers specialising toward next-token prediction rather
than holding the abstract decision.

This does not make inference cheaper: the model still runs every layer to
generate. What it does say is that the decision is available long before the
first token is emitted, which is the premise the whole method rests on.

### The six thinnest predictions

All six are correct; these are the items closest to the boundary.

```
p=0.55  label=1  read-hard  Is it a good time to take a break?
p=0.57  label=1  clock      What month are we in?
p=0.60  label=1  clock      Is it morning or afternoon?
p=0.60  label=1  clock      What time is it?
p=0.64  label=1  read-hard  Would it be silly to go out on Thursday?
p=0.65  label=1  clock      What's today's date?
```

Four of the five `clock` tasks are in there, which is exactly right and worth
keeping in view. Those are label 1 only because this agent has no clock — the
time reaches it solely as the `now` field inside a `get_schedule` result. The
model has no way to know that from the prompt, so the probe is reading a
genuinely ambiguous state and still lands on the correct side by a thin
margin. **If a clock is ever added to `SYSTEM_PROMPT`, these five labels flip
and this feature file is stale.** That is what the prompt hash is for.

## The whole thing, run twice, by accident

Every number above was measured while the agent had three tools. Restructuring
the code into `src/` closed a gap that had been open for a while — mark_done,
log_hours and remove existed in the store and were reachable only from the
keyboard — so the agent went from three tools to six.

That changes the prompt, and the prompt is most of what the probe reads. The
hash caught it on the next run:

```
FAIL  these features were extracted under prompt d8ee38bc60e882f4,
      but the agent now deploys 0ff2d8cc1ff45b99
```

Which is the first time that tripwire has fired, and it fired correctly. The
hash had been written into every feature file since Phase 2 and never read;
`probe.py` now checks it before training, because a tripwire nobody checks is
decoration.

### What three more tools cost

| | three tools | six tools |
|---|---:|---:|
| prompt | 1183 tok | **1683 tok** |
| forward pass, warm | 929 ms | **2098 ms** |
| extraction, 143 tasks | 134 s | **301 s** |

**Doubling the tool count more than doubled the prefill.** Tool descriptions
are not free and they are not paid once — they are re-read on every turn, by
the live agent exactly as much as by this pipeline. That is the argument for
keeping the registry small and pushing calculation into Python rather than
adding a tool for it.

### And the result did not move

Re-extracted and retrained on the six-tool prompt:

```
                  probe    words     n   pos
overall           1.000    0.930   143    73
easy              1.000    0.995    57    25
medium            1.000    0.964    44    26
hard              1.000    0.705    42    22

length only        0.655
shuffled labels    0.478, 0.521, 0.455
margin             z >= +0.51 for tool, z <= -1.54 for none, gap +2.04
```

Identical to three decimal places, on 500 extra tokens of prompt, with the
same six items closest to the boundary and a slightly wider margin (2.04
against 1.90). This was not planned as a robustness check, but it is one: the
finding survived a substantial change to the input it reads. What it does not
do is make the ceiling less of a problem — a set that scores 1.000 twice still
cannot tell a good probe from a better one.

## The set at 300

The refactor that unified the tools exposed three the agent had always been
able to perform and never been asked to: `mark_done`, `log_hours`,
`remove_entry`. The task set had been written when there were three tools, so
it contained no sentence that needed any of them. That is not an untested
capability — for the probe it is an absent one. A classifier that has never
seen *"I've finished assignment one"* has no reason to know it needs a tool.

157 prompts were added, taking the set from 143 to 300.

```
                 143      300
needs a tool      73      152
no tool           70      148
easy              57       79
medium            44       89
hard              42      132
```

**Six new groups.** Three cover the new tools directly — `write-done` (18),
`write-log` (15), `write-remove` (12). But all three can be caught by a
keyword rule; *mark*, *log* and *delete* are right there in the sentence. So
three more exist to make that impossible:

- **`write-hard`** (18) — writing intent with the verb left out, which is how
  people speak once they trust the thing. *"Actually I already did that one"*,
  *"Chapter four, sorted"*, *"Two hours down on the essay"*.
- **`write-trap`** (20) — soaked in the new tools' vocabulary, touching no
  user data. *"How many hours should a three thousand word essay take?"*,
  *"Can you delete things from my schedule?"*
- **`chat-idiom`** (18) — the sharpest of the three. *"I'm done"* is the same
  two words in *"I'm done with assignment one"* and *"I'm so done with this
  week"*. One is a tool call, one is a complaint, and the difference is not in
  the vocabulary.

### The ceiling broke, which is the point

```
                  probe    words     n
overall           0.994    0.923   300
easy              1.000    0.978    79
medium            1.000    0.963    89
hard              0.977    0.823   132

accuracy        293/300  251/300
```

Seven errors, every one of them in the hard slice:

```
p=0.17  label=1  write-hard    That's out of the way now
p=0.44  label=1  write-hard    Actually I already did that one
p=0.43  label=1  read-hard     Am I going to regret staying up?
p=0.83  label=0  general-hard  What time do most lectures start?
p=0.67  label=0  chat-idiom    Two more hours of this and I'm free
p=0.57  label=0  chat-idiom    Well that's me done for today
p=0.51  label=0  chat-idiom    What a day, completely done in
```

Three of the seven are `chat-idiom`, and all three fail in the predicted
direction: idiomatic exhaustion read as a `mark_done` command. That is not an
embarrassing failure. *"Well that's me done for today"* is a sentence a person
would need context to resolve.

At 143 tasks this file reported 1.000 everywhere and had to spend a section
explaining why a perfect score was a problem rather than a triumph. It no
longer has to. There are now seven points that can get better or worse, all of
them where the decision is genuinely hard, and Phase 4 finally has something
to move.

### Where the probe and the words actually differ

| group | n | probe | words |
|---|--:|--:|--:|
| `chat-idiom` | 18 | 15/18 | **10/18** |
| `clock` | 5 | 5/5 | **1/5** |
| `read-hard` | 33 | 32/33 | 26/33 |
| `knowledge` | 18 | 18/18 | 13/18 |
| `write-trap` | 20 | 20/20 | 16/20 |
| `write-log` | 15 | 15/15 | 15/15 |
| `about-self` | 20 | 20/20 | 20/20 |

`chat-idiom` is where the bag of words comes closest to a coin toss, which is
exactly what it was written to do. `write-log` and `about-self` are where it
ties — the sentences there carry their meaning in their verbs, and no hidden
state is needed to see it.

Across the whole set the probe makes 7 errors and the words make 49, and only
**2 of the probe's 7 are shared with the words**. The probe is not a better
classifier of the same items; it fails on different and harder ones.

### Controls, at 300

```
length only        0.563   (was 0.655 at 143)
shuffled labels    0.480, 0.451, 0.495
margin             overlapping, gap -6.37
```

The margin is the one that changed character. At 143 the two clouds were
separated by a gap of 1.90 with nothing between them; now they overlap, which
is the arithmetic consequence of having seven errors and is the honest shape
of a real problem. The permutation control still collapses to chance, so the
regularisation is still doing the work that keeps 94,720 features from
memorising 300 points.

## Phase 4 — prefill

`src/prefill.py`. A chat prompt ends with `<|im_start|>assistant\n` and the
model continues from there; prefilling appends our own text to that turn
first, so the model is no longer choosing how to begin.

```
soft   a sentence it may ignore
       "I need to use a tool for this question."
       "I can solve this directly without using a tool."

hard   the opening of the structure itself
       <tool_call>\n{"name":
```

**The prefix is not the paper's.** They prefill `{"name":`. Qwen wraps tool
calls in a `<tool_call>` tag, so that string would open a JSON object inside a
tag the model writes first, and the result parses as nothing.

### Hard prefill alone: fixes one thing, breaks another

Run unconditionally over the 132 hard tasks, no probe involved:

```
                        n    tool call   correct   prose bug
baseline (no prefill)  132      19         70%        20
hard prefill           132     132         42%         0
   label 1              56      56        100%         0
   label 0              76      76          0%         0
```

It repairs the broken half perfectly and destroys the healthy half just as
perfectly. Overall accuracy *falls*, 70% to 42%. This is the argument for the
probe stated as an experiment: neither fixed policy is any good, and the
decision has to be made per question.

### The threshold sweep

Both runs are greedy and deterministic over the same tasks, so combining them
per item — hard-prefill outcome where the probe says tool, baseline outcome
where it says not — gives the whole curve for free.

τ was chosen on the 92 hard items in `train.json` and then applied once to the
40 in `test.json`, which had been untouched since Phase 1:

```
  tau    train    test
  0.4     92%     92%
  0.5     95%     95%   <- best on train
  0.6     90%     90%

prompt-only        65% on test
hard always        42% on test
```

τ = 0.5 is not a tuned value — it is p = 0.5, meaning z = 0, the classifier's
own decision boundary. The sweep confirmed the default rather than discovering
something. `test.json` is now spent; reusing it would no longer be a holdout.

## Phase 5 — three architectures, measured end to end

The sweep above is arithmetic over two runs. These are real chains, each one
question at a time, on the 132 hard tasks.

```
                              routing   test    turn      models in VRAM
Ollama, no probe                 69%   26/40   1451 ms    1  (qwen2.5:7b)
Ollama + HF probe               100%   40/40   2892 ms    2  (+ Qwen3-4B)
All-HF, prefix cached            98%   38/40   1508 ms    1  (Qwen3-4B)
```

### Two things had to be true and both were

**A probe reading Qwen3-4B can gate qwen2.5:7b.** Different family, nearly
twice the parameters, and it works. The paper could not do this: their labels
are "does this model get it wrong without a tool", which is meaningless read
off another model. These labels are about access — whether the answer lives in
the user's calendar is a property of the question, not of whoever answers.

**Ollama accepts a prefill.** Send a trailing assistant message and it
continues from it. One catch: the continuation arrives as `content`, not
`tool_calls`, because the opening tag came from us. `llm.py` would need to
parse it. 56 prefills, 56 tool calls.

### The prefix cache is what makes all-HF viable

1668 of the 1683 prompt tokens are the system prompt and the tool schema,
identical every turn.

```
uncached   2086 ms per forward pass
cached      101 ms
```

Ollama was never doing less work — it was doing the same work once. With the
cache the HF path goes from four times slower than Ollama to level with it,
and the probe rides the pass that was going to happen anyway. That is the
paper's actual claim, and it took until here to earn it.

The cached hidden state is not bit-identical (bf16 kernels batch differently
at different sequence lengths) but agrees to cosine 0.9999, and no probe
decision moved.

One trap worth recording: the probe pass leaves the cache holding head+tail.
Handing `generate()` a sequence it has already cached in full leaves it no new
tokens to process, and it emits **garbage rather than failing** —
`A111331411111 AI1113142140`. `cache.crop(prefix.length)` before generating.

### What to build, if this ships

All-HF with the prefix cache: 98% routing against 69%, for 57 ms a turn. Not
the two-model split — it is both slower and needs 8.6 GB of an 8 GB card.

The one refinement: the all-HF run used the `deployed`-style probe (0.977 on
hard) because it reads the pass that already exists. The `bare` probe scores
0.988 and costs a separate 76 ms pass — on the same model, in the same
process, with no contention. Buying 0.011 AUROC for 76 ms is probably worth
it, and both probes are saved.

### Still not measured

Easy and medium slices end to end — only the hard 132 were run, where the
baseline is weakest and the gain is largest. The overall figure will be
flatter.

Soft prefill has never been run. Seven prose bugs survive at τ = 0.5, all of
them questions the probe routed to "answer directly" where the model still
announced a check. The `SOFT_DIRECT` branch exists for exactly that and is
untested.

And whether Qwen3-4B answers as well as qwen2.5:7b once it has the tool
result. Routing is not the whole job, and the system prompt was tuned against
qwen2.5.

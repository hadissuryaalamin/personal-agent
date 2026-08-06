"""Loading the research model, building its prompt, and the gate that proves
this machine can run it at all.

Everything that touches HuggingFace weights goes through here. Before this
file existed, the loader and the prompt builder were written out twice -- once
in the hardware gate and once in the extractor -- and the second copy is
exactly where a prompt silently stops matching the deployed one.

    python -m src.model                     # run the gate, 4-bit Qwen3-4B
    python -m src.model --model Qwen/Qwen3-1.7B --bf16

THE GATE IS TWO CHECKS, AND BOTH ARE NECESSARY

torch.cuda.is_available() answers whether a driver and a CUDA build are
present, not whether the wheel carries kernels for THIS card. On an RTX 5050
(Blackwell, sm_120) a cu124 torch says True and then dies on the first real
operation with "no kernel image is available for execution on the device". So
the first check runs an actual matmul in fp32 and in bf16.

A matmul passing does not prove a 4B model fits. The second check loads the
weights, runs a real forward pass on a real prompt, and times it -- because on
Windows an oversubscribed GPU does not raise OutOfMemory. The driver pages the
excess to system RAM over PCIe and the model gets 53x slower instead. Qwen3-4B
in bf16 measured 48626 ms per pass here and produced perfectly finite numbers.
A gate that only asks "was the output finite?" waves that through, so anything
above 5 s per pass fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from . import config
from .tools import SCHEMA

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Measured on the deployed prompt. Not a limit -- a tripwire: a task landing
# far from this means the system prompt or the schema moved underneath a
# feature file that was extracted before it.
#
# It was 1183 when the agent had three tools. Going to six put it here, and
# the forward pass went 929 ms -> 2098 ms with it. Tool descriptions are not
# free: they are re-read on every single turn, in the agent as much as here.
EXPECTED_TOKENS = 1683
SLOW_MS = 5000


def quant_config(quant: str):
    """Weights only. Activations stay bf16, so the hidden states the probe
    reads are full precision however the weights are stored -- which is why
    quantising is not obviously fatal to this method."""
    import torch
    from transformers import BitsAndBytesConfig

    if quant == "bf16":
        return None
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def load(model_id: str = DEFAULT_MODEL, quant: str = "int4"):
    """Tokeniser and model, on the GPU, in eval mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The agent runs with OFFLINE_MODE on and has a check that proves it can
    # work with no network. A probe service that phones huggingface.co on every
    # start would quietly break that guarantee -- the weights are already
    # cached, so there is nothing to fetch and no reason to ask.
    offline = getattr(config, "OFFLINE_MODE", False)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=offline)
    q = quant_config(quant)
    kw = ({"quantization_config": q, "device_map": "cuda:0"} if q else
          {"dtype": torch.bfloat16, "device_map": "cuda:0"})
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=offline,
                                                 **kw)
    model.eval()
    return tok, model


def build_prompt(tok, question: str, style: str = "deployed") -> str:
    """The prompt the probe reads.

    style="deployed" is the agent's own prompt: system prompt plus the tool
    schema the template inserts. Required when the probe reads the same
    forward pass the model performs to answer, because then it has no choice
    -- that pass exists already and it is 1683 tokens long.

    style="bare" is the question and nothing else. Only legitimate when the
    probe runs as a separate model from the one that answers, which is a real
    architecture here: Qwen3-4B reads, qwen2.5:7b on Ollama replies. Then the
    probe's pass is its own and can be made cheap. 1683 tokens cost 2098 ms;
    the bare form is a fortieth of that.

    The reason it is even arguable: these labels are about ACCESS, not model
    competence. Whether a question needs the user's calendar is a property of
    the question. The paper could not do this -- their labels are "does this
    model get it wrong without a tool", which is meaningless read off another
    model.
    """
    if style == "bare":
        return tok.apply_chat_template([{"role": "user", "content": question}],
                                       add_generation_prompt=True, tokenize=False)
    msgs = [{"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tools=SCHEMA,
                                   add_generation_prompt=True, tokenize=False)


def prompt_hash() -> str:
    """Identifies the prompt a feature file was extracted under. If the system
    prompt or a tool description changes, this changes, and the mismatch is
    visible instead of surfacing as a puzzling result months later."""
    blob = config.SYSTEM_PROMPT + json.dumps(SCHEMA, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def geometry(model) -> tuple[int, int, int]:
    """(layers, hidden size, probe width).

    The paper concatenates the last-token state from every layer plus the
    embedding output, hence layers + 1 -- which is what hidden_states returns.
    """
    n = model.config.num_hidden_layers
    d = model.config.hidden_size
    return n, d, (n + 1) * d


def last_token_state(out):
    """The 94,720-float feature vector, sliced free of everything else.

    Called inside the extraction loop for a reason: the full hidden_states for
    one 1183-token prompt is 224 MB, and the 4-bit configuration leaves 2.38 GB
    spare. Holding a handful of them is the one way to run this card out of
    memory.
    """
    import torch
    return torch.cat([h[0, -1, :] for h in out.hidden_states]).float()


class PrefixCache:
    """The constant head of the prompt, computed once and kept.

    1668 of the 1683 prompt tokens are the system prompt and the tool schema,
    byte-identical on every turn. Recomputing them is what made the HF path
    look unusable next to Ollama: 2086 ms per turn against 101 ms with this.
    Ollama was never doing less work -- it was doing the same work once.

    The hidden state is not bit-identical to the uncached path (bf16 kernels
    batch differently at different sequence lengths) but agrees to a cosine of
    0.9999, and the probe's decision does not move.

    Usage per turn:

        cache, ids = prefix.fork()                 # a private copy
        out = model(tail_ids, past_key_values=cache, output_hidden_states=True)
        ...read the probe...
        cache.crop(prefix.length)                  # BEFORE generating

    That crop is not optional. The probe pass leaves the cache holding
    head+tail; handing generate() a sequence it has already cached in full
    leaves it no new tokens to process, and it emits garbage rather than
    failing.
    """

    def __init__(self, model, tok, style: str = "deployed"):
        import torch

        full = build_prompt(tok, "PLACEHOLDER", style)
        self.head, self.tail_fmt = full.split("PLACEHOLDER")
        self.tok = tok
        self.head_ids = tok(self.head, return_tensors="pt").input_ids.to("cuda")
        self.length = self.head_ids.shape[1]
        with torch.inference_mode():
            self._base = model(input_ids=self.head_ids, use_cache=True).past_key_values

    def fork(self):
        """A private copy of the cache. The model mutates whatever it is given,
        so the shared one must never be passed in directly."""
        import copy
        return copy.deepcopy(self._base)

    def tail_ids(self, question: str):
        return self.tok(question + self.tail_fmt, return_tensors="pt",
                        add_special_tokens=False).input_ids.to("cuda")

    def full_ids(self, question: str):
        import torch
        return torch.cat([self.head_ids, self.tail_ids(question)], dim=1)


def check_gpu() -> bool:
    import torch

    print(f"  torch          {torch.__version__}  (CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("  FAIL           no CUDA device visible to torch")
        return False

    major, minor = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    gb = 1024 ** 3
    print(f"  device         {torch.cuda.get_device_name(0)}  sm_{major}{minor}")
    print(f"  VRAM           {free / gb:.2f} GB free of {total / gb:.2f} GB")

    arches = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in arches:
        print(f"  NOTE           sm_{major}{minor} is not in this wheel "
              f"({', '.join(arches)});")
        print("                 if the matmul below fails, that is why --")
        print("                 reinstall torch from the cu128 index")

    ok = True
    for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        try:
            a = torch.randn(1024, 1024, device="cuda", dtype=dtype)
            c = (a @ a).float().sum().item()
            assert c == c, "produced NaN"
            print(f"  PASS           {name} matmul on GPU")
        except Exception as e:
            print(f"  FAIL           {name} matmul: {type(e).__name__}: {e}")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default="int4", choices=("int4", "int8", "bf16"))
    ap.add_argument("--bf16", action="store_true", help="shorthand for --quant bf16")
    args = ap.parse_args()
    quant = "bf16" if args.bf16 else args.quant

    import torch

    if not check_gpu():
        print("\n  gate FAILED at the hardware check")
        return 1

    gb = 1024 ** 3
    print()
    print(f"  model          {args.model}")
    print(f"  quantisation   {quant}")

    t0 = time.perf_counter()
    try:
        tok, model = load(args.model, quant)
    except torch.OutOfMemoryError as e:
        print(f"  FAIL           out of memory loading weights: {e}")
        return 1
    print(f"  loaded in      {time.perf_counter() - t0:.1f}s")

    n_layers, hidden, width = geometry(model)
    print(f"  geometry       {n_layers + 1} x {hidden} = {width:,} probe features")
    print(f"  prompt hash    {prompt_hash()}")

    text = build_prompt(tok, "Can I go to the gym at four?")
    inputs = tok(text, return_tensors="pt").to("cuda")
    n_tok = inputs.input_ids.shape[1]
    print(f"  prompt tokens  {n_tok}  (expected about {EXPECTED_TOKENS})")
    print(f"  last token     {tok.decode(inputs.input_ids[0, -1])!r}")

    # Six passes, not one. The first includes kernel autotuning and runs about
    # 4x the steady-state cost; reporting it would overstate extraction by
    # minutes across the task set.
    times = []
    for _ in range(6):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True)
        except torch.OutOfMemoryError as e:
            print(f"  FAIL           out of memory on the forward pass: {e}")
            return 1
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    warm = sorted(times[1:])[len(times[1:]) // 2]
    vec = last_token_state(out)
    free = torch.cuda.mem_get_info(0)[0] / gb

    print(f"  forward pass   {warm:.0f} ms warm ({times[0]:.0f} ms cold)")
    print(f"  143 tasks      ~{warm * 143 / 1000:.0f} s")
    print(f"  feature vector {vec.shape[0]:,} floats, finite "
          f"{bool(torch.isfinite(vec).all())}")
    print(f"  headroom       {free:.2f} GB")

    ok = vec.shape[0] == width and bool(torch.isfinite(vec).all())
    if warm > SLOW_MS:
        print(f"\n  FAIL           {warm:.0f} ms per pass -- this is VRAM spilling")
        print("                 to system RAM, not a working configuration")
        ok = False
    elif free < 0.5:
        print(f"\n  WARN           only {free:.2f} GB spare; a longer prompt or any")
        print("                 batching pushes this into the same spill")

    print("\n  gate PASSED" if ok else "\n  gate FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

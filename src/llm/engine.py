"""The single model load, and the one prefill per turn.

Invariant #2: one model, through HF transformers, in one process. Hidden states
are the point of the project, and transformers is the only route that exposes
them per layer -- which is why this file does not import Ollama, llama.cpp, or
anything that would hide them.

The shape of a turn (PLAN.md section 4):

    prefill(prefix)          one forward pass, output_hidden_states=True
      -> h_L for the probe, and a KV cache
    continue_from(cache, suffix)   the gate question, or the tool schemas
      -> the cache is cropped back afterwards, so the prefix is encoded once

Run ``python -m src.llm.engine --info`` to load the weights and print what the
model actually is. CLAUDE.md: assert shapes at load rather than trusting the
numbers written in PLAN.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config

#: Rough bf16 footprint of Qwen3-4B plus working space, in GiB. Used only to
#: decide whether to quantise; the real check is whether the load succeeds.
_BF16_VRAM_GB = 9.0


@dataclass
class Prefill:
    """One encoded prompt: its cache, and the hidden states at its last token."""

    n_tokens: int
    #: (n_layers + 1, hidden_size) float32 on CPU -- layer 0 is the embedding
    #: output, so hidden[L] is the output of transformer block L.
    hidden: Any
    cache: Any = field(repr=False, default=None)
    ms: int = 0

    @property
    def n_layers(self) -> int:
        return int(self.hidden.shape[0]) - 1

    @property
    def hidden_size(self) -> int:
        return int(self.hidden.shape[1])

    def layer(self, index: int):
        """``hidden_states[L][0, -1]`` from PLAN.md section 4."""
        return self.hidden[index]


class Engine:
    """Wraps the tokenizer and model. Loads once, on first use."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        quantise: str | None = None,
        device: str | None = None,
    ) -> None:
        cfg = config.load()
        self.model_dir = Path(model_dir or cfg.model_dir)
        self.quantise = quantise or cfg.quantise
        self.device = device
        self.model = None
        self.tokenizer = None
        self.info: dict[str, Any] = {}

    # -- loading ----------------------------------------------------------

    def _resolve_quantisation(self, torch) -> str:
        if self.quantise != "auto":
            return self.quantise
        if not torch.cuda.is_available():
            return "bf16"
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024**3
        return "bf16" if free_gb >= _BF16_VRAM_GB else "4bit"

    def load(self) -> "Engine":
        if self.model is not None:
            return self

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"No weights at {self.model_dir}. Run: python scripts\\fetch_models.py"
            )

        started = time.perf_counter()
        # Invariant #8: local files only. Nothing reaches the network here.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )

        mode = self._resolve_quantisation(torch)
        kwargs: dict[str, Any] = {"local_files_only": True}
        if mode == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["dtype"] = torch.bfloat16
            kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kwargs)
        self.model.eval()

        model_config = self.model.config
        self.info = {
            "model_dir": str(self.model_dir),
            "quantisation": mode,
            "layers": int(model_config.num_hidden_layers),
            "hidden_size": int(model_config.hidden_size),
            "vocab": int(model_config.vocab_size),
            "device": str(next(self.model.parameters()).device),
            "load_ms": int((time.perf_counter() - started) * 1000),
        }
        return self

    # -- one turn ---------------------------------------------------------

    def prefix_text(self, messages: list[dict[str, str]]) -> str:
        """Chat-format the messages, stopping before the assistant's turn.

        The suffixes appended later supply their own assistant header, which is
        what lets one cache serve both the gate and the tool pass.
        """
        self.load()
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def prefill(self, text: str) -> Prefill:
        """Encode the prompt once, keeping the cache and the hidden states."""
        import torch

        self.load()
        started = time.perf_counter()
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        # PLAN.md section 4: hidden_states[L][0, -1] -- the last prompt token.
        stacked = torch.stack([layer[0, -1] for layer in outputs.hidden_states])
        hidden = stacked.to(dtype=torch.float32, device="cpu").numpy()

        # CLAUDE.md: assert the shape, do not trust the doc. These check the
        # model against itself, so they keep holding if the weights change.
        assert hidden.shape[0] == self.info["layers"] + 1, (
            f"expected {self.info['layers']} + 1 hidden states, got {hidden.shape[0]}"
        )
        assert hidden.shape[1] == self.info["hidden_size"], (
            f"hidden size {hidden.shape[1]} does not match config "
            f"{self.info['hidden_size']}"
        )

        return Prefill(
            n_tokens=int(ids.shape[1]),
            hidden=hidden,
            cache=outputs.past_key_values,
            ms=int((time.perf_counter() - started) * 1000),
        )

    def next_token_logits(self, prefill: Prefill, suffix: str):
        """Logits for the token that would come after ``suffix``.

        The prompted gate reads a single distribution rather than generating,
        which is both faster and gives it a comparable *score* -- the probe is
        measured against this baseline at M3, so it needs more than a label.
        """
        import torch

        self.load()
        device = self.model.device
        ids = self.tokenizer(
            suffix, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        try:
            with torch.inference_mode():
                past = prefill.cache.get_seq_length()
                outputs = self.model(
                    input_ids=ids,
                    past_key_values=prefill.cache,
                    use_cache=True,
                    attention_mask=torch.ones(
                        (1, past + ids.shape[1]), dtype=torch.long, device=device
                    ),
                    return_dict=True,
                )
                return outputs.logits[0, -1].float().cpu()
        finally:
            prefill.cache.crop(prefill.n_tokens)

    def token_ids_for(self, *words: str) -> list[int]:
        """Every id that could start one of ``words``, with and without a space."""
        self.load()
        found: set[int] = set()
        for word in words:
            for variant in (word, f" {word}", word.lower(), f" {word.lower()}"):
                ids = self.tokenizer(variant, add_special_tokens=False).input_ids
                if ids:
                    found.add(int(ids[0]))
        return sorted(found)

    def continue_from(
        self,
        prefill: Prefill,
        suffix: str,
        max_new_tokens: int = 96,
        stop_on: str | None = None,
        stop_when: Any = None,
    ) -> tuple[str, int]:
        """Greedy continuation from a prefilled cache, then restore the cache.

        Returns the generated text and how long it took. The cache is cropped
        back to the prefix length so the same prefill can drive a second pass --
        the gate first, then the tool schemas.

        ``stop_when`` is a predicate over the text so far. Decode is the whole
        cost of a turn here, so stopping the moment the answer is complete --
        rather than at a token limit -- is worth more than any other single
        optimisation. See PLAN.md section 5.
        """
        import torch

        self.load()
        started = time.perf_counter()
        device = self.model.device

        ids = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False)
        current = ids.input_ids.to(device)

        eos = {self.tokenizer.eos_token_id}
        for token in ("<|im_end|>", "<|endoftext|>"):
            found = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(found, int) and found >= 0:
                eos.add(found)

        produced: list[int] = []
        text = ""
        try:
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    past = prefill.cache.get_seq_length()
                    outputs = self.model(
                        input_ids=current,
                        past_key_values=prefill.cache,
                        use_cache=True,
                        attention_mask=torch.ones(
                            (1, past + current.shape[1]), dtype=torch.long, device=device
                        ),
                        return_dict=True,
                    )
                    nxt = int(outputs.logits[0, -1].argmax())
                    if nxt in eos:
                        break
                    produced.append(nxt)
                    text = self.tokenizer.decode(produced, skip_special_tokens=True)
                    if stop_on and stop_on in text:
                        break
                    if stop_when is not None and stop_when(text):
                        break
                    current = torch.tensor([[nxt]], dtype=torch.long, device=device)
        finally:
            # Put the cache back the way the prefill left it.
            prefill.cache.crop(prefill.n_tokens)

        return text.strip(), int((time.perf_counter() - started) * 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="load the model and report what it is")
    parser.add_argument("--info", action="store_true", help="load, assert shapes, print")
    parser.add_argument("--quantise", default=None, choices=["auto", "bf16", "4bit"])
    parser.add_argument("--prompt", default="What is due next Friday?")
    args = parser.parse_args(argv)

    engine = Engine(quantise=args.quantise).load()
    print("loaded:")
    for key, value in engine.info.items():
        print(f"  {key:<13} {value}")

    prefill = engine.prefill(
        engine.prefix_text([{"role": "user", "content": args.prompt}])
    )
    print("\nprefill:")
    print(f"  tokens        {prefill.n_tokens}")
    print(f"  hidden        {prefill.hidden.shape}  (layers+1, hidden_size)")
    print(f"  ms            {prefill.ms}")

    if args.info:
        reply, ms = engine.continue_from(prefill, "<|im_start|>assistant\n", 32)
        print(f"\nsample reply ({ms} ms): {reply!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

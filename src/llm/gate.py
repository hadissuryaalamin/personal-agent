"""Deciding whether a turn wants a tool at all.

This is the question the whole project is about. At M1 there is one
implementation -- ask the model and read the answer token -- and it exists to be
the *baseline*: PLAN.md section 4 says the probe has to beat it on accuracy and
latency, or D1 was the wrong call. Numbers go in docs/eval.md, produced by a
script, never typed by hand.

Both gates return the same shape, so the branch in the agent does not care
which one is running. Which one runs is ``config.gate``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from src.llm import prompts  # noqa: F401  (used by ProbeGate's staleness check)

#: PLAN.md section 4. Asymmetric on purpose: a false *call* writes wrong data, a
#: false *no-call* only produces an unhelpful reply. The band between them asks.
#: These are the M1 defaults; M3 tunes them against the held-out set.
TAU_LO = 0.35
TAU_HI = 0.65


@dataclass(frozen=True)
class Decision:
    #: "tool", "chat", or "unsure" -- the three-way outcome from one score.
    label: str
    #: Probability that the turn wants a tool, or None if the gate has no score.
    score: float | None
    source: str
    ms: int

    @property
    def is_tool(self) -> bool:
        return self.label == "tool"


class Gate(Protocol):
    def decide(self, engine, prefill) -> Decision: ...


def classify(score: float, tau_lo: float = TAU_LO, tau_hi: float = TAU_HI) -> str:
    """One score, three branches."""
    if score >= tau_hi:
        return "tool"
    if score < tau_lo:
        return "chat"
    return "unsure"


class PromptedGate:
    """The baseline: append the question, read the TOOL/CHAT distribution.

    Reading the distribution rather than the sampled token costs the same
    forward pass but yields a score, which is what makes the M3 comparison a
    like-for-like ROC rather than two accuracy numbers.
    """

    source = "prompted"

    def __init__(self, tau_lo: float = TAU_LO, tau_hi: float = TAU_HI) -> None:
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self._tool_ids: list[int] | None = None
        self._chat_ids: list[int] | None = None

    def _candidates(self, engine) -> tuple[list[int], list[int]]:
        if self._tool_ids is None:
            self._tool_ids = engine.token_ids_for("TOOL")
            self._chat_ids = engine.token_ids_for("CHAT")
        return self._tool_ids, self._chat_ids or []

    def suffix(self) -> str:
        return (
            f"<|im_start|>user\n{prompts.GATE_QUESTION}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def decide(self, engine, prefill) -> Decision:
        import time

        started = time.perf_counter()
        logits = engine.next_token_logits(prefill, self.suffix())
        tool_ids, chat_ids = self._candidates(engine)

        # Softmax over just the two candidate sets: the question is binary, and
        # normalising over the whole vocabulary only adds noise from tokens
        # that are not answers to it.
        tool_logit = max(float(logits[i]) for i in tool_ids)
        chat_logit = max(float(logits[i]) for i in chat_ids)
        score = 1.0 / (1.0 + math.exp(-(tool_logit - chat_logit)))

        return Decision(
            label=classify(score, self.tau_lo, self.tau_hi),
            score=score,
            source=self.source,
            ms=int((time.perf_counter() - started) * 1000),
        )


class ProbeGate:
    """The point of the project: read the hidden state the prefill already made.

    The prompted gate needs a forward pass over its question and a look at the
    answer distribution. This needs a matrix multiply against a vector that
    ``Engine.prefill`` produced anyway, so the gate is very nearly free -- which
    is half of what PLAN.md section 4 asks the probe to beat the baseline on.

    Thresholds come from the artefact, where `scripts/train_probe.py` recorded
    the values it tuned on a validation split. They can be overridden, but the
    defaults are the measured ones, not guesses.
    """

    source = "probe"

    def __init__(
        self,
        artifact_path=None,
        tau_lo: float | None = None,
        tau_hi: float | None = None,
    ) -> None:
        from src.config import ROOT

        self.artifact_path = artifact_path or (ROOT / "data" / "probe.joblib")
        self._override = (tau_lo, tau_hi)
        self._loaded = False
        self.pipeline = None
        self.layer = None
        self.tau_lo = tau_lo if tau_lo is not None else TAU_LO
        self.tau_hi = tau_hi if tau_hi is not None else TAU_HI

    def load(self) -> "ProbeGate":
        if self._loaded:
            return self

        import joblib

        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"No probe at {self.artifact_path}. Train one:\n"
                "    python scripts\\sweep_layers.py\n"
                "    python scripts\\train_probe.py\n"
                "or run with AGENT_GATE=prompted."
            )
        payload = joblib.load(self.artifact_path)
        self.pipeline = payload["pipeline"]
        self.layer = int(payload["layer"])
        self.hidden_size = int(payload["hidden_size"])
        self.n_layers = int(payload["n_layers"])
        self.trained_at = payload.get("trained_at")

        # The probe reads activations produced by a prefix that starts with the
        # system prompt. Change the prompt and those activations move, but the
        # probe carries on returning confident scores for a prompt it has never
        # seen. Fail loudly instead.
        # A missing fingerprint means an artefact from before this check
        # existed, which is exactly the case that cannot be verified -- so it
        # is refused too. The artefact is gitignored and regenerated locally in
        # a couple of minutes; there is nothing to be gained by trusting it.
        trained_on = payload.get("prompt_fingerprint")
        current = prompts.system_fingerprint()
        if trained_on != current:
            raise ValueError(
                f"This probe was trained against system prompt "
                f"{trained_on or '(unrecorded)'}, but the prompt is now {current}. "
                "Its scores would be meaningless. Retrain:\n"
                "    python scripts\\sweep_layers.py\n"
                "    python scripts\\train_probe.py"
            )
        if self._override[0] is None:
            self.tau_lo = float(payload["tau_lo"])
        if self._override[1] is None:
            self.tau_hi = float(payload["tau_hi"])
        self._loaded = True
        return self

    def decide(self, engine, prefill) -> Decision:
        import time

        self.load()
        started = time.perf_counter()

        # Assert the probe and the model agree about shape. A probe trained on
        # different weights would otherwise produce confident nonsense.
        if prefill.hidden.shape[0] != self.n_layers + 1:
            raise ValueError(
                f"probe was trained on a {self.n_layers}-layer model, "
                f"got {prefill.hidden.shape[0] - 1}"
            )
        if prefill.hidden.shape[1] != self.hidden_size:
            raise ValueError(
                f"probe expects hidden size {self.hidden_size}, "
                f"got {prefill.hidden.shape[1]}"
            )

        vector = prefill.hidden[self.layer].reshape(1, -1).astype("float32")
        score = float(self.pipeline.predict_proba(vector)[0, 1])

        return Decision(
            label=classify(score, self.tau_lo, self.tau_hi),
            score=score,
            source=self.source,
            ms=int((time.perf_counter() - started) * 1000),
        )


def build(cfg) -> Gate:
    if cfg.gate == "prompted":
        return PromptedGate()
    if cfg.gate == "probe":
        gate = ProbeGate()
        if cfg.probe_layer is not None:
            gate.load()
            gate.layer = cfg.probe_layer
        return gate
    raise ValueError(f"Unknown gate {cfg.gate!r} — use 'prompted' or 'probe'.")

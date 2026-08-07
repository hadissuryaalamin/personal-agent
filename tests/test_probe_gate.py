"""The probe gate (M3), against a synthetic artefact.

No weights and no GPU: a probe is a scikit-learn pipeline over a vector, so
everything except producing that vector is testable on the CPU. What the
hardware tests cover is that the vector coming out of the real model is the one
the probe was trained on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.llm import gate as gate_module
from src.llm.engine import Prefill

LAYERS, HIDDEN = 36, 2560
LAYER = 18


def _pipeline(seed=0):
    """A probe that fires on vectors pointing along a known direction."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    direction = rng.randn(HIDDEN)
    y = np.array([0, 1] * 100)
    x = rng.randn(200, HIDDEN) * 0.5 + y[:, None] * direction
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x, y)
    return model, direction


def _dump(path, model, **overrides):
    import joblib

    from src.llm import prompts

    payload = {
        "pipeline": model, "layer": LAYER, "C": 0.01,
        "tau_lo": 0.4, "tau_hi": 0.7,
        "hidden_size": HIDDEN, "n_layers": LAYERS,
        "dataset_fingerprint": "test",
        "prompt_fingerprint": prompts.system_fingerprint(),
        "trained_at": "2026-08-07T00:00:00+00:00",
    }
    payload.update(overrides)
    joblib.dump(payload, path)
    return path


@pytest.fixture
def artifact(tmp_path):
    model, direction = _pipeline()
    return _dump(tmp_path / "probe.joblib", model), direction


def prefill_with(vector, layer=LAYER):
    hidden = np.zeros((LAYERS + 1, HIDDEN), dtype=np.float32)
    hidden[layer] = vector
    return Prefill(n_tokens=11, hidden=hidden, ms=5)


# -- deciding --------------------------------------------------------------


def test_a_tool_shaped_vector_scores_high(artifact):
    path, direction = artifact
    gate = gate_module.ProbeGate(path)
    decision = gate.decide(None, prefill_with(direction))

    assert decision.source == "probe"
    assert decision.score > 0.7
    assert decision.label == "tool"


def test_a_chat_shaped_vector_scores_low(artifact):
    path, direction = artifact
    gate = gate_module.ProbeGate(path)
    decision = gate.decide(None, prefill_with(-direction))

    assert decision.score < 0.3
    assert decision.label == "chat"


def test_the_gate_reads_the_layer_it_was_trained_on(artifact):
    """Put the signal on the wrong layer and the probe must not see it."""
    path, direction = artifact
    gate = gate_module.ProbeGate(path)

    right = gate.decide(None, prefill_with(direction, layer=LAYER)).score
    wrong = gate.decide(None, prefill_with(direction, layer=3)).score
    assert right > wrong


def test_the_probe_needs_no_engine(artifact):
    """It reads a vector the prefill already produced -- that is the point."""
    path, direction = artifact
    assert gate_module.ProbeGate(path).decide(None, prefill_with(direction)) is not None


def test_the_probe_is_fast(artifact):
    import time

    path, direction = artifact
    gate = gate_module.ProbeGate(path).load()
    prefill = prefill_with(direction)

    gate.decide(None, prefill)  # warm
    started = time.perf_counter()
    for _ in range(20):
        gate.decide(None, prefill)
    per_call_ms = (time.perf_counter() - started) * 1000 / 20
    assert per_call_ms < 20, f"{per_call_ms:.1f} ms is not a matrix multiply"


# -- thresholds ------------------------------------------------------------


def test_thresholds_come_from_the_artifact(artifact):
    path, _ = artifact
    gate = gate_module.ProbeGate(path).load()
    assert (gate.tau_lo, gate.tau_hi) == (0.4, 0.7)


def test_thresholds_can_be_overridden(artifact):
    path, _ = artifact
    gate = gate_module.ProbeGate(path, tau_lo=0.1, tau_hi=0.9).load()
    assert (gate.tau_lo, gate.tau_hi) == (0.1, 0.9)


def test_a_score_in_the_band_asks(artifact):
    path, _ = artifact
    gate = gate_module.ProbeGate(path, tau_lo=0.0, tau_hi=1.0).load()
    assert gate.decide(None, prefill_with(np.zeros(HIDDEN))).label == "unsure"


def test_an_empty_band_never_asks(artifact):
    """τ_lo == τ_hi is what training actually produced: two live arms, not three."""
    path, direction = artifact
    gate = gate_module.ProbeGate(path, tau_lo=0.5, tau_hi=0.5).load()

    for vector in (direction, -direction, np.zeros(HIDDEN)):
        assert gate.decide(None, prefill_with(vector)).label in ("tool", "chat")


# -- refusing to run against the wrong model -------------------------------


def test_a_different_layer_count_is_refused(artifact):
    path, direction = artifact
    gate = gate_module.ProbeGate(path)
    wrong = Prefill(n_tokens=5, hidden=np.zeros((29, HIDDEN), dtype=np.float32), ms=1)

    with pytest.raises(ValueError, match="36-layer"):
        gate.decide(None, wrong)


def test_a_different_hidden_size_is_refused(artifact):
    path, _ = artifact
    gate = gate_module.ProbeGate(path)
    wrong = Prefill(n_tokens=5, hidden=np.zeros((LAYERS + 1, 999), dtype=np.float32), ms=1)

    with pytest.raises(ValueError, match="hidden size"):
        gate.decide(None, wrong)


def test_a_missing_artifact_says_how_to_make_one(tmp_path):
    gate = gate_module.ProbeGate(tmp_path / "nope.joblib")
    with pytest.raises(FileNotFoundError, match="train_probe"):
        gate.load()


def test_a_probe_trained_on_a_different_prompt_is_refused(tmp_path):
    """Edit the system prompt and every activation downstream of it moves."""
    model, _ = _pipeline()
    path = _dump(tmp_path / "stale.joblib", model, prompt_fingerprint="deadbeefdeadbeef")

    with pytest.raises(ValueError, match="system prompt"):
        gate_module.ProbeGate(path).load()


def test_the_refusal_says_how_to_fix_it(tmp_path):
    model, _ = _pipeline()
    path = _dump(tmp_path / "stale.joblib", model, prompt_fingerprint="0000000000000000")

    with pytest.raises(ValueError, match="sweep_layers"):
        gate_module.ProbeGate(path).load()


def test_an_artifact_with_no_fingerprint_is_refused(tmp_path):
    """Unverifiable is not the same as fine — and it is the dangerous case."""
    model, _ = _pipeline()
    path = _dump(tmp_path / "old.joblib", model, prompt_fingerprint=None)

    with pytest.raises(ValueError, match="unrecorded"):
        gate_module.ProbeGate(path).load()


def test_the_fingerprint_tracks_the_prompt():
    from src.llm import prompts

    original = prompts.SYSTEM
    first = prompts.system_fingerprint()
    try:
        prompts.SYSTEM = original + "\n- and one more rule\n"
        assert prompts.system_fingerprint() != first
    finally:
        prompts.SYSTEM = original
    assert prompts.system_fingerprint() == first


# -- wiring ----------------------------------------------------------------


def test_build_returns_the_configured_gate(cfg):
    from dataclasses import replace

    assert isinstance(gate_module.build(cfg), gate_module.PromptedGate)
    probe = gate_module.build(replace(cfg, gate="probe"))
    assert isinstance(probe, gate_module.ProbeGate)


def test_build_still_rejects_nonsense(cfg):
    from dataclasses import replace

    with pytest.raises(ValueError):
        gate_module.build(replace(cfg, gate="vibes"))


def test_the_probe_gate_is_not_loaded_until_it_is_used(cfg):
    """Constructing a gate must not need the artefact on disk."""
    from dataclasses import replace

    gate = gate_module.build(replace(cfg, gate="probe"))
    assert gate.pipeline is None


def test_both_gates_return_the_same_shape(artifact):
    path, direction = artifact
    decision = gate_module.ProbeGate(path).decide(None, prefill_with(direction))

    assert set(decision.__dataclass_fields__) == {"label", "score", "source", "ms"}
    assert decision.label in ("tool", "chat", "unsure")
    assert 0.0 <= decision.score <= 1.0

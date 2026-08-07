"""The layer sweep's arithmetic, on synthetic data.

The sweep itself needs weights, so it lives in test_hardware.py. What is
testable without a GPU is the part that decides things: how a layer is scored,
and which layer wins when two are close.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import sweep_layers
from src import dataset


def row(layer, accuracy, false_call=0.0, false_skip=0.0, c=1.0):
    return {
        "layer": layer, "accuracy": accuracy, "false_call": false_call,
        "false_skip": false_skip, "C": c,
    }


# -- choosing a layer ------------------------------------------------------


def test_best_accuracy_wins():
    rows = [row(0, 0.80), row(12, 0.92), row(30, 0.85)]
    assert sweep_layers.choose(rows)["layer"] == 12


def test_ties_go_to_the_safer_layer():
    """A false call writes wrong data; a false no-call is only unhelpful."""
    rows = [row(10, 0.90, false_call=0.20), row(20, 0.90, false_call=0.05)]
    assert sweep_layers.choose(rows)["layer"] == 20


def test_a_full_tie_goes_to_the_earlier_layer():
    """Earlier means the probe could, in principle, read it sooner."""
    rows = [row(30, 0.9, 0.1), row(8, 0.9, 0.1)]
    assert sweep_layers.choose(rows)["layer"] == 8


def test_choose_handles_a_single_layer():
    assert sweep_layers.choose([row(5, 0.7)])["layer"] == 5


# -- scoring a layer -------------------------------------------------------


def make_separable(n=200, dim=32, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    y = np.array([0, 1] * (n // 2))
    direction = rng.randn(dim)
    x = rng.randn(n, dim) * noise + y[:, None] * direction
    return x.astype(np.float32), y


def test_a_separable_layer_scores_high():
    x, y = make_separable()
    result = sweep_layers.evaluate_layer(x[:150], y[:150], x[150:], y[150:])
    assert result["accuracy"] > 0.95
    assert result["false_call"] < 0.1


def test_pure_noise_scores_around_chance():
    rng = np.random.RandomState(1)
    x = rng.randn(200, 32).astype(np.float32)
    y = np.array([0, 1] * 100)
    result = sweep_layers.evaluate_layer(x[:150], y[:150], x[150:], y[150:])
    assert 0.3 < result["accuracy"] < 0.7


def test_the_regularisation_grid_is_searched():
    x, y = make_separable(noise=3.0)
    result = sweep_layers.evaluate_layer(x[:150], y[:150], x[150:], y[150:])
    assert result["C"] in sweep_layers.C_GRID


def test_false_call_and_false_skip_are_the_right_way_round():
    """Predict everything as a tool call: no skips, every chat is a false call."""
    x = np.vstack([np.ones((50, 4)), np.ones((50, 4))]).astype(np.float32)
    y = np.array([1] * 50 + [0] * 50)
    result = sweep_layers.evaluate_layer(x, y, x, y)
    assert result["false_call"] + result["false_skip"] > 0


# -- the cache key ---------------------------------------------------------


def test_fingerprint_changes_when_the_dataset_changes():
    a = [dataset.Example("hello", 0), dataset.Example("what is on", 1)]
    b = [dataset.Example("hello", 0), dataset.Example("what is on", 0)]
    assert sweep_layers.fingerprint(a) != sweep_layers.fingerprint(b)


def test_fingerprint_is_stable_for_the_same_data():
    a = [dataset.Example("hello", 0, history=(("hi", "hello"),))]
    b = [dataset.Example("hello", 0, history=(("hi", "hello"),))]
    assert sweep_layers.fingerprint(a) == sweep_layers.fingerprint(b)


def test_fingerprint_notices_history():
    plain = [dataset.Example("make it sixty", 1)]
    with_history = [dataset.Example("make it sixty", 1, history=(("a", "b"),))]
    assert sweep_layers.fingerprint(plain) != sweep_layers.fingerprint(with_history)


@pytest.mark.parametrize("layer", [0, 18, 36])
def test_sweep_indexes_every_layer(layer):
    """A 37-row stack is layers 0..36 — the embedding output plus 36 blocks."""
    stack = np.zeros((10, 37, 8), dtype=np.float16)
    assert stack[:, layer].shape == (10, 8)


# -- guarding the recorded result ------------------------------------------
#
# data/layer_sweep.json is gitignored, so these skip on a fresh clone. They
# exist so that re-running the sweep after changing the dataset, the prompt or
# the model tells you if the result stopped holding up.


@pytest.fixture
def recorded():
    import json

    if not sweep_layers.RESULTS.exists():
        pytest.skip("no sweep results — run scripts/sweep_layers.py")
    return json.loads(sweep_layers.RESULTS.read_text(encoding="utf-8"))


def test_the_probe_beats_a_bag_of_words(recorded):
    """If TF-IDF matches it, the sweep says nothing about hidden states."""
    control = recorded.get("control")
    if not control:
        pytest.skip("no lexical control in the recorded results")
    assert recorded["chosen"]["accuracy"] > control["accuracy"], (
        f"probe {recorded['chosen']['accuracy']:.1%} vs "
        f"bag of words {control['accuracy']:.1%} — the dataset is separable by "
        "vocabulary alone, so this result is about the words, not the model"
    )


def test_the_chosen_layer_is_a_real_layer(recorded):
    layers = [r["layer"] for r in recorded["layers"]]
    assert recorded["chosen"]["layer"] in layers
    assert min(layers) == 0, "layer 0 is the embedding output and belongs in the sweep"


def test_the_held_out_set_is_big_enough_to_mean_something(recorded):
    assert recorded["dataset"]["test"] >= 100


def test_the_held_out_set_is_not_all_one_class(recorded):
    test, tool = recorded["dataset"]["test"], recorded["dataset"]["test_tool"]
    assert 0.35 <= tool / test <= 0.65


def test_no_logged_turn_reached_the_held_out_set(recorded):
    """Real turns are folded into train only — PLAN.md section 4."""
    assert recorded["dataset"]["test"] + recorded["dataset"]["train"] >= 500

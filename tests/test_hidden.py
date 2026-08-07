"""Hidden-state capture (M2).

The array is the probe's training data, so losing one is a real cost -- but
losing the user's answer because the disk filled up is a worse one. These tests
pin that priority down.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import hidden


def test_save_and_load_round_trip(tmp_path):
    array = np.random.RandomState(0).randn(37, 2560).astype(np.float32)
    path = hidden.save_turn(array, turn_id=7, session_id="abc123", root=tmp_path)

    assert path is not None and path.exists()
    loaded = hidden.load_turn(path)
    assert loaded.shape == (37, 2560)
    # float16 on disk: close, not identical, and that is the intended trade.
    assert np.allclose(loaded.astype(np.float32), array, atol=1e-2)


def test_every_layer_is_kept(tmp_path):
    """L is chosen by the sweep and may be re-chosen; keep all of them."""
    array = np.zeros((37, 2560), dtype=np.float32)
    path = hidden.save_turn(array, 1, "s", root=tmp_path)
    assert hidden.load_turn(path).shape[0] == 37


def test_files_are_grouped_by_session(tmp_path):
    a = hidden.save_turn(np.zeros((2, 2)), 1, "session-one", root=tmp_path)
    b = hidden.save_turn(np.zeros((2, 2)), 2, "session-two", root=tmp_path)
    assert a.parent != b.parent
    assert a.parent.name == "session-one"


def test_turn_ids_sort_lexicographically(tmp_path):
    first = hidden.turn_path(2, "s", tmp_path).name
    second = hidden.turn_path(10, "s", tmp_path).name
    assert first < second, "zero-padding keeps ls order matching turn order"


def test_a_hostile_session_id_cannot_escape_the_directory(tmp_path):
    path = hidden.turn_path(1, "../../etc/passwd", tmp_path)
    assert tmp_path in path.parents


def test_no_hidden_state_is_not_an_error(tmp_path):
    assert hidden.save_turn(None, 1, "s", root=tmp_path) is None


def test_a_write_failure_never_costs_the_user_their_answer(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", explode)
    assert hidden.save_turn(np.zeros((2, 2)), 1, "s", root=tmp_path) is None


def test_stored_paths_are_relative_to_the_repo(tmp_path):
    from src.config import ROOT

    inside = ROOT / "data" / "hidden" / "s" / "turn_00000001.npz"
    assert hidden.relative_to_root(inside) == str(
        inside.relative_to(ROOT)
    )
    assert not hidden.relative_to_root(inside).startswith(str(ROOT))


def test_a_path_outside_the_repo_is_kept_whole(tmp_path):
    outside = tmp_path / "elsewhere.npz"
    assert hidden.relative_to_root(outside) == str(outside)


def test_relative_to_root_passes_none_through():
    assert hidden.relative_to_root(None) is None


# -- the REPL records the path ---------------------------------------------


def test_a_spoken_turn_logs_where_its_hidden_state_went(conn, cfg, monkeypatch, tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.cli import Repl

    monkeypatch.setattr(hidden, "HIDDEN_DIR", tmp_path)

    class FakePlan:
        label, tool, args, reply, score = "chat", None, {}, "Hi.", 0.01
        gate_source, raw = "fake", None
        hidden = np.zeros((37, 2560), dtype=np.float32)
        ms_prefill = ms_gate = ms_gen = 1
        prompt_tokens = 5

    class FakeAgent:
        def plan(self, transcript, now):
            return FakePlan()

        def remember(self, user, assistant):
            pass

    now = datetime(2026, 8, 6, 10, 0, tzinfo=ZoneInfo("Australia/Sydney"))
    repl = Repl(conn, cfg, session_id="sess", now_fn=lambda: now, agent=FakeAgent())
    repl.handle("hello there")

    stored = conn.execute("SELECT hidden_state_path FROM turn_log").fetchone()[0]
    assert stored is not None
    assert "sess" in stored


def test_a_typed_tool_call_stores_no_hidden_state(conn, cfg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.cli import Repl

    now = datetime(2026, 8, 6, 10, 0, tzinfo=ZoneInfo("Australia/Sydney"))
    repl = Repl(conn, cfg, session_id="sess", now_fn=lambda: now, use_model=False)
    repl.handle("get_now")

    assert conn.execute("SELECT hidden_state_path FROM turn_log").fetchone()[0] is None

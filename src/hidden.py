"""Storing the hidden states a turn produced.

PLAN.md M2: capture `h_L` per turn into `turn_log` + `.npz`. The path goes in
`turn_log.hidden_state_path`; the array goes next to the database.

All 37 layers are kept per turn, not just the chosen one. *L* is decided by the
layer sweep and may be re-decided later -- throwing away the other 36 layers to
save 180 KB would mean re-running the model over the whole history to change
your mind. At roughly 100 turns a day this costs about 19 MB a day, and
`data/` is gitignored.

Arrays are float16. The probe standardises its input anyway, so the precision
that matters is in the classifier's weights, not in the fourth decimal place of
an activation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import ROOT

HIDDEN_DIR = ROOT / "data" / "hidden"


def turn_path(turn_id: int, session_id: str, root: Path | None = None) -> Path:
    """One file per turn, grouped by session so a session can be dropped."""
    base = Path(root) if root else HIDDEN_DIR
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:32] or "unknown"
    return base / safe / f"turn_{turn_id:08d}.npz"


def save_turn(
    hidden: Any, turn_id: int, session_id: str, root: Path | None = None
) -> Path | None:
    """Write one turn's hidden states. Returns the path to log, or None.

    Never raises: a full disk must not cost the user their answer. A missing
    array is a gap in the training data, which is recoverable; a crashed turn
    is not.
    """
    if hidden is None:
        return None
    try:
        import numpy as np

        path = turn_path(turn_id, session_id, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, hidden=np.asarray(hidden, dtype=np.float16))
        return path
    except Exception:  # noqa: BLE001 - see the docstring
        return None


def load_turn(path: Path | str) -> Any:
    import numpy as np

    with np.load(path) as data:
        return data["hidden"]


def relative_to_root(path: Path | None) -> str | None:
    """Store a repo-relative path so the database survives being moved."""
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

"""Runtime configuration.

Everything here is overridable by environment variable so tests can point at a
throwaway database without touching the real one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TZ = "Australia/Sydney"
DEFAULT_DB = ROOT / "data" / "agent.db"
DEFAULT_MODEL = ROOT / "models" / "qwen3-4b-instruct-2507"


@dataclass(frozen=True)
class Config:
    tz: ZoneInfo
    db_path: Path
    #: "prompted" until the probe exists (M3), then "probe".
    gate: str
    #: Invariant #8: weights load from here, never from the network.
    model_dir: Path
    #: "auto" picks bf16 if the card has room and 4-bit NF4 if it does not.
    #: Either way it is one load through HF transformers -- invariant #2.
    quantise: str
    #: Layer the probe reads. Chosen by the sweep at M2; unused before then.
    probe_layer: int | None

    @property
    def tz_name(self) -> str:
        return str(self.tz.key)


def load() -> Config:
    layer = os.environ.get("AGENT_PROBE_LAYER")
    return Config(
        tz=ZoneInfo(os.environ.get("AGENT_TZ", DEFAULT_TZ)),
        db_path=Path(os.environ.get("AGENT_DB", DEFAULT_DB)),
        gate=os.environ.get("AGENT_GATE", "prompted"),
        model_dir=Path(os.environ.get("AGENT_MODEL_DIR", DEFAULT_MODEL)),
        quantise=os.environ.get("AGENT_QUANTISE", "auto"),
        probe_layer=int(layer) if layer else None,
    )

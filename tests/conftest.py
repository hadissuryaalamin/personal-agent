"""Shared fixtures.

Invariant #7: this suite runs with no GPU, no audio device and no network.
Everything here is an in-memory database and a frozen clock.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from src import config
from src.store.db import connect, init_db
from src.tools import registry
from src.tools.context import ToolContext

TZ = ZoneInfo("Australia/Sydney")

#: Thursday 6 August 2026, mid-morning. Every relative expression in the tests
#: is anchored to this, so "next friday" is always the fourteenth.
NOW = datetime(2026, 8, 6, 10, 0, tzinfo=TZ)


@pytest.fixture
def conn():
    connection = connect(":memory:")
    init_db(connection)
    yield connection
    connection.close()


@pytest.fixture
def tz() -> ZoneInfo:
    return TZ


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture
def cfg(tmp_path) -> config.Config:
    return config.Config(
        tz=TZ,
        db_path=tmp_path / "test.db",
        gate="prompted",
        # Points at nothing on purpose: the default suite must never be able to
        # load weights, whatever a test gets wrong (invariant #7).
        model_dir=tmp_path / "no-model-here",
        quantise="auto",
        probe_layer=None,
    )


@pytest.fixture
def ctx(conn) -> ToolContext:
    return ToolContext(conn=conn, now=NOW, tz=TZ, turn_id=None)


@pytest.fixture
def call(ctx):
    """Invoke a tool the way the REPL and, later, the model will."""

    def _call(name: str, confirmed: bool = False, **args: Any) -> dict[str, Any]:
        ctx.confirmed = confirmed
        return registry.call(name, args, ctx)

    return _call


@pytest.fixture
def a_course(call):
    result = call(
        "add_class",
        code="COMP4020",
        title="Agentic Coding Studio",
        weekday="thursday",
        start="9am",
        end="11am",
        location="Hanna Neumann 1.21",
    )
    assert "id" in result, result
    return result

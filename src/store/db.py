"""SQLite connection and schema management.

    python -m src.store.db --init
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def utc_now_iso(moment: datetime | None = None) -> str:
    """Storage format for every timestamp: UTC, ISO-8601, seconds precision."""
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with the settings the rest of the code assumes."""
    if db_path is None:
        db_path = config.load().db_path
    db_path = Path(db_path)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="personal-agent database")
    parser.add_argument("--init", action="store_true", help="create tables if absent")
    parser.add_argument("--db", default=None, help="database path (default from config)")
    args = parser.parse_args(argv)

    path = Path(args.db) if args.db else config.load().db_path
    conn = connect(path)
    if args.init:
        init_db(conn)
        print(f"schema ready at {path}")
    else:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        print(f"{path}: {', '.join(tables) if tables else 'no tables (run --init)'}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

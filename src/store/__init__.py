"""SQLite storage: schema, audit log, soft deletes.

Import the submodules directly (``from src.store import audit``,
``from src.store.db import connect``). This package deliberately re-exports
nothing: eagerly importing ``db`` here makes ``python -m src.store.db`` warn
that the module was already in ``sys.modules``.
"""

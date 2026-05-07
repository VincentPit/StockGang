"""
tests/conftest.py — shared pytest fixtures for the Postgres-backed data layer.

Opt-in, not autouse
-------------------
Only the test files that touch ``api.db`` need a clean DB before each test.
They opt in by declaring ``pytestmark = pytest.mark.usefixtures("_isolated_db")``
at the top of the file. Tests that don't need the DB are unaffected.

Isolation strategy
------------------
TRUNCATE-before-test on Postgres (fast; avoids per-test schema setup) plus
a reset of the in-process L1 cache. If Postgres is unreachable, DB tests
skip with a clear message rather than crash with a connection error.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text


def _postgres_available() -> bool:
    try:
        from myquant.db import sync_engine
        with sync_engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


_PG_OK: bool = _postgres_available()


@pytest.fixture
def _isolated_db():
    """Truncate all tables and reset the L1 cache. Skip if Postgres is down."""
    if not _PG_OK:
        pytest.skip(
            "Postgres unavailable — set POSTGRES_HOST/PORT/USER/DB/PASSWORD "
            "(see .env) and ensure the database is reachable."
        )

    from myquant.db import sync_engine
    from myquant.db.models import Base
    import api.db as api_db

    Base.metadata.create_all(sync_engine)  # idempotent — alembic-equivalent for tests

    table_names = ", ".join(t.name for t in Base.metadata.sorted_tables)
    with sync_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"))

    api_db._mem.clear()

    yield


@pytest.fixture
def db():
    """Module handle for DB tests."""
    import api.db as _db
    return _db

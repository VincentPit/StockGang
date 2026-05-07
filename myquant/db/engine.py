"""
myquant/db/engine.py — Postgres engines.

Two engines, one database
-------------------------
  * ``async_engine`` (asyncpg)     long-term API for FastAPI handlers,
                                   Arq workers (T1b), and any new code.
  * ``sync_engine``  (psycopg2)    used by Alembic and by the api/db.py
                                   compatibility shim that thread-pool
                                   workers (runner.py, advisor.py) call into.

Both point at the same database; they share the ORM models in models.py so
the schema is one source of truth.

The sync engine uses ``pool_pre_ping=True`` because long-running worker
threads can hold idle connections across Postgres-side timeouts.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from myquant.config.settings import settings


# ── Async engine (asyncpg) ────────────────────────────────────────────────────

async_engine: AsyncEngine = create_async_engine(
    settings.POSTGRES_DSN,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)


# ── Sync engine (psycopg2) ────────────────────────────────────────────────────

sync_engine: Engine = create_engine(
    settings.POSTGRES_DSN_SYNC,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)

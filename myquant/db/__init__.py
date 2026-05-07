"""
myquant/db — Postgres data layer for MyQuant.

Layout
------
  models.py   SQLAlchemy declarative ORM models (single source of truth for
              schema; consumed by Alembic and by both engines below).
  engine.py   Async engine (asyncpg) and sync engine (psycopg2). The async
              engine is the long-term API; the sync engine exists for the
              api/db.py compatibility shim and Alembic migrations.
  session.py  AsyncSessionLocal + SessionLocal sessionmakers.

Usage
-----
  Async (preferred for new code, FastAPI handlers, future Arq workers):
      from myquant.db import AsyncSessionLocal
      async with AsyncSessionLocal() as s:
          ...

  Sync (used by api/db.py shim — thread-pool callers):
      from myquant.db import SessionLocal
      with SessionLocal() as s:
          ...
"""
from __future__ import annotations

from .engine import async_engine, sync_engine
from .models import Base, Cache, Job, PaperAccount, PaperPosition, TrainedModel
from .session import AsyncSessionLocal, SessionLocal

#: Postgres NOTIFY channel that carries job-id payloads after every job upsert.
#: WebSocket clients LISTEN here for live progress without polling.
JOB_NOTIFY_CHANNEL = "myquant_jobs"

__all__ = [
    "Base",
    "Cache",
    "Job",
    "JOB_NOTIFY_CHANNEL",
    "PaperAccount",
    "PaperPosition",
    "TrainedModel",
    "async_engine",
    "sync_engine",
    "AsyncSessionLocal",
    "SessionLocal",
]

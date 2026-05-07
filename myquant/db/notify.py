"""
myquant/db/notify.py — Postgres LISTEN listener for job updates.

A single asyncpg connection per FastAPI process holds a ``LISTEN`` on the
``myquant_jobs`` channel. Subscribers (one per WebSocket connection)
register a job_id and are pushed an asyncio.Event each time Postgres
notifies that the job's row has been updated.

Why one shared connection
-------------------------
asyncpg LISTEN is connection-scoped. Holding one shared connection avoids
opening a new socket per WebSocket and means we only consume one Postgres
backend slot for all live progress streams.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Optional

import asyncpg

from myquant.config.settings import settings
from myquant.db import JOB_NOTIFY_CHANNEL

_log = logging.getLogger(__name__)


class JobNotifier:
    """Process-wide singleton that fans out Postgres NOTIFY events to subscribers."""

    def __init__(self) -> None:
        self._conn: Optional[asyncpg.Connection] = None
        self._lock = asyncio.Lock()
        # job_id → set of asyncio.Event objects waiting on that job
        self._waiters: dict[str, set[asyncio.Event]] = defaultdict(set)

    async def _connect(self) -> None:
        """Open the shared listener connection (idempotent)."""
        if self._conn is not None and not self._conn.is_closed():
            return
        self._conn = await asyncpg.connect(settings.POSTGRES_DSN_RAW)
        await self._conn.add_listener(JOB_NOTIFY_CHANNEL, self._on_notify)
        _log.info("JobNotifier listening on Postgres channel %s", JOB_NOTIFY_CHANNEL)

    def _on_notify(self, conn: asyncpg.Connection, pid: int, channel: str, payload: str) -> None:
        """asyncpg callback — wake everyone waiting on this job_id."""
        for ev in list(self._waiters.get(payload, ())):
            ev.set()

    async def subscribe(self, job_id: str) -> asyncio.Event:
        """Return an Event that fires every time NOTIFY arrives for this job_id."""
        async with self._lock:
            await self._connect()
        ev = asyncio.Event()
        self._waiters[job_id].add(ev)
        return ev

    def unsubscribe(self, job_id: str, ev: asyncio.Event) -> None:
        bucket = self._waiters.get(job_id)
        if bucket is None:
            return
        bucket.discard(ev)
        if not bucket:
            self._waiters.pop(job_id, None)

    async def close(self) -> None:
        if self._conn is not None and not self._conn.is_closed():
            await self._conn.close()
        self._conn = None


# Module-level singleton — lazily connects on first subscribe()
job_notifier = JobNotifier()

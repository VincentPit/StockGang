"""
myquant/leader_lock.py — Redis-based leader election for the scheduler.

Why
---
APScheduler running in two scheduler replicas would double-fire every cron
trigger. Postgres LOCK semantics aren't quite right for this (we'd need a
reliably-released advisory lock); a Redis SET-NX-EX lock with periodic
extension gives us ~10s leader-failover with no schema changes.

Pattern
-------
``async with leader_lock(key) as got_it:`` blocks the body until either
acquisition or shutdown:

    async with leader_lock("myquant:scheduler:leader") as got_it:
        if got_it:
            await run_as_leader()
        else:
            await wait_then_retry()

The context manager spawns a renewal task that re-extends the TTL every
``ttl/3`` seconds. On exit (or if renewal fails) the lock is released and
another replica takes over.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis

from myquant.config.settings import settings

_log = logging.getLogger(__name__)

# Default lock TTL: 30 s with renewal every 10 s. A leader that crashes
# is replaced within ~30 s.
_TTL_SECONDS  = 30
_RENEW_PERIOD = 10


def _holder_id() -> str:
    """Stable-per-process identity used as the lock value (for safe release)."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


@asynccontextmanager
async def leader_lock(
    key: str,
    *,
    ttl: int = _TTL_SECONDS,
    renew_period: int = _RENEW_PERIOD,
) -> AsyncIterator[bool]:
    """Try to acquire ``key`` as a Redis lock.

    Yields True if this process holds the lock, False if another replica
    already holds it. The TTL is extended periodically while the body
    runs; on exit (or renewal failure) the lock is released cleanly so
    the next replica can take over without waiting for the TTL.
    """
    client: aioredis.Redis = aioredis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True,
    )
    holder = _holder_id()

    # NX + EX: claim the key only if it does not already exist; auto-expire.
    got = await client.set(key, holder, nx=True, ex=ttl)
    if not got:
        _log.info("leader_lock(%s): held by another replica — standing by", key)
        try:
            yield False
        finally:
            await client.aclose()
        return

    _log.info("leader_lock(%s): acquired by %s (TTL %ds)", key, holder, ttl)

    # Renewal task — re-extends TTL while we hold it. If the lock value no
    # longer matches our holder id (split-brain), we release the role.
    stop = asyncio.Event()

    async def _renew() -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=renew_period)
                return  # stop was set during the sleep
            except asyncio.TimeoutError:
                pass
            try:
                current = await client.get(key)
                if current != holder:
                    _log.warning(
                        "leader_lock(%s): lost ownership (now %s) — stopping renewal",
                        key, current,
                    )
                    stop.set()
                    return
                await client.expire(key, ttl)
            except Exception as exc:
                _log.warning("leader_lock(%s): renewal failed: %s", key, exc)

    renew_task = asyncio.create_task(_renew(), name="leader-lock-renew")

    try:
        yield True
    finally:
        stop.set()
        renew_task.cancel()
        try:
            await renew_task
        except (asyncio.CancelledError, Exception):
            pass
        # Release only if we still own the key (avoid clobbering a successor)
        try:
            current = await client.get(key)
            if current == holder:
                await client.delete(key)
                _log.info("leader_lock(%s): released cleanly", key)
        except Exception:
            pass
        await client.aclose()

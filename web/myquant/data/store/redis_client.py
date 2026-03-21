"""
Redis client — hot cache for latest ticks, signals, and state.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from myquant.config.logging_config import get_logger
from myquant.config.settings import settings

logger = get_logger(__name__)

_TICK_KEY    = "tick:{symbol}"
_SIGNAL_KEY  = "signal:{strategy}:{symbol}"
_STATE_KEY   = "state:{key}"


class RedisClient:
    """
    Async Redis wrapper with domain-specific helpers.
    """

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._redis = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
        await self._redis.ping()
        logger.info("Redis connected at %s:%d", settings.REDIS_HOST, settings.REDIS_PORT)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # ── Tick cache ────────────────────────────────────────────

    async def set_tick(self, symbol: str, tick_dict: dict) -> None:
        key = _TICK_KEY.format(symbol=symbol)
        await self._redis.set(key, json.dumps(tick_dict), ex=60)

    async def get_tick(self, symbol: str) -> Optional[dict]:
        key = _TICK_KEY.format(symbol=symbol)
        raw = await self._redis.get(key)
        return json.loads(raw) if raw else None

    # ── Signal cache ─────────────────────────────────────────

    async def set_signal(self, strategy: str, symbol: str, signal_dict: dict) -> None:
        key = _SIGNAL_KEY.format(strategy=strategy, symbol=symbol)
        await self._redis.set(key, json.dumps(signal_dict), ex=300)

    async def get_signal(self, strategy: str, symbol: str) -> Optional[dict]:
        key = _SIGNAL_KEY.format(strategy=strategy, symbol=symbol)
        raw = await self._redis.get(key)
        return json.loads(raw) if raw else None

    # ── Generic state ─────────────────────────────────────────

    async def set_state(self, key: str, value: Any, ttl: int = 0) -> None:
        rkey = _STATE_KEY.format(key=key)
        serialized = json.dumps(value)
        if ttl:
            await self._redis.set(rkey, serialized, ex=ttl)
        else:
            await self._redis.set(rkey, serialized)

    async def get_state(self, key: str) -> Optional[Any]:
        rkey = _STATE_KEY.format(key=key)
        raw = await self._redis.get(rkey)
        return json.loads(raw) if raw else None

    # ── Pub/Sub ───────────────────────────────────────────────

    async def publish(self, channel: str, message: dict) -> None:
        await self._redis.publish(channel, json.dumps(message))

    def pubsub(self) -> aioredis.client.PubSub:
        return self._redis.pubsub()

    # ── Throttle helper (for order rate limiting) ─────────────

    async def increment_counter(self, key: str, window_seconds: int = 60) -> int:
        """Increment a sliding window counter. Returns new count."""
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        results = await pipe.execute()
        return int(results[0])


# Singleton
redis_client = RedisClient()

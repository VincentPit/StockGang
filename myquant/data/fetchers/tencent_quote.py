"""
Tencent real-time quote fetcher.

Tencent's free endpoint:
  GET http://qt.gtimg.cn/q=sh600036,hk00700,usTSLA
Returns a plain-text response with one variable per symbol.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Iterable
from datetime import datetime

import aiohttp

from myquant.config.logging_config import get_logger
from myquant.models.tick import Tick

logger = get_logger(__name__)

_TENCENT_URL = "http://qt.gtimg.cn/q={symbols}"
_DEFAULT_INTERVAL = 3.0   # seconds between polls


class TencentQuoteFetcher:
    """
    Polls the Tencent quote API at a configurable interval and delivers
    parsed Tick objects to registered callbacks.

    Usage:
        fetcher = TencentQuoteFetcher(symbols=["sh600036", "hk00700"])
        fetcher.on_tick(my_handler)
        await fetcher.start()
    """

    def __init__(
        self,
        symbols: Iterable[str],
        interval: float = _DEFAULT_INTERVAL,
    ) -> None:
        self._symbols: list[str] = list(symbols)
        self._interval = interval
        self._handlers: list[Callable[[Tick], None]] = []
        self._running = False
        self._session: aiohttp.ClientSession | None = None

    # ── Public API ────────────────────────────────────────────

    def on_tick(self, handler: Callable[[Tick], None]) -> None:
        """Register a tick callback."""
        self._handlers.append(handler)

    def add_symbol(self, symbol: str) -> None:
        if symbol not in self._symbols:
            self._symbols.append(symbol)
            logger.info("Added symbol to fetcher: %s", symbol)

    def remove_symbol(self, symbol: str) -> None:
        if symbol in self._symbols:
            self._symbols.remove(symbol)
            logger.info("Removed symbol from fetcher: %s", symbol)

    async def start(self) -> None:
        """Start polling loop (runs until stop() is called)."""
        self._running = True
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            self._session = session
            logger.info(
                "TencentQuoteFetcher started, polling %d symbols every %.1fs",
                len(self._symbols),
                self._interval,
            )
            while self._running:
                try:
                    await self._poll()
                except Exception as exc:
                    logger.warning("Poll error: %s", exc)
                await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("TencentQuoteFetcher stopped.")

    # ── Internal ─────────────────────────────────────────────

    async def _poll(self) -> None:
        if not self._symbols:
            return

        # Tencent allows multiple symbols per request (comma-separated)
        url = _TENCENT_URL.format(symbols=",".join(self._symbols))
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                logger.warning("Tencent API returned HTTP %d", resp.status)
                return
            raw_text = await resp.text(encoding="gbk", errors="replace")

        for line in raw_text.strip().splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            try:
                tick = Tick.from_tencent_raw(line)
                self._dispatch(tick)
            except ValueError as exc:
                logger.debug("Tick parse error: %s", exc)

    def _dispatch(self, tick: Tick) -> None:
        for handler in self._handlers:
            try:
                handler(tick)
            except Exception as exc:
                logger.error("Tick handler error: %s", exc)


async def fetch_once(symbols: list[str]) -> dict[str, Tick]:
    """
    One-shot fetch — returns a dict of {symbol: Tick}.
    Useful for bootstrapping or ad-hoc price lookups.
    """
    url = _TENCENT_URL.format(symbols=",".join(symbols))
    result: dict[str, Tick] = {}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            raw_text = await resp.text(encoding="gbk", errors="replace")

    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        try:
            tick = Tick.from_tencent_raw(line)
            result[tick.symbol] = tick
        except ValueError:
            pass

    return result

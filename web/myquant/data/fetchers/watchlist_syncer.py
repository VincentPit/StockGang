"""
腾讯自选股 Watchlist Syncer.

Pulls the user's 自选股 watchlist from Tencent's internal API
and keeps it in sync as the trading universe.

Note: This uses the unofficial Tencent Stock app API.
      Requires a valid user cookie/token obtained from the app.
      In development mode, falls back to a local watchlist JSON file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import aiohttp

from myquant.config.logging_config import get_logger
from myquant.config.settings import settings

logger = get_logger(__name__)

_WATCHLIST_API = (
    "https://stockapp.finance.qq.com/mstats/#route=my/list"
    # actual endpoint reverse-engineered from the app
)

# Unofficial API endpoint for fetching grouped watchlists
_ZIXUANGU_API = "https://ifzq.gtimg.cn/cor/account/info"

# Fallback local file
_LOCAL_WATCHLIST = Path(__file__).resolve().parent.parent.parent / "data" / "watchlist.json"


class WatchlistSyncer:
    """
    Manages the trading universe from 腾讯自选股.

    In live mode : fetches from Tencent API using session cookie.
    In dev mode  : reads from data/watchlist.json.
    """

    def __init__(
        self,
        cookie: Optional[str] = None,
        local_fallback: bool = True,
    ) -> None:
        self._cookie = cookie
        self._local_fallback = local_fallback
        self._watchlist: dict[str, list[str]] = {}  # {group_name: [symbols]}
        self._flat_symbols: list[str] = []

    # ── Public API ────────────────────────────────────────────

    @property
    def symbols(self) -> list[str]:
        """Flat list of all symbols across all groups."""
        return list(self._flat_symbols)

    @property
    def groups(self) -> dict[str, list[str]]:
        """Group → symbol mapping."""
        return dict(self._watchlist)

    async def sync(self) -> list[str]:
        """
        Sync the watchlist. Returns the flat list of symbols.
        """
        if self._cookie and not settings.ENV == "development":
            try:
                symbols = await self._fetch_remote()
                logger.info("Watchlist synced from Tencent API: %d symbols", len(symbols))
                return symbols
            except Exception as exc:
                logger.warning("Remote sync failed (%s), falling back to local.", exc)

        return await self._load_local()

    async def save_local(self) -> None:
        """Persist current watchlist to the local JSON file."""
        _LOCAL_WATCHLIST.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOCAL_WATCHLIST, "w", encoding="utf-8") as f:
            json.dump(self._watchlist, f, ensure_ascii=False, indent=2)
        logger.info("Watchlist saved to %s", _LOCAL_WATCHLIST)

    def add_symbol(self, symbol: str, group: str = "default") -> None:
        self._watchlist.setdefault(group, [])
        if symbol not in self._watchlist[group]:
            self._watchlist[group].append(symbol)
            self._rebuild_flat()

    def remove_symbol(self, symbol: str) -> None:
        for symbols in self._watchlist.values():
            if symbol in symbols:
                symbols.remove(symbol)
        self._rebuild_flat()

    # ── Internal ─────────────────────────────────────────────

    async def _fetch_remote(self) -> list[str]:
        headers = {"Cookie": self._cookie, "User-Agent": "TencentStock/6.0"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _ZIXUANGU_API,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json(content_type=None)

        groups: dict[str, list[str]] = {}
        for group in data.get("data", {}).get("groups", []):
            name = group.get("name", "default")
            stocks = [s["code"] for s in group.get("stocks", []) if "code" in s]
            groups[name] = stocks

        self._watchlist = groups
        self._rebuild_flat()
        return self._flat_symbols

    async def _load_local(self) -> list[str]:
        if _LOCAL_WATCHLIST.exists():
            with open(_LOCAL_WATCHLIST, encoding="utf-8") as f:
                self._watchlist = json.load(f)
            self._rebuild_flat()
            logger.info("Watchlist loaded from local file: %d symbols", len(self._flat_symbols))
        else:
            logger.warning(
                "No local watchlist found at %s. Creating sample.", _LOCAL_WATCHLIST
            )
            self._watchlist = self._default_sample()
            self._rebuild_flat()
            await self.save_local()

        return self._flat_symbols

    def _rebuild_flat(self) -> None:
        seen: set[str] = set()
        flat: list[str] = []
        for symbols in self._watchlist.values():
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    flat.append(s)
        self._flat_symbols = flat

    @staticmethod
    def _default_sample() -> dict[str, list[str]]:
        """Sample watchlist for development / first run."""
        return {
            "港股核心": ["hk00700", "hk09988", "hk03690", "hk02318"],
            "A股蓝筹": ["sh600036", "sh601318", "sz000858", "sh600519"],
            "科技": ["usTSLA", "usAAPL", "usNVDA", "usBABA"],
        }

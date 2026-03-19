"""
News Fetcher — fetches financial news headlines for stocks and macro.

Sources:
  • Stock-level: East Money (东方财富) via AKShare ak.stock_news_em()
  • Macro:       Baidu Finance news via AKShare ak.news_economic_baidu()

Results are memory-cached per symbol for `cache_ttl_minutes` to avoid
hammering APIs on every tick/bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from myquant.config.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_CACHE_TTL_MINS = 30


@dataclass
class NewsItem:
    symbol:  Optional[str]
    title:   str
    content: str  = ""
    source:  str  = ""
    ts:      datetime = field(default_factory=datetime.now)
    url:     str  = ""

    @property
    def full_text(self) -> str:
        """Concatenated title + content for NLP processing."""
        return f"{self.title} {self.content}".strip()


class NewsFetcher:
    """
    Fetches news articles via AKShare with in-memory caching.

    Usage:
        fetcher = NewsFetcher()
        items   = fetcher.fetch_stock_news("hk00700", limit=20)
        macro   = fetcher.fetch_macro_news(limit=30)
    """

    def __init__(self, cache_ttl_minutes: int = _DEFAULT_CACHE_TTL_MINS) -> None:
        self._ttl   = timedelta(minutes=cache_ttl_minutes)
        self._cache: dict[str, tuple[datetime, list[NewsItem]]] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def fetch_stock_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Fetch recent news for a specific stock symbol."""
        key = f"stock:{symbol}"
        if (cached := self._cache.get(key)) and datetime.now() - cached[0] < self._ttl:
            return cached[1]

        items: list[NewsItem] = []
        try:
            import akshare as ak
            code = self._to_em_code(symbol)
            df   = ak.stock_news_em(symbol=code)
            if df is not None and not df.empty:
                for _, row in df.head(limit).iterrows():
                    items.append(
                        NewsItem(
                            symbol  = symbol,
                            title   = str(row.get("新闻标题", row.get("title", ""))),
                            content = str(row.get("新闻内容", row.get("content", "")))[:600],
                            source  = str(row.get("文章来源", row.get("source", "eastmoney"))),
                            ts      = self._parse_ts(row.get("发布时间", row.get("datetime", ""))),
                            url     = str(row.get("新闻链接", row.get("url", ""))),
                        )
                    )
        except Exception as e:
            logger.debug("NewsFetcher: stock_news_em failed %s: %s", symbol, e)

        self._cache[key] = (datetime.now(), items)
        logger.debug("NewsFetcher: fetched %d articles for %s", len(items), symbol)
        return items

    def fetch_macro_news(self, limit: int = 30) -> list[NewsItem]:
        """Fetch general macroeconomic news from Baidu Finance."""
        key = "macro"
        if (cached := self._cache.get(key)) and datetime.now() - cached[0] < self._ttl:
            return cached[1]

        items: list[NewsItem] = []
        try:
            import akshare as ak
            df = ak.news_economic_baidu()
            if df is not None and not df.empty:
                for _, row in df.head(limit).iterrows():
                    items.append(
                        NewsItem(
                            symbol  = None,
                            title   = str(row.get("title",   "")),
                            content = str(row.get("content", ""))[:600],
                            source  = "baidu_finance",
                            ts      = self._parse_ts(row.get("datetime", row.get("date", ""))),
                        )
                    )
        except Exception as e:
            logger.debug("NewsFetcher: macro news fetch failed: %s", e)

        self._cache[key] = (datetime.now(), items)
        return items

    def invalidate(self, symbol: Optional[str] = None) -> None:
        """Clear cache for a symbol (or all caches if symbol is None)."""
        if symbol is None:
            self._cache.clear()
        else:
            self._cache.pop(f"stock:{symbol}", None)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_em_code(symbol: str) -> str:
        """Convert internal symbol format to East Money numeric code string."""
        if symbol.startswith(("sh", "sz")):
            return symbol[2:]
        if symbol.startswith("hk"):
            return symbol[2:].lstrip("0") or "0"
        return symbol

    @staticmethod
    def _parse_ts(raw) -> datetime:
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, pd.Timestamp):
            return raw.to_pydatetime()
        try:
            return pd.to_datetime(str(raw)).to_pydatetime()
        except Exception:
            return datetime.now()

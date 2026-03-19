"""
News Strategy — generates trading signals from real-time news sentiment.

Signal logic:
  1. Fetch latest news via NewsFetcher (cached, re-fetched every N minutes)
  2. Score each headline with SentimentAnalyzer
  3. Maintain a rolling window of the most recent K scored articles per symbol
  4. Aggregate to a single [-1, +1] sentiment score
  5. Emit BUY  when aggregate ≥ buy_threshold  AND confidence ≥ min_confidence
     Emit SELL when aggregate ≤ sell_threshold AND confidence ≥ min_confidence
  6. At most one signal per symbol per trading day (avoids re-entry on same news)

In backtest mode this strategy has limited value because it needs live news.
It is most useful in live / paper trading where the news feed is current.
In backtests it is included for architecture completeness and will only fire
if news happens to be fetched during the replay run (which won't happen for
historical bars — so it effectively stays silent, cleanly).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, date
from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.data.fetchers.news_fetcher import NewsFetcher
from myquant.models.bar import Bar
from myquant.models.signal import Signal, SignalStrength, SignalType
from myquant.models.tick import Tick
from myquant.strategy.base import BaseStrategy
from myquant.strategy.nlp.sentiment_analyzer import SentimentAnalyzer, SentimentScore

logger = get_logger(__name__)


class NewsStrategy(BaseStrategy):
    """
    News-driven sentiment strategy.

    Parameters
    ----------
    buy_threshold          : Aggregate sentiment score to trigger BUY (default 0.40).
    sell_threshold         : Aggregate sentiment score to trigger SELL (default -0.40).
    window_size            : Max articles in rolling sentiment window (default 10).
    news_fetch_interval_mins : How often to refresh news per symbol (default 30).
    min_confidence         : Minimum sentiment confidence to act on (default 0.55).
    """

    def __init__(
        self,
        strategy_id:               str,
        symbols:                   list[str],
        buy_threshold:             float = 0.40,
        sell_threshold:            float = -0.40,
        window_size:               int   = 10,
        news_fetch_interval_mins:  int   = 30,
        min_confidence:            float = 0.55,
    ) -> None:
        super().__init__(strategy_id, symbols)
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold
        self.window_size    = window_size
        self.min_confidence = min_confidence
        self._fetch_interval = timedelta(minutes=news_fetch_interval_mins)

        self._fetcher  = NewsFetcher(cache_ttl_minutes=news_fetch_interval_mins)
        self._analyzer = SentimentAnalyzer()

        # Rolling window of (datetime, SentimentScore) tuples per symbol
        self._windows: dict[str, deque] = {
            sym: deque(maxlen=window_size) for sym in symbols
        }
        self._last_fetch:       dict[str, datetime]       = {}
        self._last_signal_date: dict[str, Optional[date]] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def on_start(self) -> None:
        logger.info(
            "NewsStrategy [%s] started — monitoring %d symbols",
            self.strategy_id, len(self.symbols),
        )

    # ── Handlers ──────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        super().on_bar(bar)
        return self._evaluate(bar.symbol, bar.close, bar.ts)

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        return self._evaluate(tick.symbol, tick.price, tick.ts)

    # ── Core logic ────────────────────────────────────────────────────────

    def _evaluate(
        self, symbol: str, price: float, ts: datetime
    ) -> Optional[Signal]:
        if symbol not in self.symbols:
            return None

        # Refresh news if stale
        last = self._last_fetch.get(symbol)
        if last is None or (datetime.now() - last) >= self._fetch_interval:
            self._refresh(symbol)

        window = self._windows.get(symbol)
        if not window:
            return None

        scores    = [s for _, s in window]
        avg_score = sum(s.raw_score  for s in scores) / len(scores)
        avg_conf  = sum(s.confidence for s in scores) / len(scores)

        if avg_conf < self.min_confidence:
            return None

        # One signal per symbol per calendar day
        today     = ts.date()
        last_date = self._last_signal_date.get(symbol)
        if last_date == today:
            return None

        signal: Optional[Signal] = None

        if avg_score >= self.buy_threshold:
            strength = SignalStrength.STRONG if avg_score > 0.60 else SignalStrength.NORMAL
            signal = self.make_signal(
                symbol, SignalType.BUY, price,
                strength = strength,
                metadata = {
                    "sentiment":  round(avg_score, 3),
                    "confidence": round(avg_conf,  3),
                    "articles":   len(scores),
                    "source":     "news",
                },
            )
            logger.info(
                "NewsStrategy BUY  %s | sentiment=%.2f conf=%.2f (%d articles)",
                symbol, avg_score, avg_conf, len(scores),
            )

        elif avg_score <= self.sell_threshold:
            strength = SignalStrength.STRONG if avg_score < -0.60 else SignalStrength.NORMAL
            signal = self.make_signal(
                symbol, SignalType.SELL, price,
                strength = strength,
                metadata = {
                    "sentiment":  round(avg_score, 3),
                    "confidence": round(avg_conf,  3),
                    "articles":   len(scores),
                    "source":     "news",
                },
            )
            logger.info(
                "NewsStrategy SELL %s | sentiment=%.2f conf=%.2f (%d articles)",
                symbol, avg_score, avg_conf, len(scores),
            )

        if signal:
            self._last_signal_date[symbol] = today

        return signal

    def _refresh(self, symbol: str) -> None:
        self._last_fetch[symbol] = datetime.now()
        articles = self._fetcher.fetch_stock_news(symbol, limit=self.window_size * 2)
        window   = self._windows.setdefault(symbol, deque(maxlen=self.window_size))

        for item in articles:
            score = self._analyzer.analyze(item.full_text)
            # Accept articles even if just barely above noise floor
            if score.confidence >= self.min_confidence * 0.80:
                window.append((item.ts, score))

        logger.debug(
            "NewsStrategy: refreshed %d articles for %s", len(articles), symbol
        )

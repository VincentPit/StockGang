"""
Bar — an OHLCV candle aggregated from ticks or loaded from history.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class BarInterval(str, Enum):
    M1  = "1m"
    M5  = "5m"
    M15 = "15m"
    M30 = "30m"
    H1  = "1h"
    D1  = "1d"
    W1  = "1w"


@dataclass(slots=True)
class Bar:
    """
    A single OHLCV candle.

    symbol  : e.g. "sh600036"
    ts      : Bar open time
    interval: BarInterval
    open    : Opening price
    high    : High price
    low     : Low price
    close   : Closing price
    volume  : Volume in shares/lots
    turnover: Turnover in CNY/HKD/USD
    vwap    : Volume-weighted average price (optional)
    is_complete: False if this is the still-building current bar
    """

    symbol: str
    ts: datetime
    interval: BarInterval
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    turnover: float = 0.0
    vwap: float = 0.0
    is_complete: bool = True

    @property
    def pct_chg(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0

    @property
    def body(self) -> float:
        """Absolute candle body size."""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    def __repr__(self) -> str:
        direction = "▲" if self.is_bullish else "▼"
        return (
            f"Bar({self.symbol} {self.interval.value} "
            f"O={self.open} H={self.high} L={self.low} C={self.close} "
            f"{direction}{self.pct_chg:+.2%} @ {self.ts.strftime('%Y-%m-%d %H:%M')})"
        )

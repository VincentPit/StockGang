"""
MACD Strategy — MACD line / Signal line crossover with histogram confirmation.
"""
from __future__ import annotations

from typing import Optional

from myquant.models.bar import Bar
from myquant.models.signal import Signal, SignalType, SignalStrength
from myquant.strategy.base import BaseStrategy


def _ema_series(values: list[float], period: int) -> list[float]:
    """Return full EMA series (same length as input, NaN-padded at start)."""
    if not values:
        return []
    k = 2.0 / (period + 1)
    result = [float("nan")] * len(values)
    # seed with SMA
    if len(values) >= period:
        result[period - 1] = sum(values[:period]) / period
        for i in range(period, len(values)):
            result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (macd_line, signal_line, histogram) or (None, None, None)."""
    if len(closes) < slow + signal:
        return None, None, None

    fast_ema = _ema_series(closes, fast)
    slow_ema = _ema_series(closes, slow)

    macd_line = [
        (f - s) if not (isinstance(f, float) and f != f) and not (isinstance(s, float) and s != s) else float("nan")
        for f, s in zip(fast_ema, slow_ema)
    ]

    valid_macd = [v for v in macd_line if v == v]  # filter NaN
    if len(valid_macd) < signal:
        return None, None, None

    signal_ema = _ema_series(valid_macd, signal)
    if not signal_ema or signal_ema[-1] != signal_ema[-1]:
        return None, None, None

    m = valid_macd[-1]
    s = signal_ema[-1]
    h = m - s
    prev_m = valid_macd[-2] if len(valid_macd) >= 2 else None
    prev_s = signal_ema[-2] if len(signal_ema) >= 2 else None

    return m, s, h


class MACDStrategy(BaseStrategy):
    """
    Parameters:
        fast    : Fast EMA period (default: 12)
        slow    : Slow EMA period (default: 26)
        signal  : Signal EMA period (default: 9)
        min_hist: Minimum histogram magnitude to trigger signal (default: 0.0)
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        min_hist: float = 0.0,
    ) -> None:
        super().__init__(strategy_id, symbols)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.min_hist = min_hist
        self._prev_hist: dict[str, Optional[float]] = {}

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        super().on_bar(bar)
        symbol = bar.symbol
        closes = self.closes(symbol)

        macd_line, signal_line, hist = _macd(closes, self.fast, self.slow, self.signal_period)
        if macd_line is None or hist is None:
            return None

        prev_hist = self._prev_hist.get(symbol)
        trade_signal = None

        if prev_hist is not None:
            # Histogram flips from negative → positive (bullish momentum)
            if prev_hist < 0 and hist > 0 and abs(hist) >= self.min_hist:
                trade_signal = self.make_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=bar.close,
                    confidence=min(1.0, abs(hist) * 5),
                    macd=round(macd_line, 6),
                    signal_line=round(signal_line, 6),
                    histogram=round(hist, 6),
                )
                trade_signal.strength = SignalStrength.STRONG if hist > 0.01 else SignalStrength.NORMAL

            # Histogram flips from positive → negative (bearish momentum)
            elif prev_hist > 0 and hist < 0 and abs(hist) >= self.min_hist:
                trade_signal = self.make_signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=bar.close,
                    confidence=min(1.0, abs(hist) * 5),
                    macd=round(macd_line, 6),
                    signal_line=round(signal_line, 6),
                    histogram=round(hist, 6),
                )

        self._prev_hist[symbol] = hist
        return trade_signal

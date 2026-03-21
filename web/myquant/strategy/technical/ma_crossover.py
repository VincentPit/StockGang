"""
MA Crossover Strategy — dual moving average crossover.

Golden Cross: fast MA crosses above slow MA → BUY signal
Death Cross : fast MA crosses below slow MA → SELL signal
"""
from __future__ import annotations

from typing import Optional

from myquant.models.bar import Bar
from myquant.models.signal import Signal, SignalType, SignalStrength
from myquant.models.tick import Tick
from myquant.strategy.base import BaseStrategy


def _sma(values: list[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema(values: list[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


class MACrossoverStrategy(BaseStrategy):
    """
    Parameters:
        fast_period : Short-term MA period (default: 10)
        slow_period : Long-term MA period (default: 30)
        use_ema     : Use EMA instead of SMA (default: True)
        min_bars    : Minimum bars before generating signals (default: slow_period + 5)
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        fast_period: int = 10,
        slow_period: int = 30,
        use_ema: bool = True,
    ) -> None:
        super().__init__(strategy_id, symbols)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema
        self._ma_fn = _ema if use_ema else _sma

        # Track previous cross state per symbol to detect crossover events
        self._prev_fast: dict[str, Optional[float]] = {}
        self._prev_slow: dict[str, Optional[float]] = {}

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        super().on_bar(bar)  # appends to buffer
        symbol = bar.symbol
        closes = self.closes(symbol)

        fast = self._ma_fn(closes, self.fast_period)
        slow = self._ma_fn(closes, self.slow_period)

        if fast is None or slow is None:
            return None

        prev_fast = self._prev_fast.get(symbol)
        prev_slow = self._prev_slow.get(symbol)

        signal = None

        if prev_fast is not None and prev_slow is not None:
            was_above = prev_fast > prev_slow
            is_above  = fast > slow

            if not was_above and is_above:
                # Golden Cross ✨
                signal = self.make_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=bar.close,
                    confidence=min(1.0, (fast - slow) / slow * 10 + 0.5),
                    fast_ma=round(fast, 4),
                    slow_ma=round(slow, 4),
                    cross_type="golden",
                )
                signal.strength = SignalStrength.STRONG if (fast - slow) / slow > 0.003 else SignalStrength.NORMAL

            elif was_above and not is_above:
                # Death Cross ☠️
                signal = self.make_signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=bar.close,
                    confidence=min(1.0, (slow - fast) / slow * 10 + 0.5),
                    fast_ma=round(fast, 4),
                    slow_ma=round(slow, 4),
                    cross_type="death",
                )

        self._prev_fast[symbol] = fast
        self._prev_slow[symbol] = slow

        return signal

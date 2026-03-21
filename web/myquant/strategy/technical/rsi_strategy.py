"""
RSI Mean-Reversion Strategy.

Oversold : RSI < lower_threshold → BUY
Overbought: RSI > upper_threshold → SELL
"""
from __future__ import annotations

from typing import Optional

from myquant.models.bar import Bar
from myquant.models.signal import Signal, SignalType, SignalStrength
from myquant.strategy.base import BaseStrategy


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-(period):]]
    losses = [abs(min(d, 0)) for d in deltas[-(period):]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class RSIStrategy(BaseStrategy):
    """
    Parameters:
        period          : RSI period (default: 14)
        oversold        : RSI below this → BUY (default: 30)
        overbought      : RSI above this → SELL (default: 70)
        exit_mid        : RSI crosses 50 line → exit (default: True)
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_mid: bool = True,
    ) -> None:
        super().__init__(strategy_id, symbols)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_mid = exit_mid
        self._prev_rsi: dict[str, Optional[float]] = {}

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        super().on_bar(bar)
        symbol = bar.symbol
        closes = self.closes(symbol)

        rsi = _rsi(closes, self.period)
        if rsi is None:
            return None

        prev_rsi = self._prev_rsi.get(symbol)
        signal = None

        if prev_rsi is not None:
            # Entry signals
            if prev_rsi >= self.oversold and rsi < self.oversold:
                # Just entered oversold territory
                signal = self.make_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=bar.close,
                    confidence=1.0 - (rsi / self.oversold),
                    rsi=round(rsi, 2),
                    threshold=self.oversold,
                )
                signal.strength = SignalStrength.STRONG if rsi < 25 else SignalStrength.NORMAL

            elif prev_rsi <= self.overbought and rsi > self.overbought:
                # Just entered overbought territory
                signal = self.make_signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=bar.close,
                    confidence=(rsi - self.overbought) / (100 - self.overbought),
                    rsi=round(rsi, 2),
                    threshold=self.overbought,
                )

            # Exit signals (RSI crosses 50)
            elif self.exit_mid:
                if prev_rsi < 50 and rsi >= 50:
                    signal = self.make_signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        price=bar.close,
                        rsi=round(rsi, 2),
                        reason="rsi_mid_cross_up",
                    )
                elif prev_rsi > 50 and rsi <= 50:
                    signal = self.make_signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_SHORT,
                        price=bar.close,
                        rsi=round(rsi, 2),
                        reason="rsi_mid_cross_down",
                    )

        self._prev_rsi[symbol] = rsi
        return signal

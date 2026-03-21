"""
BaseStrategy — all strategies must inherit from this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from myquant.models.bar import Bar
from myquant.models.signal import Signal
from myquant.models.tick import Tick


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Lifecycle:
        1. __init__  — set parameters
        2. on_start  — called once when engine starts (warm-up, load history, etc.)
        3. on_tick   — called on every incoming real-time tick
        4. on_bar    — called when a new completed bar is available
        5. on_stop   — called once when engine shuts down

    Subclasses emit Signals by returning them from on_tick / on_bar.
    Return None to indicate no signal.
    """

    def __init__(self, strategy_id: str, symbols: list[str]) -> None:
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.is_active: bool = True

        # Per-symbol bar history buffer (keep last N bars)
        # 756 = 3 years of daily bars (252 × 3), providing enough history for
        # the 252-bar vol-regime rolling window PLUS max_train_bars=504.
        self._bar_buffer: dict[str, deque[Bar]] = {
            sym: deque(maxlen=756) for sym in symbols
        }
        # Latest tick per symbol
        self._last_tick: dict[str, Tick] = {}

    # ── Lifecycle hooks (override as needed) ──────────────────

    async def on_start(self) -> None:
        """Called once when the engine starts. Load historical data here."""

    async def on_stop(self) -> None:
        """Called once when the engine stops."""

    # ── Event handlers (implement in subclass) ────────────────

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        """
        Called on every incoming tick for watched symbols.
        Return a Signal to trigger order placement, or None.
        """
        self._last_tick[tick.symbol] = tick
        return None

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        """
        Called each time a completed bar is published for watched symbols.
        Return a Signal or None.
        """
        self._bar_buffer[bar.symbol].append(bar)
        return None

    # ── Helper utilities ──────────────────────────────────────

    def bars(self, symbol: str, n: int = 0) -> list[Bar]:
        """Return last n bars for symbol. n=0 → all buffered."""
        buf = list(self._bar_buffer.get(symbol, []))
        return buf[-n:] if n else buf

    def closes(self, symbol: str, n: int = 0) -> list[float]:
        return [b.close for b in self.bars(symbol, n)]

    def volumes(self, symbol: str, n: int = 0) -> list[int]:
        return [b.volume for b in self.bars(symbol, n)]

    def last_price(self, symbol: str) -> float:
        tick = self._last_tick.get(symbol)
        if tick:
            return tick.price
        bars = self.bars(symbol, 1)
        return bars[0].close if bars else 0.0

    def warm_bars(self, symbol: str, bars: list[Bar]) -> None:
        """Pre-load historical bars into the buffer (called during on_start)."""
        buf = self._bar_buffer.setdefault(symbol, deque(maxlen=756))
        for bar in bars:
            buf.append(bar)

    def make_signal(
        self,
        symbol: str,
        signal_type,
        price: float,
        quantity: int = 0,
        confidence: float = 1.0,
        **metadata,
    ) -> Signal:
        from myquant.models.signal import Signal
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            quantity=quantity,
            confidence=confidence,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.strategy_id}, active={self.is_active})"

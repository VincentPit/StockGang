"""
Strategy Registry — central store of all active strategy instances.
"""
from __future__ import annotations

from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.models.bar import Bar
from myquant.models.signal import Signal
from myquant.models.tick import Tick
from myquant.strategy.base import BaseStrategy

logger = get_logger(__name__)


class StrategyRegistry:
    """
    Manages all active strategies and routes market events to them.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, BaseStrategy] = {}

    def register(self, strategy: BaseStrategy) -> None:
        if strategy.strategy_id in self._strategies:
            raise ValueError(f"Strategy '{strategy.strategy_id}' already registered.")
        self._strategies[strategy.strategy_id] = strategy
        logger.info("Registered strategy: %s (symbols=%s)", strategy.strategy_id, strategy.symbols)

    def unregister(self, strategy_id: str) -> None:
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            logger.info("Unregistered strategy: %s", strategy_id)

    def get(self, strategy_id: str) -> Optional[BaseStrategy]:
        return self._strategies.get(strategy_id)

    @property
    def all(self) -> list[BaseStrategy]:
        return list(self._strategies.values())

    async def start_all(self) -> None:
        for strategy in self._strategies.values():
            try:
                await strategy.on_start()
                logger.info("Strategy started: %s", strategy.strategy_id)
            except Exception as exc:
                logger.error("Error starting strategy %s: %s", strategy.strategy_id, exc)

    async def stop_all(self) -> None:
        for strategy in self._strategies.values():
            try:
                await strategy.on_stop()
            except Exception as exc:
                logger.error("Error stopping strategy %s: %s", strategy.strategy_id, exc)

    def dispatch_tick(self, tick: Tick) -> list[Signal]:
        """Route a tick to all interested strategies. Returns list of signals."""
        signals: list[Signal] = []
        for strategy in self._strategies.values():
            if not strategy.is_active:
                continue
            if tick.symbol not in strategy.symbols:
                continue
            try:
                signal = strategy.on_tick(tick)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                logger.error("Strategy %s on_tick error: %s", strategy.strategy_id, exc)
        return signals

    def dispatch_bar(self, bar: Bar) -> list[Signal]:
        """Route a bar to all interested strategies. Returns list of signals."""
        signals: list[Signal] = []
        for strategy in self._strategies.values():
            if not strategy.is_active:
                continue
            if bar.symbol not in strategy.symbols:
                continue
            try:
                signal = strategy.on_bar(bar)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                logger.error("Strategy %s on_bar error: %s", strategy.strategy_id, exc)
        return signals

"""
BaseBroker — abstract interface that all broker implementations must follow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from myquant.models.order import Order, OrderStatus


class BaseBroker(ABC):
    """
    Abstract broker interface.

    Concrete implementations:
        PaperBroker  — simulated fills (dev/backtest)
        FutuBroker   — Futu OpenAPI (HK/US/A-shares via Futu)
        IBKRBroker   — Interactive Brokers (US/global)
    """

    def __init__(self) -> None:
        self._fill_callbacks: list[Callable[[Order], None]] = []
        self._reject_callbacks: list[Callable[[Order, str], None]] = []

    # ── Lifecycle ─────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the broker API."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect cleanly."""

    # ── Trading ───────────────────────────────────────────────

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        Submit order to broker.
        Returns broker-assigned order ID.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.
        Returns True if cancellation request was accepted.
        """

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Poll current status of an order."""

    # ── Account ───────────────────────────────────────────────

    @abstractmethod
    async def get_cash(self) -> float:
        """Return current available cash balance."""

    @abstractmethod
    async def get_positions(self) -> dict:
        """Return current positions from broker (symbol → qty)."""

    # ── Callbacks ────────────────────────────────────────────

    def on_fill(self, callback: Callable[[Order], None]) -> None:
        self._fill_callbacks.append(callback)

    def on_reject(self, callback: Callable[[Order, str], None]) -> None:
        self._reject_callbacks.append(callback)

    def _notify_fill(self, order: Order) -> None:
        for cb in self._fill_callbacks:
            try:
                cb(order)
            except Exception:
                pass

    def _notify_reject(self, order: Order, reason: str) -> None:
        for cb in self._reject_callbacks:
            try:
                cb(order, reason)
            except Exception:
                pass

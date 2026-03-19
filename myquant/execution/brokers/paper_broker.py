"""
Paper Broker — simulates order execution locally.
Used for development, paper trading, and backtesting.

Fill model:
  MARKET orders → filled immediately at current price + slippage
  LIMIT  orders → filled when price crosses limit level
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.execution.brokers.base_broker import BaseBroker
from myquant.models.order import Order, OrderSide, OrderStatus, OrderType

logger = get_logger(__name__)

_DEFAULT_COMMISSION_RATE = 0.0003   # 0.03%
_DEFAULT_SLIPPAGE        = 0.0002   # 0.02%
_STAMP_DUTY_SELL_CN      = 0.001    # 0.1% stamp duty on A-share sells


class PaperBroker(BaseBroker):
    """
    Simulated broker for paper trading and backtesting.

    Parameters:
        initial_cash      : Starting cash balance
        commission_rate   : Per-trade commission rate (default 0.03%)
        slippage          : Market impact slippage (default 0.02%)
        apply_stamp_duty  : Apply CN stamp duty on A-share sells
        price_getter      : Optional callable(symbol) → float for live prices
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = _DEFAULT_COMMISSION_RATE,
        slippage: float = _DEFAULT_SLIPPAGE,
        apply_stamp_duty: bool = True,
        price_getter=None,
    ) -> None:
        super().__init__()
        self._cash = initial_cash
        self._commission_rate = commission_rate
        self._slippage = slippage
        self._apply_stamp_duty = apply_stamp_duty
        self._price_getter = price_getter  # callable(symbol) → float

        self._open_orders: dict[str, Order] = {}
        self._filled_orders: list[Order] = []
        self._positions: dict[str, int] = {}    # symbol → net quantity
        self._avg_prices: dict[str, float] = {}  # symbol → average fill price

        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────

    async def connect(self) -> None:
        self._connected = True
        logger.info("PaperBroker connected. Cash: %.2f", self._cash)

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBroker disconnected.")

    # ── Trading ───────────────────────────────────────────────

    async def submit_order(self, order: Order) -> str:
        broker_id = f"PAPER-{str(uuid.uuid4())[:8].upper()}"
        order.broker_order_id = broker_id
        order.status = OrderStatus.SUBMITTED
        self._open_orders[broker_id] = order
        logger.info("PaperBroker received order: %s", order)

        if order.order_type == OrderType.MARKET:
            # Fill immediately
            await asyncio.sleep(0)  # yield to event loop
            fill_price = self._get_fill_price(order)
            self._fill_order(order, fill_price)
        else:
            logger.debug("LIMIT order %s queued at %.4f", broker_id, order.limit_price)

        return broker_id

    async def cancel_order(self, broker_order_id: str) -> bool:
        order = self._open_orders.get(broker_order_id)
        if order and not order.is_terminal:
            order.status = OrderStatus.CANCELLED
            del self._open_orders[broker_order_id]
            logger.info("Order cancelled: %s", broker_order_id)
            return True
        return False

    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        order = self._open_orders.get(broker_order_id)
        if order:
            return order.status
        # Check filled orders
        for o in self._filled_orders:
            if o.broker_order_id == broker_order_id:
                return o.status
        return OrderStatus.EXPIRED

    async def get_cash(self) -> float:
        return self._cash

    async def get_positions(self) -> dict:
        return dict(self._positions)

    # ── Simulation helpers ────────────────────────────────────

    def simulate_tick(self, symbol: str, price: float) -> None:
        """
        Called by the backtester/engine to update prices and check LIMIT fills.
        """
        for broker_id, order in list(self._open_orders.items()):
            if order.symbol != symbol or order.is_terminal:
                continue
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    self._fill_order(order, order.limit_price)
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    self._fill_order(order, order.limit_price)

    def _get_fill_price(self, order: Order) -> float:
        """Market order fill price with slippage."""
        base = self._price_getter(order.symbol) if self._price_getter else order.limit_price
        if base == 0:
            base = order.limit_price or 1.0
        slip = base * self._slippage
        if order.side == OrderSide.BUY:
            return base + slip
        else:
            return base - slip

    def _fill_order(self, order: Order, fill_price: float) -> None:
        qty = order.remaining_quantity

        # Calculate commission
        notional = fill_price * qty
        commission = notional * self._commission_rate

        # Stamp duty on A-share sells
        if (
            self._apply_stamp_duty
            and order.side == OrderSide.SELL
            and order.symbol.startswith(("sh", "sz"))
        ):
            commission += notional * _STAMP_DUTY_SELL_CN

        # Check buying power
        if order.side == OrderSide.BUY:
            total_cost = notional + commission
            if total_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.notes = f"Insufficient cash: need {total_cost:.2f}, have {self._cash:.2f}"
                logger.warning("Order REJECTED (insufficient cash): %s", order.order_id[:8])
                if order.broker_order_id in self._open_orders:
                    del self._open_orders[order.broker_order_id]
                self._notify_reject(order, order.notes)
                return

        # Apply fill
        order.apply_fill(qty, fill_price, commission)

        # Update cash, position and weighted-average cost
        if order.side == OrderSide.BUY:
            old_qty = self._positions.get(order.symbol, 0)
            old_avg = self._avg_prices.get(order.symbol, 0.0)
            new_qty = old_qty + qty
            # Weighted-average cost basis
            if new_qty > 0:
                self._avg_prices[order.symbol] = (
                    (old_qty * old_avg + qty * fill_price) / new_qty
                )
            self._cash -= (fill_price * qty + commission)
            self._positions[order.symbol] = new_qty
        else:
            self._cash += (fill_price * qty - commission)
            new_qty = self._positions.get(order.symbol, 0) - qty
            self._positions[order.symbol] = new_qty
            if new_qty <= 0:
                del self._positions[order.symbol]
                self._avg_prices.pop(order.symbol, None)

        # Move to filled
        if order.broker_order_id in self._open_orders:
            del self._open_orders[order.broker_order_id]
        self._filled_orders.append(order)

        logger.info(
            "FILL: %s %s %d×%s @ %.4f (commission: %.2f)",
            order.side.value,
            order.symbol,
            qty,
            order.symbol,
            fill_price,
            commission,
        )
        self._notify_fill(order)

    @property
    def filled_orders(self) -> list[Order]:
        return list(self._filled_orders)

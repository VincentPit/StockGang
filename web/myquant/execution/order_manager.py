"""
Order Manager — owns the full order lifecycle from signal to fill.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.execution.brokers.base_broker import BaseBroker
from myquant.models.order import Order, OrderSide, OrderStatus, OrderType
from myquant.models.signal import Signal, SignalType

logger = get_logger(__name__)

# Minimum lot sizes by market
_MIN_LOT = {"sh": 100, "sz": 100, "hk": 100, "us": 1}


def _round_to_lot(qty: int, symbol: str) -> int:
    prefix = symbol[:2]
    lot = _MIN_LOT.get(prefix, 1)
    return max(lot, (qty // lot) * lot)


class OrderManager:
    """
    Converts signals into orders and tracks their lifecycle.
    """

    def __init__(self, broker: BaseBroker) -> None:
        self._broker = broker
        self._open_orders: dict[str, Order] = {}  # order_id → Order
        self._all_orders: list[Order] = []
        self._on_fill_callbacks: list = []
        broker.on_fill(self._handle_fill)
        broker.on_reject(self._handle_reject)

    # ── Public ────────────────────────────────────────────────

    def on_fill(self, callback) -> None:
        self._on_fill_callbacks.append(callback)

    async def process_signal(
        self,
        signal: Signal,
        nav: float,
        adjusted_qty: Optional[int] = None,
    ) -> Optional[Order]:
        """
        Convert a Signal into an Order and submit to broker.
        Returns the Order if submitted, None if skipped.
        """
        side = self._signal_to_side(signal.signal_type)
        if side is None:
            logger.debug("Signal type %s produces no order.", signal.signal_type)
            return None

        # Determine quantity
        qty = adjusted_qty or signal.quantity
        if qty <= 0:
            qty = self._size_order(signal, nav)
        qty = _round_to_lot(qty, signal.symbol)
        if qty <= 0:
            logger.warning("Order sizing returned 0 for %s — skipping.", signal.symbol)
            return None

        order = Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            signal_id=signal.signal_id,
            strategy_id=signal.strategy_id,
            limit_price=signal.price,
        )

        try:
            broker_id = await self._broker.submit_order(order)
            order.broker_order_id = broker_id
            self._open_orders[order.order_id] = order
            self._all_orders.append(order)
            logger.info("Order submitted: %s", order)
            return order
        except Exception as exc:
            logger.error("Order submission failed: %s — %s", order, exc)
            order.status = OrderStatus.REJECTED
            order.notes = str(exc)
            self._all_orders.append(order)
            return None

    async def cancel_all(self) -> None:
        """Cancel all open orders (e.g., on shutdown or circuit breaker)."""
        for order in list(self._open_orders.values()):
            if not order.is_terminal:
                await self._broker.cancel_order(order.broker_order_id)
                logger.info("Cancelled order: %s", order.order_id[:8])

    @property
    def open_orders(self) -> list[Order]:
        return [o for o in self._open_orders.values() if not o.is_terminal]

    @property
    def all_orders(self) -> list[Order]:
        return list(self._all_orders)

    # ── Handlers ─────────────────────────────────────────────

    def _handle_fill(self, order: Order) -> None:
        if order.order_id in self._open_orders and order.is_terminal:
            del self._open_orders[order.order_id]
        logger.info("Fill confirmed: %s", order)
        for cb in self._on_fill_callbacks:
            try:
                cb(order)
            except Exception as exc:
                logger.error("Fill callback error: %s", exc)

    def _handle_reject(self, order: Order, reason: str) -> None:
        if order.order_id in self._open_orders:
            del self._open_orders[order.order_id]
        logger.warning("Order rejected: %s — %s", order.order_id[:8], reason)

    # ── Internal ─────────────────────────────────────────────

    @staticmethod
    def _signal_to_side(signal_type: SignalType) -> Optional[OrderSide]:
        mapping = {
            SignalType.BUY:         OrderSide.BUY,
            SignalType.SHORT:       OrderSide.SELL,
            SignalType.SELL:        OrderSide.SELL,
            SignalType.CLOSE_LONG:  OrderSide.SELL,
            SignalType.CLOSE_SHORT: OrderSide.BUY,
        }
        return mapping.get(signal_type)

    @staticmethod
    def _size_order(signal: Signal, nav: float) -> int:
        """
        Default position sizing: fixed fractional (2% of NAV per trade).
        Can be overridden or replaced with a Kelly-criterion sizer.
        """
        RISK_PCT = 0.02
        notional = nav * RISK_PCT
        if signal.price <= 0:
            return 0
        return int(notional / signal.price)

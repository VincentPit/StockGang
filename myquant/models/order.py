"""
Order — represents a trade instruction sent to a broker.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    STOP   = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TWAP   = "TWAP"
    VWAP   = "VWAP"


class OrderStatus(str, Enum):
    PENDING    = "PENDING"     # Created, not yet submitted
    SUBMITTED  = "SUBMITTED"   # Sent to broker
    PARTIAL    = "PARTIAL"     # Partially filled
    FILLED     = "FILLED"      # Fully filled
    CANCELLED  = "CANCELLED"
    REJECTED   = "REJECTED"
    EXPIRED    = "EXPIRED"


@dataclass
class Order:
    """
    Full lifecycle order object.

    signal_id      : Links back to originating Signal
    strategy_id    : Originating strategy name
    symbol         : Instrument code
    side           : BUY | SELL
    order_type     : MARKET | LIMIT | STOP | TWAP | VWAP
    quantity       : Total shares requested
    limit_price    : Required for LIMIT / STOP_LIMIT orders
    stop_price     : Required for STOP / STOP_LIMIT orders
    filled_quantity: Cumulative shares filled
    avg_fill_price : Average fill price so far
    commission     : Total commission paid
    status         : Current lifecycle status
    broker_order_id: ID returned by the broker
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int

    signal_id: str = ""
    strategy_id: str = ""
    limit_price: float = 0.0
    stop_price: float = 0.0
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: str = ""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    notes: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def fill_pct(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity else 0.0

    @property
    def notional(self) -> float:
        """Estimated notional (uses limit_price, else avg_fill_price, else 0)."""
        price = self.avg_fill_price or self.limit_price
        return price * self.quantity

    def apply_fill(self, qty: int, price: float, commission: float = 0.0) -> None:
        """Update order state after a partial or full fill."""
        prev_filled = self.filled_quantity
        self.filled_quantity += qty
        # Recalculate average fill price
        total_cost = self.avg_fill_price * prev_filled + price * qty
        self.avg_fill_price = total_cost / self.filled_quantity
        self.commission += commission
        self.updated_at = datetime.now()
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

    def __repr__(self) -> str:
        return (
            f"Order({self.order_id[:8]} {self.side.value} {self.quantity}×{self.symbol} "
            f"@ {self.limit_price or 'MKT'} [{self.status.value}] "
            f"filled={self.filled_quantity})"
        )

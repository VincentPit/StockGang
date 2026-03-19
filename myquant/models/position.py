"""
Position — current holding in a single instrument.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """
    Tracks a live holding in one instrument.

    symbol         : Instrument code
    quantity       : Current net long position (negative = short)
    avg_cost       : Average cost basis per share
    realized_pnl   : Locked-in profit/loss from closed portions
    market_price   : Latest market price (updated on each tick)
    """

    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    market_price: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # ── Computed properties ───────────────────────────────────

    @property
    def market_value(self) -> float:
        return self.market_price * self.quantity

    @property
    def cost_basis(self) -> float:
        return self.avg_cost * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        return (self.market_price - self.avg_cost) * self.quantity

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        basis = self.cost_basis
        return self.unrealized_pnl / abs(basis) if basis else 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    # ── Mutation helpers ──────────────────────────────────────

    def add_fill(self, qty: int, price: float) -> float:
        """
        Apply a fill to this position.
        qty > 0 = buy fill, qty < 0 = sell fill.
        Returns the realized PnL for this fill (only non-zero when closing/reducing).
        """
        realized = 0.0
        self.updated_at = datetime.now()

        if self.quantity == 0:
            # Opening a brand-new position
            self.quantity = qty
            self.avg_cost = price
        elif (self.quantity > 0 and qty > 0) or (self.quantity < 0 and qty < 0):
            # Increasing same-direction position — recalculate avg cost
            total_cost = self.avg_cost * self.quantity + price * qty
            self.quantity += qty
            self.avg_cost = total_cost / self.quantity
        else:
            # Reducing or flipping position
            reducing = min(abs(qty), abs(self.quantity))
            realized = (price - self.avg_cost) * reducing * (1 if self.quantity > 0 else -1)
            self.realized_pnl += realized
            self.quantity += qty
            if self.quantity == 0:
                self.avg_cost = 0.0
            elif (self.quantity > 0 and qty < 0 and abs(qty) > 0) or (
                self.quantity < 0 and qty > 0
            ):
                # Flipped direction — reset avg cost to new fill price
                self.avg_cost = price

        return realized

    def update_price(self, price: float) -> None:
        self.market_price = price
        self.updated_at = datetime.now()

    def __repr__(self) -> str:
        direction = "LONG" if self.quantity > 0 else ("SHORT" if self.quantity < 0 else "FLAT")
        return (
            f"Position({self.symbol} {direction} {abs(self.quantity)} "
            f"cost={self.avg_cost:.4f} upnl={self.unrealized_pnl:+.2f})"
        )

"""
Portfolio Engine — tracks positions, cash, NAV, and PnL in real time.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.models.order import Order, OrderSide
from myquant.models.position import Position
from myquant.models.tick import Tick

logger = get_logger(__name__)


class PortfolioEngine:
    """
    Single source of truth for:
        - Cash balance
        - Open positions
        - Realized / unrealized PnL
        - NAV (Net Asset Value)
        - Performance history
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self._cash = initial_cash
        self._initial_nav = initial_cash
        self._positions: dict[str, Position] = {}

        # Performance tracking
        self._nav_history: list[tuple[datetime, float]] = []
        self._peak_nav: float = initial_cash
        self._total_trades: int = 0
        self._started_at = datetime.now()

    # ── Properties ───────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self._positions.values())

    @property
    def nav(self) -> float:
        return self._cash + self.market_value

    @property
    def total_pnl(self) -> float:
        return self.nav - self._initial_nav

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self._initial_nav if self._initial_nav else 0.0

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    @property
    def max_drawdown(self) -> float:
        if not self._nav_history:
            return 0.0
        navs = [n for _, n in self._nav_history]
        peak = navs[0]
        max_dd = 0.0
        for n in navs:
            if n > peak:
                peak = n
            dd = (n - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return max_dd

    @property
    def current_drawdown(self) -> float:
        if self.nav >= self._peak_nav:
            self._peak_nav = self.nav
            return 0.0
        return (self.nav - self._peak_nav) / self._peak_nav

    # ── Event handlers ────────────────────────────────────────

    def on_fill(self, order: Order) -> None:
        """Called by OrderManager when a fill is confirmed."""
        symbol = order.symbol
        qty = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity

        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)

        pos = self._positions[symbol]
        realized = pos.add_fill(qty, order.avg_fill_price)

        # Update cash: actual cost already deducted by broker, but we
        # reconcile here for the portfolio engine's internal ledger.
        net_cash_change = -(order.avg_fill_price * order.filled_quantity + order.commission)
        if order.side == OrderSide.SELL:
            net_cash_change = (order.avg_fill_price * order.filled_quantity - order.commission)

        self._cash += net_cash_change

        # Remove flat positions
        if pos.is_flat:
            del self._positions[symbol]

        self._total_trades += 1

        if realized != 0:
            logger.info(
                "Realized PnL from %s: %.2f (trade #%d)",
                symbol, realized, self._total_trades,
            )

        # Update peak NAV
        if self.nav > self._peak_nav:
            self._peak_nav = self.nav

    def on_tick(self, tick: Tick) -> None:
        """Update mark-to-market prices."""
        pos = self._positions.get(tick.symbol)
        if pos:
            pos.update_price(tick.price)

    def snapshot(self) -> None:
        """Record current NAV to history (call periodically)."""
        self._nav_history.append((datetime.now(), self.nav))

    # ── Reporting ─────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "nav":             round(self.nav, 2),
            "cash":            round(self._cash, 2),
            "market_value":    round(self.market_value, 2),
            "total_pnl":       round(self.total_pnl, 2),
            "total_pnl_pct":   f"{self.total_pnl_pct:.2%}",
            "unrealized_pnl":  round(self.unrealized_pnl, 2),
            "realized_pnl":    round(self.realized_pnl, 2),
            "current_drawdown":f"{self.current_drawdown:.2%}",
            "max_drawdown":    f"{self.max_drawdown:.2%}",
            "total_trades":    self._total_trades,
            "open_positions":  len(self._positions),
            "positions": {
                sym: {
                    "qty":   p.quantity,
                    "cost":  round(p.avg_cost, 4),
                    "price": round(p.market_price, 4),
                    "upnl":  round(p.unrealized_pnl, 2),
                    "pct":   f"{p.pnl_pct:.2%}",
                }
                for sym, p in self._positions.items()
            },
        }

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio from NAV history (daily)."""
        if len(self._nav_history) < 2:
            return 0.0
        navs = [n for _, n in self._nav_history]
        returns = [(navs[i] - navs[i - 1]) / navs[i - 1] for i in range(1, len(navs))]
        if not returns:
            return 0.0
        avg_ret = sum(returns) / len(returns)
        variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        daily_sharpe = (avg_ret - risk_free_rate / 252) / std
        return daily_sharpe * (252 ** 0.5)  # Annualize

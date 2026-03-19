"""
Tests for PortfolioEngine — NAV, PnL, drawdown, Sharpe.
No external data or network calls required.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from myquant.portfolio.portfolio_engine import PortfolioEngine
from myquant.models.order import Order, OrderSide, OrderType
from myquant.models.tick import Tick


# ── helpers ────────────────────────────────────────────────────────────────────

def _buy_order(symbol: str, qty: int, price: float, commission: float = 0.0) -> Order:
    o = Order(symbol, OrderSide.BUY, OrderType.MARKET, qty, limit_price=price)
    o.apply_fill(qty, price, commission=commission)
    return o


def _sell_order(symbol: str, qty: int, price: float, commission: float = 0.0) -> Order:
    o = Order(symbol, OrderSide.SELL, OrderType.MARKET, qty, limit_price=price)
    o.apply_fill(qty, price, commission=commission)
    return o


# ── NAV & cash ─────────────────────────────────────────────────────────────────

class TestNAV:
    def test_initial_nav_equals_cash(self):
        pe = PortfolioEngine(initial_cash=500_000)
        assert pe.nav == pytest.approx(500_000)
        assert pe.cash == pytest.approx(500_000)
        assert pe.market_value == pytest.approx(0.0)

    def test_nav_after_buy(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("hk00700", 100, 300.0))
        # market_price defaults to 0 until a tick arrives; send one at cost price
        pe.on_tick(Tick("hk00700", datetime.now(), price=300.0))
        assert pe.cash == pytest.approx(100_000 - 100 * 300.0)
        # nav = cash + market_value; price unchanged → nav == initial nav
        assert pe.nav == pytest.approx(100_000)

    def test_nav_reflects_price_change(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("hk00700", 100, 300.0))
        pe.on_tick(Tick("hk00700", datetime.now(), price=330.0))
        assert pe.nav == pytest.approx(100_000 + 100 * 30.0)   # +30 per share

    def test_nav_after_full_close(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("sh600036", 1000, 42.0))
        pe.on_fill(_sell_order("sh600036", 1000, 45.0))
        # Position should be gone
        assert "sh600036" not in pe.positions
        assert pe.total_pnl == pytest.approx(3000.0, abs=1.0)

    def test_multiple_symbols_nav(self):
        pe = PortfolioEngine(initial_cash=500_000)
        pe.on_fill(_buy_order("hk00700", 100, 300.0))
        pe.on_fill(_buy_order("sh600036", 1000, 42.0))
        pe.on_tick(Tick("hk00700",  datetime.now(), price=310.0))
        pe.on_tick(Tick("sh600036", datetime.now(), price=44.0))
        expected_nav = 500_000 + 100 * 10.0 + 1000 * 2.0
        assert pe.nav == pytest.approx(expected_nav)


# ── PnL ────────────────────────────────────────────────────────────────────────

class TestPnL:
    def test_unrealized_pnl(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("usTSLA", 50, 200.0))
        pe.on_tick(Tick("usTSLA", datetime.now(), price=210.0))
        assert pe.unrealized_pnl == pytest.approx(500.0)

    def test_realized_pnl_on_partial_close(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("hk00700", 200, 300.0))
        pe.on_fill(_sell_order("hk00700", 100, 310.0))
        assert pe.realized_pnl == pytest.approx(1000.0)   # 100 × (310−300)
        assert pe.positions["hk00700"].quantity == 100

    def test_total_pnl_pct(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("hk00700", 100, 300.0))
        pe.on_fill(_sell_order("hk00700", 100, 330.0))
        # Realized +3000, no open position, unrealized 0 → total_pnl_pct = 3%
        assert pe.total_pnl_pct == pytest.approx(0.03, abs=0.001)

    def test_loss_trade(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("sh600036", 500, 50.0))
        pe.on_fill(_sell_order("sh600036", 500, 45.0))
        # Once fully closed, the position is removed from _positions.
        # realized_pnl sums over open positions only → use total_pnl (nav − initial)
        assert pe.total_pnl == pytest.approx(-2500.0, abs=1.0)


# ── Drawdown ───────────────────────────────────────────────────────────────────

class TestDrawdown:
    def _build_nav_history(self, navs: list[float]) -> PortfolioEngine:
        pe = PortfolioEngine(initial_cash=navs[0])
        base = datetime(2024, 1, 1)
        for i, n in enumerate(navs):
            pe._nav_history.append((base + timedelta(days=i), n))
        return pe

    def test_no_drawdown_on_rising_nav(self):
        pe = self._build_nav_history([100_000, 102_000, 105_000, 110_000])
        assert pe.max_drawdown == pytest.approx(0.0)

    def test_max_drawdown_simple(self):
        # Peak 110k → trough 88k → drawdown = (88k-110k)/110k = -20%
        pe = self._build_nav_history([100_000, 110_000, 88_000, 95_000])
        assert pe.max_drawdown == pytest.approx(-0.2, abs=0.001)

    def test_current_drawdown_at_peak(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe._peak_nav = 100_000
        # nav == peak → 0%
        assert pe.current_drawdown == pytest.approx(0.0)

    def test_current_drawdown_below_peak(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe._peak_nav = 100_000
        pe._cash = 90_000   # nav = 90k, peak = 100k
        assert pe.current_drawdown == pytest.approx(-0.1)


# ── Sharpe ─────────────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_sharpe_zero_with_no_history(self):
        pe = PortfolioEngine()
        assert pe.sharpe_ratio() == 0.0

    def test_sharpe_zero_with_one_nav_point(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.snapshot()
        assert pe.sharpe_ratio() == 0.0

    def test_sharpe_positive_for_upward_nav(self):
        pe = PortfolioEngine(initial_cash=100_000)
        base = datetime(2024, 1, 1)
        # Steadily rising NAV → positive Sharpe
        for i in range(60):
            pe._nav_history.append((base + timedelta(days=i), 100_000 + i * 100))
        assert pe.sharpe_ratio() > 0

    def test_sharpe_negative_for_downward_nav(self):
        pe = PortfolioEngine(initial_cash=100_000)
        base = datetime(2024, 1, 1)
        for i in range(60):
            pe._nav_history.append((base + timedelta(days=i), 100_000 - i * 50))
        assert pe.sharpe_ratio() < 0

    def test_sharpe_zero_std_returns_zero(self):
        # Flat NAV → std=0, should not raise ZeroDivisionError
        pe = PortfolioEngine(initial_cash=100_000)
        base = datetime(2024, 1, 1)
        for i in range(10):
            pe._nav_history.append((base + timedelta(days=i), 100_000))
        assert pe.sharpe_ratio() == 0.0


# ── Summary dict ───────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_keys_present(self):
        pe = PortfolioEngine(initial_cash=500_000)
        s = pe.summary()
        for key in ("nav", "cash", "market_value", "total_pnl", "total_pnl_pct",
                    "unrealized_pnl", "realized_pnl", "current_drawdown",
                    "max_drawdown", "total_trades", "open_positions", "positions"):
            assert key in s, f"Missing key: {key}"

    def test_summary_open_positions_count(self):
        pe = PortfolioEngine(initial_cash=100_000)
        pe.on_fill(_buy_order("hk00700", 100, 300.0))
        pe.on_fill(_buy_order("sh600036", 500, 42.0))
        assert pe.summary()["open_positions"] == 2

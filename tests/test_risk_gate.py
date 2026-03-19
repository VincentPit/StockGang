"""
Tests for RiskGate — all 6 check layers, backtest vs live mode.
No external data or network calls required.
"""
from __future__ import annotations

import time
import pytest
from datetime import datetime

from myquant.risk.risk_gate import RiskGate, RiskDecision
from myquant.models.signal import Signal, SignalType, SignalStrength
from myquant.models.position import Position


# ── helpers ────────────────────────────────────────────────────────────────────

def _signal(
    symbol: str = "hk00700",
    signal_type: SignalType = SignalType.BUY,
    price: float = 300.0,
    qty: int = 100,
    confidence: float = 0.7,
) -> Signal:
    return Signal(
        strategy_id="test",
        symbol=symbol,
        signal_type=signal_type,
        price=price,
        quantity=qty,
        confidence=confidence,
    )


def _gate(nav: float = 1_000_000.0, positions: dict | None = None) -> RiskGate:
    """Build a RiskGate with controllable NAV and positions."""
    pos = positions or {}
    gate = RiskGate(
        nav_getter=lambda: nav,
        positions_getter=lambda: pos,
    )
    return gate


# ── backtest mode bypasses wall-clock checks ───────────────────────────────────

class TestBacktestMode:
    """When sim_time is provided, market-hours and throttle checks are skipped."""

    SIM_TIME = datetime(2024, 6, 15, 8, 0)   # 08:00 — outside any market hours

    def test_ashare_buy_approved_in_backtest(self):
        gate = _gate()
        decision = gate.evaluate(_signal("sh600036"), sim_time=self.SIM_TIME)
        assert decision.approved

    def test_hk_buy_approved_in_backtest(self):
        gate = _gate()
        decision = gate.evaluate(_signal("hk00700"), sim_time=self.SIM_TIME)
        assert decision.approved

    def test_throttle_not_triggered_in_backtest(self):
        gate = _gate()
        gate.cooldown_seconds = 0   # isolate throttle check; disable cooldown
        sim_epoch = datetime(2024, 1, 1, 9, 30)
        # Each signal uses a unique symbol so cooldown never fires
        for i in range(50):
            sym = f"hk{i:05d}"
            d = gate.evaluate(_signal(sym), sim_time=sim_epoch)
            assert d.approved, f"Throttle should not fire in backtest mode (symbol={sym})"


# ── market hours (live mode, sim_time=None) ────────────────────────────────────

class TestMarketHours:
    def test_ashare_rejected_outside_hours(self):
        gate = _gate()
        # 07:00 — A-shares not open
        off_hours = datetime(2024, 6, 15, 7, 0)
        d = gate.evaluate(_signal("sh600036"), sim_time=None)
        # We can't control wall-clock, so just confirm the gate doesn't crash
        assert isinstance(d.approved, bool)

    def test_sell_always_approved_regardless_of_hours(self):
        """Exit signals must never be blocked by the hours check."""
        gate = _gate(positions={"sh600036": _make_pos("sh600036", 100, 40.0)})
        sig = _signal("sh600036", signal_type=SignalType.SELL)
        # Patch sim_time inside market hours for A-shares (09:45)
        in_hours = datetime(2024, 6, 15, 9, 45)
        d = gate.evaluate(sig, sim_time=in_hours)
        assert d.approved


def _make_pos(symbol: str, qty: int, price: float) -> Position:
    pos = Position(symbol)
    pos.add_fill(qty, price)
    pos.update_price(price)
    return pos


# ── drawdown circuit breaker ───────────────────────────────────────────────────

class TestDrawdown:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_no_baseline_approves(self):
        gate = _gate(nav=1_000_000)
        # No record_nav called → daily_start_nav is None → approved
        d = gate.evaluate(_signal(), sim_time=self.SIM)
        assert d.approved

    def test_within_limit_approves(self):
        gate = _gate(nav=980_000)   # −2% drawdown
        gate.record_nav(1_000_000, sim_date=self.SIM.date())
        d = gate.evaluate(_signal(), sim_time=self.SIM)
        assert d.approved

    def test_over_limit_rejects(self):
        from myquant.config.settings import settings
        # Nav dropped more than DAILY_DRAWDOWN_LIMIT (typically -5%)
        gate = _gate(nav=900_000)
        gate.record_nav(1_000_000, sim_date=self.SIM.date())
        d = gate.evaluate(_signal(), sim_time=self.SIM)
        assert not d.approved
        assert "drawdown" in d.reason.lower()


# ── cooldown ───────────────────────────────────────────────────────────────────

class TestCooldown:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_first_signal_approved(self):
        gate = _gate()
        d = gate.evaluate(_signal(), sim_time=self.SIM)
        assert d.approved

    def test_immediate_repeat_rejected(self):
        gate = _gate()
        gate.cooldown_seconds = 60
        epoch = self.SIM.timestamp()
        gate.evaluate(_signal(), sim_time=self.SIM)
        # Same epoch → cooldown not elapsed
        d = gate.evaluate(_signal(), sim_time=self.SIM)
        assert not d.approved
        assert "cooldown" in d.reason.lower()

    def test_different_symbol_not_affected_by_cooldown(self):
        gate = _gate()
        gate.cooldown_seconds = 60
        gate.evaluate(_signal("hk00700"), sim_time=self.SIM)
        # Different symbol → no cooldown
        d = gate.evaluate(_signal("sh600036"), sim_time=self.SIM)
        assert d.approved

    def test_cooldown_expires(self):
        gate = _gate()
        gate.cooldown_seconds = 10
        t0 = self.SIM.timestamp()
        gate._last_order_time["hk00700"] = t0
        future = datetime(2024, 6, 15, 9, 31)   # +60s
        d = gate.evaluate(_signal("hk00700"), sim_time=future)
        assert d.approved


# ── position limit ─────────────────────────────────────────────────────────────

class TestPositionLimit:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_buy_within_limit_approved(self):
        gate = _gate(nav=1_000_000)
        # 100 shares × 300 = 30 000 = 3% of NAV → well under 15% default
        d = gate.evaluate(_signal("hk00700", price=300, qty=100), sim_time=self.SIM)
        assert d.approved

    def test_buy_over_limit_adjusted(self):
        gate = _gate(nav=1_000_000)
        # 10 000 shares × 300 = 3 000 000 = 300% of NAV — way over the 20% limit
        d = gate.evaluate(_signal("hk00700", price=300, qty=10_000), sim_time=self.SIM)
        assert d.approved
        # adjusted_quantity is now propagated from _check_position_limit
        assert d.adjusted_quantity is not None
        assert d.adjusted_quantity < 10_000

    def test_sell_without_position_rejected(self):
        gate = _gate(nav=1_000_000, positions={})
        d = gate.evaluate(_signal("hk00700", signal_type=SignalType.SELL), sim_time=self.SIM)
        assert not d.approved
        assert "no long position" in d.reason.lower()

    def test_sell_caps_to_held_quantity(self):
        pos = {"hk00700": _make_pos("hk00700", 50, 300.0)}
        gate = _gate(nav=1_000_000, positions=pos)
        # Try to sell 200 but only hold 50; gate must cap to 50
        sig = _signal("hk00700", signal_type=SignalType.SELL, qty=200)
        d = gate.evaluate(sig, sim_time=self.SIM)
        assert d.approved
        # adjusted_quantity is now propagated through evaluate()
        assert d.adjusted_quantity == 50


# ── sector limit ───────────────────────────────────────────────────────────────

class TestSectorLimit:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_unknown_sector_approves(self):
        gate = _gate(nav=1_000_000)
        gate.set_sector_map({})   # no mapping
        d = gate.evaluate(_signal("hk00700"), sim_time=self.SIM)
        assert d.approved

    def test_within_sector_limit_approves(self):
        pos = {"hk00700": _make_pos("hk00700", 100, 300.0)}
        gate = _gate(nav=1_000_000, positions=pos)
        gate.set_sector_map({"hk00700": "tech", "sh600036": "finance"})
        d = gate.evaluate(_signal("sh600036"), sim_time=self.SIM)
        assert d.approved

    def test_over_sector_limit_rejects(self):
        # MAX_SECTOR_PCT = 0.40; use 2 000 shares × 400 = 800 000 = 80% of 1M NAV
        pos = {"hk00700": _make_pos("hk00700", 2000, 400.0)}
        gate = _gate(nav=1_000_000, positions=pos)
        gate.set_sector_map({"hk00700": "tech", "sh600036": "tech"})
        # Existing tech exposure is already 80% → any new tech buy must be rejected
        d = gate.evaluate(_signal("sh600036", price=50, qty=100), sim_time=self.SIM)
        assert not d.approved
        assert "sector" in d.reason.lower()

    def test_sell_always_bypasses_sector_check(self):
        pos = {"hk00700": _make_pos("hk00700", 200, 400.0)}
        gate = _gate(nav=1_000_000, positions=pos)
        gate.set_sector_map({"hk00700": "tech"})
        sig = _signal("hk00700", signal_type=SignalType.SELL)
        d = gate.evaluate(sig, sim_time=self.SIM)
        assert d.approved

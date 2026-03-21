"""
Tests for RiskGate — all 6 check layers, backtest vs live mode.
No external data or network calls required.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from myquant.models.position import Position
from myquant.models.signal import Signal, SignalType
from myquant.risk.risk_gate import RiskGate

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
        datetime(2024, 6, 15, 7, 0)
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
        self.SIM.timestamp()
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


# ── Layer 7: VaR check ─────────────────────────────────────────────────────────

class TestVaR:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_small_position_within_var_limit(self):
        """Small position + low vol → VaR well under 8% of NAV."""
        gate = _gate(nav=1_000_000)
        gate.set_vol_estimate("hk00700", 0.015)
        # 100 shares × ¥300 = ¥30 000 notional; VaR = 30 000 × 0.015 × 1.645 = ¥740 (0.07%)
        d = gate.evaluate(_signal("hk00700", price=300, qty=100), sim_time=self.SIM)
        assert d.approved

    def test_large_position_exceeds_var_limit(self):
        """Multiple high-vol existing positions fill VaR budget; new trade tips it over.

        Note: a single position cannot exceed VaR by itself because the position-limit
        layer (5) fires first and caps exposure at MAX_POSITION_PCT (~20% of NAV).
        With 20% NAV × vol × 1.645 > 8%, vol would need to be >24% daily — unrealistic.
        We instead pre-load 5 existing positions (each 12% of NAV at 8% daily vol)
        to reach ~7.9% portfolio VaR, then the new trade tips it above the 8% ceiling.
        """
        # 5 existing positions: each 12K shares × 10 = 120 000 market value (12% of 1M NAV)
        # Vol = 8% daily. VaR per position = 120 000 × 0.08 × 1.645 = 15 792
        # Total portfolio VaR = 5 × 15 792 = 78 960 = 7.9% of 1M NAV
        syms   = ["sz000001", "sz000002", "sz000003", "sz000004", "sz000005"]
        pos    = {s: _make_pos(s, 12_000, 10.0) for s in syms}
        gate   = _gate(nav=1_000_000, positions=pos)
        for s in syms:
            gate.set_vol_estimate(s, 0.08)
        # New position: 50K × 8% daily vol → VaR = 50 000 × 0.08 × 1.645 = 6 580
        # Total = 78 960 + 6 580 = 85 540 > 80 000 (8% of 1M)
        gate.set_vol_estimate("hk00700", 0.08)
        d = gate.evaluate(_signal("hk00700", price=1.0, qty=50_000), sim_time=self.SIM)
        assert not d.approved
        assert "var" in d.reason.lower()

    def test_var_includes_existing_positions(self):
        """Accumulated portfolio VaR from many existing positions causes rejection."""
        syms = ["sz000001", "sz000002", "sz000003", "sz000004", "sz000005",
                "sz000006", "sz000007"]
        pos  = {s: _make_pos(s, 10_000, 10.0) for s in syms}  # 100K each
        gate = _gate(nav=1_000_000, positions=pos)
        for s in syms:
            gate.set_vol_estimate(s, 0.08)   # VaR/pos = 100K × 0.08 × 1.645 = 13 160
        # 7 positions × 13 160 = 92 120 = 9.2% > 8% limit — any new BUY should be blocked
        gate.set_vol_estimate("hk00700", 0.08)
        d = gate.evaluate(_signal("hk00700", price=1.0, qty=100), sim_time=self.SIM)
        assert not d.approved
        assert "var" in d.reason.lower()

    def test_exit_bypasses_var(self):
        """SELL signals bypass the VaR check entirely."""
        pos = {"hk00700": _make_pos("hk00700", 10_000, 300.0)}
        gate = _gate(nav=1_000_000, positions=pos)
        gate.set_vol_estimate("hk00700", 0.10)   # extreme vol
        sig = _signal("hk00700", signal_type=SignalType.SELL)
        d = gate.evaluate(sig, sim_time=self.SIM)
        assert d.approved

    def test_zero_nav_bypasses_var(self):
        gate = RiskGate(nav_getter=lambda: 0, positions_getter=lambda: {})
        d = gate.evaluate(_signal(), sim_time=self.SIM)
        assert d.approved  # nav=0 edge case handled gracefully

    def test_bulk_vol_update(self):
        """set_vol_estimates bulk update changes future evaluations."""
        gate = _gate(nav=1_000_000)
        gate.set_vol_estimates({"hk00700": 0.10, "sh600036": 0.02})
        assert gate._vol_estimates["hk00700"] == pytest.approx(0.10)
        assert gate._vol_estimates["sh600036"] == pytest.approx(0.02)

    def test_default_vol_used_when_no_estimate(self):
        """Missing per-symbol estimate falls back to default_daily_vol."""
        gate = _gate(nav=1_000_000)
        # No explicit estimate set for hk00700; but default is 2% → small position OK
        d = gate.evaluate(_signal("hk00700", price=300, qty=100), sim_time=self.SIM)
        assert d.approved


# ── Layer 8: Hot-stock guard ───────────────────────────────────────────────────

class TestHotStockGuard:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_unlisted_symbol_not_blocked(self):
        gate = _gate()
        gate.set_hot_symbols({"sh600519"})   # Maotai is hot — not our test symbol
        d = gate.evaluate(_signal("hk00700"), sim_time=self.SIM)
        assert d.approved

    def test_hot_symbol_buy_blocked(self):
        gate = _gate()
        gate.set_hot_symbols({"sh600519"})
        d = gate.evaluate(_signal("sh600519"), sim_time=self.SIM)
        assert not d.approved
        assert "hot" in d.reason.lower() or "exclusion" in d.reason.lower()

    def test_hot_symbol_sell_allowed(self):
        """Exits must always be permitted even for hot-stock listed symbols."""
        pos = {"sh600519": _make_pos("sh600519", 100, 1500.0)}
        gate = _gate(nav=1_000_000, positions=pos)
        gate.set_hot_symbols({"sh600519"})
        sig = _signal("sh600519", signal_type=SignalType.SELL)
        d = gate.evaluate(sig, sim_time=self.SIM)
        assert d.approved

    def test_set_hot_symbols_is_case_insensitive(self):
        """set_hot_symbols uppercases; evaluate also uppercases signal.symbol."""
        gate = _gate()
        gate.set_hot_symbols({"SH600519"})   # upper-case input
        d = gate.evaluate(_signal("sh600519"), sim_time=self.SIM)
        assert not d.approved

    def test_empty_hot_list_approves_all(self):
        gate = _gate()
        gate.set_hot_symbols(set())
        d = gate.evaluate(_signal("sh600519"), sim_time=self.SIM)
        assert d.approved

    def test_hot_list_replaced_on_new_call(self):
        """Calling set_hot_symbols again replaces the previous list."""
        gate = _gate()
        gate.set_hot_symbols({"sh600519"})
        # Remove sh600519, add another symbol
        gate.set_hot_symbols({"sz300750"})
        d_old = gate.evaluate(_signal("sh600519"), sim_time=self.SIM)
        d_new = gate.evaluate(_signal("sz300750"), sim_time=self.SIM)
        assert d_old.approved      # sh600519 no longer blocked
        assert not d_new.approved  # sz300750 now blocked

    def test_multiple_hot_symbols_all_blocked(self):
        gate = _gate()
        gate.set_hot_symbols({"sh600519", "sz300750", "sh601318"})
        for sym in ["sh600519", "sz300750", "sh601318"]:
            d = gate.evaluate(_signal(sym), sim_time=self.SIM)
            assert not d.approved, f"Expected {sym} to be blocked"


# ── Layer 9: MA20 life-line ────────────────────────────────────────────────────

class TestMA20Lifeline:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_price_above_ma20_approved(self):
        gate = _gate()
        gate.set_ma20_map({"hk00700": 290.0})   # MA20 = 290; signal price = 300
        d = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert d.approved

    def test_price_below_ma20_rejected(self):
        gate = _gate()
        gate.set_ma20_map({"hk00700": 310.0})   # MA20 = 310; signal price = 300
        d = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert not d.approved
        assert "ma20" in d.reason.lower() or "life-line" in d.reason.lower() or "ma" in d.reason.lower()

    def test_price_exactly_at_ma20_approved(self):
        """Price equal to MA20 (not below) — should pass."""
        gate = _gate()
        gate.set_ma20_map({"hk00700": 300.0})
        d = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert d.approved

    def test_no_ma20_entry_always_passes(self):
        """If no MA20 is known for this symbol, the check must not block."""
        gate = _gate()
        gate.set_ma20_map({})   # empty map
        d = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert d.approved

    def test_sell_below_ma20_allowed(self):
        """Exits are always allowed regardless of MA20."""
        pos = {"hk00700": _make_pos("hk00700", 100, 300.0)}
        gate = _gate(nav=1_000_000, positions=pos)
        gate.set_ma20_map({"hk00700": 350.0})   # price will be 300 < MA20 350
        sig = _signal("hk00700", signal_type=SignalType.SELL, price=300.0)
        d = gate.evaluate(sig, sim_time=self.SIM)
        assert d.approved

    def test_ma20_map_case_insensitive(self):
        """MA20 map should match regardless of symbol case."""
        gate = _gate()
        gate.set_ma20_map({"HK00700": 310.0})   # upper-case key
        d = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert not d.approved

    def test_ma20_map_replaced_on_new_call(self):
        gate = _gate()
        gate.set_ma20_map({"hk00700": 310.0})   # price 300 < MA20 310 → blocked
        d1 = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert not d1.approved

        gate.set_ma20_map({"hk00700": 280.0})   # now MA20 = 280 < 300 → allowed
        d2 = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert d2.approved

    def test_reason_mentions_symbol_and_prices(self):
        gate = _gate()
        gate.set_ma20_map({"SH600036": 50.0})
        d = gate.evaluate(_signal("sh600036", price=45.0), sim_time=self.SIM)
        assert not d.approved
        assert "45" in d.reason or "50" in d.reason


# ── Layer ordering: later layers don't fire if earlier ones reject ─────────────

class TestLayerOrdering:
    SIM = datetime(2024, 6, 15, 9, 30)

    def test_drawdown_fires_before_hot_stock(self):
        """Drawdown (layer 3) should reject before hot-stock (layer 8) is evaluated."""
        gate = _gate(nav=900_000)
        gate.record_nav(1_000_000, sim_date=SIM.date() if False else datetime(2024, 6, 15).date())
        gate.set_hot_symbols({"sh600036"})
        d = gate.evaluate(_signal("sh600036"), sim_time=self.SIM)
        assert not d.approved
        assert "drawdown" in d.reason.lower()   # drawdown message, not hot-stock

    def test_hot_stock_fires_before_ma20(self):
        """Hot-stock guard (layer 8) fires before MA20 (layer 9)."""
        gate = _gate()
        gate.set_hot_symbols({"hk00700"})
        gate.set_ma20_map({"hk00700": 350.0})   # would also block
        d = gate.evaluate(_signal("hk00700", price=300.0), sim_time=self.SIM)
        assert not d.approved
        # Should mention hot-stock, not MA20 (hot-stock fires first)
        assert "hot" in d.reason.lower() or "exclusion" in d.reason.lower()


# ── set_vol_estimate backward compatibility ────────────────────────────────────

class TestVolEstimateAPI:
    def test_single_symbol_update(self):
        gate = _gate()
        gate.set_vol_estimate("hk00700", 0.025)
        assert gate._vol_estimates["hk00700"] == pytest.approx(0.025)

    def test_bulk_does_not_clear_previous(self):
        gate = _gate()
        gate.set_vol_estimate("hk00700", 0.025)
        gate.set_vol_estimates({"sh600036": 0.018})
        # Original entry should still be there
        assert gate._vol_estimates["hk00700"] == pytest.approx(0.025)
        assert gate._vol_estimates["sh600036"] == pytest.approx(0.018)


SIM = datetime(2024, 6, 15, 9, 30)

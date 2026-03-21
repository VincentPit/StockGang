"""
Tests for ATR-based position sizer and MACD strategy internals.
No external data or network calls required.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from myquant.models.bar import Bar, BarInterval
from myquant.models.signal import SignalType
from myquant.strategy.sizing import atr_position_size
from myquant.strategy.technical.macd_strategy import MACDStrategy, _ema_series, _macd

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_bar(symbol: str, close: float, idx: int = 0) -> Bar:
    ts = datetime(2024, 1, 1) + timedelta(days=idx)
    return Bar(symbol=symbol, ts=ts, interval=BarInterval.D1,
               open=close * 0.99, high=close * 1.01, low=close * 0.98, close=close)


# ── ATR position sizer ─────────────────────────────────────────────────────────

class TestATRPositionSize:
    def test_returns_at_least_one_lot(self):
        qty = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.015)
        assert qty >= 100
        assert qty % 100 == 0

    def test_higher_confidence_gives_larger_position(self):
        low  = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.015, confidence=0.51)
        high = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.015, confidence=0.95)
        assert high >= low

    def test_lower_atr_gives_larger_position(self):
        """Tighter volatility → can risk same budget on more shares."""
        calm   = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.005)
        choppy = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.030)
        assert calm >= choppy

    def test_max_position_cap_respected(self):
        # Extreme confidence + tiny ATR should still be capped at max_pos_pct
        qty = atr_position_size(
            nav=1_000_000,
            price=10.0,
            atr_pct=0.001,
            confidence=1.0,
            max_pos_pct=0.10,
        )
        notional = qty * 10.0
        assert notional <= 1_000_000 * 0.10 + 10   # allow 1 lot tolerance

    def test_zero_price_returns_lot_size(self):
        qty = atr_position_size(nav=1_000_000, price=0.0, atr_pct=0.015)
        assert qty == 100   # fallback default

    def test_zero_nav_returns_lot_size(self):
        qty = atr_position_size(nav=0.0, price=300.0, atr_pct=0.015)
        assert qty == 100

    def test_custom_lot_size(self):
        qty = atr_position_size(nav=1_000_000, price=50.0, atr_pct=0.015, lot_size=200)
        assert qty % 200 == 0

    def test_tiny_atr_floored_to_minimum(self):
        """atr_pct < 0.003 is floored so we don't get astronomically large positions."""
        qty_floored = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.0001)
        qty_normal  = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.003)
        # Both should be identical because atr is floored at 0.003
        assert qty_floored == qty_normal

    def test_larger_nav_gives_larger_position(self):
        small = atr_position_size(nav=100_000,   price=300.0, atr_pct=0.015)
        large = atr_position_size(nav=1_000_000, price=300.0, atr_pct=0.015)
        assert large >= small


# ── MACD internals ─────────────────────────────────────────────────────────────

class TestMACDIndicator:
    def test_returns_none_with_insufficient_data(self):
        m, s, h = _macd([100.0] * 5)          # need at least slow+signal=35 bars
        assert m is None and s is None and h is None

    def test_returns_values_with_sufficient_data(self):
        closes = [float(100 + i * 0.5) for i in range(40)]
        m, s, h = _macd(closes)
        assert m is not None
        assert s is not None
        assert h == pytest.approx(m - s, abs=1e-9)

    def test_histogram_is_macd_minus_signal(self):
        closes = [float(50 + (i % 7)) for i in range(50)]
        m, s, h = _macd(closes)
        if m is not None:
            assert h == pytest.approx(m - s, abs=1e-9)

    def test_ema_series_length_matches_input(self):
        values = list(range(1, 31))
        result = _ema_series(values, 10)
        assert len(result) == len(values)

    def test_ema_series_constant_input(self):
        values = [5.0] * 30
        result = _ema_series(values, 10)
        # After warmup, EMA of a constant should equal the constant
        assert result[-1] == pytest.approx(5.0, abs=0.01)

    def test_ema_series_insufficient_data_returns_nans(self):
        values = [1.0, 2.0, 3.0]
        result = _ema_series(values, 10)
        import math
        assert all(math.isnan(v) for v in result)


# ── MACD strategy (signal generation) ─────────────────────────────────────────

class TestMACDStrategy:
    SYMBOL = "hk00700"

    def _feed(self, closes: list[float], fast=12, slow=26, signal=9) -> list:
        strategy = MACDStrategy(
            "test_macd", [self.SYMBOL], fast=fast, slow=slow, signal=signal
        )
        signals = []
        for i, c in enumerate(closes):
            bar = _make_bar(self.SYMBOL, c, i)
            sig = strategy.on_bar(bar)
            if sig:
                signals.append(sig)
        return signals

    def test_no_signal_before_warmup(self):
        # 20 bars — below the slow+signal threshold (26+9=35)
        sigs = self._feed([float(100 + i) for i in range(20)])
        assert len(sigs) == 0

    def test_bull_crossover_generates_buy(self):
        # 40 flat bars set EMA baseline, accelerating decline creates negative
        # histogram (hist < 0 before settling), sharp recovery flips hist > 0.
        flat = [200.0] * 40
        down = [200.0 - i * 2 - i ** 2 * 0.2 for i in range(1, 21)]
        up   = [max(down[-1], 1.0) + i * 4 for i in range(60)]
        sigs = self._feed(flat + down + up)
        buys = [s for s in sigs if s.signal_type == SignalType.BUY]
        assert len(buys) >= 1

    def test_bear_crossover_generates_sell(self):
        # 40 flat bars set EMA baseline, accelerating rise creates positive
        # histogram, sharp decline flips hist < 0.
        flat  = [50.0] * 40
        up    = [50.0 + i * 2 + i ** 2 * 0.2 for i in range(1, 21)]
        down  = [max(up[-1] - i * 4, 1.0) for i in range(60)]
        sigs  = self._feed(flat + up + down)
        sells = [s for s in sigs if s.signal_type == SignalType.SELL]
        assert len(sells) >= 1

    def test_signal_has_correct_symbol(self):
        up   = [float(50 + i * 0.8) for i in range(30)]
        down = [float(up[-1] - i * 1.2) for i in range(30)]
        sigs = self._feed(up + down)
        for s in sigs:
            assert s.symbol == self.SYMBOL

    def test_no_signal_for_flat_prices(self):
        # Constant prices → MACD=0, signal=0, no crossover
        sigs = self._feed([100.0] * 60)
        assert len(sigs) == 0

    def test_min_hist_filter(self):
        """Signals should not fire when histogram magnitude is below min_hist."""
        down = [float(100 - i * 0.8) for i in range(30)]
        up   = [float(down[-1] + i * 0.01) for i in range(30)]  # tiny move
        strategy = MACDStrategy("test_minH", [self.SYMBOL], fast=12, slow=26, signal=9, min_hist=100.0)
        sigs = []
        for i, c in enumerate(down + up):
            bar = _make_bar(self.SYMBOL, max(c, 0.1), i)
            sig = strategy.on_bar(bar)
            if sig:
                sigs.append(sig)
        assert len(sigs) == 0

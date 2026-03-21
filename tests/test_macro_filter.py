"""
tests/test_macro_filter.py

Tests for:
  • MacroSnapshot.regime / signal_multiplier boundary conditions
  • FundamentalSnapshot.value_score formula
  • MacroFilter.filter_signal — all four layers:
      L1  RISK_OFF suppresses BUY; allows SELL/CLOSE
      L2  Fundamental quality gate blocks low-value buys
      L3  Confidence scaling via signal_multiplier
      L4  Strength upgrade on RISK_ON / downgrade on RISK_OFF
"""
from __future__ import annotations

from unittest.mock import MagicMock

from myquant.data.fetchers.fundamental_fetcher import FundamentalSnapshot
from myquant.data.fetchers.macro_fetcher import MacroSnapshot
from myquant.models.signal import Signal, SignalStrength, SignalType
from myquant.strategy.macro_filter import MacroFilter

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal(
    signal_type: SignalType = SignalType.BUY,
    symbol: str = "sh600519",
    confidence: float = 0.60,
    strength: SignalStrength = SignalStrength.NORMAL,
) -> Signal:
    return Signal(
        strategy_id="test_strategy",
        symbol=symbol,
        signal_type=signal_type,
        confidence=confidence,
        strength=strength,
        price=100.0,
        quantity=100,
    )


def _neutral_snap(**kwargs) -> MacroSnapshot:
    """Return a NEUTRAL MacroSnapshot, overridable by kwargs."""
    defaults = dict(vix=18.0, china_pmi_mfg=51.0, us_10y_yield=4.5, china_cpi_yoy=2.0)
    defaults.update(kwargs)
    return MacroSnapshot(**defaults)


def _good_fundamental(symbol: str = "sh600519") -> FundamentalSnapshot:
    """value_score well above default threshold of 15."""
    return FundamentalSnapshot(symbol=symbol, pe_ttm=10.0, roe=20.0, pb=1.0)


def _bad_fundamental(symbol: str = "sh600519") -> FundamentalSnapshot:
    """value_score below threshold — high PE, low ROE."""
    return FundamentalSnapshot(symbol=symbol, pe_ttm=50.0, roe=1.0, pb=10.0)


def _make_filter(snap: MacroSnapshot, fund: FundamentalSnapshot | None = None) -> MacroFilter:
    """Construct a MacroFilter whose fetchers are fully mocked."""
    mf = MacroFilter.__new__(MacroFilter)
    mf._snapshot = snap
    mf._min_value_score = 15.0

    if fund is not None:
        mock_fund_fetcher = MagicMock()
        mock_fund_fetcher.fetch.return_value = fund
        mf._fund_fetcher = mock_fund_fetcher
    else:
        mf._fund_fetcher = None

    return mf


# ─────────────────────────────────────────────────────────────────────────────
# MacroSnapshot.regime boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroSnapshotRegime:
    def test_vix_above_30_is_risk_off(self):
        snap = _neutral_snap(vix=30.1)
        assert snap.regime == "RISK_OFF"

    def test_vix_exactly_30_is_not_risk_off(self):
        snap = _neutral_snap(vix=30.0, china_pmi_mfg=50.0, us_10y_yield=4.5)
        # vix=30 does NOT trigger RISK_OFF (condition is vix > 30)
        assert snap.regime in ("NEUTRAL", "RISK_ON")

    def test_pmi_below_49_is_risk_off(self):
        snap = _neutral_snap(china_pmi_mfg=48.9)
        assert snap.regime == "RISK_OFF"

    def test_us_10y_above_5p5_is_risk_off(self):
        snap = _neutral_snap(us_10y_yield=5.6)
        assert snap.regime == "RISK_OFF"

    def test_risk_on_all_conditions_met(self):
        snap = MacroSnapshot(vix=15.0, china_pmi_mfg=52.0, us_10y_yield=4.0, china_cpi_yoy=2.0)
        assert snap.regime == "RISK_ON"

    def test_risk_on_blocked_by_high_vix(self):
        snap = MacroSnapshot(vix=20.0, china_pmi_mfg=52.0, us_10y_yield=4.0, china_cpi_yoy=2.0)
        assert snap.regime == "NEUTRAL"

    def test_neutral_baseline(self):
        snap = _neutral_snap()
        assert snap.regime == "NEUTRAL"

    def test_signal_multiplier_risk_on(self):
        snap = MacroSnapshot(vix=14.0, china_pmi_mfg=52.0, us_10y_yield=3.9, china_cpi_yoy=1.5)
        assert snap.regime == "RISK_ON"
        assert snap.signal_multiplier == 1.20

    def test_signal_multiplier_neutral(self):
        assert _neutral_snap().signal_multiplier == 1.00

    def test_signal_multiplier_risk_off(self):
        snap = _neutral_snap(vix=35.0)
        assert snap.signal_multiplier == 0.50

    def test_cny_depreciation_pressure_true(self):
        snap = _neutral_snap()
        snap.usdcny = 7.30
        assert snap.cny_depreciation_pressure is True

    def test_cny_depreciation_pressure_false(self):
        snap = _neutral_snap()
        snap.usdcny = 7.20
        assert snap.cny_depreciation_pressure is False


# ─────────────────────────────────────────────────────────────────────────────
# FundamentalSnapshot.value_score formula
# ─────────────────────────────────────────────────────────────────────────────

class TestFundamentalSnapshot:
    def test_value_score_high_quality(self):
        """Low PE, high ROE, low PB → high score."""
        snap = FundamentalSnapshot(symbol="sh600519", pe_ttm=10.0, roe=20.0, pb=1.0)
        # pe_score = 100 - 20 = 80; roe_score = 60; pb_score = 100 - 8 = 92
        expected = 80.0 * 0.40 + 60.0 * 0.40 + 92.0 * 0.20
        assert abs(snap.value_score - expected) < 0.01

    def test_value_score_low_quality(self):
        """High PE, low ROE, high PB → low score."""
        snap = FundamentalSnapshot(symbol="sh600519", pe_ttm=50.0, roe=1.0, pb=10.0)
        assert snap.value_score < 15.0

    def test_value_score_clamped_at_zero(self):
        """Extreme values shouldn't go negative."""
        snap = FundamentalSnapshot(symbol="sh600519", pe_ttm=100.0, roe=0.0, pb=20.0)
        assert snap.value_score >= 0.0

    def test_value_score_clamped_at_100(self):
        """Best-case values shouldn't exceed 100."""
        snap = FundamentalSnapshot(symbol="sh600519", pe_ttm=0.0, roe=100.0, pb=0.0)
        assert snap.value_score <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — RISK_OFF regime gate
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroFilterRiskOff:
    def setup_method(self):
        self.snap = _neutral_snap(vix=35.0)  # RISK_OFF
        assert self.snap.regime == "RISK_OFF"
        self.mf = _make_filter(self.snap, fund=_good_fundamental())

    def test_buy_suppressed_in_risk_off(self):
        result = self.mf.filter_signal(_make_signal(SignalType.BUY))
        assert result is None

    def test_short_suppressed_in_risk_off(self):
        result = self.mf.filter_signal(_make_signal(SignalType.SHORT))
        assert result is None

    def test_sell_passes_in_risk_off(self):
        result = self.mf.filter_signal(_make_signal(SignalType.SELL))
        assert result is not None

    def test_close_long_passes_in_risk_off(self):
        result = self.mf.filter_signal(_make_signal(SignalType.CLOSE_LONG))
        assert result is not None

    def test_close_short_passes_in_risk_off(self):
        result = self.mf.filter_signal(_make_signal(SignalType.CLOSE_SHORT))
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Fundamental quality gate
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroFilterFundamental:
    def setup_method(self):
        self.snap = _neutral_snap()  # NEUTRAL — passes L1

    def test_buy_suppressed_by_poor_fundamentals(self):
        mf = _make_filter(self.snap, fund=_bad_fundamental())
        result = mf.filter_signal(_make_signal(SignalType.BUY))
        assert result is None

    def test_buy_passes_with_good_fundamentals(self):
        mf = _make_filter(self.snap, fund=_good_fundamental())
        result = mf.filter_signal(_make_signal(SignalType.BUY))
        assert result is not None

    def test_sell_not_filtered_by_fundamentals(self):
        mf = _make_filter(self.snap, fund=_bad_fundamental())
        result = mf.filter_signal(_make_signal(SignalType.SELL))
        assert result is not None

    def test_fundamental_gate_disabled_when_fund_fetcher_none(self):
        """min_value_score check still runs but no fetcher → no fundamental gate."""
        mf = _make_filter(self.snap, fund=None)
        # Should pass even though there's no fundamental data
        result = mf.filter_signal(_make_signal(SignalType.BUY))
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Confidence scaling
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroFilterConfidenceScaling:
    def test_neutral_multiplier_keeps_confidence(self):
        snap = _neutral_snap()  # multiplier = 1.00
        mf = _make_filter(snap, fund=_good_fundamental())
        sig = _make_signal(confidence=0.70)
        result = mf.filter_signal(sig)
        assert result is not None
        assert abs(result.confidence - 0.70) < 0.001

    def test_risk_on_multiplier_boosts_confidence(self):
        snap = MacroSnapshot(vix=14.0, china_pmi_mfg=52.0, us_10y_yield=3.9, china_cpi_yoy=1.5)
        assert snap.signal_multiplier == 1.20
        mf = _make_filter(snap, fund=_good_fundamental())
        sig = _make_signal(confidence=0.70)
        result = mf.filter_signal(sig)
        assert result is not None
        assert abs(result.confidence - 0.84) < 0.001  # 0.70 × 1.20

    def test_confidence_clamped_at_1(self):
        snap = MacroSnapshot(vix=14.0, china_pmi_mfg=52.0, us_10y_yield=3.9, china_cpi_yoy=1.5)
        mf = _make_filter(snap, fund=_good_fundamental())
        sig = _make_signal(confidence=0.95)  # 0.95 × 1.20 = 1.14 → clamped
        result = mf.filter_signal(sig)
        assert result is not None
        assert result.confidence <= 1.0

    def test_sell_confidence_also_scaled(self):
        snap = MacroSnapshot(vix=14.0, china_pmi_mfg=52.0, us_10y_yield=3.9, china_cpi_yoy=1.5)
        mf = _make_filter(snap)
        sig = _make_signal(signal_type=SignalType.SELL, confidence=0.60)
        result = mf.filter_signal(sig)
        assert result is not None
        assert abs(result.confidence - 0.72) < 0.001  # 0.60 × 1.20


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Strength upgrade / downgrade
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroFilterStrength:
    def test_risk_on_upgrades_normal_to_strong(self):
        snap = MacroSnapshot(vix=14.0, china_pmi_mfg=52.0, us_10y_yield=3.9, china_cpi_yoy=1.5)
        mf = _make_filter(snap, fund=_good_fundamental())
        sig = _make_signal(strength=SignalStrength.NORMAL)
        result = mf.filter_signal(sig)
        assert result is not None
        assert result.strength == SignalStrength.STRONG

    def test_risk_on_does_not_downgrade_strong(self):
        snap = MacroSnapshot(vix=14.0, china_pmi_mfg=52.0, us_10y_yield=3.9, china_cpi_yoy=1.5)
        mf = _make_filter(snap, fund=_good_fundamental())
        sig = _make_signal(strength=SignalStrength.STRONG)
        result = mf.filter_signal(sig)
        assert result is not None
        assert result.strength == SignalStrength.STRONG

    def test_risk_off_sell_set_to_weak(self):
        snap = _neutral_snap(vix=35.0)  # RISK_OFF
        mf = _make_filter(snap)
        sig = _make_signal(signal_type=SignalType.SELL, strength=SignalStrength.STRONG)
        result = mf.filter_signal(sig)
        assert result is not None
        assert result.strength == SignalStrength.WEAK

    def test_neutral_regime_preserves_strength(self):
        snap = _neutral_snap()
        mf = _make_filter(snap, fund=_good_fundamental())
        for strength in (SignalStrength.WEAK, SignalStrength.NORMAL, SignalStrength.STRONG):
            sig = _make_signal(strength=strength)
            result = mf.filter_signal(sig)
            assert result is not None
            assert result.strength == strength

    def test_metadata_contains_macro_fields(self):
        snap = _neutral_snap()
        mf = _make_filter(snap, fund=_good_fundamental())
        result = mf.filter_signal(_make_signal())
        assert result is not None
        for key in ("macro_regime", "macro_pmi", "macro_vix", "macro_us10y",
                    "macro_usdcny", "conf_multiplier"):
            assert key in result.metadata

    def test_original_signal_not_mutated(self):
        snap = MacroSnapshot(vix=14.0, china_pmi_mfg=52.0, us_10y_yield=3.9, china_cpi_yoy=1.5)
        mf = _make_filter(snap, fund=_good_fundamental())
        orig = _make_signal(confidence=0.70, strength=SignalStrength.NORMAL)
        _ = mf.filter_signal(orig)
        # Original should be unchanged
        assert orig.confidence == 0.70
        assert orig.strength == SignalStrength.NORMAL

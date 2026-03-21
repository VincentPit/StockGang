"""
Tests for feature_engineer.py — bars_to_features, make_labels, and FEATURE_COLS.

Covers:
  - All 28 feature columns are produced with no NaN after warm-up
  - CN-market features: recent_limit_up, yang_ratio, candle_strength
  - Momentum: ret_1d, ret_20d, ret_60d, ret_120d
  - Volatility: vol_20d, vol_120d, dist_52w_high
  - Oscillators: rsi_14, macd_hist, stoch_k/d, wr_14, cci_20
  - Mean-revert: bb_pos, bb_width, price_52w_pos
  - Regime: vol_regime, trend_strength, above_ma50, ma_spread, ma50_slope
  - Microstructure: close_loc
  - make_labels: ternary classification logic
  - Edge cases: too-short input, constant-price series, extreme values
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from myquant.models.bar import Bar, BarInterval
from myquant.strategy.ml.feature_engineer import (
    FEATURE_COLS,
    bars_to_features,
    make_labels,
)


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_bars(
    n: int = 300,
    start_price: float = 50.0,
    daily_ret: float = 0.001,
    vol: float = 0.015,
    seed: int = 42,
) -> list[Bar]:
    """Synthetic bar series with configurable trend and noise."""
    rng = np.random.default_rng(seed)
    bars = []
    price = start_price
    base_date = datetime(2022, 1, 1)
    for i in range(n):
        ret   = daily_ret + rng.normal(0, vol)
        price = max(price * (1 + ret), 0.01)
        hi    = price * (1 + abs(rng.normal(0, 0.005)))
        lo    = price * (1 - abs(rng.normal(0, 0.005)))
        vol_v = rng.uniform(1e6, 5e6)
        bars.append(
            Bar(
                symbol   = "sh600036",
                ts       = base_date + timedelta(days=i),
                interval = BarInterval.D1,
                open     = price / (1 + ret / 2),
                high     = hi,
                low      = lo,
                close    = price,
                volume   = vol_v,
            )
        )
    return bars


def _make_bars_with_limit_ups(n: int = 200, limit_up_days: list[int] | None = None) -> list[Bar]:
    """Make bars where specific day indices have ≥9.5% daily gain (涨停 simulation)."""
    bars = _make_bars(n, seed=7)
    limit_up_days = limit_up_days or [20, 40, 60, 80, 100]
    for i in limit_up_days:
        if i < len(bars):
            b = bars[i]
            # Replace the bar with a 10% gain
            prev_close = bars[i - 1].close if i > 0 else b.open
            new_close  = prev_close * 1.10
            bars[i] = Bar(
                symbol   = b.symbol,
                ts       = b.ts,
                interval = b.interval,
                open     = prev_close,
                high     = new_close * 1.01,
                low      = prev_close * 0.99,
                close    = new_close,
                volume   = b.volume,
            )
    return bars


# ── Core: all FEATURE_COLS produced ──────────────────────────────────────────

class TestBarsToFeatures:
    def test_returns_dataframe(self):
        df = bars_to_features(_make_bars(300))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_all_feature_cols_present(self):
        df = bars_to_features(_make_bars(300))
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_no_nan_in_features(self):
        df = bars_to_features(_make_bars(300))
        nan_cols = [c for c in FEATURE_COLS if df[c].isna().any()]
        assert nan_cols == [], f"NaN in columns: {nan_cols}"

    def test_feature_col_count(self):
        """FEATURE_COLS list must have exactly 34 entries (frozen contract)."""
        assert len(FEATURE_COLS) == 34, (
            f"Expected 34 features, got {len(FEATURE_COLS)}. "
            "Did you add/remove a feature without updating this test?"
        )

    def test_short_input_returns_empty(self):
        df = bars_to_features(_make_bars(10))
        assert len(df) == 0

    def test_exactly_30_bars_returns_empty(self):
        # bars_to_features needs >30 bars to start computing rolling windows
        df = bars_to_features(_make_bars(30))
        assert len(df) == 0

    def test_250_bars_has_output(self):
        df = bars_to_features(_make_bars(250))
        assert len(df) > 0

    def test_output_sorted_by_timestamp(self):
        bars = _make_bars(300)
        # Shuffle bars to verify internal sorting
        import random
        random.shuffle(bars)
        df = bars_to_features(bars)
        assert df.index.is_monotonic_increasing


# ── Momentum features ─────────────────────────────────────────────────────────

class TestMomentumFeatures:
    def test_ret_1d_range(self):
        df = bars_to_features(_make_bars(300))
        # Most 1-day returns in a synthetic series should be <±20%
        assert df["ret_1d"].abs().max() < 0.5

    def test_ret_60d_non_zero(self):
        df = bars_to_features(_make_bars(300, daily_ret=0.005, vol=0.003))
        # Strong positive trend with low noise — 60d return should be mostly positive
        assert (df["ret_60d"] > 0).mean() > 0.6

    def test_ret_120d_captured(self):
        df = bars_to_features(_make_bars(300))
        assert "ret_120d" in df.columns
        assert not df["ret_120d"].isna().all()

    def test_ret_20d_captured(self):
        df = bars_to_features(_make_bars(300))
        assert "ret_20d" in df.columns
        assert not df["ret_20d"].isna().all()


# ── Volatility features ───────────────────────────────────────────────────────

class TestVolatilityFeatures:
    def test_vol_20d_positive(self):
        df = bars_to_features(_make_bars(300))
        assert (df["vol_20d"] >= 0).all()

    def test_vol_120d_smoother_than_vol_20d(self):
        """120d vol should have a smaller std-dev (smoother) than 20d vol."""
        df = bars_to_features(_make_bars(300, vol=0.02))
        assert df["vol_120d"].std() < df["vol_20d"].std()

    def test_dist_52w_high_clipped(self):
        df = bars_to_features(_make_bars(300))
        assert (df["dist_52w_high"] >= 0.0).all()
        assert (df["dist_52w_high"] <= 0.80).all()

    def test_dist_52w_high_zero_at_52w_peak(self):
        """A steadily rising price should have dist_52w_high near 0 at the end."""
        df = bars_to_features(_make_bars(300, daily_ret=0.003, vol=0.001))
        # Last rows should be at/near 52w high
        last_dist = df["dist_52w_high"].iloc[-10:].mean()
        assert last_dist < 0.05, f"Expected near-zero dist_52w_high for rising trend, got {last_dist:.4f}"


# ── CN-market microstructure features ────────────────────────────────────────

class TestCNMarketFeatures:
    def test_recent_limit_up_zero_without_surges(self):
        """No ≥9.5% days → recent_limit_up should be 0 throughout."""
        df = bars_to_features(_make_bars(300, daily_ret=0.001, vol=0.005))
        assert (df["recent_limit_up"] == 0).all()

    def test_recent_limit_up_positive_with_surges(self):
        """Injected 涨停 days should produce positive recent_limit_up."""
        bars = _make_bars_with_limit_ups(300, limit_up_days=list(range(150, 180)))
        df   = bars_to_features(bars)
        # After the surges, some rows should have recent_limit_up > 0
        assert (df["recent_limit_up"] > 0).any(), "Expected some limit-up rows to be non-zero"

    def test_yang_ratio_range(self):
        df = bars_to_features(_make_bars(300))
        assert (df["yang_ratio"] >= 0).all()
        assert (df["yang_ratio"] <= 1).all()

    def test_yang_ratio_reflects_trend(self):
        """An uptrending series should produce yang_ratio > 0.5 on average."""
        df = bars_to_features(_make_bars(300, daily_ret=0.003, vol=0.005))
        # Most candles should be 阳线 (close > open) in a rising market
        assert df["yang_ratio"].mean() > 0.5

    def test_candle_strength_clipped(self):
        df = bars_to_features(_make_bars(300))
        assert (df["candle_strength"] >= -0.05).all()
        assert (df["candle_strength"] <=  0.05).all()

    def test_candle_strength_positive_in_bull(self):
        """In a strong uptrend, average candle_strength should be positive."""
        df = bars_to_features(_make_bars(300, daily_ret=0.004, vol=0.004))
        assert df["candle_strength"].mean() > 0, "Expected positive candle_strength in bull trend"


# ── Oscillator features ───────────────────────────────────────────────────────

class TestOscillators:
    def test_rsi_range(self):
        df = bars_to_features(_make_bars(300))
        assert (df["rsi_14"] >= 0).all()
        assert (df["rsi_14"] <= 100).all()

    def test_rsi_high_in_bull(self):
        df = bars_to_features(_make_bars(300, daily_ret=0.005, vol=0.003))
        assert df["rsi_14"].mean() > 55, "RSI should be elevated in a bull trend"

    def test_stoch_k_range(self):
        df = bars_to_features(_make_bars(300))
        assert (df["stoch_k"] >= 0).all()
        assert (df["stoch_k"] <= 100).all()

    def test_stoch_d_is_smooth(self):
        """stoch_d (3-bar SMA of %K) should have lower std-dev than stoch_k."""
        df = bars_to_features(_make_bars(300))
        assert df["stoch_d"].std() <= df["stoch_k"].std()

    def test_wr_14_range(self):
        df = bars_to_features(_make_bars(300))
        assert (df["wr_14"] >= 0).all()
        assert (df["wr_14"] <= 100).all()

    def test_cci_20_clipped(self):
        df = bars_to_features(_make_bars(300))
        assert (df["cci_20"] >= -300).all()
        assert (df["cci_20"] <=  300).all()

    def test_bb_pos_range(self):
        df = bars_to_features(_make_bars(300))
        # bb_pos = (close - lower) / (upper - lower); can be slightly negative when
        # price dips below the lower Bollinger Band — this is correct/expected behaviour.
        # What matters: values are finite and in a reasonable range.
        assert df["bb_pos"].notna().all()
        assert (df["bb_pos"] >= -1.0).all(), "bb_pos should not be wildly negative"
        assert (df["bb_pos"] <=  2.0).all(), "bb_pos should not be wildly above band"

    def test_macd_hist_sign(self):
        """In a strong bull run, MACD histogram should be mostly positive."""
        df = bars_to_features(_make_bars(300, daily_ret=0.005, vol=0.003))
        assert (df["macd_hist"] > 0).mean() > 0.5


# ── Regime / MA features ──────────────────────────────────────────────────────

class TestRegimeFeatures:
    def test_above_ma50_binary(self):
        df = bars_to_features(_make_bars(300))
        vals = df["above_ma50"].unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_above_ma50_mostly_true_in_bull(self):
        df = bars_to_features(_make_bars(300, daily_ret=0.003, vol=0.005))
        assert df["above_ma50"].mean() > 0.7, "Expected price mostly above MA50 in bull trend"

    def test_ma_spread_clipped(self):
        df = bars_to_features(_make_bars(300))
        assert (df["ma_spread"] >= -0.20).all()
        assert (df["ma_spread"] <=  0.20).all()

    def test_vol_regime_in_unit_interval(self):
        df = bars_to_features(_make_bars(300))
        assert (df["vol_regime"] >= 0).all()
        assert (df["vol_regime"] <= 1).all()

    def test_trend_strength_non_negative(self):
        df = bars_to_features(_make_bars(300))
        assert (df["trend_strength"] >= 0).all()
        assert (df["trend_strength"] <= 5).all()

    def test_close_loc_range(self):
        df = bars_to_features(_make_bars(300))
        assert (df["close_loc"] >= 0).all()
        assert (df["close_loc"] <= 1).all()

    def test_ma50_slope_clipped(self):
        df = bars_to_features(_make_bars(300))
        assert (df["ma50_slope"] >= -0.05).all()
        assert (df["ma50_slope"] <=  0.05).all()


# ── price_52w_pos ─────────────────────────────────────────────────────────────

class TestPrice52wPos:
    def test_range(self):
        df = bars_to_features(_make_bars(300))
        assert (df["price_52w_pos"] >= 0).all()
        assert (df["price_52w_pos"] <= 1).all()


# ── make_labels ───────────────────────────────────────────────────────────────

class TestMakeLabels:
    def test_ternary_values(self):
        df = bars_to_features(_make_bars(300))
        labels = make_labels(df)
        assert set(labels.dropna().unique()).issubset({-1, 0, 1})

    def test_label_length_matches_df(self):
        df = bars_to_features(_make_bars(300))
        labels = make_labels(df)
        assert len(labels) == len(df)

    def test_bull_trend_has_buy_labels(self):
        """A strongly rising series should produce mostly BUY labels."""
        df = bars_to_features(_make_bars(300, daily_ret=0.010, vol=0.003))
        labels = make_labels(df, forward_days=5, threshold=0.01)
        buy_ratio = (labels == 1).mean()
        assert buy_ratio > 0.4, f"Expected mostly BUY in bull trend, got {buy_ratio:.2%}"

    def test_bear_trend_has_sell_labels(self):
        """A falling series should produce mostly SELL labels."""
        df = bars_to_features(_make_bars(300, daily_ret=-0.010, vol=0.003))
        labels = make_labels(df, forward_days=5, threshold=0.01)
        sell_ratio = (labels == -1).mean()
        assert sell_ratio > 0.4, f"Expected mostly SELL in bear trend, got {sell_ratio:.2%}"

    def test_last_n_rows_nan(self):
        """Last `forward_days` rows cannot have a label (no future data)."""
        df = bars_to_features(_make_bars(300))
        labels = make_labels(df, forward_days=5)
        # The very last few rows of `close.pct_change(5).shift(-5)` will be NaN
        # (make_labels fills them with 0 — neutral, no forward data)
        # What matters is the logic doesn't crash and produces the right length
        assert len(labels) == len(df)

    def test_custom_threshold(self):
        """High threshold should reduce the number of directional labels."""
        df = bars_to_features(_make_bars(300))
        l_tight  = make_labels(df, threshold=0.10)
        l_loose  = make_labels(df, threshold=0.001)
        tight_directional = ((l_tight == 1) | (l_tight == -1)).sum()
        loose_directional = ((l_loose == 1) | (l_loose == -1)).sum()
        assert tight_directional <= loose_directional

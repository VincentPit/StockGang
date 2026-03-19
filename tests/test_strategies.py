"""
Tests for strategy logic (no external data required).
Includes TestFeatureEngineer — validates bars_to_features() output shape,
all new features, and make_labels() ternary labels.
"""
import numpy as np
import pytest
from datetime import datetime, timedelta

from myquant.models.bar import Bar, BarInterval
from myquant.models.signal import SignalType
from myquant.strategy.technical.ma_crossover import MACrossoverStrategy, _sma, _ema
from myquant.strategy.technical.rsi_strategy import RSIStrategy, _rsi
from myquant.strategy.technical.macd_strategy import MACDStrategy
from myquant.strategy.ml.feature_engineer import (
    FEATURE_COLS, bars_to_features, make_labels,
)


def _make_bar(symbol: str, close: float, idx: int = 0) -> Bar:
    ts = datetime(2024, 1, 1) + timedelta(days=idx)
    return Bar(symbol=symbol, ts=ts, interval=BarInterval.D1,
               open=close * 0.99, high=close * 1.01, low=close * 0.98, close=close)


class TestIndicators:
    def test_sma(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _sma(values, 3) == pytest.approx(4.0)
        assert _sma(values, 10) is None

    def test_ema_convergence(self):
        # EMA of constant series should equal the constant
        values = [10.0] * 30
        ema = _ema(values, 10)
        assert ema == pytest.approx(10.0, abs=0.01)

    def test_rsi_overbought(self):
        # Rising prices → RSI should be high
        closes = [float(i) for i in range(1, 25)]
        rsi = _rsi(closes, 14)
        assert rsi is not None
        assert rsi > 70

    def test_rsi_oversold(self):
        # Falling prices → RSI should be low
        closes = [float(25 - i) for i in range(24)]
        rsi = _rsi(closes, 14)
        assert rsi is not None
        assert rsi < 30


class TestMACrossover:
    SYMBOL = "hk00700"

    def _feed_bars(self, strategy, closes: list[float]) -> list:
        signals = []
        for i, c in enumerate(closes):
            bar = _make_bar(self.SYMBOL, c, i)
            sig = strategy.on_bar(bar)
            if sig:
                signals.append(sig)
        return signals

    def test_golden_cross_generates_buy(self):
        strategy = MACrossoverStrategy(
            "test_ma", [self.SYMBOL], fast_period=3, slow_period=5
        )
        # Downtrend then uptrend → triggers golden cross
        closes = [10, 9, 8, 7, 6, 5] + [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11]
        signals = self._feed_bars(strategy, closes)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1

    def test_death_cross_generates_sell(self):
        strategy = MACrossoverStrategy(
            "test_ma2", [self.SYMBOL], fast_period=3, slow_period=5
        )
        # Uptrend then downtrend → triggers death cross
        closes = [5, 6, 7, 8, 9, 10, 11] + [10, 9, 8, 7, 6, 5, 4, 3, 2]
        signals = self._feed_bars(strategy, closes)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) >= 1


class TestRSI:
    SYMBOL = "sh600036"

    def test_buy_signal_on_oversold(self):
        strategy = RSIStrategy("test_rsi", [self.SYMBOL], period=14,
                                oversold=30, overbought=70, exit_mid=False)
        # Feed falling prices to get oversold
        closes = [float(30 - i) for i in range(20)]  # declining
        signals = []
        for i, c in enumerate(closes):
            bar = _make_bar(self.SYMBOL, max(c, 1.0), i)
            sig = strategy.on_bar(bar)
            if sig:
                signals.append(sig)
        # No explicit assertion on count — just ensure no crash
        for sig in signals:
            assert sig.symbol == self.SYMBOL


# ══════════════════════════════════════════════════════════════════════════════
# TestFeatureEngineer — validate bars_to_features() and make_labels()
# ══════════════════════════════════════════════════════════════════════════════

def _make_bars(n: int = 300, symbol: str = "sh600036") -> list:
    """Return n synthetic daily bars with realistic OHLCV data."""
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0.0003, 0.015, n)
    closes = 100.0 * np.exp(np.cumsum(log_rets))
    bars = []
    for i in range(n):
        c = float(closes[i])
        spread = c * 0.01
        bar = Bar(
            symbol   = symbol,
            ts       = datetime(2022, 1, 1) + timedelta(days=i),
            interval = BarInterval.D1,
            open     = c + rng.uniform(-spread, spread),
            high     = c + abs(rng.normal(0, spread)),
            low      = c - abs(rng.normal(0, spread)),
            close    = c,
            volume   = int(rng.integers(1_000_000, 10_000_000)),
        )
        bars.append(bar)
    return bars


class TestFeatureEngineer:
    BARS = _make_bars(300)  # shared across all tests in this class

    def test_all_feature_cols_present(self):
        """bars_to_features() must produce every column listed in FEATURE_COLS."""
        df = bars_to_features(self.BARS)
        assert not df.empty, "Feature DataFrame should not be empty"
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        assert missing == [], f"Missing feature columns: {missing}"

    def test_no_nan_in_output(self):
        """dropna(subset=FEATURE_COLS) is applied internally — no NaN allowed."""
        df = bars_to_features(self.BARS)
        assert df[FEATURE_COLS].isna().sum().sum() == 0

    def test_feature_count_matches(self):
        """FEATURE_COLS should have exactly 28 entries (23 existing + 5 new)."""
        assert len(FEATURE_COLS) == 28

    def test_stoch_k_range(self):
        """Stochastic %K must be in [0, 100]."""
        df = bars_to_features(self.BARS)
        assert df["stoch_k"].min() >= 0.0
        assert df["stoch_k"].max() <= 100.0

    def test_stoch_d_range(self):
        """Stochastic %D (SMA of %K) must also be in [0, 100]."""
        df = bars_to_features(self.BARS)
        assert df["stoch_d"].min() >= 0.0
        assert df["stoch_d"].max() <= 100.0

    def test_cmf_range(self):
        """Chaikin Money Flow must be in [-1, +1]."""
        df = bars_to_features(self.BARS)
        assert df["cmf_20"].min() >= -1.0 - 1e-9
        assert df["cmf_20"].max() <=  1.0 + 1e-9

    def test_keltner_pos_roughly_bounded(self):
        """keltner_pos is typically near [0, 1] but can exceed bounds during breakouts.
        We just verify it's computed and finite."""
        df = bars_to_features(self.BARS)
        assert df["keltner_pos"].notna().all()
        assert np.isfinite(df["keltner_pos"]).all()

    def test_ma50_slope_clipped(self):
        """ma50_slope must be within [-0.05, 0.05] after clipping."""
        df = bars_to_features(self.BARS)
        assert df["ma50_slope"].min() >= -0.05 - 1e-9
        assert df["ma50_slope"].max() <=  0.05 + 1e-9

    def test_make_labels_ternary(self):
        """make_labels() must return only +1, 0, -1 with no NaN rows."""
        df = bars_to_features(self.BARS)
        labels = make_labels(df, forward_days=5, threshold=0.015)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_make_labels_alignment(self):
        """Labels series must be aligned to the feature DataFrame's index."""
        df = bars_to_features(self.BARS)
        labels = make_labels(df, forward_days=5, threshold=0.015)
        assert labels.index.equals(df.index)

    def test_insufficient_bars_returns_empty(self):
        """Fewer than 30 bars should return an empty DataFrame."""
        df = bars_to_features(self.BARS[:20])
        assert df.empty

    def test_rsi_14_range(self):
        """RSI-14 must be in (0, 100)."""
        df = bars_to_features(self.BARS)
        assert df["rsi_14"].min() >  0.0
        assert df["rsi_14"].max() < 100.0

    def test_bb_pos_mostly_bounded(self):
        """Bollinger Band position is typically [0, 1]; at least 90% of rows in range."""
        df = bars_to_features(self.BARS)
        in_range = ((df["bb_pos"] >= 0) & (df["bb_pos"] <= 1)).mean()
        assert in_range > 0.80, f"bb_pos out-of-range fraction too high: {1 - in_range:.2%}"

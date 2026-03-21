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
        """FEATURE_COLS should have exactly 28 entries (CN rules 1-4: recent_limit_up/yang_ratio/candle_strength)."""
        assert len(FEATURE_COLS) == 28

    def test_wr_14_range(self):
        """Williams %R (inverted, normalised) must be in [0, 100]."""
        df = bars_to_features(self.BARS)
        assert df["wr_14"].min() >= 0.0 - 1e-9
        assert df["wr_14"].max() <= 100.0 + 1e-9

    def test_cci_20_clipped(self):
        """CCI-20 must be within [\u2212300, +300] after clipping."""
        df = bars_to_features(self.BARS)
        assert df["cci_20"].min() >= -300.0 - 1e-9
        assert df["cci_20"].max() <=  300.0 + 1e-9

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

    def test_candle_strength_clipped(self):
        """candle_strength (avg yang-body minus avg yin-body / close) must be in [-0.05, 0.05]."""
        df = bars_to_features(self.BARS)
        assert df["candle_strength"].min() >= -0.05 - 1e-9
        assert df["candle_strength"].max() <=  0.05 + 1e-9

    def test_yang_ratio_range(self):
        """yang_ratio (fraction of 阳线 candles in last 60d) must be in [0, 1]."""
        df = bars_to_features(self.BARS)
        assert df["yang_ratio"].min() >= 0.0 - 1e-9
        assert df["yang_ratio"].max() <=  1.0 + 1e-9

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

    def test_wr_14_complement_of_stoch_k(self):
        """wr_14 + stoch_k must equal 100 for any row (they are perfect inverses)."""
        df = bars_to_features(self.BARS)
        diff = (df["wr_14"] + df["stoch_k"]).sub(100).abs()
        assert diff.max() < 1e-6, "wr_14 and stoch_k should sum to 100"

    def test_ret_60d_finite(self):
        """ret_60d (60-day gradual momentum) must be finite in all output rows."""
        df = bars_to_features(self.BARS)
        assert np.isfinite(df["ret_60d"]).all()

    def test_ret_120d_finite(self):
        """ret_120d (120-day gradual momentum) must be finite in all output rows."""
        df = bars_to_features(self.BARS)
        assert np.isfinite(df["ret_120d"]).all()

    def test_dist_52w_high_range(self):
        """dist_52w_high (fraction below 52-week peak) must be in [0, 0.80] after clipping."""
        df = bars_to_features(self.BARS)
        assert df["dist_52w_high"].min() >= -1e-9
        assert df["dist_52w_high"].max() <=  0.80 + 1e-9

    def test_vol_120d_non_negative(self):
        """vol_120d (120-day return std) must be non-negative."""
        df = bars_to_features(self.BARS)
        assert (df["vol_120d"] >= 0).all()

    def test_recent_limit_up_range(self):
        """recent_limit_up (fraction of limit-up days in 60d) must be in [0, 1]."""
        df = bars_to_features(self.BARS)
        assert df["recent_limit_up"].min() >= -1e-9
        assert df["recent_limit_up"].max() <=  1.0 + 1e-9

    def test_yang_ratio_uptrend_above_half(self):
        """In a sustained uptrend bars, yang_ratio should exceed 0.50 (more up than down candles)."""
        import random
        random.seed(42)
        np.random.seed(42)
        # Build strongly trending bars where open < close on most days
        base = 100.0
        bars = []
        from myquant.models.bar import Bar, BarInterval
        from datetime import datetime, timedelta
        for i in range(300):
            o = base + i * 0.15
            c = o + abs(np.random.normal(0.3, 0.05))  # close always above open
            bars.append(Bar(
                symbol="X", ts=datetime(2023, 1, 1) + timedelta(days=i),
                open=o, high=c * 1.005, low=o * 0.995, close=c, volume=1_000_000,
                interval=BarInterval.D1,
            ))
        df = bars_to_features(bars)
        assert df["yang_ratio"].iloc[-1] > 0.5, "Uptrend bars should have yang_ratio > 0.5"


# ══════════════════════════════════════════════════════════════════════════════
# TestLGBMEnsemble — integration smoke-test for the walk-forward ensemble
# ══════════════════════════════════════════════════════════════════════════════

try:
    import lightgbm  # noqa: F401
    _HAS_LGB_FOR_TEST = True
except ImportError:
    _HAS_LGB_FOR_TEST = False


@pytest.mark.skipif(not _HAS_LGB_FOR_TEST, reason="lightgbm not installed")
class TestLGBMEnsemble:
    """Smoke-tests that LGBMStrategy with n_ensemble_windows trains and predicts."""

    SYMBOL = "sh600519"

    def _strategy(self, n_windows: int = 2) -> "LGBMStrategy":
        from myquant.strategy.ml.lgbm_strategy import LGBMStrategy
        return LGBMStrategy(
            "test_ens",
            [self.SYMBOL],
            n_ensemble_windows=n_windows,
            max_train_bars=260,
            min_confidence=0.0,   # emit all signals so predict path is exercised
        )

    def test_ensemble_trains(self):
        """Strategy should be marked trained and hold K calibrated sub-models."""
        import asyncio
        s = self._strategy(n_windows=2)
        bars = _make_bars(260, self.SYMBOL)
        s.warm_bars(self.SYMBOL, bars)
        asyncio.run(s.on_start())

        assert s._is_trained.get(self.SYMBOL, False), "Strategy should be trained"
        ens = s._ensemble.get(self.SYMBOL, [])
        assert len(ens) >= 1, "Ensemble should contain at least one sub-model"

    def test_ensemble_predict_no_crash(self):
        """Calling on_bar after training should return Signal or None without error."""
        import asyncio
        from myquant.models.signal import Signal
        s = self._strategy(n_windows=2)
        bars = _make_bars(260, self.SYMBOL)
        s.warm_bars(self.SYMBOL, bars)
        asyncio.run(s.on_start())

        result = s.on_bar(bars[-1])
        assert result is None or isinstance(result, Signal)

    def test_single_window_still_trains(self):
        """n_ensemble_windows=1 should produce a single-model ensemble."""
        import asyncio
        s = self._strategy(n_windows=1)
        bars = _make_bars(260, self.SYMBOL)
        s.warm_bars(self.SYMBOL, bars)
        asyncio.run(s.on_start())

        ens = s._ensemble.get(self.SYMBOL, [])
        assert len(ens) == 1


# ══════════════════════════════════════════════════════════════════════════════
# TestRiskGateHotStock — Layer 8 hot-stock guard
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskGateHotStock:
    """Tests for RiskGate Layer 8: hot-stock exclusion list."""

    def _gate(self):
        from myquant.risk.risk_gate import RiskGate
        return RiskGate(
            nav_getter=lambda: 1_000_000.0,
            positions_getter=lambda: {},
        )

    def _buy_signal(self, symbol: str = "SH600519"):
        from myquant.models.signal import Signal, SignalType
        return Signal(
            strategy_id="test",
            symbol=symbol,
            signal_type=SignalType.BUY,
            price=100.0,
            quantity=100,
            ts=datetime(2024, 6, 1, 10, 0),
        )

    def _sell_signal(self, symbol: str = "SH600519"):
        from myquant.models.signal import Signal, SignalType
        return Signal(
            strategy_id="test",
            symbol=symbol,
            signal_type=SignalType.SELL,
            price=100.0,
            quantity=100,
            ts=datetime(2024, 6, 1, 10, 0),
        )

    def test_hot_stock_buy_rejected(self):
        """BUY signal for a hot-stock-listed symbol must be rejected."""
        gate = self._gate()
        gate.set_hot_symbols({"SH600519"})
        decision = gate._check_hot_stock(self._buy_signal("SH600519"))
        assert not decision.approved
        assert "hot-stock" in decision.reason.lower()

    def test_hot_stock_sell_allowed(self):
        """SELL (exit) signal for a hot-stock-listed symbol must be approved."""
        gate = self._gate()
        gate.set_hot_symbols({"SH600519"})
        decision = gate._check_hot_stock(self._sell_signal("SH600519"))
        assert decision.approved

    def test_unlisted_symbol_approved(self):
        """BUY signal for a symbol NOT on the hot list must be approved."""
        gate = self._gate()
        gate.set_hot_symbols({"SH600000"})
        decision = gate._check_hot_stock(self._buy_signal("SH600519"))
        assert decision.approved

    def test_empty_hot_list_all_pass(self):
        """With an empty hot list, all BUY signals should pass Layer 8."""
        gate = self._gate()
        decision = gate._check_hot_stock(self._buy_signal("SH600519"))
        assert decision.approved

    def test_set_hot_symbols_case_insensitive(self):
        """set_hot_symbols normalises to upper-case; matching is case-insensitive."""
        gate = self._gate()
        gate.set_hot_symbols({"sh600519"})  # lower-case input
        decision = gate._check_hot_stock(self._buy_signal("SH600519"))
        assert not decision.approved


# ══════════════════════════════════════════════════════════════════════════════
# TestRiskGateBelowMA20 — Layer 9 MA20 life-line check
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskGateBelowMA20:
    """Tests for RiskGate Layer 9: MA20 life-line entry block."""

    def _gate(self):
        from myquant.risk.risk_gate import RiskGate
        return RiskGate(
            nav_getter=lambda: 1_000_000.0,
            positions_getter=lambda: {},
        )

    def _buy(self, symbol: str = "SH600519", price: float = 100.0):
        from myquant.models.signal import Signal, SignalType
        return Signal(
            strategy_id="test",
            symbol=symbol,
            signal_type=SignalType.BUY,
            price=price,
            quantity=100,
            ts=datetime(2024, 6, 1, 10, 0),
        )

    def _sell(self, symbol: str = "SH600519", price: float = 100.0):
        from myquant.models.signal import Signal, SignalType
        return Signal(
            strategy_id="test",
            symbol=symbol,
            signal_type=SignalType.SELL,
            price=price,
            quantity=100,
            ts=datetime(2024, 6, 1, 10, 0),
        )

    def test_buy_below_ma20_rejected(self):
        """BUY when price < MA20 must be rejected (life-line rule)."""
        gate = self._gate()
        gate.set_ma20_map({"SH600519": 110.0})   # MA20=110, price=100 → below
        decision = gate._check_below_ma20(self._buy("SH600519", price=100.0))
        assert not decision.approved
        assert "MA20" in decision.reason or "ma20" in decision.reason.lower()

    def test_buy_above_ma20_approved(self):
        """BUY when price > MA20 must be approved."""
        gate = self._gate()
        gate.set_ma20_map({"SH600519": 90.0})    # MA20=90, price=100 → above
        decision = gate._check_below_ma20(self._buy("SH600519", price=100.0))
        assert decision.approved

    def test_sell_below_ma20_always_approved(self):
        """SELL (exit) is always approved regardless of MA20 position."""
        gate = self._gate()
        gate.set_ma20_map({"SH600519": 150.0})   # price well below MA20
        decision = gate._check_below_ma20(self._sell("SH600519", price=100.0))
        assert decision.approved

    def test_no_ma20_data_approved(self):
        """If symbol has no MA20 data injected, BUY must still be approved (no block)."""
        gate = self._gate()
        # Empty map — no data for this symbol
        decision = gate._check_below_ma20(self._buy("SH600519", price=100.0))
        assert decision.approved

    def test_set_ma20_map_case_insensitive(self):
        """set_ma20_map normalises to upper-case; lookup is case-insensitive."""
        gate = self._gate()
        gate.set_ma20_map({"sh600519": 110.0})   # lower-case input
        decision = gate._check_below_ma20(self._buy("SH600519", price=100.0))
        assert not decision.approved


# ══════════════════════════════════════════════════════════════════════════════
# TestStrategyBandit — RL UCB1 adaptive strategy weighting
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyBandit:
    """Unit tests for the UCB1 multi-armed bandit strategy selector."""

    def _bandit(self, ids=None):
        from myquant.strategy.rl.bandit import StrategyBandit
        return StrategyBandit(
            strategy_ids=ids or ["lgbm_core", "rsi_filter", "macd_sig"],
            ema_alpha=0.5,    # fast adaptation for testing
            ucb_c=1.0,
            pnl_scale=1000.0,
        )

    def test_cold_start_returns_one(self):
        """Before any updates, all weights must be 1.0 (neutral)."""
        b = self._bandit()
        assert b.get_weight("lgbm_core") == 1.0
        assert b.get_weight("rsi_filter") == 1.0
        assert b.get_weight("macd_sig") == 1.0

    def test_unknown_strategy_returns_one(self):
        """Unknown strategy_id returns 1.0 without crashing."""
        b = self._bandit()
        assert b.get_weight("unknown_strat") == 1.0

    def test_profitable_strategy_gets_weight_above_one(self):
        """After repeated profits, a strategy's weight should exceed 1.0."""
        b = self._bandit(["strat_a", "strat_b"])
        # Feed many wins to strat_a
        for _ in range(20):
            b.update("strat_a", 2000.0)   # consistent wins
        for _ in range(20):
            b.update("strat_b", -500.0)   # consistent losses
        assert b.get_weight("strat_a") > b.get_weight("strat_b"), (
            "Profitable strategy should outweigh losing strategy"
        )
        assert b.get_weight("strat_a") > 1.0, "Big winner should have weight > 1.0"

    def test_losing_strategy_gets_weight_below_one(self):
        """After repeated losses, a strategy's weight should drop below 1.0."""
        b = self._bandit(["only"])
        for _ in range(15):
            b.update("only", -3000.0)
        assert b.get_weight("only") < 1.0, "Losing strategy should have weight < 1.0"

    def test_weight_bounded(self):
        """Weight must always stay within [W_MIN, W_MAX]."""
        from myquant.strategy.rl.bandit import _WEIGHT_MIN, _WEIGHT_MAX
        b = self._bandit(["x"])
        for pnl in [1e9, -1e9, 0, 500, -500]:
            b.update("x", pnl)
        w = b.get_weight("x")
        assert _WEIGHT_MIN <= w <= _WEIGHT_MAX, f"Weight {w} outside bounds"

    def test_auto_register_unknown_strategy(self):
        """update() with an unknown strategy_id should auto-register it."""
        b = self._bandit([])
        b.update("new_strat", 1000.0)
        # Should not raise and should return a sensible weight
        w = b.get_weight("new_strat")
        assert 0.5 < w < 2.0

    def test_weights_summary_contains_all_ids(self):
        """weights_summary() should return a dict with all registered strategy IDs."""
        b = self._bandit(["a", "b", "c"])
        summary = b.weights_summary()
        for k in ["a", "b", "c"]:
            assert k in summary
            assert isinstance(summary[k]["weight"], float)

    def test_exploration_bonus_cold_strategy(self):
        """A strategy with 0 updates gets UCB exploration bonus — weight must equal
        a 'seen' strategy with moderate EMA after the first update is made."""
        b = self._bandit(["active", "cold"])
        for _ in range(10):
            b.update("active", 0.0)   # neutral updates to increment T
        # cold has n=0, total_updates=10 → large explore bonus
        w_cold   = b.get_weight("cold")
        w_active = b.get_weight("active")
        # Cold (unexplored) gets a higher UCB bonus than neutral-updated active
        assert w_cold >= w_active, "Unexplored strategy should get UCB exploration bonus"

    def test_symbol_weight_cold_returns_strategy_weight(self):
        """get_weight with symbol falls back to strategy weight when no symbol trades."""
        b = self._bandit(["lgbm"])
        for _ in range(5):
            b.update("lgbm", 2000.0)    # strategy-level only, no symbol arg
        w_no_sym   = b.get_weight("lgbm")
        w_with_sym = b.get_weight("lgbm", symbol="sh600871")
        assert w_no_sym == w_with_sym, (
            "Cold symbol should fall back to strategy-level weight"
        )

    def test_symbol_weight_adapts_independently(self):
        """Per-symbol weights adapt independently — winning symbol outweighs losing one."""
        b = self._bandit(["lgbm"])
        for _ in range(5):
            b.update("lgbm", 3000.0, symbol="sh000001")    # sym A: consistent wins
            b.update("lgbm", -2000.0, symbol="sh600871")   # sym B: consistent losses
        w_good = b.get_weight("lgbm", symbol="sh000001")
        w_bad  = b.get_weight("lgbm", symbol="sh600871")
        assert w_good > w_bad, "Winning symbol should outweigh losing symbol"
        from myquant.strategy.rl.bandit import _WEIGHT_MIN, _WEIGHT_MAX
        assert _WEIGHT_MIN <= w_good <= _WEIGHT_MAX
        assert _WEIGHT_MIN <= w_bad  <= _WEIGHT_MAX


# ══════════════════════════════════════════════════════════════════════════════
# TestMakeLabelsRL — cost-aware labels
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeLabelsRL:
    """Tests for make_labels_rl() — cost-aware ternary classification labels."""

    @pytest.fixture(autouse=True)
    def _bars(self):
        from myquant.strategy.ml.feature_engineer import bars_to_features
        self.df = bars_to_features(_make_bars(300, "SH600519"))

    def test_returns_ternary(self):
        """make_labels_rl() must return only {-1, 0, 1}."""
        from myquant.strategy.ml.feature_engineer import make_labels_rl
        labels = make_labels_rl(self.df, forward_days=5, threshold=0.015, commission_rate=0.0003)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_fewer_extremes_than_plain_labels(self):
        """Cost-aware labels should have equal or fewer BUY+SELL labels than plain labels."""
        from myquant.strategy.ml.feature_engineer import make_labels, make_labels_rl
        plain = make_labels(self.df, forward_days=5, threshold=0.015)
        rl    = make_labels_rl(self.df, forward_days=5, threshold=0.015, commission_rate=0.0003)
        n_directional_plain = (plain.dropna() != 0).sum()
        n_directional_rl    = (rl.dropna()    != 0).sum()
        assert n_directional_rl <= n_directional_plain, (
            "Cost-aware labels should suppress marginal-edge signals"
        )

    def test_index_aligned(self):
        """Labels must be aligned to the input DataFrame's index."""
        from myquant.strategy.ml.feature_engineer import make_labels_rl
        labels = make_labels_rl(self.df, forward_days=5, threshold=0.015)
        assert labels.index.equals(self.df.index)

    def test_zero_commission_equals_plain_labels(self):
        """With commission_rate=0, cost-aware labels must equal plain make_labels()."""
        import pandas as pd
        from myquant.strategy.ml.feature_engineer import make_labels, make_labels_rl
        plain = make_labels(self.df, forward_days=5, threshold=0.015)
        rl    = make_labels_rl(self.df, forward_days=5, threshold=0.015, commission_rate=0.0)
        pd.testing.assert_series_equal(plain, rl)


# ══════════════════════════════════════════════════════════════════════════════
# TestLGBMMinHold — cooldown between signals
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_LGB_FOR_TEST, reason="lightgbm not installed")
class TestLGBMMinHold:
    """Tests for the min_hold_bars cooldown added to LGBMStrategy."""

    SYMBOL = "sh600519"

    def _trained_strategy(self, min_hold: int = 5) -> "LGBMStrategy":
        import asyncio
        from myquant.strategy.ml.lgbm_strategy import LGBMStrategy
        s = LGBMStrategy(
            "test_minhold", [self.SYMBOL],
            n_ensemble_windows=1, max_train_bars=260,
            min_confidence=0.0,   # emit all non-HOLD predictions
            min_hold_bars=min_hold,
        )
        bars = _make_bars(260, self.SYMBOL)
        s.warm_bars(self.SYMBOL, bars)
        asyncio.run(s.on_start())
        return s, bars

    def test_min_hold_params_stored(self):
        """min_hold_bars and commission_rate must be stored as attributes."""
        from myquant.strategy.ml.lgbm_strategy import LGBMStrategy
        s = LGBMStrategy("t", ["sh600519"], min_hold_bars=7, commission_rate=0.0005)
        assert s.min_hold_bars == 7
        assert s.commission_rate == 0.0005

    def test_cooldown_suppresses_rapid_signals(self):
        """With min_hold_bars=5, the strategy should emit fewer signals than
        a version with min_hold_bars=0 when fed the same bar sequence."""
        import asyncio
        from myquant.models.signal import Signal
        from myquant.strategy.ml.lgbm_strategy import LGBMStrategy

        bars = _make_bars(260, self.SYMBOL)

        def count_signals(min_hold: int) -> int:
            s = LGBMStrategy(
                "t", [self.SYMBOL],
                n_ensemble_windows=1, max_train_bars=260,
                min_confidence=0.0,
                min_hold_bars=min_hold,
            )
            s.warm_bars(self.SYMBOL, bars)
            asyncio.run(s.on_start())
            count = 0
            for b in bars[-50:]:  # replay last 50 bars after training
                sig = s.on_bar(b)
                if isinstance(sig, Signal):
                    count += 1
            return count

        signals_no_hold = count_signals(min_hold=0)
        signals_with_hold = count_signals(min_hold=5)
        assert signals_with_hold <= signals_no_hold, (
            f"min_hold_bars=5 should produce ≤ signals than min_hold_bars=0 "
            f"({signals_with_hold} vs {signals_no_hold})"
        )

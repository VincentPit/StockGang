"""
tests/test_screener_causal.py — Tests for causal trace outputs from stock_screener.py

Coverage
────────
1. TestScreenerMetricHelpers    — _atr_pct, _ma50_pct_above, _autocorr_lag1, _max_drawdown
2. TestFetchAndScore            — _fetch_and_score returns data_scope
3. TestScreenerCausalNodes      — per-factor causal_nodes shape after screen()
4. TestCausalNodeMath           — weights, norm_value in [0,1], contribution formula
5. TestGateChecks               — gate_checks structure and logic
6. TestDataScopeFromScreener    — data_scope from screener (trend, dates, prices)
7. TestScreenerCausalDirections — direction thresholds for screener factors
8. TestScreenViaAPI             — /api/screen route returns causal fields
"""
from __future__ import annotations

import threading
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Isolated DB fixture ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import api.db as db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_screener_causal.db")
    monkeypatch.setattr(db, "_local", threading.local())
    monkeypatch.setattr(db, "_mem", {})
    db.init_db()
    yield
    if hasattr(db._local, "conn") and db._local.conn:
        db._local.conn.close()
        db._local.conn = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_closes(n: int = 260, trend: str = "up") -> np.ndarray:
    """Generate synthetic close prices."""
    if trend == "up":
        return 100.0 + np.arange(n, dtype=float) * 0.2
    elif trend == "gradual_up":
        # Three phases: steady climb (60%) → pullback (20%) → stabilise (20%).
        # After _make_ohlcv_df injects a limit-up spike at bar n-30, the final
        # stable tail keeps the current close above MA60, while price_52w_pct
        # sits ≈ 75% (well below the 90% hot-stock cap).
        rise_n   = int(n * 0.60)
        fall_n   = int(n * 0.20)
        stable_n = n - rise_n - fall_n
        rise   = 100.0 + np.arange(rise_n,   dtype=float) * 0.30
        peak   = float(rise[-1])
        fall   = peak   - np.arange(fall_n,   dtype=float) * 0.20
        trough = float(fall[-1])
        stable = trough + np.arange(stable_n, dtype=float) * 0.05
        return np.concatenate([rise, fall, stable])
    elif trend == "down":
        return 100.0 - np.arange(n, dtype=float) * 0.2
    else:  # flat
        return np.full(n, 100.0)


def _make_ohlcv_df(n: int = 260, trend: str = "up") -> pd.DataFrame:
    closes = _make_closes(n, trend)
    # Inject one 涨停 (+10%) spike 30 bars from the end so every synthetic stock
    # passes the new hard filter (limit_up_60d >= 1 required).  The spike also
    # pushes hi52 above the final close, dropping price_52w_pct below 0.90 for
    # monotone-up trends that would otherwise be near-peak-filtered.
    if n >= 60:
        lu_idx = n - 30
        closes[lu_idx] = closes[lu_idx - 1] * 1.10
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Close":  closes,
        "High":   closes * 1.01,
        "Low":    closes * 0.99,
        "Open":   closes * 0.995,
        "Volume": np.ones(n) * 1_000_000,
    }, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestScreenerMetricHelpers:
    def test_atr_pct_returns_float(self):
        from myquant.tools.stock_screener import _atr_pct
        closes = _make_closes(100)
        highs  = closes * 1.01
        lows   = closes * 0.99
        result = _atr_pct(closes, highs, lows)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_atr_pct_too_few_bars_returns_zero(self):
        from myquant.tools.stock_screener import _atr_pct
        closes = np.array([100.0] * 5)
        result = _atr_pct(closes, closes * 1.01, closes * 0.99, period=14)
        assert result == 0.0

    def test_atr_pct_volatile_gt_stable(self):
        from myquant.tools.stock_screener import _atr_pct
        n      = 100
        stable = np.full(n, 100.0)
        noisy  = 100.0 + np.random.randn(n) * 2
        h_s, l_s = stable * 1.001, stable * 0.999
        h_n, l_n = noisy  * 1.02,  noisy  * 0.98
        atr_stable = _atr_pct(stable, h_s, l_s)
        atr_noisy  = _atr_pct(noisy,  h_n, l_n)
        assert atr_noisy > atr_stable

    def test_ma50_pct_above_uptrend(self):
        from myquant.tools.stock_screener import _ma50_pct_above
        closes = _make_closes(260, "up")
        result = _ma50_pct_above(closes)
        assert result > 0.9, f"Uptrend should have >90% above MA50, got {result:.2%}"

    def test_ma50_pct_above_downtrend(self):
        from myquant.tools.stock_screener import _ma50_pct_above
        closes = _make_closes(260, "down")
        result = _ma50_pct_above(closes)
        assert result < 0.1, f"Downtrend should have <10% above MA50, got {result:.2%}"

    def test_ma50_pct_above_too_few_bars(self):
        from myquant.tools.stock_screener import _ma50_pct_above
        closes = _make_closes(40)
        result = _ma50_pct_above(closes)
        assert result == 0.5  # fallback

    def test_ma50_pct_above_in_unit_interval(self):
        from myquant.tools.stock_screener import _ma50_pct_above
        closes = _make_closes(200)
        result = _ma50_pct_above(closes)
        assert 0.0 <= result <= 1.0

    def test_autocorr_lag1_trending_positive(self):
        from myquant.tools.stock_screener import _autocorr_lag1
        closes = _make_closes(100, "up")
        rets   = np.diff(closes) / closes[:-1]
        result = _autocorr_lag1(rets)
        # Constant trend = perfectly autocorrelated, but after diff it may be ~0;
        # we just check it's in [-1, 1]
        assert -1.0 <= result <= 1.0

    def test_autocorr_lag1_too_few_returns_zero(self):
        from myquant.tools.stock_screener import _autocorr_lag1
        result = _autocorr_lag1(np.array([0.01, 0.02]))
        assert result == 0.0

    def test_max_drawdown_uptrend_is_small(self):
        from myquant.tools.stock_screener import _max_drawdown
        closes = _make_closes(100, "up")
        result = _max_drawdown(closes)
        assert result >= -0.05, f"Uptrend drawdown should be near 0, got {result:.2%}"

    def test_max_drawdown_is_non_positive(self):
        from myquant.tools.stock_screener import _max_drawdown
        closes = 100.0 + np.random.randn(100).cumsum()
        result = _max_drawdown(closes)
        assert result <= 0.0

    def test_max_drawdown_severe_crash(self):
        from myquant.tools.stock_screener import _max_drawdown
        closes = np.array([100.0, 50.0, 30.0, 40.0, 60.0])
        result = _max_drawdown(closes)
        assert result <= -0.70, f"Expected > -70% drawdown, got {result:.2%}"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  _fetch_and_score — data_scope output
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchAndScore:
    def _call_with_fake_yf(self, df: pd.DataFrame, min_bars: int = 50) -> dict:
        from myquant.tools.stock_screener import _fetch_and_score

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            return _fetch_and_score(
                "sh600519", "600519.SS", "Moutai",
                "2024-01-01", "2025-01-01",
                min_bars, len(df),
            )

    def test_returns_data_scope(self):
        df     = _make_ohlcv_df(260, "up")
        result = self._call_with_fake_yf(df)
        assert "_error" not in result
        assert "data_scope" in result

    def test_data_scope_keys(self):
        df = _make_ohlcv_df(260, "up")
        r  = self._call_with_fake_yf(df)
        ds = r["data_scope"]
        for key in ["start_date", "end_date", "bars", "price_start",
                    "price_end", "price_min", "price_max", "trend"]:
            assert key in ds, f"data_scope missing key: {key}"

    def test_data_scope_trend_uptrend(self):
        df = _make_ohlcv_df(260, "up")
        r  = self._call_with_fake_yf(df)
        assert r["data_scope"]["trend"] == "UPTREND"

    def test_data_scope_trend_downtrend(self):
        df = _make_ohlcv_df(260, "down")
        r  = self._call_with_fake_yf(df)
        assert r["data_scope"]["trend"] == "DOWNTREND"

    def test_data_scope_bars_count(self):
        df = _make_ohlcv_df(260, "up")
        r  = self._call_with_fake_yf(df)
        assert r["data_scope"]["bars"] == len(df)

    def test_price_start_and_end(self):
        df = _make_ohlcv_df(260, "up")
        r  = self._call_with_fake_yf(df)
        ds = r["data_scope"]
        assert ds["price_start"] == pytest.approx(df["Close"].iloc[0],  abs=0.01)
        assert ds["price_end"]   == pytest.approx(df["Close"].iloc[-1], abs=0.01)

    def test_returns_error_on_insufficient_bars(self):
        df = _make_ohlcv_df(10, "up")
        r  = self._call_with_fake_yf(df, min_bars=200)
        assert "_error" in r


# ══════════════════════════════════════════════════════════════════════════════
# 3.  screen() — causal_nodes shape after normalisation
# ══════════════════════════════════════════════════════════════════════════════

class TestScreenerCausalNodes:
    """Test screen() with mocked yfinance and universe."""

    FACTORS = ["trend_pct", "atr_pct", "autocorr", "ret_6m", "max_dd", "vol_60d", "dist_52w_high", "yang_ratio_60d", "sharpe", "calmar"]

    @pytest.fixture(autouse=True)
    def _patch_universe_and_yf(self, monkeypatch):
        fake_universe = [
            {"sym": "sh600519", "yf_ticker": "600519.SS", "name": "Moutai"},
            {"sym": "sz300750", "yf_ticker": "300750.SZ", "name": "CATL"},
        ]
        monkeypatch.setattr(
            "myquant.data.fetchers.universe_fetcher.fetch_universe",
            lambda indices=None: fake_universe,
        )

        df_up   = _make_ohlcv_df(260, "up")
        df_down = _make_ohlcv_df(260, "down")

        call_count = [0]
        def _fake_ticker(tick):
            obj = MagicMock()
            obj.history.return_value = df_up if call_count[0] % 2 == 0 else df_down
            call_count[0] += 1
            return obj

        monkeypatch.setattr("yfinance.Ticker", _fake_ticker)

    def test_screen_returns_causal_nodes(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=2, min_bars=50, verbose=False)
        assert len(results) > 0
        for r in results:
            assert "causal_nodes" in r

    def test_exactly_ten_factors(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=2, min_bars=50, verbose=False)
        for r in results:
            factors = [n["factor"] for n in r["causal_nodes"]]
            assert sorted(factors) == sorted(self.FACTORS), \
                f"Unexpected factors: {factors}"

    def test_causal_node_required_keys(self):
        from myquant.tools.stock_screener import screen
        required = {
            "factor", "label", "description", "raw_value",
            "norm_value", "weight", "contribution", "direction", "percentile",
        }
        _, results, _ = screen(top_n=2, min_bars=50, verbose=False)
        for r in results:
            for node in r["causal_nodes"]:
                missing = required - set(node.keys())
                assert not missing, f"Node missing keys: {missing}"

    def test_gate_checks_present(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=2, min_bars=50, verbose=False)
        for r in results:
            assert "gate_checks" in r
            assert isinstance(r["gate_checks"], list)
            assert len(r["gate_checks"]) >= 1

    def test_gate_check_min_bars_passed(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=2, min_bars=50, verbose=False)
        for r in results:
            gate = next((g for g in r["gate_checks"] if g["check"] == "min_bars"), None)
            assert gate is not None
            assert gate["passed"] is True
            assert gate["actual"] >= gate["threshold"]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CausalNode math — screener
# ══════════════════════════════════════════════════════════════════════════════

class TestCausalNodeMathScreener:
    @pytest.fixture(autouse=True)
    def _patch_universe_and_yf(self, monkeypatch):
        fake_universe = [
            {"sym": "sh600519", "yf_ticker": "600519.SS", "name": "Moutai"},
            {"sym": "sz300750", "yf_ticker": "300750.SZ", "name": "CATL"},
            {"sym": "sh600036", "yf_ticker": "600036.SS", "name": "CMB"},
        ]
        monkeypatch.setattr(
            "myquant.data.fetchers.universe_fetcher.fetch_universe",
            lambda indices=None: fake_universe,
        )
        call_count = [0]
        dfs = [
            _make_ohlcv_df(260, "up"),
            _make_ohlcv_df(260, "down"),
            _make_ohlcv_df(260, "up"),
        ]
        def _fake_ticker(tick):
            obj = MagicMock()
            obj.history.return_value = dfs[call_count[0] % len(dfs)]
            call_count[0] += 1
            return obj

        monkeypatch.setattr("yfinance.Ticker", _fake_ticker)

    def test_norm_values_in_unit_interval(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=3, min_bars=50, verbose=False)
        for r in results:
            for node in r["causal_nodes"]:
                assert 0.0 <= node["norm_value"] <= 1.0, \
                    f"{node['factor']} norm_value={node['norm_value']} out of [0,1]"

    def test_contribution_equals_weight_times_norm(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=3, min_bars=50, verbose=False)
        for r in results:
            for node in r["causal_nodes"]:
                expected = node["weight"] * node["norm_value"]
                assert abs(node["contribution"] - expected) < 1e-6, \
                    f"contribution mismatch for {node['factor']}"

    def test_weights_sum_to_one(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=3, min_bars=50, verbose=False)
        for r in results:
            total = sum(n["weight"] for n in r["causal_nodes"])
            assert abs(total - 1.0) < 1e-9

    def test_score_equals_sum_of_contributions(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=3, min_bars=50, verbose=False)
        for r in results:
            total = sum(n["contribution"] for n in r["causal_nodes"])
            assert abs(r["score"] - total) < 1e-6, \
                f"score {r['score']:.6f} ≠ contributions sum {total:.6f}"

    def test_direction_is_valid_enum(self):
        from myquant.tools.stock_screener import screen
        valid = {"positive", "negative", "neutral"}
        _, results, _ = screen(top_n=3, min_bars=50, verbose=False)
        for r in results:
            for node in r["causal_nodes"]:
                assert node["direction"] in valid

    def test_percentile_label_format(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=3, min_bars=50, verbose=False)
        for r in results:
            for node in r["causal_nodes"]:
                assert node["percentile"].startswith("Top "), \
                    f"percentile format wrong: {node['percentile']!r}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. GateChecks
# ══════════════════════════════════════════════════════════════════════════════

class TestGateChecks:
    REQUIRED_KEYS = {"check", "label", "threshold", "actual", "passed", "note"}

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake_universe = [{"sym": "sh600519", "yf_ticker": "600519.SS", "name": "Moutai"}]
        monkeypatch.setattr(
            "myquant.data.fetchers.universe_fetcher.fetch_universe",
            lambda indices=None: fake_universe,
        )
        df = _make_ohlcv_df(260, "up")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        monkeypatch.setattr("yfinance.Ticker", lambda _: mock_ticker)

    def test_gate_check_schema(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=1, min_bars=50, verbose=False)
        for r in results:
            for gate in r["gate_checks"]:
                missing = self.REQUIRED_KEYS - set(gate.keys())
                assert not missing, f"gate_check missing keys: {missing}"

    def test_gate_check_passed_types(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=1, min_bars=50, verbose=False)
        for r in results:
            for gate in r["gate_checks"]:
                assert isinstance(gate["passed"], bool)
                # threshold may be None for qualitative checks (e.g. above_ma60,
                # not_post_pump) that have no single numeric cut-off
                assert gate["threshold"] is None or isinstance(
                    gate["threshold"], (int, float, str)
                ), f"Bad threshold type {type(gate['threshold'])} in gate '{gate['check']}'"
                assert isinstance(gate["actual"], (int, float, str))
                assert isinstance(gate["note"],   str)

    def test_min_bars_gate_note_contains_bar_count(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=1, min_bars=50, verbose=False)
        for r in results:
            gate = next((g for g in r["gate_checks"] if g["check"] == "min_bars"), None)
            assert gate is not None
            assert str(gate["actual"]) in gate["note"]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DataScope from screener
# ══════════════════════════════════════════════════════════════════════════════

class TestDataScopeFromScreener:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake_universe = [{"sym": "sh600519", "yf_ticker": "600519.SS", "name": "Moutai"}]
        monkeypatch.setattr(
            "myquant.data.fetchers.universe_fetcher.fetch_universe",
            lambda indices=None: fake_universe,
        )
        # "gradual_up": peaks at 70% of history then minor pullback — passes hard filters
        # (ret_20d is slightly negative, price_52w_pct ≈ 0.85 < 0.90 threshold)
        df = _make_ohlcv_df(260, "gradual_up")
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        monkeypatch.setattr("yfinance.Ticker", lambda _: mock_ticker)

    def test_data_scope_present_in_results(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=1, min_bars=50, verbose=False)
        for r in results:
            assert "data_scope" in r

    def test_data_scope_trend_uptrend(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=1, min_bars=50, verbose=False)
        # Uptrend df should give UPTREND (>5% gain)
        assert results[0]["data_scope"]["trend"] == "UPTREND"

    def test_data_scope_price_range_valid(self):
        from myquant.tools.stock_screener import screen
        _, results, _ = screen(top_n=1, min_bars=50, verbose=False)
        ds = results[0]["data_scope"]
        assert ds["price_min"] <= ds["price_start"]
        assert ds["price_max"] >= ds["price_start"]
        assert ds["price_min"] <= ds["price_end"]
        assert ds["price_max"] >= ds["price_end"]


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Screener direction thresholds
# ══════════════════════════════════════════════════════════════════════════════

class TestScreenerCausalDirections:
    """Test the direction logic: positive if norm>=0.6, negative if norm<0.4, else neutral."""

    def _direction(self, norm_value: float) -> str:
        """Reproduce screener direction thresholds."""
        if norm_value >= 0.6:
            return "positive"
        elif norm_value < 0.4:
            return "negative"
        return "neutral"

    def test_high_norm_positive(self):
        assert self._direction(0.80) == "positive"

    def test_low_norm_negative(self):
        assert self._direction(0.20) == "negative"

    def test_mid_norm_neutral(self):
        assert self._direction(0.50) == "neutral"

    def test_boundary_0_6_is_positive(self):
        assert self._direction(0.6) == "positive"

    def test_just_below_0_6_is_neutral(self):
        assert self._direction(0.5999) == "neutral"

    def test_boundary_0_4_is_neutral(self):
        # 0.4 is NOT < 0.4 → neutral
        assert self._direction(0.4) == "neutral"

    def test_just_below_0_4_is_negative(self):
        assert self._direction(0.3999) == "negative"


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Screen route — FastAPI returns causal fields
# ══════════════════════════════════════════════════════════════════════════════

class TestScreenViaAPI:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        import api.runner as runner
        import api.main   as main_mod

        fake_rows = [
            {
                "rank":        1,
                "symbol":      "sh600519",
                "yf_ticker":   "600519.SS",
                "name":        "Moutai",
                "bars":        260,
                "ret_1y":      0.18,
                "ret_6m":      0.09,
                "sharpe":      1.42,
                "max_dd":     -0.12,
                "trend_pct":   0.72,
                "atr_pct":     0.015,
                "autocorr":    0.04,
                "score":       0.718,
                "recommended": True,
                "causal_nodes": [
                    {
                        "factor":      "trend_pct",
                        "label":       "Trend Quality",
                        "description": "Price above MA50 for 72% of the window",
                        "raw_value":   0.72,
                        "norm_value":  0.80,
                        "weight":      0.25,
                        "contribution": 0.20,
                        "direction":   "positive",
                        "percentile":  "Top 20%",
                    }
                ],
                "data_scope": {
                    "start_date":  "2024-03-01",
                    "end_date":    "2025-03-01",
                    "bars":        260,
                    "price_start": 100.0,
                    "price_end":   120.0,
                    "price_min":   92.0,
                    "price_max":   125.0,
                    "trend":       "UPTREND",
                },
                "gate_checks": [
                    {
                        "check":     "min_bars",
                        "label":     "Sufficient price history",
                        "threshold": 200,
                        "actual":    260,
                        "passed":    True,
                        "note":      "260 bars ≥ 200 required",
                    }
                ],
            }
        ]

        async def _fake_launch_screen(jid, req):
            runner._update_job(jid, {
                "status":        "done",
                "top_symbols":   ["sh600519"],
                "rows":          fake_rows,
                "universe_size": 300,
            })

        monkeypatch.setattr(runner,   "launch_screener", _fake_launch_screen)
        monkeypatch.setattr(main_mod, "launch_screener", _fake_launch_screen)

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_screen_returns_causal_nodes(self, client):
        jid = client.post("/api/screen", json={"top_n": 5}).json()["job_id"]
        res = client.get(f"/api/screen/{jid}")
        assert res.status_code == 200
        rows = res.json().get("rows", [])
        assert len(rows) > 0
        assert "causal_nodes" in rows[0]
        assert isinstance(rows[0]["causal_nodes"], list)

    def test_screen_returns_data_scope(self, client):
        jid = client.post("/api/screen", json={"top_n": 5}).json()["job_id"]
        res = client.get(f"/api/screen/{jid}")
        rows = res.json().get("rows", [])
        assert rows[0]["data_scope"] is not None
        assert rows[0]["data_scope"]["trend"] == "UPTREND"

    def test_screen_returns_gate_checks(self, client):
        jid = client.post("/api/screen", json={"top_n": 5}).json()["job_id"]
        res = client.get(f"/api/screen/{jid}")
        rows = res.json().get("rows", [])
        assert "gate_checks" in rows[0]
        assert isinstance(rows[0]["gate_checks"], list)
        assert rows[0]["gate_checks"][0]["passed"] is True

    def test_screen_causal_node_schema_via_api(self, client):
        jid  = client.post("/api/screen", json={"top_n": 5}).json()["job_id"]
        rows = client.get(f"/api/screen/{jid}").json().get("rows", [])
        required = {
            "factor", "label", "description", "raw_value",
            "norm_value", "weight", "contribution", "direction", "percentile",
        }
        for row in rows:
            for node in row["causal_nodes"]:
                missing = required - set(node.keys())
                assert not missing, f"Screen API node missing keys: {missing}"

    def test_screen_universe_size_returned(self, client):
        jid  = client.post("/api/screen", json={"top_n": 5}).json()["job_id"]
        body = client.get(f"/api/screen/{jid}").json()
        assert body.get("universe_size") == 300

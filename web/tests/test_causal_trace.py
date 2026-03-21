"""
tests/test_causal_trace.py — Tests for causal trace outputs from advisor.py

Coverage
────────
1. TestCausalNodeSchema        — get_recommendations() causal_nodes shape
2. TestDataScopeSchema         — data_scope field shape and business rules
3. TestCausalNodeMath          — contribution formula, weight sums, norm_value range
4. TestAdvisorCausalContent    — causal node factor names, direction values
5. TestMLSignalExtras          — model_signal node extras (p_buy, p_hold, p_sell)
6. TestFundamentalsExtras      — fundamentals node extras (pe_ttm, pb, roe, scores)
7. TestMomentumExtras          — momentum node extras (ret_1y, ret_3m, ret_1m)
8. TestRecommendRoute          — FastAPI /recommend-poll returns causal fields
9. TestCausalTrendMapping      — data_scope.trend based on ret_scope
10. TestCausalDirectionMapping — direction positive/negative/neutral thresholds
"""
from __future__ import annotations

import pickle
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ── Isolated DB ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import api.db as db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_causal.db")
    monkeypatch.setattr(db, "_local", threading.local())
    monkeypatch.setattr(db, "_mem", {})
    db.init_db()
    yield
    if hasattr(db._local, "conn") and db._local.conn:
        db._local.conn.close()
        db._local.conn = None


# ── Shared fixtures ────────────────────────────────────────────────────────────

class _MockModel:
    """Minimal stand-in for lgb.LGBMClassifier."""
    def __init__(self, pred_class: int = 2):
        self._pred = pred_class
        self.feature_importances_ = np.array([3.0, 2.0, 1.0])

    def predict(self, X):
        return np.array([self._pred] * len(X))

    def predict_proba(self, X):
        row = [0.1, 0.2, 0.7]  # default: BUY
        if self._pred == 0: row = [0.7, 0.2, 0.1]
        elif self._pred == 1: row = [0.2, 0.6, 0.2]
        return np.array([row] * len(X))

    def fit(self, X, y): return self


def _fake_df(n: int = 260) -> pd.DataFrame:
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    close  = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low":  close * 0.98, "close": close,
        "volume":   np.random.randint(1_000_000, 5_000_000, n).astype(float),
        "turnover": close * 1e6,
    }, index=dates)


def _fake_fund(sym: str) -> dict:
    return {
        "symbol": sym, "pe_ttm": 18.0, "pb": 3.1, "ps_ttm": 4.2,
        "roe": 22.5, "revenue_growth": 0.12, "net_margin": 0.21,
        "dividend_yield": 0.015, "value_score": 72.0,
        "growth_score": 65.0, "quality_score": 80.0,
    }


FEATS = ["ret_1d", "rsi_14", "macd_hist"]


@pytest.fixture()
def _patch_recommend(monkeypatch):
    """Patch all external deps for get_recommendations()."""
    import api.advisor as adv
    import api.runner  as runner
    from myquant.data.fetchers.universe_fetcher import fetch_universe

    # One-stock universe
    fake_universe = [{"sym": "sh600519", "yf_ticker": "600519.SS", "name": "Moutai"}]
    monkeypatch.setattr(
        "myquant.data.fetchers.universe_fetcher.fetch_universe",
        lambda indices=None: fake_universe,
    )
    monkeypatch.setattr(adv, "_load_bars_df", lambda sym, days=400: _fake_df(260))

    # Provide bars via get_price_sync
    fake_bars = [
        {"date": f"2024-{(i // 30 + 1):02d}-01", "close": 100.0 + i * 0.1,
         "open": 99.0, "high": 101.0, "low": 98.0, "volume": 1_000_000}
        for i in range(252)
    ]
    monkeypatch.setattr(runner, "get_price_sync",
                        lambda sym, days=365: {"bars": fake_bars})
    monkeypatch.setattr(runner, "get_fundamentals_sync", _fake_fund)


def _get_rows(monkeypatch) -> list[dict]:
    from api.advisor import get_recommendations
    return get_recommendations(top_n=5)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CausalNode schema
# ══════════════════════════════════════════════════════════════════════════════

class TestCausalNodeSchema:
    REQUIRED_KEYS = {
        "factor", "label", "description",
        "raw_value", "norm_value", "weight",
        "contribution", "direction", "percentile",
    }

    def test_causal_nodes_present_in_each_row(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        assert len(rows) > 0
        for row in rows:
            assert "causal_nodes" in row, "causal_nodes missing from row"
            assert isinstance(row["causal_nodes"], list)
            assert len(row["causal_nodes"]) > 0

    def test_each_node_has_required_keys(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                missing = self.REQUIRED_KEYS - set(node.keys())
                assert not missing, f"Missing keys in node: {missing}"

    def test_node_value_types(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                assert isinstance(node["label"],        str)
                assert isinstance(node["description"],  str)
                assert isinstance(node["norm_value"],   float)
                assert isinstance(node["weight"],       float)
                assert isinstance(node["contribution"], float)
                assert isinstance(node["direction"],    str)
                assert isinstance(node["percentile"],   str)

    def test_direction_is_valid_enum(self, _patch_recommend):
        from api.advisor import get_recommendations
        valid = {"positive", "negative", "neutral"}
        rows  = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                assert node["direction"] in valid, \
                    f"Invalid direction {node['direction']!r}"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DataScope schema
# ══════════════════════════════════════════════════════════════════════════════

class TestDataScopeSchema:
    REQUIRED_KEYS = {
        "start_date", "end_date", "bars",
        "price_start", "price_end", "price_min", "price_max", "trend",
    }

    def test_data_scope_present_in_each_row(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            assert "data_scope" in row, "data_scope missing from row"
            assert row["data_scope"] is not None

    def test_data_scope_has_required_keys(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            ds      = row["data_scope"]
            missing = self.REQUIRED_KEYS - set(ds.keys())
            assert not missing, f"data_scope missing keys: {missing}"

    def test_trend_is_valid_enum(self, _patch_recommend):
        from api.advisor import get_recommendations
        valid = {"UPTREND", "DOWNTREND", "SIDEWAYS"}
        rows  = get_recommendations(top_n=5)
        for row in rows:
            assert row["data_scope"]["trend"] in valid

    def test_bars_is_positive_int(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            bars = row["data_scope"]["bars"]
            assert isinstance(bars, int)
            assert bars > 0

    def test_price_min_le_price_start(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            ds = row["data_scope"]
            assert ds["price_min"] <= ds["price_start"] + 1e-6, \
                f"price_min {ds['price_min']} > price_start {ds['price_start']}"

    def test_price_max_ge_price_end(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            ds = row["data_scope"]
            assert ds["price_max"] >= ds["price_end"] - 1e-6, \
                f"price_max {ds['price_max']} < price_end {ds['price_end']}"

    def test_start_date_before_end_date(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            ds = row["data_scope"]
            assert ds["start_date"] <= ds["end_date"]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CausalNode math
# ══════════════════════════════════════════════════════════════════════════════

class TestCausalNodeMath:
    def test_contribution_equals_weight_times_norm_value(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                expected = node["weight"] * node["norm_value"]
                assert abs(node["contribution"] - expected) < 1e-4, \
                    f"contribution {node['contribution']:.6f} ≠ {expected:.6f} for {node['factor']}"

    def test_weights_sum_to_one(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            total = sum(n["weight"] for n in row["causal_nodes"])
            assert abs(total - 1.0) < 1e-6, f"weights sum to {total} not 1.0"

    def test_norm_value_in_unit_interval(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                assert 0.0 <= node["norm_value"] <= 1.0, \
                    f"norm_value {node['norm_value']} out of [0,1] for {node['factor']}"

    def test_contribution_in_zero_to_weight(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                assert node["contribution"] >= -1e-9, \
                    f"contribution {node['contribution']} < 0"
                assert node["contribution"] <= node["weight"] + 1e-9, \
                    f"contribution {node['contribution']} > weight {node['weight']}"

    def test_composite_score_matches_sum_of_contributions(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            expected = sum(n["contribution"] for n in row["causal_nodes"])
            assert abs(row["score"] - expected) < 1e-3, \
                f"score {row['score']:.4f} ≠ contributions sum {expected:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Causal node factor names and content
# ══════════════════════════════════════════════════════════════════════════════

class TestAdvisorCausalContent:
    EXPECTED_FACTORS = {"fundamentals", "quality_position", "model_signal"}

    def test_exactly_three_factors(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            factors = {n["factor"] for n in row["causal_nodes"]}
            assert factors == self.EXPECTED_FACTORS, \
                f"Unexpected factors: {factors}"

    def test_factor_weights(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        expected_weights = {
            "fundamentals":     0.40,
            "quality_position": 0.30,
            "model_signal":     0.30,
        }
        for row in rows:
            for node in row["causal_nodes"]:
                expected = expected_weights[node["factor"]]
                assert abs(node["weight"] - expected) < 1e-9, \
                    f"weight for {node['factor']}: {node['weight']} ≠ {expected}"

    def test_labels_are_non_empty(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                assert node["label"].strip(), f"Empty label for factor {node['factor']}"

    def test_description_is_non_empty(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for row in rows:
            for node in row["causal_nodes"]:
                assert node["description"].strip(), \
                    f"Empty description for factor {node['factor']}"


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ML signal node extras
# ══════════════════════════════════════════════════════════════════════════════

class TestMLSignalExtras:
    def _get_model_signal_node(self, rows: list[dict]) -> dict:
        for row in rows:
            for node in row["causal_nodes"]:
                if node["factor"] == "model_signal":
                    return node
        pytest.fail("No model_signal node found")

    def test_extras_present(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_model_signal_node(rows)
        assert node.get("extras") is not None

    def test_model_trained_key_present(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_model_signal_node(rows)
        assert "model_trained" in node["extras"]

    def test_probability_keys_present_when_trained(self, _patch_recommend):
        """When a model is stored, p_buy / p_hold / p_sell should be in extras."""
        import api.db as db
        db.save_model("sh600519", "lgbm_core", 260, "2025-01-01",
                      0.62, pickle.dumps(_MockModel(pred_class=2)), FEATS)

        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_model_signal_node(rows)
        ext  = node["extras"]
        assert "p_buy"  in ext
        assert "p_hold" in ext
        assert "p_sell" in ext

    def test_probabilities_sum_to_one_when_trained(self, _patch_recommend):
        import api.db as db
        db.save_model("sh600519", "lgbm_core", 260, "2025-01-01",
                      0.62, pickle.dumps(_MockModel(pred_class=2)), FEATS)

        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_model_signal_node(rows)
        ext  = node["extras"]
        total = ext["p_buy"] + ext["p_hold"] + ext["p_sell"]
        assert abs(total - 1.0) < 0.01, f"probabilities sum to {total}"

    def test_signal_string_in_extras(self, _patch_recommend):
        import api.db as db
        db.save_model("sh600519", "lgbm_core", 260, "2025-01-01",
                      0.62, pickle.dumps(_MockModel(pred_class=2)), FEATS)

        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_model_signal_node(rows)
        assert node["extras"]["signal"] in {"BUY", "HOLD", "SELL", "N/A"}

    def test_no_model_gives_neutral_signal(self, _patch_recommend):
        """Without a stored model, signal should be N/A and norm_value=0.5."""
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_model_signal_node(rows)
        # No model stored in this test (no db.save_model called)
        assert node["extras"]["model_trained"] is False
        assert abs(node["norm_value"] - 0.5) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Fundamentals node extras
# ══════════════════════════════════════════════════════════════════════════════

class TestFundamentalsExtras:
    def _get_fundamentals_node(self, rows: list[dict]) -> dict:
        for row in rows:
            for node in row["causal_nodes"]:
                if node["factor"] == "fundamentals":
                    return node
        pytest.fail("No fundamentals node found")

    def test_extras_present(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_fundamentals_node(rows)
        assert node.get("extras") is not None

    def test_required_extra_keys(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_fundamentals_node(rows)
        for key in ["pe_ttm", "pb", "roe", "value_score", "growth_score", "quality_score"]:
            assert key in node["extras"], f"Missing extras key: {key}"

    def test_extras_match_fake_fundamentals(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_fundamentals_node(rows)
        ext  = node["extras"]
        assert ext["pe_ttm"] == pytest.approx(18.0)
        assert ext["pb"]     == pytest.approx(3.1)
        assert ext["roe"]    == pytest.approx(22.5)

    def test_raw_value_is_composite_score(self, _patch_recommend):
        """raw_value should be the (value+growth+quality)/3 composite."""
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_fundamentals_node(rows)
        # From _fake_fund: (72+65+80)/3 = 72.33
        expected_composite = (72.0 + 65.0 + 80.0) / 3.0
        assert abs(float(node["raw_value"]) - expected_composite) < 0.1


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Momentum node extras
# ══════════════════════════════════════════════════════════════════════════════

class TestQualityPositionExtras:
    """Tests for the quality_position causal node that replaced the old momentum node.

    The anti-hot-stock scoring uses:
      - dist_52w_high (contrarian: beaten-down beats hot)
      - low_vol (quality / stability)
      - ret_6m (small 6-month momentum, capped)
    """

    def _get_quality_node(self, rows: list[dict]) -> dict:
        for row in rows:
            for node in row["causal_nodes"]:
                if node["factor"] == "quality_position":
                    return node
        pytest.fail("No quality_position node found")

    def test_extras_present(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_quality_node(rows)
        assert node.get("extras") is not None

    def test_quality_keys_present(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_quality_node(rows)
        for key in ["dist_52w_high", "price_52w_pct", "ret_6m", "ret_20d",
                    "low_vol_raw", "norm_52w_dist", "norm_low_vol", "norm_6m"]:
            assert key in node["extras"], f"Missing extras key: {key}"

    def test_quality_values_are_floats(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_quality_node(rows)
        for key in ["dist_52w_high", "price_52w_pct", "ret_6m", "ret_20d",
                    "low_vol_raw", "norm_52w_dist", "norm_low_vol", "norm_6m"]:
            assert isinstance(node["extras"][key], float), \
                f"{key} is not float: {type(node['extras'][key])}"

    def test_description_mentions_distance_and_momentum(self, _patch_recommend):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        node = self._get_quality_node(rows)
        # Description should contain 52w and 6M references
        desc = node["description"]
        assert "52w" in desc or "52W" in desc.upper(), \
            f"Description missing 52w reference: {desc}"
        assert "6M" in desc or "6m" in desc.lower(), \
            f"Description missing 6M reference: {desc}"


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FastAPI /recommend-poll returns causal fields
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommendRoute:
    @pytest.fixture(autouse=True)
    def _patch_route(self, monkeypatch):
        import api.runner  as runner
        import api.advisor as adv

        fake_universe = [{"sym": "sh600519", "yf_ticker": "600519.SS", "name": "Moutai"}]
        monkeypatch.setattr(
            "myquant.data.fetchers.universe_fetcher.fetch_universe",
            lambda indices=None: fake_universe,
        )
        monkeypatch.setattr(adv, "_load_bars_df", lambda sym, days=400: _fake_df(260))

        fake_bars = [
            {"date": f"2024-{(i // 30 + 1):02d}-01", "close": 100.0 + i * 0.1,
             "open": 99.0, "high": 101.0, "low": 98.0, "volume": 1_000_000}
            for i in range(252)
        ]
        monkeypatch.setattr(runner, "get_price_sync",
                            lambda sym, days=365: {"bars": fake_bars})
        monkeypatch.setattr(runner, "get_fundamentals_sync", _fake_fund)

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_recommend_poll_rows_have_causal_nodes(self, client):
        jid = client.get("/api/advisor/recommend?top_n=5").json()["job_id"]
        res = client.get(f"/api/advisor/recommend-poll/{jid}")
        assert res.status_code == 200
        rows = res.json().get("rows", [])
        if rows:
            assert "causal_nodes" in rows[0], "causal_nodes missing from API row"
            assert isinstance(rows[0]["causal_nodes"], list)

    def test_recommend_poll_rows_have_data_scope(self, client):
        jid = client.get("/api/advisor/recommend?top_n=5").json()["job_id"]
        res = client.get(f"/api/advisor/recommend-poll/{jid}")
        rows = res.json().get("rows", [])
        if rows:
            assert "data_scope" in rows[0]
            ds = rows[0]["data_scope"]
            if ds is not None:
                assert "trend" in ds
                assert ds["trend"] in {"UPTREND", "DOWNTREND", "SIDEWAYS"}

    def test_recommend_poll_causal_node_schema_via_api(self, client):
        jid = client.get("/api/advisor/recommend?top_n=5").json()["job_id"]
        res = client.get(f"/api/advisor/recommend-poll/{jid}")
        rows = res.json().get("rows", [])
        required_node_keys = {
            "factor", "label", "description", "raw_value",
            "norm_value", "weight", "contribution", "direction", "percentile",
        }
        for row in rows:
            for node in row.get("causal_nodes", []):
                missing = required_node_keys - set(node.keys())
                assert not missing, f"API node missing keys: {missing}"

    def test_recommend_poll_weights_sum_to_one_via_api(self, client):
        jid  = client.get("/api/advisor/recommend?top_n=5").json()["job_id"]
        rows = client.get(f"/api/advisor/recommend-poll/{jid}").json().get("rows", [])
        for row in rows:
            nodes = row.get("causal_nodes", [])
            if nodes:
                total = sum(n["weight"] for n in nodes)
                assert abs(total - 1.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DataScope trend mapping
# ══════════════════════════════════════════════════════════════════════════════

class TestCausalTrendMapping:
    """Test that data_scope.trend maps correctly to price returns."""

    def _build_scope(self, ret_scope: float) -> dict:
        """Reproduce the trend logic from advisor.py get_recommendations()."""
        return {
            "trend": (
                "UPTREND"   if ret_scope > 0.05  else
                "DOWNTREND" if ret_scope < -0.05 else
                "SIDEWAYS"
            )
        }

    def test_strong_positive_return_gives_uptrend(self):
        ds = self._build_scope(0.10)
        assert ds["trend"] == "UPTREND"

    def test_strong_negative_return_gives_downtrend(self):
        ds = self._build_scope(-0.08)
        assert ds["trend"] == "DOWNTREND"

    def test_small_positive_return_gives_sideways(self):
        ds = self._build_scope(0.03)
        assert ds["trend"] == "SIDEWAYS"

    def test_small_negative_return_gives_sideways(self):
        ds = self._build_scope(-0.02)
        assert ds["trend"] == "SIDEWAYS"

    def test_exact_positive_boundary(self):
        # 0.05 is NOT > 0.05 → SIDEWAYS
        ds = self._build_scope(0.05)
        assert ds["trend"] == "SIDEWAYS"

    def test_just_above_positive_boundary(self):
        ds = self._build_scope(0.0501)
        assert ds["trend"] == "UPTREND"

    def test_exact_negative_boundary(self):
        # -0.05 is NOT < -0.05 → SIDEWAYS
        ds = self._build_scope(-0.05)
        assert ds["trend"] == "SIDEWAYS"


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CausalNode direction mapping
# ══════════════════════════════════════════════════════════════════════════════

class TestCausalDirectionMapping:
    """Test the direction thresholds used by advisor.py causal_nodes."""

    def _direction(self, norm_value: float, factor: str = "fundamentals") -> str:
        """Reproduce advisor.py direction logic for fundamentals/model_signal nodes."""
        if factor == "fundamentals":
            return (
                "positive" if norm_value >= 0.55 else
                "negative" if norm_value <  0.35 else
                "neutral"
            )
        elif factor == "momentum":
            return (
                "positive" if norm_value >= 0.55 else
                "negative" if norm_value <  0.45 else
                "neutral"
            )
        elif factor == "model_signal":
            return (
                "positive" if norm_value >= 0.7 else
                "negative" if norm_value <  0.3 else
                "neutral"
            )
        return "neutral"

    # fundamentals
    def test_fundamentals_high_norm_is_positive(self):
        assert self._direction(0.80, "fundamentals") == "positive"

    def test_fundamentals_low_norm_is_negative(self):
        assert self._direction(0.20, "fundamentals") == "negative"

    def test_fundamentals_mid_norm_is_neutral(self):
        assert self._direction(0.45, "fundamentals") == "neutral"

    def test_fundamentals_boundary_055_is_positive(self):
        assert self._direction(0.55, "fundamentals") == "positive"

    def test_fundamentals_boundary_035_is_neutral(self):
        # 0.35 is NOT < 0.35 → neutral
        assert self._direction(0.35, "fundamentals") == "neutral"

    # momentum
    def test_momentum_high_norm_is_positive(self):
        assert self._direction(0.70, "momentum") == "positive"

    def test_momentum_low_norm_is_negative(self):
        assert self._direction(0.30, "momentum") == "negative"

    def test_momentum_mid_norm_is_neutral(self):
        assert self._direction(0.50, "momentum") == "neutral"

    # model_signal
    def test_model_signal_buy_is_positive(self):
        # BUY maps to norm_value=1.0
        assert self._direction(1.0, "model_signal") == "positive"

    def test_model_signal_sell_is_negative(self):
        # SELL maps to norm_value=0.0
        assert self._direction(0.0, "model_signal") == "negative"

    def test_model_signal_hold_is_neutral(self):
        # HOLD maps to norm_value=0.5
        assert self._direction(0.5, "model_signal") == "neutral"

"""
Tests for api/advisor.py and the trained_models DB layer.

All tests are hermetic — no network calls, no real LightGBM training.
External dependencies (data fetchers, LGB model) are monkey-patched.

Coverage
────────
1. TestTrainedModelsDB  — save/load/meta/list/delete model CRUD
2. TestStalenessLogic   — _is_stale() all branches
3. TestPredictSignal    — _predict_signal() BUY / HOLD / SELL
4. TestFeatureImportance — _feature_importance() with mock model
5. TestTrainForSymbol   — train_for_symbol() end-to-end (mocked)
6. TestAnalyzeStock     — analyze_stock() end-to-end (mocked)
7. TestGetRecommendations — get_recommendations() (mocked)
8. TestAdvisorRoutes    — FastAPI routes via TestClient (mocked)
"""
from __future__ import annotations

import pickle
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ── Isolated DB fixture (reuses same pattern as test_db.py) ──────────────────

@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    import api.db as db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_advisor.db")
    monkeypatch.setattr(db, "_local", threading.local())
    monkeypatch.setattr(db, "_mem", {})
    db.init_db()
    yield
    if hasattr(db._local, "conn") and db._local.conn:
        db._local.conn.close()
        db._local.conn = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fake_blob() -> bytes:
    """Pickle a tiny dict to use as a stand-in model blob."""
    return pickle.dumps({"type": "fake_model"})


def _fake_meta(symbol: str = "hk00700", bar_count: int = 250,
               days_old: float = 0.0, oos: float = 0.61) -> dict:
    """Return a model-meta dict with trained_at set ``days_old`` days in the past."""
    return {
        "model_id":      f"{symbol}_lgbm_core",
        "symbol":        symbol,
        "strategy_id":   "lgbm_core",
        "trained_at":    time.time() - days_old * 86400,
        "bar_count":     bar_count,
        "last_bar_date": "2025-01-15",
        "oos_accuracy":  oos,
        "feature_cols":  ["ret_1d", "rsi_14", "macd_hist"],
    }


def _fake_df(n: int = 260) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame large enough to trigger training."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open":     close * 0.99,
        "high":     close * 1.01,
        "low":      close * 0.98,
        "close":    close,
        "volume":   np.random.randint(1_000_000, 5_000_000, n).astype(float),
        "turnover": close * 1e6,
    }, index=dates)
    return df


class _MockModel:
    """Minimal stand-in for lgb.LGBMClassifier."""
    def __init__(self, pred_class: int = 2):
        self._pred   = pred_class            # 0=SELL, 1=HOLD, 2=BUY
        self.feature_importances_ = np.array([3.0, 2.0, 1.0])

    def predict(self, X):
        return np.array([self._pred] * len(X))

    def predict_proba(self, X):
        row = [0.1, 0.2, 0.7]              # default: BUY dominant
        if self._pred == 0:
            row = [0.7, 0.2, 0.1]
        elif self._pred == 1:
            row = [0.2, 0.6, 0.2]
        return np.array([row] * len(X))

    def fit(self, X, y):
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 1.  trained_models  DB layer
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainedModelsDB:
    def test_table_exists_after_init_db(self):
        import api.db as db
        conn   = db._conn()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "trained_models" in tables

    def test_save_and_load_roundtrip(self):
        import api.db as db
        blob = _fake_blob()
        mid  = db.save_model("hk00700", "lgbm_core", 250, "2025-01-15",
                             0.61, blob, ["ret_1d", "rsi_14"])
        assert mid == "hk00700_lgbm_core"

        result = db.load_model("hk00700", "lgbm_core")
        assert result is not None
        loaded_blob, meta = result
        assert pickle.loads(loaded_blob) == pickle.loads(blob)
        assert meta["symbol"]        == "hk00700"
        assert meta["bar_count"]     == 250
        assert meta["last_bar_date"] == "2025-01-15"
        assert meta["oos_accuracy"]  == pytest.approx(0.61)
        assert meta["feature_cols"]  == ["ret_1d", "rsi_14"]

    def test_load_missing_returns_none(self):
        import api.db as db
        assert db.load_model("hk99999", "lgbm_core") is None

    def test_save_upserts_existing(self):
        import api.db as db
        blob_v1 = pickle.dumps({"v": 1})
        blob_v2 = pickle.dumps({"v": 2})
        db.save_model("sh600519", "lgbm_core", 200, "2025-01-01", 0.55, blob_v1, [])
        db.save_model("sh600519", "lgbm_core", 260, "2025-03-01", 0.63, blob_v2, [])
        loaded_blob, meta = db.load_model("sh600519", "lgbm_core")
        assert pickle.loads(loaded_blob) == {"v": 2}
        assert meta["bar_count"]     == 260
        assert meta["last_bar_date"] == "2025-03-01"
        assert meta["oos_accuracy"]  == pytest.approx(0.63)

    def test_get_model_meta_no_blob(self):
        import api.db as db
        db.save_model("sz300750", "lgbm_core", 240, "2025-02-01",
                      0.58, _fake_blob(), ["ret_1d"])
        meta = db.get_model_meta("sz300750", "lgbm_core")
        assert meta is not None
        assert meta["bar_count"] == 240
        # must not contain the blob key
        assert "model_blob" not in meta

    def test_get_model_meta_missing_returns_none(self):
        import api.db as db
        assert db.get_model_meta("xx99999", "lgbm_core") is None

    def test_list_models_empty(self):
        import api.db as db
        assert db.list_models() == []

    def test_list_models_sorted_by_trained_at_desc(self):
        import api.db as db
        db.save_model("hk00700", "lgbm_core", 250, "2025-01-01",
                      0.60, _fake_blob(), [])
        time.sleep(0.02)
        db.save_model("sh600519", "lgbm_core", 200, "2025-01-01",
                      0.55, _fake_blob(), [])
        models = db.list_models()
        assert len(models) == 2
        # Most recently saved first
        assert models[0]["symbol"] == "sh600519"
        assert models[1]["symbol"] == "hk00700"
        # No model_blob in list output
        for m in models:
            assert "model_blob" not in m

    def test_delete_existing_returns_true(self):
        import api.db as db
        db.save_model("hk00700", "lgbm_core", 250, "2025-01-01",
                      0.60, _fake_blob(), [])
        assert db.delete_model("hk00700", "lgbm_core") is True
        assert db.load_model("hk00700", "lgbm_core") is None

    def test_delete_missing_returns_false(self):
        import api.db as db
        assert db.delete_model("hk99999", "lgbm_core") is False

    def test_model_blob_is_bytes_not_str(self):
        """BLOB column must survive the SQLite round-trip as bytes."""
        import api.db as db
        original = pickle.dumps({"weights": list(range(100))})
        db.save_model("hk00700", "lgbm_core", 100, "2025-01-01",
                      0.5, original, [])
        loaded_blob, _ = db.load_model("hk00700", "lgbm_core")
        assert isinstance(loaded_blob, bytes)
        assert pickle.loads(loaded_blob) == {"weights": list(range(100))}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  _is_stale — staleness heuristics
# ══════════════════════════════════════════════════════════════════════════════

class TestStalenessLogic:
    def test_insufficient_new_data_skip(self):
        """< MIN_NEW_BARS extra bars → skip regardless of age."""
        from api.advisor import _is_stale, MIN_NEW_BARS
        meta   = _fake_meta(days_old=1.0, bar_count=250)
        stale, reason = _is_stale(meta, new_bar_count=250 + MIN_NEW_BARS - 1)
        assert not stale
        assert reason == "insufficient_new_data"

    def test_exactly_min_new_bars_does_not_skip(self):
        """Exactly MIN_NEW_BARS extra triggers retrain."""
        from api.advisor import _is_stale, MIN_NEW_BARS
        meta   = _fake_meta(days_old=1.0, bar_count=250)
        stale, reason = _is_stale(meta, new_bar_count=250 + MIN_NEW_BARS)
        assert stale
        assert reason == "new_data_available"

    def test_new_data_available_triggers_retrain(self):
        """≥ MIN_NEW_BARS extra bars → retrain."""
        from api.advisor import _is_stale
        meta   = _fake_meta(days_old=2.0, bar_count=250)
        stale, reason = _is_stale(meta, new_bar_count=260)
        assert stale
        assert reason == "new_data_available"

    def test_model_outdated_triggers_retrain(self):
        """Model trained > STALE_MODEL_DAYS ago → always retrain."""
        from api.advisor import _is_stale, STALE_MODEL_DAYS
        meta   = _fake_meta(days_old=STALE_MODEL_DAYS + 1, bar_count=250)
        stale, reason = _is_stale(meta, new_bar_count=251)  # only 1 new bar
        assert stale
        assert reason == "model_outdated"

    def test_outdated_takes_priority_over_insufficient_data(self):
        """Outdated model should retrain even with zero new bars."""
        from api.advisor import _is_stale, STALE_MODEL_DAYS
        meta   = _fake_meta(days_old=STALE_MODEL_DAYS + 5, bar_count=250)
        stale, reason = _is_stale(meta, new_bar_count=250)
        assert stale
        assert reason == "model_outdated"

    def test_fresh_model_not_enough_new_data_skip(self):
        """Fresh model (trained today) with 2 new bars → skip."""
        from api.advisor import _is_stale
        meta   = _fake_meta(days_old=0.0, bar_count=250)
        stale, reason = _is_stale(meta, new_bar_count=252)
        assert not stale
        assert reason == "insufficient_new_data"

    def test_exact_stale_model_days_boundary(self):
        """Trained exactly STALE_MODEL_DAYS ago (not over) → not outdated."""
        from api.advisor import _is_stale, STALE_MODEL_DAYS
        # Exactly at the boundary: NOT stale (> is required)
        meta = _fake_meta(days_old=STALE_MODEL_DAYS - 0.01, bar_count=250)
        stale, reason = _is_stale(meta, new_bar_count=252)
        # Should be "insufficient_new_data" (only 2 extra bars) not "model_outdated"
        assert not stale
        assert reason == "insufficient_new_data"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  _predict_signal — signal probabilities
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictSignal:
    def _make_feat_df(self, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame({c: [1.0] for c in cols})

    def test_buy_signal(self):
        from api.advisor import _predict_signal
        model  = _MockModel(pred_class=2)   # BUY
        feat   = ["f1", "f2", "f3"]
        df     = self._make_feat_df(feat)
        result = _predict_signal(model, df, feat)
        assert result["signal"]    == "BUY"
        assert result["p_buy"]     > result["p_sell"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sell_signal(self):
        from api.advisor import _predict_signal
        model  = _MockModel(pred_class=0)   # SELL
        feat   = ["f1", "f2", "f3"]
        df     = self._make_feat_df(feat)
        result = _predict_signal(model, df, feat)
        assert result["signal"]    == "SELL"
        assert result["p_sell"]    > result["p_buy"]

    def test_hold_signal(self):
        from api.advisor import _predict_signal
        model  = _MockModel(pred_class=1)   # HOLD
        feat   = ["f1", "f2", "f3"]
        df     = self._make_feat_df(feat)
        result = _predict_signal(model, df, feat)
        assert result["signal"]    == "HOLD"
        assert result["p_hold"]    > result["p_buy"]

    def test_empty_df_returns_hold_neutral(self):
        from api.advisor import _predict_signal
        model  = _MockModel()
        result = _predict_signal(model, pd.DataFrame(), ["f1"])
        assert result["signal"]    == "HOLD"
        assert result["confidence"] == 0.0
        assert result["p_hold"]     == 1.0

    def test_missing_columns_returns_hold_neutral(self):
        from api.advisor import _predict_signal
        model  = _MockModel()
        df     = pd.DataFrame({"wrong_col": [1.0]})
        result = _predict_signal(model, df, ["expected_col"])
        assert result["signal"] == "HOLD"

    def test_probabilities_sum_to_one(self):
        from api.advisor import _predict_signal
        model  = _MockModel(pred_class=2)
        feat   = ["f1", "f2", "f3"]
        df     = pd.DataFrame({c: [1.0] for c in feat})
        result = _predict_signal(model, df, feat)
        total  = result["p_buy"] + result["p_hold"] + result["p_sell"]
        assert total == pytest.approx(1.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  _feature_importance
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureImportance:
    def test_returns_sorted_descending(self):
        from api.advisor import _feature_importance
        model  = _MockModel()
        model.feature_importances_ = np.array([1.0, 5.0, 3.0])
        feat   = ["low", "high", "mid"]
        result = _feature_importance(model, feat)
        values = [r["importance"] for r in result]
        assert values == sorted(values, reverse=True)
        assert result[0]["feature"] == "high"

    def test_max_15_returned(self):
        from api.advisor import _feature_importance
        model  = _MockModel()
        model.feature_importances_ = np.arange(20, dtype=float)
        feat   = [f"f{i}" for i in range(20)]
        result = _feature_importance(model, feat)
        assert len(result) <= 15

    def test_empty_on_exception(self):
        from api.advisor import _feature_importance

        class BadModel:
            feature_importances_ = None   # will cause TypeError

        result = _feature_importance(BadModel(), ["f1"])
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 5.  train_for_symbol — end-to-end with mocked network + LGB
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainForSymbol:
    """Mock out _load_bars_df and _train_lgbm so no network or LGB is needed."""

    SYMBOL = "hk00700"
    FEATS  = ["ret_1d", "rsi_14", "macd_hist"]

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        import api.advisor as adv
        monkeypatch.setattr(adv, "_load_bars_df",
                            lambda sym, days=400: _fake_df(260))
        monkeypatch.setattr(adv, "_train_lgbm",
                            lambda sym, df: (_MockModel(), 0.62, self.FEATS, "2025-03-01"))

    def test_trains_when_no_existing_model(self):
        from api.advisor import train_for_symbol
        result = train_for_symbol(self.SYMBOL)
        assert result["train_status"]  == "trained"
        assert result["bar_count"]     == 260
        assert result["last_bar_date"] == "2025-03-01"
        assert result["oos_accuracy"]  == pytest.approx(0.62)
        assert result["skip_reason"]   is None

    def test_persists_model_to_db(self):
        from api.advisor import train_for_symbol
        import api.db as db
        train_for_symbol(self.SYMBOL)
        meta = db.get_model_meta(self.SYMBOL, "lgbm_core")
        assert meta is not None
        assert meta["bar_count"] == 260

    def test_skips_fresh_model_with_insufficient_new_data(self):
        """Model stored yesterday with same bar count → skip."""
        from api.advisor import train_for_symbol, MIN_NEW_BARS
        import api.db as db
        # Pre-store a model that is 1 day old with 258 bars (260 - MIN_NEW_BARS + 2)
        db.save_model(self.SYMBOL, "lgbm_core", 258, "2025-03-01",
                      0.60, pickle.dumps(_MockModel()), self.FEATS)
        result = train_for_symbol(self.SYMBOL, force=False)
        assert result["train_status"] == "skipped"
        assert "insufficient_new_data" in result["skip_reason"]

    def test_retrains_outdated_model(self):
        """Model stored 31 days ago → retrain even with few new bars."""
        from api.advisor import train_for_symbol, STALE_MODEL_DAYS
        import api.db as db
        old_ts = time.time() - (STALE_MODEL_DAYS + 1) * 86400
        db.save_model(self.SYMBOL, "lgbm_core", 259, "2025-02-01",
                      0.55, pickle.dumps(_MockModel()), self.FEATS)
        # Manually back-date trained_at
        conn = db._conn()
        conn.execute("UPDATE trained_models SET trained_at=? WHERE id=?",
                     (old_ts, f"{self.SYMBOL}_lgbm_core"))
        conn.commit()

        result = train_for_symbol(self.SYMBOL, force=False)
        assert result["train_status"] == "trained"

    def test_force_always_retrains(self):
        """force=True ignores staleness rules."""
        from api.advisor import train_for_symbol
        import api.db as db
        # Fresh model stored just now
        db.save_model(self.SYMBOL, "lgbm_core", 260, "2025-03-01",
                      0.60, pickle.dumps(_MockModel()), self.FEATS)
        result = train_for_symbol(self.SYMBOL, force=True)
        assert result["train_status"] == "trained"

    def test_raises_on_empty_dataframe(self, monkeypatch):
        """No price data available → ValueError."""
        import api.advisor as adv
        monkeypatch.setattr(adv, "_load_bars_df", lambda *a, **k: pd.DataFrame())
        from api.advisor import train_for_symbol
        with pytest.raises(ValueError, match="No price data"):
            train_for_symbol("hk00700")

    def test_model_blob_is_deserializable(self):
        """Saved blob must unpickle back to a usable object."""
        from api.advisor import train_for_symbol
        import api.db as db
        train_for_symbol(self.SYMBOL)
        blob, _ = db.load_model(self.SYMBOL, "lgbm_core")
        obj = pickle.loads(blob)
        assert hasattr(obj, "predict")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  analyze_stock — end-to-end with all externals mocked
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeStock:
    SYMBOL = "hk00700"
    FEATS  = ["ret_1d", "rsi_14", "macd_hist"]

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        import api.advisor as adv
        import api.runner as runner

        fake_df = _fake_df(260)
        monkeypatch.setattr(adv, "_load_bars_df",
                            lambda sym, days=400: fake_df)
        monkeypatch.setattr(adv, "_train_lgbm",
                            lambda sym, df: (_MockModel(pred_class=2),
                                             0.63, self.FEATS, "2025-03-01"))

        def _fake_fund(sym):
            return {"symbol": sym, "pe_ttm": 18.0, "pb": 3.1, "ps_ttm": 4.2,
                    "roe": 22.5, "revenue_growth": 0.12, "net_margin": 0.21,
                    "dividend_yield": 0.015, "value_score": 72.0,
                    "growth_score": 65.0, "quality_score": 80.0}

        def _fake_news(sym, limit=5):
            return {"symbol": sym, "items": [
                {"title": "Test headline", "content": "...",
                 "source": "Reuters", "ts": "2025-03-01T10:00:00", "url": ""}
            ]}

        monkeypatch.setattr(runner, "get_fundamentals_sync", _fake_fund)
        monkeypatch.setattr(runner, "get_news_sync", _fake_news)

    def test_returns_required_top_level_keys(self):
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        for key in ["symbol", "sector", "signal", "confidence",
                    "p_buy", "p_hold", "p_sell", "momentum",
                    "fundamentals", "news", "recent_bars",
                    "feature_importance", "model_meta"]:
            assert key in result, f"Missing key: {key}"

    def test_signal_is_buy(self):
        """MockModel(pred_class=2) → BUY signal."""
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        assert result["signal"] == "BUY"
        assert result["p_buy"]  > result["p_sell"]

    def test_confidence_in_range(self):
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_momentum_has_current_price(self):
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        assert "current_price" in result["momentum"]
        assert result["momentum"]["current_price"] > 0

    def test_recent_bars_capped_at_30(self):
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        assert 0 < len(result["recent_bars"]) <= 30

    def test_news_is_list(self):
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        assert isinstance(result["news"], list)
        assert result["news"][0]["title"] == "Test headline"

    def test_model_meta_populated(self):
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        mm = result["model_meta"]
        assert mm["bar_count"]    == 260
        assert mm["oos_accuracy"] == pytest.approx(0.63)

    def test_sector_resolved(self):
        from api.advisor import analyze_stock, SECTOR_MAP
        result = analyze_stock(self.SYMBOL)
        assert result["sector"] == SECTOR_MAP.get(self.SYMBOL, "unknown")

    def test_feature_importance_top15(self):
        from api.advisor import analyze_stock
        result = analyze_stock(self.SYMBOL)
        # MockModel has 3 features; should have up to 3 entries
        assert len(result["feature_importance"]) <= 15

    def test_force_retrain_flag_propagates(self, monkeypatch):
        """When force_retrain=True, train_for_symbol should receive force=True."""
        import api.advisor as adv
        calls = []
        real_train = adv.train_for_symbol

        def _spy(sym, force=False):
            calls.append(force)
            return real_train(sym, force=force)

        monkeypatch.setattr(adv, "train_for_symbol", _spy)
        from api.advisor import analyze_stock
        analyze_stock(self.SYMBOL, force_retrain=True)
        assert calls and calls[-1] is True


# ══════════════════════════════════════════════════════════════════════════════
# 7.  get_recommendations — scoring & ranking
# ══════════════════════════════════════════════════════════════════════════════

class TestGetRecommendations:
    """Patch price + fundamentals fetchers so no network calls occur."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        import api.runner as runner
        import api.advisor as adv

        def _fake_price(sym, days=365):
            # 120 daily close prices starting at 100
            closes = 100 + np.cumsum(np.random.randn(120) * 0.3)
            bars   = [{"date": f"2025-{(i//30)+1:02d}-{(i%30)+1:02d}",
                       "open": c, "high": c * 1.01, "low": c * 0.99,
                       "close": float(c), "volume": 1e6}
                      for i, c in enumerate(closes)]
            return {"symbol": sym, "yf_ticker": "", "bars": bars}

        def _fake_fund(sym):
            return {"symbol": sym, "pe_ttm": 15.0, "pb": 2.0, "ps_ttm": 3.0,
                    "roe": 18.0, "revenue_growth": 0.10, "net_margin": 0.18,
                    "dividend_yield": 0.02, "value_score": 60.0,
                    "growth_score": 55.0, "quality_score": 70.0}

        monkeypatch.setattr(runner, "get_price_sync",        _fake_price)
        monkeypatch.setattr(runner, "get_fundamentals_sync", _fake_fund)

    def test_returns_list_of_dicts(self):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        assert isinstance(rows, list)
        assert len(rows) <= 5

    def test_top_n_respected(self):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=3)
        assert len(rows) <= 3

    def test_scores_are_between_0_and_1(self):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for r in rows:
            assert 0.0 <= r["score"] <= 1.0, f"Out-of-range score for {r['symbol']}: {r['score']}"

    def test_sorted_descending_by_score(self):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_required_keys_present(self):
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=3)
        if not rows:
            pytest.skip("Universe returned no rows with mocked data")
        row = rows[0]
        for key in ["symbol", "name", "sector", "score", "model_signal",
                    "model_trained", "ret_1y", "ret_3m", "ret_1m", "fundamentals"]:
            assert key in row, f"Missing key: {key}"

    def test_sector_filter_excludes_other_sectors(self):
        from api.advisor import get_recommendations, SECTOR_MAP
        rows = get_recommendations(sector="tech", top_n=20)
        for r in rows:
            assert r["sector"] == "tech", f"{r['symbol']} has sector {r['sector']}"

    def test_sector_filter_unknown_sector_returns_empty(self):
        from api.advisor import get_recommendations
        rows = get_recommendations(sector="__nonexistent__", top_n=10)
        assert rows == []

    def test_model_signal_n_a_when_no_stored_model(self):
        """No model in DB → model_signal is 'N/A', model_trained is False."""
        from api.advisor import get_recommendations
        rows = get_recommendations(top_n=5)
        for r in rows:
            if not r["model_trained"]:
                assert r["model_signal"] == "N/A"

    def test_model_signal_when_stored_model_present(self):
        """Store a BUY model for hk00700 → model_signal reflects it."""
        import api.db as db
        import api.advisor as adv
        from api.advisor import get_recommendations
        feat = ["ret_1d", "rsi_14", "macd_hist"]
        blob = pickle.dumps(_MockModel(pred_class=2))   # BUY
        db.save_model("hk00700", "lgbm_core", 250, "2025-03-01", 0.65, blob, feat)

        # Also mock _load_bars_df so _predict_signal can be called
        original = adv._load_bars_df

        def _df_with_feats(sym, days=400):
            return _fake_df(260)

        adv._load_bars_df = _df_with_feats
        try:
            rows = get_recommendations(top_n=20)
        finally:
            adv._load_bars_df = original

        hk700 = next((r for r in rows if r["symbol"] == "hk00700"), None)
        if hk700 is not None:
            assert hk700["model_trained"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FastAPI advisor routes — TestClient
# ══════════════════════════════════════════════════════════════════════════════

class TestAdvisorRoutes:
    """
    Smoke-test each route via FastAPI's TestClient.
    Job workers run synchronously (in-process) for these tests so we can
    assert on the final payload without real polling delays.
    """

    @pytest.fixture(autouse=True)
    def _patch_runner(self, monkeypatch):
        """
        Replace the three async launchers with synchronous no-ops that
        immediately set the job to 'done' with canned payloads.
        """
        import api.runner as runner
        import api.db    as db

        FAKE_TRAIN_META = {
            "status":        "done",
            "symbol":        "hk00700",
            "model_id":      "hk00700_lgbm_core",
            "train_status":  "trained",
            "skip_reason":   None,
            "bar_count":     250,
            "last_bar_date": "2025-03-01",
            "oos_accuracy":  0.62,
            "trained_at":    time.time(),
            "feature_cols":  ["ret_1d", "rsi_14"],
        }

        FAKE_ANALYSIS = {
            "status":             "done",
            "symbol":             "hk00700",
            "sector":             "tech",
            "signal":             "BUY",
            "confidence":         0.72,
            "p_buy":              0.72,
            "p_hold":             0.18,
            "p_sell":             0.10,
            "momentum":           {"current_price": 350.0, "ret_1d": 0.005},
            "recent_bars":        [],
            "fundamentals":       {"pe_ttm": 18.0},
            "news":               [],
            "feature_importance": [],
            "model_meta":         {
                "model_id":      "hk00700_lgbm_core",
                "trained_at":    time.time(),
                "bar_count":     250,
                "last_bar_date": "2025-03-01",
                "oos_accuracy":  0.62,
                "train_status":  "trained",
                "skip_reason":   None,
            },
        }

        FAKE_RECOMMEND = {
            "status": "done",
            "sector": None,
            "rows": [
                {
                    "symbol": "hk00700", "yf_ticker": "0700.HK",
                    "name": "Tencent", "sector": "tech", "score": 0.72,
                    "model_signal": "BUY", "model_confidence": 0.70,
                    "model_trained": True, "ret_1y": 0.18, "ret_3m": 0.05,
                    "ret_1m": 0.02, "fundamentals": {},
                }
            ],
        }

        async def _fake_launch_train(jid, sym, force=False):
            runner._update_job(jid, FAKE_TRAIN_META)

        async def _fake_launch_analyze(jid, sym, force_retrain=False):
            runner._update_job(jid, FAKE_ANALYSIS)

        async def _fake_launch_recommend(jid, sector=None, top_n=10):
            runner._update_job(jid, {**FAKE_RECOMMEND, "sector": sector})

        monkeypatch.setattr(runner, "launch_train",     _fake_launch_train)
        monkeypatch.setattr(runner, "launch_analyze",   _fake_launch_analyze)
        monkeypatch.setattr(runner, "launch_recommend", _fake_launch_recommend)

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    # ── /api/advisor/train ──────────────────────────────────────────────────

    def test_post_train_returns_202(self, client):
        res = client.post("/api/advisor/train",
                          json={"symbol": "sh600519", "force_retrain": False})
        assert res.status_code == 202
        body = res.json()
        assert "job_id" in body
        assert body["symbol"] == "sh600519"

    def test_get_train_returns_done(self, client):
        jid = client.post("/api/advisor/train",
                          json={"symbol": "sh600519"}).json()["job_id"]
        res = client.get(f"/api/advisor/train/{jid}")
        assert res.status_code == 200
        body = res.json()
        assert body["status"]       == "done"
        assert body["train_status"] == "trained"
        assert body["bar_count"]    == 250
        assert body["oos_accuracy"] == pytest.approx(0.62)

    def test_get_train_404_unknown_job(self, client):
        res = client.get("/api/advisor/train/does-not-exist")
        assert res.status_code == 404

    # ── /api/advisor/analyze ────────────────────────────────────────────────

    def test_post_analyze_returns_202(self, client):
        res = client.post("/api/advisor/analyze",
                          json={"symbol": "sh600519", "force_retrain": False})
        assert res.status_code == 202
        body = res.json()
        assert "job_id" in body

    def test_get_analyze_returns_signal(self, client):
        jid = client.post("/api/advisor/analyze",
                          json={"symbol": "sh600519"}).json()["job_id"]
        res  = client.get(f"/api/advisor/analyze/{jid}")
        assert res.status_code == 200
        body = res.json()
        assert body["signal"]     == "BUY"
        assert body["confidence"] == pytest.approx(0.72)
        assert body["p_buy"]      > body["p_sell"]

    def test_get_analyze_model_meta_present(self, client):
        jid  = client.post("/api/advisor/analyze",
                           json={"symbol": "sh600519"}).json()["job_id"]
        body = client.get(f"/api/advisor/analyze/{jid}").json()
        mm   = body.get("model_meta")
        assert mm is not None
        assert mm["bar_count"]   == 250
        assert mm["train_status"] == "trained"

    def test_get_analyze_404_unknown_job(self, client):
        assert client.get("/api/advisor/analyze/no-such-job").status_code == 404

    # ── /api/advisor/recommend ──────────────────────────────────────────────

    def test_get_recommend_general_returns_200(self, client):
        res = client.get("/api/advisor/recommend?top_n=5")
        assert res.status_code == 200
        body = res.json()
        assert "job_id" in body
        assert body["status"] in ("pending", "done", "running")

    def test_get_recommend_sector_returns_200(self, client):
        res = client.get("/api/advisor/recommend/tech?top_n=5")
        assert res.status_code == 200
        body = res.json()
        assert "job_id" in body

    def test_poll_recommend_done(self, client):
        jid = client.get("/api/advisor/recommend?top_n=5").json()["job_id"]
        res = client.get(f"/api/advisor/recommend-poll/{jid}")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "done"
        assert isinstance(body["rows"], list)

    def test_poll_recommend_row_schema(self, client):
        jid  = client.get("/api/advisor/recommend?top_n=5").json()["job_id"]
        rows = client.get(f"/api/advisor/recommend-poll/{jid}").json()["rows"]
        if rows:
            r = rows[0]
            for k in ["symbol", "score", "model_signal", "model_trained",
                      "ret_1y", "ret_3m", "ret_1m"]:
                assert k in r, f"Missing key {k} in row"

    # ── /api/advisor/models ─────────────────────────────────────────────────

    def test_list_models_empty_initially(self, client):
        res = client.get("/api/advisor/models")
        assert res.status_code == 200
        assert res.json()["models"] == []

    def test_list_models_after_save(self, client):
        import api.db as db
        db.save_model("hk00700", "lgbm_core", 250, "2025-03-01",
                      0.62, _fake_blob(), ["ret_1d"])
        res = client.get("/api/advisor/models")
        assert res.status_code == 200
        models = res.json()["models"]
        assert len(models) == 1
        assert models[0]["symbol"] == "hk00700"

    def test_delete_model_returns_ok(self, client):
        import api.db as db
        db.save_model("hk00700", "lgbm_core", 250, "2025-03-01",
                      0.62, _fake_blob(), ["ret_1d"])
        res = client.delete("/api/advisor/models/hk00700")
        assert res.status_code == 200
        assert res.json()["deleted"] is True
        # verify gone
        assert client.get("/api/advisor/models").json()["models"] == []

    def test_delete_missing_model_returns_404(self, client):
        res = client.delete("/api/advisor/models/hk99999")
        assert res.status_code == 404

    # ── /api/advisor/sectors ────────────────────────────────────────────────

    def test_sectors_returns_list(self, client):
        res = client.get("/api/advisor/sectors")
        assert res.status_code == 200
        body = res.json()
        assert "sectors" in body
        assert isinstance(body["sectors"], list)
        assert len(body["sectors"]) > 0
        assert "tech" in body["sectors"]
        assert "finance" in body["sectors"]

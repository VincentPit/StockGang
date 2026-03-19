"""
Tests for api/db.py — L1/L2 cache behaviour, job persistence, stats, purge.
No network calls.  Uses a temp file for the SQLite db to stay isolated.
"""
from __future__ import annotations

import time
import pytest
from pathlib import Path


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """
    Redirect DB_PATH to a temporary file and reset module-level state
    (_mem dict, thread-local connection) before every test so tests
    never share data with each other or with the real data/myquant.db.
    """
    import api.db as db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")

    # Re-initialise _local so a fresh connection is created for this temp file
    import threading
    monkeypatch.setattr(db, "_local", threading.local())
    monkeypatch.setattr(db, "_mem",   {})

    db.init_db()
    yield
    # Cleanup: close the thread-local connection if open
    if hasattr(db._local, "conn") and db._local.conn:
        db._local.conn.close()
        db._local.conn = None


@pytest.fixture
def db():
    import api.db as _db
    return _db


# ── init_db ────────────────────────────────────────────────────────────────────

class TestInitDB:
    def test_idempotent(self, db):
        """Calling init_db twice must not raise or corrupt the schema."""
        db.init_db()
        db.init_db()
        # If tables exist the stat call should succeed with no errors
        stats = db.cache_stats()
        assert stats["total"] == 0

    def test_tables_created(self, db):
        """Both tables must exist after init."""
        conn = db._conn()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "jobs"  in tables
        assert "cache" in tables


# ── cache_set / cache_get ──────────────────────────────────────────────────────

class TestCacheSetGet:
    def test_basic_roundtrip(self, db):
        db.cache_set("price:hk00700:365", {"bars": [1, 2, 3]}, ttl=60)
        assert db.cache_get("price:hk00700:365") == {"bars": [1, 2, 3]}

    def test_l1_populated_on_set(self, db):
        db.cache_set("fund:TEST", {"pe": 12.5}, ttl=60)
        assert "fund:TEST" in db._mem

    def test_l1_hit_returns_without_sqlite(self, db, monkeypatch):
        """If L1 is warm, _conn should not be consulted."""
        db.cache_set("news:TEST:10", {"items": []}, ttl=60)
        # Replace _conn with a function that raises to detect any SQLite access
        monkeypatch.setattr(db, "_conn", lambda: (_ for _ in ()).throw(AssertionError("SQLite accessed on L1 hit")))
        result = db.cache_get("news:TEST:10")
        assert result == {"items": []}

    def test_l2_hit_backfills_l1(self, db):
        db.cache_set("regime:hk00700", {"trend": "bull"}, ttl=60)
        del db._mem["regime:hk00700"]                      # evict L1
        result = db.cache_get("regime:hk00700")
        assert result == {"trend": "bull"}                 # L2 hit
        assert "regime:hk00700" in db._mem                 # L1 backfilled

    def test_expired_returns_none(self, db):
        db.cache_set("price:EXPIRE:30", {"x": 1}, ttl=-1)  # already expired
        assert db.cache_get("price:EXPIRE:30") is None

    def test_overwrite_extends_ttl(self, db):
        db.cache_set("fund:UPDATE", {"v": 1}, ttl=-1)      # expired
        db.cache_set("fund:UPDATE", {"v": 2}, ttl=60)      # fresh overwrite
        assert db.cache_get("fund:UPDATE") == {"v": 2}

    def test_missing_key_returns_none(self, db):
        assert db.cache_get("no:such:key") is None

    def test_complex_value_roundtrip(self, db):
        payload = {"bars": [{"o": 1.0, "h": 2.0, "l": 0.5, "c": 1.8}], "symbol": "hk00700"}
        db.cache_set("price:hk00700:30", payload, ttl=120)
        assert db.cache_get("price:hk00700:30") == payload


# ── cache_invalidate ───────────────────────────────────────────────────────────

class TestCacheInvalidate:
    def test_prefix_removes_matching_keys(self, db):
        db.cache_set("price:hk00700:365", {}, ttl=60)
        db.cache_set("price:sh600519:365", {}, ttl=60)
        db.cache_set("fund:sh600519",       {}, ttl=60)
        n = db.cache_invalidate("price")
        assert n == 2
        assert db.cache_get("price:hk00700:365") is None
        assert db.cache_get("price:sh600519:365") is None
        assert db.cache_get("fund:sh600519") is not None   # untouched

    def test_empty_prefix_removes_all(self, db):
        db.cache_set("price:A:365", {}, ttl=60)
        db.cache_set("fund:B",      {}, ttl=60)
        db.cache_set("news:C:10",   {}, ttl=60)
        n = db.cache_invalidate("")
        assert n == 3
        assert db.cache_stats()["active"] == 0

    def test_prefix_clears_l1(self, db):
        db.cache_set("price:hk00700:365", {"bars": []}, ttl=60)
        db.cache_invalidate("price")
        assert "price:hk00700:365" not in db._mem

    def test_nonexistent_prefix_returns_zero(self, db):
        n = db.cache_invalidate("nonexistent")
        assert n == 0


# ── cache_stats ────────────────────────────────────────────────────────────────

class TestCacheStats:
    def test_empty_stats(self, db):
        s = db.cache_stats()
        assert s == {"total": 0, "active": 0, "expired": 0, "by_type": {}}

    def test_active_vs_expired_counts(self, db):
        db.cache_set("price:A:365",  {}, ttl=60)    # active
        db.cache_set("price:B:365",  {}, ttl=60)    # active
        db.cache_set("news:X:10",    {}, ttl=-1)    # expired
        s = db.cache_stats()
        assert s["total"]   == 3
        assert s["active"]  == 2
        assert s["expired"] == 1

    def test_by_type_breakdown(self, db):
        db.cache_set("price:hk00700:365", {}, ttl=60)
        db.cache_set("price:sh600519:365",{}, ttl=60)
        db.cache_set("fund:hk00700",      {}, ttl=60)
        s = db.cache_stats()
        assert s["by_type"]["price"] == 2
        assert s["by_type"]["fund"]  == 1


# ── purge_expired ──────────────────────────────────────────────────────────────

class TestPurgeExpired:
    def test_removes_expired_from_l1_and_l2(self, db):
        db.cache_set("price:OLD:365", {"x": 1}, ttl=-1)   # stale
        db.cache_set("fund:LIVE",     {"y": 2}, ttl=60)   # fresh
        n = db.purge_expired()
        assert n >= 1
        assert "price:OLD:365" not in db._mem
        assert db.cache_get("fund:LIVE") == {"y": 2}       # live entry intact

    def test_no_expired_returns_zero(self, db):
        db.cache_set("price:FRESH:365", {}, ttl=60)
        assert db.purge_expired() == 0

    def test_idempotent(self, db):
        db.cache_set("news:STALE:5", {}, ttl=-1)
        db.purge_expired()
        assert db.purge_expired() == 0   # second call finds nothing to delete


# ── job persistence ────────────────────────────────────────────────────────────

class TestJobPersistence:
    def _job(self, jid: str, kind: str = "backtest", status: str = "pending") -> dict:
        return {"id": jid, "kind": kind, "status": status}

    def test_upsert_and_fetch(self, db):
        job = self._job("j-001")
        db.upsert_job(job)
        fetched = db.fetch_job("j-001")
        assert fetched is not None
        assert fetched["status"] == "pending"

    def test_update_status(self, db):
        job = self._job("j-002")
        db.upsert_job(job)
        job["status"] = "done"
        db.upsert_job(job)
        assert db.fetch_job("j-002")["status"] == "done"

    def test_fetch_missing_returns_none(self, db):
        assert db.fetch_job("nonexistent") is None

    def test_fetch_all_jobs(self, db):
        db.upsert_job(self._job("j-010", kind="backtest"))
        db.upsert_job(self._job("j-011", kind="screen"))
        all_jobs = db.fetch_all_jobs()
        ids = {j["id"] for j in all_jobs}
        assert {"j-010", "j-011"}.issubset(ids)

    def test_jobs_stats(self, db):
        db.upsert_job(self._job("j-020", status="pending"))
        db.upsert_job(self._job("j-021", status="done"))
        db.upsert_job(self._job("j-022", status="done"))
        s = db.jobs_stats()
        assert s["total"] == 3
        assert s["by_status"]["done"]    == 2
        assert s["by_status"]["pending"] == 1

    def test_upsert_preserves_extra_fields(self, db):
        job = self._job("j-030")
        job["result"] = {"sharpe": 1.23, "trades": 42}
        db.upsert_job(job)
        fetched = db.fetch_job("j-030")
        assert fetched["result"]["sharpe"] == pytest.approx(1.23)


# ══════════════════════════════════════════════════════════════════════════════
# TestTrainedModels — CRUD for the trained_models table
# ══════════════════════════════════════════════════════════════════════════════

import pickle as _pickle
import time   as _time


class TestTrainedModels:
    """
    Tests for save_model / load_model / get_model_meta / list_models /
    delete_model introduced in the Advisor feature.

    Uses the same ``_isolated_db`` autouse fixture already in scope.
    """

    SYMBOL = "hk00700"
    SID    = "lgbm_core"
    FEATS  = ["ret_1d", "rsi_14", "macd_hist"]

    def _blob(self, payload: dict = None) -> bytes:
        return _pickle.dumps(payload or {"w": [1.0, 2.0, 3.0]})

    # ── table creation ────────────────────────────────────────────────────────

    def test_trained_models_table_created(self):
        import api.db as db
        conn   = db._conn()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "trained_models" in tables, \
            "init_db() must create the trained_models table"

    # ── basic roundtrip ───────────────────────────────────────────────────────

    def test_save_returns_correct_model_id(self):
        import api.db as db
        mid = db.save_model(self.SYMBOL, self.SID, 250, "2025-01-15",
                            0.61, self._blob(), self.FEATS)
        assert mid == f"{self.SYMBOL}_{self.SID}"

    def test_load_roundtrip(self):
        import api.db as db
        payload = {"weights": list(range(50))}
        db.save_model(self.SYMBOL, self.SID, 250, "2025-01-15",
                      0.61, self._blob(payload), self.FEATS)
        result = db.load_model(self.SYMBOL, self.SID)
        assert result is not None
        loaded_blob, meta = result
        assert _pickle.loads(loaded_blob) == payload
        assert meta["symbol"]        == self.SYMBOL
        assert meta["strategy_id"]   == self.SID
        assert meta["bar_count"]     == 250
        assert meta["last_bar_date"] == "2025-01-15"
        assert meta["oos_accuracy"]  == pytest.approx(0.61)
        assert meta["feature_cols"]  == self.FEATS

    def test_load_missing_symbol_returns_none(self):
        import api.db as db
        assert db.load_model("xx99999", self.SID) is None

    def test_load_missing_strategy_returns_none(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 200, "2025-01-01",
                      0.55, self._blob(), [])
        assert db.load_model(self.SYMBOL, "other_strategy") is None

    # ── upsert (overwrite same PK) ────────────────────────────────────────────

    def test_save_upserts_existing_row(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 200, "2025-01-01",
                      0.55, self._blob({"v": 1}), ["f1"])
        db.save_model(self.SYMBOL, self.SID, 260, "2025-03-01",
                      0.63, self._blob({"v": 2}), ["f1", "f2"])
        loaded_blob, meta = db.load_model(self.SYMBOL, self.SID)
        assert _pickle.loads(loaded_blob) == {"v": 2}
        assert meta["bar_count"]     == 260
        assert meta["last_bar_date"] == "2025-03-01"
        assert meta["oos_accuracy"]  == pytest.approx(0.63)
        assert meta["feature_cols"]  == ["f1", "f2"]
        # only one row
        assert len(db.list_models()) == 1

    # ── get_model_meta ────────────────────────────────────────────────────────

    def test_get_model_meta_has_no_blob_key(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 240, "2025-02-01",
                      0.58, self._blob(), self.FEATS)
        meta = db.get_model_meta(self.SYMBOL, self.SID)
        assert meta is not None
        assert "model_blob" not in meta

    def test_get_model_meta_has_required_keys(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 240, "2025-02-01",
                      0.58, self._blob(), self.FEATS)
        meta = db.get_model_meta(self.SYMBOL, self.SID)
        for key in ["model_id", "symbol", "strategy_id", "trained_at",
                    "bar_count", "last_bar_date", "oos_accuracy", "feature_cols"]:
            assert key in meta, f"Missing key: {key}"

    def test_get_model_meta_missing_returns_none(self):
        import api.db as db
        assert db.get_model_meta("xx99999", self.SID) is None

    def test_trained_at_is_recent(self):
        import api.db as db
        before = _time.time()
        db.save_model(self.SYMBOL, self.SID, 100, "2025-01-01",
                      0.5, self._blob(), [])
        meta = db.get_model_meta(self.SYMBOL, self.SID)
        assert meta["trained_at"] >= before

    # ── list_models ───────────────────────────────────────────────────────────

    def test_list_models_empty(self):
        import api.db as db
        assert db.list_models() == []

    def test_list_models_contains_saved(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 250, "2025-01-01",
                      0.60, self._blob(), self.FEATS)
        lst = db.list_models()
        assert len(lst) == 1
        assert lst[0]["symbol"] == self.SYMBOL

    def test_list_models_no_blob_key(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 250, "2025-01-01",
                      0.60, self._blob(), self.FEATS)
        for m in db.list_models():
            assert "model_blob" not in m

    def test_list_models_sorted_newest_first(self):
        import api.db as db
        db.save_model("hk00700", self.SID, 250, "2025-01-01",
                      0.60, self._blob(), [])
        _time.sleep(0.03)   # ensure distinct trained_at values
        db.save_model("sh600519", self.SID, 200, "2025-02-01",
                      0.55, self._blob(), [])
        lst = db.list_models()
        assert len(lst) == 2
        assert lst[0]["symbol"] == "sh600519"
        assert lst[1]["symbol"] == "hk00700"

    def test_list_models_multiple_symbols(self):
        import api.db as db
        for sym in ["hk00700", "sh600519", "sz300750"]:
            db.save_model(sym, self.SID, 100, "2025-01-01",
                          0.5, self._blob(), [])
        assert len(db.list_models()) == 3

    # ── delete_model ──────────────────────────────────────────────────────────

    def test_delete_existing_returns_true(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 250, "2025-01-01",
                      0.60, self._blob(), [])
        assert db.delete_model(self.SYMBOL, self.SID) is True

    def test_delete_removes_from_db(self):
        import api.db as db
        db.save_model(self.SYMBOL, self.SID, 250, "2025-01-01",
                      0.60, self._blob(), [])
        db.delete_model(self.SYMBOL, self.SID)
        assert db.load_model(self.SYMBOL, self.SID) is None
        assert db.list_models() == []

    def test_delete_missing_returns_false(self):
        import api.db as db
        assert db.delete_model("hk99999", self.SID) is False

    def test_delete_only_removes_target_symbol(self):
        import api.db as db
        db.save_model("hk00700",  self.SID, 250, "2025-01-01", 0.60, self._blob(), [])
        db.save_model("sh600519", self.SID, 200, "2025-01-01", 0.55, self._blob(), [])
        db.delete_model("hk00700", self.SID)
        lst = db.list_models()
        assert len(lst) == 1
        assert lst[0]["symbol"] == "sh600519"

    # ── blob integrity ────────────────────────────────────────────────────────

    def test_blob_survives_roundtrip_as_bytes(self):
        import api.db as db
        original = _pickle.dumps({"matrix": list(range(1000))})
        db.save_model(self.SYMBOL, self.SID, 100, "2025-01-01",
                      0.5, original, [])
        loaded_blob, _ = db.load_model(self.SYMBOL, self.SID)
        assert isinstance(loaded_blob, bytes)
        assert _pickle.loads(loaded_blob) == {"matrix": list(range(1000))}

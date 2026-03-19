"""
api/db.py — SQLite persistence layer for MyQuant.

Two responsibilities
--------------------
  1. Jobs table  — write-through mirror of the in-memory _jobs dict so that
                   job history survives server restarts.
  2. Cache table — two-level TTL cache for data fetched from external APIs
                   (yfinance / AKShare) so repeated requests are served from
                   memory and never hit the network or disk.

Two-level cache
---------------
  L1  _mem  dict[str, (value, expires_at)]  — pure Python, zero I/O.
            Populated on every cache_set() call.  A cache_get() hit here
            returns immediately without touching SQLite.

  L2  SQLite cache table  — survives server restarts.  On an L1 miss the
            row is read from SQLite and backfilled into L1 so subsequent
            reads are served from memory.

  Both layers are invalidated together by cache_invalidate() and
  purge_expired(), keeping them consistent.

Thread safety
-------------
  SQLite connections are cheap — each OS thread gets its own via
  threading.local().  WAL mode lets readers proceed while a write is
  committed, so backtest threads never block API request threads.

  _wlock serialises all writes (to _mem and SQLite together).  Plain
  _mem.get() reads are GIL-safe and need no lock.

Cache TTLs (informational — callers decide)
-------------------------------------------
  price        4 h   (OHLCV bars, refreshes once per trading day)
  fundamentals 24 h  (quarterly reports)
  news         30 m  (headlines change frequently)
  regime        4 h  (daily-bar computation)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

DB_PATH   = Path("data/myquant.db")
_local    = threading.local()
_wlock    = threading.Lock()          # serialise all writes (L1 + L2)

# L1 in-memory cache: key → (value, expires_at_unix_seconds)
# Plain dict reads are GIL-safe; mutations always happen inside _wlock.
_mem: dict[str, tuple[Any, float]] = {}


# ── Connection factory ────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """Return the thread-local SQLite connection, creating (or re-creating) one if needed."""
    conn = getattr(_local, "conn", None)
    # Re-open if the connection was closed externally
    if conn is not None:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            conn = None
            _local.conn = None
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
        c.row_factory = sqlite3.Row
        # WAL = readers never block writers; NORMAL sync is safe for our use case
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA cache_size=-8000")   # 8 MB page cache per connection
        c.execute("PRAGMA temp_store=MEMORY")
        c.execute("PRAGMA busy_timeout=5000")  # 5 s retry window on SQLITE_BUSY
        _local.conn = c
    return _local.conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables and indexes.  Safe to call multiple times (IF NOT EXISTS)."""
    with _wlock:
        conn = _conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                kind        TEXT NOT NULL,
                status      TEXT NOT NULL,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                payload     TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_kind    ON jobs(kind);
            CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at DESC);

            CREATE TABLE IF NOT EXISTS cache (
                cache_key   TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                expires_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_exp ON cache(expires_at);

            -- Paper-broker persistent state (single-row tables)
            CREATE TABLE IF NOT EXISTS paper_account (
                id   INTEGER PRIMARY KEY DEFAULT 1,
                cash REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_positions (
                symbol    TEXT PRIMARY KEY,
                qty       INTEGER NOT NULL,
                avg_price REAL    NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS trained_models (
                id            TEXT PRIMARY KEY,
                symbol        TEXT NOT NULL,
                strategy_id   TEXT NOT NULL DEFAULT 'lgbm_core',
                trained_at    REAL NOT NULL,
                bar_count     INTEGER NOT NULL,
                last_bar_date TEXT NOT NULL,
                oos_accuracy  REAL,
                model_blob    BLOB NOT NULL,
                feature_cols  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_models_symbol  ON trained_models(symbol);
            CREATE INDEX IF NOT EXISTS idx_models_trained ON trained_models(trained_at DESC);
        """)
        conn.commit()


# ── Jobs API ──────────────────────────────────────────────────────────────────

def upsert_job(job: dict) -> None:
    """Insert or update a job.  Preserves the original created_at on UPDATE."""
    now     = time.time()
    payload = json.dumps(job, default=str)   # default=str handles datetime/date
    with _wlock:
        conn = _conn()
        conn.execute("""
            INSERT INTO jobs (id, kind, status, created_at, updated_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status     = excluded.status,
                updated_at = excluded.updated_at,
                payload    = excluded.payload
        """, (job["id"], job["kind"], job["status"], now, now, payload))
        conn.commit()


def fetch_job(jid: str) -> dict | None:
    row = _conn().execute(
        "SELECT payload FROM jobs WHERE id = ?", (jid,)
    ).fetchone()
    return json.loads(row["payload"]) if row else None


def fetch_all_jobs() -> list[dict]:
    rows = _conn().execute(
        "SELECT payload FROM jobs ORDER BY created_at DESC"
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def jobs_stats() -> dict:
    conn = _conn()
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
    ).fetchall()
    by_status = {r["status"]: r["n"] for r in rows}
    total = sum(by_status.values())
    return {"total": total, "by_status": by_status}


# ── Cache API ─────────────────────────────────────────────────────────────────

def cache_get(key: str) -> Any | None:
    """
    Return the cached value, or None if missing / expired.

    Check order:
      1. L1 (_mem)  — zero I/O, returns immediately on a warm hit.
      2. L2 (SQLite) — backfills L1 so the next call stays in memory.
    """
    now = time.time()

    # ── L1: pure-memory lookup (no lock needed — dict.get is GIL-safe) ──
    entry = _mem.get(key)
    if entry is not None:
        value, expires_at = entry
        if expires_at > now:
            return value                  # ← warm L1 hit, never touches disk
        # Expired in L1 — fall through to SQLite (may have been refreshed)

    # ── L2: SQLite ──────────────────────────────────────────────────────
    row = _conn().execute(
        "SELECT value, expires_at FROM cache WHERE cache_key = ? AND expires_at > ?",
        (key, now),
    ).fetchone()
    if row is None:
        return None

    data = json.loads(row["value"])
    # Backfill L1 so the next request is served from memory
    with _wlock:
        _mem[key] = (data, row["expires_at"])
    return data


def cache_set(key: str, value: Any, ttl: int) -> None:
    """Persist value under key for ttl seconds (writes L1 + L2 atomically)."""
    expires = time.time() + ttl
    encoded = json.dumps(value, default=str)
    with _wlock:
        # L1 first — so in-flight readers see the new value immediately
        _mem[key] = (value, expires)
        # L2 — SQLite for cross-restart durability
        conn = _conn()
        conn.execute("""
            INSERT INTO cache (cache_key, value, expires_at) VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                value      = excluded.value,
                expires_at = excluded.expires_at
        """, (key, encoded, expires))
        conn.commit()


def cache_invalidate(prefix: str = "") -> int:
    """Evict cache entries whose key starts with prefix (or ALL if prefix="")."""
    with _wlock:
        # L1
        if prefix:
            stale = [k for k in _mem if k.startswith(prefix)]
        else:
            stale = list(_mem.keys())
        for k in stale:
            _mem.pop(k, None)
        # L2
        conn = _conn()
        if prefix:
            cur = conn.execute(
                "DELETE FROM cache WHERE cache_key LIKE ?", (f"{prefix}%",)
            )
        else:
            cur = conn.execute("DELETE FROM cache")
        conn.commit()
        return cur.rowcount


def cache_stats() -> dict:
    conn = _conn()
    now  = time.time()
    total  = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM cache WHERE expires_at > ?", (now,)
    ).fetchone()[0]
    # key-prefix breakdown
    rows = conn.execute(
        "SELECT SUBSTR(cache_key, 1, INSTR(cache_key || ':', ':') - 1) AS pfx, "
        "COUNT(*) AS n FROM cache WHERE expires_at > ? GROUP BY pfx",
        (now,),
    ).fetchall()
    breakdown = {r["pfx"]: r["n"] for r in rows}
    return {
        "total":   total,
        "active":  active,
        "expired": total - active,
        "by_type": breakdown,
    }


def purge_expired() -> int:
    """Housekeeping: remove expired entries from both L1 and L2.  Returns count deleted."""
    now = time.time()
    with _wlock:
        # L1 — evict stale keys from _mem
        stale = [k for k, (_, exp) in _mem.items() if exp <= now]
        for k in stale:
            del _mem[k]
        # L2 — SQLite
        conn = _conn()
        cur  = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
        conn.commit()
        return cur.rowcount


# ── Trained Models API ────────────────────────────────────────────────────────

def save_model(
    symbol: str,
    strategy_id: str,
    bar_count: int,
    last_bar_date: str,
    oos_accuracy: float,
    model_blob: bytes,
    feature_cols: list[str],
) -> str:
    """
    Upsert a trained model into the database.

    The primary key is ``{symbol}_{strategy_id}``.  Saves the raw pickle
    bytes in the BLOB column so the model can be deserialised without
    reimporting LightGBM's native save_model interface.

    Returns the model id.
    """
    mid = f"{symbol}_{strategy_id}"
    encoded_cols = json.dumps(feature_cols)
    now = time.time()
    with _wlock:
        conn = _conn()
        conn.execute(
            """
            INSERT INTO trained_models
                (id, symbol, strategy_id, trained_at, bar_count,
                 last_bar_date, oos_accuracy, model_blob, feature_cols)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                trained_at    = excluded.trained_at,
                bar_count     = excluded.bar_count,
                last_bar_date = excluded.last_bar_date,
                oos_accuracy  = excluded.oos_accuracy,
                model_blob    = excluded.model_blob,
                feature_cols  = excluded.feature_cols
            """,
            (mid, symbol, strategy_id, now, bar_count,
             last_bar_date, oos_accuracy, model_blob, encoded_cols),
        )
        conn.commit()
    return mid


def load_model(symbol: str, strategy_id: str = "lgbm_core") -> tuple[bytes, dict] | None:
    """
    Load model blob + metadata for the given symbol/strategy pair.

    Returns ``(blob_bytes, meta_dict)`` or ``None`` if not found.
    The caller is responsible for deserialising the blob (e.g. via pickle).
    """
    mid = f"{symbol}_{strategy_id}"
    row = _conn().execute(
        """
        SELECT model_blob, trained_at, bar_count, last_bar_date,
               oos_accuracy, feature_cols
        FROM trained_models WHERE id = ?
        """,
        (mid,),
    ).fetchone()
    if row is None:
        return None
    meta = {
        "model_id":      mid,
        "symbol":        symbol,
        "strategy_id":   strategy_id,
        "trained_at":    row["trained_at"],
        "bar_count":     row["bar_count"],
        "last_bar_date": row["last_bar_date"],
        "oos_accuracy":  row["oos_accuracy"],
        "feature_cols":  json.loads(row["feature_cols"]),
    }
    return bytes(row["model_blob"]), meta


def get_model_meta(symbol: str, strategy_id: str = "lgbm_core") -> dict | None:
    """Return metadata for a stored model (no blob) — fast for staleness checks."""
    mid = f"{symbol}_{strategy_id}"
    row = _conn().execute(
        """
        SELECT trained_at, bar_count, last_bar_date, oos_accuracy, feature_cols
        FROM trained_models WHERE id = ?
        """,
        (mid,),
    ).fetchone()
    if row is None:
        return None
    return {
        "model_id":      mid,
        "symbol":        symbol,
        "strategy_id":   strategy_id,
        "trained_at":    row["trained_at"],
        "bar_count":     row["bar_count"],
        "last_bar_date": row["last_bar_date"],
        "oos_accuracy":  row["oos_accuracy"],
        "feature_cols":  json.loads(row["feature_cols"]),
    }


def list_models() -> list[dict]:
    """Return metadata for all stored models (no blobs)."""
    rows = _conn().execute(
        """
        SELECT symbol, strategy_id, trained_at, bar_count,
               last_bar_date, oos_accuracy, feature_cols
        FROM trained_models ORDER BY trained_at DESC
        """
    ).fetchall()
    return [
        {
            "model_id":      f"{r['symbol']}_{r['strategy_id']}",
            "symbol":        r["symbol"],
            "strategy_id":   r["strategy_id"],
            "trained_at":    r["trained_at"],
            "bar_count":     r["bar_count"],
            "last_bar_date": r["last_bar_date"],
            "oos_accuracy":  r["oos_accuracy"],
            "feature_cols":  json.loads(r["feature_cols"]),
        }
        for r in rows
    ]


def delete_model(symbol: str, strategy_id: str = "lgbm_core") -> bool:
    """Delete a stored model.  Returns True if a row was actually deleted."""
    mid = f"{symbol}_{strategy_id}"
    with _wlock:
        conn = _conn()
        cur = conn.execute("DELETE FROM trained_models WHERE id = ?", (mid,))
        conn.commit()
    return cur.rowcount > 0


# ── Paper Broker State ────────────────────────────────────────────────────────
# Persists PaperBroker cash + positions so simulator state survives restarts.

_PAPER_INITIAL_CASH: float = 500_000.0


def get_paper_state() -> tuple[float, dict[str, dict]]:
    """Return (cash, {symbol: {qty, avg_price}}) for the paper broker.

    Returns the initial cash and empty positions if no state has been saved yet.
    """
    conn = _conn()
    row = conn.execute("SELECT cash FROM paper_account WHERE id = 1").fetchone()
    cash = float(row["cash"]) if row else _PAPER_INITIAL_CASH

    pos_rows = conn.execute(
        "SELECT symbol, qty, avg_price FROM paper_positions"
    ).fetchall()
    positions = {
        r["symbol"]: {"qty": int(r["qty"]), "avg_price": float(r["avg_price"])}
        for r in pos_rows
    }
    return cash, positions


def set_paper_cash(cash: float) -> None:
    """Upsert the cash balance for the paper broker."""
    with _wlock:
        conn = _conn()
        conn.execute(
            """
            INSERT INTO paper_account (id, cash) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET cash = excluded.cash
            """,
            (cash,),
        )
        conn.commit()


def upsert_paper_position(symbol: str, qty: int, avg_price: float) -> None:
    """Upsert a position.  If qty <= 0, the position row is deleted."""
    with _wlock:
        conn = _conn()
        if qty <= 0:
            conn.execute("DELETE FROM paper_positions WHERE symbol = ?", (symbol,))
        else:
            conn.execute(
                """
                INSERT INTO paper_positions (symbol, qty, avg_price) VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty       = excluded.qty,
                    avg_price = excluded.avg_price
                """,
                (symbol, qty, avg_price),
            )
        conn.commit()


def reset_paper_state(initial_cash: float = _PAPER_INITIAL_CASH) -> None:
    """Wipe all paper-broker state and reset cash to initial_cash."""
    with _wlock:
        conn = _conn()
        conn.execute("DELETE FROM paper_positions")
        conn.execute(
            """
            INSERT INTO paper_account (id, cash) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET cash = excluded.cash
            """,
            (initial_cash,),
        )
        conn.commit()

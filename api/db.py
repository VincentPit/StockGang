"""
api/db.py — Postgres persistence facade for MyQuant.

This is a thin synchronous shim that preserves the original SQLite-era public
API while routing every read and write to Postgres via SQLAlchemy 2.0.

Two responsibilities (unchanged from the SQLite era)
----------------------------------------------------
  1. Jobs       — write-through mirror of the in-memory _jobs dict so that
                  job history survives server restarts.
  2. Cache      — two-level TTL cache for data fetched from external APIs
                  (yfinance / AKShare). L1 is a process-local dict; L2 is
                  the Postgres ``cache`` table.

Architecture note (T1a)
-----------------------
The new ``myquant/db/`` package owns the schema (SQLAlchemy ORM models) and
exposes both async (asyncpg) and sync (psycopg2) engines. This shim uses the
sync engine so existing thread-pool callers — runner.py's ``_run_*_sync``
workers, advisor.py, scheduler helpers — keep working unchanged. T1b will
convert those callers to async and let this shim go away.

The L1 in-memory cache stays exactly as it was: a process-local dict guarded
by a write lock. It is intentionally not shared across replicas — this is a
hot-path optimization that gets blown away on restart. T1c (multi-replica
API) will revisit whether to swap it for a Redis tier.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError, ProgrammingError

from myquant.db import (
    Cache,
    Job,
    PaperAccount,
    PaperPosition,
    SessionLocal,
    TrainedModel,
    sync_engine,
)

_log = logging.getLogger(__name__)

# L1 in-memory cache: key → (value, expires_at_unix_seconds)
# Plain dict reads are GIL-safe; mutations always happen inside _wlock.
_mem: dict[str, tuple[Any, float]] = {}
_wlock = threading.Lock()


# ── Schema bootstrap / connection check ───────────────────────────────────────

def init_db() -> None:
    """
    Verify Postgres connectivity and that the schema has been migrated.

    The schema is owned by Alembic — run ``alembic upgrade head`` before
    starting the app. This function does NOT create tables; if they're
    missing it raises so misconfiguration fails loudly at startup rather
    than silently corrupting later writes.
    """
    try:
        with sync_engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM jobs LIMIT 1"))
    except (OperationalError, ProgrammingError) as exc:
        raise RuntimeError(
            "Postgres schema not initialised — run `alembic upgrade head` "
            f"and ensure POSTGRES_* settings are correct. Underlying error: {exc}"
        ) from exc


# ── Jobs API ──────────────────────────────────────────────────────────────────

def upsert_job(job: dict) -> None:
    """Insert or update a job. Preserves created_at on UPDATE."""
    now = time.time()
    payload = json.dumps(job, default=str)
    stmt = (
        pg_insert(Job)
        .values(
            id=job["id"],
            kind=job["kind"],
            status=job["status"],
            created_at=now,
            updated_at=now,
            payload=payload,
        )
        .on_conflict_do_update(
            index_elements=[Job.id],
            set_={
                "status":     job["status"],
                "updated_at": now,
                "payload":    payload,
            },
        )
    )
    with SessionLocal() as s, s.begin():
        s.execute(stmt)


def fetch_job(jid: str) -> dict | None:
    with SessionLocal() as s:
        row = s.execute(select(Job.payload).where(Job.id == jid)).scalar_one_or_none()
    return json.loads(row) if row else None


def fetch_all_jobs() -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(Job.payload).order_by(Job.created_at.desc())
        ).scalars().all()
    return [json.loads(r) for r in rows]


def jobs_stats() -> dict:
    with SessionLocal() as s:
        rows = s.execute(
            select(Job.status, func.count(Job.id)).group_by(Job.status)
        ).all()
    by_status = {status: n for status, n in rows}
    return {"total": sum(by_status.values()), "by_status": by_status}


# ── Cache API ─────────────────────────────────────────────────────────────────

def cache_get(key: str) -> Any | None:
    """Return the cached value, or None if missing / expired.

    Check order:
      1. L1 (_mem) — zero I/O, returns immediately on a warm hit.
      2. L2 (Postgres) — backfills L1 so the next call stays in memory.
    """
    now = time.time()

    # L1 — pure-memory lookup, no lock needed (dict.get is GIL-safe)
    entry = _mem.get(key)
    if entry is not None:
        value, expires_at = entry
        if expires_at > now:
            return value

    # L2 — Postgres
    with SessionLocal() as s:
        row = s.execute(
            select(Cache.value, Cache.expires_at).where(
                Cache.cache_key == key, Cache.expires_at > now
            )
        ).one_or_none()
    if row is None:
        return None

    data, expires_at = row
    decoded = json.loads(data)
    with _wlock:
        _mem[key] = (decoded, expires_at)
    return decoded


def cache_set(key: str, value: Any, ttl: int) -> None:
    """Persist ``value`` under ``key`` for ``ttl`` seconds (writes L1 + L2 atomically)."""
    expires = time.time() + ttl
    encoded = json.dumps(value, default=str)
    stmt = (
        pg_insert(Cache)
        .values(cache_key=key, value=encoded, expires_at=expires)
        .on_conflict_do_update(
            index_elements=[Cache.cache_key],
            set_={"value": encoded, "expires_at": expires},
        )
    )
    with _wlock:
        # L1 first so in-flight readers see the new value immediately
        _mem[key] = (value, expires)
        with SessionLocal() as s, s.begin():
            s.execute(stmt)


def cache_invalidate(prefix: str = "") -> int:
    """Evict cache entries whose key starts with ``prefix`` (or ALL if prefix='')."""
    with _wlock:
        # L1 — snapshot keys before mutating
        if prefix:
            stale_keys = [k for k in _mem if k.startswith(prefix)]
            for k in stale_keys:
                _mem.pop(k, None)
        else:
            _mem.clear()

        # L2
        if prefix:
            stmt = delete(Cache).where(Cache.cache_key.like(f"{prefix}%"))
        else:
            stmt = delete(Cache)

        with SessionLocal() as s, s.begin():
            result = s.execute(stmt)
            return result.rowcount or 0


def cache_stats() -> dict:
    now = time.time()
    with SessionLocal() as s:
        total  = s.execute(select(func.count(Cache.cache_key))).scalar_one()
        active = s.execute(
            select(func.count(Cache.cache_key)).where(Cache.expires_at > now)
        ).scalar_one()
        # Key-prefix breakdown — match SQLite SUBSTR(key, 1, INSTR(key||':',':')-1).
        # The ``split_part`` Postgres builtin returns the first segment of
        # ``key:rest:more`` split on ':'.
        rows = s.execute(
            select(func.split_part(Cache.cache_key, ":", 1), func.count())
            .where(Cache.expires_at > now)
            .group_by(func.split_part(Cache.cache_key, ":", 1))
        ).all()
    breakdown = {pfx: n for pfx, n in rows}
    return {
        "total":   total,
        "active":  active,
        "expired": total - active,
        "by_type": breakdown,
    }


def purge_expired() -> int:
    """Housekeeping: remove expired entries from both L1 and L2. Returns count deleted."""
    now = time.time()
    with _wlock:
        stale_keys = [k for k, (_, exp) in _mem.items() if exp <= now]
        for k in stale_keys:
            del _mem[k]
        with SessionLocal() as s, s.begin():
            result = s.execute(delete(Cache).where(Cache.expires_at <= now))
            return result.rowcount or 0


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
    """Upsert a trained model. PK is ``{symbol}_{strategy_id}``. Returns the id."""
    mid = f"{symbol}_{strategy_id}"
    encoded_cols = json.dumps(feature_cols)
    now = time.time()
    stmt = (
        pg_insert(TrainedModel)
        .values(
            id=mid,
            symbol=symbol,
            strategy_id=strategy_id,
            trained_at=now,
            bar_count=bar_count,
            last_bar_date=last_bar_date,
            oos_accuracy=oos_accuracy,
            model_blob=model_blob,
            feature_cols=encoded_cols,
        )
        .on_conflict_do_update(
            index_elements=[TrainedModel.id],
            set_={
                "trained_at":    now,
                "bar_count":     bar_count,
                "last_bar_date": last_bar_date,
                "oos_accuracy":  oos_accuracy,
                "model_blob":    model_blob,
                "feature_cols":  encoded_cols,
            },
        )
    )
    with SessionLocal() as s, s.begin():
        s.execute(stmt)
    return mid


def load_model(symbol: str, strategy_id: str = "lgbm_core") -> tuple[bytes, dict] | None:
    mid = f"{symbol}_{strategy_id}"
    with SessionLocal() as s:
        row = s.execute(
            select(
                TrainedModel.model_blob,
                TrainedModel.trained_at,
                TrainedModel.bar_count,
                TrainedModel.last_bar_date,
                TrainedModel.oos_accuracy,
                TrainedModel.feature_cols,
            ).where(TrainedModel.id == mid)
        ).one_or_none()
    if row is None:
        return None
    blob, trained_at, bar_count, last_bar_date, oos_accuracy, feature_cols = row
    meta = {
        "model_id":      mid,
        "symbol":        symbol,
        "strategy_id":   strategy_id,
        "trained_at":    trained_at,
        "bar_count":     bar_count,
        "last_bar_date": last_bar_date,
        "oos_accuracy":  oos_accuracy,
        "feature_cols":  json.loads(feature_cols),
    }
    return bytes(blob), meta


def get_model_meta(symbol: str, strategy_id: str = "lgbm_core") -> dict | None:
    mid = f"{symbol}_{strategy_id}"
    with SessionLocal() as s:
        row = s.execute(
            select(
                TrainedModel.trained_at,
                TrainedModel.bar_count,
                TrainedModel.last_bar_date,
                TrainedModel.oos_accuracy,
                TrainedModel.feature_cols,
            ).where(TrainedModel.id == mid)
        ).one_or_none()
    if row is None:
        return None
    trained_at, bar_count, last_bar_date, oos_accuracy, feature_cols = row
    return {
        "model_id":      mid,
        "symbol":        symbol,
        "strategy_id":   strategy_id,
        "trained_at":    trained_at,
        "bar_count":     bar_count,
        "last_bar_date": last_bar_date,
        "oos_accuracy":  oos_accuracy,
        "feature_cols":  json.loads(feature_cols),
    }


def list_models() -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(
                TrainedModel.symbol,
                TrainedModel.strategy_id,
                TrainedModel.trained_at,
                TrainedModel.bar_count,
                TrainedModel.last_bar_date,
                TrainedModel.oos_accuracy,
                TrainedModel.feature_cols,
            ).order_by(TrainedModel.trained_at.desc())
        ).all()
    return [
        {
            "model_id":      f"{r.symbol}_{r.strategy_id}",
            "symbol":        r.symbol,
            "strategy_id":   r.strategy_id,
            "trained_at":    r.trained_at,
            "bar_count":     r.bar_count,
            "last_bar_date": r.last_bar_date,
            "oos_accuracy":  r.oos_accuracy,
            "feature_cols":  json.loads(r.feature_cols),
        }
        for r in rows
    ]


def delete_model(symbol: str, strategy_id: str = "lgbm_core") -> bool:
    mid = f"{symbol}_{strategy_id}"
    with SessionLocal() as s, s.begin():
        result = s.execute(delete(TrainedModel).where(TrainedModel.id == mid))
        return (result.rowcount or 0) > 0


# ── Paper Broker State ────────────────────────────────────────────────────────

_PAPER_INITIAL_CASH: float = 500_000.0


def get_paper_state() -> tuple[float, dict[str, dict]]:
    """Return (cash, {symbol: {qty, avg_price}}). Returns initial cash + {} if no state saved."""
    with SessionLocal() as s:
        cash_row = s.execute(
            select(PaperAccount.cash).where(PaperAccount.id == 1)
        ).scalar_one_or_none()
        cash = float(cash_row) if cash_row is not None else _PAPER_INITIAL_CASH

        pos_rows = s.execute(
            select(PaperPosition.symbol, PaperPosition.qty, PaperPosition.avg_price)
        ).all()
    positions = {
        sym: {"qty": int(qty), "avg_price": float(avg)}
        for sym, qty, avg in pos_rows
    }
    return cash, positions


def set_paper_cash(cash: float) -> None:
    stmt = (
        pg_insert(PaperAccount)
        .values(id=1, cash=cash)
        .on_conflict_do_update(
            index_elements=[PaperAccount.id],
            set_={"cash": cash},
        )
    )
    with SessionLocal() as s, s.begin():
        s.execute(stmt)


def upsert_paper_position(symbol: str, qty: int, avg_price: float) -> None:
    """Upsert a position. If qty <= 0, the position row is deleted."""
    with SessionLocal() as s, s.begin():
        if qty <= 0:
            s.execute(delete(PaperPosition).where(PaperPosition.symbol == symbol))
            return
        stmt = (
            pg_insert(PaperPosition)
            .values(symbol=symbol, qty=qty, avg_price=avg_price)
            .on_conflict_do_update(
                index_elements=[PaperPosition.symbol],
                set_={"qty": qty, "avg_price": avg_price},
            )
        )
        s.execute(stmt)


def reset_paper_state(initial_cash: float = _PAPER_INITIAL_CASH) -> None:
    """Wipe all paper-broker state and reset cash to initial_cash."""
    stmt = (
        pg_insert(PaperAccount)
        .values(id=1, cash=initial_cash)
        .on_conflict_do_update(
            index_elements=[PaperAccount.id],
            set_={"cash": initial_cash},
        )
    )
    with SessionLocal() as s, s.begin():
        s.execute(delete(PaperPosition))
        s.execute(stmt)

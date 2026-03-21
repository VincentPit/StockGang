"""
api/runner.py — Background job runner.

Runs backtests and screener jobs in a thread pool so the API stays async.
Jobs are stored in an in-process dict (sufficient for single-node; swap for
Redis if you go multi-process).
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import db as _db

_log = logging.getLogger(__name__)

# Scale thread pool to CPU count; cap at 16 so we don't starve the event loop
_executor = ThreadPoolExecutor(max_workers=min((os.cpu_count() or 2) * 2, 16))
atexit.register(_executor.shutdown, wait=False)  # graceful drain on process exit

# In-memory mirror — fast O(1) reads during a session
_jobs: dict[str, dict[str, Any]] = {}

# Maximum number of done/error jobs to keep in memory to prevent unbounded growth
_MAX_JOBS: int = 500

# ── Bootstrap: initialise DB schema and restore jobs from last session ────────
_db.init_db()
_jobs.update({j["id"]: j for j in _db.fetch_all_jobs()})


# ── helpers ──────────────────────────────────────────────────────────────────
def _evict_stale_jobs() -> None:
    """Evict the oldest done/error jobs once _jobs exceeds _MAX_JOBS entries."""
    if len(_jobs) <= _MAX_JOBS:
        return
    candidates = sorted(
        [j for j in _jobs.values() if j["status"] in ("done", "error")],
        key=lambda j: j.get("_ts", 0),
    )
    to_drop = max(0, len(_jobs) - _MAX_JOBS)
    for j in candidates[:to_drop]:
        _jobs.pop(j["id"], None)
def _update_job(jid: str, updates: dict) -> None:
    """Mutate in-memory dict and write-through to SQLite atomically."""
    _jobs[jid].update(updates)
    _db.upsert_job(_jobs[jid])
    if "status" in updates:
        _log.info("job %s → %s", jid[:8], updates["status"])


def new_job(kind: str) -> str:
    jid = str(uuid.uuid4())
    job = {"id": jid, "kind": kind, "status": "pending", "_ts": time.time()}
    _jobs[jid] = job
    _db.upsert_job(job)
    _evict_stale_jobs()
    return jid


def get_job(jid: str) -> dict[str, Any] | None:
    return _jobs.get(jid)


def list_jobs() -> list[dict[str, Any]]:
    return list(_jobs.values())


# ── backtest ─────────────────────────────────────────────────────────────────

def _run_backtest_sync(jid: str, req: dict) -> None:
    """Executed in a thread — thin wrapper around _backtest_core."""
    try:
        _update_job(jid, {"status": "running", "pct": 5, "step": "Loading price data…"})
        result = _backtest_core(req["symbols"], req)
        _update_job(jid, {"status": "done", "pct": 100, "step": "Done", **result})
    except Exception:
        tb = traceback.format_exc()
        _log.error("backtest job %s failed: %s", jid[:8], tb.splitlines()[-1])
        _update_job(jid, {"status": "error", "error": tb})


async def launch_backtest(jid: str, req: dict) -> None:
    asyncio.get_running_loop().run_in_executor(_executor, _run_backtest_sync, jid, req)


# ── screener ─────────────────────────────────────────────────────────────────

def _run_screener_sync(jid: str, req: dict) -> None:
    try:
        _update_job(jid, {"status": "running", "pct": 5, "step": "Fetching CSI universe…"})
        from myquant.tools.stock_screener import screen

        _update_job(jid, {"pct": 20, "step": "Downloading price bars…"})
        top_syms, results, universe_size = screen(
            top_n=req["top_n"],
            min_bars=req["min_bars"],
            lookback_years=req["lookback_years"],
            indices=req.get("indices", ["000300"]),
            verbose=False,
        )
        _update_job(jid, {"pct": 85, "step": "Scoring and ranking…"})

        rows = []
        for rank, r in enumerate(results, 1):
            rows.append({
                "rank":        rank,
                "symbol":      r["sym"],
                "yf_ticker":   r["yf"],
                "name":        r["name"],
                "bars":        r["bars"],
                "ret_1y":      round(r["ret_1y"], 6),
                "ret_6m":      round(r["ret_6m"], 6),
                "sharpe":      round(r["sharpe"], 4),
                "max_dd":      round(r["max_dd"], 6),
                "trend_pct":   round(r["trend_pct"], 4),
                "atr_pct":     round(r["atr_pct"], 6),
                "autocorr":    round(r["autocorr"], 6),
                "score":        round(r["score"], 6),
                "recommended":  r["sym"] in top_syms,
                "causal_nodes": r.get("causal_nodes", []),
                "data_scope":   r.get("data_scope"),
                "gate_checks":  r.get("gate_checks", []),
                # CN-market metrics — guard against None values from screener
                "ret_20d":        round(float(r.get("ret_20d")        or 0.0), 6),
                "price_52w_pct":  round(float(r.get("price_52w_pct")  or 0.0), 4),
                "dist_52w_high":  round(float(r.get("dist_52w_high")  or 0.0), 4),
                "vol_60d":        round(float(r.get("vol_60d")        or 0.0), 4),
                "limit_up_60d":   int(r.get("limit_up_60d")   or 0),
                "yang_ratio_60d": round(float(r.get("yang_ratio_60d") or 0.0), 4),
            })

        _update_job(jid, {
            "status":        "done",
            "pct":           100,
            "step":          "Done",
            "top_symbols":   top_syms,
            "rows":          rows,
            "universe_size": universe_size,
        })

    except Exception:
        tb = traceback.format_exc()
        _log.error("screener job %s failed: %s", jid[:8], tb.splitlines()[-1])
        _update_job(jid, {"status": "error", "error": tb})


async def launch_screener(jid: str, req: dict) -> None:
    asyncio.get_running_loop().run_in_executor(_executor, _run_screener_sync, jid, req)


# ── backtest core (shared by backtest + workflow) ─────────────────────────────

def _backtest_core(symbols: list, req: dict) -> dict:
    """
    Runs the full backtest pipeline and returns a plain result dict.
    Does NOT touch _jobs — callers handle status updates.
    """
    # ── Load best_params.json written by train_loop.py (if present) ───────────
    # The training loop optimises model and risk params across the full param grid
    # and saves the winning configuration here.  Params in the file silently
    # override the hard-coded defaults below; the user-supplied ``req`` values
    # still take precedence for explicit risk overrides (stop_loss_pct, etc.).
    import json as _json
    _bp: dict = {}
    _bp_path = Path(__file__).parent.parent / "best_params.json"
    if _bp_path.exists():
        try:
            _bp = _json.loads(_bp_path.read_text())
            _log.info(
                "_backtest_core: loaded best_params.json → conf=%.2f thresh=%.3f hold=%d",
                _bp.get("min_confidence", 0.60),
                _bp.get("threshold",      0.015),
                _bp.get("min_hold_bars",  5),
            )
        except Exception as _e:
            _log.warning("_backtest_core: could not parse best_params.json: %s", _e)

    from myquant.backtest.simulator import Backtester, BacktestConfig
    from myquant.models.bar import BarInterval
    from myquant.strategy.ml.lgbm_strategy import LGBMStrategy
    from myquant.strategy.technical.ma_crossover import MACrossoverStrategy
    from myquant.strategy.technical.macd_strategy import MACDStrategy
    from myquant.strategy.technical.rsi_strategy import RSIStrategy
    from myquant.strategy.nlp.news_strategy import NewsStrategy
    from myquant.config.logging_config import setup_logging
    setup_logging()

    end_date   = datetime.now()
    test_start = end_date - timedelta(days=req["lookback_days"])

    config = BacktestConfig(
        symbols           = symbols,
        start_date        = test_start,
        end_date          = end_date,
        initial_cash      = req["initial_cash"],
        interval          = BarInterval.D1,
        commission_rate   = req["commission_rate"],
        slippage          = 0.0002,
        apply_stamp_duty  = True,
        train_years       = 2,
        stop_loss_pct     = req["stop_loss_pct"],
        symbol_loss_cap   = req["symbol_loss_cap"],
        trailing_stop_pct = req.get("trailing_stop_pct") or _bp.get("trailing_stop_pct", 0.0),
        take_profit_pct   = req.get("take_profit_pct")   or _bp.get("take_profit_pct",   0.0),
    )

    lgbm = LGBMStrategy(
        strategy_id="lgbm_core", symbols=symbols,
        forward_days=5, threshold=_bp.get("threshold", 0.015), train_ratio=0.70,
        min_confidence=_bp.get("min_confidence", 0.60),
        retrain_every=21, max_train_bars=504,
        use_macro=False, num_leaves=31, n_estimators=300,
        min_hold_bars=_bp.get("min_hold_bars", 5),
        commission_rate=req.get("commission_rate", _bp.get("commission_rate", 0.0003)),
    )
    backtester = (
        Backtester(config)
        .add_strategy(lgbm)
        .add_strategy(MACrossoverStrategy("ma_cross", symbols, fast_period=10, slow_period=30, use_ema=True))
        .add_strategy(RSIStrategy("rsi_filter", symbols, period=14, oversold=35, overbought=65, exit_mid=False))
        .add_strategy(MACDStrategy("macd_sig", symbols, fast=12, slow=26, signal=9, min_hist=0.0))
        .add_strategy(NewsStrategy("news_sentiment", symbols, buy_threshold=0.45, sell_threshold=-0.45,
                                   window_size=8, news_fetch_interval_mins=30, min_confidence=0.60))
    )

    result = asyncio.run(backtester.run())

    # ── nav series ────────────────────────────────────────────────
    nav_series = []
    if not result.nav_series.empty:
        for ts, nav in result.nav_series.items():
            nav_series.append({"date": str(ts)[:10], "nav": round(float(nav), 2)})

    # ── trades ────────────────────────────────────────────────────
    trades = []
    if not result.trades.empty:
        for _, row in result.trades.iterrows():
            trades.append({
                "time":       str(row.get("time", "")),
                "symbol":     str(row.get("symbol", "")),
                "side":       str(row.get("side", "")),
                "qty":        float(row.get("qty", 0)),
                "price":      float(row.get("price", 0)),
                "commission": float(row.get("commission", 0)),
                "strategy":   str(row.get("strategy", "")),
            })

    # ── per-symbol P&L ────────────────────────────────────────────
    symbol_pnl = []
    if not result.trades.empty and "symbol" in result.trades.columns:
        df = result.trades

        def fifo_pnl(grp):
            grp = grp.sort_values("time").reset_index(drop=True)
            buy_q: list = []
            pnl = 0.0
            for _, r in grp.iterrows():
                qty, price, comm = float(r["qty"]), float(r["price"]), float(r["commission"])
                if r["side"] == "BUY":
                    buy_q.append((qty, price + comm / max(qty, 1)))
                else:
                    rem = qty
                    trade_pnl = -comm
                    while rem > 0 and buy_q:
                        bqty, bprice = buy_q[0]
                        matched = min(rem, bqty)
                        trade_pnl += matched * (price - bprice)
                        rem -= matched
                        buy_q[0] = (bqty - matched, bprice)
                        if buy_q[0][0] <= 0:
                            buy_q.pop(0)
                    pnl += trade_pnl
            return pnl

        for sym, grp in df.groupby("symbol"):
            symbol_pnl.append({
                "symbol":  str(sym),
                "fills":   len(grp),
                "net_pnl": round(fifo_pnl(grp), 2),
                "buys":    int((grp["side"] == "BUY").sum()),
                "sells":   int((grp["side"] == "SELL").sum()),
            })
        symbol_pnl.sort(key=lambda x: x["net_pnl"])

    # ── per-strategy ──────────────────────────────────────────────
    strategy_fills = []
    if not result.trades.empty and "strategy" in result.trades.columns:
        for strat, grp in result.trades.groupby("strategy"):
            strategy_fills.append({
                "strategy": str(strat),
                "fills":    len(grp),
                "buys":     int((grp["side"] == "BUY").sum()),
                "sells":    int((grp["side"] == "SELL").sum()),
            })

    return {
        "period_start":   str(config.start_date.date()),
        "period_end":     str(config.end_date.date()),
        "symbols":        symbols,
        "initial_nav":    config.initial_cash,
        "final_nav":      round(config.initial_cash + result.total_pnl, 2),
        "total_pnl":      round(result.total_pnl, 2),
        "total_pnl_pct":  round(result.total_pnl_pct, 6),
        "sharpe":         round(result.sharpe_ratio, 4),
        "max_dd":         round(result.max_drawdown, 6),
        "num_trades":     result.num_trades,
        "win_rate":       round(result.win_rate, 6),
        "avg_win":        round(result.avg_win, 2),
        "avg_loss":       round(result.avg_loss, 2),
        "profit_factor":  round(result.profit_factor, 4),
        "nav_series":     nav_series,
        "trades":         trades,
        "symbol_pnl":     symbol_pnl,
        "strategy_fills": strategy_fills,
    }


# ── price / OHLCV ─────────────────────────────────────────────────────────────

# Cache TTLs (seconds)
_TTL_PRICE  = 4 * 3600     # 4 h  — one fetch per trading day is enough
_TTL_FUND   = 24 * 3600    # 24 h — fundamentals change quarterly
_TTL_NEWS   = 30 * 60      # 30 m — headlines are time-sensitive
_TTL_REGIME = 4 * 3600     # 4 h  — daily-bar regime rarely flips intra-day


def get_price_sync(symbol: str, days: int = 365) -> dict:
    """Fetch OHLCV bars via yfinance — SQLite-cached for 4 hours."""
    key = f"price:{symbol}:{days}"
    hit = _db.cache_get(key)
    if hit is not None:
        return hit

    from myquant.data.fetchers.historical_loader import _symbol_to_yfinance
    import yfinance as yf
    from datetime import date as date_type, timedelta as td

    yf_ticker = _symbol_to_yfinance(symbol)
    if not yf_ticker:
        return {"symbol": symbol, "yf_ticker": "", "bars": [], "error": f"Unknown symbol format: {symbol}"}

    end   = date_type.today()
    start = end - td(days=days)
    try:
        raw = yf.download(yf_ticker, start=str(start), end=str(end),
                          auto_adjust=True, progress=False, multi_level_index=False)
        if raw.empty:
            return {"symbol": symbol, "yf_ticker": yf_ticker, "bars": [], "error": "No data returned"}

        bars = []
        for ts, row in raw.iterrows():
            bars.append({
                "date":   str(ts)[:10],
                "open":   round(float(row["Open"]),   4),
                "high":   round(float(row["High"]),   4),
                "low":    round(float(row["Low"]),    4),
                "close":  round(float(row["Close"]),  4),
                "volume": float(row.get("Volume", 0)),
            })
        result = {"symbol": symbol, "yf_ticker": yf_ticker, "bars": bars}
        _db.cache_set(key, result, ttl=_TTL_PRICE)
        return result
    except Exception as exc:
        return {"symbol": symbol, "yf_ticker": yf_ticker, "bars": [], "error": str(exc)}


# ── fundamentals ─────────────────────────────────────────────────────────────

def get_fundamentals_sync(symbol: str) -> dict:
    """Fetch fundamental snapshot — SQLite-cached for 24 hours."""
    key = f"fund:{symbol}"
    hit = _db.cache_get(key)
    if hit is not None:
        return hit

    try:
        from myquant.data.fetchers.fundamental_fetcher import FundamentalFetcher
        snap = FundamentalFetcher().fetch(symbol)
        result = {
            "symbol":         symbol,
            "pe_ttm":         round(snap.pe_ttm, 2),
            "pb":             round(snap.pb, 2),
            "ps_ttm":         round(snap.ps_ttm, 2),
            "roe":            round(snap.roe, 2),
            "revenue_growth": round(snap.revenue_growth, 2),
            "net_margin":     round(snap.net_margin, 2),
            "dividend_yield": round(snap.dividend_yield, 2),
            "value_score":    round(snap.value_score, 1),
            "growth_score":   round(snap.growth_score, 1),
            "quality_score":  round(snap.quality_score, 1),
        }
        _db.cache_set(key, result, ttl=_TTL_FUND)
        return result
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc),
                "pe_ttm": 0, "pb": 0, "ps_ttm": 0, "roe": 0,
                "revenue_growth": 0, "net_margin": 0, "dividend_yield": 0,
                "value_score": 0, "growth_score": 0, "quality_score": 0}


# ── news ─────────────────────────────────────────────────────────────────────

def get_news_sync(symbol: str, limit: int = 20) -> dict:
    """Fetch stock news headlines — SQLite-cached for 30 minutes."""
    key = f"news:{symbol}:{limit}"
    hit = _db.cache_get(key)
    if hit is not None:
        return hit

    try:
        from myquant.data.fetchers.news_fetcher import NewsFetcher
        items = NewsFetcher().fetch_stock_news(symbol, limit=limit)
        result = {
            "symbol": symbol,
            "items": [
                {"title": i.title, "content": i.content[:300],
                 "source": i.source, "ts": i.ts.isoformat(), "url": i.url}
                for i in items
            ],
        }
        _db.cache_set(key, result, ttl=_TTL_NEWS)
        return result
    except Exception as exc:
        return {"symbol": symbol, "items": [], "error": str(exc)}


def get_macro_news_sync(limit: int = 30) -> dict:
    """Fetch macro news headlines — SQLite-cached for 30 minutes."""
    key = f"news:macro:{limit}"
    hit = _db.cache_get(key)
    if hit is not None:
        return hit

    try:
        from myquant.data.fetchers.news_fetcher import NewsFetcher
        items = NewsFetcher().fetch_macro_news(limit=limit)
        result = {
            "symbol": None,
            "items": [
                {"title": i.title, "content": i.content[:300],
                 "source": i.source, "ts": i.ts.isoformat(), "url": i.url}
                for i in items
            ],
        }
        _db.cache_set(key, result, ttl=_TTL_NEWS)
        return result
    except Exception as exc:
        return {"symbol": None, "items": [], "error": str(exc)}


# ── regime ────────────────────────────────────────────────────────────────────

def get_regime_sync(symbols: list[str]) -> dict:
    """
    Feed 1-year of yfinance closes into HistoricalRegimeDetector.
    Result is SQLite-cached per sorted symbol set for 4 hours.
    """
    # Cache key is deterministic regardless of argument order
    key = "regime:" + ",".join(sorted(symbols))
    hit = _db.cache_get(key)
    if hit is not None:
        return hit

    from myquant.data.fetchers.macro_proxy import HistoricalRegimeDetector
    from myquant.data.fetchers.historical_loader import _symbol_to_yfinance
    import yfinance as yf
    from datetime import date as date_type, timedelta as td

    detector   = HistoricalRegimeDetector()
    analyzed   = 0
    end        = date_type.today()
    start      = end - td(days=365)

    for sym in symbols:
        yf_ticker = _symbol_to_yfinance(sym)
        if not yf_ticker:
            continue
        # Re-use cached price bars when available so regime calls are free
        price_hit = _db.cache_get(f"price:{sym}:365")
        if price_hit and price_hit.get("bars"):
            for bar in price_hit["bars"]:
                detector.update(sym, bar["close"])
            analyzed += 1
            continue
        try:
            raw = yf.download(yf_ticker, start=str(start), end=str(end),
                              auto_adjust=True, progress=False, multi_level_index=False)
            for ts, row in raw.iterrows():
                detector.update(sym, float(row["Close"]))
            analyzed += 1
        except Exception:
            pass

    result = {
        "regime":            detector.regime,
        "signal_multiplier": detector.signal_multiplier,
        "symbols_analyzed":  analyzed,
    }
    _db.cache_set(key, result, ttl=_TTL_REGIME)
    return result


# ── workflow (screen → backtest) ─────────────────────────────────────────────

def _run_workflow_sync(jid: str, req: dict) -> None:
    """Screen for top stocks, then immediately backtest them."""
    try:
        # Phase 1 — screening
        _update_job(jid, {"status": "screening", "pct": 5, "step": "Fetching CSI universe…"})
        from myquant.tools.stock_screener import screen

        _update_job(jid, {"pct": 15, "step": "Downloading price bars…"})
        top_syms, results, universe_size = screen(
            top_n=req["top_n"],
            min_bars=req["min_bars"],
            lookback_years=req["lookback_years"],
            indices=req.get("indices", ["000300"]),
            verbose=False,
        )
        _update_job(jid, {"pct": 35, "step": "Scoring and ranking…"})

        screen_rows = []
        for rank, r in enumerate(results, 1):
            screen_rows.append({
                "rank":        rank,
                "symbol":      r["sym"],
                "yf_ticker":   r["yf"],
                "name":        r["name"],
                "bars":        r["bars"],
                "ret_1y":      round(r["ret_1y"], 6),
                "ret_6m":      round(r["ret_6m"], 6),
                "sharpe":      round(r["sharpe"], 4),
                "max_dd":      round(r["max_dd"], 6),
                "trend_pct":   round(r["trend_pct"], 4),
                "atr_pct":     round(r["atr_pct"], 6),
                "autocorr":    round(r["autocorr"], 6),
                "score":        round(r["score"], 6),
                "recommended":  r["sym"] in top_syms,
                "causal_nodes": r.get("causal_nodes", []),
                "data_scope":   r.get("data_scope"),
                "gate_checks":  r.get("gate_checks", []),
                # CN-market metrics — same None-guards as _run_screener_sync
                "ret_20d":        round(float(r.get("ret_20d")        or 0.0), 6),
                "price_52w_pct":  round(float(r.get("price_52w_pct")  or 0.0), 4),
                "dist_52w_high":  round(float(r.get("dist_52w_high")  or 0.0), 4),
                "vol_60d":        round(float(r.get("vol_60d")        or 0.0), 4),
                "limit_up_60d":   int(r.get("limit_up_60d")   or 0),
                "yang_ratio_60d": round(float(r.get("yang_ratio_60d") or 0.0), 4),
            })

        _update_job(jid, {"top_symbols": top_syms, "screen_rows": screen_rows, "universe_size": universe_size})

        if not top_syms:
            _update_job(jid, {
                "status": "done",
                "pct": 100,
                "step": "Done — no qualifying stocks found",
                "error": "Screener found no qualifying stocks — try widening the universe or reducing min_bars.",
            })
            return

        # Phase 2 — backtesting the top picks
        _update_job(jid, {"status": "backtesting", "pct": 40, "step": f"Backtesting {len(top_syms)} picks…"})
        backtest_req = {
            "lookback_days":   req["backtest_days"],
            "initial_cash":    req["initial_cash"],
            "commission_rate": req["commission_rate"],
            "stop_loss_pct":   req["stop_loss_pct"],
            "symbol_loss_cap": req["symbol_loss_cap"],
            "trailing_stop_pct": req.get("trailing_stop_pct", 0.0),
            "take_profit_pct":   req.get("take_profit_pct", 0.0),
        }
        result = _backtest_core(top_syms, backtest_req)
        _update_job(jid, {"status": "done", "pct": 100, "step": "Done", **result})

    except Exception:
        tb = traceback.format_exc()
        _log.error("workflow job %s failed: %s", jid[:8], tb.splitlines()[-1])
        _update_job(jid, {"status": "error", "error": tb})


async def launch_workflow(jid: str, req: dict) -> None:
    asyncio.get_running_loop().run_in_executor(_executor, _run_workflow_sync, jid, req)


# ── train loop (screen → backtest → tune) ────────────────────────────────────

def _run_train_loop_sync(jid: str, req: dict) -> None:
    """Run the automated screen→backtest→tune loop in a background thread."""
    try:
        _update_job(jid, {"status": "running", "pct": 1, "step": "Initialising…"})

        from train_loop import run_loop

        def _cb(d: dict) -> None:
            update: dict = {"status": "running"}
            if "pct" in d:
                update["pct"] = d["pct"]
            if "step" in d:
                update["step"] = d["step"]
            _update_job(jid, update)

        result = run_loop(
            symbols      = req.get("symbols"),
            top_n        = req.get("top_n", 3),
            lookback_days= req.get("lookback_days", 180),
            train_years  = req.get("train_years", 1),
            configs      = req.get("configs", "fast"),
            progress_cb  = _cb,
        )

        if "error" in result and not result.get("best"):
            _update_job(jid, {"status": "error", "error": result["error"]})
            return

        best      = result.get("best") or {}
        best_res  = best.get("result") or {}
        # Flatten trial list for JSON storage (keep result sub-dict)
        flat_trials = [
            {
                "symbol":        t["symbol"],
                "config":        t["config"],
                "passes":        t["passes"],
                "score":         t["score"],
                "elapsed_s":     t["elapsed_s"],
                "total_pnl":     t["result"].get("total_pnl"),
                "profit_factor": t["result"].get("profit_factor"),
                "win_rate":      t["result"].get("win_rate"),
                "num_trades":    t["result"].get("num_trades"),
                "error":         t.get("error"),
            }
            for t in result.get("all_trials", [])
        ]

        _update_job(jid, {
            "status":        "done",
            "pct":           100,
            "step":          "Done ✅" if result.get("found_passing") else "Done ⚠️ (no profitable config)",
            "found_passing": result.get("found_passing", False),
            "best_symbol":   best.get("symbol"),
            "best_config":   best.get("config"),
            "best_pf":       best_res.get("profit_factor"),
            "best_wr":       best_res.get("win_rate"),
            "best_pnl":      best_res.get("total_pnl"),
            "best_trades":   best_res.get("num_trades"),
            "symbols_tested":result.get("symbols_tested", []),
            "all_trials":    flat_trials,
        })

    except Exception:
        tb = traceback.format_exc()
        _log.error("train_loop job %s failed: %s", jid[:8], tb.splitlines()[-1])
        _update_job(jid, {"status": "error", "error": tb})


async def launch_train_loop(jid: str, req: dict) -> None:
    asyncio.get_running_loop().run_in_executor(_executor, _run_train_loop_sync, jid, req)


# ── advisor: train ────────────────────────────────────────────────────────────

def _run_train_sync(jid: str, symbol: str, force: bool) -> None:
    """Train (or reload) an LGBM model for ``symbol`` in a background thread."""
    try:
        _update_job(jid, {"status": "running", "symbol": symbol})
        from api.advisor import train_for_symbol
        meta = train_for_symbol(symbol, force=force)
        _update_job(jid, {
            "status":        "done",
            "symbol":        symbol,
            "model_id":      meta["model_id"],
            "train_status":  meta["train_status"],
            "skip_reason":   meta.get("skip_reason"),
            "bar_count":     meta["bar_count"],
            "last_bar_date": meta["last_bar_date"],
            "oos_accuracy":  meta.get("oos_accuracy"),
            "trained_at":    meta.get("trained_at"),
            "feature_cols":  meta.get("feature_cols", []),
        })
    except Exception:
        tb = traceback.format_exc()
        _log.error("train job %s failed: %s", jid[:8], tb.splitlines()[-1])
        _update_job(jid, {"status": "error", "error": tb})


async def launch_train(jid: str, symbol: str, force: bool = False) -> None:
    asyncio.get_running_loop().run_in_executor(_executor, _run_train_sync, jid, symbol, force)


# ── advisor: analyze ──────────────────────────────────────────────────────────

def _run_analyze_sync(jid: str, symbol: str, force_retrain: bool) -> None:
    """Full stock analysis in a background thread."""
    try:
        _update_job(jid, {"status": "running", "symbol": symbol})
        from api.advisor import analyze_stock
        result = analyze_stock(symbol, force_retrain=force_retrain)
        _update_job(jid, {"status": "done", **result})
    except Exception:
        tb = traceback.format_exc()
        _log.error("analyze job %s failed: %s", jid[:8], tb.splitlines()[-1])
        _update_job(jid, {"status": "error", "error": tb})


async def launch_analyze(jid: str, symbol: str, force_retrain: bool = False) -> None:
    asyncio.get_running_loop().run_in_executor(_executor, _run_analyze_sync, jid, symbol, force_retrain)


# ── advisor: recommend ────────────────────────────────────────────────────────

def _run_recommend_sync(jid: str, sector: str | None, top_n: int) -> None:
    """Generate ranked recommendations in a background thread."""
    try:
        _update_job(jid, {"status": "running", "sector": sector})
        from api.advisor import get_recommendations
        rows = get_recommendations(sector=sector, top_n=top_n)
        _update_job(jid, {"status": "done", "rows": rows, "sector": sector})
    except Exception:
        tb = traceback.format_exc()
        _log.error("recommend job %s failed: %s", jid[:8], tb.splitlines()[-1])
        _update_job(jid, {"status": "error", "error": tb})


async def launch_recommend(jid: str, sector: str | None = None, top_n: int = 10) -> None:
    asyncio.get_running_loop().run_in_executor(_executor, _run_recommend_sync, jid, sector, top_n)

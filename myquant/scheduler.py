"""
myquant/scheduler.py — Automated data-fetch → retrain → strategy-update pipeline.

This module is the brain that makes MyQuant self-improving:

  1. **Data Refresh** (daily after market close, 15:30 CST)
     - Fetches latest OHLCV bars for all tracked symbols via AKShare/yfinance
     - Updates parquet cache and SQLite store
     - Fetches latest fundamentals + macro data

  2. **Model Retrain** (triggered after fresh data arrives)
     - Re-runs the LightGBM training pipeline on updated data
     - Walk-forward validation to prevent overfitting
     - Saves new model artifacts with timestamp versioning

  3. **Strategy Update** (triggered after retrain completes)
     - Updates best_params.json with new optimal parameters
     - Refreshes screener weights based on latest market conditions
     - Triggers auto-tune if model quality degrades

  4. **Health Monitoring**
     - Tracks last run times, success/failure counts
     - Exposes status dict for the API layer

Architecture
────────────
  APScheduler (AsyncIOScheduler) runs inside the FastAPI process.
  Jobs are lightweight — heavy work (backtest, train) runs in ThreadPoolExecutor.
  All state is persisted in SQLite so restarts don't lose track.

Usage
─────
  # As part of FastAPI (normal mode — wired in api/main.py lifespan)
  from myquant.scheduler import scheduler_manager
  await scheduler_manager.start()    # on startup
  await scheduler_manager.shutdown() # on shutdown

  # Standalone CLI
  python -m myquant.scheduler
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from myquant.config.settings import settings

_log = logging.getLogger("myquant.scheduler")

# ── Configuration ─────────────────────────────────────────────────────────────

# Market close time (CST = UTC+8) — schedule jobs after this
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MIN  = 30   # 15:30 — 30 min buffer after 15:00 close

# Default schedule: run daily at 16:00 CST (1 hour after close)
_DEFAULT_DATA_CRON   = {"hour": 16, "minute": 0}
_DEFAULT_RETRAIN_CRON = {"hour": 16, "minute": 30}

# Paths
_BEST_PARAMS_PATH      = _ROOT / "best_params.json"
_SCREENER_WEIGHTS_PATH = _ROOT / "screener_weights.json"
_MODEL_VERSIONS_DIR    = _ROOT / "data" / "model_versions"
_SCHEDULER_STATE_PATH  = _ROOT / "data" / "scheduler_state.json"

# How many model versions to keep
_MAX_MODEL_VERSIONS = 10

# Thread pool for CPU-bound work
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sched_worker")


# ── State tracking ────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    IDLE     = "idle"
    RUNNING  = "running"
    SUCCESS  = "success"
    FAILED   = "failed"
    PAUSED   = "paused"


@dataclass
class PipelineState:
    """Tracks the full auto-update pipeline state."""
    # Overall
    enabled: bool = True
    paused: bool = False

    # Data fetch
    data_status: str = "idle"
    data_last_run: Optional[str] = None
    data_last_success: Optional[str] = None
    data_last_error: Optional[str] = None
    data_symbols_updated: int = 0
    data_run_count: int = 0
    data_fail_count: int = 0

    # Model retrain
    retrain_status: str = "idle"
    retrain_last_run: Optional[str] = None
    retrain_last_success: Optional[str] = None
    retrain_last_error: Optional[str] = None
    retrain_models_updated: int = 0
    retrain_run_count: int = 0
    retrain_fail_count: int = 0

    # Strategy update
    strategy_status: str = "idle"
    strategy_last_run: Optional[str] = None
    strategy_last_success: Optional[str] = None
    strategy_last_error: Optional[str] = None
    strategy_run_count: int = 0

    # Quality metrics
    last_oos_accuracy: Optional[float] = None
    last_sharpe: Optional[float] = None
    last_profit_factor: Optional[float] = None
    last_model_score: Optional[float] = None

    # History of recent runs
    recent_runs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Keep only last 20 runs
        d["recent_runs"] = d["recent_runs"][-20:]
        return d


# ── Core Pipeline Functions ───────────────────────────────────────────────────

def _get_tracked_symbols() -> list[str]:
    """
    Get the list of symbols to track.
    Priority: screener top picks > watchlist.json > hardcoded defaults.
    """
    # 1. Try loading from recent screener results
    try:
        from api import db as _db
        jobs = _db.fetch_all_jobs()
        screen_jobs = [
            j for j in jobs
            if j.get("kind") == "screen" and j.get("status") == "done"
        ]
        if screen_jobs:
            latest = max(screen_jobs, key=lambda j: j.get("_ts", 0))
            top_syms = latest.get("top_symbols", [])
            if top_syms:
                _log.info("Using %d symbols from latest screener run", len(top_syms))
                return top_syms[:15]
    except Exception as e:
        _log.debug("Could not load screener symbols: %s", e)

    # 2. Try watchlist.json
    wl_path = _ROOT / "myquant" / "data" / "watchlist.json"
    if wl_path.exists():
        try:
            wl = json.loads(wl_path.read_text())
            symbols = wl.get("symbols", [])
            if symbols:
                _log.info("Using %d symbols from watchlist.json", len(symbols))
                return symbols[:15]
        except Exception:
            pass

    # 3. Fallback defaults
    return ["sh600519", "sh600036", "sz000858", "sh601318", "sz300750", "sh600900"]


def _refresh_data_sync(state: PipelineState) -> dict:
    """
    Synchronous: Fetch latest OHLCV + fundamentals for all tracked symbols.
    Returns summary dict.
    """
    from myquant.data.fetchers.historical_loader import HistoricalLoader
    from myquant.models.bar import BarInterval

    symbols = _get_tracked_symbols()
    loader = HistoricalLoader()
    now = datetime.now()
    end_date = now.date()
    # Fetch last 2 years to ensure enough training data
    start_date = (now - timedelta(days=730)).date()

    results = {"symbols": [], "errors": [], "bars_fetched": 0}

    for sym in symbols:
        try:
            bars = loader.load_bars(
                symbol=sym,
                interval=BarInterval.D1,
                start=start_date,
                end=end_date,
            )
            n_bars = len(bars) if bars else 0
            results["symbols"].append({"symbol": sym, "bars": n_bars})
            results["bars_fetched"] += n_bars
            _log.info("Fetched %d bars for %s", n_bars, sym)
        except Exception as e:
            _log.warning("Failed to fetch data for %s: %s", sym, e)
            results["errors"].append({"symbol": sym, "error": str(e)})

    # Also refresh fundamental data
    try:
        from myquant.data.fetchers.fundamental_fetcher import FundamentalFetcher
        ff = FundamentalFetcher()
        for sym in symbols[:6]:  # Top 6 only to avoid rate limits
            try:
                ff.fetch(sym)
            except Exception:
                pass
    except Exception as e:
        _log.debug("Fundamental refresh skipped: %s", e)

    # Refresh macro data
    try:
        from myquant.data.fetchers.macro_fetcher import MacroFetcher
        mf = MacroFetcher()
        mf.fetch()
    except Exception as e:
        _log.debug("Macro refresh skipped: %s", e)

    results["symbols_count"] = len(results["symbols"])
    results["error_count"] = len(results["errors"])
    return results


def _retrain_models_sync(state: PipelineState) -> dict:
    """
    Synchronous: Retrain LightGBM models for tracked symbols.
    Uses the existing advisor.py train pipeline with quality checks.
    """
    from api.advisor import analyze_stock, train_for_symbol

    symbols = _get_tracked_symbols()
    results = {
        "models_trained": [],
        "models_skipped": [],
        "errors": [],
        "quality_scores": [],
    }

    for sym in symbols:
        try:
            # Train (or reload if fresh enough)
            train_result = train_for_symbol(sym, force=False)
            action = train_result.get("action", "unknown")

            if action == "trained":
                # Verify quality via analysis
                try:
                    analysis = analyze_stock(sym, force_retrain=False)
                    oos = float((analysis.get("model_meta") or {}).get("oos_accuracy", 0))
                    conf = float(analysis.get("confidence", 0))
                    signal = analysis.get("signal", "HOLD")

                    results["models_trained"].append({
                        "symbol": sym,
                        "oos_accuracy": round(oos, 4),
                        "confidence": round(conf, 4),
                        "signal": signal,
                    })
                    results["quality_scores"].append(oos)
                    _log.info(
                        "Trained %s: OOS=%.1f%% conf=%.1f%% signal=%s",
                        sym, oos * 100, conf * 100, signal,
                    )
                except Exception as e:
                    results["models_trained"].append({
                        "symbol": sym, "oos_accuracy": 0, "analysis_error": str(e),
                    })

            elif action in ("fresh", "insufficient_new_data"):
                results["models_skipped"].append({
                    "symbol": sym,
                    "reason": action,
                    "details": train_result.get("reason", ""),
                })
            else:
                results["models_trained"].append({"symbol": sym, "action": action})

        except Exception as e:
            _log.warning("Failed to retrain %s: %s", sym, e)
            results["errors"].append({"symbol": sym, "error": str(e)})

    # Version the model artifacts
    _version_models(results)

    results["total_trained"] = len(results["models_trained"])
    results["total_skipped"] = len(results["models_skipped"])
    results["total_errors"]  = len(results["errors"])
    avg_oos = (
        sum(results["quality_scores"]) / len(results["quality_scores"])
        if results["quality_scores"] else 0.0
    )
    results["avg_oos_accuracy"] = round(avg_oos, 4)

    return results


def _update_strategy_sync(state: PipelineState, retrain_results: dict) -> dict:
    """
    Synchronous: Update strategy parameters based on retrained models.
    If quality is good → lock in params. If poor → trigger auto-tune adjustment.
    """
    results = {"actions": [], "auto_tune_triggered": False}

    avg_oos = retrain_results.get("avg_oos_accuracy", 0)
    trained = retrain_results.get("models_trained", [])

    # Check if quality warrants a full auto-tune cycle
    quality_ok = avg_oos >= 0.54 and len(trained) > 0

    if quality_ok:
        # Quality is acceptable — just ensure best_params.json reflects latest
        _log.info("Model quality OK (avg OOS=%.1f%%) — keeping current params", avg_oos * 100)
        results["actions"].append({
            "action": "keep_params",
            "reason": f"Model quality acceptable (avg OOS={avg_oos:.1%})",
        })

        # Check if we should run a quick backtest to validate
        try:
            symbols = _get_tracked_symbols()[:3]
            from train_loop import self_test_loop
            test_result = self_test_loop(
                symbols=symbols,
                top_n=3,
                max_rounds=1,
                progress_cb=None,
            )
            best = test_result.get("best", {})
            bt_res = best.get("result", {})

            if best.get("passes"):
                # Save winning config
                config = best.get("config", {})
                if config:
                    _BEST_PARAMS_PATH.write_text(json.dumps(config, indent=2))
                    results["actions"].append({
                        "action": "updated_best_params",
                        "config": config,
                        "sharpe": bt_res.get("sharpe_ratio"),
                        "profit_factor": bt_res.get("profit_factor"),
                    })
                    state.last_sharpe = bt_res.get("sharpe_ratio")
                    state.last_profit_factor = bt_res.get("profit_factor")
            else:
                results["actions"].append({
                    "action": "validation_backtest_failed",
                    "reason": "No config passed quality gates",
                    "best_score": best.get("score"),
                })
        except Exception as e:
            _log.warning("Strategy validation backtest failed: %s", e)
            results["actions"].append({
                "action": "validation_error", "error": str(e),
            })
    else:
        # Quality degraded — trigger lightweight auto-tune
        _log.warning(
            "Model quality degraded (avg OOS=%.1f%%) — triggering auto-tune",
            avg_oos * 100,
        )
        results["auto_tune_triggered"] = True
        try:
            from api.auto_tune import auto_tune_loop
            tune_result = auto_tune_loop(
                symbols=_get_tracked_symbols()[:3],
                top_n=3,
                max_iterations=2,  # Keep it lightweight for scheduled runs
            )
            results["auto_tune_result"] = {
                "converged": tune_result.get("converged"),
                "iterations_run": tune_result.get("iterations_run"),
                "final_score": tune_result.get("final_score"),
            }
            state.last_model_score = tune_result.get("final_score")
        except Exception as e:
            _log.warning("Auto-tune failed: %s", e)
            results["actions"].append({
                "action": "auto_tune_error", "error": str(e),
            })

    return results


def _version_models(retrain_results: dict) -> None:
    """Save a timestamped snapshot of model quality metrics."""
    _MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "models": retrain_results.get("models_trained", []),
        "avg_oos": retrain_results.get("avg_oos_accuracy", 0),
    }

    fname = f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    (_MODEL_VERSIONS_DIR / fname).write_text(json.dumps(snapshot, indent=2))

    # Prune old versions
    versions = sorted(_MODEL_VERSIONS_DIR.glob("models_*.json"))
    for old in versions[:-_MAX_MODEL_VERSIONS]:
        old.unlink()
        _log.debug("Pruned old model version: %s", old.name)


# ── Scheduler Manager ─────────────────────────────────────────────────────────

class SchedulerManager:
    """
    Manages the APScheduler instance and pipeline state.
    Designed to run inside the FastAPI process.
    """

    def __init__(self) -> None:
        self._scheduler = None
        self._state = PipelineState()
        self._started = False
        self._load_state()

    def _load_state(self) -> None:
        """Restore state from disk."""
        if _SCHEDULER_STATE_PATH.exists():
            try:
                data = json.loads(_SCHEDULER_STATE_PATH.read_text())
                for k, v in data.items():
                    if hasattr(self._state, k) and k != "recent_runs":
                        setattr(self._state, k, v)
                self._state.recent_runs = data.get("recent_runs", [])
                _log.info("Restored scheduler state from disk")
            except Exception as e:
                _log.warning("Could not restore scheduler state: %s", e)

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            _SCHEDULER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _SCHEDULER_STATE_PATH.write_text(
                json.dumps(self._state.to_dict(), indent=2, default=str)
            )
        except Exception as e:
            _log.warning("Could not save scheduler state: %s", e)

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._started and not self._state.paused

    async def start(self) -> None:
        """Start the scheduler with configured cron jobs."""
        if self._started:
            _log.warning("Scheduler already started")
            return

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            _log.error(
                "APScheduler not installed. Install with: pip install apscheduler>=3.10"
            )
            _log.info("Scheduler will not start — auto-updates disabled")
            return

        self._scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

        # Job 1: Daily data refresh at 16:00 CST (Mon-Fri)
        self._scheduler.add_job(
            self._run_data_refresh,
            CronTrigger(
                day_of_week="mon-fri",
                hour=_DEFAULT_DATA_CRON["hour"],
                minute=_DEFAULT_DATA_CRON["minute"],
                timezone="Asia/Shanghai",
            ),
            id="daily_data_refresh",
            name="Daily Data Refresh",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,  # Allow up to 1 hour late
        )

        # Job 2: Daily retrain at 16:30 CST (Mon-Fri)
        self._scheduler.add_job(
            self._run_retrain,
            CronTrigger(
                day_of_week="mon-fri",
                hour=_DEFAULT_RETRAIN_CRON["hour"],
                minute=_DEFAULT_RETRAIN_CRON["minute"],
                timezone="Asia/Shanghai",
            ),
            id="daily_retrain",
            name="Daily Model Retrain",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )

        # Job 3: Weekly deep auto-tune (Saturday 10:00 CST)
        self._scheduler.add_job(
            self._run_deep_auto_tune,
            CronTrigger(
                day_of_week="sat",
                hour=10,
                minute=0,
                timezone="Asia/Shanghai",
            ),
            id="weekly_auto_tune",
            name="Weekly Deep Auto-Tune",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=7200,
        )

        # Job 4: Heartbeat every 5 minutes (keeps state fresh)
        self._scheduler.add_job(
            self._heartbeat,
            IntervalTrigger(minutes=5),
            id="heartbeat",
            name="Scheduler Heartbeat",
            replace_existing=True,
        )

        self._scheduler.start()
        self._started = True
        self._state.enabled = True
        self._state.paused = False
        self._save_state()
        _log.info("Scheduler started — auto-updates enabled (Mon-Fri 16:00/16:30 CST)")

    async def shutdown(self) -> None:
        """Gracefully stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
        self._started = False
        self._save_state()
        _log.info("Scheduler stopped")

    async def pause(self) -> None:
        """Pause all scheduled jobs."""
        if self._scheduler:
            self._scheduler.pause()
        self._state.paused = True
        self._save_state()
        _log.info("Scheduler paused")

    async def resume(self) -> None:
        """Resume paused jobs."""
        if self._scheduler:
            self._scheduler.resume()
        self._state.paused = False
        self._save_state()
        _log.info("Scheduler resumed")

    async def trigger_now(self, job_type: str = "full") -> dict:
        """
        Manually trigger a pipeline run.
        job_type: "data" | "retrain" | "strategy" | "full"
        """
        if self._state.data_status == "running" or self._state.retrain_status == "running":
            return {"error": "A pipeline job is already running"}

        if job_type == "data":
            asyncio.create_task(self._run_data_refresh())
        elif job_type == "retrain":
            asyncio.create_task(self._run_retrain())
        elif job_type == "strategy":
            asyncio.create_task(self._run_strategy_update({}))
        elif job_type == "full":
            asyncio.create_task(self._run_full_pipeline())
        else:
            return {"error": f"Unknown job_type: {job_type}"}

        return {"triggered": job_type, "time": datetime.now().isoformat()}

    def get_status(self) -> dict:
        """Return current pipeline status for the API."""
        next_runs = {}
        if self._scheduler and self._started:
            for job in self._scheduler.get_jobs():
                if job.next_run_time:
                    next_runs[job.id] = job.next_run_time.isoformat()

        return {
            **self._state.to_dict(),
            "scheduler_running": self._started,
            "next_scheduled_runs": next_runs,
            "tracked_symbols": _get_tracked_symbols(),
        }

    # ── Pipeline Steps ────────────────────────────────────────────────────────

    async def _run_full_pipeline(self) -> None:
        """Run the complete data → retrain → strategy update pipeline."""
        _log.info("Starting full pipeline run")
        await self._run_data_refresh()

        if self._state.data_status == "success":
            await self._run_retrain()
        else:
            _log.warning("Skipping retrain — data refresh failed")

    async def _run_data_refresh(self) -> None:
        """Fetch latest market data for all tracked symbols."""
        if self._state.paused:
            _log.info("Scheduler paused — skipping data refresh")
            return

        self._state.data_status = "running"
        self._state.data_last_run = datetime.now().isoformat()
        self._state.data_run_count += 1
        self._save_state()

        run_record = {
            "type": "data_refresh",
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_pool, _refresh_data_sync, self._state)

            self._state.data_status = "success"
            self._state.data_last_success = datetime.now().isoformat()
            self._state.data_symbols_updated = result.get("symbols_count", 0)
            self._state.data_last_error = None

            run_record["status"] = "success"
            run_record["symbols_updated"] = result.get("symbols_count", 0)
            run_record["bars_fetched"] = result.get("bars_fetched", 0)
            run_record["errors"] = result.get("error_count", 0)

            _log.info(
                "Data refresh complete: %d symbols, %d bars fetched, %d errors",
                result.get("symbols_count", 0),
                result.get("bars_fetched", 0),
                result.get("error_count", 0),
            )

        except Exception as e:
            self._state.data_status = "failed"
            self._state.data_last_error = str(e)
            self._state.data_fail_count += 1

            run_record["status"] = "failed"
            run_record["error"] = str(e)
            _log.error("Data refresh failed: %s", traceback.format_exc())

        run_record["finished_at"] = datetime.now().isoformat()
        self._state.recent_runs.append(run_record)
        self._state.recent_runs = self._state.recent_runs[-20:]
        self._save_state()

    async def _run_retrain(self) -> None:
        """Retrain models and update strategies."""
        if self._state.paused:
            _log.info("Scheduler paused — skipping retrain")
            return

        self._state.retrain_status = "running"
        self._state.retrain_last_run = datetime.now().isoformat()
        self._state.retrain_run_count += 1
        self._save_state()

        run_record = {
            "type": "retrain",
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }

        try:
            loop = asyncio.get_event_loop()
            retrain_result = await loop.run_in_executor(
                _pool, _retrain_models_sync, self._state
            )

            self._state.retrain_status = "success"
            self._state.retrain_last_success = datetime.now().isoformat()
            self._state.retrain_models_updated = retrain_result.get("total_trained", 0)
            self._state.retrain_last_error = None
            self._state.last_oos_accuracy = retrain_result.get("avg_oos_accuracy")

            run_record["status"] = "success"
            run_record["models_trained"] = retrain_result.get("total_trained", 0)
            run_record["models_skipped"] = retrain_result.get("total_skipped", 0)
            run_record["avg_oos"] = retrain_result.get("avg_oos_accuracy", 0)

            _log.info(
                "Retrain complete: %d trained, %d skipped, avg OOS=%.1f%%",
                retrain_result.get("total_trained", 0),
                retrain_result.get("total_skipped", 0),
                retrain_result.get("avg_oos_accuracy", 0) * 100,
            )

            # Chain: strategy update
            await self._run_strategy_update(retrain_result)

        except Exception as e:
            self._state.retrain_status = "failed"
            self._state.retrain_last_error = str(e)
            self._state.retrain_fail_count += 1

            run_record["status"] = "failed"
            run_record["error"] = str(e)
            _log.error("Retrain failed: %s", traceback.format_exc())

        run_record["finished_at"] = datetime.now().isoformat()
        self._state.recent_runs.append(run_record)
        self._state.recent_runs = self._state.recent_runs[-20:]
        self._save_state()

    async def _run_strategy_update(self, retrain_results: dict) -> None:
        """Update strategy params based on retrain quality."""
        self._state.strategy_status = "running"
        self._state.strategy_last_run = datetime.now().isoformat()
        self._state.strategy_run_count += 1
        self._save_state()

        run_record = {
            "type": "strategy_update",
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _pool, _update_strategy_sync, self._state, retrain_results
            )

            self._state.strategy_status = "success"
            self._state.strategy_last_success = datetime.now().isoformat()
            self._state.strategy_last_error = None

            run_record["status"] = "success"
            run_record["actions"] = len(result.get("actions", []))
            run_record["auto_tune_triggered"] = result.get("auto_tune_triggered", False)

            _log.info("Strategy update complete: %d actions", len(result.get("actions", [])))

        except Exception as e:
            self._state.strategy_status = "failed"
            self._state.strategy_last_error = str(e)

            run_record["status"] = "failed"
            run_record["error"] = str(e)
            _log.error("Strategy update failed: %s", traceback.format_exc())

        run_record["finished_at"] = datetime.now().isoformat()
        self._state.recent_runs.append(run_record)
        self._state.recent_runs = self._state.recent_runs[-20:]
        self._save_state()

    async def _run_deep_auto_tune(self) -> None:
        """Weekly deep auto-tune with more iterations."""
        if self._state.paused:
            return

        _log.info("Starting weekly deep auto-tune")
        run_record = {
            "type": "deep_auto_tune",
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }

        try:
            loop = asyncio.get_event_loop()

            def _deep_tune():
                from api.auto_tune import auto_tune_loop
                return auto_tune_loop(
                    symbols=_get_tracked_symbols()[:5],
                    top_n=5,
                    max_iterations=3,
                )

            result = await loop.run_in_executor(_pool, _deep_tune)

            run_record["status"] = "success"
            run_record["converged"] = result.get("converged")
            run_record["final_score"] = result.get("final_score")
            run_record["iterations"] = result.get("iterations_run")

            self._state.last_model_score = result.get("final_score")
            _log.info(
                "Deep auto-tune complete: converged=%s score=%.0f",
                result.get("converged"), result.get("final_score", 0),
            )

        except Exception as e:
            run_record["status"] = "failed"
            run_record["error"] = str(e)
            _log.error("Deep auto-tune failed: %s", traceback.format_exc())

        run_record["finished_at"] = datetime.now().isoformat()
        self._state.recent_runs.append(run_record)
        self._state.recent_runs = self._state.recent_runs[-20:]
        self._save_state()

    async def _heartbeat(self) -> None:
        """Periodic heartbeat to keep state fresh."""
        self._save_state()


# ── Module-level singleton ────────────────────────────────────────────────────

scheduler_manager = SchedulerManager()


# ── CLI entry point ───────────────────────────────────────────────────────────

async def _cli_main():
    """Run the scheduler standalone (not inside FastAPI)."""
    import signal as _sig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    print("=" * 60)
    print("  MyQuant Scheduler — Standalone Mode")
    print("  Auto-updates: data fetch → retrain → strategy update")
    print("=" * 60)

    await scheduler_manager.start()

    stop_event = asyncio.Event()

    def _handle_signal():
        print("\nShutting down scheduler...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (_sig.SIGINT, _sig.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    print("Scheduler running. Press Ctrl+C to stop.")
    print(f"Tracked symbols: {_get_tracked_symbols()}")
    print(f"Next data refresh: daily at {_DEFAULT_DATA_CRON['hour']}:{_DEFAULT_DATA_CRON['minute']:02d} CST")
    print(f"Next model retrain: daily at {_DEFAULT_RETRAIN_CRON['hour']}:{_DEFAULT_RETRAIN_CRON['minute']:02d} CST")
    print()

    await stop_event.wait()
    await scheduler_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(_cli_main())

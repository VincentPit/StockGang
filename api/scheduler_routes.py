"""
api/scheduler_routes.py — FastAPI routes for the auto-update scheduler.

Endpoints:
  GET  /api/scheduler/status    → current pipeline status + next run times
  POST /api/scheduler/trigger   → manually trigger a pipeline run
  POST /api/scheduler/pause     → pause scheduled jobs
  POST /api/scheduler/resume    → resume scheduled jobs
  GET  /api/scheduler/history   → recent pipeline run history
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SchedulerStatus(BaseModel):
    scheduler_running: bool = False
    enabled: bool = True
    paused: bool = False

    # Data refresh
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

    # Quality
    last_oos_accuracy: Optional[float] = None
    last_sharpe: Optional[float] = None
    last_profit_factor: Optional[float] = None
    last_model_score: Optional[float] = None

    # Schedule
    next_scheduled_runs: dict = {}
    tracked_symbols: list[str] = []

    # History
    recent_runs: list[dict] = []


class TriggerRequest(BaseModel):
    job_type: str = Field(
        default="full",
        description="Pipeline to run: 'data', 'retrain', 'strategy', or 'full'",
    )


class TriggerResponse(BaseModel):
    triggered: Optional[str] = None
    time: Optional[str] = None
    error: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=SchedulerStatus)
async def get_scheduler_status():
    """Get current scheduler and pipeline status."""
    from myquant.scheduler import scheduler_manager
    return scheduler_manager.get_status()


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_pipeline(req: TriggerRequest):
    """Manually trigger a pipeline run (data refresh, retrain, or full)."""
    from myquant.scheduler import scheduler_manager
    _log.info("Manual trigger requested: %s", req.job_type)
    result = await scheduler_manager.trigger_now(req.job_type)
    return result


@router.post("/pause")
async def pause_scheduler():
    """Pause all scheduled auto-update jobs."""
    from myquant.scheduler import scheduler_manager
    await scheduler_manager.pause()
    return {"paused": True}


@router.post("/resume")
async def resume_scheduler():
    """Resume paused scheduler jobs."""
    from myquant.scheduler import scheduler_manager
    await scheduler_manager.resume()
    return {"paused": False}


@router.get("/history")
async def get_pipeline_history(limit: int = Query(default=20, ge=1, le=100)):
    """Get recent pipeline run history."""
    from myquant.scheduler import scheduler_manager
    runs = scheduler_manager.state.recent_runs[-limit:]
    return {"runs": runs, "total": len(scheduler_manager.state.recent_runs)}

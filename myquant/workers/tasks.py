"""
myquant/workers/tasks.py — Arq task wrappers.

Each task delegates to the matching ``_run_*_sync`` function in api/runner.py
(executed in a thread because the underlying code is sync and CPU-bound).
Job state is persisted in Postgres by ``_update_job`` inside each runner —
notifications are emitted via pg_notify in api/db.upsert_job, so WebSocket
listeners pick up progress without polling.

Why import inside the task body
-------------------------------
api/runner.py performs heavy bootstrapping at import time (init_db, restore
jobs from disk). We delay the import to first task invocation so a worker
process that never receives a task — e.g. one starting before migrations
finish in CI — does not touch Postgres prematurely.
"""
from __future__ import annotations

import asyncio
from typing import Any


async def run_backtest(ctx: dict[str, Any], jid: str, req: dict) -> dict:
    from api.runner import _run_backtest_sync
    await asyncio.to_thread(_run_backtest_sync, jid, req)
    return {"jid": jid}


async def run_screener(ctx: dict[str, Any], jid: str, req: dict) -> dict:
    from api.runner import _run_screener_sync
    await asyncio.to_thread(_run_screener_sync, jid, req)
    return {"jid": jid}


async def run_workflow(ctx: dict[str, Any], jid: str, req: dict) -> dict:
    from api.runner import _run_workflow_sync
    await asyncio.to_thread(_run_workflow_sync, jid, req)
    return {"jid": jid}


async def run_train_loop(ctx: dict[str, Any], jid: str, req: dict) -> dict:
    from api.runner import _run_train_loop_sync
    await asyncio.to_thread(_run_train_loop_sync, jid, req)
    return {"jid": jid}


async def run_auto_tune(ctx: dict[str, Any], jid: str, req: dict) -> dict:
    from api.runner import _run_auto_tune_sync
    await asyncio.to_thread(_run_auto_tune_sync, jid, req)
    return {"jid": jid}


async def run_train(ctx: dict[str, Any], jid: str, symbol: str, force: bool) -> dict:
    from api.runner import _run_train_sync
    await asyncio.to_thread(_run_train_sync, jid, symbol, force)
    return {"jid": jid}


async def run_analyze(ctx: dict[str, Any], jid: str, symbol: str, force_retrain: bool) -> dict:
    from api.runner import _run_analyze_sync
    await asyncio.to_thread(_run_analyze_sync, jid, symbol, force_retrain)
    return {"jid": jid}


async def run_recommend(ctx: dict[str, Any], jid: str, sector: str | None, top_n: int) -> dict:
    from api.runner import _run_recommend_sync
    await asyncio.to_thread(_run_recommend_sync, jid, sector, top_n)
    return {"jid": jid}


async def run_walk_forward(ctx: dict[str, Any], jid: str, req: dict) -> dict:
    from api.runner import _run_walk_forward_sync
    await asyncio.to_thread(_run_walk_forward_sync, jid, req)
    return {"jid": jid}


async def run_monte_carlo(ctx: dict[str, Any], jid: str, req: dict) -> dict:
    from api.runner import _run_monte_carlo_sync
    await asyncio.to_thread(_run_monte_carlo_sync, jid, req)
    return {"jid": jid}

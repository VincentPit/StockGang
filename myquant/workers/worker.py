"""
myquant/workers/worker.py — Arq worker settings.

Run with:
    arq myquant.workers.worker.WorkerSettings

Connection details come from myquant.config.settings so the API and the
worker container share one Redis configuration.

Long-running jobs
-----------------
Auto-tune and walk-forward can run for tens of minutes. Arq's default
``job_timeout`` is 5 minutes, which would silently kill them. We bump
it to 60 minutes; individual tasks can override via @task(timeout=...).
"""
from __future__ import annotations

import logging
from typing import Any

from arq.connections import RedisSettings

from myquant.config.settings import settings

from .tasks import (
    run_analyze,
    run_auto_tune,
    run_backtest,
    run_monte_carlo,
    run_recommend,
    run_screener,
    run_train,
    run_train_loop,
    run_walk_forward,
    run_workflow,
)

_log = logging.getLogger("myquant.workers")


def redis_settings() -> RedisSettings:
    """Build Arq RedisSettings from project settings (single source of truth)."""
    return RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=settings.REDIS_DB,
    )


_METRICS_PORT = 9101  # exposed inside the container; Prometheus scrapes worker:9101


async def on_startup(ctx: dict[str, Any]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _log.info(
        "Arq worker online — redis=%s:%d db=%d",
        settings.REDIS_HOST, settings.REDIS_PORT, settings.REDIS_DB,
    )

    # Bring up a tiny HTTP server publishing the custom metrics registry.
    # Arq runs each worker as its own process, so port collisions on a
    # single host are avoided by docker-compose port-mapping the service.
    try:
        from prometheus_client import start_http_server
        from myquant.observability.metrics import REGISTRY
        start_http_server(_METRICS_PORT, registry=REGISTRY)
        _log.info("Worker metrics endpoint listening on :%d/metrics", _METRICS_PORT)
    except Exception as exc:  # pragma: no cover — observability optional
        _log.warning("Worker metrics endpoint disabled: %s", exc)

    # OTel tracing — instrument SQLAlchemy/Redis so worker spans connect
    # back to the API span via the X-Request-ID and traceparent headers.
    try:
        from myquant.observability.tracing import setup_tracing
        setup_tracing(app=None, service_name="myquant-worker")
    except Exception as exc:  # pragma: no cover
        _log.warning("Worker OTel disabled: %s", exc)


async def on_shutdown(ctx: dict[str, Any]) -> None:
    _log.info("Arq worker shutting down")


class WorkerSettings:
    """Arq picks this class up when invoked as `arq myquant.workers.worker.WorkerSettings`."""

    functions = [
        run_backtest,
        run_screener,
        run_workflow,
        run_train_loop,
        run_auto_tune,
        run_train,
        run_analyze,
        run_recommend,
        run_walk_forward,
        run_monte_carlo,
    ]
    redis_settings: RedisSettings = redis_settings()
    on_startup = on_startup
    on_shutdown = on_shutdown

    job_timeout = 3600          # 60 min — auto-tune/walk-forward can be long
    max_jobs = 4                # concurrent jobs per worker process
    keep_result = 60            # discard completed Arq metadata after 1 min
    queue_name = "myquant:jobs"

"""
myquant/observability/metrics.py — Custom Prometheus metrics.

Used by both the API process (mounted on /api/metrics) and the Arq worker
(exposed via the standalone HTTP server in worker.py). Keeping the
definitions in one place means dashboards rendering the same metric name
get a consistent label set across processes.

Conventions
-----------
* Counters end in ``_total``.
* Histograms record durations in seconds.
* Labels stay low-cardinality — ``kind`` is one of ~10 known job types,
  not a free-form string.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

#: Process-local registry. The instrumentator reuses the global default
#: registry; we keep ours separate so the worker can publish *just* its
#: custom metrics on its dedicated port without leaking FastAPI internals.
REGISTRY = CollectorRegistry(auto_describe=True)


# ── Job lifecycle ─────────────────────────────────────────────────────────────
JOB_DURATION = Histogram(
    "myquant_job_duration_seconds",
    "End-to-end duration of a background job, by kind and outcome",
    labelnames=("kind", "outcome"),  # outcome: done | error
    registry=REGISTRY,
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

JOB_FAILURES_TOTAL = Counter(
    "myquant_job_failures_total",
    "Background jobs that ended in error",
    labelnames=("kind",),
    registry=REGISTRY,
)

JOB_STARTED_TOTAL = Counter(
    "myquant_job_started_total",
    "Background jobs entering the running state",
    labelnames=("kind",),
    registry=REGISTRY,
)

JOBS_INFLIGHT = Gauge(
    "myquant_jobs_inflight",
    "Background jobs currently in pending or running state",
    labelnames=("kind",),
    registry=REGISTRY,
)


# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_HITS_TOTAL = Counter(
    "myquant_cache_hits_total",
    "Successful cache lookups (L1 + L2)",
    labelnames=("tier",),  # tier: l1 | l2
    registry=REGISTRY,
)

CACHE_MISSES_TOTAL = Counter(
    "myquant_cache_misses_total",
    "Cache lookups that missed both tiers",
    registry=REGISTRY,
)


# ── Broker / orders ───────────────────────────────────────────────────────────
BROKER_ORDERS_TOTAL = Counter(
    "myquant_broker_orders_total",
    "Orders submitted to a broker, by side and broker mode",
    labelnames=("action", "mode"),  # action: BUY|SELL ; mode: live|simulator
    registry=REGISTRY,
)

BROKER_ORDER_FAILURES_TOTAL = Counter(
    "myquant_broker_order_failures_total",
    "Order submissions that raised an error",
    labelnames=("mode",),
    registry=REGISTRY,
)

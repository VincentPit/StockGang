"""
myquant/observability — Metrics, logs, and tracing primitives (T2).

Layout
------
  metrics.py    Custom Prometheus metrics (job duration, failures, cache
                hit ratio, broker order counts) shared by the API and the
                Arq worker.

The observability package is intentionally optional: importing it must not
raise even when prometheus_client is missing, so that tests and dev
environments without observability deps still boot.
"""
from __future__ import annotations

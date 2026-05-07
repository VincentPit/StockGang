"""
myquant/observability/tracing.py — OpenTelemetry tracing setup (T2c).

Initialises a tracer provider that exports spans to an OTLP collector
(Tempo by default). Wires auto-instrumentation for FastAPI, SQLAlchemy,
and Redis so any HTTP request → DB query → Redis call automatically lands
on a single connected trace.

Activation
----------
Call ``setup_tracing(app)`` from ``api/main.py`` after the FastAPI app is
created. Disabling: set ``OTEL_TRACES_EXPORTER=none`` in the env.
"""
from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

_OTEL_ENDPOINT_DEFAULT = "http://tempo:4317"


def setup_tracing(app: Any | None = None, *, service_name: str = "myquant-api") -> None:
    """Wire up OTLP tracing. Safe to call without the optional deps installed."""
    if os.getenv("OTEL_TRACES_EXPORTER", "otlp").lower() == "none":
        _log.info("OTel tracing disabled via OTEL_TRACES_EXPORTER=none")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:
        _log.warning("OTel tracing disabled (SDK missing): %s", exc)
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", _OTEL_ENDPOINT_DEFAULT)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)

    # Best-effort instrumentation — each integration is optional
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/health,/api/metrics,/api/metrics/myquant")
        except Exception as exc:
            _log.warning("FastAPI OTel instrumentation skipped: %s", exc)

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        SQLAlchemyInstrumentor().instrument()
    except Exception as exc:
        _log.warning("SQLAlchemy OTel instrumentation skipped: %s", exc)

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
    except Exception as exc:
        _log.warning("Redis OTel instrumentation skipped: %s", exc)

    _log.info("OTel tracing → %s (service=%s)", endpoint, service_name)

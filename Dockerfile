# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile (T3c). Stage 1 uses uv for ~10x-faster installs; the
# final image carries only the Python runtime + installed wheels, so it stays
# slim and reproducible. Runs as non-root user 1000:1000.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder — installs Python deps with uv ──────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install deps in their own layer — invalidated only when requirements.txt changes
COPY requirements.txt .
RUN uv pip install --system --no-cache --requirement requirements.txt


# ── Stage 2: runtime — slim Python image with the installed packages ─────────
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libs only — curl for healthcheck, libpq for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching the builder UID/GID
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid 1000 --create-home app

WORKDIR /app

# Copy installed packages from the builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application source — owned by the non-root user
COPY --chown=app:app . .

# Runtime directories the app expects
RUN mkdir -p logs data data/model_versions data/cache && chown -R app:app /app

USER 1000:1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

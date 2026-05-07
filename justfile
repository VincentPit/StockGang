# MyQuant — task runner. Install `just` (https://github.com/casey/just) and
# run `just` for the recipe list. The aim is one-line commands for everything
# a contributor reaches for daily.

# Default — show recipe list
default:
    @just --list

# ── Local dev ─────────────────────────────────────────────────────────────────
# Bring up only the dependencies a Python/Next dev needs (Postgres + Redis).
# The API and worker run on the host so reloads are fast.
dev-deps:
    docker compose up -d db redis migrate

# Run the FastAPI server with auto-reload
dev-api:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run an Arq worker against the local Redis/Postgres
dev-worker:
    MYQUANT_QUEUE=arq arq myquant.workers.worker.WorkerSettings

# Run the standalone scheduler (replaces the in-API one)
dev-scheduler:
    python -m myquant.scheduler_main

# Run the Next.js dev server
dev-web:
    cd web && npm run dev

# ── Tests / lint ─────────────────────────────────────────────────────────────
test:
    pytest tests/ -q --tb=short

test-web:
    cd web && npm test

lint:
    ruff check api/ tests/ myquant/

format:
    ruff format api/ tests/ myquant/

typecheck:
    mypy --ignore-missing-imports --no-strict-optional api/ myquant/observability/ myquant/workers/

audit:
    pip-audit --requirement requirements.txt --vulnerability-service osv

# ── Migrations ───────────────────────────────────────────────────────────────
migrate:
    alembic upgrade head

migrate-rev message="":
    alembic revision --autogenerate -m "{{message}}"

# ── Docker ───────────────────────────────────────────────────────────────────
# Bring up the full stack (api, worker, scheduler, web, db, redis, observability)
docker:
    docker compose up -d --build

docker-logs service="":
    docker compose logs -f --tail=100 {{service}}

docker-down:
    docker compose down

# ── Observability shortcuts ──────────────────────────────────────────────────
grafana:
    @echo "http://localhost:3001 (admin / myquant123)"

prom:
    @echo "http://localhost:9090"

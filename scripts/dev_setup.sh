#!/usr/bin/env bash
set -euo pipefail

# ─── MyQuant — Local DB bring-up ─────────────────────────────────────────────
# Brings up Postgres + Redis via docker-compose, applies Alembic migrations,
# and verifies the api/db.py shim can connect. Idempotent — safe to re-run.
#
# Usage:  chmod +x scripts/dev_setup.sh && ./scripts/dev_setup.sh
# ──────────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'
step() { echo -e "\n${CYAN}▸ $1${NC}"; }
ok()   { echo -e "  ${GREEN}✔ $1${NC}"; }

step "Bringing up db + redis"
docker compose up -d db redis
ok "containers started"

step "Waiting for Postgres to accept connections"
until docker compose exec -T db pg_isready -U myquant >/dev/null 2>&1; do
    sleep 1
done
ok "Postgres is ready"

step "Applying Alembic migrations"
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
alembic upgrade head
ok "schema is at head"

step "Pinging the data layer"
python3 -c "import api.db; api.db.init_db(); print('  api.db connected OK')"
ok "api/db.py connects to Postgres"

echo ""
echo "Next:"
echo "  uvicorn api.main:app --reload --port 8000"

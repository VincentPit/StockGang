#!/usr/bin/env bash
set -euo pipefail

# ─── MyQuant Local Setup ─────────────────────────────────────────────────────
# One-command setup: creates venv, installs deps, seeds the DB, and verifies.
# Usage:  chmod +x scripts/setup_local.sh && ./scripts/setup_local.sh
# ──────────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step() { echo -e "\n${CYAN}▸ $1${NC}"; }
ok()   { echo -e "  ${GREEN}✔ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${NC}"; }

echo "═══════════════════════════════════════════"
echo "   MyQuant — Local Environment Setup"
echo "═══════════════════════════════════════════"

# ── 1. Python venv ───────────────────────────────────────────────────────────
step "Setting up Python virtual environment"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists"
fi
source .venv/bin/activate
ok "Activated .venv ($(python3 --version))"

# ── 2. Python dependencies ──────────────────────────────────────────────────
step "Installing Python dependencies"
pip install --upgrade pip -q
pip install -r requirements.txt -q 2>&1 | tail -1
pip install -e . -q 2>&1 | tail -1
ok "All Python packages installed"

# ── 3. .env file ────────────────────────────────────────────────────────────
step "Checking .env configuration"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        warn "Created .env from .env.example — review and fill in your API keys"
    else
        cat > .env <<'EOF'
ENV=development
TUSHARE_TOKEN=
FUTU_HOST=127.0.0.1
FUTU_PORT=11111
REDIS_URL=redis://localhost:6379/0
SCHEDULER_ENABLED=true
EOF
        warn "Created default .env — fill in your API tokens"
    fi
else
    ok ".env file exists"
fi

# ── 4. Data directories ─────────────────────────────────────────────────────
step "Ensuring data directories"
mkdir -p data/cache data/model_versions logs
ok "data/, logs/ directories ready"

# ── 5. Verify imports ───────────────────────────────────────────────────────
step "Verifying core imports"
python3 -c "
from myquant.scheduler import scheduler_manager
from api.main import app
from myquant.backtest.simulator import Backtester
from myquant.strategy.registry import StrategyRegistry
print('  ✔ All core imports OK')
"

# ── 6. Run tests ────────────────────────────────────────────────────────────
step "Running test suite"
PASS_COUNT=$(python3 -m pytest tests/ -q --tb=no 2>&1 | tail -1)
ok "Tests: $PASS_COUNT"

# ── 7. Node.js frontend ─────────────────────────────────────────────────────
step "Setting up web frontend"
if command -v node &>/dev/null; then
    cd web
    if [ ! -d "node_modules" ]; then
        npm install --silent 2>&1 | tail -1
        ok "npm packages installed"
    else
        ok "node_modules already exists"
    fi
    cd "$ROOT"
else
    warn "Node.js not found — skip frontend setup (install Node 18+ to enable)"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo -e "  ${GREEN}✔ Setup complete!${NC}"
echo "═══════════════════════════════════════════"
echo ""
echo "  Quick start:"
echo "    source .venv/bin/activate"
echo ""
echo "    # Start the API server (with auto-update scheduler)"
echo "    uvicorn api.main:app --reload --port 8000"
echo ""
echo "    # Start the web UI"
echo "    cd web && npm run dev"
echo ""
echo "    # Or run the CLI scheduler standalone"
echo "    python -m myquant scheduler"
echo ""
echo "    # Trigger a one-time pipeline run"
echo "    python -m myquant trigger"
echo ""

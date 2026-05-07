# MyQuant — A-Share Quantitative Trading Platform

An institutional-grade quantitative trading system for Shanghai & Shenzhen A-share markets. Combines a multi-strategy backtester, LightGBM ML signals, walk-forward validation, Monte Carlo robustness analysis, an autonomous parameter-tuning loop, and a cron scheduler — all wired up to a modern Next.js dashboard.

---

## What's in the box

### Discovery & analysis
- **Stock Screener** — parallel scan of CSI 300 / CSI 300+500 universe ranked on five factors (trend strength, ATR adequacy, autocorrelation, 6-month momentum, max-drawdown penalty). Each result ships with a **causal trace** showing the factor breakdown and gate-check pass/fail.
- **AI Advisor** — per-symbol LightGBM classifier on ~1y of OHLCV-derived features. Returns BUY/HOLD/SELL with class probabilities and feature importance. Recommendations rank a curated 33-stock universe across 9 sectors using fundamentals × momentum × ML composite.
- **Research Panel** — OHLCV chart (90d/180d/1y/2y), fundamentals (P/E, P/B, ROE, growth, margin, dividend, composite scores), stock & macro news headlines, market regime badge (RISK_ON / NEUTRAL / RISK_OFF) with signal multiplier.

### Backtesting & validation
- **Backtester** — event-driven replay with a 5-strategy ensemble (LightGBM, MA crossover, RSI filter, MACD, news sentiment). Realistic cost model: 0.03% commission + stamp duty + slippage. Per-position stop-loss and per-symbol cumulative loss caps.
- **Walk-Forward** — rolling out-of-sample validation across N folds. Reports per-fold Sharpe / return / drawdown plus a consistency score so you can spot strategies that only work in-sample.
- **Monte Carlo** — bootstrap-resamples the trade log to produce a distribution of equity curves, percentile bands, VaR/CVaR, and probability-of-loss.
- **Workflow Pipeline** — one-click screen → backtest with live status (Queued → Screening → Backtesting → Done).

### Autonomous loops
- **Train Loop** — screen → backtest → adjust → retrain in a closed feedback cycle, persisting trial history and best-params.
- **Auto-Tune** — autonomous parameter tuner that diagnoses each iteration's weaknesses (overtrading, drawdown, low Sharpe, etc.) and proposes targeted adjustments. Records reasoning per iteration.
- **Scheduler** — APScheduler-backed cron jobs for nightly screening, daily model retrains, and weekly walk-forward refreshes. State persisted to disk so restarts don't drop schedules.

### Execution & risk
- **Paper trading** — full order book (BUY/SELL, MARKET/LIMIT, review-then-confirm), live position tracking with avg cost / unrealised P&L, persistent across restarts, reset-to-¥500k anytime. Available inline from the screener, recommendation cards, and analyzer.
- **RiskGate** — every signal passes through 7 layers before reaching the broker: market state, throttle, daily-drawdown circuit breaker (−3%), position limit (20% NAV), sector exposure (40% NAV), 1-day 95% VaR (<2% NAV), per-symbol cooldown.
- **Analytics** — fund-grade performance metrics (Sharpe, Sortino, Calmar, rolling stats, tail risk, monthly/yearly heatmaps) and factor attribution.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Next.js 14 (web/)                          │
│  Screener · Advisor · Backtest · Workflow · TrainLoop             │
│  AutoTune · WalkForward · MonteCarlo · Scheduler · Account        │
└─────────────────────────┬────────────────────────────────────────┘
                          │ HTTP / WebSocket
┌─────────────────────────▼────────────────────────────────────────┐
│                       FastAPI (api/)                              │
│  main.py — routes, rate limit, security headers, CORS, lifespan   │
│  runner.py — thread-pool job launcher (jobs survive restarts)     │
│  advisor.py — LightGBM train / analyze / recommend                │
│  auto_tune.py — diagnose-and-adjust autonomous loop               │
│  scheduler_routes.py — cron job CRUD                              │
│  schemas.py — Pydantic v2 request/response models                 │
│  db.py — SQLite persistence (jobs, cache, models, broker)         │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│                    myquant/  (core library)                       │
│                                                                   │
│  backtest/        simulator.py, walk_forward.py                   │
│  strategy/        alpha_model, regime, macro_filter, sizing,      │
│                   technical (MA/RSI/MACD), nlp/news_strategy      │
│  risk/            risk_gate (7 layers), advanced_risk             │
│  analytics/       performance.py, attribution.py                  │
│  portfolio/       portfolio_engine, optimizer                     │
│  execution/       order_manager, algorithms, brokers/PaperBroker  │
│  data/fetchers/   yfinance, AKShare, fundamentals, news, regime   │
│  monitoring/      Streamlit dashboard, alerts                     │
│  scheduler.py     APScheduler manager (cron-driven jobs)          │
│  tools/           parallel universe screener with causal trace    │
└──────────────────────────────────────────────────────────────────┘
```

**Background jobs.** Every expensive operation (backtest, screener, training, auto-tune, walk-forward, Monte Carlo) runs on a `ThreadPoolExecutor`. The API returns a `job_id` immediately; the frontend polls or subscribes via WebSocket. Job state is written through to SQLite so it survives restarts.

**Two-level cache.** External data fetches (price, fundamentals, news, regime) are cached in memory (L1) and SQLite (L2) with per-type TTLs (4h price · 24h fundamentals · 30m news). L1 is consulted first with zero I/O.

**Causal trace.** Every screener result and recommendation includes a serialised factor breakdown — score components, gate checks, and the final decision path — so the dashboard can render *why* a stock was picked, not just *that* it was.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript 5, Tailwind, Recharts, Lucide |
| Backend | FastAPI, Pydantic v2, Uvicorn (ASGI), APScheduler |
| ML | LightGBM, scikit-learn, scipy |
| NLP | SnowNLP (Chinese sentiment), jieba |
| Data sources | yfinance, AKShare, tushare |
| Persistence (dev) | SQLite (jobs, models, cache, paper broker) |
| Persistence (Docker) | TimescaleDB (PostgreSQL 16) + Redis 7 |
| Monitoring | Grafana, Streamlit |
| Browser automation | Playwright (WebBroker) |
| Testing | pytest, ruff, tsc |

---

## Quick start (local)

### Prerequisites
- Python 3.11+
- Node.js 18+

### Install

```bash
# Python backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Next.js frontend
cd web && npm install && cd ..
```

### Configure

```bash
cp .env.example .env
# .env ships with safe defaults — no edits needed for local dev
```

### Run

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null
source .venv/bin/activate
(uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &) && \
  cd web && npm run dev
```

Open **http://localhost:3000** — API at **http://localhost:8000** — interactive docs at **http://localhost:8000/docs**.

---

## Docker (full stack)

```bash
cp .env.example .env
docker compose up --build
```

| Service | Port | Role |
|---|---|---|
| `web` (Next.js) | 3000 | Dashboard |
| `api` (FastAPI) | 8000 | Backend + scheduler |
| `db` (TimescaleDB) | 5432 | Time-series storage |
| `redis` | 6379 | Cache & queue backbone |
| `grafana` | 3001 | Monitoring dashboards |
| `dashboard` (Streamlit) | 8501 | Live engine monitor |
| `engine` | — | Trading engine (paper mode, background) |

Default Grafana login: `admin` / `myquant123`.

---

## Tests & CI

```bash
# Python (~530 tests)
pytest tests/ -q

# Lint
ruff check api/ tests/

# Frontend type-check
cd web && npx tsc --noEmit
```

GitHub Actions (`.github/workflows/ci.yml`) runs pytest, ruff, `tsc --noEmit`, and a Docker build smoke test on every push to `main`/`develop` and PRs into `main`. The web Jest config currently runs with `--passWithNoTests`; component-level tests are a known gap.

---

## API reference (key endpoints)

| Method | Path | Description |
|---|---|---|
| `POST` / `GET` | `/api/screen[/{id}]` | Universe screener |
| `POST` / `GET` | `/api/backtest[/{id}]` | Single backtest |
| `GET` | `/api/backtest/{id}/nav` | NAV time-series for charting |
| `POST` / `GET` | `/api/workflow[/{id}]` | Screen → backtest pipeline |
| `POST` / `GET` | `/api/train-loop[/{id}]` | Closed-loop training cycle |
| `POST` / `GET` | `/api/auto-tune[/{id}]` | Autonomous parameter tuner |
| `POST` / `GET` | `/api/walk-forward[/{id}]` | Rolling out-of-sample validation |
| `POST` / `GET` | `/api/monte-carlo[/{id}]` | Bootstrap robustness analysis |
| `POST` / `GET` | `/api/advisor/train[/{id}]` | Per-symbol LightGBM training |
| `POST` / `GET` | `/api/advisor/analyze[/{id}]` | Full stock analysis |
| `GET` | `/api/advisor/recommend[/{sector}]` | Ranked picks (optionally per-sector) |
| `GET` / `DELETE` | `/api/advisor/models[/{symbol}]` | Stored model registry |
| `GET` | `/api/price/{symbol}` | OHLCV bars |
| `GET` | `/api/fundamentals/{symbol}` | P/E, P/B, ROE, composite scores |
| `GET` | `/api/news/{symbol}` · `/api/news/macro` | Headlines |
| `GET` | `/api/regime` | Market regime + signal multiplier |
| `GET` | `/api/universe` | Candidate symbol list |
| `*` | `/api/scheduler/*` | Cron job CRUD |
| `POST` / `GET` / `DELETE` | `/api/orders` · `/api/account[/reset]` | Paper trading |
| `WS` | `/api/ws/{job_id}` | Live job progress stream |
| `GET` | `/api/health` | Liveness + uptime |

Full interactive docs: **http://localhost:8000/docs**

---

## Project layout

```
MyQuant/
├── api/
│   ├── main.py             # FastAPI app — routes, middleware, lifespan
│   ├── runner.py           # Thread-pool job launcher
│   ├── advisor.py          # LightGBM train / analyze / recommend
│   ├── auto_tune.py        # Autonomous diagnose-and-adjust loop
│   ├── scheduler_routes.py # Cron job CRUD endpoints
│   ├── schemas.py          # Pydantic v2 models
│   └── db.py               # SQLite persistence
│
├── myquant/
│   ├── backtest/           # simulator.py, walk_forward.py
│   ├── strategy/           # alpha_model, regime, macro_filter, sizing, …
│   ├── risk/               # risk_gate.py, advanced_risk.py
│   ├── analytics/          # performance.py, attribution.py
│   ├── portfolio/          # portfolio_engine.py, optimizer.py
│   ├── execution/          # order_manager, algorithms, brokers/
│   ├── data/               # fetchers/, store/
│   ├── monitoring/         # Streamlit dashboard, alerts
│   ├── engine/             # trading_engine.py
│   ├── tools/              # stock_screener.py
│   └── scheduler.py        # APScheduler manager
│
├── web/
│   ├── app/                # advisor, screener, backtest, workflow,
│   │                       # trainloop, autotune, walkforward,
│   │                       # montecarlo, scheduler
│   ├── components/         # Panels (one per feature) + shared widgets
│   └── lib/                # API client, account & nav contexts
│
├── tests/                  # pytest suite
├── infra/grafana/          # Datasource + dashboard provisioning
├── scripts/                # setup_local.sh
├── data/                   # SQLite, model versions, cache, scheduler state
├── docker-compose.yml      # Full stack (db, redis, api, web, engine, …)
├── Dockerfile              # Shared Python image
└── .github/workflows/ci.yml
```

---

## Curated universe (33 stocks across 9 sectors)

| Sector | Stocks |
|---|---|
| Finance | 招商银行, 中国平安, 工商银行, 兴业银行, 平安银行, 东方财富 |
| Consumer | 贵州茅台, 五粮液, 美的集团, 伊利股份, 中国中免, 泸州老窖, 正大食品, 海天味业 |
| EV / Tech | 宁德时代 (CATL), 比亚迪, 海康威视, TCL科技 |
| Energy | 长江电力, 中国神华 |
| Healthcare | 恒瑞医药, 爱尔眼科 |
| Materials | 紫金矿业, 隆基绿能, 海螺水泥 |
| Industrial | 三一重工 |
| Transport | 上海机场, 顺丰控股 |

---

## License

MIT

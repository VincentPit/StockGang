# MyQuant — A-Share Quantitative Trading Platform

A full-stack quantitative trading system for Shanghai & Shenzhen A-share markets. Combines a multi-strategy backtesting engine, LightGBM ML signals, live data fetching, and a paper-trading simulator — all wired up to a modern Next.js dashboard.

---

## Screenshots

| Find Stocks | AI Signals | Research |
|---|---|---|
| Screener with causal trace | Recommendations & order form | Price chart + fundamentals |

---

## Feature Overview

### 🔍 Stock Screener
- Scans CSI 300 or CSI 300+500 universe (configurable)
- Ranks stocks on 5 quantitative signals: trend strength, ATR adequacy, autocorrelation, 6-month momentum, max-drawdown penalty
- Every result ships with a **causal trace** — a factor breakdown showing exactly why a stock scored the way it did, with gate-check pass/fail badges

### 🤖 AI Signals (Advisor)
- **Stock Analyzer** — trains a LightGBM classifier per symbol on ~1 year of daily OHLCV features; returns BUY/HOLD/SELL signal + class probabilities + feature importance
- **Recommendations** — scores the full curated universe (33 A-share leaders across 9 sectors) using fundamentals × momentum × ML signal composite; each pick includes a causal trace
- **Stored Models** — view, manage, and delete per-symbol trained models; staleness rules prevent unnecessary retraining (30-day calendar / 5-bar minimum thresholds)

### 📈 Research Panel
- OHLCV price chart (90d / 180d / 1yr / 2yr)
- Fundamentals card: P/E, P/B, ROE, revenue growth, net margin, dividend yield, composite value/growth/quality scores
- Stock news headlines (30-minute cache)
- Macro news feed
- Market regime badge (RISK_ON / NEUTRAL / RISK_OFF) with signal multiplier

### 📊 Backtester
- Event-driven replay of historical bars
- 5-strategy ensemble: LightGBM, MA Crossover, RSI Filter, MACD, News Sentiment
- Realistic cost model: 0.03% commission + stamp duty + slippage
- Stop-loss per position (-8%) and per-symbol cumulative loss cap
- Output: total return, Sharpe, max drawdown, win rate, NAV chart, per-symbol P&L, full trade log

### ⚡ Workflow Pipeline
- Single-click screen → backtest: runs the screener to find top-N stocks, then immediately backtests them
- Configurable: universe index, screener lookback, top-N, backtest window, initial cash
- Live status bar: Queued → Screening → Backtesting → Done

### 💼 Paper Trading
- Full order book: BUY / SELL, MARKET / LIMIT, review-then-confirm flow
- Live position tracking with avg cost, current price, unrealised P&L %
- Persistent across server restarts (SQLite)
- Reset to ¥500,000 anytime
- Available inline from screener results, recommendation cards, and the analyzer

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Next.js 14 (web/)                  │
│  ScreenerPanel · AdvisorPanel · ResearchPanel        │
│  BacktestPanel · WorkflowPanel · AccountWidget       │
│  CausalTracePanel · PriceChart · NavChart …          │
└────────────────────┬────────────────────────────────┘
                     │ HTTP / WebSocket
┌────────────────────▼────────────────────────────────┐
│               FastAPI  (api/)                        │
│  main.py · schemas.py · advisor.py                   │
│  runner.py (thread pool) · db.py (SQLite)            │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              myquant/  (core library)                │
│                                                      │
│  backtest/simulator.py    — event-driven engine      │
│  strategy/ml/             — LightGBM feature eng.    │
│  strategy/technical/      — MA, RSI, MACD            │
│  strategy/nlp/            — news sentiment           │
│  risk/risk_gate.py        — 7-layer signal filter    │
│  portfolio/               — NAV & position tracking  │
│  execution/brokers/       — PaperBroker              │
│  data/fetchers/           — yfinance, AKShare,       │
│                             fundamentals, news       │
│  tools/stock_screener.py  — parallel universe scan   │
└─────────────────────────────────────────────────────┘
```

**Background jobs** — every expensive operation (backtest, screener, ML training, recommendations) runs in a `ThreadPoolExecutor` thread. The API returns a `job_id` immediately; the frontend polls until `status == "done"`. Jobs survive server restarts via SQLite write-through.

**Two-level cache** — all external data fetches (price, fundamentals, news, regime) are cached in memory (L1) and SQLite (L2) with per-type TTLs (4h price · 24h fundamentals · 30m news). L1 is consulted first with zero I/O.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS, Recharts, Lucide |
| Backend | FastAPI, Pydantic v2, Uvicorn (ASGI) |
| ML | LightGBM, scikit-learn |
| NLP | SnowNLP (Chinese sentiment), jieba |
| Data | yfinance, AKShare, tushare |
| Persistence | SQLite (jobs, models, cache, paper broker) |
| Testing | pytest (307 tests), Jest + Testing Library (67 tests) |

---

## Quick Start (local)

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1 — Clone & install

```bash
git clone https://github.com/VincentPit/StockGang.git
cd StockGang

# Python backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Next.js frontend (run from web/ subdirectory)
cd web && npm install && cd ..
```

### 2 — Configure

```bash
cp .env.example .env
# .env ships with safe defaults for local dev — no changes needed to just run it
```

### 3 — Run (one command)

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null; \
source .venv/bin/activate && \
(uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &) && \
cd web && npm run dev
```

Open **http://localhost:3000**

- API runs at **http://localhost:8000**
- Interactive API docs at **http://localhost:8000/docs**

---

## Docker (full stack)

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| Next.js frontend | http://localhost:3000 |
| FastAPI backend | http://localhost:8000 |
| Grafana dashboards | http://localhost:3001 |
| Streamlit monitor | http://localhost:8501 |

---

## Running Tests

```bash
# Python (307 tests)
pytest tests/ -q

# Frontend (67 tests)
cd web && npm test
```

---

## Project Structure

```
MyQuant/
├── api/
│   ├── main.py          # FastAPI routes + rate limiter + lifespan
│   ├── schemas.py       # Pydantic request/response models
│   ├── advisor.py       # LightGBM training, analysis, recommendations
│   ├── runner.py        # Background job runner (thread pool + job store)
│   └── db.py            # SQLite persistence (jobs, cache, models, broker)
│
├── myquant/
│   ├── backtest/
│   │   └── simulator.py         # Event-driven backtesting engine
│   ├── strategy/
│   │   ├── ml/
│   │   │   ├── lgbm_strategy.py       # LightGBM signal strategy
│   │   │   └── feature_engineer.py    # 22 OHLCV-derived features
│   │   ├── technical/
│   │   │   ├── ma_crossover.py
│   │   │   ├── rsi_strategy.py
│   │   │   └── macd_strategy.py
│   │   ├── nlp/
│   │   │   └── news_strategy.py       # SnowNLP sentiment
│   │   ├── risk_gate.py  → risk/
│   │   └── sizing.py                  # ATR position sizing
│   ├── risk/
│   │   └── risk_gate.py     # 7-layer signal filter (drawdown, VaR, sector, …)
│   ├── data/fetchers/
│   │   ├── historical_loader.py   # yfinance OHLCV + symbol mapping
│   │   ├── universe_fetcher.py    # CSI 300 / CSI 500 constituents (AKShare)
│   │   ├── fundamental_fetcher.py # P/E, P/B, ROE, composite scores
│   │   ├── news_fetcher.py        # Stock & macro headlines
│   │   └── macro_proxy.py         # Regime detection (RISK_ON/OFF)
│   ├── execution/brokers/
│   │   └── paper_broker.py        # Paper trading with cost model
│   ├── portfolio/
│   │   └── portfolio_engine.py    # NAV, drawdown, Sharpe tracking
│   └── tools/
│       └── stock_screener.py      # Parallel universe scanner + causal trace
│
├── web/
│   ├── app/
│   │   ├── page.tsx         # App shell (tabs, account widget)
│   │   └── layout.tsx
│   ├── components/
│   │   ├── ScreenerPanel.tsx
│   │   ├── AdvisorPanel.tsx
│   │   ├── ResearchPanel.tsx
│   │   ├── BacktestPanel.tsx
│   │   ├── WorkflowPanel.tsx
│   │   ├── CausalTracePanel.tsx   # Factor attribution visualiser
│   │   ├── AccountWidget.tsx
│   │   ├── PriceChart.tsx
│   │   ├── NavChart.tsx
│   │   └── …
│   └── lib/
│       ├── api.ts              # Typed API client
│       ├── account-context.tsx # Global account polling (30s interval)
│       └── nav-context.tsx     # Tab routing + cross-panel symbol jump
│
├── tests/                  # 307 pytest tests
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API Reference (key endpoints)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/screen` | Launch screener job |
| `GET` | `/api/screen/{id}` | Poll screener results |
| `POST` | `/api/backtest` | Launch backtest job |
| `GET` | `/api/backtest/{id}` | Poll backtest results |
| `POST` | `/api/workflow` | Launch screen→backtest pipeline |
| `GET` | `/api/workflow/{id}` | Poll workflow results |
| `POST` | `/api/advisor/analyze` | Launch full stock analysis |
| `GET` | `/api/advisor/recommend` | Get ranked recommendations (all sectors) |
| `GET` | `/api/advisor/recommend/{sector}` | Filter by sector |
| `GET` | `/api/price/{symbol}` | OHLCV bars |
| `GET` | `/api/fundamentals/{symbol}` | P/E, P/B, ROE, composite scores |
| `GET` | `/api/news/{symbol}` | Stock news headlines |
| `GET` | `/api/regime` | Market regime + signal multiplier |
| `POST` | `/api/orders` | Submit paper order |
| `GET` | `/api/account` | Account cash + live positions |
| `DELETE` | `/api/account/reset` | Reset simulator to ¥500,000 |
| `WS` | `/api/ws/{job_id}` | Live job progress stream |

Full interactive docs: **http://localhost:8000/docs**

---

## Curated Universe (33 stocks across 9 sectors)

| Sector | Key stocks |
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

## Risk Controls

The `RiskGate` checks every signal through 7 layers before execution reaches the broker:

1. **Market state** — no trading during halt / auction / off-hours
2. **Throttle** — max orders per minute (configurable)
3. **Daily drawdown circuit breaker** — halts all trading if daily P&L < −3%
4. **Position limit** — max 20% NAV in any single symbol
5. **Sector exposure** — max 40% NAV in any single sector
6. **VaR check** — 1-day 95% VaR must stay under 2% NAV
7. **Cooldown timer** — prevents repeated trades on same symbol within N seconds

---

## License

MIT

"""
api/main.py — FastAPI application for MyQuant.

Endpoints:
  POST /api/backtest              → start a backtest job
  GET  /api/backtest/{id}         → poll job status / results
  GET  /api/backtest/{id}/nav     → nav timeseries (chart data)
  POST /api/screen                → start a screener job
  GET  /api/screen/{id}           → poll screener results
  POST /api/workflow              → screen → backtest pipeline job
  GET  /api/workflow/{id}         → poll workflow results
  GET  /api/price/{symbol}        → OHLCV bars (yfinance)
  GET  /api/fundamentals/{symbol} → PE / PB / ROE / scores
  GET  /api/news/{symbol}         → stock news headlines
  GET  /api/news/macro            → macro news headlines
  GET  /api/regime                → market regime (RISK_ON/OFF/NEUTRAL)
  GET  /api/universe              → full candidate symbol list
  GET  /api/jobs                  → list all jobs
  GET  /api/health                → health check
  WS   /api/ws/{job_id}           → live progress stream
  ── Advisor ────────────────────────────────────────────────────────
  POST /api/advisor/train         → start model training job for a symbol
  GET  /api/advisor/train/{id}    → poll training job
  POST /api/advisor/analyze       → start full stock analysis job
  GET  /api/advisor/analyze/{id}  → poll analysis job result
  GET  /api/advisor/recommend     → top picks (all sectors)
  GET  /api/advisor/recommend/{sector} → top picks filtered by sector
  GET  /api/advisor/models        → list all stored trained models
  DELETE /api/advisor/models/{symbol} → delete a stored model (force retrain)
  GET  /api/advisor/sectors       → list available sectors
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger(__name__)

_VERSION = "1.0.0"  # single source of truth — consumed by FastAPI() and /api/health

from api import db as _db
from api.schemas import (
    AccountInfo,
    AccountPosition,
    AnalysisResponse,
    AnalyzeRequest,
    BacktestRequest,
    BacktestResponse,
    FundamentalsResponse,
    ModelListResponse,
    NewsResponse,
    OrderRequest,
    OrderResponse,
    PriceResponse,
    RecommendResponse,
    RegimeResponse,
    ScreenRequest,
    ScreenResponse,
    TrainLoopRequest,
    TrainLoopResponse,
    TrainRequest,
    TrainResponse,
    TrainTrialRow,
    UniverseResponse,
    WorkflowRequest,
    WorkflowResponse,
)
from api.runner import (
    get_fundamentals_sync,
    get_job,
    get_macro_news_sync,
    get_news_sync,
    get_price_sync,
    get_regime_sync,
    launch_analyze,
    launch_backtest,
    launch_recommend,
    launch_screener,
    launch_train,
    launch_train_loop,
    launch_workflow,
    list_jobs,
    new_job,
)


# ── Request-ID middleware ─────────────────────────────────────────────────────

class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID to every response for end-to-end tracing."""
    async def dispatch(self, request: StarletteRequest, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add defensive browser-security headers to every response."""
    _HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options":        "DENY",
        "X-XSS-Protection":       "1; mode=block",
        "Referrer-Policy":        "strict-origin-when-cross-origin",
        "Permissions-Policy":     "geolocation=(), microphone=(), camera=()",
    }
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers.update(self._HEADERS)
        return response


# ── In-memory rate limiter ────────────────────────────────────────────────────

class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter — no external dependencies required.

    Per remote IP, per 60-second rolling window:
    • 60 requests  — global cap on all routes
    • 10 requests  — cap on expensive POST routes that launch background jobs
    """

    _WINDOW       = 60.0
    _GLOBAL_MAX   = 60
    _LAUNCH_MAX   = 10
    _LAUNCH_PATHS = frozenset({
        "/api/backtest",
        "/api/screen",
        "/api/workflow",
        "/api/advisor/train",
        "/api/advisor/analyze",
        "/api/train-loop",
    })

    def __init__(self, app) -> None:
        super().__init__(app)
        self._hits: dict[str, collections.deque[float]] = {}

    def _count(self, key: str, now: float) -> int:
        """Trim expired entries and return current-window hit count."""
        cutoff = now - self._WINDOW
        dq = self._hits.setdefault(key, collections.deque())
        while dq and dq[0] < cutoff:
            dq.popleft()
        return len(dq)

    def _record(self, key: str, now: float) -> None:
        self._hits.setdefault(key, collections.deque()).append(now)

    async def dispatch(self, request: StarletteRequest, call_next) -> Response:
        ip  = request.client.host if request.client else "unknown"
        now = time.time()

        if self._count(f"g:{ip}", now) >= self._GLOBAL_MAX:
            return Response(
                content='{"detail":"Rate limit exceeded \u2014 please slow down"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        path = request.url.path
        # Normalise the sector-variant of the recommend path so both
        # /api/advisor/recommend and /api/advisor/recommend/{sector}
        # share a single rate-limit bucket.
        is_recommend_launch = (
            request.method == "GET"
            and (
                path == "/api/advisor/recommend"
                or path.startswith("/api/advisor/recommend/")
            )
        )
        rl_path = "/api/advisor/recommend" if is_recommend_launch else path
        if (request.method == "POST" and path in self._LAUNCH_PATHS) or is_recommend_launch:
            if self._count(f"r:{ip}:{rl_path}", now) >= self._LAUNCH_MAX:
                return Response(
                    content='{"detail":"Too many job submissions \u2014 wait 60 s before retrying"}',
                    status_code=429,
                    media_type="application/json",
                    headers={"Retry-After": "60"},
                )
            self._record(f"r:{ip}:{rl_path}", now)

        self._record(f"g:{ip}", now)
        return await call_next(request)


# ── Startup timestamp (exposed via /api/health) ───────────────────────────────────────
_startup_at: float = 0.0

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """DB initialisation and cache pruning on startup; graceful teardown on shutdown."""
    global _startup_at
    _startup_at = time.time()
    await asyncio.to_thread(_db.init_db)
    deleted = await asyncio.to_thread(_db.purge_expired)
    if deleted:
        _log.info("Pruned %d expired cache rows on startup", deleted)
    _log.info("MyQuant API started")
    yield
    _log.info("MyQuant API shutting down")


# ── CORS origin whitelist (comma-separated env var) ───────────────────────────

_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]


app = FastAPI(
    title="MyQuant API",
    description="Quantitative trading backtest & stock screener API",
    version=_VERSION,
    lifespan=_lifespan,
)

# Middleware order matters: added last = executed first
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(_RateLimitMiddleware)
app.add_middleware(_RequestIDMiddleware)
app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _unhandled_exc(request: StarletteRequest, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return a structured 500 instead of a raw traceback."""
    _log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    jstat  = await asyncio.to_thread(_db.jobs_stats)
    cstat  = await asyncio.to_thread(_db.cache_stats)
    uptime = round(time.time() - _startup_at, 1) if _startup_at else 0.0
    return {
        "status":   "ok",
        "version":  _VERSION,
        "uptime_s": uptime,
        "jobs":     jstat,
        "cache":    cstat,
    }


# ── Backtest ──────────────────────────────────────────────────────────────────

@app.post("/api/backtest", response_model=BacktestResponse, status_code=202)
async def start_backtest(req: BacktestRequest):
    jid = new_job("backtest")
    await launch_backtest(jid, req.model_dump())
    return BacktestResponse(job_id=jid, status="pending", symbols=req.symbols)


@app.get("/api/backtest/{job_id}", response_model=BacktestResponse)
async def get_backtest(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _backtest_response(job)


@app.get("/api/backtest/{job_id}/nav")
async def get_nav(job_id: str):
    """Return just the NAV time series for the chart."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        return {"status": job["status"], "data": []}
    return {"status": "done", "data": job.get("nav_series", [])}


# ── Screener ──────────────────────────────────────────────────────────────────

@app.post("/api/screen", response_model=ScreenResponse, status_code=202)
async def start_screen(req: ScreenRequest):
    jid = new_job("screen")
    await launch_screener(jid, req.model_dump())
    return ScreenResponse(job_id=jid, status="pending")


@app.get("/api/screen/{job_id}", response_model=ScreenResponse)
async def get_screen(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _screen_response(job)


# ── Job list ──────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
async def jobs():
    return [
        {
            "id":         j["id"],
            "kind":       j["kind"],
            "status":     j["status"],
            "created_at": j.get("created_at"),
        }
        for j in list_jobs()
    ]


# ── WebSocket: live progress ──────────────────────────────────────────────────

@app.websocket("/api/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """Poll job every 500 ms and push status updates until done/error."""
    await websocket.accept()
    try:
        while True:
            job = get_job(job_id)
            if job is None:
                await websocket.send_json({"status": "not_found"})
                break
            status = job["status"]
            if status == "done":
                if job["kind"] == "backtest":
                    await websocket.send_json({"status": "done", "data": _backtest_response(job).model_dump()})
                else:
                    await websocket.send_json({"status": "done", "data": _screen_response(job).model_dump()})
                break
            elif status == "error":
                raw_err  = job.get("error") or "unknown error"
                safe_err = raw_err.splitlines()[-1] if "\n" in raw_err else raw_err
                await websocket.send_json({"status": "error", "error": safe_err})
                break
            else:
                msg: dict = {"status": status}
                if job.get("pct") is not None:
                    msg["pct"] = job["pct"]
                if job.get("step"):
                    msg["step"] = job["step"]
                await websocket.send_json(msg)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ── serialiser helpers ────────────────────────────────────────────────────────

def _backtest_response(job: dict) -> BacktestResponse:
    return BacktestResponse(
        job_id          = job["id"],
        status          = job["status"],
        pct             = job.get("pct"),
        step            = job.get("step"),
        period_start    = job.get("period_start"),
        period_end      = job.get("period_end"),
        symbols         = job.get("symbols", []),
        initial_nav     = job.get("initial_nav"),
        final_nav       = job.get("final_nav"),
        total_pnl       = job.get("total_pnl"),
        total_pnl_pct   = job.get("total_pnl_pct"),
        sharpe          = job.get("sharpe"),
        max_dd          = job.get("max_dd"),
        num_trades      = job.get("num_trades"),
        win_rate        = job.get("win_rate"),
        avg_win         = job.get("avg_win"),
        avg_loss        = job.get("avg_loss"),
        profit_factor   = job.get("profit_factor"),
        nav_series      = job.get("nav_series", []),
        trades          = job.get("trades", []),
        symbol_pnl      = job.get("symbol_pnl", []),
        strategy_fills  = job.get("strategy_fills", []),
        error           = job.get("error"),
    )


def _screen_response(job: dict) -> ScreenResponse:
    return ScreenResponse(
        job_id        = job["id"],
        status        = job["status"],
        pct           = job.get("pct"),
        step          = job.get("step"),
        top_symbols   = job.get("top_symbols", []),
        rows          = job.get("rows", []),
        universe_size = job.get("universe_size", 0),
        error         = job.get("error"),
    )


# ── Train Loop (screen → backtest → tune) ───────────────────────────────────

@app.post("/api/train-loop", response_model=TrainLoopResponse, status_code=202)
async def start_train_loop(req: TrainLoopRequest):
    """
    Launch the automated screen→backtest→tune feedback loop as a background job.
    Poll GET /api/train-loop/{job_id} for live progress (pct + step).
    """
    jid = new_job("train_loop")
    await launch_train_loop(jid, req.model_dump())
    return TrainLoopResponse(job_id=jid, status="pending")


@app.get("/api/train-loop/{job_id}", response_model=TrainLoopResponse)
async def get_train_loop(job_id: str):
    """Poll a train-loop job by ID."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _train_loop_response(job)


def _train_loop_response(job: dict) -> TrainLoopResponse:
    raw_trials = job.get("all_trials", [])
    trials = []
    for t in raw_trials:
        trials.append(TrainTrialRow(
            symbol        = t.get("symbol", ""),
            config        = t.get("config", ""),
            passes        = bool(t.get("passes")),
            score         = float(t.get("score", -999)),
            elapsed_s     = float(t.get("elapsed_s", 0)),
            total_pnl     = t.get("total_pnl"),
            profit_factor = t.get("profit_factor"),
            win_rate      = t.get("win_rate"),
            sharpe_ratio  = t.get("sharpe_ratio"),
            num_trades    = t.get("num_trades"),
            error         = t.get("error"),
        ))
    return TrainLoopResponse(
        job_id        = job["id"],
        status        = job["status"],
        pct           = job.get("pct"),
        step          = job.get("step"),
        found_passing = job.get("found_passing"),
        rounds_run    = job.get("rounds_run"),
        best_symbol   = job.get("best_symbol"),
        best_config   = job.get("best_config"),
        best_pf       = job.get("best_pf"),
        best_wr       = job.get("best_wr"),
        best_pnl      = job.get("best_pnl"),
        best_trades   = job.get("best_trades"),
        best_sharpe   = job.get("best_sharpe"),
        symbols_tested= job.get("symbols_tested", []),
        all_trials    = trials,
        error         = job.get("error"),
    )


# ── Workflow (screen → backtest) ──────────────────────────────────────────────

@app.post("/api/workflow", response_model=WorkflowResponse, status_code=202)
async def start_workflow(req: WorkflowRequest):
    jid = new_job("workflow")
    await launch_workflow(jid, req.model_dump())
    return WorkflowResponse(job_id=jid, status="pending")


@app.get("/api/workflow/{job_id}", response_model=WorkflowResponse)
async def get_workflow(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _workflow_response(job)


def _workflow_response(job: dict) -> WorkflowResponse:
    return WorkflowResponse(
        job_id        = job["id"],
        status        = job["status"],
        pct           = job.get("pct"),
        step          = job.get("step"),
        top_symbols   = job.get("top_symbols", []),
        screen_rows   = job.get("screen_rows", []),
        period_start  = job.get("period_start"),
        period_end    = job.get("period_end"),
        initial_nav   = job.get("initial_nav"),
        final_nav     = job.get("final_nav"),
        total_pnl     = job.get("total_pnl"),
        total_pnl_pct = job.get("total_pnl_pct"),
        sharpe        = job.get("sharpe"),
        max_dd        = job.get("max_dd"),
        num_trades    = job.get("num_trades"),
        win_rate      = job.get("win_rate"),
        nav_series    = job.get("nav_series", []),
        trades        = job.get("trades", []),
        symbol_pnl    = job.get("symbol_pnl", []),
        error         = job.get("error"),
    )


# ── Price / OHLCV ─────────────────────────────────────────────────────────────

@app.get("/api/price/{symbol}", response_model=PriceResponse)
async def get_price(
    symbol: str,
    days: int = Query(default=365, ge=30, le=1095),
    response: Response = None,
):
    """Return OHLCV bars for a symbol via yfinance. Cached 4 h client-side."""
    result = await asyncio.to_thread(get_price_sync, symbol, days)
    if response is not None:
        response.headers["Cache-Control"] = "public, max-age=14400"
    return PriceResponse(**result)


# ── Fundamentals ─────────────────────────────────────────────────────────────

@app.get("/api/fundamentals/{symbol}", response_model=FundamentalsResponse)
async def get_fundamentals(symbol: str, response: Response = None):
    """Return fundamental snapshot: PE, PB, ROE, value/growth/quality scores. Cached 24 h."""
    result = await asyncio.to_thread(get_fundamentals_sync, symbol)
    if response is not None:
        response.headers["Cache-Control"] = "public, max-age=86400"
    return FundamentalsResponse(**result)


# ── News ──────────────────────────────────────────────────────────────────────

@app.get("/api/news/macro", response_model=NewsResponse)
async def get_macro_news(
    limit: int = Query(default=30, ge=1, le=100),
    response: Response = None,
):
    """Return macro/economic news headlines. Cached 30 m."""
    result = await asyncio.to_thread(get_macro_news_sync, limit)
    if response is not None:
        response.headers["Cache-Control"] = "public, max-age=1800"
    return NewsResponse(**result)


@app.get("/api/news/{symbol}", response_model=NewsResponse)
async def get_news(
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100),
    response: Response = None,
):
    """Return recent news headlines for a symbol. Cached 30 m."""
    result = await asyncio.to_thread(get_news_sync, symbol, limit)
    if response is not None:
        response.headers["Cache-Control"] = "public, max-age=1800"
    return NewsResponse(**result)


# ── Regime ────────────────────────────────────────────────────────────────────

@app.get("/api/regime", response_model=RegimeResponse)
async def get_regime(
    symbols: str = Query(
        default="",
        description="Comma-separated symbol list to analyse (e.g. sh600519,sz300750)"
    ),
    response: Response = None,
):
    """Derive market regime from 1-year price history. Cached 4 h."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        # No symbols provided — return a neutral stance instead of using hardcoded defaults
        return RegimeResponse(regime="NEUTRAL", signal_multiplier=1.0, symbols_analyzed=0)
    result   = await asyncio.to_thread(get_regime_sync, sym_list)
    if response is not None:
        response.headers["Cache-Control"] = "public, max-age=14400"
    return RegimeResponse(**result)

# ── Cache management ─────────────────────────────────────────────────────────────

@app.get("/api/cache")
async def cache_stats():
    """Return cache statistics (hit counts, TTL breakdown)."""
    return await asyncio.to_thread(_db.cache_stats)


@app.delete("/api/cache")
async def cache_invalidate(prefix: str = Query(default="", description="Key prefix to delete, or empty to wipe all")):
    """
    Invalidate cached data.  Use prefix= to target a subset:
      price      → all price caches
      fund       → all fundamentals caches
      news       → all news caches (stock + macro)
      regime     → all regime caches
      price:hk00700:365  → one specific entry
    """
    deleted = await asyncio.to_thread(_db.cache_invalidate, prefix)
    return {"deleted": deleted, "prefix": prefix or "(all)"}


@app.post("/api/cache/purge")
async def cache_purge():
    """Manually delete expired cache entries (normally done at startup)."""
    deleted = await asyncio.to_thread(_db.purge_expired)
    return {"purged": deleted}

# ── Universe ─────────────────────────────────────────────────────────────────

@app.get("/api/universe", response_model=UniverseResponse)
async def get_universe(indices: list[str] = Query(default=["000300"])):
    """Return the live CSI-index constituent universe used by the screener."""
    from myquant.data.fetchers.universe_fetcher import fetch_universe
    raw = fetch_universe(indices=indices)
    syms = [{"symbol": r["sym"], "yf_ticker": r["yf_ticker"], "name": r["name"]} for r in raw]
    return UniverseResponse(symbols=syms)


# ── Advisor ───────────────────────────────────────────────────────────────────

# ── Train ─────────────────────────────────────────────────────────────────────

@app.post("/api/advisor/train", response_model=TrainResponse, status_code=202)
async def start_train(req: TrainRequest):
    """
    Launch a background job to train (or reload) a LightGBM model for one symbol.

    The job respects staleness rules — if the model is fresh and force_retrain=False,
    training is skipped and the existing model metadata is returned immediately.
    """
    jid = new_job("train")
    await launch_train(jid, req.symbol, req.force_retrain)
    return TrainResponse(job_id=jid, status="pending", symbol=req.symbol)


@app.get("/api/advisor/train/{job_id}", response_model=TrainResponse)
async def get_train(job_id: str):
    """Poll a training job by ID."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return TrainResponse(
        job_id        = job["id"],
        status        = job["status"],
        symbol        = job.get("symbol", ""),
        model_id      = job.get("model_id"),
        train_status  = job.get("train_status"),
        skip_reason   = job.get("skip_reason"),
        bar_count     = job.get("bar_count"),
        last_bar_date = job.get("last_bar_date"),
        oos_accuracy  = job.get("oos_accuracy"),
        trained_at    = job.get("trained_at"),
        feature_cols  = job.get("feature_cols", []),
        error         = job.get("error"),
    )


# ── Analyze ───────────────────────────────────────────────────────────────────

@app.post("/api/advisor/analyze", response_model=AnalysisResponse, status_code=202)
async def start_analyze(req: AnalyzeRequest):
    """
    Launch a background job for full stock analysis:
    trains / loads LGBM model, computes signal, fetches fundamentals + news.
    """
    jid = new_job("analyze")
    await launch_analyze(jid, req.symbol, req.force_retrain)
    return AnalysisResponse(job_id=jid, status="pending", symbol=req.symbol)


@app.get("/api/advisor/analyze/{job_id}", response_model=AnalysisResponse)
async def get_analyze(job_id: str):
    """Poll an analysis job by ID."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _analysis_response(job)


def _analysis_response(job: dict) -> AnalysisResponse:
    from api.schemas import ModelMeta
    mm = None
    raw_meta = job.get("model_meta")
    if isinstance(raw_meta, dict):
        mm = ModelMeta(
            model_id      = raw_meta.get("model_id", ""),
            trained_at    = raw_meta.get("trained_at", 0.0),
            bar_count     = raw_meta.get("bar_count", 0),
            last_bar_date = raw_meta.get("last_bar_date", ""),
            oos_accuracy  = raw_meta.get("oos_accuracy"),
            train_status  = raw_meta.get("train_status", "loaded"),
            skip_reason   = raw_meta.get("skip_reason"),
        )
    return AnalysisResponse(
        job_id             = job["id"],
        status             = job["status"],
        symbol             = job.get("symbol", ""),
        sector             = job.get("sector", ""),
        signal             = job.get("signal", ""),
        confidence         = job.get("confidence", 0.0),
        p_buy              = job.get("p_buy", 0.0),
        p_hold             = job.get("p_hold", 0.0),
        p_sell             = job.get("p_sell", 0.0),
        momentum           = job.get("momentum", {}),
        recent_bars        = job.get("recent_bars", []),
        fundamentals       = job.get("fundamentals", {}),
        news               = job.get("news", []),
        feature_importance = job.get("feature_importance", []),
        model_meta         = mm,
        error              = job.get("error"),
    )


# ── Recommend ─────────────────────────────────────────────────────────────────

@app.get("/api/advisor/recommend", response_model=RecommendResponse)
async def recommend_general(
    top_n: int = Query(default=10, ge=1, le=30),
):
    """
    Launch a background job to rank the full universe and return top picks.
    Uses fundamentals + momentum + stored LGBM signals (if available).
    """
    jid = new_job("recommend")
    await launch_recommend(jid, sector=None, top_n=top_n)
    return RecommendResponse(job_id=jid, status="pending")


@app.get("/api/advisor/recommend/{sector}", response_model=RecommendResponse)
async def recommend_sector(
    sector: str,
    top_n: int = Query(default=10, ge=1, le=30),
):
    """
    Launch a background job to rank stocks within a specific sector.
    sector must be one of: tech, finance, consumer, ev, energy, healthcare,
    materials, telco, transport.
    """
    jid = new_job("recommend")
    await launch_recommend(jid, sector=sector, top_n=top_n)
    return RecommendResponse(job_id=jid, status="pending", sector=sector)


@app.get("/api/advisor/recommend-poll/{job_id}", response_model=RecommendResponse)
async def poll_recommend(job_id: str):
    """Poll a recommendation job by ID."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return RecommendResponse(
        job_id = job["id"],
        status = job["status"],
        sector = job.get("sector"),
        rows   = job.get("rows", []),
        error  = job.get("error"),
    )


# ── Stored models ─────────────────────────────────────────────────────────────

@app.get("/api/advisor/models", response_model=ModelListResponse)
async def list_models():
    """Return metadata for all stored trained models (no blobs)."""
    models = await asyncio.to_thread(_db.list_models)
    return ModelListResponse(models=models)


@app.delete("/api/advisor/models/{symbol}")
async def delete_model(symbol: str):
    """
    Delete a stored model for the given symbol.
    The next call to train or analyze will trigger a fresh model training run.
    """
    deleted = await asyncio.to_thread(_db.delete_model, symbol)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No stored model for {symbol}")
    return {"deleted": True, "symbol": symbol}


# ── Sectors ───────────────────────────────────────────────────────────────────

@app.get("/api/advisor/sectors")
async def get_sectors():
    """Return the list of available sectors for sector-filtered recommendations."""
    from api.advisor import KNOWN_SECTORS
    return {"sectors": KNOWN_SECTORS}


# ── Broker singletons (lazily connected) ─────────────────────────────────────

_PAPER_INITIAL_CASH: float = 500_000.0

_web_broker        = None
_web_broker_lock   = asyncio.Lock()
_paper_broker      = None
_paper_broker_lock = asyncio.Lock()


async def _get_web_broker():
    """Return the connected WebBroker singleton, connecting if necessary."""
    global _web_broker
    async with _web_broker_lock:
        if _web_broker is None:
            from myquant.execution.brokers.web_broker import WebBroker
            _web_broker = WebBroker()
            await _web_broker.connect()
            _log.info("WebBroker connected for order execution")
        return _web_broker


async def _get_paper_broker():
    """Return the PaperBroker singleton, restoring persisted state on first call."""
    global _paper_broker
    async with _paper_broker_lock:
        if _paper_broker is None:
            from myquant.execution.brokers.paper_broker import PaperBroker
            cash, positions = await asyncio.to_thread(_db.get_paper_state)
            pb = PaperBroker(initial_cash=cash)
            await pb.connect()
            pb._positions  = {s: d["qty"]      for s, d in positions.items()}
            pb._avg_prices = {s: d["avg_price"] for s, d in positions.items()}
            _paper_broker = pb
            _log.info("PaperBroker restored: cash=%.2f positions=%d", cash, len(positions))
        return _paper_broker


def _is_live() -> bool:
    return bool(os.getenv("WEB_BROKER_USERNAME"))


async def _get_broker():
    if _is_live():
        return await _get_web_broker()
    return await _get_paper_broker()


async def _save_paper_state(broker) -> None:
    """Persist PaperBroker cash + positions to SQLite after each fill."""
    await asyncio.to_thread(_db.set_paper_cash, broker._cash)
    for sym, qty in list(broker._positions.items()):
        avg = broker._avg_prices.get(sym, 0.0)
        await asyncio.to_thread(_db.upsert_paper_position, sym, qty, avg)
    _, stored = await asyncio.to_thread(_db.get_paper_state)
    for sym in stored:
        if sym not in broker._positions:
            await asyncio.to_thread(_db.upsert_paper_position, sym, 0, 0.0)


# ── Account info ──────────────────────────────────────────────────────────────

@app.get("/api/account", response_model=AccountInfo)
async def get_account():
    """Return cash balance, open positions with live prices, and broker mode."""
    live   = _is_live()
    broker = await _get_broker()
    cash          = await broker.get_cash()
    raw_positions = await broker.get_positions()

    positions: list[AccountPosition] = []
    total_value = cash

    for sym, qty in raw_positions.items():
        avg_price = broker._avg_prices.get(sym, 0.0) if hasattr(broker, "_avg_prices") else 0.0
        cur_price = avg_price
        try:
            price_data = await asyncio.to_thread(get_price_sync, sym, days=5)
            bars = price_data.get("bars", [])
            if bars:
                cur_price = float(bars[-1]["close"])
        except Exception:
            pass

        market_value = qty * cur_price
        total_value += market_value
        pnl_pct = (cur_price / avg_price - 1.0) if avg_price > 0 else 0.0

        positions.append(AccountPosition(
            symbol        = sym,
            qty           = qty,
            avg_price     = round(avg_price, 4),
            current_price = round(cur_price, 4),
            market_value  = round(market_value, 2),
            pnl_pct       = round(pnl_pct, 6),
        ))

    return AccountInfo(
        cash         = round(cash, 2),
        total_value  = round(total_value, 2),
        positions    = positions,
        is_simulated = not live,
        broker_mode  = "live" if live else "simulator",
        initial_cash = _PAPER_INITIAL_CASH,
    )


@app.delete("/api/account/reset", status_code=200)
async def reset_account():
    """Reset the simulator to initial cash (no positions). Simulator only."""
    if _is_live():
        raise HTTPException(status_code=400, detail="Cannot reset a live broker account.")
    global _paper_broker
    async with _paper_broker_lock:
        await asyncio.to_thread(_db.reset_paper_state, _PAPER_INITIAL_CASH)
        _paper_broker = None
    _log.info("PaperBroker reset to initial_cash=%.2f", _PAPER_INITIAL_CASH)
    return {"reset": True, "initial_cash": _PAPER_INITIAL_CASH}


# ── Orders ────────────────────────────────────────────────────────────────────

@app.post("/api/orders", response_model=OrderResponse, status_code=201)
async def submit_order(req: OrderRequest):
    """
    Submit a BUY or SELL order.
    Simulator mode (default): instant fill via PaperBroker; state persists in SQLite.
    Live mode: Playwright browser automation via WebBroker.
    """
    live = _is_live()

    from myquant.models.order import Order, OrderSide, OrderType

    side  = OrderSide.BUY    if req.side       == "BUY"    else OrderSide.SELL
    otype = OrderType.MARKET if req.order_type == "MARKET" else OrderType.LIMIT
    order = Order(
        symbol      = req.symbol,
        side        = side,
        order_type  = otype,
        quantity    = req.quantity,
        limit_price = req.limit_price or 0.0,
    )

    try:
        broker    = await _get_broker()
        broker_id = await broker.submit_order(order)
        _log.info(
            "Order submitted (%s): %s %s x%d broker_id=%s",
            "live" if live else "sim",
            req.side, req.symbol, req.quantity, broker_id,
        )

        cash_after: float | None = None
        if not live:
            await _save_paper_state(broker)
            cash_after = round(broker._cash, 2)

        return OrderResponse(
            broker_order_id = broker_id,
            symbol          = req.symbol,
            side            = req.side,
            order_type      = req.order_type,
            quantity        = req.quantity,
            limit_price     = req.limit_price,
            status          = "SUBMITTED",
            is_simulated    = not live,
            cash_after      = cash_after,
        )

    except Exception as exc:
        _log.exception("Broker order failed: %s", exc)
        if live:
            global _web_broker
            _web_broker = None
        raise HTTPException(status_code=502, detail=f"Broker error: {exc}")

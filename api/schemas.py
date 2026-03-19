"""
api/schemas.py — Pydantic request/response models for the MyQuant API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Backtest ─────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    symbols: list[str] = Field(
        default=["sh600519", "sz300750"],
        description="List of SH/SZ A-share symbols, e.g. sh600519, sz300750"
    )
    lookback_days: int = Field(default=365, ge=30, le=730, description="Test window in days")
    initial_cash: float = Field(default=1_000_000.0, ge=10_000)
    commission_rate: float = Field(default=0.0003)
    stop_loss_pct: float = Field(default=-0.08, le=0)
    symbol_loss_cap: float = Field(default=-20_000.0, le=0)
    trailing_stop_pct: float = Field(default=0.0, ge=0.0, description="Trailing stop fraction (0=disabled)")
    take_profit_pct: float = Field(default=0.0, ge=0.0, description="Take-profit fraction (0=disabled)")

    @field_validator("symbols")
    @classmethod
    def symbols_sh_sz_only(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip().lower() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("symbols must contain at least one non-blank entry")
        bad = [s for s in cleaned if not (s.startswith("sh") or s.startswith("sz"))]
        if bad:
            raise ValueError(
                f"Only Shanghai (sh) and Shenzhen (sz) A-share symbols are supported. "
                f"Unsupported: {', '.join(bad)}"
            )
        return cleaned


class TradeRow(BaseModel):
    time: str
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    strategy: Optional[str] = None


class SymbolPnL(BaseModel):
    symbol: str
    fills: int
    net_pnl: float
    buys: int
    sells: int


class StrategyFills(BaseModel):
    strategy: str
    fills: int
    buys: int
    sells: int


class BacktestResponse(BaseModel):
    job_id: str
    status: str                          # "pending" | "running" | "done" | "error"
    # populated when status == "done"
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    symbols: list[str] = []
    initial_nav: Optional[float] = None
    final_nav: Optional[float] = None
    total_pnl: Optional[float] = None
    total_pnl_pct: Optional[float] = None
    sharpe: Optional[float] = None
    max_dd: Optional[float] = None
    num_trades: Optional[int] = None
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    nav_series: list[dict[str, Any]] = []   # [{date, nav}, ...]
    trades: list[TradeRow] = []
    symbol_pnl: list[SymbolPnL] = []
    strategy_fills: list[StrategyFills] = []
    error: Optional[str] = None


# ── Screener ─────────────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    top_n: int = Field(default=6, ge=1, le=20)
    min_bars: int = Field(default=200, ge=50)
    lookback_years: int = Field(default=1, ge=1, le=3)
    indices: list[str] = Field(
        default=["000300"],
        description="CSI index codes to screen: 000300=CSI300, 000905=CSI500, 000852=CSI1000",
    )

    @field_validator("indices")
    @classmethod
    def valid_indices(cls, v: list[str]) -> list[str]:
        valid = {"000300", "000905", "000852"}
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"Unknown index codes: {bad}. Valid: {sorted(valid)}")
        return v


class ScreenRow(BaseModel):
    rank: int
    symbol: str
    yf_ticker: str
    name: str
    bars: int
    ret_1y: float
    ret_6m: float
    sharpe: float
    max_dd: float
    trend_pct: float
    atr_pct: float
    autocorr: float
    score: float
    recommended: bool
    causal_nodes: list[dict[str, Any]] = []
    data_scope:   Optional[dict[str, Any]] = None
    gate_checks:  list[dict[str, Any]] = []


class ScreenResponse(BaseModel):
    job_id: str
    status: str
    top_symbols: list[str] = []
    rows: list[ScreenRow] = []
    universe_size: int = 0
    error: Optional[str] = None


# ── Price / OHLCV ─────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceResponse(BaseModel):
    symbol: str
    yf_ticker: str = ""
    bars: list[OHLCVBar] = []
    error: Optional[str] = None


# ── Fundamentals ──────────────────────────────────────────────────────────────

class FundamentalsResponse(BaseModel):
    symbol: str
    pe_ttm: float = 0.0
    pb: float = 0.0
    ps_ttm: float = 0.0
    roe: float = 0.0
    revenue_growth: float = 0.0
    net_margin: float = 0.0
    dividend_yield: float = 0.0
    value_score: float = 0.0
    growth_score: float = 0.0
    quality_score: float = 0.0
    error: Optional[str] = None


# ── News ──────────────────────────────────────────────────────────────────────

class NewsItemSchema(BaseModel):
    title: str
    content: str = ""
    source: str = ""
    ts: str
    url: str = ""


class NewsResponse(BaseModel):
    symbol: Optional[str] = None
    items: list[NewsItemSchema] = []
    error: Optional[str] = None


# ── Regime ────────────────────────────────────────────────────────────────────

class RegimeResponse(BaseModel):
    regime: str                  # RISK_ON | NEUTRAL | RISK_OFF
    signal_multiplier: float
    symbols_analyzed: int
    error: Optional[str] = None


# ── Universe ──────────────────────────────────────────────────────────────────

class UniverseSymbol(BaseModel):
    symbol: str
    yf_ticker: str
    name: str


class UniverseResponse(BaseModel):
    symbols: list[UniverseSymbol]


# ── Workflow (screen → backtest pipeline) ─────────────────────────────────────

class WorkflowRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    top_n: int = Field(default=6, ge=1, le=20)
    min_bars: int = Field(default=200, ge=50)
    lookback_years: int = Field(default=1, ge=1, le=3)
    backtest_days: int = Field(default=365, ge=30, le=730)
    initial_cash: float = Field(default=1_000_000.0, ge=10_000)
    commission_rate: float = Field(default=0.0003)
    stop_loss_pct: float = Field(default=-0.08, le=0)
    symbol_loss_cap: float = Field(default=-20_000.0, le=0)
    trailing_stop_pct: float = Field(default=0.0, ge=0.0, description="Trailing stop fraction (0=disabled)")
    take_profit_pct: float = Field(default=0.0, ge=0.0, description="Take-profit fraction (0=disabled)")
    indices: list[str] = Field(
        default=["000300"],
        description="CSI index codes to screen: 000300=CSI300, 000905=CSI500",
    )


class WorkflowResponse(BaseModel):
    job_id: str
    status: str                           # pending|screening|backtesting|done|error
    # ── Screener section ──────────────────────────────────────
    top_symbols: list[str] = []
    screen_rows: list[ScreenRow] = []
    # ── Backtest section (populated once status == "done") ────
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    initial_nav: Optional[float] = None
    final_nav: Optional[float] = None
    total_pnl: Optional[float] = None
    total_pnl_pct: Optional[float] = None
    sharpe: Optional[float] = None
    max_dd: Optional[float] = None
    num_trades: Optional[int] = None
    win_rate: Optional[float] = None
    nav_series: list[dict[str, Any]] = []
    trades: list[TradeRow] = []
    symbol_pnl: list[SymbolPnL] = []
    error: Optional[str] = None


# ── Advisor: Train ────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    symbol:        str
    force_retrain: bool = False

    @field_validator("symbol")
    @classmethod
    def symbol_sh_sz(cls, v: str) -> str:
        s = v.strip().lower()
        if not s:
            raise ValueError("symbol must not be blank")
        if not (s.startswith("sh") or s.startswith("sz")):
            raise ValueError("Only SH/SZ A-share symbols are supported (e.g. sh600519, sz300750)")
        return s


class TrainResponse(BaseModel):
    job_id:        str
    status:        str
    symbol:        str  = ""
    model_id:      Optional[str]   = None
    train_status:  Optional[str]   = None   # "trained" | "loaded" | "skipped"
    skip_reason:   Optional[str]   = None
    bar_count:     Optional[int]   = None
    last_bar_date: Optional[str]   = None
    oos_accuracy:  Optional[float] = None
    trained_at:    Optional[float] = None
    feature_cols:  list[str]       = []
    error:         Optional[str]   = None


# ── Advisor: Analyze ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    symbol:        str
    force_retrain: bool = False

    @field_validator("symbol")
    @classmethod
    def symbol_sh_sz(cls, v: str) -> str:
        s = v.strip().lower()
        if not s:
            raise ValueError("symbol must not be blank")
        if not (s.startswith("sh") or s.startswith("sz")):
            raise ValueError("Only SH/SZ A-share symbols are supported (e.g. sh600519, sz300750)")
        return s


class ModelMeta(BaseModel):
    model_id:      str
    trained_at:    float
    bar_count:     int
    last_bar_date: str
    oos_accuracy:  Optional[float] = None
    train_status:  str = "loaded"
    skip_reason:   Optional[str]   = None


class AnalysisResponse(BaseModel):
    job_id:    str
    status:    str
    symbol:    str  = ""
    sector:    str  = ""
    signal:     str   = ""
    confidence: float = 0.0
    p_buy:      float = 0.0
    p_hold:     float = 0.0
    p_sell:     float = 0.0
    momentum:    dict[str, Any] = {}
    recent_bars: list[dict[str, Any]] = []
    fundamentals: dict[str, Any] = {}
    news:         list[dict[str, Any]] = []
    feature_importance: list[dict[str, Any]] = []
    model_meta:   Optional[ModelMeta] = None
    error:        Optional[str] = None


# ── Advisor: Recommend ────────────────────────────────────────────────────────

class RecommendationRow(BaseModel):
    symbol:           str
    yf_ticker:        str = ""
    name:             str = ""
    sector:           str = ""
    score:            float = 0.0
    model_signal:     str   = "N/A"
    model_confidence: float = 0.0
    model_trained:    bool  = False
    ret_1y:           float = 0.0
    ret_3m:           float = 0.0
    ret_1m:           float = 0.0
    fundamentals:     dict[str, Any] = {}
    causal_nodes:     list[dict[str, Any]] = []
    data_scope:       Optional[dict[str, Any]] = None


class RecommendResponse(BaseModel):
    job_id: str
    status: str
    sector: Optional[str]           = None
    rows:   list[RecommendationRow] = []
    error:  Optional[str]           = None


# ── Advisor: Model list ───────────────────────────────────────────────────────

class StoredModelInfo(BaseModel):
    model_id:      str
    symbol:        str
    strategy_id:   str
    trained_at:    float
    bar_count:     int
    last_bar_date: str
    oos_accuracy:  Optional[float] = None
    feature_cols:  list[str] = []


class ModelListResponse(BaseModel):
    models: list[StoredModelInfo]


# ── Orders (WebBroker semi-auto execution) ──────────────────────────────

class OrderRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    symbol:      str
    side:        str   = Field(..., pattern="^(BUY|SELL)$")
    order_type:  str   = Field(default="MARKET", pattern="^(MARKET|LIMIT)$")
    quantity:    int   = Field(..., ge=1)
    limit_price: Optional[float] = Field(default=None, gt=0)

    @field_validator("symbol")
    @classmethod
    def symbol_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol must not be blank")
        return v.strip().lower()


class OrderResponse(BaseModel):
    broker_order_id: str
    symbol:          str
    side:            str
    order_type:      str
    quantity:        int
    limit_price:     Optional[float] = None
    status:          str   # SUBMITTED | REJECTED | error
    error:           Optional[str] = None
    is_simulated:    bool  = False
    cash_after:      Optional[float] = None   # available cash after fill (simulator only)


# ── Account (positions + cash) ────────────────────────────────────────────────

class AccountPosition(BaseModel):
    symbol:        str
    qty:           int
    avg_price:     float = 0.0
    current_price: float = 0.0
    market_value:  float = 0.0
    pnl_pct:       float = 0.0


class AccountInfo(BaseModel):
    cash:          float
    total_value:   float
    positions:     list[AccountPosition] = []
    is_simulated:  bool
    broker_mode:   str   # "simulator" | "live"
    initial_cash:  float = 500_000.0

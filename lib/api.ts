// lib/api.ts — typed API client for MyQuant backend

const BASE = "/api";

// ── Error parsing ────────────────────────────────────────────────────────────
// Extracts a human-readable message from FastAPI error bodies ({detail: "..."}).
async function _throwApiError(res: Response): Promise<never> {
  let message = `HTTP ${res.status}`;
  try {
    const body = await res.json() as Record<string, unknown>;
    if (typeof body.detail === "string")       message = body.detail;
    else if (typeof body.message === "string") message = body.message;
    else                                       message = JSON.stringify(body);
  } catch {
    try { message = (await res.text()) || message; } catch { /* ignore */ }
  }
  throw new Error(message);
}

// ── Fetch wrapper ─────────────────────────────────────────────────────────────
// • Throws a human-readable error on non-2xx responses.
// • Aborts the request after timeoutMs (default 30 s) to prevent hung UI.
async function apiFetch(
  url: string,
  init?: RequestInit,
  timeoutMs = 30_000,
): Promise<Response> {
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) await _throwApiError(res);
    return res;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
    }
    throw err;
  } finally {
    clearTimeout(tid);
  }
}

export type JobStatus = "pending" | "running" | "done" | "error";

export interface TradeRow {
  time: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  commission: number;
  strategy: string;
}

export interface SymbolPnL {
  symbol: string;
  fills: number;
  net_pnl: number;
  buys: number;
  sells: number;
}

export interface StrategyFills {
  strategy: string;
  fills: number;
  buys: number;
  sells: number;
}

export interface NavPoint {
  date: string;
  nav: number;
}

export interface BacktestResult {
  job_id: string;
  status: JobStatus;
  period_start?: string;
  period_end?: string;
  symbols: string[];
  initial_nav?: number;
  final_nav?: number;
  total_pnl?: number;
  total_pnl_pct?: number;
  sharpe?: number;
  max_dd?: number;
  num_trades?: number;
  win_rate?: number;
  avg_win?: number;
  avg_loss?: number;
  profit_factor?: number;
  nav_series: NavPoint[];
  trades: TradeRow[];
  symbol_pnl: SymbolPnL[];
  strategy_fills: StrategyFills[];
  error?: string;
}

export interface BacktestRequest {
  symbols: string[];
  lookback_days?: number;
  initial_cash?: number;
  commission_rate?: number;
  stop_loss_pct?: number;
  symbol_loss_cap?: number;
}

// ── Causal Trace types ──────────────────────────────────────────────────────
export interface CausalNode {
  factor: string;
  label: string;
  description: string;
  raw_value: number | string;
  norm_value: number;
  weight: number;
  contribution: number;
  direction: "positive" | "negative" | "neutral";
  percentile: string;
  extras?: Record<string, unknown>;
}

export interface DataScope {
  start_date: string;
  end_date: string;
  bars: number;
  price_start: number;
  price_end: number;
  price_min: number;
  price_max: number;
  trend: "UPTREND" | "DOWNTREND" | "SIDEWAYS";
}

export interface GateCheck {
  check: string;
  label: string;
  threshold: number;
  actual: number;
  passed: boolean;
  note: string;
}

export interface ScreenRow {
  rank: number;
  symbol: string;
  yf_ticker: string;
  name: string;
  bars: number;
  ret_1y: number;
  ret_6m: number;
  sharpe: number;
  max_dd: number;
  trend_pct: number;
  atr_pct: number;
  autocorr: number;
  score: number;
  recommended: boolean;
  causal_nodes: CausalNode[];
  data_scope?: DataScope;
  gate_checks: GateCheck[];
}

export interface ScreenResult {
  job_id: string;
  status: JobStatus;
  top_symbols: string[];
  rows: ScreenRow[];
  universe_size?: number;
  error?: string;
}

export interface ScreenRequest {
  top_n?: number;
  min_bars?: number;
  lookback_years?: number;
  indices?: string[];  // CSI index codes, e.g. ["000300"] or ["000300","000905"]
}

// ── API calls ──────────────────────────────────────────────────────────────

export async function startBacktest(req: BacktestRequest): Promise<BacktestResult> {
  const res = await apiFetch(`${BASE}/backtest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function getBacktest(jobId: string): Promise<BacktestResult> {
  const res = await apiFetch(`${BASE}/backtest/${jobId}`);
  return res.json();
}

export async function startScreen(req: ScreenRequest): Promise<ScreenResult> {
  const res = await apiFetch(`${BASE}/screen`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function getScreen(jobId: string): Promise<ScreenResult> {
  const res = await apiFetch(`${BASE}/screen/${jobId}`);
  return res.json();
}

/**
 * Poll a job every `intervalMs` ms until done or error, then resolve.
 *
 * Uses recursive setTimeout (not setInterval) so a slow response never
 * causes overlapping in-flight requests.
 */
export async function pollJob<T extends { status: JobStatus | string; error?: string }>(
  jobId: string,
  fetcher: (id: string) => Promise<T>,
  onProgress?: (status: string) => void,
  intervalMs = 1500,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const data = await fetcher(jobId);
        onProgress?.(data.status);
        if (data.status === "done") {
          resolve(data);
        } else if (data.status === "error") {
          reject(new Error(data.error ?? "Job failed"));
        } else {
          setTimeout(tick, intervalMs);
        }
      } catch (e) {
        reject(e);
      }
    };
    setTimeout(tick, intervalMs);
  });
}

// ── Price / OHLCV ─────────────────────────────────────────────────────────────

export interface OHLCVBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PriceResponse {
  symbol: string;
  yf_ticker: string;
  bars: OHLCVBar[];
  error?: string;
}

export async function getPrice(symbol: string, days = 365): Promise<PriceResponse> {
  const res = await apiFetch(`${BASE}/price/${encodeURIComponent(symbol)}?days=${days}`);
  return res.json();
}

// ── Fundamentals ─────────────────────────────────────────────────────────────

export interface FundamentalsResponse {
  symbol: string;
  pe_ttm: number;
  pb: number;
  ps_ttm: number;
  roe: number;
  revenue_growth: number;
  net_margin: number;
  dividend_yield: number;
  value_score: number;
  growth_score: number;
  quality_score: number;
  error?: string;
}

export async function getFundamentals(symbol: string): Promise<FundamentalsResponse> {
  const res = await apiFetch(`${BASE}/fundamentals/${encodeURIComponent(symbol)}`);
  return res.json();
}

// ── News ──────────────────────────────────────────────────────────────────────

export interface NewsItem {
  title: string;
  content: string;
  source: string;
  ts: string;
  url: string;
}

export interface NewsResponse {
  symbol?: string;
  items: NewsItem[];
  error?: string;
}

export async function getStockNews(symbol: string, limit = 20): Promise<NewsResponse> {
  const res = await apiFetch(`${BASE}/news/${encodeURIComponent(symbol)}?limit=${limit}`);
  return res.json();
}

export async function getMacroNews(limit = 30): Promise<NewsResponse> {
  const res = await apiFetch(`${BASE}/news/macro?limit=${limit}`);
  return res.json();
}

// ── Regime ────────────────────────────────────────────────────────────────────

export interface RegimeResponse {
  regime: string;            // RISK_ON | NEUTRAL | RISK_OFF
  signal_multiplier: number;
  symbols_analyzed: number;
  error?: string;
}

export async function getRegime(symbols: string[]): Promise<RegimeResponse> {
  const q = symbols.join(",");
  const res = await apiFetch(`${BASE}/regime?symbols=${encodeURIComponent(q)}`);
  return res.json();
}

// ── Universe ──────────────────────────────────────────────────────────────────

export interface UniverseSymbol {
  symbol: string;
  yf_ticker: string;
  name: string;
}

export interface UniverseResponse {
  symbols: UniverseSymbol[];
}

export async function getUniverse(): Promise<UniverseResponse> {
  const res = await apiFetch(`${BASE}/universe`);
  return res.json();
}

// ── Workflow ──────────────────────────────────────────────────────────────────

export interface WorkflowRequest {
  top_n?: number;
  min_bars?: number;
  lookback_years?: number;
  backtest_days?: number;
  initial_cash?: number;
  commission_rate?: number;
  stop_loss_pct?: number;
  symbol_loss_cap?: number;
  indices?: string[];  // CSI index codes for the screener phase, e.g. ["000300","000905"]
}

export interface WorkflowResult {
  job_id: string;
  status: string;
  // screener
  top_symbols: string[];
  screen_rows: ScreenRow[];
  // backtest (once done)
  period_start?: string;
  period_end?: string;
  initial_nav?: number;
  final_nav?: number;
  total_pnl?: number;
  total_pnl_pct?: number;
  sharpe?: number;
  max_dd?: number;
  num_trades?: number;
  win_rate?: number;
  nav_series: NavPoint[];
  trades: TradeRow[];
  symbol_pnl: SymbolPnL[];
  error?: string;
}

export async function startWorkflow(req: WorkflowRequest): Promise<WorkflowResult> {
  const res = await apiFetch(`${BASE}/workflow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function getWorkflow(jobId: string): Promise<WorkflowResult> {
  const res = await apiFetch(`${BASE}/workflow/${jobId}`);
  return res.json();
}

// ── Advisor ───────────────────────────────────────────────────────────────────

export type TrainStatus = "trained" | "loaded" | "skipped";

export interface TrainResult {
  job_id: string;
  status: JobStatus;
  symbol: string;
  model_id?: string;
  train_status?: TrainStatus;
  skip_reason?: string;
  bar_count?: number;
  last_bar_date?: string;
  oos_accuracy?: number;
  trained_at?: number;
  feature_cols: string[];
  error?: string;
}

export interface ModelMeta {
  model_id: string;
  trained_at: number;
  bar_count: number;
  last_bar_date: string;
  oos_accuracy?: number;
  train_status: string;
  skip_reason?: string;
}

export interface AnalysisResult {
  job_id: string;
  status: JobStatus;
  symbol: string;
  sector: string;
  signal: string;
  confidence: number;
  p_buy: number;
  p_hold: number;
  p_sell: number;
  momentum: Record<string, number>;
  recent_bars: Array<{ date: string; open: number; high: number; low: number; close: number; volume: number }>;
  fundamentals: Record<string, number>;
  news: Array<{ title: string; content: string; source: string; ts: string; url: string }>;
  feature_importance: Array<{ feature: string; importance: number }>;
  model_meta?: ModelMeta;
  error?: string;
}

export interface RecommendationRow {
  symbol: string;
  yf_ticker: string;
  name: string;
  sector: string;
  score: number;
  model_signal: string;
  model_confidence: number;
  model_trained: boolean;
  ret_1y: number;
  ret_3m: number;
  ret_1m: number;
  fundamentals: Record<string, number>;
  causal_nodes: CausalNode[];
  data_scope?: DataScope;
}

export interface RecommendResult {
  job_id: string;
  status: JobStatus;
  sector?: string;
  rows: RecommendationRow[];
  error?: string;
}

export interface StoredModelInfo {
  model_id: string;
  symbol: string;
  strategy_id: string;
  trained_at: number;
  bar_count: number;
  last_bar_date: string;
  oos_accuracy?: number;
  feature_cols: string[];
}

// Train a model for a symbol
export async function startTrain(symbol: string, forceRetrain = false): Promise<TrainResult> {
  const res = await apiFetch(`${BASE}/advisor/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, force_retrain: forceRetrain }),
  });
  return res.json();
}

export async function getTrain(jobId: string): Promise<TrainResult> {
  const res = await apiFetch(`${BASE}/advisor/train/${jobId}`);
  return res.json();
}

// Analyze a stock (train + signal + fundamentals + news)
export async function startAnalyze(symbol: string, forceRetrain = false): Promise<AnalysisResult> {
  const res = await apiFetch(`${BASE}/advisor/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, force_retrain: forceRetrain }),
  });
  return res.json();
}

export async function getAnalyze(jobId: string): Promise<AnalysisResult> {
  const res = await apiFetch(`${BASE}/advisor/analyze/${jobId}`);
  return res.json();
}

// Get recommendations
export async function startRecommend(sector?: string, topN = 10): Promise<RecommendResult> {
  const url = sector
    ? `${BASE}/advisor/recommend/${encodeURIComponent(sector)}?top_n=${topN}`
    : `${BASE}/advisor/recommend?top_n=${topN}`;
  const res = await apiFetch(url);
  return res.json();
}

export async function pollRecommend(jobId: string): Promise<RecommendResult> {
  const res = await apiFetch(`${BASE}/advisor/recommend-poll/${jobId}`);
  return res.json();
}

// List stored models
export async function listModels(): Promise<{ models: StoredModelInfo[] }> {
  const res = await apiFetch(`${BASE}/advisor/models`);
  return res.json();
}

export async function deleteModel(symbol: string): Promise<{ deleted: boolean; symbol: string }> {
  const res = await apiFetch(`${BASE}/advisor/models/${encodeURIComponent(symbol)}`, {
    method: "DELETE",
  });
  return res.json();
}

// List sectors
export async function getSectors(): Promise<{ sectors: string[] }> {
  const res = await apiFetch(`${BASE}/advisor/sectors`);
  return res.json();
}

export interface OrderRequest {
  symbol:       string;
  side:         "BUY" | "SELL";
  order_type:   "MARKET" | "LIMIT";
  quantity:     number;
  limit_price?: number;
}

export interface OrderResponse {
  broker_order_id: string;
  symbol:          string;
  side:            string;
  order_type:      string;
  quantity:        number;
  limit_price?:    number;
  status:          string;
  error?:          string;
  is_simulated:    boolean;
  cash_after?:     number;
}

export async function submitOrder(req: OrderRequest): Promise<OrderResponse> {
  const res = await apiFetch(`${BASE}/orders`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(req),
  }, 60_000);   // 60 s — browser automation can be slow
  return res.json();
}

// ── Account ───────────────────────────────────────────────────────────────────

export interface AccountPosition {
  symbol:        string;
  qty:           number;
  avg_price:     number;
  current_price: number;
  market_value:  number;
  pnl_pct:       number;
}

export interface AccountInfo {
  cash:          number;
  total_value:   number;
  positions:     AccountPosition[];
  is_simulated:  boolean;
  broker_mode:   string;   // "simulator" | "live"
  initial_cash:  number;
}

export async function getAccount(): Promise<AccountInfo> {
  const res = await apiFetch(`${BASE}/account`);
  return res.json();
}

export async function resetSimulator(): Promise<{ reset: boolean; initial_cash: number }> {
  const res = await apiFetch(`${BASE}/account/reset`, { method: "DELETE" });
  return res.json();
}

"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import {
  BrainCircuit,
  TrendingUp,
  LayoutGrid,
  Cpu,
  Loader2,
  RefreshCw,
  Search,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Trash2,
  ShoppingCart,
  BookOpen,
  GitBranch,
} from "lucide-react";
import { CausalTracePanel } from "./CausalTracePanel";
import {
  startAnalyze,
  getAnalyze,
  startTrain,
  getTrain,
  startRecommend,
  pollRecommend,
  listModels,
  deleteModel,
  getSectors,
  submitOrder,
  pollJob,
  type AnalysisResult,
  type TrainResult,
  type RecommendResult,
  type RecommendationRow,
  type StoredModelInfo,
  type OrderRequest,
  type OrderResponse,
} from "@/lib/api";
import { useAccountCtx } from "@/lib/account-context";
import { useNav } from "@/lib/nav-context";

// ── sub-tab type ──────────────────────────────────────────────────────────────
type AdvisorTab = "analyze" | "recommend" | "models";

const SUBTABS: { id: AdvisorTab; label: string; icon: React.ReactNode }[] = [
  { id: "analyze",   label: "Stock Analyzer",     icon: <BrainCircuit className="w-4 h-4" /> },
  { id: "recommend", label: "Recommendations",    icon: <TrendingUp    className="w-4 h-4" /> },
  { id: "models",    label: "Stored Models",      icon: <LayoutGrid    className="w-4 h-4" /> },
];

// ── helpers ───────────────────────────────────────────────────────────────────

function pct(v: number | undefined) {
  if (v === undefined || v === null) return "—";
  return `${(v * 100).toFixed(2)}%`;
}

function num(v: number | undefined, dp = 2) {
  if (v === undefined || v === null) return "—";
  return v.toFixed(dp);
}

function ts(epoch: number | undefined) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleString();
}

function SignalBadge({ signal }: { signal: string }) {
  const cfg: Record<string, string> = {
    BUY:  "bg-emerald-900 text-emerald-300 border border-emerald-700",
    SELL: "bg-red-900 text-red-300 border border-red-700",
    HOLD: "bg-gray-800 text-gray-300 border border-gray-600",
    "":   "bg-gray-800 text-gray-500 border border-gray-700",
  };
  return (
    <span className={clsx("px-2.5 py-0.5 rounded-full text-xs font-bold tracking-widest uppercase", cfg[signal] ?? cfg[""])}>
      {signal || "—"}
    </span>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-10 text-xs text-gray-400 text-right">{label}</span>
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className={clsx("h-full rounded-full", color)} style={{ width: `${(value * 100).toFixed(1)}%` }} />
      </div>
      <span className="w-12 text-xs text-gray-300 text-right">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

// ── Score bar (0–1) ───────────────────────────────────────────────────────────
function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(Math.max(score, 0), 1) * 100;
  const color =
    pct >= 65 ? "bg-emerald-500" : pct >= 45 ? "bg-sky-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={clsx("h-full rounded-full", color)} style={{ width: `${pct.toFixed(1)}%` }} />
      </div>
      <span className="text-xs text-gray-300">{pct.toFixed(0)}</span>
    </div>
  );
}

// ── Status indicator ──────────────────────────────────────────────────────────
function StatusDot({ status }: { status: string }) {
  if (status === "done") return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
  if (status === "error") return <XCircle className="w-4 h-4 text-red-400" />;
  if (status === "running" || status === "pending")
    return <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />;
  return <AlertCircle className="w-4 h-4 text-gray-500" />;
}

// ── Research jump button (reusable) ──────────────────────────────────────────
function ResearchButton({ symbol }: { symbol: string }) {
  const { jumpTo } = useNav();
  return (
    <button
      onClick={() => jumpTo("research", symbol)}
      className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-900/60 hover:bg-indigo-800 border border-indigo-800/50 text-indigo-300 hover:text-indigo-200 rounded-lg text-xs font-medium transition-colors whitespace-nowrap"
      title={`Deep-dive research on ${symbol}`}
    >
      <BookOpen className="w-3 h-3" />
      Research
    </button>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  STOCK ANALYZER SUB-PANEL
// ═══════════════════════════════════════════════════════════════════════════════

// Popular A-share picks for quick access
const ANALYZE_PICKS = [
  { symbol: "sz300059", name: "东方财富" },
  { symbol: "sz300750", name: "宁德时代" },
  { symbol: "sz000858", name: "五粮液" },
  { symbol: "sz000333", name: "美的" },
  { symbol: "sh601318", name: "平安" },
  { symbol: "sz002594", name: "BYD" },
];

function AnalyzeSubPanel() {
  const { jumpSymbol, clearJumpSymbol, activeTab } = useNav();

  const [symbol, setSymbol]             = useState("");
  const [forceRetrain, setForceRetrain] = useState(false);
  const [loading, setLoading]           = useState(false);
  const [status, setStatus]             = useState("");
  const [result, setResult]             = useState<AnalysisResult | null>(null);
  const [error, setError]               = useState("");
  const [showImp, setShowImp]           = useState(false);

  // Pre-fill symbol when jumped from another panel — only consume when
  // this tab (advisor) is active to avoid bleeding into ResearchPanel.
  useEffect(() => {
    if (jumpSymbol && activeTab === "advisor") {
      setSymbol(jumpSymbol);
      clearJumpSymbol();
    }
  }, [jumpSymbol, activeTab, clearJumpSymbol]);

  async function handleAnalyze(overrideSym?: string) {
    const sym = (overrideSym ?? symbol).trim().toLowerCase();
    if (!sym) return;
    // Sync input field when called from a chip
    if (overrideSym) setSymbol(overrideSym);
    setLoading(true);
    setError("");
    setResult(null);
    setStatus("Starting analysis…");
    try {
      const init = await startAnalyze(sym, forceRetrain);
      const data = await pollJob(init.job_id, getAnalyze, (s) =>
        setStatus(
          s === "running"
            ? "Training model & computing signals…"
            : s === "pending"
            ? "Queued…"
            : s
        )
      );
      setResult(data);
      setStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const fund = result?.fundamentals ?? {};
  const mom  = result?.momentum    ?? {};

  return (
    <div className="space-y-6">
      {/* Search bar */}
      <div className="space-y-3">
        <div className="flex gap-2 items-end flex-wrap">
          <div className="flex-1 min-w-48">
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Symbol</label>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !loading && handleAnalyze()}
              placeholder="e.g. sz300059, sz300750, sz000858"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            />
          </div>
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer select-none pb-2">
            <input
              type="checkbox"
              checked={forceRetrain}
              onChange={(e) => setForceRetrain(e.target.checked)}
              className="accent-sky-500"
            />
            Force retrain
          </label>
          <button
            onClick={() => handleAnalyze()}
            disabled={loading || !symbol.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>

        {/* Quick chips */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[11px] text-gray-600 uppercase tracking-wider">Quick picks:</span>
          {ANALYZE_PICKS.map((q) => (
            <button
              key={q.symbol}
              onClick={() => handleAnalyze(q.symbol)}
              className="inline-flex items-center gap-1 text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-sky-700 text-gray-300 hover:text-sky-300 px-2.5 py-1 rounded-full transition-colors"
            >
              <span className="font-medium">{q.name}</span>
              <span className="text-gray-500 font-mono text-[10px]">{q.symbol}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Progress */}
      {loading && (
        <div className="text-sm text-sky-400 flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin" /> {status}
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-300">{error}</div>
      )}

      {result && result.status === "done" && (
        <div className="space-y-4">
          {/* Contextual next-step: deep dive into this stock */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">
              Analyzed <span className="font-mono text-sky-400">{result.symbol.toUpperCase()}</span>
            </span>
            <ResearchButton symbol={result.symbol} />
          </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* ── Signal card ── */}
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-300">ML Signal</h3>
              <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">
                {result.sector}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <SignalBadge signal={result.signal} />
              <span className="text-lg font-bold text-white">
                {((result.confidence ?? 0) * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-gray-500">confidence</span>
            </div>
            <div className="space-y-2">
              <ProbBar label="BUY"  value={result.p_buy}  color="bg-emerald-500" />
              <ProbBar label="HOLD" value={result.p_hold} color="bg-sky-500" />
              <ProbBar label="SELL" value={result.p_sell} color="bg-red-500" />
            </div>
            {result.model_meta && (
              <div className="mt-2 pt-3 border-t border-gray-800 text-xs text-gray-500 space-y-1">
                <div className="flex justify-between">
                  <span>OOS accuracy</span>
                  <span className="text-gray-300">
                    {result.model_meta.oos_accuracy !== undefined
                      ? `${(result.model_meta.oos_accuracy * 100).toFixed(1)}%`
                      : "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Training bars</span>
                  <span className="text-gray-300">{result.model_meta.bar_count}</span>
                </div>
                <div className="flex justify-between">
                  <span>Last bar</span>
                  <span className="text-gray-300">{result.model_meta.last_bar_date}</span>
                </div>
                <div className="flex justify-between">
                  <span>Status</span>
                  <span className={clsx(
                    "font-medium",
                    result.model_meta.train_status === "trained" ? "text-emerald-400"
                    : result.model_meta.train_status === "skipped" ? "text-yellow-400"
                    : "text-sky-400"
                  )}>
                    {result.model_meta.train_status}
                    {result.model_meta.skip_reason ? ` (${result.model_meta.skip_reason})` : ""}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Trained at</span>
                  <span className="text-gray-300">{ts(result.model_meta.trained_at)}</span>
                </div>
              </div>
            )}
          </div>

          {/* ── Momentum + Fundamentals ── */}
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800 space-y-4">
            <h3 className="text-sm font-semibold text-gray-300">Momentum</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {(["ret_1d","ret_5d","ret_1m","ret_3m","ret_1y"] as const).map((k) => (
                (mom as Record<string, number>)[k] !== undefined && (
                  <div key={k} className="flex justify-between items-center bg-gray-800 rounded-lg px-3 py-2">
                    <span className="text-gray-400">{k.replace("ret_","")}</span>
                    <span className={clsx(
                      "font-medium",
                      (mom as Record<string, number>)[k] >= 0 ? "text-emerald-400" : "text-red-400"
                    )}>
                      {pct((mom as Record<string, number>)[k])}
                    </span>
                  </div>
                )
              ))}
              {mom.current_price !== undefined && (
                <div className="col-span-2 flex justify-between items-center bg-gray-800 rounded-lg px-3 py-2">
                  <span className="text-gray-400">Current price</span>
                  <span className="font-bold text-white">{num(mom.current_price, 4)}</span>
                </div>
              )}
            </div>

            <h3 className="text-sm font-semibold text-gray-300 pt-2">Fundamentals</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                { k: "pe_ttm",         label: "P/E" },
                { k: "pb",             label: "P/B" },
                { k: "roe",            label: "ROE %" },
                { k: "revenue_growth", label: "Rev. growth" },
                { k: "net_margin",     label: "Net margin" },
                { k: "dividend_yield", label: "Yield" },
              ].map(({ k, label }) => (
                <div key={k} className="flex justify-between items-center bg-gray-800 rounded-lg px-3 py-2">
                  <span className="text-gray-400">{label}</span>
                  <span className="text-gray-200">{num(fund[k])}</span>
                </div>
              ))}
            </div>
            <div className="flex gap-2 mt-2">
              {(["value_score","growth_score","quality_score"] as const).map((k) => (
                <div key={k} className="flex-1 bg-gray-800 rounded-lg p-2 text-center">
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">{k.replace("_score","")}</div>
                  <div className="text-base font-bold text-sky-400">{num(fund[k], 0)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* ── News ── */}
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800 space-y-3">
            <h3 className="text-sm font-semibold text-gray-300">Recent News</h3>
            {result.news.length === 0 ? (
              <p className="text-xs text-gray-500">No news available.</p>
            ) : (
              result.news.map((n, i) => (
                <div key={i} className="border-b border-gray-800 pb-2 last:border-0">
                  <p className="text-xs font-medium text-gray-200 leading-snug">
                    {n.url ? (
                      <a href={n.url} target="_blank" rel="noreferrer" className="hover:text-sky-400">
                        {n.title}
                      </a>
                    ) : n.title}
                  </p>
                  <p className="text-[11px] text-gray-500 mt-0.5">
                    {n.source} · {n.ts.slice(0, 10)}
                  </p>
                </div>
              ))
            )}
          </div>

          {/* ── Feature importance (collapsible) ── */}
          {result.feature_importance.length > 0 && (
            <div className="lg:col-span-3 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
              <button
                onClick={() => setShowImp((v) => !v)}
                className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-gray-300 hover:bg-gray-800 transition-colors"
              >
                <span className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-sky-400" />
                  Feature Importance (Top 15)
                </span>
                {showImp ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              {showImp && (
                <div className="px-5 pb-5 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                  {result.feature_importance.map(({ feature, importance }) => {
                    const max = result.feature_importance[0].importance;
                    return (
                      <div key={feature} className="bg-gray-800 rounded-lg p-2">
                        <div className="text-[10px] text-gray-400 truncate">{feature}</div>
                        <div className="mt-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-sky-500 rounded-full"
                            style={{ width: `${(importance / max) * 100}%` }}
                          />
                        </div>
                        <div className="text-[10px] text-gray-500 mt-0.5 text-right">{importance.toFixed(0)}</div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RECOMMENDATIONS SUB-PANEL
// ═══════════════════════════════════════════════════════════════════════════════

function RecommendSubPanel() {
  const [sectors, setSectors]   = useState<string[]>([]);
  const [sector, setSector]     = useState("all");
  const [topN, setTopN]         = useState(10);
  const [loading, setLoading]   = useState(false);
  const [status, setStatus]     = useState("");
  const [result, setResult]     = useState<RecommendResult | null>(null);
  const [error, setError]       = useState("");
  const [trainTarget, setTrainTarget] = useState<string | null>(null);
  const [trainStatus, setTrainStatus] = useState<Record<string, string>>({});

  // Load sectors on mount
  useEffect(() => {
    getSectors()
      .then((r) => setSectors(r.sectors))
      .catch(() => {});
  }, []);

  async function handleRecommend() {
    setLoading(true);
    setError("");
    setResult(null);
    setStatus("Scoring universe…");
    try {
      const init = await startRecommend(sector === "all" ? undefined : sector, topN);
      const data = await pollJob(init.job_id, pollRecommend, (s) =>
        setStatus(
          s === "running" ? "Scoring & ranking stocks…"
          : s === "pending" ? "Queued…"
          : s
        )
      );
      setResult(data);
      setStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleTrain(sym: string) {
    setTrainTarget(sym);
    setTrainStatus((prev) => ({ ...prev, [sym]: "training" }));
    try {
      const init = await startTrain(sym);
      const data = await pollJob(init.job_id, getTrain);
      setTrainStatus((prev) => ({
        ...prev,
        [sym]: data.train_status === "trained"
          ? `Trained · OOS ${((data.oos_accuracy ?? 0) * 100).toFixed(1)}%`
          : data.train_status === "skipped"
          ? `Up to date (${data.skip_reason})`
          : "loaded",
      }));
    } catch (e: unknown) {
      setTrainStatus((prev) => ({
        ...prev,
        [sym]: "error: " + (e instanceof Error ? e.message : String(e)),
      }));
    } finally {
      setTrainTarget(null);
    }
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex gap-2 items-end flex-wrap">
        <div>
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Sector</label>
          <select
            value={sector}
            onChange={(e) => setSector(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
          >
            <option value="all">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Top N</label>
          <select
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
          >
            {[5, 10, 15, 20].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
        <button
          onClick={handleRecommend}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
          {loading ? "Scoring…" : "Get Recommendations"}
        </button>
      </div>

      {loading && (
        <div className="text-sm text-sky-400 flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin" /> {status}
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-300">{error}</div>
      )}

      {result && result.status === "done" && result.rows.length === 0 && (
        <p className="text-sm text-gray-500">No stocks found for this sector.</p>
      )}

      {result && result.status === "done" && result.rows.length > 0 && (
        <div className="space-y-3">
          {result.rows.map((row: RecommendationRow, i: number) => (
            <RecommendCard
              key={row.symbol}
              rank={i + 1}
              row={row}
              trainStatus={trainStatus[row.symbol]}
              isTraining={trainTarget === row.symbol}
              onTrain={() => handleTrain(row.symbol)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function RecommendCard({
  rank,
  row,
  trainStatus,
  isTraining,
  onTrain,
}: {
  rank: number;
  row: RecommendationRow;
  trainStatus?: string;
  isTraining: boolean;
  onTrain: () => void;
}) {
  const { account, refresh: refreshAccount } = useAccountCtx();

  const [expanded,      setExpanded]      = useState(false);
  const [showOrderForm, setShowOrderForm] = useState(false);
  const [showConfirm,   setShowConfirm]   = useState(false);
  const [orderSide,     setOrderSide]     = useState<"BUY" | "SELL">("BUY");
  const [orderType,     setOrderType]     = useState<"MARKET" | "LIMIT">("MARKET");
  const [orderQty,      setOrderQty]      = useState(100);
  const [limitPrice,    setLimitPrice]    = useState("");
  const [submitting,    setSubmitting]    = useState(false);
  const [orderResult,   setOrderResult]   = useState<OrderResponse | null>(null);
  const [orderError,    setOrderError]    = useState("");

  const fund        = row.fundamentals;
  const existingPos = account?.positions.find((p) => p.symbol === row.symbol);

  // Smart-prefill when the order form opens
  useEffect(() => {
    if (showOrderForm && !orderResult) {
      if (existingPos) {
        setOrderSide("SELL");
        setOrderQty(existingPos.qty);
      } else {
        setOrderSide("BUY");
        setOrderQty(100);
      }
      setShowConfirm(false);
      setOrderError("");
    }
  }, [showOrderForm]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleSubmitOrder() {
    setSubmitting(true);
    setOrderError("");
    setOrderResult(null);
    const req: OrderRequest = {
      symbol:     row.symbol,
      side:       orderSide,
      order_type: orderType,
      quantity:   orderQty,
      ...(orderType === "LIMIT" && limitPrice ? { limit_price: parseFloat(limitPrice) } : {}),
    };
    try {
      const res = await submitOrder(req);
      setOrderResult(res);
      setShowConfirm(false);
      refreshAccount();
    } catch (e: unknown) {
      setOrderError(e instanceof Error ? e.message : String(e));
      setShowConfirm(false);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3">
        {/* rank */}
        <span className="w-6 text-center text-xs font-bold text-gray-500">#{rank}</span>

        {/* symbol + name */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono text-sm font-bold text-sky-400">{row.symbol}</span>
            <span className="text-xs text-gray-400 truncate">{row.name}</span>
            <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">
              {row.sector}
            </span>
          </div>
        </div>

        {/* score bar */}
        <div className="hidden sm:flex flex-col items-end gap-0.5">
          <span className="text-[10px] text-gray-500">Score</span>
          <ScoreBar score={row.score} />
        </div>

        {/* momentum */}
        <div className="hidden md:flex gap-3 text-xs">
          <div className="text-center">
            <div className="text-gray-500">1M</div>
            <div className={row.ret_1m >= 0 ? "text-emerald-400" : "text-red-400"}>
              {pct(row.ret_1m)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-500">1Y</div>
            <div className={row.ret_1y >= 0 ? "text-emerald-400" : "text-red-400"}>
              {pct(row.ret_1y)}
            </div>
          </div>
        </div>

        {/* signal */}
        <SignalBadge signal={row.model_signal === "N/A" ? "" : row.model_signal} />

        {/* train button */}
        <button
          onClick={onTrain}
          disabled={isTraining}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-700 hover:bg-indigo-600 disabled:bg-gray-700 rounded-lg text-xs font-medium transition-colors whitespace-nowrap"
        >
          {isTraining ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <Cpu className="w-3 h-3" />
          )}
          {isTraining ? "Training…" : row.model_trained ? "Retrain" : "Train"}
        </button>

        {/* trade button */}
        <button
          onClick={() => { setShowOrderForm((v) => !v); setOrderResult(null); setOrderError(""); setShowConfirm(false); }}
          className={clsx(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap",
            showOrderForm
              ? "bg-emerald-800 text-emerald-200"
              : "bg-emerald-700 hover:bg-emerald-600 text-white"
          )}
        >
          <ShoppingCart className="w-3 h-3" />
          {existingPos ? `Trade (hold ${existingPos.qty})` : "Trade"}
        </button>

        {/* research button */}
        <ResearchButton symbol={row.symbol} />

        {/* expand */}
        <button onClick={() => setExpanded((v) => !v)} className="text-gray-500 hover:text-gray-300">
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {/* inline order form */}
      {showOrderForm && (
        <div className="border-t border-gray-700 bg-gray-850 px-4 py-4 space-y-3">
          {/* compact causal context */}
          {row.causal_nodes && row.causal_nodes.length > 0 && (() => {
            const top = [...row.causal_nodes].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))[0];
            const ds  = row.data_scope;
            return (
              <div className="flex flex-wrap items-center gap-2 px-3 py-2 rounded-lg bg-violet-950/40 border border-violet-800/40 text-[11px] text-violet-300">
                <GitBranch className="w-3.5 h-3.5 text-violet-400 shrink-0" />
                <span className="font-semibold text-violet-200">Score {row.score.toFixed(3)}</span>
                {top && (
                  <span>· Top factor: <span className="text-white font-medium">{top.label}</span> <span className="text-violet-400">{top.percentile}</span></span>
                )}
                {ds && (
                  <span className="text-violet-400 ml-auto">{ds.start_date} → {ds.end_date} ({ds.bars} bars)</span>
                )}
              </div>
            );
          })()}
          {orderResult ? (
            /* ── success ── */
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-emerald-300 bg-emerald-950 border border-emerald-800 rounded-lg p-3">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <div className="flex-1">
                  <span className="font-semibold">
                    {orderResult.is_simulated ? "Simulated order filled" : "Order submitted"}
                  </span>
                  <span className="text-emerald-400 ml-2 font-mono text-xs">{orderResult.broker_order_id}</span>
                  <span className="text-gray-400 ml-2 text-xs">
                    {orderResult.side} {orderResult.quantity}×{orderResult.symbol}
                    {orderResult.limit_price ? ` @ ¥${orderResult.limit_price}` : " (market)"}
                  </span>
                </div>
              </div>
              {orderResult.cash_after !== undefined && (
                <p className="text-xs text-gray-400">
                  Cash remaining:{" "}
                  <span className="font-mono text-gray-200">
                    ¥{orderResult.cash_after.toLocaleString("zh-CN", { minimumFractionDigits: 2 })}
                  </span>
                </p>
              )}
              <button
                onClick={() => { setOrderResult(null); setShowConfirm(false); }}
                className="text-xs text-sky-400 hover:text-sky-300"
              >
                Place another order
              </button>
            </div>
          ) : showConfirm ? (
            /* ── confirmation ── */
            <div className="space-y-3">
              <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">Confirm order</p>
              <div className="bg-gray-800 rounded-lg px-4 py-3 space-y-1.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Action</span>
                  <span className={clsx("font-bold", orderSide === "BUY" ? "text-emerald-400" : "text-red-400")}>
                    {orderSide}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Symbol</span>
                  <span className="font-mono text-sky-400">{row.symbol}{row.name ? ` · ${row.name}` : ""}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Quantity</span>
                  <span className="text-gray-100">{orderQty.toLocaleString()} shares</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Order type</span>
                  <span className="text-gray-100">
                    {orderType}{orderType === "LIMIT" && limitPrice ? ` @ ¥${limitPrice}` : ""}
                  </span>
                </div>
                {account && (
                  <div className="flex justify-between border-t border-gray-700 pt-1.5 mt-0.5">
                    <span className="text-gray-400">Available cash</span>
                    <span className="font-mono text-gray-200">
                      ¥{account.cash.toLocaleString("zh-CN", { minimumFractionDigits: 2 })}
                    </span>
                  </div>
                )}
                {account?.is_simulated && (
                  <div className="flex justify-between">
                    <span className="text-gray-500 text-xs">Mode</span>
                    <span className="text-xs text-amber-400 font-medium">SIMULATOR</span>
                  </div>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleSubmitOrder}
                  disabled={submitting}
                  className={clsx(
                    "flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50",
                    orderSide === "BUY" ? "bg-emerald-600 hover:bg-emerald-500" : "bg-red-600 hover:bg-red-500"
                  )}
                >
                  {submitting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ShoppingCart className="w-3.5 h-3.5" />}
                  {submitting ? "Submitting…" : `Confirm ${orderSide}`}
                </button>
                <button
                  onClick={() => setShowConfirm(false)}
                  disabled={submitting}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
                >
                  Edit
                </button>
              </div>
              {orderError && (
                <div className="p-2.5 bg-red-950 border border-red-800 rounded-lg text-xs text-red-300">
                  {orderError}
                </div>
              )}
            </div>
          ) : (
            /* ── entry form ── */
            <>
              <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">
                Place order — {row.symbol}{row.name ? ` · ${row.name}` : ""}
              </p>

              {/* Position banner */}
              {existingPos && (
                <div className="flex items-center gap-2 px-3 py-2 bg-amber-950/50 border border-amber-800/60 rounded-lg text-xs text-amber-300">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  You hold{" "}
                  <span className="font-bold font-mono mx-0.5">{existingPos.qty}</span> shares @ avg{" "}
                  <span className="font-mono mx-0.5">¥{existingPos.avg_price.toFixed(2)}</span>
                  {" "}(P&amp;L:{" "}
                  <span className={clsx("font-semibold ml-0.5", existingPos.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400")}>
                    {existingPos.pnl_pct >= 0 ? "+" : ""}{(existingPos.pnl_pct * 100).toFixed(2)}%
                  </span>)
                </div>
              )}

              {/* Cash balance */}
              {account && (
                <div className="text-xs text-gray-500">
                  Cash available:{" "}
                  <span className="font-mono text-gray-300">
                    ¥{account.cash.toLocaleString("zh-CN", { minimumFractionDigits: 2 })}
                  </span>
                  {account.is_simulated && (
                    <span className="ml-2 text-amber-500 font-medium">[SIMULATOR]</span>
                  )}
                </div>
              )}

              <div className="flex flex-wrap gap-3 items-end">
                {/* Side */}
                <div>
                  <label className="block text-[10px] text-gray-500 mb-1">Side</label>
                  <div className="flex rounded-lg overflow-hidden border border-gray-700">
                    {(["BUY", "SELL"] as const).map((s) => (
                      <button
                        key={s}
                        onClick={() => setOrderSide(s)}
                        className={clsx(
                          "px-4 py-1.5 text-xs font-semibold transition-colors",
                          orderSide === s
                            ? s === "BUY" ? "bg-emerald-700 text-white" : "bg-red-700 text-white"
                            : "bg-gray-800 text-gray-400 hover:text-gray-200"
                        )}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Order type */}
                <div>
                  <label className="block text-[10px] text-gray-500 mb-1">Type</label>
                  <div className="flex rounded-lg overflow-hidden border border-gray-700">
                    {(["MARKET", "LIMIT"] as const).map((t) => (
                      <button
                        key={t}
                        onClick={() => setOrderType(t)}
                        className={clsx(
                          "px-3 py-1.5 text-xs font-medium transition-colors",
                          orderType === t
                            ? "bg-sky-700 text-white"
                            : "bg-gray-800 text-gray-400 hover:text-gray-200"
                        )}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Qty */}
                <div>
                  <label className="block text-[10px] text-gray-500 mb-1">Quantity</label>
                  <input
                    type="number"
                    min={1}
                    value={orderQty}
                    onChange={(e) => setOrderQty(Math.max(1, parseInt(e.target.value) || 1))}
                    className="w-24 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-sky-500"
                  />
                </div>

                {/* Limit price (only for LIMIT orders) */}
                {orderType === "LIMIT" && (
                  <div>
                    <label className="block text-[10px] text-gray-500 mb-1">Limit price</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.01"
                      placeholder="e.g. 38.50"
                      value={limitPrice}
                      onChange={(e) => setLimitPrice(e.target.value)}
                      className="w-28 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-sky-500"
                    />
                  </div>
                )}

                {/* Review button → goes to confirm step */}
                <button
                  onClick={() => setShowConfirm(true)}
                  disabled={orderType === "LIMIT" && !limitPrice}
                  className={clsx(
                    "flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50",
                    orderSide === "BUY" ? "bg-emerald-600 hover:bg-emerald-500" : "bg-red-600 hover:bg-red-500"
                  )}
                >
                  <ShoppingCart className="w-3.5 h-3.5" />
                  Review {orderSide} {orderQty}
                </button>
              </div>

              {orderError && (
                <div className="p-2.5 bg-red-950 border border-red-800 rounded-lg text-xs text-red-300">
                  {orderError}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* training status */}
      {trainStatus && (
        <div className="px-4 py-1.5 bg-gray-800 text-xs text-sky-300 border-t border-gray-700">
          {trainStatus}
        </div>
      )}

      {/* expanded: causal trace + fundamentals */}
      {expanded && (
        <div className="px-4 pb-4 pt-3 border-t border-gray-800 space-y-4">
          {/* CausalTracePanel (primary) */}
          {row.causal_nodes && row.causal_nodes.length > 0 ? (
            <CausalTracePanel
              decision="RECOMMENDED"
              score={row.score}
              causalNodes={row.causal_nodes}
              dataScope={row.data_scope}
            />
          ) : (
            /* fallback: old fundamentals grid */
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              {[
                { k: "pe_ttm",         label: "P/E" },
                { k: "pb",             label: "P/B" },
                { k: "roe",            label: "ROE %" },
                { k: "revenue_growth", label: "Rev. Gr." },
                { k: "net_margin",     label: "Net Margin" },
                { k: "dividend_yield", label: "Yield" },
                { k: "value_score",    label: "Value" },
                { k: "growth_score",   label: "Growth" },
              ].map(({ k, label }) => (
                <div key={k} className="bg-gray-800 rounded-lg px-3 py-2 flex justify-between">
                  <span className="text-gray-500">{label}</span>
                  <span className="text-gray-200">{num(fund[k])}</span>
                </div>
              ))}
              {row.model_trained && (
                <div className="col-span-2 sm:col-span-4 bg-indigo-950 rounded-lg px-3 py-2 flex justify-between items-center border border-indigo-800">
                  <span className="text-indigo-400">ML signal</span>
                  <div className="flex items-center gap-2">
                    <SignalBadge signal={row.model_signal} />
                    <span className="text-indigo-300">{(row.model_confidence * 100).toFixed(1)}% conf.</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  STORED MODELS SUB-PANEL
// ═══════════════════════════════════════════════════════════════════════════════

function ModelsSubPanel() {
  const [models, setModels]               = useState<StoredModelInfo[]>([]);
  const [loading, setLoading]             = useState(false);
  const [deleting, setDeleting]           = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [error, setError]                 = useState("");

  async function load() {
    setLoading(true);
    setError("");
    try {
      const r = await listModels();
      setModels(r.models);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleDelete(sym: string) {
    setDeleting(sym);
    setConfirmDelete(null);
    try {
      await deleteModel(sym);
      setModels((prev) => prev.filter((m) => m.symbol !== sym));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">
          {models.length} stored model{models.length !== 1 ? "s" : ""}
        </h3>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-xs font-medium transition-colors"
        >
          <RefreshCw className={clsx("w-3.5 h-3.5", loading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-300">{error}</div>
      )}

      {models.length === 0 && !loading && (
        <p className="text-sm text-gray-500">
          No trained models stored yet. Use the{" "}
          <span className="text-sky-400">Stock Analyzer</span> or{" "}
          <span className="text-sky-400">Recommendations</span> tab to train models.
        </p>
      )}

      {models.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-gray-800">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-900">
                {["Symbol", "Strategy", "Bars", "Last Bar", "OOS Acc.", "Trained At", ""].map((h) => (
                  <th key={h} className="px-3 py-2.5 text-left text-gray-400 font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.model_id} className="border-b border-gray-800 hover:bg-gray-900 transition-colors">
                  <td className="px-3 py-2.5 font-mono font-bold text-sky-400">{m.symbol}</td>
                  <td className="px-3 py-2.5 text-gray-400">{m.strategy_id}</td>
                  <td className="px-3 py-2.5 text-gray-300">{m.bar_count}</td>
                  <td className="px-3 py-2.5 text-gray-300">{m.last_bar_date}</td>
                  <td className="px-3 py-2.5">
                    {m.oos_accuracy !== undefined ? (
                      <span className={clsx(
                        "font-medium",
                        m.oos_accuracy >= 0.55 ? "text-emerald-400" : "text-yellow-400"
                      )}>
                        {(m.oos_accuracy * 100).toFixed(1)}%
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-3 py-2.5 text-gray-400 whitespace-nowrap">{ts(m.trained_at)}</td>
                  <td className="px-3 py-2.5">
                    {confirmDelete === m.symbol ? (
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleDelete(m.symbol)}
                          className="px-2 py-1 rounded bg-red-700 hover:bg-red-600 text-red-100 text-[10px] font-medium"
                        >
                          Confirm
                        </button>
                        <button
                          onClick={() => setConfirmDelete(null)}
                          className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 text-[10px] font-medium"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setConfirmDelete(m.symbol)}
                        disabled={deleting === m.symbol}
                        className="p-1.5 rounded hover:bg-red-900 text-gray-500 hover:text-red-400 transition-colors"
                        title="Delete model (triggers retrain)"
                      >
                        {deleting === m.symbol ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Trash2 className="w-3.5 h-3.5" />
                        )}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN ADVISOR PANEL
// ═══════════════════════════════════════════════════════════════════════════════

export default function AdvisorPanel() {
  const [sub, setSub] = useState<AdvisorTab>("recommend");

  return (
    <div className="space-y-6">
      {/* Sub-tab bar */}
      <div className="flex gap-1 border-b border-gray-800">
        {SUBTABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setSub(t.id)}
            className={clsx(
              "flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-t transition-colors",
              sub === t.id
                ? "bg-gray-900 text-sky-400 border-b-2 border-sky-400"
                : "text-gray-400 hover:text-gray-200"
            )}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {sub === "analyze"   && <AnalyzeSubPanel />}
      {sub === "recommend" && <RecommendSubPanel />}
      {sub === "models"    && <ModelsSubPanel />}
    </div>
  );
}

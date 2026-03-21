"use client";

import { useState, useCallback, useEffect } from "react";
import clsx from "clsx";
import {
  Brain, Play, Loader2, AlertCircle, Search, Trash2, RefreshCw,
  TrendingUp, TrendingDown, Minus, ChevronDown, ChevronRight,
} from "lucide-react";
import {
  startAnalyze, getAnalyze, startRecommend, pollRecommend,
  listModels, deleteModel, getSectors, pollJob,
} from "@/lib/api";
import type {
  AnalysisResult, RecommendResult, RecommendationRow, StoredModelInfo,
} from "@/lib/api";
import { ProgressBar } from "./ProgressBar";
import { ModelExplainerCard } from "./ModelExplainerCard";
import { CausalTracePanel } from "./CausalTracePanel";

function fmtPct(n?: number, d = 1) { return n == null ? "—" : `${(n * 100).toFixed(d)}%`; }
function fmt2(n?: number) { return n == null ? "—" : n.toFixed(2); }

function SignalBadge({ signal }: { signal: string }) {
  const colors: Record<string, string> = {
    BUY:  "bg-emerald-500/15 text-emerald-400 border-emerald-500/40",
    SELL: "bg-red-500/15 text-red-400 border-red-500/40",
    HOLD: "bg-yellow-500/15 text-yellow-400 border-yellow-500/40",
  };
  const Icon = signal === "BUY" ? TrendingUp : signal === "SELL" ? TrendingDown : Minus;
  return (
    <span className={clsx("inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold border", colors[signal] ?? "text-gray-400")}>
      <Icon className="w-3 h-3" />
      {signal}
    </span>
  );
}

// ── Analyse tab ───────────────────────────────────────────────────────────────
function AnalyseTab() {
  const [symbol, setSymbol] = useState("sh600519");
  const [forceRetrain, setForceRetrain] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState("");
  const [pct, setPct] = useState(0);
  const [showNews, setShowNews] = useState(false);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPct(0);
    setStep("Starting…");
    try {
      const init = await startAnalyze(symbol.trim(), forceRetrain);
      const final = await pollJob(
        init.job_id,
        getAnalyze,
        undefined,
        1500,
        (d) => { setStep(d.step ?? ""); setPct(d.pct ?? 0); },
      );
      setResult(final);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [symbol, forceRetrain]);

  return (
    <div className="space-y-4">
      <div className="flex gap-3 items-end">
        <div className="flex-1">
          <label className="block text-xs text-gray-400 mb-1">Symbol</label>
          <input
            className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={symbol}
            onChange={e => setSymbol(e.target.value)}
            placeholder="sh600519"
          />
        </div>
        <label className="flex items-center gap-1.5 text-xs text-gray-400 pb-2 cursor-pointer">
          <input type="checkbox" checked={forceRetrain} onChange={e => setForceRetrain(e.target.checked)}
            className="rounded border-gray-600 bg-gray-800 text-indigo-500" />
          Force retrain
        </label>
        <button
          onClick={run}
          disabled={loading}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium text-white transition-colors"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          Analyse
        </button>
      </div>

      {loading && (
        <div className="space-y-1">
          <ProgressBar value={pct} />
          <p className="text-xs text-gray-400">{step}</p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {result?.status === "done" && (
        <div className="space-y-4">
          {/* Signal */}
          <div className="flex items-center gap-4">
            <SignalBadge signal={result.signal} />
            <div className="text-sm text-gray-400">{result.symbol} · {result.sector}</div>
            <div className="ml-auto text-xs text-gray-500">Confidence: {fmtPct(result.confidence)}</div>
          </div>

          {/* Probability bars */}
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: "Buy", val: result.p_buy, color: "bg-emerald-500" },
              { label: "Hold", val: result.p_hold, color: "bg-yellow-500" },
              { label: "Sell", val: result.p_sell, color: "bg-red-500" },
            ].map(({ label, val, color }) => (
              <div key={label} className="bg-gray-800/60 rounded-lg p-3">
                <div className="text-[10px] text-gray-500 uppercase mb-1">{label}</div>
                <div className="h-1.5 rounded bg-gray-700 mb-1">
                  <div className={clsx("h-full rounded", color)} style={{ width: `${val * 100}%` }} />
                </div>
                <div className="text-xs font-medium text-gray-200">{fmtPct(val)}</div>
              </div>
            ))}
          </div>

          {/* Momentum */}
          {Object.keys(result.momentum ?? {}).length > 0 && (
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
              {Object.entries(result.momentum).map(([k, v]) => (
                <div key={k} className="bg-gray-800/40 rounded p-2 text-center">
                  <div className="text-[10px] text-gray-500">{k}</div>
                  <div className={clsx("text-xs font-medium", v >= 0 ? "text-emerald-400" : "text-red-400")}>
                    {(v * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Fundamentals */}
          {Object.keys(result.fundamentals ?? {}).length > 0 && (
            <div>
              <div className="text-xs font-medium text-gray-400 mb-2">Fundamentals</div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {Object.entries(result.fundamentals).map(([k, v]) => (
                  <div key={k} className="bg-gray-800/40 rounded p-2">
                    <div className="text-[10px] text-gray-500">{k}</div>
                    <div className="text-xs text-gray-200 font-medium">{typeof v === "number" ? v.toFixed(2) : v}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Feature importance (model) */}
          {result.model_meta && (
            <ModelExplainerCard
              model={{
                model_id: result.model_meta.model_id,
                symbol: result.symbol,
                strategy_id: "lgbm",
                trained_at: result.model_meta.trained_at,
                bar_count: result.model_meta.bar_count,
                last_bar_date: result.model_meta.last_bar_date,
                oos_accuracy: result.model_meta.oos_accuracy,
                feature_cols: result.feature_importance.map(f => f.feature),
              }}
              features={result.feature_importance}
            />
          )}

          {/* News */}
          {result.news?.length > 0 && (
            <div>
              <button
                onClick={() => setShowNews(!showNews)}
                className="flex items-center gap-1.5 text-xs font-medium text-gray-400 hover:text-gray-200 transition-colors mb-2"
              >
                {showNews ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                News ({result.news.length})
              </button>
              {showNews && (
                <div className="space-y-2">
                  {result.news.slice(0, 5).map((n, i) => (
                    <div key={i} className="bg-gray-800/40 rounded p-3">
                      <div className="text-xs font-medium text-gray-200">{n.title}</div>
                      <div className="text-[10px] text-gray-500 mt-1">{n.source} · {n.ts}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Recommend tab ─────────────────────────────────────────────────────────────
function RecommendRow({ row }: { row: RecommendationRow }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-gray-800/40 transition-colors"
      >
        <span className="font-mono text-sm text-gray-100 w-20 shrink-0">{row.symbol}</span>
        <span className="text-xs text-gray-400 flex-1 truncate">{row.name}</span>
        <SignalBadge signal={row.model_signal} />
        <span className={clsx("text-xs font-medium w-12 text-right shrink-0",
          row.score >= 0 ? "text-emerald-400" : "text-red-400")}>{row.score.toFixed(2)}</span>
        {open ? <ChevronDown className="w-3.5 h-3.5 text-gray-500" /> : <ChevronRight className="w-3.5 h-3.5 text-gray-500" />}
      </button>
      <div className="px-4 pb-2 text-[11px] text-gray-500 flex flex-wrap gap-3">
        <span>1Y: <span className={clsx("font-medium", row.ret_1y >= 0 ? "text-emerald-400" : "text-red-400")}>{fmtPct(row.ret_1y)}</span></span>
        <span>3M: <span className={clsx("font-medium", row.ret_3m >= 0 ? "text-emerald-400" : "text-red-400")}>{fmtPct(row.ret_3m)}</span></span>
        <span>1M: <span className={clsx("font-medium", row.ret_1m >= 0 ? "text-emerald-400" : "text-red-400")}>{fmtPct(row.ret_1m)}</span></span>
        <span>Conf: <span className="text-gray-300">{fmtPct(row.model_confidence)}</span></span>
        {!row.model_trained && <span className="text-orange-400">no model</span>}
      </div>
      {open && row.causal_nodes?.length > 0 && (
        <div className="px-4 pb-3 border-t border-gray-800 pt-3">
          <CausalTracePanel nodes={row.causal_nodes} scope={row.data_scope} />
        </div>
      )}
    </div>
  );
}

function RecommendTab() {
  const [sectors, setSectors] = useState<string[]>([]);
  const [sector, setSector] = useState("");
  const [topN, setTopN] = useState(10);
  const [result, setResult] = useState<RecommendResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState("");
  const [pct, setPct] = useState(0);

  useEffect(() => {
    getSectors().then(r => setSectors(r.sectors)).catch(() => null);
  }, []);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPct(0);
    setStep("Fetching recommendations…");
    try {
      const init = await startRecommend(sector || undefined, topN);
      const final = await pollJob(
        init.job_id,
        pollRecommend,
        undefined,
        1500,
        (d) => { setStep(d.step ?? ""); setPct(d.pct ?? 0); },
      );
      setResult(final);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [sector, topN]);

  return (
    <div className="space-y-4">
      <div className="flex gap-3 items-end flex-wrap">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Sector</label>
          <select
            className="rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={sector}
            onChange={e => setSector(e.target.value)}
          >
            <option value="">All sectors</option>
            {sectors.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Top N</label>
          <input type="number" className="w-20 rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={topN} onChange={e => setTopN(Number(e.target.value))} />
        </div>
        <button
          onClick={run}
          disabled={loading}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium text-white transition-colors"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          Recommend
        </button>
      </div>

      {loading && (
        <div className="space-y-1">
          <ProgressBar value={pct} />
          <p className="text-xs text-gray-400">{step}</p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {result?.status === "done" && (
        <div className="space-y-2">
          <p className="text-xs text-gray-400">{result.rows.length} recommendations</p>
          {result.rows.map(row => <RecommendRow key={row.symbol} row={row} />)}
        </div>
      )}
    </div>
  );
}

// ── Models tab ────────────────────────────────────────────────────────────────
function ModelsTab() {
  const [models, setModels] = useState<StoredModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await listModels();
      setModels(r.models);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleDelete = useCallback(async (symbol: string) => {
    if (!confirm(`Delete model for ${symbol}?`)) return;
    try {
      await deleteModel(symbol);
      setModels(prev => prev.filter(m => m.symbol !== symbol));
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  }, []);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-xs font-medium text-gray-200 transition-colors"
        >
          <RefreshCw className={clsx("w-3.5 h-3.5", loading && "animate-spin")} />
          Refresh
        </button>
        <span className="text-xs text-gray-500">{models.length} stored model{models.length !== 1 ? "s" : ""}</span>
      </div>

      {error && (
        <div className="flex items-start gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {models.length === 0 && !loading && (
        <p className="text-sm text-gray-500">No stored models yet. Run Analyse to train one.</p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {models.map(m => (
          <ModelExplainerCard key={m.model_id} model={m} onDelete={handleDelete} />
        ))}
      </div>
    </div>
  );
}

// ── Main Advisor panel ────────────────────────────────────────────────────────
const TABS = ["Analyse", "Recommend", "Models"] as const;
type Tab = typeof TABS[number];

export default function AdvisorPanel() {
  const [tab, setTab] = useState<Tab>("Analyse");

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      <div className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold text-gray-100">Advisor</h2>
      </div>

      <div className="flex gap-1 border-b border-gray-800 pb-0">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={clsx(
              "px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors",
              tab === t
                ? "border-indigo-500 text-indigo-400"
                : "border-transparent text-gray-500 hover:text-gray-300",
            )}
          >
            {t}
          </button>
        ))}
      </div>

      <div>
        {tab === "Analyse" && <AnalyseTab />}
        {tab === "Recommend" && <RecommendTab />}
        {tab === "Models" && <ModelsTab />}
      </div>
    </div>
  );
}

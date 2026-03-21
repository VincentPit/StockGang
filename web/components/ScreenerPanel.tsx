"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { Search, Play, Loader2, AlertCircle, ChevronDown, ChevronRight } from "lucide-react";
import { startScreen, getScreen, pollJob } from "@/lib/api";
import type { ScreenRequest, ScreenResult, ScreenRow } from "@/lib/api";
import { ProgressBar } from "./ProgressBar";
import { CausalTracePanel } from "./CausalTracePanel";

function fmtPct(n?: number) { return n == null ? "—" : `${(n * 100).toFixed(1)}%`; }
function fmt2(n?: number) { return n == null ? "—" : n.toFixed(2); }

const BADGE: Record<string, string> = {
  UPTREND: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
  DOWNTREND: "bg-red-500/10 text-red-400 border-red-500/30",
  SIDEWAYS: "bg-yellow-500/10 text-yellow-400 border-yellow-500/30",
};

function ScreenRowCard({ row }: { row: ScreenRow }) {
  const [open, setOpen] = useState(false);
  const trend = row.data_scope?.trend;

  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-gray-800/40 transition-colors"
      >
        <span className="text-xs text-gray-500 w-4 shrink-0">#{row.rank}</span>
        <span className="font-mono text-sm text-gray-100 w-20 shrink-0">{row.symbol}</span>
        <span className="text-xs text-gray-400 flex-1 truncate">{row.name}</span>
        {trend && (
          <span className={clsx("px-1.5 py-0.5 rounded-full text-[10px] border", BADGE[trend] ?? "text-gray-400")}>
            {trend}
          </span>
        )}
        <span className={clsx("text-xs font-semibold w-12 text-right shrink-0",
          row.score >= 0 ? "text-emerald-400" : "text-red-400")}>{fmt2(row.score)}</span>
        {open ? <ChevronDown className="w-3.5 h-3.5 text-gray-500" /> : <ChevronRight className="w-3.5 h-3.5 text-gray-500" />}
      </button>

      {/* Collapsed summary row */}
      <div className="flex gap-4 px-4 pb-2 text-[11px] text-gray-500 flex-wrap">
        <span>1Y: <span className={clsx("font-medium", row.ret_1y >= 0 ? "text-emerald-400" : "text-red-400")}>{fmtPct(row.ret_1y)}</span></span>
        <span>Sharpe: <span className="text-gray-300">{fmt2(row.sharpe)}</span></span>
        <span>Max DD: <span className="text-orange-400">{fmtPct(row.max_dd)}</span></span>
        <span>Trend%: <span className="text-gray-300">{fmtPct(row.trend_pct)}</span></span>
        <span>ATR%: <span className="text-gray-300">{fmtPct(row.atr_pct)}</span></span>
        {row.recommended && <span className="text-indigo-400 font-medium">✓ Recommended</span>}
      </div>

      {open && (
        <div className="px-4 pb-3 border-t border-gray-800 pt-3 space-y-3">
          {row.causal_nodes?.length > 0 && (
            <CausalTracePanel
              nodes={row.causal_nodes}
              gates={row.gate_checks}
              scope={row.data_scope}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default function ScreenerPanel() {
  const [topN, setTopN] = useState(20);
  const [minBars, setMinBars] = useState(60);
  const [lookbackYears, setLookbackYears] = useState(2);
  const [indices, setIndices] = useState("000300");
  const [result, setResult] = useState<ScreenResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState("");
  const [pct, setPct] = useState(0);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPct(0);
    setStep("Starting…");
    try {
      const req: ScreenRequest = {
        top_n: topN,
        min_bars: minBars,
        lookback_years: lookbackYears,
        indices: indices.split(",").map(s => s.trim()).filter(Boolean),
      };
      const init = await startScreen(req);
      const final = await pollJob(
        init.job_id,
        getScreen,
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
  }, [topN, minBars, lookbackYears, indices]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      <div className="flex items-center gap-2">
        <Search className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold text-gray-100">Screener</h2>
        {result?.universe_size && (
          <span className="text-xs text-gray-500 ml-auto">
            Universe: {result.universe_size} symbols
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Top N</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={topN} onChange={e => setTopN(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Min Bars</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={minBars} onChange={e => setMinBars(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Lookback (years)</label>
          <input type="number" step="0.5" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={lookbackYears} onChange={e => setLookbackYears(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Indices (CSI codes)</label>
          <input className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={indices} onChange={e => setIndices(e.target.value)} placeholder="000300,000905" />
        </div>
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium text-white transition-colors"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        {loading ? "Screening…" : "Run Screener"}
      </button>

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

      {result?.status === "done" && result.rows.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-gray-400">
            Top {result.rows.length} symbols — recommended: {result.top_symbols.length}
          </p>
          {result.rows.map(row => (
            <ScreenRowCard key={row.symbol} row={row} />
          ))}
        </div>
      )}

      {result?.status === "done" && result.rows.length === 0 && (
        <p className="text-sm text-gray-500">No symbols passed screening.</p>
      )}
    </div>
  );
}

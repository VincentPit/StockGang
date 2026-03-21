"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { GitMerge, Play, Loader2, AlertCircle, TrendingUp, List } from "lucide-react";
import { startWorkflow, getWorkflow, pollJob } from "@/lib/api";
import type { WorkflowRequest, WorkflowResult, ScreenRow } from "@/lib/api";
import { ProgressBar } from "./ProgressBar";
import { NavChart } from "./NavChart";
import { TradesTable } from "./TradesTable";

function fmt2(n?: number) { return n == null ? "—" : n.toFixed(2); }
function fmtPct(n?: number) { return n == null ? "—" : `${(n * 100).toFixed(2)}%`; }
function fmtPnl(n?: number) {
  if (n == null) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}¥${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

function ScreenBadges({ rows }: { rows: ScreenRow[] }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {rows.map(r => (
        <span
          key={r.symbol}
          title={`Score: ${r.score.toFixed(2)} | Sharpe: ${r.sharpe.toFixed(2)} | 1Y: ${(r.ret_1y * 100).toFixed(1)}%`}
          className={clsx(
            "px-2 py-0.5 rounded-full text-[10px] font-mono border",
            r.recommended
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
              : "bg-gray-700/40 border-gray-700 text-gray-400"
          )}
        >
          {r.symbol}
        </span>
      ))}
    </div>
  );
}

export default function WorkflowPanel() {
  const [topN, setTopN] = useState(10);
  const [minBars, setMinBars] = useState(60);
  const [lookbackYears, setLookbackYears] = useState(2);
  const [backtestDays, setBacktestDays] = useState(365);
  const [cash, setCash] = useState(1_000_000);
  const [commission, setCommission] = useState(0.001);
  const [stopLoss, setStopLoss] = useState(0.08);
  const [indices, setIndices] = useState("000300");

  const [result, setResult] = useState<WorkflowResult | null>(null);
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
      const req: WorkflowRequest = {
        top_n: topN,
        min_bars: minBars,
        lookback_years: lookbackYears,
        backtest_days: backtestDays,
        initial_cash: cash,
        commission_rate: commission,
        stop_loss_pct: stopLoss,
        indices: indices.split(",").map(s => s.trim()).filter(Boolean),
      };
      const init = await startWorkflow(req);
      const final = await pollJob(
        init.job_id,
        getWorkflow,
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
  }, [topN, minBars, lookbackYears, backtestDays, cash, commission, stopLoss, indices]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      <div className="flex items-center gap-2">
        <GitMerge className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold text-gray-100">Workflow</h2>
        <span className="text-xs text-gray-500">Screen → Backtest pipeline</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Top N symbols</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={topN} onChange={e => setTopN(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Indices (CSI)</label>
          <input className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={indices} onChange={e => setIndices(e.target.value)} placeholder="000300,000905" />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Screen lookback (yrs)</label>
          <input type="number" step="0.5" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={lookbackYears} onChange={e => setLookbackYears(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Min Bars</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={minBars} onChange={e => setMinBars(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Backtest days</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={backtestDays} onChange={e => setBacktestDays(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Initial Cash</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={cash} onChange={e => setCash(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Commission</label>
          <input type="number" step="0.0001" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={commission} onChange={e => setCommission(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Stop Loss %</label>
          <input type="number" step="0.01" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={stopLoss} onChange={e => setStopLoss(Number(e.target.value))} />
        </div>
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium text-white transition-colors"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        {loading ? "Running…" : "Run Workflow"}
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

      {result?.status === "done" && (
        <div className="space-y-5">
          {/* Screened symbols */}
          {result.screen_rows?.length > 0 && (
            <div>
              <div className="text-xs font-medium text-gray-400 mb-2">
                Screened symbols ({result.screen_rows.length})
              </div>
              <ScreenBadges rows={result.screen_rows} />
            </div>
          )}

          {/* Backtest stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Total PnL", value: fmtPnl(result.total_pnl), color: (result.total_pnl ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
              { label: "Return", value: fmtPct(result.total_pnl_pct), color: (result.total_pnl_pct ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
              { label: "Sharpe", value: fmt2(result.sharpe), color: "text-gray-100" },
              { label: "Max DD", value: fmtPct(result.max_dd), color: "text-orange-400" },
              { label: "Win Rate", value: fmtPct(result.win_rate), color: "text-gray-100" },
              { label: "# Trades", value: String(result.num_trades ?? "—"), color: "text-gray-100" },
              { label: "Period", value: `${result.period_start ?? ""} → ${result.period_end ?? ""}`, color: "text-gray-400" },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-gray-800/60 rounded-lg p-3">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
                <div className={clsx("text-sm font-semibold", color)}>{value}</div>
              </div>
            ))}
          </div>

          {result.nav_series?.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">NAV</span>
              </div>
              <NavChart data={result.nav_series} />
            </div>
          )}

          {result.trades?.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <List className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">Trades ({result.trades.length})</span>
              </div>
              <TradesTable trades={result.trades} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

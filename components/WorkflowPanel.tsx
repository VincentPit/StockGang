"use client";

import { useState } from "react";
import {
  startWorkflow,
  getWorkflow,
  pollJob,
  type WorkflowResult,
  type WorkflowRequest,
} from "@/lib/api";
import { MetricCard } from "@/components/MetricCard";
import { NavChart } from "@/components/NavChart";
import { SymbolPnLChart } from "@/components/SymbolPnLChart";
import { TradesTable } from "@/components/TradesTable";
import {
  Zap,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import clsx from "clsx";

const STATUS_LABELS: Record<string, string> = {
  pending:     "Queued…",
  screening:   "Screening stocks…",
  backtesting: "Backtesting top picks…",
  done:        "Complete",
  error:       "Failed",
};

export default function WorkflowPanel() {
  const [topN,         setTopN]         = useState(6);
  const [backtestDays, setBacktestDays] = useState(365);
  const [initialCash,  setInitialCash]  = useState(1_000_000);
  const [lookbackYrs,  setLookbackYrs]  = useState(1);
  const [indices,      setIndices]      = useState<string[]>(["000300"]);

  const [status,   setStatus]   = useState("");
  const [result,   setResult]   = useState<WorkflowResult | null>(null);
  const [error,    setError]    = useState("");
  const [running,  setRunning]  = useState(false);
  const [showTrades, setShowTrades] = useState(false);

  async function run() {
    setRunning(true);
    setError("");
    setResult(null);
    setStatus("pending");

    const req: WorkflowRequest = {
      top_n:          topN,
      backtest_days:  backtestDays,
      initial_cash:   initialCash,
      lookback_years: lookbackYrs,
      indices,
    };

    try {
      const { job_id } = await startWorkflow(req);
      const data = await pollJob(job_id, getWorkflow, setStatus, 2000);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    } finally {
      setRunning(false);
    }
  }

  const pnlPct = result?.total_pnl_pct != null
    ? (result.total_pnl_pct * 100).toFixed(2)
    : null;

  return (
    <div className="space-y-6">
      {/* ── Config ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div>
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Top N stocks</label>
          <input
            type="number"
            min={1} max={20}
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Screener lookback</label>
          <select
            value={lookbackYrs}
            onChange={(e) => setLookbackYrs(Number(e.target.value))}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
          >
            <option value={1}>1 year</option>
            <option value={2}>2 years</option>
            <option value={3}>3 years</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Backtest window</label>
          <select
            value={backtestDays}
            onChange={(e) => setBacktestDays(Number(e.target.value))}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
          >
            <option value={180}>6 months</option>
            <option value={365}>1 year</option>
            <option value={730}>2 years</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Initial cash (¥)</label>
          <select
            value={initialCash}
            onChange={(e) => setInitialCash(Number(e.target.value))}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
          >
            <option value={500_000}>¥500,000</option>
            <option value={1_000_000}>¥1,000,000</option>
            <option value={5_000_000}>¥5,000,000</option>
          </select>
        </div>
      </div>

      {/* ── Index universe selector ── */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-500 uppercase tracking-wider">Universe</span>
        {([
          { label: "CSI 300",      value: ["000300"] },
          { label: "CSI 300+500",  value: ["000300", "000905"] },
        ] as const).map((opt) => {
          const active = JSON.stringify(opt.value) === JSON.stringify(indices);
          return (
            <button
              key={opt.label}
              onClick={() => setIndices([...opt.value])}
              className={clsx(
                "text-xs px-3 py-1.5 rounded-lg border transition-colors",
                active
                  ? "bg-sky-700 border-sky-600 text-white"
                  : "bg-gray-900 border-gray-700 text-gray-400 hover:text-gray-200"
              )}
            >
              {opt.label}
            </button>
          );
        })}
      </div>

      <button
        onClick={run}
        disabled={running}
        className="flex items-center gap-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded-lg px-6 py-2.5 text-sm font-medium transition-colors"
      >
        {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
        {running ? "Running workflow…" : "Run full workflow"}
      </button>

      {/* ── Status bar ── */}
      {status && (
        <div className={clsx(
          "flex items-center gap-2 text-sm px-3 py-2 rounded-lg",
          status === "done"  && "bg-emerald-950/40 text-emerald-400",
          status === "error" && "bg-rose-950/40 text-rose-400",
          !["done","error"].includes(status) && "bg-gray-900 text-sky-400",
        )}>
          {status === "done"  && <CheckCircle2 className="w-4 h-4" />}
          {status === "error" && <AlertCircle  className="w-4 h-4" />}
          {!["done","error"].includes(status) && <Loader2 className="w-4 h-4 animate-spin" />}
          {STATUS_LABELS[status] ?? status}
        </div>
      )}

      {error && (
        <div className="text-rose-400 text-sm bg-rose-950/30 border border-rose-900 rounded-lg p-3">
          {error}
        </div>
      )}

      {/* ── Screener picks ── */}
      {result && result.top_symbols.length > 0 && (
        <div className="bg-gray-950 border border-gray-800 rounded-xl p-4">
          <p className="text-sm font-medium mb-3">Selected by Screener</p>
          <div className="flex flex-wrap gap-2">
            {result.top_symbols.map((s) => (
              <span key={s} className="text-xs bg-sky-900/50 text-sky-300 border border-sky-800 rounded-full px-2.5 py-1">
                {s}
              </span>
            ))}
          </div>
          {result.screen_rows.length > 0 && (
            <table className="mt-3 w-full text-xs text-gray-400">
              <thead>
                <tr className="border-b border-gray-800 text-gray-600">
                  <th className="text-left py-1 pr-3">#</th>
                  <th className="text-left py-1 pr-3">Symbol</th>
                  <th className="text-left py-1 pr-3">Name</th>
                  <th className="text-right py-1 pr-3">Score</th>
                  <th className="text-right py-1 pr-3">Ret 1Y</th>
                  <th className="text-right py-1">Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {result.screen_rows.map((r) => (
                  <tr
                    key={r.symbol}
                    className={clsx("border-b border-gray-900", r.recommended && "text-sky-400")}
                  >
                    <td className="py-1 pr-3">{r.rank}</td>
                    <td className="py-1 pr-3 font-mono">{r.symbol}</td>
                    <td className="py-1 pr-3">{r.name}</td>
                    <td className="py-1 pr-3 text-right">{r.score.toFixed(3)}</td>
                    <td className={clsx("py-1 pr-3 text-right", r.ret_1y >= 0 ? "text-emerald-400" : "text-rose-400")}>
                      {(r.ret_1y * 100).toFixed(1)}%
                    </td>
                    <td className="py-1 text-right">{r.sharpe.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* ── Backtest summary ── */}
      {result?.status === "done" && result.total_pnl != null && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard
              label="Total Return"
              value={pnlPct !== null ? `${pnlPct}%` : "—"}
              positive={result.total_pnl >= 0}
            />
            <MetricCard
              label="Sharpe Ratio"
              value={result.sharpe?.toFixed(3) ?? "—"}
              positive={(result.sharpe ?? 0) > 0}
            />
            <MetricCard
              label="Max Drawdown"
              value={`${((result.max_dd ?? 0) * 100).toFixed(2)}%`}
              positive={false}
            />
            <MetricCard
              label={`Win Rate (${result.num_trades ?? 0} trades)`}
              value={`${((result.win_rate ?? 0) * 100).toFixed(1)}%`}
              positive={(result.win_rate ?? 0) >= 0.5}
            />
          </div>

          {result.nav_series?.length > 0 && (
            <NavChart data={result.nav_series} initial={result.initial_nav ?? 1_000_000} />
          )}

          {result.symbol_pnl?.length > 0 && (
            <SymbolPnLChart data={result.symbol_pnl} />
          )}

          {result.trades?.length > 0 && (
            <div>
              <button
                onClick={() => setShowTrades((v) => !v)}
                className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-200 mb-2"
              >
                {showTrades ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                {showTrades ? "Hide" : "Show"} trades ({result.trades.length})
              </button>
              {showTrades && <TradesTable trades={result.trades} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

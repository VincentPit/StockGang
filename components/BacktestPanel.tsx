"use client";

import { useState } from "react";
import {
  startBacktest,
  getBacktest,
  pollJob,
  type BacktestResult,
} from "@/lib/api";
import { NavChart } from "./NavChart";
import { MetricCard } from "./MetricCard";
import { TradesTable } from "./TradesTable";
import { SymbolPnLChart } from "./SymbolPnLChart";
import { Loader2, Play } from "lucide-react";

const DEFAULT_SYMBOLS = [
  "sz300059", "sz300750", "sz000858", "sz000333", "sh601318", "sh600036",
];

function fmt(n: number, pct = false) {
  if (pct) return `${(n * 100).toFixed(2)}%`;
  return n.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

export default function BacktestPanel() {
  const [symbolInput, setSymbolInput] = useState(DEFAULT_SYMBOLS.join(", "));
  const [lookback, setLookback] = useState(365);
  const [cash, setCash] = useState(1_000_000);
  const [status, setStatus] = useState<string>("");
  const [error, setError]   = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);

  async function run() {
    const symbols = symbolInput.split(",").map((s) => s.trim()).filter(Boolean);
    if (!symbols.length) return;
    setLoading(true);
    setResult(null);
    setError("");
    setStatus("Submitting…");
    try {
      const job = await startBacktest({ symbols, lookback_days: lookback, initial_cash: cash });
      setStatus("Running backtest (this takes ~1 min)…");
      const done = await pollJob(job.job_id, getBacktest, (s) => setStatus(`Status: ${s}…`));
      setResult(done);
      setStatus("Done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8">
      {/* Config form */}
      <div className="bg-gray-900 rounded-xl p-6 space-y-4">
        <h2 className="text-lg font-semibold text-sky-400">Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-3">
            <label className="block text-xs text-gray-400 mb-1">Symbols (comma-separated)</label>
            <input
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value)}
              placeholder="e.g. sz300059, sz300750, sz000858"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Lookback (days)</label>
            <input
              type="number"
              value={lookback}
              onChange={(e) => setLookback(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Initial Cash (¥)</label>
            <input
              type="number"
              value={cash}
              onChange={(e) => setCash(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={run}
              disabled={loading}
              className="flex items-center gap-2 w-full justify-center bg-sky-600 hover:bg-sky-500 disabled:opacity-50 rounded-lg px-4 py-2 text-sm font-medium transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {loading ? "Running…" : "Run Backtest"}
            </button>
          </div>
        </div>

        {status && (
          <p className="text-xs text-gray-400 mt-1">{status}</p>
        )}
      </div>

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      {result && result.status === "done" && (
        <>
          {/* Metric grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <MetricCard
              label="Total Return"
              value={fmt(result.total_pnl_pct ?? 0, true)}
              positive={(result.total_pnl_pct ?? 0) >= 0}
            />
            <MetricCard label="Sharpe" value={(result.sharpe ?? 0).toFixed(3)} />
            <MetricCard
              label="Max Drawdown"
              value={fmt(result.max_dd ?? 0, true)}
              positive={false}
            />
            <MetricCard label="# Trades" value={String(result.num_trades ?? 0)} />
            <MetricCard
              label="Win Rate"
              value={fmt(result.win_rate ?? 0, true)}
              positive={(result.win_rate ?? 0) >= 0.5}
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            <MetricCard label="Final NAV" value={`¥${fmt(result.final_nav ?? 0)}`} />
            <MetricCard
              label="Total P&L"
              value={`¥${fmt(result.total_pnl ?? 0)}`}
              positive={(result.total_pnl ?? 0) >= 0}
            />
            <MetricCard label="Avg Win" value={`¥${fmt(result.avg_win ?? 0)}`} positive />
            <MetricCard label="Avg Loss" value={`¥${fmt(result.avg_loss ?? 0)}`} positive={false} />
          </div>

          {/* NAV chart */}
          {result.nav_series.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6">
              <h2 className="text-sm font-semibold text-gray-300 mb-4">Portfolio NAV</h2>
              <NavChart data={result.nav_series} initial={result.initial_nav ?? 1_000_000} />
            </div>
          )}

          {/* Symbol P&L */}
          {result.symbol_pnl.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6">
              <h2 className="text-sm font-semibold text-gray-300 mb-4">Per-Symbol P&amp;L</h2>
              <SymbolPnLChart data={result.symbol_pnl} />
            </div>
          )}

          {/* Trades table */}
          {result.trades.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6">
              <h2 className="text-sm font-semibold text-gray-300 mb-4">
                Trade Log ({result.trades.length} fills)
              </h2>
              <TradesTable trades={result.trades} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

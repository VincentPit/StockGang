"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { Play, Loader2, AlertCircle, TrendingUp, BarChart2, List, RefreshCw } from "lucide-react";
import { startBacktest, getBacktest, pollJob } from "@/lib/api";
import type { BacktestRequest, BacktestResult } from "@/lib/api";
import { NavChart } from "./NavChart";
import { TradesTable } from "./TradesTable";
import { ProgressBar } from "./ProgressBar";

const DEFAULT_SYMBOLS = "sh600519,sh600036,sz000858";

function fmt2(n?: number) { return n == null ? "—" : n.toFixed(2); }
function fmtPct(n?: number) { return n == null ? "—" : `${(n * 100).toFixed(2)}%`; }
function fmtPnl(n?: number) {
  if (n == null) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}¥${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export default function BacktestPanel() {
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS);
  const [lookback, setLookback] = useState(365);
  const [cash, setCash] = useState(1_000_000);
  const [commission, setCommission] = useState(0.001);
  const [stopLoss, setStopLoss] = useState(0.08);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<string>("");
  const [pct, setPct] = useState(0);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setStep("Starting…");
    setPct(0);
    try {
      const req: BacktestRequest = {
        symbols: symbols.split(",").map(s => s.trim()).filter(Boolean),
        lookback_days: lookback,
        initial_cash: cash,
        commission_rate: commission,
        stop_loss_pct: stopLoss,
      };
      const init = await startBacktest(req);
      const final = await pollJob(
        init.job_id,
        getBacktest,
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
  }, [symbols, lookback, cash, commission, stopLoss]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      <div className="flex items-center gap-2">
        <BarChart2 className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold text-gray-100">Backtest</h2>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div className="col-span-2 sm:col-span-3">
          <label className="block text-xs text-gray-400 mb-1">Symbols (comma-separated)</label>
          <input
            className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={symbols}
            onChange={e => setSymbols(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Lookback (days)</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={lookback} onChange={e => setLookback(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Initial Cash</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={cash} onChange={e => setCash(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Commission Rate</label>
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
        {loading ? "Running…" : "Run Backtest"}
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

      {result && result.status === "done" && (
        <div className="space-y-5">
          {/* Summary metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Total PnL", value: fmtPnl(result.total_pnl), color: (result.total_pnl ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
              { label: "Return", value: fmtPct(result.total_pnl_pct), color: (result.total_pnl_pct ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
              { label: "Sharpe", value: fmt2(result.sharpe), color: "text-gray-100" },
              { label: "Max DD", value: fmtPct(result.max_dd), color: "text-orange-400" },
              { label: "Win Rate", value: fmtPct(result.win_rate), color: "text-gray-100" },
              { label: "Profit Factor", value: fmt2(result.profit_factor), color: "text-gray-100" },
              { label: "# Trades", value: String(result.num_trades ?? "—"), color: "text-gray-100" },
              { label: "Period", value: `${result.period_start ?? ""} → ${result.period_end ?? ""}`, color: "text-gray-400" },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-gray-800/60 rounded-lg p-3">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
                <div className={clsx("text-sm font-semibold", color)}>{value}</div>
              </div>
            ))}
          </div>

          {/* NAV Chart */}
          {result.nav_series?.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">NAV</span>
              </div>
              <NavChart data={result.nav_series} />
            </div>
          )}

          {/* Symbol PnL */}
          {result.symbol_pnl?.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <List className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">Per-Symbol PnL</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-800">
                      <th className="text-left py-1 pr-4">Symbol</th>
                      <th className="text-right py-1 pr-4">Net PnL</th>
                      <th className="text-right py-1 pr-4">Fills</th>
                      <th className="text-right py-1">B/S</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.symbol_pnl.map(s => (
                      <tr key={s.symbol} className="border-b border-gray-800/40">
                        <td className="py-1 pr-4 text-gray-200 font-mono">{s.symbol}</td>
                        <td className={clsx("py-1 pr-4 text-right", s.net_pnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                          {fmtPnl(s.net_pnl)}
                        </td>
                        <td className="py-1 pr-4 text-right text-gray-300">{s.fills}</td>
                        <td className="py-1 text-right text-gray-300">{s.buys}/{s.sells}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Trades */}
          {result.trades?.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <RefreshCw className="w-4 h-4 text-indigo-400" />
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

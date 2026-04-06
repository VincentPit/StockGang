"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { Play, Loader2, AlertCircle, TrendingUp, BarChart2, List, RefreshCw, Shield, Activity, Target, Zap } from "lucide-react";
import { startBacktest, getBacktest, pollJob } from "@/lib/api";
import type { BacktestRequest, BacktestResult } from "@/lib/api";
import { NavChart } from "./NavChart";
import { TradesTable } from "./TradesTable";
import { ProgressBar } from "./ProgressBar";

const DEFAULT_SYMBOLS = "sh600519,sh600036,sz000858";

function fmt2(n?: number) { return n == null ? "—" : n.toFixed(2); }
function fmt3(n?: number) { return n == null ? "—" : n.toFixed(3); }
function fmt4(n?: number) { return n == null ? "—" : n.toFixed(4); }
function fmtPct(n?: number) { return n == null ? "—" : `${(n * 100).toFixed(2)}%`; }
function fmtPnl(n?: number) {
  if (n == null) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}¥${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

type Tab = "summary" | "risk" | "attribution" | "trades";

export default function BacktestPanel() {
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS);
  const [lookback, setLookback] = useState(365);
  const [cash, setCash] = useState(1_000_000);
  const [commission, setCommission] = useState(0.001);
  const [stopLoss, setStopLoss] = useState(8);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<string>("");
  const [pct, setPct] = useState(0);
  const [tab, setTab] = useState<Tab>("summary");

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
        stop_loss_pct: -(stopLoss / 100),
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
        <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
          Institutional Analytics
        </span>
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
          <label className="block text-xs text-gray-400 mb-1">Stop Loss <span className="text-gray-500">(%, e.g. 8 = 8%)</span></label>
          <input type="number" step="1" min="0" max="50" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
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
          {/* Tab bar */}
          <div className="flex gap-1 border-b border-gray-800 pb-1">
            {([
              { key: "summary" as Tab, label: "Summary", icon: <TrendingUp className="w-3.5 h-3.5" /> },
              { key: "risk" as Tab,    label: "Risk",    icon: <Shield className="w-3.5 h-3.5" /> },
              { key: "attribution" as Tab, label: "Attribution", icon: <Target className="w-3.5 h-3.5" /> },
              { key: "trades" as Tab,  label: "Trades",  icon: <Activity className="w-3.5 h-3.5" /> },
            ]).map(t => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-t text-xs font-medium transition-colors",
                  tab === t.key
                    ? "bg-gray-800 text-indigo-400 border-b-2 border-indigo-500"
                    : "text-gray-500 hover:text-gray-300"
                )}
              >
                {t.icon}
                {t.label}
              </button>
            ))}
          </div>

          {/* ═══════ SUMMARY TAB ═══════ */}
          {tab === "summary" && (
            <>
              {/* Key metrics row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "Total PnL", value: fmtPnl(result.total_pnl), color: (result.total_pnl ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
                  { label: "Ann. Return", value: fmtPct(result.annualised_return ?? result.total_pnl_pct), color: (result.annualised_return ?? result.total_pnl_pct ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
                  { label: "Sharpe", value: fmt3(result.sharpe), color: "text-gray-100" },
                  { label: "Max DD", value: fmtPct(result.max_dd), color: "text-orange-400" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-gray-800/60 rounded-lg p-3">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
                    <div className={clsx("text-sm font-semibold", color)}>{value}</div>
                  </div>
                ))}
              </div>

              {/* Risk-adjusted metrics */}
              <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
                {[
                  { label: "Sortino", value: fmt3(result.sortino) },
                  { label: "Calmar", value: fmt3(result.calmar) },
                  { label: "Omega", value: fmt3(result.omega) },
                  { label: "Gain/Pain", value: fmt3(result.gain_to_pain) },
                  { label: "Win Rate", value: fmtPct(result.win_rate) },
                  { label: "Profit Factor", value: fmt2(result.profit_factor) },
                  { label: "# Trades", value: String(result.num_trades ?? "—") },
                  { label: "Ann. Vol", value: fmtPct(result.annualised_volatility) },
                  { label: "DD Days", value: result.max_dd_duration_days != null ? String(result.max_dd_duration_days) : "—" },
                  { label: "Period", value: `${result.period_start ?? ""} → ${result.period_end ?? ""}`, span: true },
                ].map(({ label, value, span }) => (
                  <div key={label} className={clsx("bg-gray-800/40 rounded-lg p-2", span && "col-span-3 sm:col-span-3")}>
                    <div className="text-[9px] text-gray-500 uppercase tracking-wider">{label}</div>
                    <div className="text-xs font-medium text-gray-200 mt-0.5">{value}</div>
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
            </>
          )}

          {/* ═══════ RISK TAB ═══════ */}
          {tab === "risk" && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "VaR (95%)", value: fmtPct(result.var_95), desc: "Daily Value at Risk" },
                  { label: "CVaR (95%)", value: fmtPct(result.cvar_95), desc: "Expected tail loss" },
                  { label: "Ulcer Index", value: fmt4(result.ulcer_index), desc: "Pain from drawdowns" },
                  { label: "Max DD Duration", value: result.max_dd_duration_days != null ? `${result.max_dd_duration_days} days` : "—", desc: "Longest recovery" },
                  { label: "Skewness", value: fmt3(result.skewness), desc: "Return distribution asymmetry" },
                  { label: "Kurtosis", value: fmt3(result.kurtosis), desc: "Tail heaviness" },
                  { label: "Avg Win", value: fmtPnl(result.avg_win), desc: "Mean winning trade" },
                  { label: "Avg Loss", value: fmtPnl(result.avg_loss), desc: "Mean losing trade" },
                ].map(({ label, value, desc }) => (
                  <div key={label} className="bg-gray-800/60 rounded-lg p-3">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
                    <div className="text-sm font-semibold text-gray-100">{value}</div>
                    <div className="text-[9px] text-gray-600 mt-0.5">{desc}</div>
                  </div>
                ))}
              </div>

              {/* Monthly Returns Heatmap */}
              {result.monthly_returns && Object.keys(result.monthly_returns).length > 0 && (
                <div>
                  <h3 className="text-xs font-medium text-gray-300 mb-2 flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-indigo-400" />
                    Monthly Returns
                  </h3>
                  <div className="grid grid-cols-6 sm:grid-cols-12 gap-1">
                    {Object.entries(result.monthly_returns).map(([month, ret]) => (
                      <div
                        key={month}
                        className={clsx(
                          "rounded p-1.5 text-center text-[9px] font-mono",
                          ret > 0.05 ? "bg-emerald-600/40 text-emerald-300" :
                          ret > 0 ? "bg-emerald-900/40 text-emerald-400" :
                          ret > -0.05 ? "bg-red-900/40 text-red-400" :
                          "bg-red-600/40 text-red-300"
                        )}
                        title={`${month}: ${(ret * 100).toFixed(2)}%`}
                      >
                        <div className="text-[8px] text-gray-500">{month.slice(5)}</div>
                        <div>{(ret * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ═══════ ATTRIBUTION TAB ═══════ */}
          {tab === "attribution" && (
            <div className="space-y-4">
              {/* Strategy Attribution */}
              {result.strategy_attribution && result.strategy_attribution.length > 0 && (
                <div>
                  <h3 className="text-xs font-medium text-gray-300 mb-2">Strategy Attribution</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-gray-500 border-b border-gray-800">
                          <th className="text-left py-1 pr-4">Strategy</th>
                          <th className="text-right py-1 pr-4">P&L</th>
                          <th className="text-right py-1 pr-4">% of Total</th>
                          <th className="text-right py-1 pr-4">Trades</th>
                          <th className="text-right py-1">Win Rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.strategy_attribution.map(s => (
                          <tr key={s.strategy} className="border-b border-gray-800/40">
                            <td className="py-1.5 pr-4 text-gray-200 font-mono text-xs">{s.strategy}</td>
                            <td className={clsx("py-1.5 pr-4 text-right", s.pnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                              {fmtPnl(s.pnl)}
                            </td>
                            <td className="py-1.5 pr-4 text-right text-gray-300">{fmtPct(s.pct)}</td>
                            <td className="py-1.5 pr-4 text-right text-gray-300">{s.trades}</td>
                            <td className="py-1.5 text-right text-gray-300">{fmtPct(s.win_rate)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Strategy P&L bar chart */}
                  <div className="mt-3 space-y-1">
                    {result.strategy_attribution.map(s => {
                      const maxPnl = Math.max(...result.strategy_attribution!.map(x => Math.abs(x.pnl)), 1);
                      const width = Math.abs(s.pnl) / maxPnl * 100;
                      return (
                        <div key={s.strategy} className="flex items-center gap-2 text-[10px]">
                          <span className="w-24 text-gray-400 truncate">{s.strategy}</span>
                          <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden relative">
                            <div
                              className={clsx("h-full rounded-full", s.pnl >= 0 ? "bg-emerald-500/60" : "bg-red-500/60")}
                              style={{ width: `${width}%` }}
                            />
                          </div>
                          <span className={clsx("w-16 text-right", s.pnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                            {fmtPnl(s.pnl)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Top Winners & Losers */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {result.top_winners && result.top_winners.length > 0 && (
                  <div>
                    <h3 className="text-xs font-medium text-emerald-400 mb-2">🏆 Top Winners</h3>
                    {result.top_winners.map(s => (
                      <div key={s.symbol} className="flex justify-between items-center py-1 border-b border-gray-800/30">
                        <span className="text-xs text-gray-200 font-mono">{s.symbol}</span>
                        <span className="text-xs text-emerald-400">{fmtPnl(s.pnl)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {result.top_losers && result.top_losers.length > 0 && (
                  <div>
                    <h3 className="text-xs font-medium text-red-400 mb-2">📉 Top Losers</h3>
                    {result.top_losers.map(s => (
                      <div key={s.symbol} className="flex justify-between items-center py-1 border-b border-gray-800/30">
                        <span className="text-xs text-gray-200 font-mono">{s.symbol}</span>
                        <span className="text-xs text-red-400">{fmtPnl(s.pnl)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Strategy Fills */}
              {result.strategy_fills?.length > 0 && (
                <div>
                  <h3 className="text-xs font-medium text-gray-300 mb-2">Strategy Fill Count</h3>
                  <div className="flex flex-wrap gap-2">
                    {result.strategy_fills.map(sf => (
                      <span key={sf.strategy} className="px-2 py-1 rounded bg-gray-800 text-[10px] text-gray-300">
                        <span className="text-indigo-400 font-medium">{sf.strategy}</span>: {sf.fills} fills ({sf.buys}B/{sf.sells}S)
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ═══════ TRADES TAB ═══════ */}
          {tab === "trades" && result.trades?.length > 0 && (
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

"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { Play, Loader2, AlertCircle, Dice5, TrendingUp, TrendingDown, Percent } from "lucide-react";
import { startMonteCarlo, getMonteCarlo, pollJob } from "@/lib/api";
import type { MonteCarloRequest, MonteCarloResult } from "@/lib/api";
import { ProgressBar } from "./ProgressBar";

const DEFAULT_SYMBOLS = "sh600519,sh600036,sz000858";

function fmt2(n?: number) { return n == null ? "—" : n.toFixed(2); }
function fmtPct(n?: number) { return n == null ? "—" : `${(n * 100).toFixed(2)}%`; }

export default function MonteCarloPanel() {
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS);
  const [lookback, setLookback] = useState(365);
  const [cash, setCash] = useState(1_000_000);
  const [nSims, setNSims] = useState(5000);
  const [result, setResult] = useState<MonteCarloResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState("");
  const [pct, setPct] = useState(0);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setStep("Starting…");
    setPct(0);
    try {
      const req: MonteCarloRequest = {
        symbols: symbols.split(",").map(s => s.trim()).filter(Boolean),
        lookback_days: lookback,
        initial_cash: cash,
        n_simulations: nSims,
        stop_loss_pct: -0.08,
      };
      const init = await startMonteCarlo(req);
      const final = await pollJob(
        init.job_id,
        getMonteCarlo,
        undefined,
        2000,
        (d) => { setStep(d.step ?? ""); setPct(d.pct ?? 0); },
      );
      setResult(final);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [symbols, lookback, cash, nSims]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      <div className="flex items-center gap-2">
        <Dice5 className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold text-gray-100">Monte Carlo Simulation</h2>
        <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
          {nSims.toLocaleString()} Simulations
        </span>
      </div>

      <p className="text-xs text-gray-500">
        Bootstrap resamples your strategy&apos;s trade-level P&L to build a probability distribution
        of possible outcomes. Reveals the true range of expected returns and tail risk.
      </p>

      {/* Controls */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="col-span-2 sm:col-span-4">
          <label className="block text-xs text-gray-400 mb-1">Symbols</label>
          <input className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={symbols} onChange={e => setSymbols(e.target.value)} />
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
          <label className="block text-xs text-gray-400 mb-1"># Simulations</label>
          <input type="number" min={100} max={50000} step={1000}
            className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={nSims} onChange={e => setNSims(Number(e.target.value))} />
        </div>
      </div>

      <button onClick={run} disabled={loading}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium text-white transition-colors">
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        {loading ? "Simulating…" : "Run Monte Carlo"}
      </button>

      {loading && (
        <div className="space-y-1">
          <ProgressBar value={pct} />
          <p className="text-xs text-gray-400">{step}</p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />{error}
        </div>
      )}

      {result && result.status === "done" && (
        <div className="space-y-5">
          {/* Headline: probability of profit */}
          <div className={clsx(
            "text-center p-6 rounded-lg border",
            (result.prob_profit ?? 0) >= 0.5
              ? "bg-emerald-900/20 border-emerald-700/40"
              : "bg-red-900/20 border-red-700/40"
          )}>
            <div className="text-3xl font-bold tracking-tight">
              <span className={(result.prob_profit ?? 0) >= 0.5 ? "text-emerald-400" : "text-red-400"}>
                {fmtPct(result.prob_profit)}
              </span>
            </div>
            <div className="text-xs text-gray-400 mt-1">Probability of Profit</div>
            <div className="text-[10px] text-gray-600 mt-0.5">
              P(Sharpe &gt; 1.0) = {fmtPct(result.prob_sharpe_above_1)}
            </div>
          </div>

          {/* Stat cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Mean Return", value: fmtPct(result.mean_total_return), icon: <TrendingUp className="w-3.5 h-3.5" />, color: (result.mean_total_return ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
              { label: "Median Return", value: fmtPct(result.median_total_return), icon: <Percent className="w-3.5 h-3.5" />, color: (result.median_total_return ?? 0) >= 0 ? "text-emerald-400" : "text-red-400" },
              { label: "VaR (95%)", value: fmtPct(result.var_95), icon: <TrendingDown className="w-3.5 h-3.5" />, color: "text-orange-400" },
              { label: "CVaR (95%)", value: fmtPct(result.cvar_95), icon: <TrendingDown className="w-3.5 h-3.5" />, color: "text-red-400" },
            ].map(m => (
              <div key={m.label} className="bg-gray-800/60 rounded-lg p-3">
                <div className="flex items-center gap-1 text-[10px] text-gray-500 uppercase tracking-wider mb-1">
                  {m.icon}{m.label}
                </div>
                <div className={clsx("text-sm font-semibold", m.color)}>{m.value}</div>
              </div>
            ))}
          </div>

          {/* Confidence intervals */}
          <div className="bg-gray-800/40 rounded-lg p-4">
            <h3 className="text-xs font-medium text-gray-300 mb-3">Return Distribution Percentiles</h3>
            <div className="flex items-center gap-1 h-8 relative">
              {/* Visual bar showing the range */}
              <div className="absolute inset-0 flex items-center">
                {/* 5-95 range */}
                <div className="h-2 bg-gray-700 rounded-full w-full relative">
                  {/* 25-75 range (IQR) */}
                  <div className="absolute h-full bg-indigo-600/40 rounded-full"
                    style={{
                      left: "25%",
                      width: "50%",
                    }}
                  />
                  {/* Median line */}
                  <div className="absolute h-4 w-0.5 bg-indigo-400 -top-1"
                    style={{ left: "50%" }}
                  />
                </div>
              </div>
            </div>
            <div className="flex justify-between text-[9px] text-gray-500 mt-2">
              <span>P5: {fmtPct(result.percentile_5)}</span>
              <span>P25: {fmtPct(result.percentile_25)}</span>
              <span className="text-indigo-400 font-medium">Median: {fmtPct(result.median_total_return)}</span>
              <span>P75: {fmtPct(result.percentile_75)}</span>
              <span>P95: {fmtPct(result.percentile_95)}</span>
            </div>
          </div>

          {/* Histogram */}
          {result.distribution_histogram && result.distribution_histogram.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-gray-300 mb-2">Return Distribution</h3>
              <div className="flex items-end gap-px h-32">
                {(() => {
                  const maxCount = Math.max(...result.distribution_histogram!.map(b => b.count), 1);
                  return result.distribution_histogram!.map((bin, i) => {
                    const height = (bin.count / maxCount) * 100;
                    const midpoint = (bin.bin_start + bin.bin_end) / 2;
                    const isPositive = midpoint >= 0;
                    return (
                      <div
                        key={i}
                        className={clsx(
                          "flex-1 rounded-t transition-all",
                          isPositive ? "bg-emerald-500/50 hover:bg-emerald-500/70" : "bg-red-500/50 hover:bg-red-500/70"
                        )}
                        style={{ height: `${Math.max(height, 1)}%` }}
                        title={`${(midpoint * 100).toFixed(1)}%: ${bin.count} sims`}
                      />
                    );
                  });
                })()}
              </div>
              <div className="flex justify-between text-[8px] text-gray-600 mt-1">
                <span>{(result.distribution_histogram![0].bin_start * 100).toFixed(0)}%</span>
                <span>0%</span>
                <span>{(result.distribution_histogram![result.distribution_histogram!.length - 1].bin_end * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}

          {/* Volatility of returns */}
          <div className="text-center text-xs text-gray-500">
            σ(returns) = {fmtPct(result.std_total_return)} across {nSims.toLocaleString()} bootstrap simulations
          </div>
        </div>
      )}
    </div>
  );
}

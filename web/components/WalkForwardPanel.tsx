"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { Play, Loader2, AlertCircle, Shield, CheckCircle2, XCircle } from "lucide-react";
import { startWalkForward, getWalkForward, pollJob } from "@/lib/api";
import type { WalkForwardRequest, WalkForwardResult } from "@/lib/api";
import { ProgressBar } from "./ProgressBar";

const DEFAULT_SYMBOLS = "sh600519,sh600036,sz000858";

function fmt3(n?: number) { return n == null ? "—" : n.toFixed(3); }
function fmtPct(n?: number) { return n == null ? "—" : `${(n * 100).toFixed(2)}%`; }

export default function WalkForwardPanel() {
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS);
  const [lookback, setLookback] = useState(730);
  const [cash, setCash] = useState(1_000_000);
  const [nSplits, setNSplits] = useState(5);
  const [trainRatio, setTrainRatio] = useState(0.7);
  const [result, setResult] = useState<WalkForwardResult | null>(null);
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
      const req: WalkForwardRequest = {
        symbols: symbols.split(",").map(s => s.trim()).filter(Boolean),
        lookback_days: lookback,
        initial_cash: cash,
        n_splits: nSplits,
        train_ratio: trainRatio,
        stop_loss_pct: -0.08,
      };
      const init = await startWalkForward(req);
      const final = await pollJob(
        init.job_id,
        getWalkForward,
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
  }, [symbols, lookback, cash, nSplits, trainRatio]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      <div className="flex items-center gap-2">
        <Shield className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-semibold text-gray-100">Walk-Forward Validation</h2>
        <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
          Out-of-Sample Test
        </span>
      </div>

      <p className="text-xs text-gray-500">
        Splits historical data into rolling train/test windows to measure out-of-sample Sharpe
        degradation and strategy robustness. A strategy is considered robust if OOS Sharpe &gt; 0.3,
        degradation &lt; 50%, and consistency ≥ 60%.
      </p>

      {/* Controls */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div className="col-span-2 sm:col-span-3">
          <label className="block text-xs text-gray-400 mb-1">Symbols</label>
          <input className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={symbols} onChange={e => setSymbols(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Total Lookback (days)</label>
          <input type="number" className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={lookback} onChange={e => setLookback(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1"># Splits</label>
          <input type="number" min={2} max={10} className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={nSplits} onChange={e => setNSplits(Number(e.target.value))} />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Train Ratio</label>
          <input type="number" step={0.05} min={0.5} max={0.9} className="w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
            value={trainRatio} onChange={e => setTrainRatio(Number(e.target.value))} />
        </div>
      </div>

      <button onClick={run} disabled={loading}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium text-white transition-colors">
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        {loading ? "Validating…" : "Run Walk-Forward"}
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
          {/* Robustness verdict */}
          <div className={clsx(
            "flex items-center gap-3 p-4 rounded-lg border",
            result.is_robust
              ? "bg-emerald-900/20 border-emerald-700/40"
              : "bg-red-900/20 border-red-700/40"
          )}>
            {result.is_robust
              ? <CheckCircle2 className="w-6 h-6 text-emerald-400" />
              : <XCircle className="w-6 h-6 text-red-400" />
            }
            <div>
              <div className={clsx("text-sm font-semibold", result.is_robust ? "text-emerald-300" : "text-red-300")}>
                {result.is_robust ? "✅ Strategy is ROBUST" : "⚠️ Strategy NOT robust"}
              </div>
              <div className="text-xs text-gray-400 mt-0.5">
                OOS Sharpe: {fmt3(result.aggregate_oos_sharpe)} | Degradation: {fmtPct(result.sharpe_degradation)} | Consistency: {fmtPct(result.consistency_score)}
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "OOS Sharpe", value: fmt3(result.aggregate_oos_sharpe) },
              { label: "Sharpe Degradation", value: fmtPct(result.sharpe_degradation) },
              { label: "Consistency", value: fmtPct(result.consistency_score) },
              { label: "Stability", value: fmtPct(result.stability_score) },
            ].map(m => (
              <div key={m.label} className="bg-gray-800/60 rounded-lg p-3">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{m.label}</div>
                <div className="text-sm font-semibold text-gray-100">{m.value}</div>
              </div>
            ))}
          </div>

          {/* Folds table */}
          {result.folds.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800">
                    <th className="text-left py-1 pr-3">Fold</th>
                    <th className="text-right py-1 pr-3">IS Sharpe</th>
                    <th className="text-right py-1 pr-3">OOS Sharpe</th>
                    <th className="text-right py-1 pr-3">OOS Return</th>
                    <th className="text-right py-1 pr-3">OOS Max DD</th>
                    <th className="text-right py-1">OOS Win Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {result.folds.map(f => (
                    <tr key={f.fold} className="border-b border-gray-800/40">
                      <td className="py-1.5 pr-3 text-gray-200">Fold {f.fold}</td>
                      <td className="py-1.5 pr-3 text-right text-gray-300">{f.in_sample_sharpe.toFixed(3)}</td>
                      <td className={clsx("py-1.5 pr-3 text-right", f.out_of_sample_sharpe > 0 ? "text-emerald-400" : "text-red-400")}>
                        {f.out_of_sample_sharpe.toFixed(3)}
                      </td>
                      <td className={clsx("py-1.5 pr-3 text-right", f.oos_return >= 0 ? "text-emerald-400" : "text-red-400")}>
                        {(f.oos_return * 100).toFixed(2)}%
                      </td>
                      <td className="py-1.5 pr-3 text-right text-orange-400">{(f.oos_max_dd * 100).toFixed(2)}%</td>
                      <td className="py-1.5 text-right text-gray-300">{(f.oos_win_rate * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Visual fold comparison */}
          {result.folds.length > 0 && (
            <div className="space-y-1.5">
              <h3 className="text-xs font-medium text-gray-400">In-Sample vs Out-of-Sample Sharpe</h3>
              {result.folds.map(f => {
                const maxS = Math.max(...result.folds.map(x => Math.max(Math.abs(x.in_sample_sharpe), Math.abs(x.out_of_sample_sharpe))), 0.1);
                return (
                  <div key={f.fold} className="flex items-center gap-2 text-[10px]">
                    <span className="w-12 text-gray-500">F{f.fold}</span>
                    <div className="flex-1 flex gap-1">
                      <div className="h-3 bg-blue-500/40 rounded" style={{ width: `${Math.abs(f.in_sample_sharpe) / maxS * 50}%` }} title={`IS: ${f.in_sample_sharpe.toFixed(3)}`} />
                      <div className={clsx("h-3 rounded", f.out_of_sample_sharpe > 0 ? "bg-emerald-500/40" : "bg-red-500/40")}
                        style={{ width: `${Math.abs(f.out_of_sample_sharpe) / maxS * 50}%` }} title={`OOS: ${f.out_of_sample_sharpe.toFixed(3)}`} />
                    </div>
                    <span className="w-20 text-right text-gray-400">
                      <span className="text-blue-400">{f.in_sample_sharpe.toFixed(2)}</span> / <span className={f.out_of_sample_sharpe > 0 ? "text-emerald-400" : "text-red-400"}>{f.out_of_sample_sharpe.toFixed(2)}</span>
                    </span>
                  </div>
                );
              })}
              <div className="flex gap-4 mt-1 text-[9px] text-gray-600">
                <span><span className="inline-block w-2 h-2 rounded bg-blue-500/40 mr-1" />In-Sample</span>
                <span><span className="inline-block w-2 h-2 rounded bg-emerald-500/40 mr-1" />Out-of-Sample</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

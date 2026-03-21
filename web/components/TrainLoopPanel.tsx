"use client";

import { useState } from "react";
import clsx from "clsx";
import { Loader2, Play, RefreshCw, CheckCircle2, AlertCircle, Zap, TrendingUp, RotateCcw } from "lucide-react";
import {
  startTrainLoop,
  getTrainLoop,
  pollJob,
  type TrainLoopRequest,
  type TrainLoopResult,
  type TrainTrialRow,
} from "@/lib/api";
import { ProgressBar } from "./ProgressBar";

// ── helpers ───────────────────────────────────────────────────────────────────

function fmtPct(v?: number) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtPnl(v?: number) {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}¥${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

function fmtPF(v?: number) {
  if (v == null) return "—";
  return v.toFixed(3);
}

function PassBadge({ passes }: { passes: boolean }) {
  return (
    <span className={clsx(
      "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold",
      passes
        ? "bg-emerald-900/60 text-emerald-300 border border-emerald-700"
        : "bg-gray-800 text-gray-500 border border-gray-700"
    )}>
      {passes ? <CheckCircle2 className="w-2.5 h-2.5" /> : <AlertCircle className="w-2.5 h-2.5" />}
      {passes ? "PASS" : "fail"}
    </span>
  );
}

// ── Ladder info table ─────────────────────────────────────────────────────────

const LADDER = [
  { r: 1, lookback: "180d", train: "1yr",  grid: "fast", stocks: "×1" },
  { r: 2, lookback: "360d", train: "1yr",  grid: "full", stocks: "×1" },
  { r: 3, lookback: "360d", train: "2yr",  grid: "full", stocks: "×1" },
  { r: 4, lookback: "365d", train: "2yr",  grid: "full", stocks: "×2" },
  { r: 5, lookback: "540d", train: "3yr",  grid: "full", stocks: "×3" },
];

// ── Main component ────────────────────────────────────────────────────────────

export default function TrainLoopPanel() {
  // Config
  const [topN,        setTopN]        = useState(3);
  const [maxRounds,   setMaxRounds]   = useState(1);
  const [symbolInput, setSymbolInput] = useState("");

  // Job state
  const [loading,  setLoading]  = useState(false);
  const [progress, setProgress] = useState<{ pct: number; step: string } | null>(null);
  const [result,   setResult]   = useState<TrainLoopResult | null>(null);
  const [error,    setError]    = useState("");

  async function run() {
    setLoading(true);
    setResult(null);
    setError("");
    setProgress({ pct: 1, step: "Submitting…" });

    const req: TrainLoopRequest = {
      top_n:      topN,
      max_rounds: maxRounds,
    };
    const rawSyms = symbolInput.split(",").map((s) => s.trim()).filter(Boolean);
    if (rawSyms.length) req.symbols = rawSyms;

    try {
      const job = await startTrainLoop(req);

      const done = await pollJob(
        job.job_id,
        getTrainLoop,
        undefined,           // no plain-text status needed
        2000,
        (data) => {          // onUpdate — live pct + step
          if (data.pct != null) {
            setProgress({ pct: data.pct, step: data.step ?? "" });
          }
        },
      );

      setResult(done);
      setProgress({ pct: 100, step: done.step ?? "Done" });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setProgress(null);
    } finally {
      setLoading(false);
    }
  }

  const trials = result?.all_trials ?? [];
  const passing = trials.filter((t) => t.passes);

  return (
    <div className="space-y-6">

      {/* ── Header explainer ── */}
      <div className="bg-sky-950/20 border border-sky-900/40 rounded-xl p-4 text-sm text-gray-400 space-y-1">
        <p className="text-sky-300 font-semibold flex items-center gap-2">
          <RotateCcw className="w-4 h-4" /> Self-Test Optimizer — auto-escalating retry loop
        </p>
        <p className="text-xs">
          Screens live CSI stocks, runs a parameter grid, and saves the best config to{" "}
          <code className="text-sky-300">best_params.json</code>. If it doesn’t pass, it
          automatically escalates: wider backtest window, more training history, more stocks,
          and a larger parameter grid — no manual re-run needed.
        </p>
        <p className="text-[11px] text-gray-600">
          PASS: Profit Factor ≥ 1.30 · Win Rate ≥ 33% · PnL &gt; 0 · ≥ 6 trades · Sharpe ≥ 0.15
        </p>
      </div>

      {/* ── Escalation ladder ── */}
      <div className="bg-gray-900 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <h2 className="text-xs font-semibold text-sky-400 uppercase tracking-wider">Escalation Ladder</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-gray-500">
            <thead>
              <tr className="border-b border-gray-800 text-gray-600">
                <th className="text-left px-4 py-2">Round</th>
                <th className="text-left px-3 py-2">Test window</th>
                <th className="text-left px-3 py-2">ML warm-up</th>
                <th className="text-left px-3 py-2">Grid</th>
                <th className="text-left px-3 py-2">Stocks</th>
                <th className="text-left px-3 py-2">Screener nudge</th>
              </tr>
            </thead>
            <tbody>
              {LADDER.filter((l) => l.r <= maxRounds).map((l) => (
                <tr key={l.r} className="border-b border-gray-800/40">
                  <td className="px-4 py-2 font-mono text-gray-300">{l.r}</td>
                  <td className="px-3 py-2 font-mono">{l.lookback}</td>
                  <td className="px-3 py-2 font-mono">{l.train}</td>
                  <td className="px-3 py-2 font-mono">{l.grid}</td>
                  <td className="px-3 py-2 font-mono">top-N{l.stocks}</td>
                  <td className="px-3 py-2 text-[10px] text-gray-600">
                    {l.r === 1 ? "\u2014" : `×${(1 + (l.r - 2) * 0.5).toFixed(1)} safety nudge`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Configuration ── */}
      <div className="bg-gray-900 rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-semibold text-sky-400 uppercase tracking-wider">Configuration</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
              Top N stocks <span className="normal-case text-gray-600">(round 1; auto-expands)</span>
            </label>
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              disabled={loading}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500 disabled:opacity-50"
            >
              {[1, 2, 3, 5, 8, 10].map((v) => (
                <option key={v} value={v}>{v} stocks</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
              Max rounds
            </label>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((r) => (
                <button
                  key={r}
                  onClick={() => setMaxRounds(r)}
                  disabled={loading}
                  className={clsx(
                    "flex-1 py-2 rounded-lg text-xs font-medium border transition-colors disabled:opacity-50",
                    maxRounds === r
                      ? "bg-sky-700 border-sky-600 text-white"
                      : "bg-gray-800 border-gray-700 text-gray-400 hover:text-white"
                  )}
                >
                  {r}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-gray-600 mt-1">
              {maxRounds === 1 ? "Quick (180d / 1yr / fast grid)" :
               maxRounds <= 3 ? "Moderate — stops once it passes" :
               "Full self-test — escalates to max data if needed"}
            </p>
          </div>
        </div>

        {/* Optional symbol override */}
        <div>
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
            Symbol override <span className="normal-case text-gray-600">(optional — skip screener)</span>
          </label>
          <input
            value={symbolInput}
            onChange={(e) => setSymbolInput(e.target.value)}
            disabled={loading}
            placeholder="e.g. sh600522, sz002648 — leave blank to use live screener"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500 disabled:opacity-50"
          />
        </div>

        {/* Run button */}
        <button
          onClick={run}
          disabled={loading}
          className="flex items-center gap-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded-lg px-6 py-2.5 text-sm font-medium transition-colors"
        >
          {loading
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : <Zap className="w-4 h-4" />}
          {loading ? "Training…" : "Run Auto-Optimizer"}
        </button>
      </div>

      {/* ── Live progress ── */}
      {(loading || progress) && (
        <div className="bg-gray-900 rounded-xl p-5">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Progress</h2>
          <ProgressBar
            pct={progress?.pct ?? 0}
            step={progress?.step}
            running={loading}
          />
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="flex items-start gap-2 p-4 bg-red-950/40 border border-red-800 rounded-xl text-sm text-red-300">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {/* ── Results ── */}
      {result && result.status === "done" && (
        <div className="space-y-5">

          {/* Best params summary card */}
          <div className={clsx(
            "rounded-xl p-5 border",
            result.found_passing
              ? "bg-emerald-950/30 border-emerald-800/50"
              : "bg-amber-950/30 border-amber-800/50"
          )}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  {result.found_passing
                    ? <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    : <AlertCircle  className="w-5 h-5 text-amber-400" />}
                  <h3 className={clsx(
                    "font-semibold",
                    result.found_passing ? "text-emerald-300" : "text-amber-300"
                  )}>
                    {result.found_passing
                      ? `\u2705 Profitable config found & saved`
                      : `\u26a0\ufe0f All ${result.rounds_run ?? "?"} rounds exhausted \u2014 best applied`}
                  </h3>
                </div>
                {result.best_symbol && (
                  <p className="text-xs text-gray-400 mt-1">
                    Best: <span className="text-gray-200 font-mono">{result.best_symbol}</span>
                    {" / "}
                    <span className="text-gray-200 font-mono">{result.best_config}</span>
                  </p>
                )}
                {result.rounds_run != null && (
                  <p className="text-xs text-gray-500 mt-0.5">
                    Completed in{" "}
                    <span className="text-gray-300 font-mono">{result.rounds_run}</span>
                    {" "}{result.rounds_run === 1 ? "round" : "rounds"}
                  </p>
                )}
              </div>

              {/* Key metrics */}
              <div className="flex gap-6 text-right shrink-0">
                <div>
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">PF</p>
                  <p className={clsx(
                    "text-lg font-bold",
                    (result.best_pf ?? 0) >= 1.3 ? "text-emerald-400" : "text-amber-400"
                  )}>
                    {fmtPF(result.best_pf)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">Win Rate</p>
                  <p className={clsx(
                    "text-lg font-bold",
                    (result.best_wr ?? 0) >= 0.33 ? "text-emerald-400" : "text-amber-400"
                  )}>
                    {fmtPct(result.best_wr)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">PnL</p>
                  <p className={clsx(
                    "text-lg font-bold",
                    (result.best_pnl ?? 0) > 0 ? "text-emerald-400" : "text-rose-400"
                  )}>
                    {fmtPnl(result.best_pnl)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">Trades</p>
                  <p className="text-lg font-bold text-gray-300">
                    {result.best_trades ?? "—"}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">Sharpe</p>
                  <p className={clsx(
                    "text-lg font-bold",
                    (result.best_sharpe ?? 0) >= 0.15 ? "text-emerald-400" : "text-amber-400"
                  )}>
                    {result.best_sharpe != null ? result.best_sharpe.toFixed(3) : "—"}
                  </p>
                </div>
              </div>
            </div>

            {!result.found_passing && (
              <p className="text-xs text-amber-500/80 mt-3">
                Screener weights were nudged toward safety each round. Run again to search with the updated weights.
              </p>
            )}
          </div>

          {/* Stats bar */}
          <div className="flex flex-wrap gap-4 text-xs text-gray-400">
            <span>
              Symbols tested:{" "}
              <span className="text-gray-200 font-mono">{result.symbols_tested.join(", ") || "—"}</span>
            </span>
            <span>
              Passing configs:{" "}
              <span className={passing.length > 0 ? "text-emerald-400 font-semibold" : "text-gray-400"}>
                {passing.length} / {trials.length}
              </span>
            </span>
          </div>

          {/* Trial table */}
          {trials.length > 0 && (
            <div className="bg-gray-900 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-gray-200">
                  All Trials ({trials.length})
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-gray-400">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-600">
                      <th className="text-left px-4 py-2">Symbol</th>
                      <th className="text-left px-4 py-2">Config</th>
                      <th className="text-center px-3 py-2">Result</th>
                      <th className="text-right px-3 py-2">PF</th>
                      <th className="text-right px-3 py-2">Win Rate</th>
                      <th className="text-right px-3 py-2">Sharpe</th>
                      <th className="text-right px-3 py-2">PnL (¥)</th>
                      <th className="text-right px-3 py-2">Trades</th>
                      <th className="text-right px-3 py-2">Score</th>
                      <th className="text-right px-3 py-2">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trials.map((t, i) => (
                      <TrialRow
                        key={i}
                        t={t}
                        isBest={
                          t.symbol === result.best_symbol &&
                          t.config === result.best_config
                        }
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Trial row sub-component ───────────────────────────────────────────────────

function TrialRow({ t, isBest }: { t: TrainTrialRow; isBest: boolean }) {
  return (
    <tr className={clsx(
      "border-b border-gray-800/50 transition-colors",
      isBest
        ? "bg-sky-950/30 text-sky-200"
        : t.passes
        ? "bg-emerald-950/20"
        : "hover:bg-gray-800/30"
    )}>
      <td className="px-4 py-2 font-mono font-semibold">{t.symbol}</td>
      <td className="px-4 py-2 font-mono text-gray-300">
        {t.config}
        {isBest && (
          <span className="ml-1.5 text-[10px] text-sky-400 border border-sky-700 rounded px-1">best</span>
        )}
      </td>
      <td className="px-3 py-2 text-center">
        {t.error
          ? <span className="text-rose-500 text-[10px]">error</span>
          : <PassBadge passes={t.passes} />}
      </td>
      <td className={clsx(
        "px-3 py-2 text-right font-mono",
        (t.profit_factor ?? 0) >= 1.3 ? "text-emerald-400" : "text-gray-400"
      )}>
        {t.profit_factor != null ? t.profit_factor.toFixed(3) : "—"}
      </td>
      <td className={clsx(
        "px-3 py-2 text-right font-mono",
        (t.win_rate ?? 0) >= 0.33 ? "text-emerald-400" : "text-gray-400"
      )}>
        {t.win_rate != null ? `${(t.win_rate * 100).toFixed(1)}%` : "—"}
      </td>
      <td className={clsx(
        "px-3 py-2 text-right font-mono",
        (t.sharpe_ratio ?? 0) >= 0.15 ? "text-emerald-400" : "text-gray-400"
      )}>
        {t.sharpe_ratio != null ? t.sharpe_ratio.toFixed(3) : "—"}
      </td>
      <td className={clsx(
        "px-3 py-2 text-right font-mono",
        t.total_pnl == null ? "text-gray-500"
          : t.total_pnl > 0 ? "text-emerald-400"
          : "text-rose-400"
      )}>
        {t.total_pnl != null
          ? `${t.total_pnl >= 0 ? "+" : ""}${t.total_pnl.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
          : "—"}
      </td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">
        {t.num_trades ?? "—"}
      </td>
      <td className={clsx(
        "px-3 py-2 text-right font-mono",
        t.score >= 0 ? "text-gray-300" : "text-gray-600"
      )}>
        {t.score.toFixed(2)}
      </td>
      <td className="px-3 py-2 text-right text-gray-500">
        {t.elapsed_s != null ? `${t.elapsed_s}s` : "—"}
      </td>
    </tr>
  );
}

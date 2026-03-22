"use client";

import { useState } from "react";
import clsx from "clsx";
import {
  Loader2, Play, CheckCircle2, AlertCircle, Brain, Zap, TrendingUp,
  ChevronDown, ChevronRight, ArrowRight, RotateCcw, Sliders, Target,
  Activity, BarChart3
} from "lucide-react";
import {
  startAutoTune, getAutoTune, pollJob,
  type AutoTuneRequest,
  type AutoTuneResult,
  type AutoTuneIteration,
  type AutoTuneAdjustment,
  type AutoTuneIterAnalysis,
} from "@/lib/api";
import { ProgressBar } from "./ProgressBar";

// ── helpers ───────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 2) {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function fmtPct(v: number | null | undefined) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtPF(v: number | null | undefined) {
  if (v == null) return "—";
  return v.toFixed(3);
}

function fmtPnl(v: number | null | undefined) {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}¥${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

// ── Score gauge ───────────────────────────────────────────────────────────────

function ScoreGauge({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(100, score));
  const isGood   = clamped >= 70;
  const isMid    = clamped >= 40;
  const color    = isGood ? "text-emerald-400" : isMid ? "text-amber-400" : "text-red-400";
  const bgTrack  = isGood ? "bg-emerald-900/40" : isMid ? "bg-amber-900/40" : "bg-red-900/40";
  const barColor = isGood ? "bg-emerald-500"    : isMid ? "bg-amber-500"    : "bg-red-500";
  return (
    <div className="flex items-center gap-3 min-w-[120px]">
      <div className={clsx("text-2xl font-extrabold tabular-nums leading-none", color)}>
        {clamped.toFixed(0)}
      </div>
      <div className="flex-1 space-y-1">
        <div className="text-[10px] text-gray-600 uppercase">/ 100</div>
        <div className={clsx("h-1.5 rounded-full", bgTrack)}>
          <div
            className={clsx("h-full rounded-full transition-all", barColor)}
            style={{ width: `${clamped}%` }}
          />
        </div>
      </div>
    </div>
  );
}

// ── Signal badge ──────────────────────────────────────────────────────────────

function SignalBadge({ signal }: { signal?: string }) {
  const s = (signal ?? "HOLD").toUpperCase();
  const style =
    s === "BUY"  ? "bg-emerald-900/60 text-emerald-300 border-emerald-700" :
    s === "SELL" ? "bg-red-900/60    text-red-300    border-red-700"    :
                   "bg-gray-800      text-gray-500   border-gray-700";
  return (
    <span className={clsx(
      "inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase border",
      style
    )}>
      {s}
    </span>
  );
}

// ── Adjustment row ────────────────────────────────────────────────────────────

function AdjustmentRow({ adj }: { adj: AutoTuneAdjustment }) {
  const typeColor =
    adj.type === "model_hyperparams" ? "text-sky-400"    :
    adj.type === "signal_params"     ? "text-violet-400" :
    adj.type === "screener_weights"  ? "text-amber-400"  :
    "text-gray-400";
  const typeLabel =
    adj.type === "model_hyperparams" ? "Model"    :
    adj.type === "signal_params"     ? "Signal"   :
    adj.type === "screener_weights"  ? "Screener" :
    adj.type;
  return (
    <div className="flex items-start gap-2 text-xs">
      <span className={clsx("shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold border", typeColor,
        adj.type === "model_hyperparams" ? "border-sky-800 bg-sky-950/30" :
        adj.type === "signal_params"     ? "border-violet-800 bg-violet-950/30" :
        "border-amber-800 bg-amber-950/30"
      )}>
        {typeLabel}
      </span>
      <div className="flex-1 min-w-0">
        <span className="font-mono text-gray-300">{adj.param}</span>
        {adj.old != null && adj.new != null && (
          <span className="text-gray-500 ml-2">
            <span className="text-red-400/80">{String(adj.old)}</span>
            {" "}<ArrowRight className="inline w-2.5 h-2.5" />{" "}
            <span className="text-emerald-400/80">{String(adj.new)}</span>
          </span>
        )}
        <p className="text-gray-600 text-[10px] mt-0.5 truncate" title={adj.reason}>{adj.reason}</p>
      </div>
    </div>
  );
}

// ── Analysis chip ─────────────────────────────────────────────────────────────

function AnalysisChip({ a }: { a: AutoTuneIterAnalysis }) {
  const oos = a.oos_accuracy;
  const oosOk = oos != null && oos >= 0.54;
  const confOk = (a.confidence ?? 0) >= 0.48;
  return (
    <div className={clsx(
      "rounded-lg p-3 border space-y-1.5",
      oosOk && confOk ? "border-emerald-800/60 bg-emerald-950/20" :
                        "border-gray-700/60 bg-gray-900/60"
    )}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono font-bold text-sm text-gray-200">{a.symbol}</span>
        <SignalBadge signal={a.signal} />
      </div>
      {a.error ? (
        <p className="text-[10px] text-red-400">{a.error}</p>
      ) : (
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px]">
          <div className="flex justify-between">
            <span className="text-gray-600">OOS</span>
            <span className={oosOk ? "text-emerald-400" : "text-red-400"}>
              {fmtPct(oos)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Conf</span>
            <span className={confOk ? "text-emerald-400" : "text-amber-400"}>
              {fmtPct(a.confidence)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">P(buy)</span>
            <span className="text-gray-400">{fmtPct(a.p_buy)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">P(hold)</span>
            <span className="text-gray-400">{fmtPct(a.p_hold)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Iteration card ────────────────────────────────────────────────────────────

function IterationCard({
  iter,
  isLast,
  converged,
}: {
  iter: AutoTuneIteration;
  isLast: boolean;
  converged: boolean;
}) {
  const [open, setOpen] = useState(isLast); // expand the last one by default
  const ok = iter.score >= 70;
  const mid = iter.score >= 40;

  const bt = iter.backtest ?? {};

  return (
    <div className={clsx(
      "rounded-xl border overflow-hidden transition-all",
      ok  ? "border-emerald-800/60 bg-emerald-950/10" :
      mid ? "border-amber-800/60  bg-amber-950/10"   :
            "border-gray-800      bg-gray-900/40"
    )}>
      {/* header — always visible */}
      <button
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/5 transition-colors"
        onClick={() => setOpen((v) => !v)}
      >
        {/* iteration number */}
        <div className={clsx(
          "shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold",
          ok  ? "bg-emerald-700 text-white" :
          mid ? "bg-amber-700  text-white"  :
                "bg-gray-700   text-gray-300"
        )}>
          {iter.iteration}
        </div>

        {/* status indicator */}
        <div className="shrink-0">
          {ok ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          ) : (
            <AlertCircle className="w-4 h-4 text-amber-400" />
          )}
        </div>

        {/* score */}
        <ScoreGauge score={iter.score} />

        {/* quick stats */}
        <div className="hidden sm:flex items-center gap-4 ml-2 text-[11px] text-gray-500">
          <div>BT {iter.backtest_ok ? <span className="text-emerald-400">✓</span> : <span className="text-red-400">✗</span>}</div>
          <div>ML {iter.model_ok   ? <span className="text-emerald-400">✓</span> : <span className="text-red-400">✗</span>}</div>
          {bt.profit_factor != null && <div>PF {fmtPF(bt.profit_factor)}</div>}
          {bt.num_trades    != null && <div>{bt.num_trades} trades</div>}
        </div>

        {/* converged badge */}
        {ok && isLast && converged && (
          <span className="ml-auto shrink-0 px-2.5 py-1 rounded-full bg-emerald-900 text-emerald-300 text-[10px] font-bold border border-emerald-700">
            CONVERGED ✓
          </span>
        )}

        {/* chevron */}
        <div className="ml-auto shrink-0 text-gray-600">
          {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </div>
      </button>

      {/* expanded body */}
      {open && (
        <div className="px-4 pb-4 pt-2 space-y-4 border-t border-gray-800/60">

          {/* backtest stats */}
          <div>
            <h4 className="text-[10px] font-semibold text-sky-400 uppercase tracking-wider mb-2 flex items-center gap-1">
              <BarChart3 className="w-3 h-3" /> Backtest
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-2">
              {[
                { label: "Profit Factor", value: fmtPF(bt.profit_factor), ok: (bt.profit_factor ?? 0) >= 1.30 },
                { label: "Sharpe",        value: fmt(bt.sharpe),          ok: (bt.sharpe ?? 0) >= 0.15 },
                { label: "Win Rate",      value: fmtPct(bt.win_rate),     ok: (bt.win_rate ?? 0) >= 0.33 },
                { label: "Trades",        value: String(bt.num_trades ?? "—"), ok: (bt.num_trades ?? 0) >= 5 },
                { label: "PnL",           value: fmtPnl(bt.total_pnl),   ok: (bt.total_pnl ?? 0) > 0 },
              ].map((s) => (
                <div key={s.label} className="bg-gray-900 rounded-lg p-2.5">
                  <div className="text-[10px] text-gray-600 mb-1">{s.label}</div>
                  <div className={clsx("text-sm font-bold", s.ok ? "text-emerald-400" : "text-red-400")}>
                    {s.value}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* model analyses */}
          {iter.analyses.length > 0 && (
            <div>
              <h4 className="text-[10px] font-semibold text-sky-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                <Brain className="w-3 h-3" /> Model Quality
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                {iter.analyses.map((a) => (
                  <AnalysisChip key={a.symbol} a={a} />
                ))}
              </div>
            </div>
          )}

          {/* failures */}
          {iter.failures.length > 0 && (
            <div>
              <h4 className="text-[10px] font-semibold text-red-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> Failures Detected
              </h4>
              <ul className="space-y-1">
                {iter.failures.map((f, i) => (
                  <li key={i} className="text-xs text-red-300/80 flex items-center gap-1.5">
                    <span className="text-red-600">✗</span> {f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* adjustments */}
          {iter.adjustments.length > 0 && (
            <div>
              <h4 className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                <Sliders className="w-3 h-3" /> Adjustments Applied
              </h4>
              <div className="space-y-2">
                {iter.adjustments.map((adj, i) => (
                  <AdjustmentRow key={i} adj={adj} />
                ))}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function AutoTunePanel() {
  const [topN,         setTopN]         = useState(3);
  const [maxIter,      setMaxIter]      = useState(3);
  const [symbolInput,  setSymbolInput]  = useState("");

  const [loading,   setLoading]   = useState(false);
  const [progress,  setProgress]  = useState<{ pct: number; step: string } | null>(null);
  const [result,    setResult]    = useState<AutoTuneResult | null>(null);
  const [error,     setError]     = useState("");

  async function run() {
    setLoading(true);
    setResult(null);
    setError("");
    setProgress({ pct: 1, step: "Submitting…" });

    const req: AutoTuneRequest = {
      top_n:           topN,
      max_iterations:  maxIter,
    };
    const rawSyms = symbolInput.split(",").map((s) => s.trim()).filter(Boolean);
    if (rawSyms.length) req.symbols = rawSyms;

    try {
      const job = await startAutoTune(req);

      const done = await pollJob(
        job.job_id,
        getAutoTune,
        undefined,
        3000,
        (data) => {
          if (data.pct != null) {
            setProgress({ pct: data.pct, step: data.step ?? "" });
          }
        },
      );

      setResult(done);
      setProgress({ pct: 100, step: done.converged ? "Converged ✓" : "Done (max iterations reached)" });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setProgress(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">

      {/* ── Header explainer ── */}
      <div className="bg-violet-950/20 border border-violet-900/40 rounded-xl p-4 text-sm space-y-2">
        <p className="text-violet-300 font-semibold flex items-center gap-2">
          <Brain className="w-4 h-4" /> Autonomous Auto-Tune — train → analyse → diagnose → adjust → retrain
        </p>
        <p className="text-xs text-gray-400">
          The system screens stocks, trains ML models, backtests them, then <em>reads its own quality
          metrics</em>. If OOS accuracy is low, it adjusts tree depth &amp; regularization. If profit factor
          is poor, it tightens signal thresholds. If trades are too few, it relaxes confidence filters.
          It loops until the score reaches 70 / 100 or the iteration limit is hit.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1">
          {[
            { icon: <Activity className="w-3 h-3" />, label: "Score ≥ 70", desc: "converged" },
            { icon: <Brain className="w-3 h-3" />,    label: "OOS ≥ 54%",  desc: "model quality" },
            { icon: <Target className="w-3 h-3" />,   label: "PF ≥ 1.30",  desc: "backtest pass" },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-2 text-[11px] text-gray-500">
              <span className="text-violet-400">{s.icon}</span>
              <span><span className="text-gray-300">{s.label}</span> {s.desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Config ── */}
      <div className="bg-gray-900 rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-semibold text-violet-400 uppercase tracking-wider">Configuration</h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* top_n */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
              Top N stocks
            </label>
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              disabled={loading}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-violet-500 disabled:opacity-50"
            >
              {[1,2,3,5,7,10].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>

          {/* max_iterations */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
              Max iterations
              <span className="ml-1 text-gray-600 normal-case">(each trains fresh models)</span>
            </label>
            <select
              value={maxIter}
              onChange={(e) => setMaxIter(Number(e.target.value))}
              disabled={loading}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-violet-500 disabled:opacity-50"
            >
              {[1,2,3,4,5].map((n) => (
                <option key={n} value={n}>{n} iteration{n !== 1 ? "s" : ""}</option>
              ))}
            </select>
          </div>

          {/* symbol override */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">
              Symbols override
              <span className="ml-1 text-gray-600 normal-case">(optional, comma-sep)</span>
            </label>
            <input
              type="text"
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value)}
              disabled={loading}
              placeholder="600519, 000858, …"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono placeholder-gray-700 focus:outline-none focus:border-violet-500 disabled:opacity-50"
            />
          </div>
        </div>

        {/* time estimate */}
        <p className="text-[11px] text-amber-600/80 flex items-center gap-1.5">
          <AlertCircle className="w-3 h-3 shrink-0" />
          Estimated runtime: ~{Math.ceil(maxIter * 4)}–{maxIter * 12} min
          {" "}({maxIter} iter × ~4–12 min per iteration including model training)
        </p>
      </div>

      {/* ── Run button ── */}
      <button
        onClick={run}
        disabled={loading}
        className={clsx(
          "w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm",
          "transition-all",
          loading
            ? "bg-gray-800 text-gray-500 cursor-not-allowed"
            : "bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-900/30"
        )}
      >
        {loading
          ? <><Loader2 className="w-4 h-4 animate-spin" /> Auto-Tuning…</>
          : <><Play className="w-4 h-4" /> Run Auto-Tune</>
        }
      </button>

      {/* ── Live progress ── */}
      {loading && progress && (
        <div className="space-y-2">
          <ProgressBar pct={progress.pct} />
          <p className="text-xs text-gray-500 text-center">{progress.step}</p>
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="bg-red-950/30 border border-red-800/50 rounded-xl p-4 text-sm text-red-300 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* ── Result iterations ── */}
      {result && (
        <div className="space-y-4">

          {/* summary header */}
          <div className={clsx(
            "rounded-xl p-4 border",
            result.converged
              ? "bg-emerald-950/20 border-emerald-800/60"
              : "bg-amber-950/20 border-amber-800/60"
          )}>
            <div className="flex flex-col sm:flex-row sm:items-center gap-3">
              <div>
                {result.converged
                  ? <CheckCircle2 className="w-7 h-7 text-emerald-400" />
                  : <AlertCircle  className="w-7 h-7 text-amber-400"   />
                }
              </div>
              <div className="flex-1 min-w-0">
                <p className={clsx(
                  "font-bold text-base",
                  result.converged ? "text-emerald-300" : "text-amber-300"
                )}>
                  {result.converged ? "Converged — models are tuned ✓" : "Max iterations reached"}
                </p>
                <p className="text-xs text-gray-500 mt-0.5">
                  {result.iterations_run} iteration{result.iterations_run !== 1 ? "s" : ""} ·{" "}
                  Final score {result.final_score?.toFixed(1)} / 100
                  {result.best_symbol && ` · Best symbol ${result.best_symbol}`}
                </p>
              </div>
              <ScoreGauge score={result.final_score ?? 0} />
            </div>

            {result.best_config && (
              <div className="mt-3 pt-3 border-t border-gray-800">
                <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Best config saved</p>
                <code className="text-[11px] text-sky-300 break-all">{result.best_config}</code>
              </div>
            )}
          </div>

          {/* iteration cards */}
          <div className="space-y-3">
            {result.iterations.map((iter, idx) => (
              <IterationCard
                key={iter.iteration}
                iter={iter}
                isLast={idx === result.iterations.length - 1}
                converged={result.converged}
              />
            ))}
          </div>

          {/* next steps CTA */}
          {result.converged && (
            <div className="bg-sky-950/20 border border-sky-800/50 rounded-xl p-4 text-sm text-gray-400">
              <p className="text-sky-300 font-semibold flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4" /> Next steps
              </p>
              <ul className="space-y-1 text-xs">
                <li className="flex items-center gap-1.5"><Zap className="w-3 h-3 text-sky-400 shrink-0" />
                  Switch to the <strong className="text-gray-200">Advisor</strong> tab and run a fresh analysis — models are already trained with optimal params.
                </li>
                <li className="flex items-center gap-1.5"><RotateCcw className="w-3 h-3 text-sky-400 shrink-0" />
                  Run <strong className="text-gray-200">Backtest</strong> with the auto-saved{" "}
                  <code className="text-sky-300 text-[10px]">best_params.json</code> settings.
                </li>
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

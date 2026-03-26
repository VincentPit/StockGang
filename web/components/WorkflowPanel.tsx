"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import {
  GitMerge, Play, Loader2, AlertCircle, TrendingUp, List,
  Zap, BarChart2, Microscope, ChevronDown, ChevronRight,
  Check, RotateCcw, Settings2, Info, Sparkles,
} from "lucide-react";
import {
  startScreen, getScreen, startBacktest, getBacktest, pollJob,
} from "@/lib/api";
import type {
  ScreenRequest, ScreenResult, ScreenRow,
  BacktestRequest, BacktestResult,
} from "@/lib/api";
import { ProgressBar } from "./ProgressBar";
import { NavChart } from "./NavChart";
import { TradesTable } from "./TradesTable";

// ── Formatters ────────────────────────────────────────────────────────────────

function fmt2(n?: number) { return n == null ? "—" : n.toFixed(2); }
function fmtPct(n?: number) { return n == null ? "—" : `${(n * 100).toFixed(2)}%`; }
function fmtPnl(n?: number) {
  if (n == null) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}¥${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

// ── Presets ────────────────────────────────────────────────────────────────────

type PresetKey = "quick" | "standard" | "deep";

interface PresetConfig {
  key: PresetKey;
  label: string;
  desc: string;
  Icon: typeof Zap;
  color: string;
  borderActive: string;
  bgActive: string;
  screen: Required<ScreenRequest>;
  backtest: { days: number; cash: number; commission: number; stopLoss: number };
  estimate: string;
}

const PRESETS: PresetConfig[] = [
  {
    key: "quick",
    label: "Quick Scan",
    desc: "CSI 300 · Top 3 picks · 6-month backtest",
    Icon: Zap,
    color: "text-amber-400",
    borderActive: "border-amber-500/40",
    bgActive: "bg-amber-950/20",
    screen: { top_n: 3, min_bars: 60, lookback_years: 1, indices: ["000300"] },
    backtest: { days: 180, cash: 1_000_000, commission: 0.001, stopLoss: 8 },
    estimate: "~3–5 min",
  },
  {
    key: "standard",
    label: "Standard",
    desc: "CSI 300 + 500 · Top 6 picks · 1-year backtest",
    Icon: BarChart2,
    color: "text-indigo-400",
    borderActive: "border-indigo-500/40",
    bgActive: "bg-indigo-950/20",
    screen: { top_n: 6, min_bars: 60, lookback_years: 2, indices: ["000300", "000905"] },
    backtest: { days: 365, cash: 1_000_000, commission: 0.001, stopLoss: 8 },
    estimate: "~5–10 min",
  },
  {
    key: "deep",
    label: "Deep Research",
    desc: "CSI 300 + 500 + 1000 · Top 10 picks · 1-year backtest",
    Icon: Microscope,
    color: "text-emerald-400",
    borderActive: "border-emerald-500/40",
    bgActive: "bg-emerald-950/20",
    screen: { top_n: 10, min_bars: 120, lookback_years: 2, indices: ["000300", "000905", "000852"] },
    backtest: { days: 365, cash: 1_000_000, commission: 0.001, stopLoss: 8 },
    estimate: "~10–20 min",
  },
];

// ── Phase types ───────────────────────────────────────────────────────────────

type Phase = "configure" | "screening" | "review" | "backtesting" | "results";

const STEPS = [
  { phase: "configure" as const, label: "Configure", num: 1 },
  { phase: "review" as const, label: "Screen & Select", num: 2 },
  { phase: "results" as const, label: "Backtest & Review", num: 3 },
];

function phaseToStepIndex(phase: Phase): number {
  if (phase === "configure") return 0;
  if (phase === "screening" || phase === "review") return 1;
  return 2; // backtesting | results
}

// ── Step Indicator ────────────────────────────────────────────────────────────

function StepIndicator({ phase }: { phase: Phase }) {
  const current = phaseToStepIndex(phase);
  const inProgress = phase === "screening" || phase === "backtesting";

  return (
    <div className="flex items-center gap-0">
      {STEPS.map((s, i) => {
        const done = i < current || (i === current && phase === "results");
        const active = i === current;
        return (
          <div key={s.phase} className="flex items-center">
            <div className="flex items-center gap-2">
              <div className={clsx(
                "w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-all",
                done    ? "bg-emerald-600 text-white" :
                active  ? "bg-indigo-600 text-white ring-2 ring-indigo-400/30" :
                          "bg-gray-800 text-gray-500",
                active && inProgress && "animate-pulse",
              )}>
                {done ? <Check className="w-3.5 h-3.5" /> : s.num}
              </div>
              <span className={clsx(
                "text-xs font-medium whitespace-nowrap hidden sm:inline",
                done    ? "text-emerald-400" :
                active  ? "text-indigo-300" :
                          "text-gray-600",
              )}>
                {s.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={clsx(
                "mx-3 h-px w-6 sm:w-10",
                i < current ? "bg-emerald-600" : "bg-gray-800",
              )} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Symbol Chip (toggleable) ─────────────────────────────────────────────────

function SymbolChip({
  row,
  selected,
  onToggle,
}: {
  row: ScreenRow;
  selected: boolean;
  onToggle: () => void;
}) {
  const trend = row.data_scope?.trend;
  return (
    <button
      onClick={onToggle}
      title={[
        `Score: ${row.score.toFixed(2)}`,
        `Sharpe: ${row.sharpe.toFixed(2)}`,
        `1Y: ${(row.ret_1y * 100).toFixed(1)}%`,
        trend ?? "",
      ].filter(Boolean).join(" | ")}
      className={clsx(
        "group flex items-center gap-2.5 px-3 py-2.5 rounded-lg border text-left transition-all text-sm",
        selected
          ? "bg-indigo-950/40 border-indigo-500/40 hover:border-indigo-400/60"
          : "bg-gray-900/40 border-gray-800 opacity-50 hover:opacity-75",
      )}
    >
      {/* checkbox */}
      <div className={clsx(
        "w-4 h-4 rounded border-2 flex items-center justify-center shrink-0 transition-all",
        selected
          ? "border-indigo-500 bg-indigo-600"
          : "border-gray-700 bg-transparent",
      )}>
        {selected && <Check className="w-2.5 h-2.5 text-white" />}
      </div>

      {/* info */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="font-mono font-semibold text-gray-200">{row.symbol}</span>
          <span className={clsx(
            "text-[10px] font-bold px-1 py-0.5 rounded",
            row.score >= 0.5 ? "text-emerald-400 bg-emerald-950/50" : "text-amber-400 bg-amber-950/50",
          )}>
            {row.score.toFixed(2)}
          </span>
          {trend && (
            <span className={clsx("text-[9px] px-1 rounded",
              trend === "UPTREND"   ? "text-emerald-500 bg-emerald-900/30" :
              trend === "DOWNTREND" ? "text-red-500 bg-red-900/30" :
                                     "text-yellow-500 bg-yellow-900/30",
            )}>
              {trend === "UPTREND" ? "↑" : trend === "DOWNTREND" ? "↓" : "→"}
            </span>
          )}
        </div>
        <div className="text-[10px] text-gray-500 truncate">{row.name}</div>
      </div>

      {/* metrics */}
      <div className="ml-auto text-right shrink-0">
        <div className={clsx("text-[10px] font-medium",
          row.ret_1y >= 0 ? "text-emerald-400" : "text-red-400",
        )}>
          {(row.ret_1y * 100).toFixed(1)}%
        </div>
        <div className="text-[9px] text-gray-600">S:{row.sharpe.toFixed(1)}</div>
      </div>
    </button>
  );
}

// ── Shared input class ───────────────────────────────────────────────────────

const inputClass =
  "w-full rounded bg-gray-800 border border-gray-700 px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500";

// ══════════════════════════════════════════════════════════════════════════════
//  WorkflowPanel — 3-phase stepper wizard
// ══════════════════════════════════════════════════════════════════════════════

export default function WorkflowPanel() {
  // ── Phase ──────────────────────────────────────────────────────────────────
  const [phase, setPhase] = useState<Phase>("configure");

  // ── Configure ──────────────────────────────────────────────────────────────
  const [preset, setPreset]               = useState<PresetKey>("standard");
  const [showAdvanced, setShowAdvanced]   = useState(false);
  const [topN, setTopN]                   = useState(6);
  const [minBars, setMinBars]             = useState(60);
  const [lookbackYears, setLookbackYears] = useState(2);
  const [indices, setIndices]             = useState("000300,000905");
  const [backtestDays, setBacktestDays]   = useState(365);
  const [cash, setCash]                   = useState(1_000_000);
  const [commission, setCommission]       = useState(0.001);
  const [stopLoss, setStopLoss]           = useState(8);

  // ── Screen results ─────────────────────────────────────────────────────────
  const [screenResult, setScreenResult]       = useState<ScreenResult | null>(null);
  const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set());

  // ── Backtest results ───────────────────────────────────────────────────────
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);

  // ── Shared UI ──────────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [step, setStep]       = useState("");
  const [pct, setPct]         = useState(0);

  // ── Helpers ────────────────────────────────────────────────────────────────

  const applyPreset = useCallback((key: PresetKey) => {
    setPreset(key);
    const p = PRESETS.find(pr => pr.key === key)!;
    setTopN(p.screen.top_n);
    setMinBars(p.screen.min_bars);
    setLookbackYears(p.screen.lookback_years);
    setIndices(p.screen.indices.join(","));
    setBacktestDays(p.backtest.days);
    setCash(p.backtest.cash);
    setCommission(p.backtest.commission);
    setStopLoss(p.backtest.stopLoss);
  }, []);

  const startOver = useCallback(() => {
    setPhase("configure");
    setScreenResult(null);
    setBacktestResult(null);
    setSelectedSymbols(new Set());
    setError(null);
    setStep("");
    setPct(0);
  }, []);

  const toggleSymbol = useCallback((sym: string) => {
    setSelectedSymbols(prev => {
      const next = new Set(prev);
      if (next.has(sym)) next.delete(sym);
      else next.add(sym);
      return next;
    });
  }, []);

  // ── Run screening ──────────────────────────────────────────────────────────

  const runScreen = useCallback(async () => {
    setLoading(true);
    setError(null);
    setScreenResult(null);
    setPct(0);
    setStep("Starting screener…");
    setPhase("screening");
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
      setScreenResult(final);
      // Pre-select the recommended symbols
      const recommended = new Set(
        final.rows.filter(r => r.recommended).map(r => r.symbol),
      );
      setSelectedSymbols(recommended);
      setPhase("review");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("configure");
    } finally {
      setLoading(false);
    }
  }, [topN, minBars, lookbackYears, indices]);

  // ── Run backtest ───────────────────────────────────────────────────────────

  const runBacktest = useCallback(async () => {
    setLoading(true);
    setError(null);
    setBacktestResult(null);
    setPct(0);
    setStep("Starting backtest…");
    setPhase("backtesting");
    try {
      const syms = Array.from(selectedSymbols);
      const req: BacktestRequest = {
        symbols: syms,
        lookback_days: backtestDays,
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
      setBacktestResult(final);
      setPhase("results");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("review"); // let user retry from review
    } finally {
      setLoading(false);
    }
  }, [selectedSymbols, backtestDays, cash, commission, stopLoss]);

  const selectedCount = selectedSymbols.size;
  const currentPreset = PRESETS.find(p => p.key === preset)!;

  // ══════════════════════════════════════════════════════════════════════════
  //  Render
  // ══════════════════════════════════════════════════════════════════════════

  return (
    <div className="space-y-5">

      {/* ── Header + step indicator ──────────────────────────────────────── */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-3">
          <div className="flex items-center gap-2">
            <GitMerge className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-semibold text-gray-100">Workflow Pipeline</h2>
          </div>
          <StepIndicator phase={phase} />
        </div>
        <p className="text-xs text-gray-500">
          Screen the market for top stocks, review &amp; curate your picks, then
          backtest them — all in one guided flow.
        </p>
        <p className="text-xs text-emerald-400/80 mt-2">
          ✓ Automatically uses your trained model from{" "}
          <span className="font-mono">best_params.json</span> — run Auto-Tune
          first for best results.
        </p>
      </div>

      {/* ── Error banner ──────────────────────────────────────────────────── */}
      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-800/50 bg-red-950/20 p-4 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <div>
            <p>{error}</p>
            <button
              onClick={startOver}
              className="text-xs text-red-300 underline mt-1 hover:text-red-200"
            >
              ← Back to configure
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════ PHASE 1: CONFIGURE ═══════════════════════════════ */}
      {phase === "configure" && (
        <div className="space-y-4">
          {/* Preset cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {PRESETS.map(p => {
              const active = preset === p.key;
              return (
                <button
                  key={p.key}
                  onClick={() => applyPreset(p.key)}
                  className={clsx(
                    "relative rounded-xl border p-4 text-left transition-all",
                    active
                      ? `${p.bgActive} ${p.borderActive}`
                      : "bg-gray-900/40 border-gray-800 hover:border-gray-700",
                  )}
                >
                  {active && (
                    <div className="absolute top-2.5 right-2.5">
                      <Check className={clsx("w-4 h-4", p.color)} />
                    </div>
                  )}
                  <div className="flex items-center gap-2 mb-2">
                    <p.Icon className={clsx("w-4 h-4", active ? p.color : "text-gray-600")} />
                    <span className={clsx(
                      "text-sm font-semibold",
                      active ? "text-gray-100" : "text-gray-400",
                    )}>
                      {p.label}
                    </span>
                  </div>
                  <p className={clsx("text-xs", active ? "text-gray-400" : "text-gray-600")}>
                    {p.desc}
                  </p>
                  <p className={clsx("text-[10px] mt-2", active ? "text-gray-500" : "text-gray-700")}>
                    ⏱ {p.estimate}
                  </p>
                </button>
              );
            })}
          </div>

          {/* Advanced settings */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            <Settings2 className="w-3.5 h-3.5" />
            Advanced settings
            {showAdvanced
              ? <ChevronDown className="w-3 h-3" />
              : <ChevronRight className="w-3 h-3" />}
          </button>

          {showAdvanced && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Top N symbols</label>
                  <input type="number" className={inputClass}
                    value={topN} onChange={e => setTopN(Number(e.target.value))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Indices (CSI)</label>
                  <input className={inputClass}
                    value={indices} onChange={e => setIndices(e.target.value)}
                    placeholder="000300,000905" />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Screen lookback (yrs)</label>
                  <input type="number" step="0.5" className={inputClass}
                    value={lookbackYears} onChange={e => setLookbackYears(Number(e.target.value))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Min Bars</label>
                  <input type="number" className={inputClass}
                    value={minBars} onChange={e => setMinBars(Number(e.target.value))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Backtest days</label>
                  <input type="number" className={inputClass}
                    value={backtestDays} onChange={e => setBacktestDays(Number(e.target.value))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Initial Cash</label>
                  <input type="number" className={inputClass}
                    value={cash} onChange={e => setCash(Number(e.target.value))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Commission</label>
                  <input type="number" step="0.0001" className={inputClass}
                    value={commission} onChange={e => setCommission(Number(e.target.value))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Stop Loss <span className="text-gray-500">(%, e.g. 8 = 8%)</span>
                  </label>
                  <input type="number" step="1" min="0" max="50" className={inputClass}
                    value={stopLoss} onChange={e => setStopLoss(Number(e.target.value))} />
                </div>
              </div>
            </div>
          )}

          {/* Run Screener CTA */}
          <button
            onClick={runScreen}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-sm font-medium text-white transition-colors"
          >
            <Play className="w-4 h-4" />
            Start Screening
            <span className="text-indigo-300/60 ml-1">({currentPreset.label})</span>
          </button>
        </div>
      )}

      {/* ═══════════════ SCREENING PROGRESS ═══════════════════════════════ */}
      {phase === "screening" && loading && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6 space-y-3">
          <div className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
            <span className="text-sm font-medium text-gray-200">Screening the market…</span>
          </div>
          <ProgressBar value={pct} />
          <p className="text-xs text-gray-500">{step}</p>
        </div>
      )}

      {/* ═══════════════ PHASE 2: REVIEW & SELECT ═════════════════════════ */}
      {phase === "review" && screenResult && (
        <div className="space-y-4">
          {/* Summary banner */}
          <div className="rounded-xl border border-emerald-800/40 bg-emerald-950/10 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-emerald-300">
                ✓ Screened {screenResult.rows.length} symbols
                {screenResult.universe_size
                  ? ` from ${screenResult.universe_size} candidates`
                  : ""}
              </span>
              <span className="text-xs text-gray-500">
                {selectedCount} selected for backtest
              </span>
            </div>
            <p className="text-xs text-gray-500">
              Toggle any symbol to include or exclude it from the backtest.
              Recommended picks are pre-selected.
            </p>
          </div>

          {/* Symbol grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {screenResult.rows.map(row => (
              <SymbolChip
                key={row.symbol}
                row={row}
                selected={selectedSymbols.has(row.symbol)}
                onToggle={() => toggleSymbol(row.symbol)}
              />
            ))}
          </div>

          {/* Quick selection actions */}
          <div className="flex flex-wrap gap-2 text-[10px]">
            <button
              onClick={() =>
                setSelectedSymbols(
                  new Set(screenResult.rows.filter(r => r.recommended).map(r => r.symbol)),
                )
              }
              className="px-2.5 py-1 rounded border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
            >
              Reset to recommended
            </button>
            <button
              onClick={() =>
                setSelectedSymbols(new Set(screenResult.rows.map(r => r.symbol)))
              }
              className="px-2.5 py-1 rounded border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
            >
              Select all
            </button>
            <button
              onClick={() => setSelectedSymbols(new Set())}
              className="px-2.5 py-1 rounded border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
            >
              Clear all
            </button>
          </div>

          {/* Backtest config summary */}
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div className="flex items-center gap-2 mb-2">
              <Info className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-xs font-medium text-gray-400">
                Backtest configuration
              </span>
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-gray-500">
              <span>Period: <span className="text-gray-300">{backtestDays} days</span></span>
              <span>Cash: <span className="text-gray-300">¥{cash.toLocaleString()}</span></span>
              <span>
                Commission:{" "}
                <span className="text-gray-300">{(commission * 100).toFixed(2)}%</span>
              </span>
              <span>Stop Loss: <span className="text-gray-300">{stopLoss}%</span></span>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={runBacktest}
              disabled={selectedCount === 0}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium text-white transition-colors"
            >
              <Play className="w-4 h-4" />
              Run Backtest on {selectedCount} symbol{selectedCount !== 1 ? "s" : ""}
            </button>
            <button
              onClick={startOver}
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600 text-sm transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Restart
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════ BACKTESTING PROGRESS ═════════════════════════════ */}
      {phase === "backtesting" && loading && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6 space-y-3">
          <div className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
            <span className="text-sm font-medium text-gray-200">
              Backtesting {selectedCount} symbol{selectedCount !== 1 ? "s" : ""}…
            </span>
          </div>
          <ProgressBar value={pct} />
          <p className="text-xs text-gray-500">{step}</p>
          <p className="text-[10px] text-gray-600">
            Training LGBM walk-forward models — this may take several minutes.
          </p>
        </div>
      )}

      {/* ═══════════════ PHASE 3: BACKTEST RESULTS ════════════════════════ */}
      {phase === "results" && backtestResult?.status === "done" && (
        <div className="space-y-5">
          {/* Hero result banner */}
          {(() => {
            const pnl = backtestResult.total_pnl ?? 0;
            const positive = pnl >= 0;
            return (
              <div
                className={clsx(
                  "rounded-xl border p-5",
                  positive
                    ? "border-emerald-800/50 bg-emerald-950/10"
                    : "border-red-800/50 bg-red-950/10",
                )}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p
                      className={clsx(
                        "text-2xl font-bold",
                        positive ? "text-emerald-400" : "text-red-400",
                      )}
                    >
                      {fmtPnl(backtestResult.total_pnl)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {backtestResult.period_start} → {backtestResult.period_end} ·{" "}
                      {backtestResult.num_trades ?? 0} trades ·{" "}
                      {backtestResult.symbols?.length ?? selectedCount} symbols
                    </p>
                  </div>
                  <div className="text-right">
                    <p
                      className={clsx(
                        "text-xl font-bold",
                        positive ? "text-emerald-400" : "text-red-400",
                      )}
                    >
                      {fmtPct(backtestResult.total_pnl_pct)}
                    </p>
                    <p className="text-[10px] text-gray-500 uppercase tracking-wider">
                      Return
                    </p>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Metric cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Sharpe", value: fmt2(backtestResult.sharpe), color: "text-gray-100" },
              { label: "Max DD", value: fmtPct(backtestResult.max_dd), color: "text-orange-400" },
              { label: "Win Rate", value: fmtPct(backtestResult.win_rate), color: "text-gray-100" },
              { label: "Profit Factor", value: fmt2(backtestResult.profit_factor), color: "text-gray-100" },
              { label: "Avg Win", value: fmtPnl(backtestResult.avg_win), color: "text-emerald-400" },
              { label: "Avg Loss", value: fmtPnl(backtestResult.avg_loss), color: "text-red-400" },
              { label: "# Trades", value: String(backtestResult.num_trades ?? "—"), color: "text-gray-100" },
              {
                label: "Period",
                value: `${backtestResult.period_start ?? ""} → ${backtestResult.period_end ?? ""}`,
                color: "text-gray-400",
              },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-gray-800/60 rounded-lg p-3">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">
                  {label}
                </div>
                <div className={clsx("text-sm font-semibold", color)}>{value}</div>
              </div>
            ))}
          </div>

          {/* NAV Chart */}
          {backtestResult.nav_series?.length > 0 && (
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
              <div className="flex items-center gap-1.5 mb-3">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">NAV Curve</span>
              </div>
              <NavChart data={backtestResult.nav_series} />
            </div>
          )}

          {/* Per-symbol PnL */}
          {backtestResult.symbol_pnl?.length > 0 && (
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
              <div className="flex items-center gap-1.5 mb-3">
                <BarChart2 className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">
                  Per-Symbol P&amp;L
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-800">
                      <th className="text-left py-1.5 pr-4">Symbol</th>
                      <th className="text-right py-1.5 pr-4">Net PnL</th>
                      <th className="text-right py-1.5 pr-4">Fills</th>
                      <th className="text-right py-1.5">Buys / Sells</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtestResult.symbol_pnl.map(s => (
                      <tr key={s.symbol} className="border-b border-gray-800/40">
                        <td className="py-1.5 pr-4 text-gray-200 font-mono">
                          {s.symbol}
                        </td>
                        <td
                          className={clsx(
                            "py-1.5 pr-4 text-right font-medium",
                            s.net_pnl >= 0 ? "text-emerald-400" : "text-red-400",
                          )}
                        >
                          {fmtPnl(s.net_pnl)}
                        </td>
                        <td className="py-1.5 pr-4 text-right text-gray-400">
                          {s.fills}
                        </td>
                        <td className="py-1.5 text-right text-gray-400">
                          {s.buys} / {s.sells}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Strategy fills */}
          {backtestResult.strategy_fills?.length > 0 && (
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
              <div className="flex items-center gap-1.5 mb-3">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">
                  Per-Strategy Fills
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-800">
                      <th className="text-left py-1.5 pr-4">Strategy</th>
                      <th className="text-right py-1.5 pr-4">Fills</th>
                      <th className="text-right py-1.5 pr-4">Buys</th>
                      <th className="text-right py-1.5">Sells</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtestResult.strategy_fills.map(sf => (
                      <tr key={sf.strategy} className="border-b border-gray-800/40">
                        <td className="py-1.5 pr-4 text-gray-200 font-mono text-[11px]">
                          {sf.strategy}
                        </td>
                        <td className="py-1.5 pr-4 text-right text-gray-300">
                          {sf.fills}
                        </td>
                        <td className="py-1.5 pr-4 text-right text-gray-400">
                          {sf.buys}
                        </td>
                        <td className="py-1.5 text-right text-gray-400">
                          {sf.sells}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Trades Table */}
          {backtestResult.trades?.length > 0 && (
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
              <div className="flex items-center gap-1.5 mb-3">
                <List className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-gray-300">
                  All Trades ({backtestResult.trades.length})
                </span>
              </div>
              <TradesTable trades={backtestResult.trades} />
            </div>
          )}

          {/* Footer: restart + next steps */}
          <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
            <div className="flex flex-col sm:flex-row sm:items-center gap-4">
              <button
                onClick={startOver}
                className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg border border-gray-700 text-gray-300 hover:text-gray-100 hover:border-gray-600 text-sm font-medium transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Run Again
              </button>
              <div className="flex items-start gap-2 text-xs text-gray-600">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-gray-400">Next:</strong> Switch to the{" "}
                  <strong className="text-gray-400">Advisor</strong> tab to get live
                  BUY / SELL signals on these symbols, or run{" "}
                  <strong className="text-gray-400">Auto-Tune</strong> to optimise
                  model parameters.
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

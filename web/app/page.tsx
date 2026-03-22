"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import clsx from "clsx";
import { BarChart2, Search, GitMerge, Brain, Repeat2, ChevronRight, X, RefreshCw, Sparkles } from "lucide-react";
import { listModels } from "@/lib/api";
import type { StoredModelInfo } from "@/lib/api";

// All panels lazy-loaded (no SSR) to avoid hydration issues with Recharts
const BacktestPanel   = dynamic(() => import("@/components/BacktestPanel"),   { ssr: false });
const ScreenerPanel   = dynamic(() => import("@/components/ScreenerPanel"),   { ssr: false });
const WorkflowPanel   = dynamic(() => import("@/components/WorkflowPanel"),   { ssr: false });
const AdvisorPanel    = dynamic(() => import("@/components/AdvisorPanel"),    { ssr: false });
const TrainLoopPanel  = dynamic(() => import("@/components/TrainLoopPanel"),  { ssr: false });
const AutoTunePanel   = dynamic(() => import("@/components/AutoTunePanel"),   { ssr: false });

const TABS = [
  {
    id: "autotune",
    step: 0,
    label: "Auto-Tune",
    Icon: Sparkles,
    tagline: "Fully automated",
    desc: "One click: screens stocks, trains models, reads its own quality metrics, diagnoses failures, adjusts LGB hyper-params + signal thresholds, and retrains — up to 5 iterations until score ≥ 70 / 100.",
  },
  {
    id: "trainloop",
    step: 1,
    label: "Train Loop",
    Icon: Repeat2,
    tagline: "Manual training",
    desc: "Screens the market, trains an LGBM model on the best stocks, and auto-tunes strategy parameters. Run this first — everything else uses what it finds.",
  },
  {
    id: "advisor",
    step: 2,
    label: "Advisor",
    Icon: Brain,
    tagline: "Get signals",
    desc: "Uses your trained models to score every stock in the universe and surface today's best BUY/HOLD/SELL recommendations.",
  },
  {
    id: "screener",
    step: 3,
    label: "Screener",
    Icon: Search,
    tagline: "Find stocks",
    desc: "Ranks all A-share stocks by momentum, trend quality, Sharpe, and drawdown. Drill into each result to see the causal factors behind the score.",
  },
  {
    id: "workflow",
    step: 4,
    label: "Workflow",
    Icon: GitMerge,
    tagline: "Screen → Backtest",
    desc: "Runs the screener and immediately backtests the top picks in one shot. Uses your saved best_params automatically.",
  },
  {
    id: "backtest",
    step: 5,
    label: "Backtest",
    Icon: BarChart2,
    tagline: "Test any symbols",
    desc: "Run a historical simulation on specific symbols you choose. Great for validating a specific idea or comparing individual stocks.",
  },
] as const;

type TabId = typeof TABS[number]["id"];

const FLOW_STEPS = [
  { step: 0, tab: "autotune"  as TabId, label: "Auto-Tune",  note: "train + self-improve automatically" },
  { step: 2, tab: "advisor"   as TabId, label: "Advisor",     note: "get today's BUY signals"            },
  { step: 4, tab: "workflow"  as TabId, label: "Workflow",    note: "screen + backtest in one click"     },
];

function fmtModelDate(ts?: number) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" });
}

function TrainedModelsBar({
  models,
  loading,
  onRefresh,
  onOpen,
  onTrainLoop,
}: {
  models: StoredModelInfo[];
  loading: boolean;
  onRefresh: () => void;
  onOpen: (symbol: string) => void;
  onTrainLoop: () => void;
}) {
  return (
    <div className="mb-4 rounded-lg border border-gray-800 bg-gray-900/40 px-4 py-2.5 flex items-center gap-3 flex-wrap">
      <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide shrink-0">Trained models</span>
      {loading ? (
        <span className="text-xs text-gray-600">Loading…</span>
      ) : models.length === 0 ? (
        <span className="text-xs text-gray-600 italic">
          None yet —{" "}
          <button onClick={onTrainLoop} className="text-indigo-400 hover:text-indigo-300 underline">
            run Train Loop first
          </button>
        </span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {models.map(m => (
            <button
              key={m.model_id}
              onClick={() => onOpen(m.symbol)}
              title={`Click to analyse · OOS: ${
                m.oos_accuracy != null ? (m.oos_accuracy * 100).toFixed(1) + "%" : "—"
              } · Trained: ${fmtModelDate(m.trained_at)} · ${m.bar_count} bars`}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-900/30 border border-indigo-500/30 text-indigo-300 text-xs font-mono hover:bg-indigo-800/40 hover:border-indigo-400/60 transition-colors"
            >
              {m.symbol}
              {m.oos_accuracy != null && (
                <span className="text-[10px] text-indigo-400/70">{(m.oos_accuracy * 100).toFixed(0)}%</span>
              )}
            </button>
          ))}
        </div>
      )}
      <button
        onClick={onRefresh}
        disabled={loading}
        className="ml-auto shrink-0 flex items-center gap-1 text-[10px] text-gray-600 hover:text-gray-400 transition-colors disabled:opacity-50"
        title="Refresh model list"
      >
        <RefreshCw className={clsx("w-3 h-3", loading && "animate-spin")} />
        refresh
      </button>
    </div>
  );
}

function GettingStartedBanner({ onNavigate }: { onNavigate: (id: TabId) => void }) {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) return null;

  return (
    <div className="rounded-xl border border-indigo-500/20 bg-indigo-950/30 p-4 mb-6 relative">
      <button
        onClick={() => setDismissed(true)}
        className="absolute top-3 right-3 text-gray-600 hover:text-gray-400 transition-colors"
        aria-label="Dismiss"
      >
        <X className="w-4 h-4" />
      </button>

      <div className="pr-6">
        <h2 className="text-sm font-semibold text-indigo-300 mb-1">👋 Getting Started</h2>
        <p className="text-xs text-gray-400 mb-4">
          Follow these three steps to go from raw data to live trade signals.
        </p>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          {FLOW_STEPS.map(({ step, tab, label, note }, i) => (
            <div key={tab} className="flex items-center gap-3">
              <button
                onClick={() => onNavigate(tab)}
                className="flex items-start gap-3 p-3 rounded-lg border border-gray-700 bg-gray-900/60 hover:border-indigo-500/50 hover:bg-indigo-900/20 transition-all text-left group"
              >
                <div className="w-6 h-6 rounded-full bg-indigo-600 flex items-center justify-center text-[10px] font-bold text-white shrink-0 mt-0.5">
                  {step}
                </div>
                <div>
                  <div className="text-xs font-semibold text-gray-200 group-hover:text-indigo-300 transition-colors">
                    {label}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-0.5">{note}</div>
                </div>
              </button>
              {i < FLOW_STEPS.length - 1 && (
                <ChevronRight className="w-4 h-4 text-gray-600 shrink-0 hidden sm:block" />
              )}
            </div>
          ))}
        </div>

        <p className="text-[10px] text-gray-600 mt-3">
          Once you have a trained model, the <strong className="text-gray-500">Advisor → Recommend</strong> tab surfaces daily signals automatically.
          Use <strong className="text-gray-500">Screener</strong> and <strong className="text-gray-500">Backtest</strong> for deeper research.
        </p>
      </div>
    </div>
  );
}

export default function Home() {
  const [active, setActive] = useState<TabId>("trainloop");
  const [models, setModels] = useState<StoredModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [advisorSymbol, setAdvisorSymbol] = useState<string | undefined>(undefined);

  const loadModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const r = await listModels();
      setModels(r.models);
    } catch { /* silent */ }
    finally { setModelsLoading(false); }
  }, []);

  useEffect(() => { loadModels(); }, [loadModels]);

  // Refresh model list whenever the user switches tabs (picks up newly-trained models)
  useEffect(() => { loadModels(); }, [active, loadModels]);

  const openAdvisor = useCallback((symbol: string) => {
    setAdvisorSymbol(symbol);
    setActive("advisor");
  }, []);

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-3">
          <span className="text-lg font-bold text-indigo-400 mr-2 shrink-0">MyQuant</span>
          <nav className="flex gap-0.5 overflow-x-auto">
            {TABS.map(({ id, step, label, Icon, tagline }) => (
              <button
                key={id}
                onClick={() => setActive(id)}
                title={tagline}
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-colors relative",
                  id === "autotune"
                    ? active === id
                      ? "bg-violet-600/20 text-violet-400"
                      : "text-violet-500/70 hover:text-violet-300 hover:bg-violet-900/20"
                    : active === id
                      ? "bg-indigo-600/20 text-indigo-400"
                      : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50",
                )}
              >
                {/* Step number badge */}
                <span className={clsx(
                  "w-4 h-4 rounded-full text-[9px] font-bold flex items-center justify-center shrink-0",
                  id === "autotune" && active === id ? "bg-violet-500 text-white" :
                  id === "autotune"                  ? "bg-violet-900/60 text-violet-400" :
                  active === id                      ? "bg-indigo-500 text-white" :
                                                       "bg-gray-700 text-gray-400",
                )}>
                  {step}
                </span>
                <Icon className="w-3.5 h-3.5" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {/* Trained models bar — persistent across all tabs */}
        <TrainedModelsBar
          models={models}
          loading={modelsLoading}
          onRefresh={loadModels}
          onOpen={openAdvisor}
          onTrainLoop={() => setActive("trainloop")}
        />

        {/* Getting started banner — shown on every tab until dismissed */}
        <GettingStartedBanner onNavigate={setActive} />

        {/* Panel description strip */}
        {(() => {
          const tab = TABS.find(t => t.id === active)!;
          return (
            <div className={clsx(
              "flex items-start gap-3 mb-5 p-3 rounded-lg border",
              tab.id === "autotune"
                ? "bg-violet-950/20 border-violet-800/40"
                : "bg-gray-900/40 border-gray-800"
            )}>
              <tab.Icon className={clsx(
                "w-4 h-4 mt-0.5 shrink-0",
                tab.id === "autotune" ? "text-violet-400" : "text-indigo-400"
              )} />
              <div>
                <span className="text-xs font-semibold text-gray-200">{tab.label}</span>
                <span className="mx-2 text-gray-600">·</span>
                <span className={clsx(
                  "text-xs font-medium",
                  tab.id === "autotune" ? "text-violet-400/90" : "text-indigo-400/80"
                )}>{tab.tagline}</span>
                <p className="text-xs text-gray-500 mt-0.5">{tab.desc}</p>
              </div>
            </div>
          );
        })()}

        {active === "autotune"  && <AutoTunePanel />}
        {active === "trainloop" && <TrainLoopPanel />}
        {active === "screener"  && <ScreenerPanel />}
        {active === "workflow"  && <WorkflowPanel />}
        {active === "advisor"   && <AdvisorPanel initialSymbol={advisorSymbol} />}
        {active === "backtest"  && <BacktestPanel />}
      </div>
    </main>
  );
}

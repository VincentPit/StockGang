"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import clsx from "clsx";
import { BarChart2, Search, GitMerge, Brain, Repeat2, ChevronRight, X } from "lucide-react";

// All panels lazy-loaded (no SSR) to avoid hydration issues with Recharts
const BacktestPanel   = dynamic(() => import("@/components/BacktestPanel"),   { ssr: false });
const ScreenerPanel   = dynamic(() => import("@/components/ScreenerPanel"),   { ssr: false });
const WorkflowPanel   = dynamic(() => import("@/components/WorkflowPanel"),   { ssr: false });
const AdvisorPanel    = dynamic(() => import("@/components/AdvisorPanel"),    { ssr: false });
const TrainLoopPanel  = dynamic(() => import("@/components/TrainLoopPanel"),  { ssr: false });

const TABS = [
  {
    id: "trainloop",
    step: 1,
    label: "Train Loop",
    Icon: Repeat2,
    tagline: "Start here",
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
  { step: 1, tab: "trainloop" as TabId, label: "Train Loop",  note: "trains models + saves best params" },
  { step: 2, tab: "advisor"   as TabId, label: "Advisor",     note: "get today's BUY signals"           },
  { step: 3, tab: "workflow"  as TabId, label: "Workflow",    note: "screen + backtest in one click"    },
];

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
                  active === id
                    ? "bg-indigo-600/20 text-indigo-400"
                    : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50",
                )}
              >
                {/* Step number badge */}
                <span className={clsx(
                  "w-4 h-4 rounded-full text-[9px] font-bold flex items-center justify-center shrink-0",
                  active === id ? "bg-indigo-500 text-white" : "bg-gray-700 text-gray-400",
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
        {/* Getting started banner — shown on every tab until dismissed */}
        <GettingStartedBanner onNavigate={setActive} />

        {/* Panel description strip */}
        {(() => {
          const tab = TABS.find(t => t.id === active)!;
          return (
            <div className="flex items-start gap-3 mb-5 p-3 rounded-lg bg-gray-900/40 border border-gray-800">
              <tab.Icon className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
              <div>
                <span className="text-xs font-semibold text-gray-200">{tab.label}</span>
                <span className="mx-2 text-gray-600">·</span>
                <span className="text-xs text-indigo-400/80 font-medium">{tab.tagline}</span>
                <p className="text-xs text-gray-500 mt-0.5">{tab.desc}</p>
              </div>
            </div>
          );
        })()}

        {active === "trainloop" && <TrainLoopPanel />}
        {active === "screener"  && <ScreenerPanel />}
        {active === "workflow"  && <WorkflowPanel />}
        {active === "advisor"   && <AdvisorPanel />}
        {active === "backtest"  && <BacktestPanel />}
      </div>
    </main>
  );
}

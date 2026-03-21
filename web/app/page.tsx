"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import clsx from "clsx";
import { BarChart2, Search, GitMerge, Brain, Repeat2 } from "lucide-react";

// All panels lazy-loaded (no SSR) to avoid hydration issues with Recharts
const BacktestPanel   = dynamic(() => import("@/components/BacktestPanel"),   { ssr: false });
const ScreenerPanel   = dynamic(() => import("@/components/ScreenerPanel"),   { ssr: false });
const WorkflowPanel   = dynamic(() => import("@/components/WorkflowPanel"),   { ssr: false });
const AdvisorPanel    = dynamic(() => import("@/components/AdvisorPanel"),    { ssr: false });
const TrainLoopPanel  = dynamic(() => import("@/components/TrainLoopPanel"),  { ssr: false });

const TABS = [
  { id: "backtest",  label: "Backtest",   Icon: BarChart2  },
  { id: "screener",  label: "Screener",   Icon: Search     },
  { id: "workflow",  label: "Workflow",   Icon: GitMerge   },
  { id: "advisor",   label: "Advisor",    Icon: Brain      },
  { id: "trainloop", label: "Train Loop", Icon: Repeat2    },
] as const;

type TabId = typeof TABS[number]["id"];

export default function Home() {
  const [active, setActive] = useState<TabId>("backtest");

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center gap-3 h-14">
          <span className="text-lg font-bold text-indigo-400 mr-4">MyQuant</span>
          <nav className="flex gap-0.5 overflow-x-auto">
            {TABS.map(({ id, label, Icon }) => (
              <button
                key={id}
                onClick={() => setActive(id)}
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-colors",
                  active === id
                    ? "bg-indigo-600/20 text-indigo-400"
                    : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50",
                )}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {active === "backtest"  && <BacktestPanel />}
        {active === "screener"  && <ScreenerPanel />}
        {active === "workflow"  && <WorkflowPanel />}
        {active === "advisor"   && <AdvisorPanel />}
        {active === "trainloop" && <TrainLoopPanel />}
      </div>
    </main>
  );
}


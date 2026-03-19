"use client";

import { lazy, Suspense } from "react";
import { BarChart2, Search, BookOpen, Zap, BrainCircuit, Loader2 } from "lucide-react";
import clsx from "clsx";
import { AccountProvider } from "@/lib/account-context";
import { NavProvider, useNav, type Tab } from "@/lib/nav-context";
import AccountWidget from "@/components/AccountWidget";

const ScreenerPanel  = lazy(() => import("@/components/ScreenerPanel"));
const AdvisorPanel   = lazy(() => import("@/components/AdvisorPanel"));
const ResearchPanel  = lazy(() => import("@/components/ResearchPanel"));
const BacktestPanel  = lazy(() => import("@/components/BacktestPanel"));
const WorkflowPanel  = lazy(() => import("@/components/WorkflowPanel"));

interface TabMeta {
  id:    Tab;
  label: string;
  sub:   string;
  icon:  React.ReactNode;
}

// Natural user flow: find → signals → research → test → automate
const TABS: TabMeta[] = [
  { id: "screener", label: "Find Stocks", sub: "Scan A-shares",     icon: <Search       className="w-4 h-4" /> },
  { id: "advisor",  label: "AI Signals",  sub: "ML-powered picks",  icon: <BrainCircuit className="w-4 h-4" /> },
  { id: "research", label: "Research",    sub: "Charts & news",     icon: <BookOpen     className="w-4 h-4" /> },
  { id: "backtest", label: "Backtest",    sub: "Test strategies",   icon: <BarChart2    className="w-4 h-4" /> },
  { id: "workflow", label: "Pipeline",    sub: "Auto screen→test",  icon: <Zap          className="w-4 h-4" /> },
];

const PANELS: Record<Tab, React.ComponentType> = {
  screener: ScreenerPanel,
  advisor:  AdvisorPanel,
  research: ResearchPanel,
  backtest: BacktestPanel,
  workflow: WorkflowPanel,
};

function TabFallback() {
  return (
    <div className="flex justify-center py-20">
      <Loader2 className="w-6 h-6 animate-spin text-sky-400" />
    </div>
  );
}

function AppShell() {
  const { activeTab, mounted, switchTab } = useNav();

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">

      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="flex items-center gap-2.5">
          <BarChart2 className="w-7 h-7 text-sky-400 shrink-0" />
          <div>
            <h1 className="text-xl font-bold tracking-tight leading-none">MyQuant</h1>
            <p className="text-[11px] text-gray-500 mt-0.5">Shanghai &amp; Shenzhen A-shares</p>
          </div>
        </div>
        <span className="ml-auto text-[10px] bg-sky-950 text-sky-400 border border-sky-800/60 px-2.5 py-0.5 rounded-full font-medium tracking-wide">
          LIVE DATA
        </span>
      </div>

      {/* Account widget */}
      <div className="mb-6">
        <AccountWidget />
      </div>

      {/* Tabs — natural left-to-right workflow */}
      <div className="flex gap-0 border-b border-gray-800 overflow-x-auto">
        {TABS.map((t, i) => (
          <button
            key={t.id}
            onClick={() => switchTab(t.id)}
            className={clsx(
              "flex flex-col items-start shrink-0 px-5 pt-3 pb-2.5 rounded-t-lg transition-all select-none",
              i > 0 && "ml-0.5",
              activeTab === t.id
                ? "bg-gray-900 border-b-2 border-sky-500 text-sky-400"
                : "text-gray-500 hover:text-gray-200 hover:bg-gray-900/40",
            )}
          >
            <span className="flex items-center gap-1.5 text-sm font-medium whitespace-nowrap">
              {t.icon}
              {t.label}
            </span>
            <span className={clsx(
              "text-[10px] mt-0.5 leading-none whitespace-nowrap",
              activeTab === t.id ? "text-sky-600" : "opacity-40",
            )}>
              {t.sub}
            </span>
          </button>
        ))}
      </div>

      {/* Panel content */}
      <div className="pt-6">
        {TABS.map(({ id }) => {
          if (!mounted.has(id)) return null;
          const Panel = PANELS[id];
          return (
            <div key={id} className={activeTab !== id ? "hidden" : undefined}>
              <Suspense fallback={<TabFallback />}>
                <Panel />
              </Suspense>
            </div>
          );
        })}
      </div>

    </div>
  );
}

export default function Home() {
  return (
    <AccountProvider>
      <NavProvider>
        <AppShell />
      </NavProvider>
    </AccountProvider>
  );
}



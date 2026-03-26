"use client";

import { useState, useEffect, useCallback, type ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import {
  BarChart2, Search, Rocket, Brain, Repeat2, ChevronRight, X,
  RefreshCw, Sparkles,
} from "lucide-react";
import { listModels } from "@/lib/api";
import type { StoredModelInfo } from "@/lib/api";

// ── Route / nav config ──────────────────────────────────────────────────────

const NAV_ITEMS = [
  {
    href: "/",
    group: "Research",
    label: "Workflow",
    Icon: Rocket,
    tagline: "Screen → Select → Backtest",
    desc: "The main research pipeline. Screen the market, hand-pick the best stocks, then backtest them — all in a guided 3-step wizard.",
  },
  {
    href: "/screener",
    group: "Research",
    label: "Screener",
    Icon: Search,
    tagline: "Find stocks",
    desc: "Ranks all A-share stocks by momentum, trend quality, Sharpe, and drawdown. Drill into each result to see the causal factors behind the score.",
  },
  {
    href: "/backtest",
    group: "Research",
    label: "Backtest",
    Icon: BarChart2,
    tagline: "Test any symbols",
    desc: "Run a historical simulation on specific symbols you choose. Great for validating a specific idea or comparing individual stocks.",
  },
  {
    href: "/autotune",
    group: "Models",
    label: "Auto-Tune",
    Icon: Sparkles,
    tagline: "Fully automated",
    desc: "One click: screens stocks, trains models, reads its own quality metrics, diagnoses failures, adjusts hyper-params + signal thresholds, and retrains — up to 5 iterations.",
  },
  {
    href: "/advisor",
    group: "Models",
    label: "Advisor",
    Icon: Brain,
    tagline: "Get signals",
    desc: "Uses your trained models to score every stock in the universe and surface today's best BUY/HOLD/SELL recommendations.",
  },
  {
    href: "/trainloop",
    group: "Models",
    label: "Train Loop",
    Icon: Repeat2,
    tagline: "Manual training",
    desc: "Screens the market, trains an LGBM model on the best stocks, and auto-tunes strategy parameters. For advanced users who want fine-grained control.",
  },
] as const;

const FLOW_STEPS = [
  { href: "/autotune", label: "Auto-Tune", note: "train a model (one click)" },
  { href: "/",         label: "Workflow",   note: "screen → pick stocks → backtest" },
  { href: "/advisor",  label: "Advisor",    note: "get today's BUY / SELL signals" },
];

function fmtModelDate(ts?: number) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "2-digit",
  });
}

// ── Trained models bar ──────────────────────────────────────────────────────

function TrainedModelsBar({
  models,
  loading,
  onRefresh,
}: {
  models: StoredModelInfo[];
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <div className="mb-4 rounded-lg border border-gray-800 bg-gray-900/40 px-4 py-2.5 flex items-center gap-3 flex-wrap">
      <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide shrink-0">
        Trained models
      </span>
      {loading ? (
        <span className="text-xs text-gray-600">Loading…</span>
      ) : models.length === 0 ? (
        <span className="text-xs text-gray-600 italic">
          None yet —{" "}
          <Link href="/trainloop" className="text-indigo-400 hover:text-indigo-300 underline">
            run Train Loop first
          </Link>
        </span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {models.map((m) => (
            <Link
              key={m.model_id}
              href={`/advisor?symbol=${m.symbol}`}
              title={`Click to analyse · OOS: ${
                m.oos_accuracy != null ? (m.oos_accuracy * 100).toFixed(1) + "%" : "—"
              } · Trained: ${fmtModelDate(m.trained_at)} · ${m.bar_count} bars`}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-900/30 border border-indigo-500/30 text-indigo-300 text-xs font-mono hover:bg-indigo-800/40 hover:border-indigo-400/60 transition-colors"
            >
              {m.symbol}
              {m.oos_accuracy != null && (
                <span className="text-[10px] text-indigo-400/70">
                  {(m.oos_accuracy * 100).toFixed(0)}%
                </span>
              )}
            </Link>
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

// ── Getting started banner ──────────────────────────────────────────────────

function GettingStartedBanner() {
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
          {FLOW_STEPS.map(({ href, label, note }, i) => (
            <div key={href} className="flex items-center gap-3">
              <Link
                href={href}
                className="flex items-start gap-3 p-3 rounded-lg border border-gray-700 bg-gray-900/60 hover:border-indigo-500/50 hover:bg-indigo-900/20 transition-all text-left group"
              >
                <div className="w-6 h-6 rounded-full bg-indigo-600 flex items-center justify-center text-[10px] font-bold text-white shrink-0 mt-0.5">
                  {i + 1}
                </div>
                <div>
                  <div className="text-xs font-semibold text-gray-200 group-hover:text-indigo-300 transition-colors">
                    {label}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-0.5">{note}</div>
                </div>
              </Link>
              {i < FLOW_STEPS.length - 1 && (
                <ChevronRight className="w-4 h-4 text-gray-600 shrink-0 hidden sm:block" />
              )}
            </div>
          ))}
        </div>

        <p className="text-[10px] text-gray-600 mt-3">
          You can skip straight to{" "}
          <Link href="/" className="text-gray-500 font-bold hover:text-indigo-300">
            Workflow
          </Link>{" "}
          without a model — it works out of the box. Train a model first for better signal quality.
        </p>
      </div>
    </div>
  );
}

// ── Dashboard shell ─────────────────────────────────────────────────────────

export default function DashboardShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [models, setModels] = useState<StoredModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  const loadModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const r = await listModels();
      setModels(r.models);
    } catch { /* silent */ }
    finally { setModelsLoading(false); }
  }, []);

  // Load on mount + whenever pathname changes (picks up newly-trained models)
  useEffect(() => { loadModels(); }, [pathname, loadModels]);

  // Find the current nav item for the description strip
  const current = NAV_ITEMS.find(
    (n) => n.href === pathname || (n.href !== "/" && pathname.startsWith(n.href)),
  ) ?? NAV_ITEMS[0];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* ── Header / nav bar ── */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-3">
          <Link href="/" className="text-lg font-bold text-indigo-400 mr-2 shrink-0">
            MyQuant
          </Link>
          <nav className="flex items-center gap-0.5 overflow-x-auto">
            {NAV_ITEMS.map((item, i) => {
              const prevGroup = i > 0 ? NAV_ITEMS[i - 1].group : null;
              const showDivider = item.group !== prevGroup;
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              return (
                <div key={item.href} className="flex items-center">
                  {showDivider && (
                    <span className="flex items-center gap-1 mr-1 ml-2 first:ml-0">
                      {i > 0 && <span className="w-px h-4 bg-gray-700 mr-2" />}
                      <span className="text-[9px] font-semibold uppercase tracking-widest text-gray-600 whitespace-nowrap">
                        {item.group}
                      </span>
                    </span>
                  )}
                  <Link
                    href={item.href}
                    title={item.tagline}
                    className={clsx(
                      "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-colors",
                      isActive
                        ? "bg-indigo-600/20 text-indigo-400"
                        : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50",
                    )}
                  >
                    <item.Icon className="w-3.5 h-3.5" />
                    {item.label}
                  </Link>
                </div>
              );
            })}
          </nav>
        </div>
      </header>

      {/* ── Page content ── */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <TrainedModelsBar
          models={models}
          loading={modelsLoading}
          onRefresh={loadModels}
        />
        <GettingStartedBanner />

        {/* Page description strip */}
        <div className="flex items-start gap-3 mb-5 p-3 rounded-lg border bg-gray-900/40 border-gray-800">
          <current.Icon className="w-4 h-4 mt-0.5 shrink-0 text-indigo-400" />
          <div>
            <span className="text-xs font-semibold text-gray-200">{current.label}</span>
            <span className="mx-2 text-gray-600">·</span>
            <span className="text-xs font-medium text-indigo-400/80">{current.tagline}</span>
            <p className="text-xs text-gray-500 mt-0.5">{current.desc}</p>
          </div>
        </div>

        {children}
      </div>
    </div>
  );
}

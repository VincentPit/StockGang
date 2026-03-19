"use client";

import { useState } from "react";
import clsx from "clsx";
import {
  Wallet,
  RefreshCw,
  RotateCcw,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  resetSimulator,
  type AccountInfo,
  type AccountPosition,
} from "@/lib/api";
import { useAccountCtx } from "@/lib/account-context";
import { useNav } from "@/lib/nav-context";

// ── helpers ───────────────────────────────────────────────────────────────────

function cny(n: number) {
  return "¥" + n.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function pctFmt(v: number) {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(2)}%`;
}

// ── sub-component: single position row ───────────────────────────────────────

function PositionRow({ p }: { p: AccountPosition }) {
  const pos = p.pnl_pct >= 0;
  return (
    <tr className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
      <td className="px-3 py-1.5 font-mono text-xs font-bold text-sky-400">{p.symbol}</td>
      <td className="px-3 py-1.5 text-xs text-right text-gray-200">{p.qty.toLocaleString()}</td>
      <td className="px-3 py-1.5 text-xs text-right text-gray-400">{p.avg_price.toFixed(2)}</td>
      <td className="px-3 py-1.5 text-xs text-right text-gray-300">{p.current_price.toFixed(2)}</td>
      <td className="px-3 py-1.5 text-xs text-right text-gray-200">{cny(p.market_value)}</td>
      <td className={clsx("px-3 py-1.5 text-xs text-right font-semibold", pos ? "text-emerald-400" : "text-red-400")}>
        <span className="flex items-center justify-end gap-0.5">
          {pos ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          {pctFmt(p.pnl_pct)}
        </span>
      </td>
    </tr>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function AccountWidget() {
  const { account, refresh: onRefresh, loading } = useAccountCtx();
  const { jumpTo } = useNav();
  const [expanded,  setExpanded]  = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetErr,  setResetErr]  = useState("");

  async function handleReset() {
    if (!confirm("Reset simulator to ¥500,000 with no positions?")) return;
    setResetting(true);
    setResetErr("");
    try {
      await resetSimulator();
      onRefresh();
    } catch (e: unknown) {
      setResetErr(e instanceof Error ? e.message : String(e));
    } finally {
      setResetting(false);
    }
  }

  if (!account) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-900 border border-gray-800 text-xs text-gray-500">
        <Wallet className="w-3.5 h-3.5" />
        {loading ? "Loading account…" : "Account unavailable"}
        <button onClick={onRefresh} disabled={loading} className="ml-1 text-sky-500 hover:text-sky-400">
          <RefreshCw className={clsx("w-3 h-3", loading && "animate-spin")} />
        </button>
      </div>
    );
  }

  const pnl = account.total_value - account.initial_cash;
  const pnlPct = account.initial_cash > 0 ? pnl / account.initial_cash : 0;

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 overflow-hidden">
      {/* Header row */}
      <div className="flex items-center gap-3 px-4 py-2.5 flex-wrap">
        <Wallet className="w-4 h-4 text-gray-400 shrink-0" />

        {/* Broker mode badge */}
        <span className={clsx(
          "text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full",
          account.is_simulated
            ? "bg-amber-900/60 text-amber-300 border border-amber-700"
            : "bg-emerald-900/60 text-emerald-300 border border-emerald-700"
        )}>
          {account.is_simulated ? "SIMULATOR" : "LIVE"}
        </span>

        {/* Cash */}
        <div className="flex flex-col">
          <span className="text-[10px] text-gray-500 leading-none">Cash</span>
          <span className="text-sm font-mono font-semibold text-gray-100">{cny(account.cash)}</span>
        </div>

        <div className="w-px h-6 bg-gray-700" />

        {/* Portfolio value */}
        <div className="flex flex-col">
          <span className="text-[10px] text-gray-500 leading-none">Portfolio</span>
          <span className="text-sm font-mono font-semibold text-gray-100">{cny(account.total_value)}</span>
        </div>

        {/* Total P&L */}
        <div className="flex flex-col">
          <span className="text-[10px] text-gray-500 leading-none">Total P&L</span>
          <span className={clsx(
            "text-sm font-mono font-semibold",
            pnl >= 0 ? "text-emerald-400" : "text-red-400"
          )}>
            {pnl >= 0 ? "+" : ""}{cny(pnl)}
            <span className="text-xs ml-1 opacity-75">({pctFmt(pnlPct)})</span>
          </span>
        </div>

        {/* Positions count */}
        {account.positions.length > 0 && (
          <span className="text-xs text-gray-500">
            {account.positions.length} position{account.positions.length !== 1 ? "s" : ""}
          </span>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Reset button (sim only) */}
        {account.is_simulated && (
          <button
            onClick={handleReset}
            disabled={resetting}
            title="Reset simulator to ¥500,000"
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs text-gray-400 hover:text-red-400 transition-colors"
          >
            <RotateCcw className={clsx("w-3 h-3", resetting && "animate-spin")} />
            Reset
          </button>
        )}

        {/* Refresh */}
        <button
          onClick={onRefresh}
          disabled={loading}
          title="Refresh account"
          className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs text-gray-400 hover:text-sky-400 transition-colors"
        >
          <RefreshCw className={clsx("w-3 h-3", loading && "animate-spin")} />
        </button>

        {/* Expand / collapse positions table */}
        {account.positions.length > 0 && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="text-gray-500 hover:text-gray-300 transition-colors"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        )}
      </div>

      {resetErr && (
        <div className="px-4 py-1 text-xs text-red-400 flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" /> {resetErr}
        </div>
      )}

      {/* Onboarding hint — shown only when account is completely fresh */}
      {account.positions.length === 0 && account.cash >= account.initial_cash * 0.999 && (
        <div className="border-t border-gray-800/50 px-4 py-2.5 flex items-center gap-3 bg-gradient-to-r from-sky-950/40 to-transparent">
          <span className="text-xs text-gray-400 flex-1">
            <span className="text-sky-300 font-semibold">Your simulator is funded.</span>
            {" "}Start by scanning stocks, then use AI signals to pick your best entry.
          </span>
          <button
            onClick={() => jumpTo("screener")}
            className="text-xs text-sky-400 hover:text-sky-300 font-semibold whitespace-nowrap flex items-center gap-1 shrink-0"
          >
            Find Stocks →
          </button>
        </div>
      )}

      {/* Positions table */}
      {expanded && account.positions.length > 0 && (
        <div className="border-t border-gray-800 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-gray-800/50">
                {["Symbol", "Qty", "Avg Price", "Cur Price", "Value", "P&L"].map((h) => (
                  <th key={h} className="px-3 py-1.5 text-left text-gray-500 font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {account.positions.map((p) => (
                <PositionRow key={p.symbol} p={p} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}



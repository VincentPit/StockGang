"use client";

import { Fragment, useState } from "react";
import {
  startScreen,
  getScreen,
  pollJob,
  submitOrder,
  type ScreenResult,
  type ScreenRow,
  type OrderRequest,
  type OrderResponse,
} from "@/lib/api";
import { useAccountCtx } from "@/lib/account-context";
import { useNav } from "@/lib/nav-context";
import { Loader2, Search, TrendingUp, TrendingDown, ShoppingCart, CheckCircle2, AlertCircle, BookOpen, ChevronDown, ChevronUp, GitBranch } from "lucide-react";
import { CausalTracePanel } from "./CausalTracePanel";
import clsx from "clsx";

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

function scoreColor(score: number) {
  if (score >= 0.7) return "text-emerald-400";
  if (score >= 0.5) return "text-sky-400";
  if (score >= 0.3) return "text-yellow-400";
  return "text-rose-400";
}

function ScoreBar({ score }: { score: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={clsx(
            "h-full rounded-full",
            score >= 0.7 ? "bg-emerald-400" : score >= 0.5 ? "bg-sky-400" : score >= 0.3 ? "bg-yellow-400" : "bg-rose-400"
          )}
          style={{ width: `${(score * 100).toFixed(0)}%` }}
        />
      </div>
      <span className={clsx("text-xs font-mono font-semibold w-10 text-right", scoreColor(score))}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}

export default function ScreenerPanel() {
  const [topN, setTopN] = useState(6);
  const [lookback, setLookback] = useState(1);
  const [indices, setIndices] = useState<string[]>(["000300"]);
  const [status, setStatus] = useState("");
  const [error, setError]   = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScreenResult | null>(null);

  // Trading state
  const { account, refresh: refreshAccount } = useAccountCtx();
  const { jumpTo } = useNav();
  const [tradeSymbol,  setTradeSymbol]  = useState<string | null>(null);
  const [expandedSym,  setExpandedSym]  = useState<string | null>(null);
  const [showConfirm,  setShowConfirm]  = useState(false);
  const [orderSide,    setOrderSide]    = useState<"BUY" | "SELL">("BUY");
  const [orderType,    setOrderType]    = useState<"MARKET" | "LIMIT">("MARKET");
  const [orderQty,     setOrderQty]     = useState(100);
  const [limitPrice,   setLimitPrice]   = useState("");
  const [submitting,   setSubmitting]   = useState(false);
  const [orderResult,  setOrderResult]  = useState<OrderResponse | null>(null);
  const [orderError,   setOrderError]   = useState("");

  function openTrade(row: ScreenRow) {
    const existingPos = account?.positions.find((p) => p.symbol === row.symbol);
    if (existingPos) {
      setOrderSide("SELL");
      setOrderQty(existingPos.qty);
    } else {
      setOrderSide("BUY");
      setOrderQty(100);
    }
    setTradeSymbol(row.symbol);
    setShowConfirm(false);
    setOrderResult(null);
    setOrderError("");
    setLimitPrice("");
    setOrderType("MARKET");
  }

  async function handleSubmitOrder(symbol: string) {
    setSubmitting(true);
    setOrderError("");
    setOrderResult(null);
    const req: OrderRequest = {
      symbol,
      side: orderSide,
      order_type: orderType,
      quantity: orderQty,
      ...(orderType === "LIMIT" && limitPrice ? { limit_price: parseFloat(limitPrice) } : {}),
    };
    try {
      const res = await submitOrder(req);
      setOrderResult(res);
      setShowConfirm(false);
      refreshAccount();
    } catch (e: unknown) {
      setOrderError(e instanceof Error ? e.message : String(e));
      setShowConfirm(false);
    } finally {
      setSubmitting(false);
    }
  }

  async function run() {
    setLoading(true);
    setResult(null);
    setError("");
    setStatus("Submitting…");
    try {
      const job = await startScreen({ top_n: topN, lookback_years: lookback, indices });
      setStatus("Downloading real market data (~30s)…");
      const done = await pollJob(job.job_id, getScreen, (s) => setStatus(`Status: ${s}…`), 2000);
      setResult(done);
      setStatus("Done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Config */}
      <div className="bg-gray-900 rounded-xl p-6 space-y-4">
        <h2 className="text-lg font-semibold text-sky-400">Universe Screener</h2>
        <p className="text-xs text-gray-400">
          Pulls live CSI index constituents via akshare, fetches real OHLCV data in parallel,
          then ranks by trend %, ATR%, momentum autocorrelation, 6M return, and max drawdown.
        </p>

        {/* Index selector */}
        <div>
          <label className="block text-xs text-gray-400 mb-2">Index Universe</label>
          <div className="flex gap-2 flex-wrap">
            {([
              { value: ["000300"],               label: "CSI 300",       count: "~300 stocks" },
              { value: ["000300", "000905"],      label: "CSI 300+500",  count: "~800 stocks" },
            ] as { value: string[]; label: string; count: string }[]).map((opt) => {
              const active = JSON.stringify(opt.value) === JSON.stringify(indices);
              return (
                <button
                  key={opt.label}
                  onClick={() => setIndices(opt.value)}
                  className={`px-4 py-2 rounded-lg text-xs font-medium border transition-colors ${
                    active
                      ? "bg-sky-700 border-sky-500 text-white"
                      : "bg-gray-800 border-gray-700 text-gray-400 hover:border-sky-600 hover:text-sky-300"
                  }`}
                >
                  {opt.label}
                  <span className="ml-1.5 text-gray-400 font-normal">{opt.count}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Top N to select</label>
            <input
              type="number"
              value={topN}
              min={1}
              max={20}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Lookback (years)</label>
            <input
              type="number"
              value={lookback}
              min={1}
              max={3}
              onChange={(e) => setLookback(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            />
          </div>
          <div className="flex items-end md:col-span-2">
            <button
              onClick={run}
              disabled={loading}
              className="flex items-center gap-2 justify-center w-full bg-sky-600 hover:bg-sky-500 disabled:opacity-50 rounded-lg px-4 py-2 text-sm font-medium transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              {loading ? "Screening…" : "Run Screener"}
            </button>
          </div>
        </div>

        {status && <p className="text-xs text-gray-400">{status}</p>}
      </div>

      {error && (
        <div className="p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      {result && result.status === "done" && (
        <>
          {/* Top picks */}
          <div className="bg-gray-900 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-emerald-400 mb-3 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Top {topN} Picks
            </h3>
            <div className="flex flex-wrap gap-2">
              {result.top_symbols.map((sym) => {
                const row = result.rows.find((r) => r.symbol === sym);
                return (
                  <div
                    key={sym}
                    className="bg-emerald-900/30 border border-emerald-800 rounded-lg px-3 py-2 text-sm"
                  >
                    <span className="font-mono font-semibold text-emerald-300">{sym}</span>
                    {row && (
                      <span className="ml-2 text-xs text-gray-400">
                        {row.name} · {pct(row.ret_1y)} 1Y
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Full ranked table */}
          <div className="bg-gray-900 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              {result.rows.length} scored
              {result.universe_size ? ` / ${result.universe_size} screened` : ""}
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-800">
                    {["#", "Symbol", "Name", "1Y Ret", "6M Ret", "Sharpe", "MaxDD", "Trend%", "ATR%", "Score", "Signal", ""].map((h) => (
                      <th key={h} className="text-left px-3 py-2 font-medium whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row: ScreenRow) => (
                    <Fragment key={row.symbol}>
                    <tr
                      className={clsx(
                        "border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors",
                        row.recommended && "bg-emerald-950/20"
                      )}
                    >
                      <td className="px-3 py-2 text-gray-500">{row.rank}</td>
                      <td className="px-3 py-2 font-mono font-semibold">{row.symbol}</td>
                      <td className="px-3 py-2 text-gray-300 whitespace-nowrap">{row.name}</td>
                      <td className={clsx("px-3 py-2 font-mono", row.ret_1y >= 0 ? "text-emerald-400" : "text-rose-400")}>
                        {pct(row.ret_1y)}
                      </td>
                      <td className={clsx("px-3 py-2 font-mono", row.ret_6m >= 0 ? "text-emerald-400" : "text-rose-400")}>
                        {pct(row.ret_6m)}
                      </td>
                      <td className="px-3 py-2 font-mono">{row.sharpe.toFixed(2)}</td>
                      <td className="px-3 py-2 font-mono text-rose-400">{pct(row.max_dd)}</td>
                      <td className="px-3 py-2 font-mono">{pct(row.trend_pct)}</td>
                      <td className="px-3 py-2 font-mono">{pct(row.atr_pct)}</td>
                      <td className="px-3 py-2 w-36">
                        <ScoreBar score={row.score} />
                      </td>
                      <td className="px-3 py-2">
                        {row.recommended ? (
                          <span className="text-emerald-400 font-semibold flex items-center gap-1">
                            <TrendingUp className="w-3 h-3" /> ADD
                          </span>
                        ) : row.rank > result.rows.length - 3 ? (
                          <span className="text-rose-400 flex items-center gap-1">
                            <TrendingDown className="w-3 h-3" /> DROP
                          </span>
                        ) : null}
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() =>
                              tradeSymbol === row.symbol
                                ? setTradeSymbol(null)
                                : openTrade(row)
                            }
                            className={clsx(
                              "flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors whitespace-nowrap",
                              tradeSymbol === row.symbol
                                ? "bg-emerald-800 text-emerald-200"
                                : "bg-emerald-700 hover:bg-emerald-600 text-white"
                            )}
                          >
                            <ShoppingCart className="w-3 h-3" />
                            {account?.positions.find((p) => p.symbol === row.symbol)
                              ? `Trade (hold ${account.positions.find((p) => p.symbol === row.symbol)!.qty})`
                              : "Trade"}
                          </button>
                          <button
                            onClick={() => jumpTo("research", row.symbol)}
                            className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-indigo-900/60 hover:bg-indigo-800 border border-indigo-800/50 text-indigo-300 hover:text-indigo-200 transition-colors whitespace-nowrap"
                            title={`Deep-dive research on ${row.symbol}`}
                          >
                            <BookOpen className="w-3 h-3" />
                            Research
                          </button>
                          {row.causal_nodes && row.causal_nodes.length > 0 && (
                            <button
                              onClick={() => setExpandedSym(expandedSym === row.symbol ? null : row.symbol)}
                              className={clsx(
                                "flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors whitespace-nowrap",
                                expandedSym === row.symbol
                                  ? "bg-violet-800 border-violet-600 text-violet-200"
                                  : "bg-violet-900/40 border-violet-800/50 text-violet-400 hover:text-violet-200 hover:border-violet-600",
                              )}
                              title="Show causal trace"
                            >
                              <GitBranch className="w-3 h-3" />
                              {expandedSym === row.symbol ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>

                    {/* Causal trace expand row */}
                    {expandedSym === row.symbol && row.causal_nodes && row.causal_nodes.length > 0 && (
                      <tr>
                        <td colSpan={12} className="p-0">
                          <div className="px-6 py-4 bg-gray-900/60 border-t border-violet-900/30">
                            <CausalTracePanel
                              decision={row.recommended ? "SELECTED" : "RANKED"}
                              score={row.score}
                              rank={row.rank}
                              universeSize={result.universe_size}
                              causalNodes={row.causal_nodes}
                              dataScope={row.data_scope}
                              gateChecks={row.gate_checks}
                            />
                          </div>
                        </td>
                      </tr>
                    )}

                    {/* Inline order form row */}
                    {tradeSymbol === row.symbol && (
                      <tr>
                        <td colSpan={12} className="p-0">
                          <div className="bg-gray-800/60 border-t border-gray-700 px-6 py-4 space-y-3">
                            {/* Causal context summary */}
                            {row.causal_nodes && row.causal_nodes.length > 0 && (() => {
                              const topNode = [...row.causal_nodes].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))[0];
                              return (
                                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 bg-violet-950/40 border border-violet-800/40 rounded-lg text-[11px] text-violet-300">
                                  <GitBranch className="w-3 h-3 text-violet-500 shrink-0" />
                                  <span className="font-mono font-semibold">Score {row.score.toFixed(3)}</span>
                                  <span className="text-violet-600">·</span>
                                  <span>Top factor: <span className="text-violet-200 font-medium">{topNode.label}</span></span>
                                  {topNode.percentile && <span className="text-violet-500">{topNode.percentile}</span>}
                                  {row.data_scope && (
                                    <>
                                      <span className="text-violet-600">·</span>
                                      <span className="text-violet-400">
                                        {row.data_scope.start_date} → {row.data_scope.end_date}
                                        <span className="text-violet-600 ml-1">({row.data_scope.bars} bars)</span>
                                      </span>
                                    </>
                                  )}
                                </div>
                              );
                            })()}
                            {orderResult ? (
                              <div className="space-y-2">
                                <div className="flex items-center gap-2 text-sm text-emerald-300 bg-emerald-950 border border-emerald-800 rounded-lg p-3">
                                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                                  <div className="flex-1">
                                    <span className="font-semibold">
                                      {orderResult.is_simulated ? "Simulated order filled" : "Order submitted"}
                                    </span>
                                    <span className="text-emerald-400 ml-2 font-mono text-xs">{orderResult.broker_order_id}</span>
                                    <span className="text-gray-400 ml-2 text-xs">
                                      {orderResult.side} {orderResult.quantity}×{orderResult.symbol}
                                      {orderResult.limit_price ? ` @ ¥${orderResult.limit_price}` : " (market)"}
                                    </span>
                                  </div>
                                </div>
                                {orderResult.cash_after !== undefined && (
                                  <p className="text-xs text-gray-400">
                                    Cash remaining:{" "}
                                    <span className="font-mono text-gray-200">
                                      ¥{orderResult.cash_after.toLocaleString("zh-CN", { minimumFractionDigits: 2 })}
                                    </span>
                                  </p>
                                )}
                                <button
                                  onClick={() => { setOrderResult(null); setShowConfirm(false); }}
                                  className="text-xs text-sky-400 hover:text-sky-300"
                                >
                                  Place another order
                                </button>
                              </div>
                            ) : showConfirm ? (
                              <div className="space-y-3">
                                <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">Confirm order</p>
                                <div className="bg-gray-800 rounded-lg px-4 py-3 space-y-1.5 text-sm max-w-md">
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">Action</span>
                                    <span className={clsx("font-bold", orderSide === "BUY" ? "text-emerald-400" : "text-red-400")}>{orderSide}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">Symbol</span>
                                    <span className="font-mono text-sky-400">{row.symbol} · {row.name}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">Quantity</span>
                                    <span className="text-gray-100">{orderQty.toLocaleString()} shares</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">Order type</span>
                                    <span className="text-gray-100">{orderType}{orderType === "LIMIT" && limitPrice ? ` @ ¥${limitPrice}` : ""}</span>
                                  </div>
                                  {account && (
                                    <div className="flex justify-between border-t border-gray-700 pt-1.5 mt-0.5">
                                      <span className="text-gray-400">Available cash</span>
                                      <span className="font-mono text-gray-200">
                                        ¥{account.cash.toLocaleString("zh-CN", { minimumFractionDigits: 2 })}
                                      </span>
                                    </div>
                                  )}
                                  {account?.is_simulated && (
                                    <div className="flex justify-between">
                                      <span className="text-xs text-gray-500">Mode</span>
                                      <span className="text-xs text-amber-400 font-medium">SIMULATOR</span>
                                    </div>
                                  )}
                                </div>
                                <div className="flex gap-2">
                                  <button
                                    onClick={() => handleSubmitOrder(row.symbol)}
                                    disabled={submitting}
                                    className={clsx(
                                      "flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50",
                                      orderSide === "BUY" ? "bg-emerald-600 hover:bg-emerald-500" : "bg-red-600 hover:bg-red-500"
                                    )}
                                  >
                                    {submitting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ShoppingCart className="w-3.5 h-3.5" />}
                                    {submitting ? "Submitting…" : `Confirm ${orderSide}`}
                                  </button>
                                  <button
                                    onClick={() => setShowConfirm(false)}
                                    disabled={submitting}
                                    className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
                                  >
                                    Edit
                                  </button>
                                </div>
                                {orderError && (
                                  <div className="p-2.5 bg-red-950 border border-red-800 rounded-lg text-xs text-red-300">{orderError}</div>
                                )}
                              </div>
                            ) : (
                              <div className="space-y-3">
                                <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">
                                  Place order — {row.symbol} · {row.name}
                                </p>

                                {/* Position banner */}
                                {(() => {
                                  const ep = account?.positions.find((p) => p.symbol === row.symbol);
                                  return ep ? (
                                    <div className="flex items-center gap-2 px-3 py-2 bg-amber-950/50 border border-amber-800/60 rounded-lg text-xs text-amber-300">
                                      <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                                      You hold <span className="font-bold font-mono mx-0.5">{ep.qty}</span> shares
                                      @ avg <span className="font-mono mx-0.5">¥{ep.avg_price.toFixed(2)}</span>
                                      {" "}(P&amp;L:{" "}
                                      <span className={clsx("font-semibold ml-0.5", ep.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400")}>
                                        {ep.pnl_pct >= 0 ? "+" : ""}{(ep.pnl_pct * 100).toFixed(2)}%
                                      </span>)
                                    </div>
                                  ) : null;
                                })()}

                                {account && (
                                  <p className="text-xs text-gray-500">
                                    Cash available:{" "}
                                    <span className="font-mono text-gray-300">
                                      ¥{account.cash.toLocaleString("zh-CN", { minimumFractionDigits: 2 })}
                                    </span>
                                    {account.is_simulated && <span className="ml-2 text-amber-500 font-medium">[SIMULATOR]</span>}
                                  </p>
                                )}

                                <div className="flex flex-wrap gap-3 items-end">
                                  <div>
                                    <label className="block text-[10px] text-gray-500 mb-1">Side</label>
                                    <div className="flex rounded-lg overflow-hidden border border-gray-700">
                                      {(["BUY", "SELL"] as const).map((s) => (
                                        <button key={s} onClick={() => setOrderSide(s)}
                                          className={clsx("px-4 py-1.5 text-xs font-semibold transition-colors",
                                            orderSide === s
                                              ? s === "BUY" ? "bg-emerald-700 text-white" : "bg-red-700 text-white"
                                              : "bg-gray-800 text-gray-400 hover:text-gray-200"
                                          )}
                                        >{s}</button>
                                      ))}
                                    </div>
                                  </div>
                                  <div>
                                    <label className="block text-[10px] text-gray-500 mb-1">Type</label>
                                    <div className="flex rounded-lg overflow-hidden border border-gray-700">
                                      {(["MARKET", "LIMIT"] as const).map((t) => (
                                        <button key={t} onClick={() => setOrderType(t)}
                                          className={clsx("px-3 py-1.5 text-xs font-medium transition-colors",
                                            orderType === t ? "bg-sky-700 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
                                          )}
                                        >{t}</button>
                                      ))}
                                    </div>
                                  </div>
                                  <div>
                                    <label className="block text-[10px] text-gray-500 mb-1">Quantity</label>
                                    <input type="number" min={1} value={orderQty}
                                      onChange={(e) => setOrderQty(Math.max(1, parseInt(e.target.value) || 1))}
                                      className="w-24 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-sky-500"
                                    />
                                  </div>
                                  {orderType === "LIMIT" && (
                                    <div>
                                      <label className="block text-[10px] text-gray-500 mb-1">Limit price</label>
                                      <input type="number" step="0.01" min="0.01" placeholder="e.g. 38.50" value={limitPrice}
                                        onChange={(e) => setLimitPrice(e.target.value)}
                                        className="w-28 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-sky-500"
                                      />
                                    </div>
                                  )}
                                  <button
                                    onClick={() => setShowConfirm(true)}
                                    disabled={orderType === "LIMIT" && !limitPrice}
                                    className={clsx(
                                      "flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50",
                                      orderSide === "BUY" ? "bg-emerald-600 hover:bg-emerald-500" : "bg-red-600 hover:bg-red-500"
                                    )}
                                  >
                                    <ShoppingCart className="w-3.5 h-3.5" />
                                    Review {orderSide} {orderQty}
                                  </button>
                                </div>
                                {orderError && (
                                  <div className="p-2.5 bg-red-950 border border-red-800 rounded-lg text-xs text-red-300">{orderError}</div>
                                )}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

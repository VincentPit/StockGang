"use client";

import { useEffect, useState } from "react";
import {
  getFundamentals,
  getPrice,
  getRegime,
  getStockNews,
  getMacroNews,
  type FundamentalsResponse,
  type NewsResponse,
  type PriceResponse,
  type RegimeResponse,
} from "@/lib/api";
import { useNav } from "@/lib/nav-context";
import PriceChart from "@/components/PriceChart";
import FundamentalsCard from "@/components/FundamentalsCard";
import NewsPanel from "@/components/NewsPanel";
import RegimeBadge from "@/components/RegimeBadge";
import { Search, Loader2, Globe, TrendingUp } from "lucide-react";

// Quick-access popular A-share picks
const QUICK_PICKS = [
  { symbol: "sz300059", name: "东方财富" },
  { symbol: "sz300750", name: "宁德时代" },
  { symbol: "sz000858", name: "五粮液" },
  { symbol: "sz000333", name: "美的集团" },
  { symbol: "sh601318", name: "中国平安" },
  { symbol: "sz002594", name: "BYD" },
];

export default function ResearchPanel() {
  const { jumpSymbol, clearJumpSymbol, activeTab } = useNav();

  const [symbol, setSymbol]       = useState("");
  const [days, setDays]           = useState(365);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const [macroLoading, setMacroLoading] = useState(false);

  const [price,   setPrice]   = useState<PriceResponse | null>(null);
  const [fund,    setFund]    = useState<FundamentalsResponse | null>(null);
  const [news,    setNews]    = useState<NewsResponse | null>(null);
  const [macro,   setMacro]   = useState<NewsResponse | null>(null);
  const [regime,  setRegime]  = useState<RegimeResponse | null>(null);

  // Jump from another panel (e.g. screener "Research →") — only consume when
  // this tab is actually active to avoid bleeding into other mounted panels.
  useEffect(() => {
    if (jumpSymbol && activeTab === "research") {
      const sym = jumpSymbol;
      clearJumpSymbol();
      setSymbol(sym);
      doSearch(sym, days);
    }
  }, [jumpSymbol, activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  async function doSearch(sym: string, lookback = days) {
    const s = sym.trim().toLowerCase();
    if (!s) return;
    setLoading(true);
    setError("");
    const context = QUICK_PICKS.map((q) => q.symbol).filter((x) => x !== s).slice(0, 3);
    try {
      const [p, f, n, r] = await Promise.all([
        getPrice(s, lookback),
        getFundamentals(s),
        getStockNews(s, 25),
        getRegime([s, ...context]),
      ]);
      setPrice(p);
      setFund(f);
      setNews(n);
      setRegime(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch() {
    doSearch(symbol, days);
  }

  async function loadMacroNews() {
    setMacroLoading(true);
    try {
      const m = await getMacroNews(30);
      setMacro(m);
    } catch {/* ignore */} finally {
      setMacroLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* ── Search bar ── */}
      <div className="space-y-3">
        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Symbol</label>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="e.g. sz300059, sz300750, sz000858"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider">Period</label>
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
            >
              <option value={90}>90d</option>
              <option value={180}>180d</option>
              <option value={365}>1yr</option>
              <option value={730}>2yr</option>
            </select>
          </div>
          <button
            onClick={handleSearch}
            disabled={loading || !symbol.trim()}
            className="flex items-center gap-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            Analyse
          </button>
        </div>

        {/* Quick-pick chips */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[11px] text-gray-600 uppercase tracking-wider">Quick picks:</span>
          {QUICK_PICKS.map((q) => (
            <button
              key={q.symbol}
              onClick={() => { setSymbol(q.symbol); doSearch(q.symbol, days); }}
              className="inline-flex items-center gap-1 text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-sky-700 text-gray-300 hover:text-sky-300 px-2.5 py-1 rounded-full transition-colors"
            >
              <span className="font-medium">{q.name}</span>
              <span className="text-gray-500 font-mono text-[10px]">{q.symbol}</span>
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="text-rose-400 text-sm bg-rose-950/30 border border-rose-900 rounded-lg p-3">
          {error}
        </div>
      )}

      {/* ── Regime badge ── */}
      {regime && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Market Regime</span>
          <RegimeBadge
            regime={regime.regime}
            multiplier={regime.signal_multiplier}
            symbolsAnalyzed={regime.symbols_analyzed}
          />
        </div>
      )}

      {/* ── Main grid: chart + fundamentals ── */}
      {(price || fund) && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          {/* Price chart — wider */}
          <div className="lg:col-span-3 bg-gray-950 border border-gray-800 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-4 h-4 text-sky-400" />
              <span className="text-sm font-medium">Price History</span>
            </div>
            {price && <PriceChart bars={price.bars} symbol={symbol.toUpperCase()} />}
            {price?.error && (
              <p className="text-rose-400 text-xs mt-2">{price.error}</p>
            )}
          </div>

          {/* Fundamentals — narrower */}
          <div className="lg:col-span-2 bg-gray-950 border border-gray-800 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-medium">Fundamentals</span>
            </div>
            {fund && <FundamentalsCard data={fund} />}
          </div>
        </div>
      )}

      {/* ── News columns ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Stock news */}
        {news && (
          <div className="bg-gray-950 border border-gray-800 rounded-xl p-4">
            <p className="text-sm font-medium mb-3">
              Stock News — {symbol.toUpperCase()}
            </p>
            <NewsPanel items={news.items} />
          </div>
        )}

        {/* Macro news */}
        <div className="bg-gray-950 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium">Macro News</p>
            {!macro && (
              <button
                onClick={loadMacroNews}
                disabled={macroLoading}
                className="flex items-center gap-1.5 text-xs text-sky-400 hover:text-sky-300 disabled:opacity-50"
              >
                {macroLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Globe className="w-3 h-3" />}
                Load
              </button>
            )}
          </div>
          {macro ? (
            <NewsPanel items={macro.items} loading={macroLoading} />
          ) : (
            <div className="text-gray-600 text-sm text-center py-8">
              Click &quot;Load&quot; to fetch macro headlines
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

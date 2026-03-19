"use client";

import type { FundamentalsResponse } from "@/lib/api";

interface Props {
  data: FundamentalsResponse;
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  const color =
    value >= 70 ? "bg-emerald-500" : value >= 40 ? "bg-sky-500" : "bg-rose-500";
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span className="font-mono">{value.toFixed(1)}</span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.min(value, 100)}%` }} />
      </div>
    </div>
  );
}

function Metric({ label, value, unit = "" }: { label: string; value: number; unit?: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-3 flex flex-col gap-0.5">
      <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
      <span className="text-lg font-semibold font-mono text-gray-100">
        {value.toFixed(1)}<span className="text-xs text-gray-500 ml-0.5">{unit}</span>
      </span>
    </div>
  );
}

export default function FundamentalsCard({ data }: Props) {
  if (data.error) return (
    <div className="text-rose-400 text-sm p-3 bg-rose-950/30 rounded-lg">{data.error}</div>
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        <Metric label="P/E (TTM)"      value={data.pe_ttm} />
        <Metric label="P/B"            value={data.pb} />
        <Metric label="P/S (TTM)"      value={data.ps_ttm} />
        <Metric label="ROE"            value={data.roe}            unit="%" />
        <Metric label="Rev. Growth"    value={data.revenue_growth} unit="%" />
        <Metric label="Net Margin"     value={data.net_margin}     unit="%" />
        <Metric label="Div. Yield"     value={data.dividend_yield} unit="%" />
      </div>
      <div className="space-y-2 pt-1">
        <ScoreBar value={data.value_score}   label="Value Score" />
        <ScoreBar value={data.growth_score}  label="Growth Score" />
        <ScoreBar value={data.quality_score} label="Quality Score" />
      </div>
    </div>
  );
}

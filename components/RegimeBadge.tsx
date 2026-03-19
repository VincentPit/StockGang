"use client";

import clsx from "clsx";

interface Props {
  regime: string;
  multiplier: number;
  symbolsAnalyzed: number;
}

const MAP: Record<string, { label: string; cls: string; dot: string }> = {
  RISK_ON:  { label: "Risk On",  cls: "bg-emerald-950 text-emerald-400 border-emerald-800", dot: "bg-emerald-400" },
  NEUTRAL:  { label: "Neutral",  cls: "bg-amber-950  text-amber-400  border-amber-800",  dot: "bg-amber-400"  },
  RISK_OFF: { label: "Risk Off", cls: "bg-rose-950   text-rose-400   border-rose-800",   dot: "bg-rose-400"   },
};

export default function RegimeBadge({ regime, multiplier, symbolsAnalyzed }: Props) {
  const meta = MAP[regime] ?? MAP["NEUTRAL"];

  return (
    <div className={clsx("inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium", meta.cls)}>
      <span className={clsx("h-2 w-2 rounded-full animate-pulse", meta.dot)} />
      <span>{meta.label}</span>
      <span className="opacity-60">·</span>
      <span className="font-mono">{multiplier.toFixed(2)}×</span>
      <span className="opacity-60">·</span>
      <span>{symbolsAnalyzed} sym</span>
    </div>
  );
}

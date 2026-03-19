"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { SymbolPnL } from "@/lib/api";

interface Props {
  data: SymbolPnL[];
}

export function SymbolPnLChart({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 16, bottom: 0, left: 16 }}>
        <XAxis dataKey="symbol" tick={{ fill: "#9ca3af", fontSize: 12 }} tickLine={false} />
        <YAxis
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          tickLine={false}
          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
        />
        <Tooltip
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
          formatter={(v: number) => [`$${v.toLocaleString()}`, "Net P&L"]}
        />
        <ReferenceLine y={0} stroke="#374151" />
        <Bar dataKey="net_pnl" radius={[4, 4, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.net_pnl >= 0 ? "#34d399" : "#f87171"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

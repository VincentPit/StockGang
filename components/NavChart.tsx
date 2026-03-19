"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { NavPoint } from "@/lib/api";

interface Props {
  data: NavPoint[];
  initial: number;
}

export function NavChart({ data, initial }: Props) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 4, right: 16, bottom: 0, left: 16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          tickLine={false}
          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          domain={["auto", "auto"]}
        />
        <Tooltip
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
          labelStyle={{ color: "#e5e7eb", fontSize: 12 }}
          formatter={(v: number) => [`$${v.toLocaleString()}`, "NAV"]}
        />
        <ReferenceLine y={initial} stroke="#374151" strokeDasharray="4 4" />
        <Line
          type="monotone"
          dataKey="nav"
          stroke="#38bdf8"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#38bdf8" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

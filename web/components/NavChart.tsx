"use client";

import { useMemo } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
} from "recharts";
import type { NavPoint } from "@/lib/api";

interface NavChartProps {
  data: NavPoint[];
  height?: number;
}

export function NavChart({ data, height = 240 }: NavChartProps) {
  const formatted = useMemo(() =>
    data.map(d => ({ ...d, nav: Number(d.nav.toFixed(2)) })),
    [data],
  );

  const min = useMemo(() => Math.min(...formatted.map(d => d.nav)), [formatted]);
  const max = useMemo(() => Math.max(...formatted.map(d => d.nav)), [formatted]);
  const pad = (max - min) * 0.05;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={formatted} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: "#6b7280" }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[min - pad, max + pad]}
          tick={{ fontSize: 10, fill: "#6b7280" }}
          tickLine={false}
          axisLine={false}
          width={60}
          tickFormatter={v => `¥${(v / 1000).toFixed(0)}k`}
        />
        <Tooltip
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, fontSize: 12 }}
          labelStyle={{ color: "#9ca3af" }}
          itemStyle={{ color: "#e5e7eb" }}
          formatter={(v: number) => [`¥${v.toLocaleString("en-US")}`, "NAV"]}
        />
        <Line
          type="monotone"
          dataKey="nav"
          stroke="#6366f1"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

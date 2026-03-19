"use client";

import {
  Area,
  AreaChart,
  Bar,
  CartesianGrid,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { OHLCVBar } from "@/lib/api";

interface Props {
  bars: OHLCVBar[];
  symbol: string;
}

const fmt = (v: number) =>
  v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(2);

export default function PriceChart({ bars, symbol }: Props) {
  if (!bars.length) return (
    <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
      No price data
    </div>
  );

  // thin out to at most 252 points for performance
  const step = Math.max(1, Math.floor(bars.length / 252));
  const data = bars.filter((_, i) => i % step === 0);

  // normalise volume to 0-100 range for dual-axis display
  const maxVol = Math.max(...data.map((b) => b.volume), 1);
  const normed = data.map((b) => ({
    ...b,
    vol_pct: (b.volume / maxVol) * 100,
  }));

  return (
    <div>
      <p className="text-xs text-gray-400 mb-2 uppercase tracking-wider">
        {symbol} — close price
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={normed} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickLine={false}
            interval={Math.floor(normed.length / 6)}
          />
          <YAxis
            yAxisId="price"
            orientation="right"
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickLine={false}
            tickFormatter={fmt}
            width={52}
          />
          <YAxis yAxisId="vol" orientation="left" hide domain={[0, 300]} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
            labelStyle={{ color: "#94a3b8", fontSize: 11 }}
            formatter={(val: number, name: string) =>
              name === "vol_pct"
                ? [`${(val / 100 * maxVol / 1e6).toFixed(1)} M`, "Volume"]
                : [val.toFixed(2), "Close"]
            }
          />
          {/* volume bars behind */}
          <Bar yAxisId="vol" dataKey="vol_pct" fill="#1e3a5f" opacity={0.6} radius={[1, 1, 0, 0]} />
          {/* close price area */}
          <Area
            yAxisId="price"
            type="monotone"
            dataKey="close"
            stroke="#38bdf8"
            strokeWidth={1.5}
            fill="url(#priceGrad)"
            dot={false}
          />
          <defs>
            <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#38bdf8" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} />
            </linearGradient>
          </defs>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

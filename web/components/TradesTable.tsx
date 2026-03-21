"use client";

import clsx from "clsx";
import type { TradeRow } from "@/lib/api";

interface TradesTableProps {
  trades: TradeRow[];
  maxRows?: number;
}

export function TradesTable({ trades, maxRows = 100 }: TradesTableProps) {
  const rows = trades.slice(0, maxRows);

  return (
    <div className="overflow-x-auto max-h-64 overflow-y-auto">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-gray-900">
          <tr className="text-gray-500 border-b border-gray-800">
            <th className="text-left py-1 pr-3">Time</th>
            <th className="text-left py-1 pr-3">Symbol</th>
            <th className="text-left py-1 pr-3">Side</th>
            <th className="text-right py-1 pr-3">Qty</th>
            <th className="text-right py-1 pr-3">Price</th>
            <th className="text-right py-1 pr-3">Comm.</th>
            <th className="text-left py-1">Strategy</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t, i) => (
            <tr key={i} className="border-b border-gray-800/40 hover:bg-gray-800/30">
              <td className="py-1 pr-3 text-gray-400 font-mono">{t.time.slice(0, 10)}</td>
              <td className="py-1 pr-3 text-gray-200 font-mono">{t.symbol}</td>
              <td className={clsx("py-1 pr-3 font-medium", t.side === "BUY" ? "text-emerald-400" : "text-red-400")}>
                {t.side}
              </td>
              <td className="py-1 pr-3 text-right text-gray-300">{t.qty}</td>
              <td className="py-1 pr-3 text-right text-gray-300">¥{t.price.toFixed(2)}</td>
              <td className="py-1 pr-3 text-right text-gray-500">¥{t.commission.toFixed(2)}</td>
              <td className="py-1 text-gray-400 truncate max-w-[100px]">{t.strategy}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {trades.length > maxRows && (
        <p className="text-xs text-gray-500 text-center py-2">
          Showing {maxRows} of {trades.length} trades
        </p>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import clsx from "clsx";
import type { TradeRow } from "@/lib/api";

interface Props {
  trades: TradeRow[];
}

const PAGE = 20;

export function TradesTable({ trades }: Props) {
  const [page, setPage] = useState(0);
  const total = Math.ceil(trades.length / PAGE);
  const slice = trades.slice(page * PAGE, (page + 1) * PAGE);

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-400 border-b border-gray-800">
              {["Time", "Symbol", "Side", "Qty", "Price", "Commission", "Strategy"].map((h) => (
                <th key={h} className="text-left px-3 py-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {slice.map((t, i) => (
              <tr
                key={i}
                className="border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors"
              >
                <td className="px-3 py-1.5 text-gray-400">{t.time.slice(0, 16)}</td>
                <td className="px-3 py-1.5 font-mono">{t.symbol}</td>
                <td className="px-3 py-1.5">
                  <span
                    className={clsx(
                      "px-1.5 py-0.5 rounded text-xs font-semibold",
                      t.side === "BUY"
                        ? "bg-emerald-900/60 text-emerald-300"
                        : "bg-rose-900/60 text-rose-300"
                    )}
                  >
                    {t.side}
                  </span>
                </td>
                <td className="px-3 py-1.5 font-mono">{t.qty.toLocaleString()}</td>
                <td className="px-3 py-1.5 font-mono">{t.price.toFixed(2)}</td>
                <td className="px-3 py-1.5 font-mono text-gray-400">
                  {t.commission.toFixed(2)}
                </td>
                <td className="px-3 py-1.5 text-gray-400">{t.strategy}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {total > 1 && (
        <div className="flex items-center gap-2 justify-end">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="text-xs px-3 py-1 bg-gray-800 rounded disabled:opacity-40 hover:bg-gray-700"
          >
            ‹ Prev
          </button>
          <span className="text-xs text-gray-400">
            {page + 1} / {total}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(total - 1, p + 1))}
            disabled={page === total - 1}
            className="text-xs px-3 py-1 bg-gray-800 rounded disabled:opacity-40 hover:bg-gray-700"
          >
            Next ›
          </button>
        </div>
      )}
    </div>
  );
}

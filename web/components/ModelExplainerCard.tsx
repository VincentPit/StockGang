"use client";

import clsx from "clsx";
import { Trash2 } from "lucide-react";
import type { StoredModelInfo } from "@/lib/api";

interface ModelExplainerCardProps {
  model: StoredModelInfo;
  features?: Array<{ feature: string; importance: number }>;
  onDelete?: (symbol: string) => void;
  onAnalyse?: (symbol: string) => void;
}

function fmt(n?: number, d = 3) { return n == null ? "—" : n.toFixed(d); }
function fmtDate(ts?: number) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleDateString("en-US", { year: "2-digit", month: "short", day: "numeric" });
}

export function ModelExplainerCard({ model, features, onDelete, onAnalyse }: ModelExplainerCardProps) {
  const sortedFeats = features
    ? [...features].sort((a, b) => b.importance - a.importance).slice(0, 10)
    : [];
  const maxImp = sortedFeats[0]?.importance ?? 0.001;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-semibold text-gray-100 font-mono">{model.symbol}</div>
          <div className="text-[10px] text-gray-500 mt-0.5">{model.strategy_id} · {model.model_id}</div>
        </div>
        <div className="flex items-center gap-1">
          {onAnalyse && (
            <button
              onClick={() => onAnalyse(model.symbol)}
              className="px-2 py-0.5 rounded text-[11px] text-indigo-400 hover:bg-indigo-900/40 transition-colors font-medium"
              title="Analyse this symbol"
            >
              Analyse →
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(model.symbol)}
              className="p-1 rounded text-gray-600 hover:text-red-400 transition-colors"
              title="Delete model"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[10px]">
        {[
          { label: "Trained", value: fmtDate(model.trained_at) },
          { label: "OOS Acc", value: model.oos_accuracy != null ? `${(model.oos_accuracy * 100).toFixed(1)}%` : "—" },
          { label: "Bars", value: String(model.bar_count) },
          { label: "Last Bar", value: model.last_bar_date ?? "—" },
          { label: "Features", value: String(model.feature_cols.length) },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-800/40 rounded p-1.5">
            <div className="text-gray-500 uppercase tracking-wide">{label}</div>
            <div className="text-gray-200 font-medium mt-0.5">{value}</div>
          </div>
        ))}
      </div>

      {sortedFeats.length > 0 && (
        <div>
          <div className="text-[10px] text-gray-500 uppercase mb-1.5">Feature Importance</div>
          <div className="space-y-1">
            {sortedFeats.map(f => (
              <div key={f.feature} className="flex items-center gap-2">
                <div className="w-28 text-[10px] text-gray-400 truncate" title={f.feature}>
                  {f.feature}
                </div>
                <div className="flex-1 h-1.5 rounded bg-gray-800">
                  <div
                    className="h-full rounded bg-indigo-500"
                    style={{ width: `${(f.importance / maxImp) * 100}%` }}
                  />
                </div>
                <div className="text-[10px] text-gray-400 w-10 text-right">
                  {fmt(f.importance)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

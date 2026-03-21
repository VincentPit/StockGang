"use client";

import clsx from "clsx";
import type { CausalNode, DataScope, GateCheck } from "@/lib/api";

function Bar({ pct, positive }: { pct: number; positive: boolean }) {
  return (
    <div className="flex-1 h-2 rounded-full bg-gray-800 overflow-hidden">
      <div
        className={clsx("h-full rounded-full", positive ? "bg-emerald-500" : "bg-red-500")}
        style={{ width: `${Math.min(100, Math.abs(pct))}%` }}
      />
    </div>
  );
}

interface CausalTracePanelProps {
  nodes: CausalNode[];
  gates?: GateCheck[];
  scope?: DataScope;
}

export function CausalTracePanel({ nodes, gates = [], scope }: CausalTracePanelProps) {
  const sorted = [...nodes].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const maxAbs = Math.max(...sorted.map(n => Math.abs(n.contribution)), 0.001);

  return (
    <div className="space-y-4">
      {scope && (
        <div className="flex flex-wrap gap-3 text-xs text-gray-400">
          <span className="font-mono">{scope.start_date} → {scope.end_date}</span>
          <span>{scope.bars} bars</span>
          <span className={clsx("font-semibold",
            scope.trend === "UPTREND" ? "text-emerald-400" :
            scope.trend === "DOWNTREND" ? "text-red-400" : "text-yellow-400"
          )}>{scope.trend}</span>
        </div>
      )}

      <div className="space-y-1.5">
        {sorted.map(node => {
          const pct = (Math.abs(node.contribution) / maxAbs) * 100;
          const pos = node.direction === "positive";
          return (
            <div key={node.factor} className="flex items-center gap-3 group">
              <div className="w-32 shrink-0 text-xs text-gray-300 truncate" title={node.label}>
                {node.label}
              </div>
              <Bar pct={pct} positive={pos} />
              <div className={clsx("w-14 text-right text-xs font-mono shrink-0",
                pos ? "text-emerald-400" : "text-red-400"
              )}>
                {node.contribution >= 0 ? "+" : ""}{node.contribution.toFixed(3)}
              </div>
              <div className="hidden group-hover:block text-[10px] text-gray-500 truncate max-w-xs">
                {node.description}
              </div>
            </div>
          );
        })}
      </div>

      {gates.length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-medium text-gray-400 mb-2">Gate Checks</div>
          <div className="flex flex-wrap gap-2">
            {gates.map(g => (
              <div
                key={g.check}
                title={`${g.label}: actual ${g.actual.toFixed(3)} vs threshold ${g.threshold.toFixed(3)} — ${g.note}`}
                className={clsx(
                  "px-2 py-0.5 rounded-full text-[10px] font-medium border",
                  g.passed
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    : "bg-red-500/10 border-red-500/30 text-red-400"
                )}
              >
                {g.passed ? "✓" : "✗"} {g.label}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

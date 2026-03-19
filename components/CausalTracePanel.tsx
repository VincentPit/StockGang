"use client";

/**
 * CausalTracePanel — explains WHY a stock was selected, ranked, or recommended.
 *
 * Shows:
 *  1. Decision badge + data scope (date range, price range, trend)
 *  2. Per-factor score bars with direction colouring
 *  3. Gate checks (passed/failed thresholds)
 *  4. For ML signal nodes: BUY/HOLD/SELL probability pills
 */

import clsx from "clsx";
import { CheckCircle2, XCircle, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { CausalNode, DataScope, GateCheck } from "@/lib/api";

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  decision: string;            // "SELECTED" | "RANKED" | "RECOMMENDED" | "REJECTED"
  score: number;
  rank?: number;
  universeSize?: number;
  causalNodes: CausalNode[];
  dataScope?: DataScope;
  gateChecks?: GateCheck[];
  compact?: boolean;           // tighter padding for inline use
}

// ── Small helpers ─────────────────────────────────────────────────────────────

function TrendBadge({ trend }: { trend: string }) {
  if (trend === "UPTREND")
    return (
      <span className="inline-flex items-center gap-1 text-emerald-400 text-[10px] font-semibold">
        <TrendingUp className="w-3 h-3" /> UPTREND
      </span>
    );
  if (trend === "DOWNTREND")
    return (
      <span className="inline-flex items-center gap-1 text-rose-400 text-[10px] font-semibold">
        <TrendingDown className="w-3 h-3" /> DOWNTREND
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 text-amber-400 text-[10px] font-semibold">
      <Minus className="w-3 h-3" /> SIDEWAYS
    </span>
  );
}

function DecisionBadge({ decision }: { decision: string }) {
  const styles: Record<string, string> = {
    SELECTED:    "bg-emerald-900/60 border-emerald-700 text-emerald-300",
    RECOMMENDED: "bg-sky-900/60 border-sky-700 text-sky-300",
    RANKED:      "bg-gray-800 border-gray-600 text-gray-300",
    REJECTED:    "bg-rose-900/60 border-rose-700 text-rose-300",
  };
  return (
    <span
      className={clsx(
        "text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border",
        styles[decision] ?? styles.RANKED,
      )}
    >
      {decision}
    </span>
  );
}

// ── Single factor row ─────────────────────────────────────────────────────────

function NodeBar({ node }: { node: CausalNode }) {
  const barPct  = Math.round(node.norm_value * 100);
  const barColor =
    node.direction === "positive" ? "bg-emerald-500" :
    node.direction === "negative" ? "bg-rose-500"    :
    "bg-amber-500";

  const ext    = node.extras as Record<string, unknown> | undefined;
  const isML   = node.factor === "model_signal" && !!ext?.model_trained;
  const isFund = node.factor === "fundamentals"  && !!ext;

  return (
    <div className="space-y-1.5">
      {/* header row */}
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="font-semibold text-gray-200 shrink-0">{node.label}</span>
        <div className="flex items-center gap-3 shrink-0 ml-auto">
          {node.percentile && (
            <span className="text-[10px] text-gray-500">{node.percentile}</span>
          )}
          <span className="font-mono tabular-nums text-gray-300">
            {node.contribution >= 0 ? "+" : ""}{node.contribution.toFixed(3)}
          </span>
          <span className="text-[10px] text-gray-600 w-12 text-right">
            {(node.weight * 100).toFixed(0)}% wt
          </span>
        </div>
      </div>

      {/* progress bar */}
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all duration-300", barColor)}
          style={{ width: `${barPct}%` }}
        />
      </div>

      {/* description */}
      <p className="text-[10px] text-gray-500 leading-snug">{node.description}</p>

      {/* ML signal: BUY / HOLD / SELL probability pills */}
      {isML && (
        <div className="flex gap-3 mt-0.5">
          {[
            { k: "p_buy",  label: "BUY",  cls: "text-emerald-400 bg-emerald-950/60 border-emerald-800/70" },
            { k: "p_hold", label: "HOLD", cls: "text-amber-400 bg-amber-950/60 border-amber-800/70" },
            { k: "p_sell", label: "SELL", cls: "text-rose-400 bg-rose-950/60 border-rose-800/70" },
          ].map(({ k, label, cls }) => (
            <span
              key={k}
              className={clsx(
                "inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded border",
                cls,
              )}
            >
              {label}{" "}{(((ext?.[k] as number) ?? 0) * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      {/* Fundamentals: sub-score detail pills */}
      {isFund && (
        <div className="flex flex-wrap gap-2 mt-0.5">
          {[
            { k: "value_score",   label: "Value" },
            { k: "growth_score",  label: "Growth" },
            { k: "quality_score", label: "Quality" },
            { k: "pe_ttm",        label: "P/E" },
            { k: "pb",            label: "P/B" },
            { k: "roe",           label: "ROE%" },
          ].map(({ k, label }) => {
            const v = ext?.[k];
            if (v === undefined) return null;
            const display = typeof v === "number" ? v.toFixed(1) : `${v}`;
            return (
              <span
                key={k}
                className="text-[10px] bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-gray-400"
              >
                <span className="text-gray-500">{label} </span>
                {display}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function CausalTracePanel({
  decision,
  score,
  rank,
  universeSize,
  causalNodes,
  dataScope,
  gateChecks,
  compact = false,
}: Props) {
  if (!causalNodes || causalNodes.length === 0) return null;

  return (
    <div className="rounded-xl border border-violet-900/40 bg-gray-950 overflow-hidden">
      {/* ── header ── */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-violet-950/30 border-b border-violet-900/40">
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="text-[11px] text-gray-400 font-medium">Causal trace</span>
          <DecisionBadge decision={decision.toUpperCase()} />
          {rank != null && (
            <span className="text-[10px] text-gray-500">
              Rank #{rank}{universeSize ? ` of ${universeSize}` : ""}
            </span>
          )}
        </div>
        <span className="font-mono text-sm font-bold text-violet-300">{score.toFixed(3)}</span>
      </div>

      <div className={clsx("px-4 space-y-4", compact ? "py-3" : "py-4")}>
        {/* ── data scope ── */}
        {dataScope && (
          <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-[11px] pb-3 border-b border-gray-800">
            <span>
              <span className="text-gray-500">Period </span>
              <span className="text-gray-300">
                {dataScope.start_date} → {dataScope.end_date}
              </span>
              <span className="text-gray-600 ml-1">({dataScope.bars} bars)</span>
            </span>
            <span>
              <span className="text-gray-500">Price </span>
              <span className="text-gray-300">
                ¥{dataScope.price_start.toLocaleString()} → ¥{dataScope.price_end.toLocaleString()}
              </span>
            </span>
            <span>
              <span className="text-gray-500">Range </span>
              <span className="text-gray-300">
                ¥{dataScope.price_min.toLocaleString()} – ¥{dataScope.price_max.toLocaleString()}
              </span>
            </span>
            <TrendBadge trend={dataScope.trend} />
          </div>
        )}

        {/* ── score nodes ── */}
        <div className={clsx("space-y-4", compact && "space-y-3")}>
          {causalNodes.map((node) => (
            <NodeBar key={node.factor} node={node} />
          ))}
        </div>

        {/* ── gate checks ── */}
        {gateChecks && gateChecks.length > 0 && (
          <div className="space-y-1 pt-2 border-t border-gray-800">
            {gateChecks.map((g) => (
              <div key={g.check} className="flex items-center gap-2 text-[11px]">
                {g.passed ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-rose-500 shrink-0" />
                )}
                <span className={g.passed ? "text-gray-500" : "text-rose-400"}>
                  {g.note || g.label}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

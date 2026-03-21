"use client";

interface ProgressBarProps {
  /** 0–100 progress percentage (preferred prop name) */
  value?: number;
  /** Alias for value — accepted for back-compat with TrainLoopPanel */
  pct?: number;
  /** Optional text displayed above the bar (also accepts `step` alias) */
  label?: string;
  step?: string;
  /** When true the bar pulses to indicate ongoing work */
  running?: boolean;
  className?: string;
}

export function ProgressBar({ value, pct, label, step, running, className = "" }: ProgressBarProps) {
  const raw = value ?? pct ?? 0;
  const displayLabel = label ?? step;
  const clamped = Math.min(100, Math.max(0, raw));
  return (
    <div className={`w-full ${className}`}>
      {displayLabel && (
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>{displayLabel}</span>
          <span>{clamped.toFixed(0)}%</span>
        </div>
      )}
      <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-full bg-indigo-500 rounded-full transition-all duration-300 ${running && clamped === 0 ? "animate-pulse w-full opacity-40" : ""}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

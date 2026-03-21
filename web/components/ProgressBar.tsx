"use client";

interface ProgressBarProps {
  value: number;   // 0–100
  label?: string;
  className?: string;
}

export function ProgressBar({ value, label, className = "" }: ProgressBarProps) {
  const clamped = Math.min(100, Math.max(0, value));
  return (
    <div className={`w-full ${className}`}>
      {label && (
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>{label}</span>
          <span>{clamped.toFixed(0)}%</span>
        </div>
      )}
      <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

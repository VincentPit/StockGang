"use client";

import { useState, useEffect, useCallback } from "react";
import clsx from "clsx";
import {
  RefreshCcw,
  Pause,
  Play,
  Zap,
  Activity,
  Database,
  Brain,
  Settings2,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  getSchedulerStatus,
  triggerScheduler,
  pauseScheduler,
  resumeScheduler,
  type SchedulerStatus,
} from "@/lib/api";

function timeAgo(iso?: string): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { color: string; icon: React.ReactNode }> = {
    idle:    { color: "text-gray-400 bg-gray-800",     icon: <Clock className="w-3 h-3" /> },
    running: { color: "text-blue-400 bg-blue-900/40",  icon: <Loader2 className="w-3 h-3 animate-spin" /> },
    success: { color: "text-emerald-400 bg-emerald-900/30", icon: <CheckCircle2 className="w-3 h-3" /> },
    failed:  { color: "text-red-400 bg-red-900/30",    icon: <XCircle className="w-3 h-3" /> },
  };
  const s = map[status] ?? map.idle;
  return (
    <span className={clsx("inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full", s.color)}>
      {s.icon}{status}
    </span>
  );
}

export default function SchedulerPanel() {
  const [status, setStatus] = useState<SchedulerStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const s = await getSchedulerStatus();
      setStatus(s);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh every 10 seconds
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 10_000);
    return () => clearInterval(timer);
  }, [refresh]);

  const handleTrigger = async (jobType: string) => {
    setActionLoading(jobType);
    try {
      await triggerScheduler(jobType);
      setTimeout(refresh, 1000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setActionLoading(null);
    }
  };

  const handlePauseResume = async () => {
    setActionLoading("pause");
    try {
      if (status?.paused) {
        await resumeScheduler();
      } else {
        await pauseScheduler();
      }
      setTimeout(refresh, 500);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-8 flex items-center justify-center gap-2 text-gray-400">
        <Loader2 className="w-5 h-5 animate-spin" /> Loading scheduler status…
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-gray-100">Auto-Update Pipeline</h2>
          {status?.scheduler_running ? (
            <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> Active
            </span>
          ) : (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-gray-700 text-gray-400">
              Inactive
            </span>
          )}
          {status?.paused && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-300 border border-yellow-500/30">
              Paused
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => refresh()} className="p-1.5 rounded hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors">
            <RefreshCcw className="w-4 h-4" />
          </button>
          <button
            onClick={handlePauseResume}
            disabled={actionLoading === "pause"}
            className={clsx(
              "flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors",
              status?.paused
                ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                : "bg-yellow-600/80 hover:bg-yellow-500 text-white"
            )}
          >
            {status?.paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            {status?.paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>

      <p className="text-xs text-gray-500">
        Automatically fetches new market data, retrains ML models, and updates strategy
        parameters daily after market close. Runs a deep auto-tune every weekend.
      </p>

      {error && (
        <div className="flex items-start gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />{error}
        </div>
      )}

      {/* Pipeline steps */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Data Refresh */}
        <div className="bg-gray-800/60 rounded-lg p-4 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-200">
              <Database className="w-4 h-4 text-blue-400" /> Data Refresh
            </div>
            <StatusBadge status={status?.data_status ?? "idle"} />
          </div>
          <div className="text-[10px] text-gray-500 space-y-0.5">
            <p>Last run: {timeAgo(status?.data_last_run)}</p>
            <p>Symbols: {status?.data_symbols_updated ?? 0}</p>
            <p>Runs: {status?.data_run_count ?? 0} ({status?.data_fail_count ?? 0} failed)</p>
            {status?.data_last_error && <p className="text-red-400 truncate">⚠ {status.data_last_error}</p>}
          </div>
          <button
            onClick={() => handleTrigger("data")}
            disabled={actionLoading !== null}
            className="w-full flex items-center justify-center gap-1 py-1.5 rounded bg-blue-600/30 hover:bg-blue-600/50 text-blue-300 text-[11px] font-medium transition-colors disabled:opacity-40"
          >
            {actionLoading === "data" ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
            Fetch Now
          </button>
        </div>

        {/* Model Retrain */}
        <div className="bg-gray-800/60 rounded-lg p-4 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-200">
              <Brain className="w-4 h-4 text-purple-400" /> Model Retrain
            </div>
            <StatusBadge status={status?.retrain_status ?? "idle"} />
          </div>
          <div className="text-[10px] text-gray-500 space-y-0.5">
            <p>Last run: {timeAgo(status?.retrain_last_run)}</p>
            <p>Models updated: {status?.retrain_models_updated ?? 0}</p>
            <p>Avg OOS: {status?.last_oos_accuracy != null ? `${(status.last_oos_accuracy * 100).toFixed(1)}%` : "—"}</p>
            {status?.retrain_last_error && <p className="text-red-400 truncate">⚠ {status.retrain_last_error}</p>}
          </div>
          <button
            onClick={() => handleTrigger("retrain")}
            disabled={actionLoading !== null}
            className="w-full flex items-center justify-center gap-1 py-1.5 rounded bg-purple-600/30 hover:bg-purple-600/50 text-purple-300 text-[11px] font-medium transition-colors disabled:opacity-40"
          >
            {actionLoading === "retrain" ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
            Retrain Now
          </button>
        </div>

        {/* Strategy Update */}
        <div className="bg-gray-800/60 rounded-lg p-4 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-200">
              <Settings2 className="w-4 h-4 text-amber-400" /> Strategy Update
            </div>
            <StatusBadge status={status?.strategy_status ?? "idle"} />
          </div>
          <div className="text-[10px] text-gray-500 space-y-0.5">
            <p>Last run: {timeAgo(status?.strategy_last_run)}</p>
            <p>Sharpe: {status?.last_sharpe != null ? status.last_sharpe.toFixed(2) : "—"}</p>
            <p>PF: {status?.last_profit_factor != null ? status.last_profit_factor.toFixed(2) : "—"}</p>
            <p>Score: {status?.last_model_score != null ? `${status.last_model_score.toFixed(0)}/100` : "—"}</p>
          </div>
          <button
            onClick={() => handleTrigger("strategy")}
            disabled={actionLoading !== null}
            className="w-full flex items-center justify-center gap-1 py-1.5 rounded bg-amber-600/30 hover:bg-amber-600/50 text-amber-300 text-[11px] font-medium transition-colors disabled:opacity-40"
          >
            {actionLoading === "strategy" ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
            Update Now
          </button>
        </div>
      </div>

      {/* Full pipeline trigger */}
      <button
        onClick={() => handleTrigger("full")}
        disabled={actionLoading !== null}
        className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-sm font-semibold transition-all disabled:opacity-40"
      >
        {actionLoading === "full" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
        Run Full Pipeline (Data → Retrain → Strategy)
      </button>

      {/* Next scheduled runs */}
      {status?.next_scheduled_runs && Object.keys(status.next_scheduled_runs).length > 0 && (
        <div className="bg-gray-800/30 rounded-lg p-3">
          <h3 className="text-xs font-medium text-gray-300 mb-2">Next Scheduled Runs</h3>
          <div className="space-y-1">
            {Object.entries(status.next_scheduled_runs).map(([id, time]) => (
              <div key={id} className="flex items-center justify-between text-[10px]">
                <span className="text-gray-400">{id.replace(/_/g, " ")}</span>
                <span className="text-gray-500">{new Date(time).toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tracked symbols */}
      {status?.tracked_symbols && status.tracked_symbols.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {status.tracked_symbols.map(sym => (
            <span key={sym} className="text-[10px] px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 border border-gray-700">
              {sym}
            </span>
          ))}
        </div>
      )}

      {/* History toggle */}
      <button
        onClick={() => setShowHistory(!showHistory)}
        className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
      >
        {showHistory ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        Recent Runs ({status?.recent_runs?.length ?? 0})
      </button>

      {showHistory && status?.recent_runs && status.recent_runs.length > 0 && (
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {[...status.recent_runs].reverse().map((run, i) => (
            <div key={i} className="flex items-center justify-between bg-gray-800/40 rounded px-3 py-1.5 text-[10px]">
              <div className="flex items-center gap-2">
                <StatusBadge status={run.status} />
                <span className="text-gray-300 font-medium">{run.type.replace(/_/g, " ")}</span>
              </div>
              <div className="flex items-center gap-3 text-gray-500">
                {run.error && <span className="text-red-400 truncate max-w-[200px]" title={run.error}>⚠ {run.error}</span>}
                <span>{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

"use client";

import DashboardShell from "@/components/DashboardShell";
import SchedulerPanel from "@/components/SchedulerPanel";

export default function SchedulerPage() {
  return (
    <DashboardShell>
      <div className="max-w-5xl mx-auto space-y-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Auto-Update Scheduler</h1>
          <p className="text-sm text-gray-500 mt-1">
            Monitor and control the automated pipeline that keeps your data fresh,
            models retrained, and strategy parameters optimised.
          </p>
        </div>
        <SchedulerPanel />
      </div>
    </DashboardShell>
  );
}

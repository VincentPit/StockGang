import { render, screen } from "@testing-library/react";
import SchedulerPanel from "@/components/SchedulerPanel";

jest.mock("@/lib/api", () => ({
  getSchedulerStatus: jest.fn().mockResolvedValue({
    scheduler_running: false, enabled: false, paused: false,
    data_status: "idle", retrain_status: "idle", strategy_status: "idle",
    data_symbols_updated: 0, data_run_count: 0, data_fail_count: 0,
    retrain_models_updated: 0, retrain_run_count: 0, retrain_fail_count: 0,
    strategy_run_count: 0,
    next_scheduled_runs: {}, tracked_symbols: [], recent_runs: [],
  }),
  triggerScheduler:    jest.fn(),
  pauseScheduler:      jest.fn(),
  resumeScheduler:     jest.fn(),
  getSchedulerHistory: jest.fn().mockResolvedValue({ runs: [], total: 0 }),
}));

describe("SchedulerPanel", () => {
  it("renders the Auto-Update Pipeline heading after initial load", async () => {
    render(<SchedulerPanel />);
    expect(await screen.findByText("Auto-Update Pipeline")).toBeInTheDocument();
  });
});

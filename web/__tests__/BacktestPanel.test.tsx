import { render, screen } from "@testing-library/react";
import BacktestPanel from "@/components/BacktestPanel";

jest.mock("@/lib/api", () => ({
  startBacktest: jest.fn(),
  getBacktest:   jest.fn(),
  pollJob:       jest.fn(),
  listJobs:      jest.fn().mockResolvedValue([]),
  getUniverse:   jest.fn().mockResolvedValue({ symbols: [] }),
}));

describe("BacktestPanel", () => {
  it("renders the Backtest heading", () => {
    render(<BacktestPanel />);
    expect(screen.getByText("Backtest")).toBeInTheDocument();
  });
});

/**
 * Smoke test: AdvisorPanel renders without crashing and displays its heading.
 * The full panel state machine is exercised manually for now; this guards
 * against import/JSX/typing regressions.
 */
import { render, screen } from "@testing-library/react";
import AdvisorPanel from "@/components/AdvisorPanel";

jest.mock("@/lib/api", () => ({
  startAnalyze:    jest.fn(),
  getAnalyze:      jest.fn(),
  startRecommend:  jest.fn(),
  pollRecommend:   jest.fn(),
  listModels:      jest.fn().mockResolvedValue({ models: [] }),
  deleteModel:     jest.fn(),
  getSectors:      jest.fn().mockResolvedValue({ sectors: [] }),
  pollJob:         jest.fn(),
}));

describe("AdvisorPanel", () => {
  it("renders the Advisor heading", () => {
    render(<AdvisorPanel />);
    expect(screen.getByText("Advisor")).toBeInTheDocument();
  });
});

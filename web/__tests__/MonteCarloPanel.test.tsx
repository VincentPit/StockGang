import { render, screen } from "@testing-library/react";
import MonteCarloPanel from "@/components/MonteCarloPanel";

jest.mock("@/lib/api", () => ({
  startMonteCarlo: jest.fn(),
  getMonteCarlo:   jest.fn(),
  pollJob:         jest.fn(),
}));

describe("MonteCarloPanel", () => {
  it("renders the Monte Carlo heading", () => {
    render(<MonteCarloPanel />);
    expect(screen.getByText("Monte Carlo Simulation")).toBeInTheDocument();
  });
});

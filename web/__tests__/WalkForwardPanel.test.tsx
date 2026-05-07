import { render, screen } from "@testing-library/react";
import WalkForwardPanel from "@/components/WalkForwardPanel";

jest.mock("@/lib/api", () => ({
  startWalkForward: jest.fn(),
  getWalkForward:   jest.fn(),
  pollJob:          jest.fn(),
}));

describe("WalkForwardPanel", () => {
  it("renders the Walk-Forward heading", () => {
    render(<WalkForwardPanel />);
    expect(screen.getByText("Walk-Forward Validation")).toBeInTheDocument();
  });
});

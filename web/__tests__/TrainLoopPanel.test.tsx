import { render, screen } from "@testing-library/react";
import TrainLoopPanel from "@/components/TrainLoopPanel";

jest.mock("@/lib/api", () => ({
  startTrainLoop: jest.fn(),
  getTrainLoop:   jest.fn(),
  pollJob:        jest.fn(),
}));

describe("TrainLoopPanel", () => {
  it("renders the Configuration heading", () => {
    render(<TrainLoopPanel />);
    expect(screen.getByText(/Configuration/i)).toBeInTheDocument();
  });
});

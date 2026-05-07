import { render, screen } from "@testing-library/react";
import AutoTunePanel from "@/components/AutoTunePanel";

jest.mock("@/lib/api", () => ({
  startAutoTune: jest.fn(),
  getAutoTune:   jest.fn(),
  pollJob:       jest.fn(),
}));

describe("AutoTunePanel", () => {
  it("renders the Configuration heading", () => {
    render(<AutoTunePanel />);
    expect(screen.getByText(/Configuration/i)).toBeInTheDocument();
  });
});

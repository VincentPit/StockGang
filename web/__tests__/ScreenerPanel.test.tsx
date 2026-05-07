import { render, screen } from "@testing-library/react";
import ScreenerPanel from "@/components/ScreenerPanel";

jest.mock("@/lib/api", () => ({
  startScreen: jest.fn(),
  getScreen:   jest.fn(),
  pollJob:     jest.fn(),
}));

describe("ScreenerPanel", () => {
  it("renders without crashing", () => {
    const { container } = render(<ScreenerPanel />);
    expect(container.firstChild).toBeInTheDocument();
  });
});

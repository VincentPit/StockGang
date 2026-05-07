import { render, screen } from "@testing-library/react";
import DashboardShell from "@/components/DashboardShell";

jest.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

jest.mock("@/lib/api", () => ({
  listModels: jest.fn().mockResolvedValue({ models: [] }),
}));

describe("DashboardShell", () => {
  it("renders nav items and child content", () => {
    render(
      <DashboardShell>
        <div>child-marker</div>
      </DashboardShell>
    );
    // Multiple matches because nav items appear in both desktop and mobile nav
    expect(screen.getAllByText("Workflow").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Screener").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Backtest").length).toBeGreaterThan(0);
    expect(screen.getByText("child-marker")).toBeInTheDocument();
  });
});

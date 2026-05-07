import { render } from "@testing-library/react";
import { NavChart } from "@/components/NavChart";

describe("NavChart", () => {
  it("renders a recharts container for the given series", () => {
    const data = [
      { date: "2025-01-01", nav: 1_000_000 },
      { date: "2025-01-02", nav: 1_010_000 },
      { date: "2025-01-03", nav:   995_000 },
    ];
    const { container } = render(<NavChart data={data} />);
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();
  });

  it("renders even when nav values are missing (defensive)", () => {
    const data = [
      { date: "2025-01-01", nav: undefined as unknown as number },
      { date: "2025-01-02", nav: 1_005_000 },
    ];
    const { container } = render(<NavChart data={data} />);
    expect(container.firstChild).toBeInTheDocument();
  });
});

import { render, screen } from "@testing-library/react";
import { ProgressBar } from "@/components/ProgressBar";

describe("ProgressBar", () => {
  it("renders the label and rounded percentage", () => {
    render(<ProgressBar value={42.7} label="Loading…" />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
    expect(screen.getByText("43%")).toBeInTheDocument();
  });

  it("clamps values above 100", () => {
    render(<ProgressBar value={250} label="Done" />);
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("clamps values below 0", () => {
    render(<ProgressBar value={-10} label="Neg" />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  it("falls back to pct/step aliases for back-compat", () => {
    render(<ProgressBar pct={55} step="Step 2" />);
    expect(screen.getByText("Step 2")).toBeInTheDocument();
    expect(screen.getByText("55%")).toBeInTheDocument();
  });

  it("hides the label row when no label/step is provided", () => {
    const { container } = render(<ProgressBar value={50} />);
    expect(container.querySelector("span")).toBeNull();
  });
});

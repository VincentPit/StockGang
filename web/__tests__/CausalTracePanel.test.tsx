import { render, screen } from "@testing-library/react";
import { CausalTracePanel } from "@/components/CausalTracePanel";
import type { CausalNode, DataScope, GateCheck } from "@/lib/api";

const node = (over: Partial<CausalNode>): CausalNode => ({
  factor:       "rsi_14",
  label:        "RSI(14)",
  description:  "Momentum oscillator",
  raw_value:    52.3,
  norm_value:   0.52,
  weight:       0.25,
  contribution: 0.12,
  direction:    "positive",
  percentile:   "p70",
  ...over,
});

const scope: DataScope = {
  start_date:  "2024-01-01",
  end_date:    "2025-01-01",
  bars:        252,
  price_start: 100,
  price_end:   115,
  price_min:   90,
  price_max:   120,
  trend:       "UPTREND",
};

const gate = (over: Partial<GateCheck>): GateCheck => ({
  check:     "min_bars",
  label:     "Min Bars",
  threshold: 100,
  actual:    252,
  passed:    true,
  note:      "ok",
  ...over,
});

describe("CausalTracePanel", () => {
  it("renders sorted nodes with positive/negative styling", () => {
    const nodes = [
      node({ factor: "a", label: "A", contribution:  0.05 }),
      node({ factor: "b", label: "B", contribution: -0.30, direction: "negative" }),
      node({ factor: "c", label: "C", contribution:  0.15 }),
    ];
    render(<CausalTracePanel nodes={nodes} />);
    expect(screen.getByText("A")).toBeInTheDocument();
    expect(screen.getByText("B")).toBeInTheDocument();
    expect(screen.getByText("C")).toBeInTheDocument();
    // Largest |contribution| (B at -0.30) shows formatted value
    expect(screen.getByText("-0.300")).toBeInTheDocument();
  });

  it("renders the scope summary when provided", () => {
    render(<CausalTracePanel nodes={[node({})]} scope={scope} />);
    expect(screen.getByText("UPTREND")).toBeInTheDocument();
    expect(screen.getByText("252 bars")).toBeInTheDocument();
  });

  it("renders gate checks with pass/fail icons", () => {
    const gates = [gate({ label: "G1" }), gate({ check: "g2", label: "G2", passed: false })];
    render(<CausalTracePanel nodes={[node({})]} gates={gates} />);
    expect(screen.getByText(/✓ G1/)).toBeInTheDocument();
    expect(screen.getByText(/✗ G2/)).toBeInTheDocument();
  });
});

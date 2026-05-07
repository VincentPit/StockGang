import { render, screen } from "@testing-library/react";
import { TradesTable } from "@/components/TradesTable";
import type { TradeRow } from "@/lib/api";

const mkTrade = (over: Partial<TradeRow> = {}): TradeRow => ({
  time:       "2025-01-02T09:30:00",
  symbol:     "sh600519",
  side:       "BUY",
  qty:        100,
  price:      1500.5,
  commission: 0.45,
  strategy:   "lgbm_core",
  ...over,
});

describe("TradesTable", () => {
  it("renders the column headers and a trade row", () => {
    render(<TradesTable trades={[mkTrade()]} />);
    expect(screen.getByText("Time")).toBeInTheDocument();
    expect(screen.getByText("Symbol")).toBeInTheDocument();
    expect(screen.getByText("sh600519")).toBeInTheDocument();
    expect(screen.getByText("BUY")).toBeInTheDocument();
  });

  it("clips to maxRows and shows the overflow line", () => {
    const trades = Array.from({ length: 150 }, (_, i) => mkTrade({ symbol: `S${i}` }));
    render(<TradesTable trades={trades} maxRows={10} />);
    expect(screen.getByText(/Showing 10 of 150 trades/)).toBeInTheDocument();
    expect(screen.queryByText("S140")).toBeNull();
  });

  it("renders SELLs distinctly from BUYs", () => {
    render(<TradesTable trades={[mkTrade({ side: "SELL" })]} />);
    expect(screen.getByText("SELL")).toBeInTheDocument();
  });
});

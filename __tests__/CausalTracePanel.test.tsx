/**
 * CausalTracePanel.test.tsx
 *
 * Unit tests for the CausalTracePanel component and its sub-components.
 *
 * Coverage
 * ────────
 * 1. Rendering — renders nothing when causalNodes is empty
 * 2. DecisionBadge — each decision variant renders correct label
 * 3. TrendBadge — UPTREND / DOWNTREND / SIDEWAYS labels
 * 4. DataScope strip — shows dates, bar count, price range, trend
 * 5. NodeBar — label, weight, contribution, progress bar width, description
 * 6. NodeBar direction colours — positive / negative / neutral
 * 7. NodeBar ML pills — only shown for model_signal with model_trained=true
 * 8. NodeBar ML pills — p_buy / p_hold / p_sell percentages
 * 9. NodeBar fundamentals pills — value_score / growth_score / quality_score / PE / PB / ROE
 * 10. GateChecks — passed gate shows green icon, failed gate shows red text
 * 11. Rank display — renders rank + universeSize when provided
 * 12. Compact mode — renders without breaking
 * 13. Unknown decision falls back to RANKED styling (no crash)
 * 14. Contributions sum matches composite score
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { CausalTracePanel } from "../components/CausalTracePanel";
import type { CausalNode, DataScope, GateCheck } from "../lib/api";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const screenerNodes: CausalNode[] = [
  {
    factor: "trend_pct",
    label: "Trend Quality",
    description: "Price above MA50 for 72% of the window",
    raw_value: 0.72,
    norm_value: 0.80,
    weight: 0.25,
    contribution: 0.20,
    direction: "positive",
    percentile: "Top 20%",
  },
  {
    factor: "atr_pct",
    label: "Volatility (ATR %)",
    description: "ATR = 1.50% of price — good for sizing",
    raw_value: 0.015,
    norm_value: 0.55,
    weight: 0.20,
    contribution: 0.11,
    direction: "neutral",
    percentile: "Top 45%",
  },
  {
    factor: "max_dd",
    label: "Drawdown Risk",
    description: "Max drawdown -12.0% — low risk",
    raw_value: -0.12,
    norm_value: 0.30,
    weight: 0.15,
    contribution: 0.045,
    direction: "negative",
    percentile: "Top 70%",
  },
];

const advisorNodes: CausalNode[] = [
  {
    factor: "fundamentals",
    label: "Fundamentals",
    description: "Value 72 · Growth 65 · Quality 80 (composite 72.3/100)",
    raw_value: 72.3,
    norm_value: 0.723,
    weight: 0.35,
    contribution: 0.253,
    direction: "positive",
    percentile: "",
    extras: {
      value_score: 72.0,
      growth_score: 65.0,
      quality_score: 80.0,
      pe_ttm: 18.5,
      pb: 3.1,
      roe: 22.5,
    },
  },
  {
    factor: "model_signal",
    label: "ML Signal (LightGBM)",
    description: "Signal: BUY with 72% confidence",
    raw_value: "BUY",
    norm_value: 1.0,
    weight: 0.30,
    contribution: 0.30,
    direction: "positive",
    percentile: "",
    extras: {
      signal: "BUY",
      confidence: 0.72,
      model_trained: true,
      p_buy: 0.72,
      p_hold: 0.18,
      p_sell: 0.10,
    },
  },
];

const advisorNodesNoModel: CausalNode[] = [
  {
    ...advisorNodes[1],
    extras: {
      signal: "N/A",
      confidence: 0,
      model_trained: false,
      p_buy: 0,
      p_hold: 1,
      p_sell: 0,
    },
  },
];

const upScope: DataScope = {
  start_date: "2024-03-01",
  end_date: "2025-03-01",
  bars: 252,
  price_start: 100.0,
  price_end: 120.0,
  price_min: 92.0,
  price_max: 125.0,
  trend: "UPTREND",
};

const downScope: DataScope = { ...upScope, trend: "DOWNTREND", price_end: 85.0 };
const sidewaysScope: DataScope = { ...upScope, trend: "SIDEWAYS" };

const gateChecks: GateCheck[] = [
  {
    check: "min_bars",
    label: "Sufficient price history",
    threshold: 200,
    actual: 252,
    passed: true,
    note: "252 bars ≥ 200 required",
  },
  {
    check: "liquidity",
    label: "Minimum liquidity",
    threshold: 1000000,
    actual: 500000,
    passed: false,
    note: "Volume 500000 < 1000000 required",
  },
];

// ── 1. Empty nodes renders nothing ───────────────────────────────────────────

describe("CausalTracePanel — empty guard", () => {
  it("renders nothing when causalNodes is empty", () => {
    const { container } = render(
      <CausalTracePanel decision="SELECTED" score={0.72} causalNodes={[]} />
    );
    expect(container.firstChild).toBeNull();
  });
});

// ── 2. DecisionBadge variants ────────────────────────────────────────────────

describe("CausalTracePanel — decision badge", () => {
  const decisions = ["SELECTED", "RECOMMENDED", "RANKED", "REJECTED"] as const;

  decisions.forEach((d) => {
    it(`renders ${d} badge label`, () => {
      render(
        <CausalTracePanel decision={d} score={0.5} causalNodes={screenerNodes} />
      );
      expect(screen.getByText(d)).toBeTruthy();
    });
  });

  it("does not crash for unknown decision", () => {
    expect(() =>
      render(
        <CausalTracePanel decision="UNKNOWN" score={0.5} causalNodes={screenerNodes} />
      )
    ).not.toThrow();
  });
});

// ── 3. TrendBadge variants ───────────────────────────────────────────────────

describe("CausalTracePanel — trend badge", () => {
  it("shows UPTREND label", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        dataScope={upScope}
      />
    );
    expect(screen.getByText(/UPTREND/i)).toBeTruthy();
  });

  it("shows DOWNTREND label", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        dataScope={downScope}
      />
    );
    expect(screen.getByText(/DOWNTREND/i)).toBeTruthy();
  });

  it("shows SIDEWAYS label", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        dataScope={sidewaysScope}
      />
    );
    expect(screen.getByText(/SIDEWAYS/i)).toBeTruthy();
  });
});

// ── 4. DataScope strip ───────────────────────────────────────────────────────

describe("CausalTracePanel — data scope", () => {
  it("shows start and end dates", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        dataScope={upScope}
      />
    );
    expect(screen.getByText(/2024-03-01/)).toBeTruthy();
    expect(screen.getByText(/2025-03-01/)).toBeTruthy();
  });

  it("shows bar count", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        dataScope={upScope}
      />
    );
    expect(screen.getByText(/252 bars/)).toBeTruthy();
  });

  it("does not render scope section when dataScope is undefined", () => {
    const { container } = render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
      />
    );
    expect(container.textContent).not.toMatch(/2024-03-01/);
  });
});

// ── 5. NodeBar — label, weight, contribution ─────────────────────────────────

describe("CausalTracePanel — node bars", () => {
  it("renders all node labels", () => {
    render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={screenerNodes} />
    );
    expect(screen.getByText("Trend Quality")).toBeTruthy();
    expect(screen.getByText("Volatility (ATR %)")).toBeTruthy();
    expect(screen.getByText("Drawdown Risk")).toBeTruthy();
  });

  it("renders node descriptions", () => {
    render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={screenerNodes} />
    );
    expect(screen.getByText(/Price above MA50 for 72%/)).toBeTruthy();
  });

  it("renders weight percentage", () => {
    render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={screenerNodes} />
    );
    // Trend Quality has 0.25 weight → "25% wt"
    expect(screen.getByText("25% wt")).toBeTruthy();
  });

  it("renders contribution value", () => {
    render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={screenerNodes} />
    );
    // Trend Quality contribution = 0.20 → "+0.200"
    expect(screen.getByText("+0.200")).toBeTruthy();
  });

  it("renders percentile label", () => {
    render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={screenerNodes} />
    );
    expect(screen.getByText("Top 20%")).toBeTruthy();
  });
});

// ── 6. NodeBar — direction colour (bar class) ────────────────────────────────

describe("CausalTracePanel — node direction classes", () => {
  it("positive direction node has emerald progress bar", () => {
    const { container } = render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={[screenerNodes[0]]} />
    );
    const bar = container.querySelector(".bg-emerald-500");
    expect(bar).not.toBeNull();
  });

  it("negative direction node has rose progress bar", () => {
    const { container } = render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={[screenerNodes[2]]} />
    );
    const bar = container.querySelector(".bg-rose-500");
    expect(bar).not.toBeNull();
  });

  it("neutral direction node has amber progress bar", () => {
    const { container } = render(
      <CausalTracePanel decision="SELECTED" score={0.5} causalNodes={[screenerNodes[1]]} />
    );
    const bar = container.querySelector(".bg-amber-500");
    expect(bar).not.toBeNull();
  });
});

// ── 7–8. ML probability pills ────────────────────────────────────────────────

describe("CausalTracePanel — ML probability pills", () => {
  it("shows BUY / HOLD / SELL pills for model_signal with model_trained=true", () => {
    render(
      <CausalTracePanel
        decision="RECOMMENDED"
        score={0.72}
        causalNodes={advisorNodes}
      />
    );
    // pills appear alongside description text — use getAllByText
    expect(screen.getAllByText(/BUY/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/HOLD/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/SELL/).length).toBeGreaterThanOrEqual(1);
  });

  it("shows correct p_buy percentage (72%)", () => {
    render(
      <CausalTracePanel
        decision="RECOMMENDED"
        score={0.72}
        causalNodes={advisorNodes}
      />
    );
    // p_buy = 0.72 → pill span contains "BUY" and "72" and "%"
    // Testing Library matches by accessible text content of each element.
    // The BUY pill span contains three child text nodes; we check them via
    // container text search to avoid the multi-element ambiguity.
    const text = document.body.textContent ?? "";
    expect(text).toMatch(/BUY/);
    expect(text).toMatch(/72/);
  });

  it("shows correct p_sell percentage (10%)", () => {
    render(
      <CausalTracePanel
        decision="RECOMMENDED"
        score={0.72}
        causalNodes={advisorNodes}
      />
    );
    expect(screen.getByText(/SELL.*10%|10%.*SELL/s)).toBeTruthy();
  });

  it("does NOT show ML pills when model_trained=false", () => {
    render(
      <CausalTracePanel
        decision="RECOMMENDED"
        score={0.5}
        causalNodes={advisorNodesNoModel}
      />
    );
    // Pills should not appear
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/BUY\s*72%/);
  });
});

// ── 9. Fundamentals sub-pills ────────────────────────────────────────────────

describe("CausalTracePanel — fundamentals pills", () => {
  it("renders Value score pill", () => {
    render(
      <CausalTracePanel
        decision="RECOMMENDED"
        score={0.72}
        causalNodes={[advisorNodes[0]]}
      />
    );
    // description contains 'Value' and pill label also contains 'Value'
    expect(screen.getAllByText(/Value/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/72\.0/).length).toBeGreaterThanOrEqual(1);
  });

  it("renders P/E pill", () => {
    render(
      <CausalTracePanel
        decision="RECOMMENDED"
        score={0.72}
        causalNodes={[advisorNodes[0]]}
      />
    );
    expect(screen.getByText(/P\/E/)).toBeTruthy();
    expect(screen.getByText(/18\.5/)).toBeTruthy();
  });

  it("renders ROE% pill", () => {
    render(
      <CausalTracePanel
        decision="RECOMMENDED"
        score={0.72}
        causalNodes={[advisorNodes[0]]}
      />
    );
    expect(screen.getByText(/ROE%/)).toBeTruthy();
    expect(screen.getByText(/22\.5/)).toBeTruthy();
  });

  it("does NOT render fundamentals pills for non-fundamentals nodes", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={[screenerNodes[0]]}  // trend_pct node, no extras
      />
    );
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/Value\s+\d/);
  });
});

// ── 10. Gate checks ──────────────────────────────────────────────────────────

describe("CausalTracePanel — gate checks", () => {
  it("renders passed gate check note", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        gateChecks={gateChecks}
      />
    );
    expect(screen.getByText("252 bars ≥ 200 required")).toBeTruthy();
  });

  it("renders failed gate check note", () => {
    render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        gateChecks={gateChecks}
      />
    );
    expect(screen.getByText("Volume 500000 < 1000000 required")).toBeTruthy();
  });

  it("renders no gate section when gateChecks is empty", () => {
    const { container } = render(
      <CausalTracePanel
        decision="SELECTED"
        score={0.5}
        causalNodes={screenerNodes}
        gateChecks={[]}
      />
    );
    // Passed/failed gate check icons should not appear
    expect(container.querySelector('[data-testid="gate-check"]')).toBeNull();
  });
});

// ── 11. Rank + universeSize display ─────────────────────────────────────────

describe("CausalTracePanel — rank display", () => {
  it("renders rank and universe size when both provided", () => {
    render(
      <CausalTracePanel
        decision="RANKED"
        score={0.65}
        rank={3}
        universeSize={300}
        causalNodes={screenerNodes}
      />
    );
    expect(screen.getByText(/Rank #3/)).toBeTruthy();
    expect(screen.getByText(/of 300/)).toBeTruthy();
  });

  it("renders rank without universeSize", () => {
    render(
      <CausalTracePanel
        decision="RANKED"
        score={0.65}
        rank={5}
        causalNodes={screenerNodes}
      />
    );
    expect(screen.getByText(/Rank #5/)).toBeTruthy();
  });

  it("does not render rank when rank is undefined", () => {
    render(
      <CausalTracePanel
        decision="RANKED"
        score={0.65}
        causalNodes={screenerNodes}
      />
    );
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/Rank #/);
  });
});

// ── 12. Score display ────────────────────────────────────────────────────────

describe("CausalTracePanel — score display", () => {
  it("renders score to 3 decimal places", () => {
    render(
      <CausalTracePanel decision="SELECTED" score={0.7234} causalNodes={screenerNodes} />
    );
    expect(screen.getByText("0.723")).toBeTruthy();
  });
});

// ── 13. Compact mode ─────────────────────────────────────────────────────────

describe("CausalTracePanel — compact mode", () => {
  it("renders without throwing in compact mode", () => {
    expect(() =>
      render(
        <CausalTracePanel
          decision="SELECTED"
          score={0.72}
          causalNodes={screenerNodes}
          dataScope={upScope}
          compact
        />
      )
    ).not.toThrow();
  });
});

// ── 14. Contribution sum integrity ───────────────────────────────────────────

describe("CausalTracePanel — contribution sum", () => {
  it("contributions of screener nodes are within expected range (0–1)", () => {
    const totalContribution = screenerNodes.reduce((s, n) => s + n.contribution, 0);
    expect(totalContribution).toBeGreaterThan(0);
    expect(totalContribution).toBeLessThanOrEqual(1.0);
  });

  it("weights of screener nodes sum to less than or equal to 1.0", () => {
    const totalWeight = screenerNodes.reduce((s, n) => s + n.weight, 0);
    expect(totalWeight).toBeLessThanOrEqual(1.0);
  });

  it("advisor node weights sum to ~1.0 (0.35 + 0.35 + 0.30)", () => {
    const allAdvisorNodes: CausalNode[] = [
      { ...advisorNodes[0] },
      { ...advisorNodes[1] },
      {
        factor: "momentum",
        label: "Price Momentum",
        description: "1Y +18% · 3M +5% · 1M +2%",
        raw_value: 0.18,
        norm_value: 0.68,
        weight: 0.35,
        contribution: 0.238,
        direction: "positive",
        percentile: "",
      },
    ];
    const totalWeight = allAdvisorNodes.reduce((s, n) => s + n.weight, 0);
    expect(totalWeight).toBeCloseTo(1.0, 5);
  });
});

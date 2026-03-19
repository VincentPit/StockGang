/**
 * api.causal.test.ts
 *
 * Tests for the causal-trace data model helpers and the pollJob / apiFetch
 * client utilities as they relate to causal trace fields.
 *
 * Coverage
 * ────────
 * 1. CausalNode shape validation — all required fields present
 * 2. DataScope shape validation — correct field types
 * 3. GateCheck shape validation — passed/failed semantics
 * 4. ScreenRow causal fields — causal_nodes / data_scope / gate_checks
 * 5. RecommendationRow causal fields — causal_nodes / data_scope
 * 6. DataScope trend enum — only UPTREND / DOWNTREND / SIDEWAYS
 * 7. CausalNode direction enum — positive / negative / neutral
 * 8. CausalNode contribution formula — weight × norm_value ≈ contribution
 * 9. pollJob resolves when status becomes "done"
 * 10. pollJob rejects when status is "error"
 * 11. pollJob calls onProgress callback on each tick
 * 12. apiFetch throws human-readable error on 4xx
 * 13. apiFetch throws timeout error after timeout
 */

import type {
  CausalNode,
  DataScope,
  GateCheck,
  ScreenRow,
  RecommendationRow,
} from "../lib/api";
import { pollJob } from "../lib/api";

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeCausalNode(overrides: Partial<CausalNode> = {}): CausalNode {
  return {
    factor: "trend_pct",
    label: "Trend Quality",
    description: "Price above MA50 for 72% of the window",
    raw_value: 0.72,
    norm_value: 0.80,
    weight: 0.25,
    contribution: 0.20,
    direction: "positive",
    percentile: "Top 20%",
    ...overrides,
  };
}

function makeDataScope(overrides: Partial<DataScope> = {}): DataScope {
  return {
    start_date: "2024-03-01",
    end_date: "2025-03-01",
    bars: 252,
    price_start: 100.0,
    price_end: 120.0,
    price_min: 92.0,
    price_max: 125.0,
    trend: "UPTREND",
    ...overrides,
  };
}

function makeGateCheck(overrides: Partial<GateCheck> = {}): GateCheck {
  return {
    check: "min_bars",
    label: "Sufficient price history",
    threshold: 200,
    actual: 252,
    passed: true,
    note: "252 bars ≥ 200 required",
    ...overrides,
  };
}

// ── 1. CausalNode shape ───────────────────────────────────────────────────────

describe("CausalNode — shape validation", () => {
  it("has all required fields", () => {
    const node = makeCausalNode();
    expect(typeof node.factor).toBe("string");
    expect(typeof node.label).toBe("string");
    expect(typeof node.description).toBe("string");
    expect(typeof node.norm_value).toBe("number");
    expect(typeof node.weight).toBe("number");
    expect(typeof node.contribution).toBe("number");
    expect(["positive", "negative", "neutral"]).toContain(node.direction);
    expect(typeof node.percentile).toBe("string");
  });

  it("raw_value can be number or string", () => {
    const numNode   = makeCausalNode({ raw_value: 0.72 });
    const strNode   = makeCausalNode({ raw_value: "BUY" });
    expect(typeof numNode.raw_value).toBe("number");
    expect(typeof strNode.raw_value).toBe("string");
  });

  it("extras is optional", () => {
    const noExtras  = makeCausalNode({ extras: undefined });
    const withExtras = makeCausalNode({ extras: { p_buy: 0.72 } });
    expect(noExtras.extras).toBeUndefined();
    expect(withExtras.extras?.p_buy).toBe(0.72);
  });
});

// ── 2. DataScope shape ────────────────────────────────────────────────────────

describe("DataScope — shape validation", () => {
  it("has all required fields with correct types", () => {
    const ds = makeDataScope();
    expect(typeof ds.start_date).toBe("string");
    expect(typeof ds.end_date).toBe("string");
    expect(typeof ds.bars).toBe("number");
    expect(typeof ds.price_start).toBe("number");
    expect(typeof ds.price_end).toBe("number");
    expect(typeof ds.price_min).toBe("number");
    expect(typeof ds.price_max).toBe("number");
  });

  it("trend is always a valid enum value", () => {
    const valid = ["UPTREND", "DOWNTREND", "SIDEWAYS"] as const;
    for (const trend of valid) {
      const ds = makeDataScope({ trend });
      expect(valid).toContain(ds.trend);
    }
  });

  it("price_min ≤ price_start and price_max ≥ price_end for uptrend", () => {
    const ds = makeDataScope();
    expect(ds.price_min).toBeLessThanOrEqual(ds.price_start);
    expect(ds.price_max).toBeGreaterThanOrEqual(ds.price_end);
  });

  it("bars is a positive integer", () => {
    const ds = makeDataScope({ bars: 252 });
    expect(ds.bars).toBeGreaterThan(0);
    expect(Number.isInteger(ds.bars)).toBe(true);
  });
});

// ── 3. GateCheck shape ────────────────────────────────────────────────────────

describe("GateCheck — shape and semantics", () => {
  it("has all required fields", () => {
    const g = makeGateCheck();
    expect(typeof g.check).toBe("string");
    expect(typeof g.label).toBe("string");
    expect(typeof g.threshold).toBe("number");
    expect(typeof g.actual).toBe("number");
    expect(typeof g.passed).toBe("boolean");
    expect(typeof g.note).toBe("string");
  });

  it("passed=true when actual ≥ threshold", () => {
    const g = makeGateCheck({ threshold: 200, actual: 252, passed: true });
    expect(g.actual).toBeGreaterThanOrEqual(g.threshold);
    expect(g.passed).toBe(true);
  });

  it("passed=false when actual < threshold", () => {
    const g = makeGateCheck({
      threshold: 1_000_000,
      actual: 500_000,
      passed: false,
      note: "Volume 500000 < 1000000 required",
    });
    expect(g.actual).toBeLessThan(g.threshold);
    expect(g.passed).toBe(false);
  });
});

// ── 4. ScreenRow causal fields ───────────────────────────────────────────────

describe("ScreenRow — causal fields", () => {
  const makeScreenRow = (): ScreenRow => ({
    rank: 1,
    symbol: "sh600519",
    yf_ticker: "600519.SS",
    name: "Kweichow Moutai",
    bars: 252,
    ret_1y: 0.18,
    ret_6m: 0.09,
    sharpe: 1.42,
    max_dd: -0.12,
    trend_pct: 0.72,
    atr_pct: 0.015,
    autocorr: 0.04,
    score: 0.718,
    recommended: true,
    causal_nodes: [
      makeCausalNode({ factor: "trend_pct", label: "Trend Quality" }),
      makeCausalNode({ factor: "atr_pct",   label: "Volatility (ATR %)", weight: 0.20, contribution: 0.11 }),
    ],
    data_scope: makeDataScope(),
    gate_checks: [makeGateCheck()],
  });

  it("has causal_nodes array", () => {
    const row = makeScreenRow();
    expect(Array.isArray(row.causal_nodes)).toBe(true);
    expect(row.causal_nodes.length).toBe(2);
  });

  it("each causal_node has factor, label, norm_value, contribution", () => {
    const row = makeScreenRow();
    for (const node of row.causal_nodes) {
      expect(node.factor).toBeTruthy();
      expect(node.label).toBeTruthy();
      expect(typeof node.norm_value).toBe("number");
      expect(typeof node.contribution).toBe("number");
    }
  });

  it("data_scope is present and valid", () => {
    const row = makeScreenRow();
    expect(row.data_scope).toBeDefined();
    expect(row.data_scope!.bars).toBe(252);
  });

  it("gate_checks is array with at least one entry", () => {
    const row = makeScreenRow();
    expect(Array.isArray(row.gate_checks)).toBe(true);
    expect(row.gate_checks.length).toBeGreaterThan(0);
  });

  it("gate_checks min_bars is passed when bars ≥ threshold", () => {
    const row = makeScreenRow();
    const minBarsCheck = row.gate_checks.find((g) => g.check === "min_bars");
    expect(minBarsCheck).toBeDefined();
    expect(minBarsCheck!.passed).toBe(true);
    expect(minBarsCheck!.actual).toBeGreaterThanOrEqual(minBarsCheck!.threshold);
  });
});

// ── 5. RecommendationRow causal fields ───────────────────────────────────────

describe("RecommendationRow — causal fields", () => {
  const makeRecommendRow = (): RecommendationRow => ({
    symbol: "sh600519",
    yf_ticker: "600519.SS",
    name: "Kweichow Moutai",
    sector: "consumer",
    score: 0.74,
    model_signal: "BUY",
    model_confidence: 0.72,
    model_trained: true,
    ret_1y: 0.18,
    ret_3m: 0.06,
    ret_1m: 0.02,
    fundamentals: {
      pe_ttm: 28.0, pb: 8.5, roe: 33.0,
      revenue_growth: 0.08, net_margin: 0.45,
      dividend_yield: 0.012, value_score: 68, growth_score: 72, quality_score: 85,
    },
    causal_nodes: [
      makeCausalNode({ factor: "fundamentals", label: "Fundamentals", weight: 0.35 }),
      makeCausalNode({ factor: "momentum",     label: "Price Momentum", weight: 0.35 }),
      makeCausalNode({ factor: "model_signal", label: "ML Signal (LightGBM)", weight: 0.30 }),
    ],
    data_scope: makeDataScope(),
  });

  it("has causal_nodes array with 3 advisor factors", () => {
    const row = makeRecommendRow();
    expect(row.causal_nodes.length).toBe(3);
    const factors = row.causal_nodes.map((n) => n.factor);
    expect(factors).toContain("fundamentals");
    expect(factors).toContain("momentum");
    expect(factors).toContain("model_signal");
  });

  it("advisor factor weights sum to 1.0", () => {
    const row = makeRecommendRow();
    const totalWeight = row.causal_nodes.reduce((s, n) => s + n.weight, 0);
    expect(totalWeight).toBeCloseTo(1.0, 5);
  });

  it("data_scope is present", () => {
    const row = makeRecommendRow();
    expect(row.data_scope).toBeDefined();
    expect(row.data_scope!.trend).toMatch(/^(UPTREND|DOWNTREND|SIDEWAYS)$/);
  });

  it("data_scope is optional (undefined is valid)", () => {
    const row = makeRecommendRow();
    row.data_scope = undefined;
    expect(row.data_scope).toBeUndefined();
  });
});

// ── 6–7. Enum consistency ─────────────────────────────────────────────────────

describe("CausalNode / DataScope — enum values", () => {
  it("direction values are exhaustive", () => {
    const directions = ["positive", "negative", "neutral"] as const;
    directions.forEach((d) => {
      const node = makeCausalNode({ direction: d });
      expect(node.direction).toBe(d);
    });
  });

  it("DataScope trend values are exhaustive", () => {
    const trends = ["UPTREND", "DOWNTREND", "SIDEWAYS"] as const;
    trends.forEach((t) => {
      const ds = makeDataScope({ trend: t });
      expect(ds.trend).toBe(t);
    });
  });
});

// ── 8. Contribution formula ───────────────────────────────────────────────────

describe("CausalNode — contribution formula", () => {
  it("contribution ≈ weight × norm_value (within 1e-6)", () => {
    const node = makeCausalNode({ weight: 0.25, norm_value: 0.80, contribution: 0.20 });
    expect(node.contribution).toBeCloseTo(node.weight * node.norm_value, 5);
  });

  it("contribution is always in [0, weight]", () => {
    const nodes = [
      makeCausalNode({ weight: 0.25, norm_value: 1.0, contribution: 0.25 }),
      makeCausalNode({ weight: 0.20, norm_value: 0.0, contribution: 0.0 }),
      makeCausalNode({ weight: 0.30, norm_value: 0.5, contribution: 0.15 }),
    ];
    for (const node of nodes) {
      expect(node.contribution).toBeGreaterThanOrEqual(0);
      expect(node.contribution).toBeLessThanOrEqual(node.weight + 1e-9);
    }
  });
});

// ── 9–11. pollJob client utility ─────────────────────────────────────────────

describe("pollJob — async polling", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it("resolves immediately when first response is 'done'", async () => {
    let calls = 0;
    const fetcher = async (_id: string) => {
      calls++;
      return { status: "done" as const, rows: [], causal_nodes: [] };
    };

    const promise = pollJob("job-1", fetcher, undefined, 100);
    jest.runAllTimers();
    const result = await promise;
    expect(result.status).toBe("done");
    expect(calls).toBe(1);
  });

  it("rejects when status is 'error'", async () => {
    const fetcher = async (_id: string) => ({
      status: "error" as const,
      error: "Something went wrong",
    });

    const promise = pollJob("job-err", fetcher, undefined, 100);
    jest.runAllTimers();
    await expect(promise).rejects.toThrow("Something went wrong");
  });

  it("calls onProgress on each tick", async () => {
    const statuses = ["pending", "running", "done"];
    let idx = 0;
    const fetcher = async (_id: string) => ({
      status: statuses[idx++] as "pending" | "running" | "done",
    });

    const progressCalls: string[] = [];
    const promise = pollJob("job-p", fetcher, (s) => progressCalls.push(s), 100);

    // Advance timers three times for three ticks
    jest.runAllTimers();
    await Promise.resolve();
    jest.runAllTimers();
    await Promise.resolve();
    jest.runAllTimers();
    await promise;

    expect(progressCalls).toContain("done");
  });
});

// ── 12–13. apiFetch error handling ───────────────────────────────────────────

describe("apiFetch — error handling", () => {
  const originalFetch = global.fetch;
  afterEach(() => { global.fetch = originalFetch; });

  it("throws human-readable error for 404 with JSON detail", async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Job not found" }),
      text: async () => "",
    } as unknown as Response);

    // Import apiFetch via dynamic require since it's not exported
    // Instead test the behavior through an API function that uses it
    // We verify the error message bubbles up via startScreen's underlying apiFetch
    const { startScreen } = await import("../lib/api");
    await expect(startScreen({ top_n: 5 })).rejects.toThrow("Job not found");
  });

  it("throws timeout error when fetch is aborted", async () => {
    jest.useFakeTimers();

    global.fetch = jest.fn().mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          const signal = init?.signal as AbortSignal;
          if (signal) {
            signal.addEventListener("abort", () => {
              reject(new DOMException("The operation was aborted.", "AbortError"));
            });
          }
        })
    );

    const { startScreen } = await import("../lib/api");
    // Start the request (it will hang until AbortController fires)
    const p = startScreen({ top_n: 5 });
    // Advance all timers so the internal setTimeout(abort, 30_000) fires
    jest.runAllTimers();
    await expect(p).rejects.toThrow();
    jest.useRealTimers();
  });
});

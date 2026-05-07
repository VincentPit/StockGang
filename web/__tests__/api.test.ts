/**
 * api.ts unit tests — fetch wrapper, error parsing, and high-traffic methods.
 *
 * We mock global.fetch (and crypto.getRandomValues for the traceparent
 * generator) so each call is deterministic. The goal is broad coverage of
 * the file's surface; we don't try to assert every endpoint, just enough
 * to lock in the contract and lift the file off 0%.
 */
import * as api from "@/lib/api";

// ── Mock harness ─────────────────────────────────────────────────────────────
const _origFetch = global.fetch;
const _origCrypto = global.crypto;

// jsdom doesn't ship a usable getRandomValues — install one before imports run
beforeAll(() => {
  if (!global.crypto || !global.crypto.getRandomValues) {
    Object.defineProperty(global, "crypto", {
      value: {
        getRandomValues: (buf: Uint8Array) => {
          for (let i = 0; i < buf.length; i++) buf[i] = i & 0xff;
          return buf;
        },
      },
      configurable: true,
    });
  }
});

afterAll(() => {
  global.fetch = _origFetch;
  if (_origCrypto) Object.defineProperty(global, "crypto", { value: _origCrypto, configurable: true });
});

function mockFetch(payload: unknown, init: { ok?: boolean; status?: number } = {}) {
  const ok = init.ok ?? true;
  const fn = jest.fn().mockResolvedValue({
    ok,
    status: init.status ?? (ok ? 200 : 500),
    json:   async () => payload,
    text:   async () => JSON.stringify(payload),
  });
  global.fetch = fn as unknown as typeof fetch;
  return fn;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("apiFetch (via startBacktest)", () => {
  it("sends JSON body and stamps a traceparent header", async () => {
    const fn = mockFetch({ job_id: "j1", status: "pending", symbols: ["sh600519"] });
    await api.startBacktest({ symbols: ["sh600519"] });
    expect(fn).toHaveBeenCalledTimes(1);
    const [url, init] = fn.mock.calls[0];
    expect(String(url)).toContain("/api/backtest");
    expect((init as RequestInit).method).toBe("POST");
    const headers = new Headers((init as RequestInit).headers);
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("traceparent")).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);
  });

  it("propagates the FastAPI `detail` field as the Error message", async () => {
    mockFetch({ detail: "something exploded" }, { ok: false, status: 422 });
    await expect(api.startBacktest({ symbols: ["x"] })).rejects.toThrow("something exploded");
  });

  it("propagates `message` when `detail` is missing", async () => {
    mockFetch({ message: "kaboom" }, { ok: false, status: 500 });
    await expect(api.startBacktest({ symbols: ["x"] })).rejects.toThrow("kaboom");
  });
});

describe("GET helpers", () => {
  it("getPrice URL-encodes the symbol and appends ?days", async () => {
    const fn = mockFetch({ symbol: "sh600519", yf_ticker: "600519.SS", bars: [] });
    await api.getPrice("sh600519", 90);
    const url = String(fn.mock.calls[0][0]);
    expect(url).toBe("/api/price/sh600519?days=90");
  });

  it("getStockNews respects custom limit", async () => {
    const fn = mockFetch({ items: [] });
    await api.getStockNews("hk00700", 5);
    expect(String(fn.mock.calls[0][0])).toBe("/api/news/hk00700?limit=5");
  });

  it("getRegime joins symbols with a comma and URL-encodes", async () => {
    const fn = mockFetch({ regime: "NEUTRAL", signal_multiplier: 1.0, symbols_analyzed: 0 });
    await api.getRegime(["sh600519", "sz300750"]);
    expect(String(fn.mock.calls[0][0])).toBe("/api/regime?symbols=sh600519%2Csz300750");
  });

  it("listJobs hits /api/jobs", async () => {
    const fn = mockFetch([{ id: "1", kind: "backtest", status: "done" }]);
    const jobs = await api.listJobs();
    expect(String(fn.mock.calls[0][0])).toBe("/api/jobs");
    expect(jobs).toHaveLength(1);
  });

  it("getAccount hits /api/account", async () => {
    const fn = mockFetch({ cash: 100, total_value: 100, positions: [], is_simulated: true, broker_mode: "simulator", initial_cash: 100 });
    await api.getAccount();
    expect(String(fn.mock.calls[0][0])).toBe("/api/account");
  });

  it("getSchedulerStatus hits /api/scheduler/status", async () => {
    const fn = mockFetch({});
    await api.getSchedulerStatus();
    expect(String(fn.mock.calls[0][0])).toBe("/api/scheduler/status");
  });
});

describe("recommend variants", () => {
  it("startRecommend without sector hits the bare endpoint", async () => {
    const fn = mockFetch({ job_id: "j", status: "pending", rows: [] });
    await api.startRecommend(undefined, 5);
    expect(String(fn.mock.calls[0][0])).toBe("/api/advisor/recommend?top_n=5");
  });

  it("startRecommend with sector encodes it into the path", async () => {
    const fn = mockFetch({ job_id: "j", status: "pending", rows: [] });
    await api.startRecommend("tech", 7);
    expect(String(fn.mock.calls[0][0])).toBe("/api/advisor/recommend/tech?top_n=7");
  });
});

describe("pollJob", () => {
  it("resolves on `done`", async () => {
    const fetcher = jest.fn()
      .mockResolvedValueOnce({ status: "running" })
      .mockResolvedValueOnce({ status: "done", payload: 42 });
    const updates: string[] = [];
    const out = await api.pollJob("j1", fetcher, (s) => updates.push(s), 1);
    expect(out.status).toBe("done");
    expect(updates).toEqual(["running", "done"]);
  });

  it("rejects on `error` with the embedded message", async () => {
    const fetcher = jest.fn().mockResolvedValueOnce({ status: "error", error: "boom" });
    await expect(api.pollJob("j2", fetcher, undefined, 1)).rejects.toThrow("boom");
  });
});

// ── Coverage round-up: walk through the remaining endpoint helpers ───────────
describe("endpoint round-up", () => {
  beforeEach(() => mockFetch({}));

  it.each([
    ["getBacktest",     () => api.getBacktest("J"),               "/api/backtest/J"],
    ["getScreen",       () => api.getScreen("J"),                 "/api/screen/J"],
    ["getWorkflow",     () => api.getWorkflow("J"),               "/api/workflow/J"],
    ["getFundamentals", () => api.getFundamentals("hk00700"),     "/api/fundamentals/hk00700"],
    ["getMacroNews",    () => api.getMacroNews(50),               "/api/news/macro?limit=50"],
    ["getUniverse",     () => api.getUniverse(),                  "/api/universe"],
    ["getTrain",        () => api.getTrain("T"),                  "/api/advisor/train/T"],
    ["getAnalyze",      () => api.getAnalyze("A"),                "/api/advisor/analyze/A"],
    ["pollRecommend",   () => api.pollRecommend("R"),             "/api/advisor/recommend-poll/R"],
    ["listModels",      () => api.listModels(),                   "/api/advisor/models"],
    ["deleteModel",     () => api.deleteModel("sh600519"),        "/api/advisor/models/sh600519"],
    ["getSectors",      () => api.getSectors(),                   "/api/advisor/sectors"],
    ["getTrainLoop",    () => api.getTrainLoop("L"),              "/api/train-loop/L"],
    ["getAutoTune",     () => api.getAutoTune("A"),               "/api/auto-tune/A"],
    ["getWalkForward",  () => api.getWalkForward("W"),            "/api/walk-forward/W"],
    ["getMonteCarlo",   () => api.getMonteCarlo("M"),             "/api/monte-carlo/M"],
    ["resumeScheduler", () => api.resumeScheduler(),              "/api/scheduler/resume"],
    ["pauseScheduler",  () => api.pauseScheduler(),               "/api/scheduler/pause"],
    ["getSchedulerHistory", () => api.getSchedulerHistory(5),     "/api/scheduler/history?limit=5"],
  ])("%s hits %s", async (_name, call, expectedUrl) => {
    const fn = mockFetch({});
    await call();
    expect(String(fn.mock.calls[0][0])).toBe(expectedUrl);
  });
});

describe("POST helpers", () => {
  it.each([
    ["startScreen",       () => api.startScreen({ top_n: 5 }),                            "/api/screen"],
    ["startWorkflow",     () => api.startWorkflow({ top_n: 5 }),                          "/api/workflow"],
    ["startTrain",        () => api.startTrain("hk00700", true),                          "/api/advisor/train"],
    ["startAnalyze",      () => api.startAnalyze("hk00700"),                              "/api/advisor/analyze"],
    ["startTrainLoop",    () => api.startTrainLoop({ top_n: 3 }),                         "/api/train-loop"],
    ["startAutoTune",     () => api.startAutoTune({ top_n: 3 }),                          "/api/auto-tune"],
    ["startWalkForward",  () => api.startWalkForward({ symbols: ["x"] }),                 "/api/walk-forward"],
    ["startMonteCarlo",   () => api.startMonteCarlo({ symbols: ["x"] }),                  "/api/monte-carlo"],
    ["submitOrder",       () => api.submitOrder({ symbol: "x", side: "BUY", order_type: "MARKET", quantity: 100 }), "/api/orders"],
    ["triggerScheduler",  () => api.triggerScheduler("data"),                             "/api/scheduler/trigger"],
  ])("%s POSTs to %s", async (_name, call, expectedUrl) => {
    const fn = mockFetch({});
    await call();
    expect(String(fn.mock.calls[0][0])).toBe(expectedUrl);
    expect((fn.mock.calls[0][1] as RequestInit).method).toBe("POST");
  });

  it("resetSimulator uses DELETE", async () => {
    const fn = mockFetch({ reset: true, initial_cash: 100 });
    await api.resetSimulator();
    expect(String(fn.mock.calls[0][0])).toBe("/api/account/reset");
    expect((fn.mock.calls[0][1] as RequestInit).method).toBe("DELETE");
  });
});

#!/usr/bin/env python3
"""
train_loop.py — Automated screen → backtest → tune feedback loop.

Logic
-----
1.  Run live screener → rank A-shares → pick top-N stocks
2.  For each top stock, try a grid of parameter configurations
3.  Track which config produces the best backtest (profit_factor, win_rate, PnL)
4.  Save winning params to best_params.json  (api/runner.py auto-loads on every run)
5.  If NO config is profitable across all top stocks:
      - Nudge screener weights toward safety (less momentum, more drawdown penalty)
      - Save to screener_weights.json  (screener auto-loads on next run)
      - Signal to re-run

Usage
-----
  python train_loop.py                              # screen + tune (default)
  python train_loop.py --symbol sh600871            # skip screener, use this symbol
  python train_loop.py --symbol sh600871 sh600519   # multiple symbols
  python train_loop.py --top 5 --lookback 180       # wider search
  python train_loop.py --symbol sh600871 --configs all  # run full 13-config grid
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Silence the noisy training / data-fetching logs during the loop
logging.basicConfig(level=logging.WARNING)
for _noisy in ("myquant", "lightgbm", "yfinance", "urllib3", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger("train_loop")

# ── Pass / fail targets ───────────────────────────────────────────────────────
PASS = {
    "min_trades":    3,
    "profit_factor": 1.20,
    "win_rate":      0.40,
    "min_pnl":       0.0,
}

# ── Tunable parameters ────────────────────────────────────────────────────────
@dataclass
class Params:
    # LGBM signal quality
    min_confidence:    float = 0.60
    threshold:         float = 0.015
    min_hold_bars:     int   = 5
    forward_days:      int   = 5
    # Risk management
    stop_loss_pct:     float = -0.08
    trailing_stop_pct: float = 0.00
    take_profit_pct:   float = 0.00
    commission_rate:   float = 0.0003


# ── Parameter grid ────────────────────────────────────────────────────────────
# "fast" = default 8 configs  |  "all" = full 13 configs
_GRID_FAST: list[tuple[str, dict]] = [
    ("baseline",       {}),
    ("strict_conf",    {"min_confidence": 0.65}),
    ("high_thresh",    {"threshold": 0.020}),
    ("long_hold",      {"min_hold_bars": 10}),
    ("tight_stop",     {"stop_loss_pct": -0.06}),
    ("trailing_06",    {"trailing_stop_pct": 0.06}),
    ("take_profit_12", {"take_profit_pct": 0.12}),
    ("safe_combo",     {"min_confidence": 0.65, "threshold": 0.020, "trailing_stop_pct": 0.06}),
]

_GRID_FULL: list[tuple[str, dict]] = _GRID_FAST + [
    ("very_strict",    {"min_confidence": 0.70}),
    ("very_high_thresh",{"threshold": 0.025}),
    ("long_hold_15",   {"min_hold_bars": 15}),
    ("trailing_10",    {"trailing_stop_pct": 0.10}),
    ("ultra_combo",    {"min_confidence": 0.70, "threshold": 0.020, "trailing_stop_pct": 0.08,
                        "take_profit_pct": 0.15}),
]


def _mk(overrides: dict) -> Params:
    return replace(Params(), **overrides)


# ── Evaluation helpers ────────────────────────────────────────────────────────
def _passes(r: dict) -> bool:
    return (
        r.get("num_trades",    0) >= PASS["min_trades"]
        and r.get("profit_factor", 0) >= PASS["profit_factor"]
        and r.get("win_rate",      0) >= PASS["win_rate"]
        and r.get("total_pnl",     0) >  PASS["min_pnl"]
    )


def _score(r: dict) -> float:
    """Higher is better. -999 if too few trades to evaluate."""
    if r.get("num_trades", 0) < PASS["min_trades"]:
        return -999.0
    pf  = r.get("profit_factor", 0.0)
    wr  = r.get("win_rate",      0.0)
    pnl = r.get("total_pnl",     0.0)
    # Primary: profit_factor  Secondary: win_rate  Tertiary: positive PnL
    return pf * 10.0 + wr * 5.0 + (1.0 if pnl > 0 else -0.5)


# ── One backtest trial ────────────────────────────────────────────────────────
def run_backtest(symbol: str, p: Params, lookback_days: int, train_years: int) -> dict:
    """
    Run a full single-symbol backtest with the given Params.
    Returns a flat dict of metrics.
    """
    from myquant.backtest.simulator import Backtester, BacktestConfig
    from myquant.models.bar import BarInterval
    from myquant.strategy.ml.lgbm_strategy import LGBMStrategy
    from myquant.strategy.technical.ma_crossover import MACrossoverStrategy
    from myquant.strategy.technical.macd_strategy import MACDStrategy
    from myquant.strategy.technical.rsi_strategy import RSIStrategy

    end_date   = datetime.now()
    test_start = end_date - timedelta(days=lookback_days)

    config = BacktestConfig(
        symbols            = [symbol],
        start_date         = test_start,
        end_date           = end_date,
        initial_cash       = 1_000_000.0,
        interval           = BarInterval.D1,
        commission_rate    = p.commission_rate,
        slippage           = 0.0002,
        apply_stamp_duty   = True,
        train_years        = train_years,
        stop_loss_pct      = p.stop_loss_pct,
        trailing_stop_pct  = p.trailing_stop_pct,
        take_profit_pct    = p.take_profit_pct,
        symbol_loss_cap    = -20_000.0,
    )

    bt = (
        Backtester(config)
        .add_strategy(LGBMStrategy(
            strategy_id    = "lgbm_core",
            symbols        = [symbol],
            forward_days   = p.forward_days,
            threshold      = p.threshold,
            train_ratio    = 0.70,
            min_confidence = p.min_confidence,
            retrain_every  = 21,
            max_train_bars = 504,
            use_macro      = False,
            num_leaves     = 31,
            n_estimators   = 300,
            min_hold_bars  = p.min_hold_bars,
            commission_rate= p.commission_rate,
        ))
        .add_strategy(MACrossoverStrategy("ma_cross",   [symbol], fast_period=10, slow_period=30, use_ema=True))
        .add_strategy(MACDStrategy(        "macd_sig",   [symbol], fast=12, slow=26, signal=9, min_hist=0.0))
        .add_strategy(MACrossoverStrategy( "rsi_proxy",  [symbol], fast_period=5,  slow_period=20, use_ema=True))
    )

    result = asyncio.run(bt.run())

    return {
        "total_pnl":     round(result.total_pnl, 2),
        "total_pnl_pct": round(result.total_pnl_pct, 6),
        "sharpe_ratio":  round(result.sharpe_ratio, 4),
        "max_drawdown":  round(result.max_drawdown, 4),
        "num_trades":    result.num_trades,
        "win_rate":      round(result.win_rate, 4),
        "avg_win":       round(result.avg_win, 2),
        "avg_loss":      round(result.avg_loss, 2),
        "profit_factor": round(result.profit_factor, 4),
    }


# ── Persistence ───────────────────────────────────────────────────────────────
def save_best_params(p: Params, symbol: str, result: dict) -> Path:
    """Write best_params.json. api/runner.py auto-loads it on the next backtest."""
    data = {
        **asdict(p),
        "_source_symbol": symbol,
        "_saved_at":      datetime.now().isoformat(),
        "_result":        result,
    }
    path = ROOT / "best_params.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def load_best_params() -> Optional[Params]:
    """Load best_params.json if it exists."""
    path = ROOT / "best_params.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        fields = {f for f in asdict(Params())}
        return _mk({k: v for k, v in d.items() if k in fields})
    except Exception:
        return None


def adjust_screener_weights(delta: dict[str, float]) -> Path:
    """
    Nudge screener scoring weights by delta amounts and save to screener_weights.json.
    stock_screener.py auto-loads this file at the start of each screen() call.
    Weights are re-normalised to sum to 1.0 after adjustment.
    """
    sw_path = ROOT / "screener_weights.json"
    if sw_path.exists():
        w: dict[str, float] = json.loads(sw_path.read_text())
    else:
        # Mirrors the defaults in stock_screener.py
        w = {
            "W_TREND":    0.20, "W_ATR":      0.10, "W_AUTOCORR": 0.10,
            "W_MOM6M":    0.05, "W_DD":       0.20, "W_LOW_VOL":  0.15,
            "W_DIST_52W": 0.10, "W_YANG":     0.10,
        }
    for k, dv in delta.items():
        w[k] = round(max(0.02, min(0.45, w.get(k, 0.10) + dv)), 3)
    total = sum(w.values())
    w = {k: round(v / total, 3) for k, v in w.items()}
    sw_path.write_text(json.dumps(w, indent=2))
    return sw_path


# ── Diagnostics ───────────────────────────────────────────────────────────────
def _diagnose(trials: list[dict]) -> str:
    """
    Look at all trials for a symbol and explain WHY nothing is working.
    Returns a human-readable summary.
    """
    if not trials:
        return "no trials ran"

    passing = [t for t in trials if t.get("passes")]
    if passing:
        return f"{len(passing)}/{len(trials)} configs pass"

    results = [t["result"] for t in trials if t.get("result")]
    if not results:
        return "all trials errored"

    avg_pf  = sum(r.get("profit_factor", 0) for r in results) / len(results)
    avg_wr  = sum(r.get("win_rate",      0) for r in results) / len(results)
    avg_t   = sum(r.get("num_trades",    0) for r in results) / len(results)
    avg_pnl = sum(r.get("total_pnl",     0) for r in results) / len(results)

    issues = []
    if avg_t < PASS["min_trades"]:
        issues.append(f"too few trades (avg {avg_t:.1f} < {PASS['min_trades']})")
    if avg_pf < PASS["profit_factor"]:
        issues.append(f"profit_factor too low (avg {avg_pf:.2f} < {PASS['profit_factor']})")
    if avg_wr < PASS["win_rate"]:
        issues.append(f"win_rate too low (avg {avg_wr:.0%} < {PASS['win_rate']:.0%})")
    if avg_pnl < 0:
        issues.append(f"avg PnL negative ({avg_pnl:+,.0f})")
    return "; ".join(issues) if issues else "unknown cause"


# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Automated screen → backtest → tune loop")
    parser.add_argument("--symbol",   nargs="*", default=None,
                        help="Skip screener; test these specific symbols")
    parser.add_argument("--top",      type=int,  default=3,
                        help="Top N stocks to pull from screener (default: 3)")
    parser.add_argument("--lookback", type=int,  default=180,
                        help="Backtest test window in calendar days (default: 180)")
    parser.add_argument("--train",    type=int,  default=1,
                        help="ML warm-up years before test window (default: 1)")
    parser.add_argument("--configs",  choices=["fast", "all"], default="fast",
                        help="Parameter grid size: fast=8, all=13 (default: fast)")
    args = parser.parse_args()

    grid = _GRID_FAST if args.configs == "fast" else _GRID_FULL

    _w = "=" * 66
    print(f"\n{_w}")
    print("  TRAIN LOOP  —  screen → backtest → tune → apply")
    print(_w)
    print(f"  Pass criteria : PF ≥ {PASS['profit_factor']:.2f}  |  "
          f"WR ≥ {PASS['win_rate']:.0%}  |  PnL > 0  |  trades ≥ {PASS['min_trades']}")
    print(f"  Grid size     : {len(grid)} configs  ({args.configs})")
    print(f"  Test window   : {args.lookback}d  |  Warm-up : {args.train}yr")

    # ── Step 1: get symbols ───────────────────────────────────────────────────
    screener_map: dict[str, dict] = {}   # symbol → screener result row
    if args.symbol:
        test_symbols = args.symbol
        print(f"\n[1/3] Symbols (CLI): {test_symbols}")
    else:
        print("\n[1/3] Running live screener…")
        from myquant.tools.stock_screener import screen
        top_syms, all_results, universe_size = screen(
            top_n=args.top,
            min_bars=200,
            lookback_years=1,
            verbose=True,
        )
        if not top_syms:
            print("  ERROR: screener returned 0 qualifying stocks. Exiting.")
            sys.exit(1)
        test_symbols = top_syms[: args.top]
        screener_map = {r["sym"]: r for r in all_results}
        print(f"\n  Top {len(test_symbols)} stocks: {test_symbols}")

    # ── Step 2: parameter grid search ────────────────────────────────────────
    n_total = len(grid) * len(test_symbols)
    print(f"\n[2/3] Running {n_total} trials "
          f"({len(grid)} configs × {len(test_symbols)} symbols)…\n")

    all_trials: list[dict]  = []
    best_score: float       = -float("inf")
    best_trial: Optional[dict] = None
    found_passing           = False
    sym_diagnoses: list[str] = []

    for sym_idx, symbol in enumerate(test_symbols):
        sr = screener_map.get(symbol, {})
        label = (
            f"{symbol}  ({sr.get('name', '')}  score={sr.get('score', 0):.3f})"
            if sr else symbol
        )
        print(f"  ── Stock #{sym_idx + 1}: {label}")

        sym_trials: list[dict] = []

        for cfg_name, overrides in grid:
            p = _mk(overrides)
            t0 = time.time()
            sys.stdout.write(f"    [{cfg_name:<20}] … ")
            sys.stdout.flush()

            try:
                res     = run_backtest(symbol, p, args.lookback, args.train)
                elapsed = time.time() - t0
                sc      = _score(res)
                passes  = _passes(res)
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"ERROR ({exc})")
                trial = {
                    "symbol": symbol, "config": cfg_name,
                    "params": asdict(p), "result": {}, "score": -999.0,
                    "passes": False, "elapsed_s": round(elapsed, 1),
                    "error": str(exc),
                }
                all_trials.append(trial)
                sym_trials.append(trial)
                continue

            status = "✅ PASS" if passes else ("⚡ ok  " if sc > 5 else "❌    ")
            print(
                f"{status}  "
                f"PF={res['profit_factor']:.3f}  "
                f"WR={res['win_rate']:.0%}  "
                f"PnL={res['total_pnl']:>+10,.0f}  "
                f"T={res['num_trades']:>3}  "
                f"({elapsed:.0f}s)"
            )

            trial = {
                "symbol":    symbol,
                "config":    cfg_name,
                "params":    asdict(p),
                "result":    res,
                "score":     round(sc, 3),
                "passes":    passes,
                "elapsed_s": round(elapsed, 1),
            }
            all_trials.append(trial)
            sym_trials.append(trial)

            if sc > best_score:
                best_score = sc
                best_trial = trial

            if passes:
                found_passing = True

        diag = _diagnose(sym_trials)
        sym_diagnoses.append(f"{symbol}: {diag}")
        print()

        # Stop as soon as the first profitable config is found on the #1 stock.
        # For stocks #2+, stop as soon as any config passes.
        if found_passing:
            print(f"  ✅ Found passing config — skipping remaining symbols.\n")
            break

    # ── Step 3: apply & persist ───────────────────────────────────────────────
    print("[3/3] Applying results…\n")

    if best_trial is None:
        print("  ERROR: all trials crashed. Check for import errors.")
        sys.exit(1)

    best_p = _mk(best_trial["params"])
    res    = best_trial["result"]

    print(f"  Best trial  : {best_trial['symbol']}  /  {best_trial['config']}")
    print(f"  Score       : {best_trial['score']:.2f}  "
          f"{'✅ PASS' if best_trial['passes'] else '⚠️  not profitable (best available)'}")
    print(f"  Metrics     : PF={res.get('profit_factor', 0):.3f}  "
          f"WR={res.get('win_rate', 0):.0%}  "
          f"PnL={res.get('total_pnl', 0):+,.0f}  "
          f"trades={res.get('num_trades', 0)}")
    print(f"  Params      : {json.dumps(asdict(best_p), separators=(',', ':'))}")

    bp_path = save_best_params(best_p, best_trial["symbol"], res)
    print(f"\n  ✓ Saved → {bp_path.name}")
    print(f"    api/runner.py auto-loads this on every subsequent backtest.")

    # If no config was profitable, nudge screener weights and report
    if not found_passing:
        print("\n  ⚠️  No config was profitable across all tested stocks.")
        print("  Diagnosis:")
        for d in sym_diagnoses:
            print(f"    {d}")
        print("\n  Nudging screener weights: ↑ safety  ↓ momentum…")
        sw_path = adjust_screener_weights({
            "W_MOM6M":    -0.02,   # 6-month momentum → reduce (rewards hot stocks)
            "W_AUTOCORR": -0.02,   # autocorrelation  → reduce (same issue)
            "W_DD":       +0.02,   # max-drawdown penalty → increase
            "W_LOW_VOL":  +0.02,   # stability reward → increase
        })
        print(f"  ✓ Saved → {sw_path.name}")
        print(f"    Re-run train_loop.py to search with re-weighted screener.")

    # Save full trial log
    log_path = ROOT / "train_loop_results.json"
    log_path.write_text(json.dumps({
        "run_at":          datetime.now().isoformat(),
        "symbols_tested":  test_symbols,
        "found_passing":   found_passing,
        "best":            best_trial,
        "all_trials":      all_trials,
        "diagnoses":       sym_diagnoses,
    }, indent=2, default=str))
    print(f"  ✓ Full log  → {log_path.name}")

    print(f"\n{_w}")
    if found_passing:
        print(f"  ✅ Done — profitable config found and applied.")
        print(f"     Backtest {best_trial['symbol']} in the UI to verify.")
    else:
        print(f"  ⚠️  Done — best available (not profitable) config applied.")
        print(f"     Screener weights nudged. Re-run train_loop.py to retry.")
    print(f"{_w}\n")


if __name__ == "__main__":
    main()

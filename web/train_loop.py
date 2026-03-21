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
    "min_trades":    6,     # 6 closed round-trips minimum (reduced from 8 — 480d window feasible)
    "profit_factor": 1.30,  # was 1.20
    "win_rate":      0.33,  # trend-following: low WR is OK when PF > 1.3 (break-even at ~0.30)
    "min_pnl":       0.0,
    "min_sharpe":    0.15,  # was 0.0 — require positive risk-adjusted return
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
    # Longer horizon: 10-day label is less noisy on slow-moving quality stocks
    ("fwd10",          {"forward_days": 10, "threshold": 0.020, "min_hold_bars": 8}),
    # Lower bar: more trades generated — helps when model is right but cautious
    ("low_conf",       {"min_confidence": 0.55, "min_hold_bars": 8}),
]

_GRID_FULL: list[tuple[str, dict]] = _GRID_FAST + [
    ("very_strict",    {"min_confidence": 0.70}),
    ("very_high_thresh",{"threshold": 0.025}),
    ("long_hold_15",   {"min_hold_bars": 15}),
    ("trailing_10",    {"trailing_stop_pct": 0.10}),
    ("ultra_combo",    {"min_confidence": 0.70, "threshold": 0.020, "trailing_stop_pct": 0.08,
                        "take_profit_pct": 0.15}),
    ("fwd10_strict",   {"forward_days": 10, "threshold": 0.025, "min_confidence": 0.65,
                        "min_hold_bars": 10}),
    ("tp_trail_combo", {"take_profit_pct": 0.10, "trailing_stop_pct": 0.07,
                        "min_confidence": 0.60}),
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
        and r.get("sharpe_ratio",  0) >= PASS["min_sharpe"]
    )


def _score(r: dict) -> float:
    """Higher is better. -999 if too few trades to evaluate."""
    if r.get("num_trades", 0) < PASS["min_trades"]:
        return -999.0
    n_trades = r.get("num_trades", 0)
    pf     = r.get("profit_factor", 0.0)
    wr     = r.get("win_rate",      0.0)
    pnl    = r.get("total_pnl",     0.0)
    sharpe = r.get("sharpe_ratio",  0.0)
    # Cap extreme PF values when trade count is low — prevents "PF=30" with 5 wins skewing scores
    pf_capped = min(pf, 6.0) if n_trades < 12 else pf
    base = pf_capped * 8.0 + wr * 5.0 + sharpe * 3.0 + (1.0 if pnl > 0 else -0.5)
    # Passing trials always beat non-passing — 1000 pt bonus guarantees it
    if _passes(r):
        base += 1000.0
    return base


# ── One backtest trial ────────────────────────────────────────────────────────
def run_backtest(symbol: str, p: Params, lookback_days: int, train_years: int) -> dict:
    """
    Run a full single-symbol backtest with the given Params.
    Returns a flat dict of metrics.
    """
    from myquant.backtest.simulator import Backtester, BacktestConfig
    from myquant.models.bar import BarInterval
    from myquant.strategy.ml.lgbm_strategy import LGBMStrategy

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
            num_leaves     = 63,      # matches default — was erroneously 31
            n_estimators   = 500,     # more rounds = better calibrated proba
            min_hold_bars  = p.min_hold_bars,
            commission_rate= p.commission_rate,
        ))
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
        # Mirror ALL 10 defaults from stock_screener.py (must stay in sync)
        w = {
            "W_TREND":    0.20, "W_ATR":      0.10, "W_AUTOCORR": 0.10,
            "W_MOM6M":    0.05, "W_DD":       0.14, "W_LOW_VOL":  0.13,
            "W_DIST_52W": 0.05, "W_YANG":     0.05,
            "W_SHARPE":   0.10, "W_CALMAR":   0.08,
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


# ── Programmatic entry-point (used by the API job runner) ────────────────────
def run_loop(
    symbols: list[str] | None = None,
    top_n: int = 3,
    lookback_days: int = 180,
    train_years: int = 1,
    configs: str = "fast",
    progress_cb=None,          # callable({"pct": int, "step": str}) → None
) -> dict:
    """
    Run the full training loop programmatically with optional progress reporting.

    Returns a dict with keys:
        found_passing, best, all_trials, diagnoses, symbols_tested, error
    """
    grid   = _GRID_FAST if configs == "fast" else _GRID_FULL
    _emit  = progress_cb or (lambda d: None)

    _emit({"pct": 2, "step": "Initialising train loop…"})

    # ── Step 1: get symbols ───────────────────────────────────────────────────
    screener_map: dict[str, dict] = {}
    if symbols:
        test_symbols = list(symbols)
        _emit({"pct": 10, "step": f"Using {len(test_symbols)} symbol(s) from request"})
    else:
        _emit({"pct": 5, "step": "Screening live stocks…"})
        from myquant.tools.stock_screener import screen
        top_syms, all_results, _ = screen(
            top_n=top_n, min_bars=200, lookback_years=1, verbose=False,
        )
        if not top_syms:
            return {"error": "Screener returned 0 qualifying stocks — try widening the universe."}
        test_symbols  = top_syms[:top_n]
        screener_map  = {r["sym"]: r for r in all_results}
        _emit({"pct": 15, "step": f"Screener done — top {len(test_symbols)}: {', '.join(test_symbols)}"})

    # ── Step 2: parameter grid search ────────────────────────────────────────
    n_total       = len(grid) * len(test_symbols)
    all_trials: list[dict]    = []
    best_score: float         = -float("inf")
    best_trial: dict | None   = None
    found_passing             = False
    sym_diagnoses: list[str]  = []
    trial_idx                 = 0

    for symbol in test_symbols:
        sym_trials: list[dict] = []

        for cfg_name, overrides in grid:
            pct_now = 15 + int(75 * trial_idx / max(n_total, 1))
            _emit({"pct": pct_now, "step": f"{symbol} / {cfg_name}"})
            p   = _mk(overrides)
            t0  = time.time()
            try:
                res     = run_backtest(symbol, p, lookback_days, train_years)
                elapsed = time.time() - t0
                sc      = _score(res)
                passes  = _passes(res)
            except Exception as exc:
                elapsed = time.time() - t0
                trial = {
                    "symbol": symbol, "config": cfg_name,
                    "params": asdict(p), "result": {}, "score": -999.0,
                    "passes": False, "elapsed_s": round(elapsed, 1),
                    "error": str(exc),
                }
                all_trials.append(trial)
                sym_trials.append(trial)
                trial_idx += 1
                continue

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
            trial_idx += 1

        diag = _diagnose(sym_trials)
        sym_diagnoses.append(f"{symbol}: {diag}")

        if found_passing:
            _emit({"pct": 88, "step": f"✅ Found passing config — skipping remaining symbols"})
            break

    # ── Step 3: apply & persist ───────────────────────────────────────────────
    _emit({"pct": 92, "step": "Saving best params…"})

    if best_trial is None:
        return {"error": "All trials crashed. Check symbol data availability.", "all_trials": all_trials}

    best_p = _mk(best_trial["params"])
    res    = best_trial["result"]
    save_best_params(best_p, best_trial["symbol"], res)

    if not found_passing:
        adjust_screener_weights({
            "W_MOM6M": -0.02, "W_AUTOCORR": -0.02,
            "W_DD":    +0.02, "W_LOW_VOL":  +0.02,
        })

    # Full trial log
    log_path = ROOT / "train_loop_results.json"
    log_path.write_text(json.dumps({
        "run_at":         datetime.now().isoformat(),
        "symbols_tested": test_symbols,
        "found_passing":  found_passing,
        "best":           best_trial,
        "all_trials":     all_trials,
        "diagnoses":      sym_diagnoses,
    }, indent=2, default=str))

    _emit({"pct": 100, "step": "Done ✅" if found_passing else "Done ⚠️ (no profitable config)"})
    return {
        "found_passing":  found_passing,
        "best":           best_trial,
        "all_trials":     all_trials,
        "diagnoses":      sym_diagnoses,
        "symbols_tested": test_symbols,
    }


# ── Autonomous self-test / retry ladder ─────────────────────────────────────
# Each rung escalates: more backtest data, longer LGBM warm-up, more symbols.
# Screener weights are nudged (scaled up each round) toward safety on failure.
_RETRY_LADDER: list[dict] = [
    # Round 1: Quick baseline — fast grid, 180d test window, 1yr training
    {"lookback_days": 180, "train_years": 1, "top_n_mult": 1, "configs": "fast"},
    # Round 2: Full grid, 360d window (captures a full market cycle)
    {"lookback_days": 360, "train_years": 1, "top_n_mult": 1, "configs": "all"},
    # Round 3: 2yr training warm-up (more history → better LGBM generalisation)
    {"lookback_days": 360, "train_years": 2, "top_n_mult": 1, "configs": "all"},
    # Round 4: 2× screener candidates (wider stock search)
    {"lookback_days": 365, "train_years": 2, "top_n_mult": 2, "configs": "all"},
    # Round 5: Maximum data — 3yr training, 540d window, 3× stock candidates
    {"lookback_days": 540, "train_years": 3, "top_n_mult": 3, "configs": "all"},
]


def self_test_loop(
    symbols: list[str] | None = None,
    top_n: int = 3,
    max_rounds: int = 5,
    progress_cb=None,
) -> dict:
    """
    Autonomous screen → backtest → tune loop with escalating retry strategies.

    Wraps ``run_loop()`` and automatically escalates through ``_RETRY_LADDER``
    until a passing config is found or all rounds are exhausted.  On each
    failed round the screener weights are nudged toward safety (drawdown
    penalty up, momentum weight down) — no human re-run required.

    Parameters
    ----------
    symbols :
        If given, skip the screener and test these symbols on every round.
    top_n :
        Number of screener picks in round 1; later rounds multiply by
        ``_RETRY_LADDER[i]["top_n_mult"]`` so the search widens automatically.
    max_rounds :
        Cap at this many rounds (capped at ``len(_RETRY_LADDER)`` = 5).
    progress_cb :
        Optional ``callable({"pct": int, "step": str}) → None``.

    Returns the same dict shape as ``run_loop()`` with two extra keys:
        ``rounds_run``  — how many rounds were actually executed
        ``all_rounds``  — per-round result dicts (list of ``run_loop`` returns)
    """
    _emit  = progress_cb or (lambda _: None)
    rounds = _RETRY_LADDER[:max(1, min(max_rounds, len(_RETRY_LADDER)))]
    n      = len(rounds)

    all_rounds: list[dict]    = []
    best_overall: dict | None = None
    best_score_overall: float = -float("inf")

    for idx, rung in enumerate(rounds):
        round_num       = idx + 1
        effective_top_n = top_n * rung["top_n_mult"]
        pct_base        = int(idx / n * 88)

        _emit({
            "pct":  pct_base,
            "step": (
                f"🔁 Round {round_num}/{n} — "
                f"lookback={rung['lookback_days']}d  "
                f"train={rung['train_years']}yr  "
                f"grid={rung['configs']}  "
                f"top_n={effective_top_n}"
            ),
        })
        logger.info(
            "self_test_loop round %d/%d  lookback=%d  train=%d  configs=%s  top_n=%d",
            round_num, n,
            rung["lookback_days"], rung["train_years"],
            rung["configs"], effective_top_n,
        )

        # Map run_loop's 0–100 range into this round's slice of the 0–88 band
        _round_slice = max(1, int(88 / n))

        def _scoped_cb(
            d: dict,
            _base: int = pct_base,
            _slice: int = _round_slice,
            _rn: int = round_num,
        ) -> None:
            inner = d.get("pct", 0)
            mapped = _base + int(inner / 100 * _slice)
            _emit({"pct": min(mapped, _base + _slice - 1), "step": d.get("step", "")})

        result = run_loop(
            symbols      =symbols,
            top_n        =effective_top_n,
            lookback_days=rung["lookback_days"],
            train_years  =rung["train_years"],
            configs      =rung["configs"],
            progress_cb  =_scoped_cb,  # per-trial progress now visible in UI
        )

        result["_round"] = round_num
        result["_rung"]  = rung
        all_rounds.append(result)

        # Track best trial seen across ALL rounds
        best = result.get("best")
        if best:
            sc = _score(best.get("result", {}))
            if sc > best_score_overall:
                best_score_overall = sc
                best_overall       = best

        # ── Success path ──────────────────────────────────────────────────────
        if result.get("found_passing"):
            _emit({"pct": 100, "step": f"✅ Passed on round {round_num}/{n}"})
            logger.info("self_test_loop: passed on round %d", round_num)
            return {
                "found_passing":  True,
                "rounds_run":     round_num,
                "best":           best_overall,
                "all_rounds":     all_rounds,
                "diagnoses":      result.get("diagnoses", []),
                "symbols_tested": result.get("symbols_tested", []),
            }

        # ── Failed round: diagnose → nudge screener → escalate ────────────────
        diag = result.get("diagnoses", [])
        logger.warning("self_test_loop round %d failed: %s", round_num, diag)
        _emit({
            "pct":  pct_base + int(88 / n * 0.9),
            "step": (
                f"  ↳ round {round_num} failed "
                f"({'; '.join(str(d) for d in diag[:2]) if diag else 'no diagnosis'})"
                "  — escalating…"
            ),
        })

        # Nudge screener weights (scale ×1.0, ×1.5, ×2.0, ×2.5, ×3.0 per round)
        scale = 1.0 + idx * 0.5
        adjust_screener_weights({
            "W_MOM6M":    round(-0.02 * scale, 4),
            "W_AUTOCORR": round(-0.01 * scale, 4),
            "W_DD":       round(+0.02 * scale, 4),
            "W_LOW_VOL":  round(+0.01 * scale, 4),
        })

    # ── All rounds exhausted ──────────────────────────────────────────────────
    _emit({"pct": 100, "step": f"⚠️ All {n} rounds exhausted — best available config applied"})
    logger.warning("self_test_loop: all %d rounds exhausted", n)
    last = all_rounds[-1] if all_rounds else {}
    return {
        "found_passing":  False,
        "rounds_run":     n,
        "best":           best_overall,
        "all_rounds":     all_rounds,
        "diagnoses":      last.get("diagnoses", []),
        "symbols_tested": last.get("symbols_tested", []),
    }


# ── CLI entry-point ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Automated screen → backtest → tune loop")
    parser.add_argument("--symbol",  nargs="*", default=None,
                        help="Skip screener; test these specific symbols")
    parser.add_argument("--top",     type=int,  default=3,
                        help="Top-N stocks from screener (round 1; auto-expands in later rounds)")
    parser.add_argument("--rounds",  type=int,  default=5,
                        help="Max self-test rounds before giving up (default: 5)")
    args = parser.parse_args()

    _w = "=" * 66
    print(f"\n{_w}")
    print("  SELF-TEST TRAIN LOOP  —  screen → backtest → tune → auto-retry")
    print(_w)
    print(f"  Pass criteria : PF ≥ {PASS['profit_factor']:.2f}  |  "
          f"WR ≥ {PASS['win_rate']:.0%}  |  PnL > 0  |  trades ≥ {PASS['min_trades']}  |  "
          f"Sharpe ≥ {PASS['min_sharpe']:.1f}")
    print(f"  Max rounds    : {args.rounds}  "
          f"({'specific symbols' if args.symbol else f'top-{args.top} from screener, auto-expands'})")
    print()

    def _cb(d: dict) -> None:
        step = d.get("step", "")
        pct  = d.get("pct")
        if step:
            prefix = f"  [{pct:>3}%] " if pct is not None else "         "
            print(prefix + step)

    result = self_test_loop(
        symbols    =args.symbol,
        top_n      =args.top,
        max_rounds =args.rounds,
        progress_cb=_cb,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    best   = result.get("best") or {}
    res    = best.get("result") or {}
    found  = result.get("found_passing", False)
    n_done = result.get("rounds_run", 0)

    print(f"\n{_w}")
    if found:
        print(f"  ✅ PASSED  after {n_done} round(s)")
    else:
        print(f"  ⚠️  All {n_done} round(s) exhausted — best available config applied")

    if best:
        print(f"\n  Best trial  : {best.get('symbol')}  /  {best.get('config')}")
        print(f"  Score       : {best.get('score', 0):.2f}  "
              f"({'PASS' if best.get('passes') else 'not profitable'})")
        print(f"  Metrics     : "
              f"PF={res.get('profit_factor', 0):.3f}  "
              f"WR={res.get('win_rate', 0):.0%}  "
              f"PnL={res.get('total_pnl', 0):+,.0f}  "
              f"trades={res.get('num_trades', 0)}  "
              f"Sharpe={res.get('sharpe_ratio', res.get('sharpe', 0)):.2f}")
        best_p = _mk(best.get("params", {}))
        print(f"  Params      : {json.dumps(asdict(best_p), separators=(',', ':'))}")
    else:
        print("\n  ERROR: no trials completed. Check logs for data/import errors.")

    if result.get("diagnoses"):
        print("\n  Diagnosis:")
        for d in result.get("diagnoses", []):
            print(f"    {d}")

    # ── Save full self-test results log ───────────────────────────────────────
    log_path = ROOT / "self_test_results.json"
    log_path.write_text(json.dumps({
        "run_at":        datetime.now().isoformat(),
        "found_passing": found,
        "rounds_run":    n_done,
        "best":          best,
        "all_rounds":    result.get("all_rounds", []),
        "diagnoses":     result.get("diagnoses", []),
    }, indent=2, default=str))
    print(f"\n  ✓ Full log  → {log_path.name}")
    print(f"{_w}\n")


if __name__ == "__main__":
    main()

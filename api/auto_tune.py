"""
api/auto_tune.py — Autonomous train → analyse → diagnose → adjust → retrain loop.

Pipeline per iteration
──────────────────────
1. Run self_test_loop   (screen → backtest param grid → save best_params.json)
2. Analyse trained models via analyze_stock (OOS accuracy, signal, confidence)
3. Score combined quality: backtest metrics + model quality
4. If score ≥ SUCCESS_THRESHOLD → converged ✅
5. Else diagnose root cause → write model_overrides.json + nudge screener weights
6. Force-delete cached model blobs so next iteration trains fresh with new params
7. Repeat up to max_iterations

Quality thresholds
──────────────────
  OOS accuracy ≥ 0.54   (meaningfully above 50% random baseline)
  Confidence   ≥ 0.48   (model has directional conviction)
  Profit factor ≥ 1.20
  Sharpe        ≥ 0.10
  Trades        ≥ 5
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_log = logging.getLogger(__name__)

# ── Quality thresholds ─────────────────────────────────────────────────────────
_Q = {
    "oos_ok":      0.54,
    "oos_good":    0.60,
    "conf_ok":     0.48,
    "conf_good":   0.55,
    "pf_ok":       1.20,
    "sharpe_ok":   0.10,
    "trades_ok":   5,
    "success_score": 70,   # combined score out of 100 to declare convergence
}

# ── Paths ─────────────────────────────────────────────────────────────────────
_OVERRIDES_PATH = ROOT / "model_overrides.json"
_BEST_PARAMS    = ROOT / "best_params.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_overrides() -> dict:
    if _OVERRIDES_PATH.exists():
        try:
            return json.loads(_OVERRIDES_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_overrides(d: dict) -> None:
    _OVERRIDES_PATH.write_text(json.dumps(d, indent=2))


def _score_iteration(
    bt_best: dict,
    analyses: list[dict],
) -> tuple[float, dict, list[str]]:
    """
    Return (total_score_0_100, breakdown_dict, failure_messages).
    50 pts from backtest, 50 pts from model quality.
    """
    failures: list[str] = []
    bt_res = bt_best.get("result", {})

    # ── Backtest score (50 pts) ───────────────────────────────────────────────
    pf      = bt_res.get("profit_factor", 0.0)
    sharpe  = bt_res.get("sharpe_ratio",  0.0)
    wr      = bt_res.get("win_rate",      0.0)
    trades  = bt_res.get("num_trades",    0)
    pnl     = bt_res.get("total_pnl",     0.0)

    bt_score = 0.0
    if pf >= 1.40:      bt_score += 20
    elif pf >= 1.30:    bt_score += 16
    elif pf >= 1.20:    bt_score += 10
    elif pf >= 1.0:     bt_score += 4
    else:               failures.append(f"profit_factor {pf:.2f} (need ≥ 1.20)")

    if sharpe >= 0.30:  bt_score += 12
    elif sharpe >= 0.15: bt_score += 8
    elif sharpe >= 0.10: bt_score += 5
    else:               failures.append(f"sharpe {sharpe:.2f} (need ≥ 0.10)")

    if wr >= 0.42:      bt_score += 10
    elif wr >= 0.33:    bt_score += 7
    elif wr >= 0.25:    bt_score += 3
    else:               failures.append(f"win_rate {wr:.0%} (need ≥ 33%)")

    if trades >= 10:    bt_score += 5
    elif trades >= 5:   bt_score += 3
    else:               failures.append(f"only {trades} trades (need ≥ 5)")

    if pnl > 0:         bt_score += 3
    else:               failures.append(f"PnL negative ({pnl:+,.0f})")

    # ── Model quality score (50 pts, averaged across all symbols) ─────────────
    model_summaries: list[dict] = []
    model_scores: list[float]   = []

    for a in analyses:
        mm   = a.get("model_meta") or {}
        oos  = float(mm.get("oos_accuracy") or 0)
        conf = float(a.get("confidence") or 0)
        p_hold = float(a.get("p_hold") or 0)
        signal = a.get("signal", "HOLD")

        ms = 0.0
        if oos >= 0.65:     ms += 25
        elif oos >= 0.60:   ms += 20
        elif oos >= 0.54:   ms += 13
        elif oos >= 0.50:   ms += 6
        else:               failures.append(f"{a['symbol']} OOS {oos:.1%} (need ≥ 54%)")

        if conf >= 0.60:    ms += 15
        elif conf >= 0.55:  ms += 12
        elif conf >= 0.48:  ms += 8
        elif conf >= 0.40:  ms += 4
        else:               failures.append(f"{a['symbol']} confidence {conf:.1%} (need ≥ 48%)")

        directionality = max(float(a.get("p_buy") or 0), float(a.get("p_sell") or 0))
        if directionality >= 0.45: ms += 10
        elif directionality >= 0.38: ms += 5
        else:               failures.append(f"{a['symbol']} model always predicts HOLD")

        model_summaries.append({
            "symbol":       a["symbol"],
            "oos_accuracy": round(oos, 4),
            "confidence":   round(conf, 4),
            "signal":       signal,
            "p_buy":        a.get("p_buy"),
            "p_hold":       a.get("p_hold"),
            "p_sell":       a.get("p_sell"),
            "score":        round(ms, 1),
        })
        model_scores.append(ms)

    avg_model = (sum(model_scores) / len(model_scores)) if model_scores else 0.0

    total = round(bt_score + avg_model, 1)
    breakdown = {
        "total":         total,
        "backtest":      round(bt_score, 1),
        "model":         round(avg_model, 1),
        "backtest_ok":   bt_score >= 30,
        "model_ok":      avg_model >= 25,
        "pf":     pf,  "sharpe": sharpe, "win_rate": wr,
        "trades": trades, "pnl":  pnl,
        "model_summaries": model_summaries,
    }
    return total, breakdown, failures


def _diagnose_and_adjust(
    failures: list[str],
    analyses: list[dict],
    bt_best: dict,
    iteration: int,
) -> list[dict]:
    """
    Inspect failure causes and emit concrete parameter adjustments.
    Also writes model_overrides.json and nudges screener_weights.json.
    """
    from train_loop import adjust_screener_weights

    bt_res  = bt_best.get("result", {})
    pf      = bt_res.get("profit_factor", 0.0)
    wr      = bt_res.get("win_rate",      0.0)
    trades  = bt_res.get("num_trades",    0)
    scale   = 1.0 + iteration * 0.5      # escalate adjustment magnitude each round

    ov      = _load_overrides()
    actions: list[dict] = []

    def _act(param: str, old, new, reason: str, kind: str = "model") -> None:
        actions.append({"type": kind, "param": param, "old": old, "new": new, "reason": reason})
        ov[param] = new

    # ── Per-symbol model quality ───────────────────────────────────────────────
    for a in analyses:
        mm   = a.get("model_meta") or {}
        oos  = float(mm.get("oos_accuracy") or 0)
        conf = float(a.get("confidence") or 0)
        ph   = float(a.get("p_hold") or 0)

        if oos < 0.52:
            # Model is overfitting or labels are too noisy
            old_l  = ov.get("num_leaves", 63)
            new_l  = max(15, int(old_l * 0.75))
            _act("num_leaves", old_l, new_l,
                 f"OOS {oos:.1%} < 52% → simplify trees to prevent overfit")

            old_fd = ov.get("forward_days", 5)
            new_fd = max(3, old_fd - 1)
            if new_fd != old_fd:
                _act("forward_days", old_fd, new_fd,
                     "Shorter label horizon reduces label noise when OOS is low")

            old_reg = ov.get("reg_lambda", 0.10)
            new_reg = min(0.50, round(old_reg * 1.5 * scale, 3))
            _act("reg_lambda", old_reg, new_reg,
                 "Increase L2 regularisation to improve generalisation")

        elif oos < 0.55:
            old_est = ov.get("n_estimators", 500)
            new_est = min(800, int(old_est * 1.2))
            _act("n_estimators", old_est, new_est,
                 f"OOS {oos:.1%} borderline → more boosting rounds may help")

        if conf < 0.45 and ph > 0.52:
            # Model always says HOLD → allow finer leaf splits
            old_mc = ov.get("min_child_samples", 25)
            new_mc = max(8, int(old_mc * 0.75))
            _act("min_child_samples", old_mc, new_mc,
                 f"p_hold={ph:.0%} too high → allow finer splits so model commits to directions",
                 kind="model")

    # ── Backtest quality ───────────────────────────────────────────────────────
    if trades < _Q["trades_ok"]:
        old_thr = ov.get("threshold", 0.015)
        new_thr = max(0.008, round(old_thr * 0.85, 4))
        _act("threshold", old_thr, new_thr,
             f"only {trades} trades → lower entry bar to generate more signals",
             kind="signal")

        old_mc2 = ov.get("min_confidence", 0.60)
        new_mc2 = max(0.50, round(old_mc2 - 0.04, 2))
        _act("min_confidence", old_mc2, new_mc2,
             "Reduce min-confidence so more signals pass the filter",
             kind="signal")

    elif pf < _Q["pf_ok"]:
        old_thr = ov.get("threshold", 0.015)
        new_thr = min(0.030, round(old_thr * 1.15 * (1 + 0.1 * iteration), 4))
        _act("threshold", old_thr, new_thr,
             f"PF={pf:.2f} < 1.20 → raise entry bar, trade only high-quality moves",
             kind="signal")

        if wr < 0.33:
            old_mc2 = ov.get("min_confidence", 0.60)
            new_mc2 = min(0.75, round(old_mc2 + 0.05, 2))
            _act("min_confidence", old_mc2, new_mc2,
                 f"WR={wr:.0%} < 33% → only take highest-confidence signals",
                 kind="signal")

    # ── Screener weights ───────────────────────────────────────────────────────
    sw_delta = {
        "W_MOM6M":   round(-0.015 * scale, 4),
        "W_DD":      round(+0.015 * scale, 4),
        "W_LOW_VOL": round(+0.010 * scale, 4),
        "W_SHARPE":  round(+0.010 * scale, 4),
        "W_CALMAR":  round(+0.005 * scale, 4),
    }
    adjust_screener_weights(sw_delta)
    actions.append({
        "type":   "screener_weights",
        "param":  "W_DD / W_LOW_VOL / W_SHARPE",
        "old":    "previous",
        "new":    f"+{0.015*scale:.3f} drawdown/sharpe/vol, −{0.015*scale:.3f} momentum",
        "reason": "Favour lower-volatility, stronger-drawdown-profile stocks in next screen",
    })

    _save_overrides(ov)
    return actions


def _delete_cached_models(symbols: list[str]) -> None:
    """Force-delete stored LGBM model blobs so next analyse() call retrains fresh."""
    try:
        from api import db as _db
        for sym in symbols:
            _db.delete_model(sym)
            _log.info("auto_tune: deleted cached model for %s", sym)
    except Exception as e:
        _log.warning("auto_tune: could not delete model for %s: %s", symbols, e)


# ── Core loop ─────────────────────────────────────────────────────────────────

def auto_tune_loop(
    symbols: list[str] | None = None,
    top_n: int = 3,
    max_iterations: int = 3,
    progress_cb=None,
) -> dict:
    """
    Autonomous train → analyse → diagnose → adjust → retrain loop.

    Parameters
    ----------
    symbols       : specific symbols to test (skips screener if set)
    top_n         : screener top-N in round 1
    max_iterations: max feedback iterations (each ~5–15 min)
    progress_cb   : callable({pct, step}) for live UI updates

    Returns
    -------
    dict with keys: converged, iterations_run, final_score, best_symbol,
                    best_config, iterations (audit trail), error
    """
    from train_loop import self_test_loop
    from api.advisor import analyze_stock

    _emit = progress_cb or (lambda d: None)
    iterations_log: list[dict] = []
    converged = False
    final_score = 0.0
    best_symbol_overall = None
    best_config_overall = None

    _emit({"pct": 1, "step": "🚀 Auto-Tune starting — train → analyse → adjust loop"})

    for it in range(max_iterations):
        it_num = it + 1
        pct_base = int(it / max_iterations * 90)
        _slice   = int(90 / max_iterations)

        _emit({"pct": pct_base, "step": f"━━ Iteration {it_num}/{max_iterations} ━━"})

        iter_record: dict = {
            "iteration":   it_num,
            "started_at":  datetime.now().isoformat(),
            "phase":       "training",
            "adjustments": [],
            "failures":    [],
            "score":       0.0,
            "breakdown":   {},
            "analyses":    [],
            "backtest":    {},
        }

        # ── Phase 1: Train ─────────────────────────────────────────────────────
        _emit({"pct": pct_base + 2, "step": f"  [1/3] Training models (iter {it_num})…"})

        def _train_cb(d: dict, _b: int = pct_base, _s: int = _slice) -> None:
            inner = d.get("pct", 0)
            mapped = _b + 2 + int(inner / 100 * (_s * 0.60))
            _emit({"pct": min(mapped, _b + int(_s * 0.62)), "step": d.get("step", "")})

        train_result = self_test_loop(
            symbols    = symbols,
            top_n      = top_n,
            max_rounds = 1,          # one round per auto-tune iteration — we control the outer loop
            progress_cb= _train_cb,
        )

        bt_best = train_result.get("best") or {}
        tested_symbols = train_result.get("symbols_tested") or (symbols or [])

        if bt_best:
            best_symbol_overall = bt_best.get("symbol")
            best_config_overall = bt_best.get("config")

        bt_res = bt_best.get("result", {})
        iter_record["backtest"] = {
            "symbol":        bt_best.get("symbol"),
            "config":        bt_best.get("config"),
            "passes":        bt_best.get("passes", False),
            "score":         bt_best.get("score"),
            "profit_factor": bt_res.get("profit_factor"),
            "sharpe":        bt_res.get("sharpe_ratio"),
            "win_rate":      bt_res.get("win_rate"),
            "num_trades":    bt_res.get("num_trades"),
            "total_pnl":     bt_res.get("total_pnl"),
        }

        # ── Phase 2: Analyse trained models ────────────────────────────────────
        iter_record["phase"] = "analysing"
        analyses: list[dict] = []

        for i, sym in enumerate(tested_symbols[:3]):
            ap = pct_base + int(_slice * 0.65) + int(i / max(len(tested_symbols), 1) * _slice * 0.20)
            _emit({"pct": ap, "step": f"  [2/3] Analysing model quality for {sym}…"})
            try:
                result = analyze_stock(sym, force_retrain=False)
                analyses.append(result)
                mm = result.get("model_meta") or {}
                _emit({
                    "pct": ap,
                    "step": (
                        f"    {sym}: signal={result.get('signal')}  "
                        f"OOS={float(mm.get('oos_accuracy') or 0):.1%}  "
                        f"conf={float(result.get('confidence') or 0):.1%}"
                    ),
                })
            except Exception as exc:
                _log.warning("auto_tune: analyze_stock(%s) failed: %s", sym, exc)
                analyses.append({
                    "symbol": sym, "signal": "ERROR", "confidence": 0,
                    "p_buy": 0, "p_hold": 1, "p_sell": 0,
                    "model_meta": {"oos_accuracy": 0},
                    "_error": str(exc),
                })

        iter_record["analyses"] = [
            {
                "symbol":       a["symbol"],
                "signal":       a.get("signal"),
                "confidence":   a.get("confidence"),
                "p_buy":        a.get("p_buy"),
                "p_hold":       a.get("p_hold"),
                "p_sell":       a.get("p_sell"),
                "oos_accuracy": (a.get("model_meta") or {}).get("oos_accuracy"),
                "error":        a.get("_error"),
            }
            for a in analyses
        ]

        # ── Phase 3: Evaluate ──────────────────────────────────────────────────
        iter_record["phase"] = "evaluating"
        ep = pct_base + int(_slice * 0.87)
        _emit({"pct": ep, "step": f"  [3/3] Scoring iteration {it_num}…"})

        score, breakdown, failures = _score_iteration(bt_best, analyses)
        iter_record.update({
            "score":      score,
            "breakdown":  breakdown,
            "failures":   failures,
            "backtest_ok": breakdown.get("backtest_ok", False),
            "model_ok":    breakdown.get("model_ok", False),
        })

        _emit({
            "pct": ep + 1,
            "step": (
                f"  → Score {score:.0f}/100  "
                f"(backtest {breakdown['backtest']:.0f}  model {breakdown['model']:.0f})"
            ),
        })

        final_score = score

        # ── Success? ───────────────────────────────────────────────────────────
        if score >= _Q["success_score"] or bt_best.get("passes"):
            converged = True
            iter_record["phase"] = "converged"
            _emit({"pct": pct_base + _slice, "step": f"  ✅ Converged! Score {score:.0f}/100"})
            iterations_log.append(iter_record)
            break

        # ── Phase 4: Diagnose + Adjust (only if more iterations remain) ────────
        if it_num < max_iterations:
            iter_record["phase"] = "adjusting"
            adj_p = ep + 2
            _emit({"pct": adj_p, "step": f"  ⚙️  Diagnosing {len(failures)} failure(s) and adjusting…"})

            for f in failures:
                _emit({"pct": adj_p, "step": f"      ✗ {f}"})

            actions = _diagnose_and_adjust(failures, analyses, bt_best, it)

            iter_record["adjustments"] = actions
            for act in actions:
                _emit({
                    "pct": adj_p,
                    "step": f"      ↳ [{act['type']}] {act['param']}: {act['old']} → {act['new']}",
                })

            # Force-retrain next iteration (clear stale model blobs)
            _delete_cached_models(tested_symbols[:3])
            _emit({"pct": adj_p + 1, "step": "      ✓ Cached models cleared — will retrain with new params"})

        iterations_log.append(iter_record)

    _emit({"pct": 100, "step": "✅ Auto-Tune complete!" if converged else "⚠️ Max iterations reached — best params applied"})

    return {
        "converged":       converged,
        "iterations_run":  len(iterations_log),
        "final_score":     final_score,
        "best_symbol":     best_symbol_overall,
        "best_config":     best_config_overall,
        "iterations":      iterations_log,
    }

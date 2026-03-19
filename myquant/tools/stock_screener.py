"""
tools/stock_screener.py — Live A-share quantitative screener.

Fetches the investable universe from CSI index constituents (via akshare),
downloads OHLCV data in parallel (16 threads via yfinance), computes 5
quantitative signals, normalises them, and returns a ranked list.

Scoring weights are tuned to this system's known strengths:
  Trend (MA50 time above):   0.25  — MA50 gate needs trending stocks
  ATR adequacy:              0.20  — ATR sizer needs enough volatility
  Trend autocorrelation:     0.20  — ML features reward momentum
  6-month momentum:          0.20  — recent directional bias
  Max-drawdown penalty:      0.15  — punish blow-up candidates

Usage (CLI):
    python tools/stock_screener.py
    python tools/stock_screener.py --top 8 --indices 000300 000905
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from myquant.data.fetchers.universe_fetcher import fetch_universe   # noqa: E402


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _atr_pct(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    atr = float(np.mean(trs[-period:]))
    return atr / closes[-1] if closes[-1] > 0 else 0.0


def _ma50_pct_above(closes: np.ndarray) -> float:
    if len(closes) < 51:
        return 0.5
    total = len(closes) - 50
    above = sum(1 for i in range(50, len(closes)) if closes[i] > float(np.mean(closes[i-50:i])))
    return above / total


def _autocorr_lag1(returns: np.ndarray) -> float:
    if len(returns) < 10:
        return 0.0
    r = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 5:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns))
    return (mu / sd) * np.sqrt(252) if sd > 0 else 0.0


def _max_drawdown(closes: np.ndarray) -> float:
    peak = closes[0]
    max_dd = 0.0
    for c in closes:
        if c > peak:
            peak = c
        dd = (c - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


# ---------------------------------------------------------------------------
# Per-stock worker (runs in a thread-pool thread)
# ---------------------------------------------------------------------------

def _fetch_and_score(sym, yf_tick, name, start_iso, end_iso, min_bars, n_score):
    try:
        import yfinance as yf
        df = yf.Ticker(yf_tick).history(start=start_iso, end=end_iso, auto_adjust=True)
        if df is None or len(df) < min_bars:
            return {"_error": f"{sym} — only {len(df) if df is not None else 0} bars"}
        df_s    = df.tail(n_score)
        closes  = df_s["Close"].values.astype(float)
        highs   = df_s["High"].values.astype(float)
        lows    = df_s["Low"].values.astype(float)
        rets    = np.diff(closes) / closes[:-1]
        mid_idx = len(closes) // 2
        return {
            "sym":        sym,
            "yf":         yf_tick,
            "name":       name,
            "bars":       len(df_s),
            "ret_1y":     (closes[-1] - closes[0]) / closes[0],
            "ret_6m":     (closes[-1] - closes[mid_idx]) / closes[mid_idx],
            "sharpe":     _sharpe(rets),
            "max_dd":     _max_drawdown(closes),
            "trend_pct":  _ma50_pct_above(closes),
            "atr_pct":    _atr_pct(closes, highs, lows),
            "autocorr":   _autocorr_lag1(rets),
            "last_close": closes[-1],
            "data_scope": {
                "start_date":  str(df_s.index[0])[:10],
                "end_date":    str(df_s.index[-1])[:10],
                "bars":        len(df_s),
                "price_start": round(float(closes[0]), 2),
                "price_end":   round(float(closes[-1]), 2),
                "price_min":   round(float(np.min(closes)), 2),
                "price_max":   round(float(np.max(closes)), 2),
                "trend": (
                    "UPTREND"   if (closes[-1] - closes[0]) / closes[0] > 0.05 else
                    "DOWNTREND" if (closes[-1] - closes[0]) / closes[0] < -0.05 else
                    "SIDEWAYS"
                ),
            },
        }
    except Exception as exc:
        return {"_error": f"{sym} — {exc}"}


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------

def screen(
    top_n: int = 8,
    min_bars: int = 200,
    lookback_years: int = 1,
    indices: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[str], list[dict], int]:
    """
    Screen the live A-share universe and rank by composite score.

    Returns
    -------
    top_syms      : List of top-N symbol strings, ranked by score.
    results       : All ranked result dicts (stocks with sufficient data).
    universe_size : Total candidates from the index before min_bars filter.
    """
    if indices is None:
        indices = ["000300"]

    candidates    = fetch_universe(indices=indices)
    universe_size = len(candidates)

    end     = date.today()
    start   = end - timedelta(days=int(lookback_years * 365 * 2.2))
    n_score = int(lookback_years * 252)

    if verbose:
        print(f"\n{'='*78}")
        print(f"  SCREENER  |  {'+'.join(indices)}  |  universe: {universe_size}  |  window: {lookback_years}yr")
        print(f"{'='*78}")
        print(f"  Fetching data in parallel (16 threads)…\n")

    results: list[dict] = []
    errors:  list[str]  = []

    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {
            pool.submit(
                _fetch_and_score,
                c["sym"], c["yf_ticker"], c["name"],
                start.isoformat(), end.isoformat(),
                min_bars, n_score,
            ): c
            for c in candidates
        }
        for fut in as_completed(futs):
            r = fut.result()
            if "_error" in r:
                errors.append(r["_error"])
            else:
                results.append(r)

    if not results:
        if verbose:
            print("  No data retrieved — check internet connection.")
        return [], [], universe_size

    # Normalise each metric to [0, 1] across the scored universe
    def _minmax(vals):
        lo, hi = min(vals), max(vals)
        return [0.5] * len(vals) if hi == lo else [(v - lo) / (hi - lo) for v in vals]

    norm_trend    = _minmax([r["trend_pct"] for r in results])
    norm_atr      = _minmax([r["atr_pct"]   for r in results])
    norm_autocorr = _minmax([r["autocorr"]  for r in results])
    norm_6m       = _minmax([r["ret_6m"]    for r in results])
    norm_dd       = _minmax([-r["max_dd"]   for r in results])

    W_TREND, W_ATR, W_AUTOCORR, W_MOM6M, W_DD = 0.25, 0.20, 0.20, 0.20, 0.15
    for i, r in enumerate(results):
        nt, na, nac, n6, ndd = (
            norm_trend[i], norm_atr[i], norm_autocorr[i], norm_6m[i], norm_dd[i],
        )
        r["score"] = W_TREND * nt + W_ATR * na + W_AUTOCORR * nac + W_MOM6M * n6 + W_DD * ndd
        # ── Causal trace: per-factor breakdown ───────────────────────────────
        r["causal_nodes"] = [
            {
                "factor":       "trend_pct",
                "label":        "Trend Quality",
                "description":  f"Price above MA50 for {r['trend_pct']:.0%} of the window",
                "raw_value":    round(r["trend_pct"], 4),
                "norm_value":   round(nt, 4),
                "weight":       W_TREND,
                "contribution": round(W_TREND * nt, 4),
                "direction":    "positive" if nt >= 0.6 else "negative" if nt < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - nt) * 100))}%",
            },
            {
                "factor":       "atr_pct",
                "label":        "Volatility (ATR %)",
                "description":  f"ATR = {r['atr_pct']:.2%} of price — {'good for sizing' if r['atr_pct'] > 0.012 else 'low volatility'}",
                "raw_value":    round(r["atr_pct"], 6),
                "norm_value":   round(na, 4),
                "weight":       W_ATR,
                "contribution": round(W_ATR * na, 4),
                "direction":    "positive" if na >= 0.6 else "negative" if na < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - na) * 100))}%",
            },
            {
                "factor":       "autocorr",
                "label":        "Return Momentum",
                "description":  f"Lag-1 autocorrelation {r['autocorr']:+.3f} — {'trending returns' if r['autocorr'] > 0.02 else 'mean-reverting' if r['autocorr'] < -0.02 else 'random walk'}",
                "raw_value":    round(r["autocorr"], 6),
                "norm_value":   round(nac, 4),
                "weight":       W_AUTOCORR,
                "contribution": round(W_AUTOCORR * nac, 4),
                "direction":    "positive" if nac >= 0.6 else "negative" if nac < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - nac) * 100))}%",
            },
            {
                "factor":       "ret_6m",
                "label":        "6-Month Return",
                "description":  f"6-month price change {r['ret_6m']:+.1%}",
                "raw_value":    round(r["ret_6m"], 6),
                "norm_value":   round(n6, 4),
                "weight":       W_MOM6M,
                "contribution": round(W_MOM6M * n6, 4),
                "direction":    "positive" if n6 >= 0.6 else "negative" if n6 < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - n6) * 100))}%",
            },
            {
                "factor":       "max_dd",
                "label":        "Drawdown Risk",
                "description":  f"Max drawdown {r['max_dd']:.1%} — {'low risk' if r['max_dd'] > -0.15 else 'moderate risk' if r['max_dd'] > -0.25 else 'high risk'}",
                "raw_value":    round(r["max_dd"], 6),
                "norm_value":   round(ndd, 4),
                "weight":       W_DD,
                "contribution": round(W_DD * ndd, 4),
                "direction":    "positive" if ndd >= 0.6 else "negative" if ndd < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - ndd) * 100))}%",
            },
        ]
        r["gate_checks"] = [
            {
                "check":     "min_bars",
                "label":     "Sufficient price history",
                "threshold": min_bars,
                "actual":    r["bars"],
                "passed":    True,
                "note":      f"{r['bars']} bars ≥ {min_bars} required",
            }
        ]

    results.sort(key=lambda r: r["score"], reverse=True)
    top_syms = [r["sym"] for r in results[:top_n]]

    if verbose:
        print(f"  Ranked {len(results)} / {universe_size} stocks\n")
        header = (
            f"  {'#':<3} {'Symbol':<12} {'Name':<22} {'1Y Ret':>7} {'6M Ret':>7} "
            f"{'Sharpe':>7} {'MaxDD':>7} {'Trend%':>7} {'ATR%':>6} {'Score':>7}"
        )
        print(header)
        print("  " + "-"*95)
        for rank, r in enumerate(results, 1):
            flag = "  ★ ADD" if rank <= top_n else ("  ✗ DROP" if rank > len(results) - 3 else "")
            print(
                f"  {rank:<3} {r['sym']:<12} {r['name']:<22} "
                f"{r['ret_1y']:>+7.1%} {r['ret_6m']:>+7.1%} "
                f"{r['sharpe']:>7.2f} {r['max_dd']:>7.1%} "
                f"{r['trend_pct']:>7.1%} {r['atr_pct']:>6.2%} "
                f"{r['score']:>7.3f}" + flag
            )
        if errors:
            print(f"\n  Skipped {len(errors)} stocks (insufficient data / fetch error)")
        print()

    return top_syms, results, universe_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live A-share screener")
    parser.add_argument("--top",      type=int,  default=6,          help="Top N (default: 6)")
    parser.add_argument("--min-bars", type=int,  default=200,        help="Min bars (default: 200)")
    parser.add_argument("--lookback", type=int,  default=1,          help="Lookback years (default: 1)")
    parser.add_argument("--indices",  nargs="+", default=["000300"], help="CSI codes, e.g. 000300 000905")
    args = parser.parse_args()
    screen(top_n=args.top, min_bars=args.min_bars, lookback_years=args.lookback, indices=args.indices)

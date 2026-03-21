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

        # ── Hot-stock filter metrics ────────────────────────────────────────
        # ret_20d: 20-day price change — used to detect recent surging stocks
        ret_20d = float((closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0.0)
        # price_52w_pct: where price sits in its 52-week high/low range (0=bottom, 1=top)
        hi52 = float(np.max(closes[-252:])) if len(closes) >= 252 else float(np.max(closes))
        lo52 = float(np.min(closes[-252:])) if len(closes) >= 252 else float(np.min(closes))
        price_52w_pct = float((closes[-1] - lo52) / (hi52 - lo52 + 1e-10))
        # dist_52w_high: fraction below 52-week peak (0=AT peak, 0.3=30% below)
        dist_52w_high = float(max(0.0, (hi52 - closes[-1]) / (hi52 + 1e-10)))
        # vol_60d: 60-day annualised daily return volatility (lower = more stable)
        vol_60d = float(np.std(rets[-60:]) * np.sqrt(252)) if len(rets) >= 60 else float(np.std(rets) * np.sqrt(252))

        # ── Moving averages: are we currently in an uptrend? ──────────────────────
        # MA60 filter: if price < 60-day MA, the stock is in a medium-term downtrend.
        # Stocks that peaked and are now falling will fail this regardless of 1Y return.
        ma20       = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
        ma60       = float(np.mean(closes[-60:])) if len(closes) >= 60 else closes[-1]
        above_ma20 = bool(closes[-1] >= ma20)
        above_ma60 = bool(closes[-1] >= ma60)

        # Recent return metrics (1M and 3M) — for display and post-pump filter
        ret_1m = float((closes[-1] / closes[-20]  - 1) if len(closes) >= 20 else 0.0)
        ret_3m = float((closes[-1] / closes[-63]  - 1) if len(closes) >= 63 else 0.0)

        # limit_up_60d: number of 涨停 days (≥9.5% daily gain) in last 60 bars
        # Non-zero = institutional presence (主力) confirmed; zero = stay away
        rets_60       = rets[-60:] if len(rets) >= 60 else rets
        limit_up_60d  = int((rets_60 >= 0.095).sum())

        # yang_ratio_60d: fraction of 阳线 (close > open) candles in last 60 bars
        # 阳多绿少 pattern: strong stocks have big up-candles and small down-candles
        opens_60       = df_s["Open"].values.astype(float)[-60:]
        closes_60      = closes[-60:]
        n60            = len(closes_60)
        yang_ratio_60d = float((closes_60 > opens_60).sum() / n60) if n60 >= 20 else 0.5

        return {
            "sym":           sym,
            "yf":            yf_tick,
            "name":          name,
            "bars":          len(df_s),
            "ret_1y":        (closes[-1] - closes[0]) / closes[0],
            "ret_6m":        (closes[-1] - closes[mid_idx]) / closes[mid_idx],
            "ret_3m":        ret_3m,
            "ret_1m":        ret_1m,
            "sharpe":        _sharpe(rets),
            "max_dd":        _max_drawdown(closes),
            "trend_pct":     _ma50_pct_above(closes),
            "atr_pct":       _atr_pct(closes, highs, lows),
            "autocorr":      _autocorr_lag1(rets),
            "last_close":    closes[-1],
            "ma20":          round(ma20, 2),
            "ma60":          round(ma60, 2),
            "above_ma20":    above_ma20,
            "above_ma60":    above_ma60,
            # Hot-stock guard metrics
            "ret_20d":        ret_20d,
            "price_52w_pct":  price_52w_pct,
            "dist_52w_high":  dist_52w_high,
            "vol_60d":        vol_60d,
            # CN-rule 1 metrics
            "limit_up_60d":   limit_up_60d,
            "yang_ratio_60d": yang_ratio_60d,
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
    # ── Hard filters: block hot stocks & near-peak stocks ──────────────────
    # Rule 1: 20-day gain > 25% → recent surge, likely over-heated
    # Rule 2: price in top 10% of 52-week range → near historical high, avoid
    hot_excluded: list[str] = []
    filtered: list[dict] = []
    for r in results:
        if r.get("ret_20d", 0.0) > 0.25:
            hot_excluded.append(f"{r['sym']} (20d +{r['ret_20d']:.0%})")
            continue
        if r.get("price_52w_pct", 0.5) > 0.90:
            hot_excluded.append(f"{r['sym']} (52w pos {r['price_52w_pct']:.0%})")
            continue
        filtered.append(r)
    if verbose and hot_excluded:
        print(f"  Hot-stock filter excluded {len(hot_excluded)}: {', '.join(hot_excluded[:8])}")
    results = filtered

    if not results:
        if verbose:
            print("  All candidates excluded by hot-stock filter.")
        return [], [], universe_size

    # ── Hard filter: must have had at least one 涨停 in last 60 days ──────────────
    # Per CN rule 1: if no 涨停 (≥9.5% day), no 主力 activity — don’t touch it.
    no_player: list[str] = []
    with_player: list[dict] = []
    for r in results:
        if r.get("limit_up_60d", 0) == 0:
            no_player.append(f"{r['sym']} (0 limit-ups/60d)")
        else:
            with_player.append(r)
    if verbose and no_player:
        print(f"  No-主力 filter excluded {len(no_player)}: {', '.join(no_player[:6])}")
    results = with_player

    if not results:
        if verbose:
            print("  All candidates excluded by no-主力 filter.")
        return [], [], universe_size

    # ── Hard filter: current price must be above 60-day MA ──────────────────────
    # A stock below MA60 is in a medium-term downtrend — it has peaked and is
    # falling. The trend_pct score is HISTORICAL (looks at the full lookback window)
    # and will still be high for a stock that went up 150% then crashed. The MA60
    # filter catches what trend_pct misses: currently declining stocks.
    ma_excluded: list[str] = []
    ma_passed:   list[dict] = []
    for r in results:
        if not r.get("above_ma60", True):
            ma_excluded.append(f"{r['sym']} (price {r['last_close']:.2f} < MA60 {r.get('ma60', 0):.2f})")
        else:
            ma_passed.append(r)
    if verbose and ma_excluded:
        print(f"  MA60 downtrend filter excluded {len(ma_excluded)}: {', '.join(ma_excluded[:6])}")
    results = ma_passed

    if not results:
        if verbose:
            print("  All candidates excluded by MA60 downtrend filter.")
        return [], [], universe_size

    # ── Hard filter: post-pump reversal rejection ────────────────────────────────
    # A stock with 1Y return > 100% that is now 10%+ off its 52-week peak is
    # almost certainly a commodity/theme pump that has turned over or is topping out.
    # 0.20 threshold was too loose: 天孚通信 +377% 1Y but only -16.5% off peak
    # slipped through. Now 0.10: any 100%+ gainer more than 10% off peak is excluded.
    pump_excluded: list[str] = []
    pump_passed:   list[dict] = []
    for r in results:
        if r.get("ret_1y", 0.0) > 1.0 and r.get("dist_52w_high", 0.0) > 0.10:
            pump_excluded.append(
                f"{r['sym']} (1Y +{r['ret_1y']:.0%}, {r['dist_52w_high']:.0%} off peak)"
            )
        else:
            pump_passed.append(r)
    if verbose and pump_excluded:
        print(f"  Post-pump filter excluded {len(pump_excluded)}: {', '.join(pump_excluded[:6])}")
    results = pump_passed

    if not results:
        if verbose:
            print("  All candidates excluded by post-pump filter.")
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
    # New value/safety signals
    norm_low_vol  = _minmax([-r.get("vol_60d",  0.02) for r in results])   # lower vol = higher score
    norm_52w_dist = _minmax([r.get("dist_52w_high", 0) for r in results])  # further from peak = higher score
    norm_yang     = _minmax([r.get("yang_ratio_60d", 0.5) for r in results])  # more 阳线 = higher score

    # Weights rebalanced to de-emphasise short-term momentum and reward
    # value, stability, and distance from recent highs (per CN market advice):
    #   TREND     0.20 (was 0.25) — still important but less dominant
    #   ATR       0.10 (was 0.20) — keep, but quality > tradability
    #   AUTOCORR  0.10 (was 0.20) — dampened; yang_ratio captures same info
    #   MOM_6M    0.05 (was 0.20) — 6m momentum rewards hot stocks → minimal weight
    #   DD        0.20 (was 0.15) — stronger safety penalty
    #   LOW_VOL   0.15 (NEW)      — reward low-volatility quality names
    #   DIST_52W  0.10 (NEW)      — reward stocks far from 52-week high
    #   YANG      0.10 (NEW)      — 阳多绿少 candle structure (阳线 fraction)
    W_TREND, W_ATR, W_AUTOCORR, W_MOM6M, W_DD, W_LOW_VOL, W_DIST_52W, W_YANG = (
        0.20, 0.10, 0.10, 0.05, 0.20, 0.15, 0.10, 0.10
    )
    for i, r in enumerate(results):
        nt, na, nac, n6, ndd, nlv, n52, nyang = (
            norm_trend[i], norm_atr[i], norm_autocorr[i], norm_6m[i], norm_dd[i],
            norm_low_vol[i], norm_52w_dist[i], norm_yang[i],
        )
        r["score"] = (
            W_TREND * nt + W_ATR * na + W_AUTOCORR * nac + W_MOM6M * n6
            + W_DD * ndd + W_LOW_VOL * nlv + W_DIST_52W * n52 + W_YANG * nyang
        )
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
            {
                "factor":       "vol_60d",
                "label":        "60d Volatility",
                "description":  f"Annualised 60d vol {r.get('vol_60d', 0):.1%} — {'stable/quality' if r.get('vol_60d', 0) < 0.20 else 'moderate' if r.get('vol_60d', 0) < 0.35 else 'high vol'}",
                "raw_value":    round(r.get("vol_60d", 0), 6),
                "norm_value":   round(nlv, 4),
                "weight":       W_LOW_VOL,
                "contribution": round(W_LOW_VOL * nlv, 4),
                "direction":    "positive" if nlv >= 0.6 else "negative" if nlv < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - nlv) * 100))}%",
            },
            {
                "factor":       "dist_52w_high",
                "label":        "Distance from 52w High",
                "description":  f"{r.get('dist_52w_high', 0):.1%} below 52-week peak — {'deep pullback' if r.get('dist_52w_high', 0) > 0.25 else 'moderate pullback' if r.get('dist_52w_high', 0) > 0.10 else 'near peak'}",
                "raw_value":    round(r.get("dist_52w_high", 0), 6),
                "norm_value":   round(n52, 4),
                "weight":       W_DIST_52W,
                "contribution": round(W_DIST_52W * n52, 4),
                "direction":    "positive" if n52 >= 0.6 else "negative" if n52 < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - n52) * 100))}%",
            },
            {
                "factor":       "yang_ratio_60d",
                "label":        "阳多绿少 Candle Structure",
                "description":  f"{r.get('yang_ratio_60d', 0.5):.0%} of last 60 candles are 阳线 (close>open) — {'strong structure' if r.get('yang_ratio_60d', 0.5) > 0.55 else 'weak structure' if r.get('yang_ratio_60d', 0.5) < 0.45 else 'neutral'}",
                "raw_value":    round(r.get("yang_ratio_60d", 0.5), 4),
                "norm_value":   round(nyang, 4),
                "weight":       W_YANG,
                "contribution": round(W_YANG * nyang, 4),
                "direction":    "positive" if nyang >= 0.6 else "negative" if nyang < 0.4 else "neutral",
                "percentile":   f"Top {max(1, int((1 - nyang) * 100))}%",
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
            },
            {
                "check":     "hot_stock",
                "label":     "Not a hot stock (20d gain ≤ 25%)",
                "threshold": 0.25,
                "actual":    round(r.get("ret_20d", 0), 4),
                "passed":    True,
                "note":      f"20d gain {r.get('ret_20d', 0):.1%} passed hard filter",
            },
            {
                "check":     "near_52w_high",
                "label":     "Not near 52-week high (≤ 90%)",
                "threshold": 0.90,
                "actual":    round(r.get("price_52w_pct", 0.5), 4),
                "passed":    True,
                "note":      f"52w position {r.get('price_52w_pct', 0.5):.0%} passed hard filter",
            },
            {
                "check":     "has_limit_up",
                "label":     "Has 涨停 in last 60 days (主力 present)",
                "threshold": 1,
                "actual":    r.get("limit_up_60d", 0),
                "passed":    True,
                "note":      f"{r.get('limit_up_60d', 0)} limit-up day(s) in last 60 bars",
            },
            {
                "check":     "above_ma60",
                "label":     "Price above 60-day MA (uptrend)",
                "threshold": None,
                "actual":    round(r.get("ma60", r["last_close"]), 2),
                "passed":    True,
                "note":      f"Close {r['last_close']:.2f} ≥ MA60 {r.get('ma60', r['last_close']):.2f}",
            },
            {
                "check":     "not_post_pump",
                "label":     "Not a post-pump reversal (1Y≤100% OR ≤10% off peak)",
                "threshold": None,
                "actual":    round(r.get("ret_1y", 0), 4),
                "passed":    True,
                "note":      f"1Y {r.get('ret_1y', 0):+.0%} · {r.get('dist_52w_high', 0):.0%} off peak",
            },
        ]

    results.sort(key=lambda r: r["score"], reverse=True)

    # ── Sector diversity cap: max 3 stocks per sector in the final top_n ────────────
    # Prevents the screener from returning all metals/gold/aluminum stocks when
    # a single commodity sector dominates the scoring in a given period.
    # Uses keyword detection on Chinese names (most reliable without external API).
    _SECTOR_KEYWORDS: dict[str, list[str]] = {
        "aluminum": ["铝"],
        "copper":   ["铜"],
        "gold":     ["黄金", "金矿", "黄金"],
        "steel":    ["钢", "锃铁"],
        "coal":     ["煤", "焦炭"],
        "oil_gas":  ["油", "石化", "天然气"],
        "mining":   ["矿业", "矿"],
        "bank":     ["银行"],
        "insurance":["保险"],
        "pharma":   ["医药", "制药", "生物"],
        "ev":       ["动力电池", "新能源", "电动"],
    }
    def _infer_sector(name: str) -> str:
        for sector, keywords in _SECTOR_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                return sector
        return "other"

    MAX_PER_SECTOR = 3
    sector_counts: dict[str, int] = {}
    diversified: list[dict] = []
    sector_capped: list[str] = []
    for r in results:
        sec = _infer_sector(r["name"])
        r["_inferred_sector"] = sec
        count = sector_counts.get(sec, 0)
        if sec != "other" and count >= MAX_PER_SECTOR:
            sector_capped.append(f"{r['sym']} ({sec})")
        else:
            sector_counts[sec] = count + 1
            diversified.append(r)
    if verbose and sector_capped:
        print(f"  Sector cap (max {MAX_PER_SECTOR}/sector) dropped: {', '.join(sector_capped[:8])}")
    results = diversified

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

"""
Feature Engineer — converts a list of Bar objects into an ML-ready DataFrame.

Features produced:
  Momentum    : ret_1d, ret_5d, ret_10d, ret_20d, roc_10, roc_20
  Volatility  : vol_5d, vol_20d, atr_14
  Trend       : rsi_14, macd_hist, macd_signal_ratio
  Mean-revert : bb_pos (Bollinger Band position), bb_width
  Volume      : vol_ratio (vs 20-day avg)
  Range       : price_52w_pos (position within 52-week high/low)

Labels (for supervised training):
  make_labels() → ternary classification: 1=BUY, 0=HOLD, -1=SELL
  based on forward N-day return exceeding ±threshold
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from myquant.models.bar import Bar

# Columns used as model input features (order matters — don't change without retraining)
FEATURE_COLS: list[str] = [
    # ── Core momentum & volatility ───────────────────────────────────────
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_20d",
    # ── Oscillators ──────────────────────────────────────────────────────
    "rsi_14",
    "macd_hist", "macd_signal_ratio",
    "bb_pos", "bb_width",
    # ── Volume & range ───────────────────────────────────────────────────
    "vol_ratio",
    "atr_14",
    "roc_10", "roc_20",
    "price_52w_pos",
    # ── NEW: regime & microstructure features ────────────────────────────
    "vol_regime",      # rolling vol percentile — where is current vol vs history
    "trend_strength",  # |ret_20d| / vol_20d — signal-to-noise of trend
    "close_loc",       # where close sits in day's high/low range
    "gap_pct",         # overnight gap (open vs prev close)
    "vol_trend",       # volume momentum (5d avg / 20d avg)
    "above_ma50",      # binary: is price above its own 50-day MA?
    "ma_spread",       # (MA50 - MA200) / MA200 — trend direction & strength
    # ── NEW: momentum breadth & flow ─────────────────────────────────────────
    "stoch_k",         # Stochastic %K (14-period): position in recent high/low range
    "stoch_d",         # Stochastic %D (3-period SMA of %K): slower signal line
    "cmf_20",          # Chaikin Money Flow (20-period): volume-weighted directional pressure
    "keltner_pos",     # Position in Keltner channel (ATR-based, complements bb_pos)
    "ma50_slope",      # 5-bar rate of change of MA50 — trend acceleration signal
]


# ── Main conversion ─────────────────────────────────────────────────────────

def bars_to_features(bars: list[Bar]) -> pd.DataFrame:
    """
    Convert a list of Bar objects to a feature DataFrame indexed by timestamp.
    Rows with any NaN are dropped (warm-up period for rolling windows).

    Args:
        bars: List of Bar objects, need not be pre-sorted (will be sorted internally).

    Returns:
        pd.DataFrame with FEATURE_COLS columns plus OHLCV source columns.
    """
    if len(bars) < 30:
        return pd.DataFrame()

    df = (
        pd.DataFrame(
            [
                {
                    "ts":     b.ts,
                    "open":   b.open,
                    "high":   b.high,
                    "low":    b.low,
                    "close":  b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]
        )
        .set_index("ts")
        .sort_index()
    )

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    ret    = close.pct_change()

    # ── Momentum ─────────────────────────────────────────────
    df["ret_1d"]  = ret
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["roc_10"]  = close.pct_change(10)
    df["roc_20"]  = close.pct_change(20)

    # ── Volatility ───────────────────────────────────────────
    df["vol_5d"]  = ret.rolling(5).std()
    df["vol_20d"] = ret.rolling(20).std()

    # ATR (normalized by close)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / (close + 1e-10)

    # ── RSI ──────────────────────────────────────────────────
    df["rsi_14"] = _rsi(close, 14)

    # ── Stochastic Oscillator (%K / %D) ─────────────────────
    # %K measures where close sits within the 14-day high/low range (0–100).
    # %D is a 3-period SMA of %K — the slow signal line.
    # Together they capture momentum divergences RSI can miss.
    roll_high14   = high.rolling(14).max()
    roll_low14    = low.rolling(14).min()
    df["stoch_k"] = (close - roll_low14) / (roll_high14 - roll_low14 + 1e-10) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── MACD ─────────────────────────────────────────────────
    ema12        = close.ewm(span=12, adjust=False).mean()
    ema26        = close.ewm(span=26, adjust=False).mean()
    macd_line    = ema12 - ema26
    signal_line  = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"]         = macd_line - signal_line
    df["macd_signal_ratio"] = macd_line / (close + 1e-10)

    # ── Bollinger Bands ──────────────────────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower + 1e-10
    df["bb_pos"]   = (close - bb_lower) / bb_range
    df["bb_width"] = bb_range / (bb_mid + 1e-10)

    # ── Volume ratio ─────────────────────────────────────────
    df["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-10)

    # ── Chaikin Money Flow (CMF, 20-period) ──────────────────
    # Money Flow Multiplier: +1.0 if close at high, −1.0 if close at low.
    # CMF > 0 = buying pressure; CMF < 0 = selling pressure.
    mfm            = (2 * close - high - low) / (high - low + 1e-10)
    mfv            = mfm * volume
    df["cmf_20"]  = mfv.rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)

    # ── 52-week price position ───────────────────────────────
    high_52w = high.rolling(252, min_periods=20).max()
    low_52w  = low.rolling(252,  min_periods=20).min()
    df["price_52w_pos"] = (close - low_52w) / (high_52w - low_52w + 1e-10)
    # ── Vol regime: rolling percentile of 20d vol over 252-bar window ──
    # High vol_regime (near 1.0) = unusually volatile → mean-reverting environment
    # Low vol_regime (near 0.0)  = calm market → trending environment
    df["vol_regime"] = df["vol_20d"].rolling(252, min_periods=60).rank(pct=True)

    # ── Trend strength: |ret_20d| / vol_20d (signal-to-noise ratio) ────
    # High value = strong directional move relative to noise → trend-following bet
    df["trend_strength"] = (df["ret_20d"].abs() / (df["vol_20d"] + 1e-10)).clip(0, 5)

    # ── Close location within day's range (0 = at low, 1 = at high) ───
    # Near 1.0 on up-day = buyers in control; near 0.0 on down-day = sellers in control
    df["close_loc"] = (close - low) / (high - low + 1e-10)

    # ── Overnight gap: open vs previous close ───────────────────────────
    df["gap_pct"] = (df["open"] / close.shift(1) - 1).clip(-0.10, 0.10)

    # ── Volume trend: 5d avg vs 20d avg ─────────────────────────────────
    # > 1.0 = recent volume surge; < 1.0 = volume drying up
    df["vol_trend"] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)

    # ── MA regime: self-contained trend signals ──────────────────────────
    # These let the model learn regime-conditional behaviour without look-ahead:
    # above_ma50  : binary flag (1 = price > own 50-day MA)
    # ma_spread   : (MA50 - MA200) / MA200 — positive = golden cross territory
    ma50  = close.rolling(50,  min_periods=20).mean()
    ma200 = close.rolling(200, min_periods=60).mean()
    df["above_ma50"] = (close > ma50).astype(float)
    df["ma_spread"]  = ((ma50 - ma200) / (ma200 + 1e-10)).clip(-0.20, 0.20)

    # ── Keltner Channel position ──────────────────────────────
    # ATR-based channel (1.5× ATR width) centred on the 20-day MA.
    # Complements bb_pos: Bollinger uses price std, Keltner uses ATR.
    # Values outside [0, 1] mean price has broken out of the channel.
    atr_abs         = df["atr_14"] * (close + 1e-10)  # convert pct ATR back to price units
    kc_mid          = close.rolling(20).mean()
    kc_upper        = kc_mid + 1.5 * atr_abs
    kc_lower        = kc_mid - 1.5 * atr_abs
    df["keltner_pos"] = (close - kc_lower) / (kc_upper - kc_lower + 1e-10)

    # ── MA50 slope (5-bar rate of change of the 50-day MA) ───
    # Positive slope = trend accelerating upward; negative = decelerating.
    # Clipped to ±5% to suppress extreme values during thin early-history periods.
    df["ma50_slope"] = ma50.pct_change(5).clip(-0.05, 0.05)

    return df.dropna(subset=FEATURE_COLS)


def make_labels(
    df: pd.DataFrame,
    forward_days: int = 5,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Create ternary classification labels from forward returns.

        +1  = forward return > +threshold  → BUY
        -1  = forward return < -threshold  → SELL
         0  = neutral band                 → HOLD

    Args:
        df          : DataFrame with a 'close' column, indexed by timestamp.
        forward_days: Number of bars ahead for the label horizon.
        threshold   : Minimum absolute return to trigger a directional label.

    Returns:
        pd.Series aligned to df.index (last `forward_days` rows will be NaN).
    """
    fwd_ret = df["close"].pct_change(forward_days).shift(-forward_days)
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[fwd_ret >  threshold] =  1
    labels[fwd_ret < -threshold] = -1
    return labels


def add_macro_features(df: pd.DataFrame, macro_snap) -> pd.DataFrame:
    """
    Broadcast scalar macro indicators onto every row of the features DataFrame.
    Useful for informing the model of the macro regime at training / inference time.

    Args:
        df         : Feature DataFrame from bars_to_features().
        macro_snap : MacroSnapshot instance (from myquant.data.fetchers.macro_fetcher).

    Returns:
        DataFrame with additional macro columns appended.
    """
    df = df.copy()
    df["macro_pmi"]      = macro_snap.china_pmi_mfg
    df["macro_cpi"]      = macro_snap.china_cpi_yoy
    df["macro_usdcny"]   = macro_snap.usdcny
    df["macro_us10y"]    = macro_snap.us_10y_yield
    df["macro_vix"]      = macro_snap.vix
    df["macro_risk_on"]  = 1.0 if macro_snap.regime == "RISK_ON"  else 0.0
    df["macro_risk_off"] = 1.0 if macro_snap.regime == "RISK_OFF" else 0.0
    return df


# ── Helpers ─────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

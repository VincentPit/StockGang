"""
Feature Engineer — converts a list of Bar objects into an ML-ready DataFrame.

Features produced:
  Momentum    : ret_1d, ret_60d, ret_120d, ret_20d   (gradual — no short-term surge signals)
  Volatility  : vol_20d, vol_120d, atr_14
  Oscillators : rsi_14 (Wilder EWM), macd_hist, stoch_k/d, wr_14, cci_20
  Mean-revert : bb_pos, bb_width, dist_52w_high
  Volume      : vol_ratio, candle_strength
  Range       : price_52w_pos, atr_14
  Regime      : vol_regime, trend_strength, above_ma50, ma_spread, ma50_slope
  Microstr.   : close_loc, yang_ratio, recent_limit_up

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
    # ── Gradual momentum (60d/120d) — punishes short-term hot surges ────
    # ret_5d / ret_10d removed: they bias the model toward recently surging (hot)
    # stocks. Replaced with slower momentum that favours steady climbers.
    "ret_1d", "ret_60d", "ret_120d", "ret_20d",
    # ── Volatility: long-window stability replaces 5d noise ─────────────
    # dist_52w_high (fraction below 52-week peak) rewards beaten-down stocks;
    # vol_120d captures long-term stability, favouring quality low-vol names.
    "dist_52w_high", "vol_20d", "vol_120d",
    # ── Oscillators ──────────────────────────────────────────────────────
    "rsi_14",
    "macd_hist", "macd_signal_ratio",
    "bb_pos", "bb_width",
    # ── Volume & range ───────────────────────────────────────────────────
    "vol_ratio",
    "atr_14",
    "wr_14", "cci_20",        # Williams %R (14-period) + Commodity Channel Index (20-period)
    "price_52w_pos",
    # ── Regime & microstructure features ─────────────────────────────────
    "vol_regime",         # rolling vol percentile — where is current vol vs history
    "trend_strength",     # |ret_20d| / vol_20d — signal-to-noise of trend
    "close_loc",          # where close sits in day's high/low range
    # ── CN-market regime features (rule 1 / rule 4) ───────────────────────
    # recent_limit_up: fraction of last 60d with ≥9.5% daily gain — non-zero
    #   confirms 主力 (institutional) presence; zero = avoid (no big-money action)
    # yang_ratio: fraction of 阳线 (close>open) in last 60d — 阳多绿少 strong-stock
    #   pattern; >0.55 = structurally bullish candle structure
    # candle_strength: (avg yang-body − avg yin-body) / close — big up candles
    #   + small down candles is the hallmark of a sustained uptrend
    "recent_limit_up",    # fraction of 60d bars that hit A-share limit-up (≥9.5%)
    "yang_ratio",         # fraction of candles where close > open (last 60 bars)
    "candle_strength",    # (avg_yang_body − avg_yin_body) / close, clipped ±0.05
    "above_ma50",         # binary: is price above its own 50-day MA?
    "ma_spread",          # (MA50 - MA200) / MA200 — trend direction & strength
    # ── Momentum breadth ─────────────────────────────────────────────────
    "stoch_k",            # Stochastic %K (14-period): position in recent high/low range
    "stoch_d",            # Stochastic %D (3-period SMA of %K): slower signal line
    "ma50_slope",         # 5-bar rate of change of MA50 — trend acceleration signal
    # ── New quality features ─────────────────────────────────────────────────
    "ret_5d",        # 5-day return — short-term momentum (useful with ADX filter)
    "adx_14",        # Average Directional Index (14-period): trend strength 0-100
    "cmf_20",        # Chaikin Money Flow (20-period): volume-weighted price direction
    "price_accel",   # momentum acceleration: ret_5d − ret_20d×0.25
    # ── Session structure & volatility regime ─────────────────────────────
    "overnight_gap", # (open − prev_close) / prev_close, ±10% clip: pre-session sentiment
    "vol_accel",     # vol_20d / vol_120d: volatility expansion ratio (regime-change signal)
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
    # Using 60d/120d instead of 5d/10d: rewards gradual uptrends, not hot surges.
    # A stock up 5% in 5 days (hot) looks the same as a stock up 50% in 60 days
    # (steady climber) when viewed through a 60d lens — but the model can
    # distinguish them via vol_20d and dist_52w_high.
    df["ret_1d"]   = ret
    df["ret_5d"]   = close.pct_change(5)
    df["ret_20d"]  = close.pct_change(20)
    df["ret_60d"]  = close.pct_change(60)
    df["ret_120d"] = close.pct_change(120)

    # ── Volatility ───────────────────────────────────────────
    # vol_120d: long-term stability signal — low value = quality, low-vol stock
    # (min_periods=60 so it becomes valid before the full 120-bar warm-up)
    df["vol_20d"]  = ret.rolling(20).std()
    df["vol_120d"] = ret.rolling(120, min_periods=60).std()

    # ── Overnight gap (pre-session sentiment) ────────────────────────────
    # Measures the gap between today's open and yesterday's close.
    # Positive = bullish overnight news/sentiment; negative = bearish gap.
    # Particularly informative for A-shares where limit-up/down gaps are common.
    df["overnight_gap"] = ((df["open"] - close.shift(1)) / (close.shift(1) + 1e-10)).clip(-0.10, 0.10)

    # ── Volatility acceleration (regime-change signal) ────────────────────
    # Ratio of short-term to long-term vol.  > 1.0 = vol expanding vs baseline
    # (potential breakout or breakdown); < 1.0 = vol compressing (calm market).
    df["vol_accel"] = (df["vol_20d"] / (df["vol_120d"] + 1e-10)).clip(0.0, 5.0)

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

    # ── Williams %R (14-period) ───────────────────────────────
    # Mirrors the 14-day high/low range in [0, 100] — inverse of Stochastic %K.
    # 0 = price at 14-day high (overbought); 100 = price at 14-day low (oversold).
    # Complementary to stoch_k: same window, opposite polarity.
    df["wr_14"] = (roll_high14 - close) / (roll_high14 - roll_low14 + 1e-10) * 100

    # ── Commodity Channel Index (20-period) ───────────────────
    # Measures how far the typical price has deviated from its 20-day MA,
    # scaled by mean absolute deviation.  Values outside ±100 signal extremes.
    # Clipped to ±300 to suppress outliers in illiquid/thin-bar periods.
    tp         = (high + low + close) / 3
    tp_ma      = tp.rolling(20).mean()
    tp_mad     = (tp - tp_ma).abs().rolling(20).mean()
    df["cci_20"] = ((tp - tp_ma) / (0.015 * tp_mad + 1e-10)).clip(-300, 300)

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

    # ── Candle structure (阳多绿少) ────────────────────────────
    # recent_limit_up: density of A-share 涨停 (≥9.5% daily gain) in last 60 bars.
    #   Non-zero value confirms institutional (主力) presence — a prerequisite.
    # yang_ratio: fraction of 阳线 (close > open) candles in last 60 bars.
    #   Strong stocks show 阳多绿少: many large up-candles, few small down-candles.
    # candle_strength: average yang-body minus average yin-body, normalised by
    #   close.  Positive = up-candles dominate; clipped ±0.05 to suppress outliers.
    limit_up_flag        = (ret >= 0.095).astype(float)
    df["recent_limit_up"] = limit_up_flag.rolling(60, min_periods=30).mean()

    yang_flag            = (close > df["open"]).astype(float)
    df["yang_ratio"]      = yang_flag.rolling(60, min_periods=30).mean()

    yang_body            = (close - df["open"]).clip(lower=0.0)
    yin_body             = (df["open"] - close).clip(lower=0.0)
    avg_yang_body        = yang_body.rolling(60, min_periods=20).mean()
    avg_yin_body         = yin_body.rolling(60,  min_periods=20).mean()
    df["candle_strength"] = ((avg_yang_body - avg_yin_body) / (close + 1e-10)).clip(-0.05, 0.05)

    # ── 52-week price position ───────────────────────────────
    high_52w = high.rolling(252, min_periods=20).max()
    low_52w  = low.rolling(252,  min_periods=20).min()
    df["price_52w_pos"] = (close - low_52w) / (high_52w - low_52w + 1e-10)

    # ── Distance from 52-week high ────────────────────────────────────────
    # Fraction below the 52-week peak: 0.0 = AT the 52w high (avoid — likely hot),
    # 0.30 = 30% below the 52w high (potential mean-reversion candidate).
    # Clipped at 0.80 to cap extreme outliers from deeply distressed stocks.
    df["dist_52w_high"] = ((high_52w - close) / (high_52w + 1e-10)).clip(0.0, 0.80)

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

    # ── MA regime: self-contained trend signals ──────────────────────────
    # These let the model learn regime-conditional behaviour without look-ahead:
    # above_ma50  : binary flag (1 = price > own 50-day MA)
    # ma_spread   : (MA50 - MA200) / MA200 — positive = golden cross territory
    ma50  = close.rolling(50,  min_periods=20).mean()
    ma200 = close.rolling(200, min_periods=60).mean()
    df["above_ma50"] = (close > ma50).astype(float)
    df["ma_spread"]  = ((ma50 - ma200) / (ma200 + 1e-10)).clip(-0.20, 0.20)

    # ── MA50 slope (5-bar rate of change of the 50-day MA) ───
    # Positive slope = trend accelerating upward; negative = decelerating.
    # Clipped to ±5% to suppress extreme values during thin early-history periods.
    df["ma50_slope"] = ma50.pct_change(5).clip(-0.05, 0.05)

    # ── ADX (Average Directional Index, 14-period) ────────────────────────────
    # Trend strength indicator: 0 = no trend, 100 = very strong trend.
    # ADX > 25 = trending; < 20 = choppy.  Uses Wilder EWM (alpha = 1/14).
    alpha14   = 1.0 / 14
    plus_dm   = (high - high.shift(1)).clip(lower=0)
    minus_dm  = (low.shift(1) - low).clip(lower=0)
    dm_mask   = plus_dm >= minus_dm          # bar where +DM dominates
    plus_dm   = plus_dm.where(dm_mask,    0.0)
    minus_dm  = minus_dm.where(~dm_mask,  0.0)
    smooth_tr  = tr.ewm(alpha=alpha14, min_periods=1, adjust=False).mean()
    smooth_pdm = plus_dm.ewm(alpha=alpha14, min_periods=1, adjust=False).mean()
    smooth_mdm = minus_dm.ewm(alpha=alpha14, min_periods=1, adjust=False).mean()
    plus_di14  = 100.0 * smooth_pdm / (smooth_tr + 1e-10)
    minus_di14 = 100.0 * smooth_mdm / (smooth_tr + 1e-10)
    dx         = 100.0 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14 + 1e-10)
    df["adx_14"] = dx.ewm(alpha=alpha14, min_periods=1, adjust=False).mean()

    # ── Chaikin Money Flow (CMF, 20-period) ───────────────────────────────────
    # Where the close sits in the bar's H-L range, weighted by volume.
    # Positive = persistent buying pressure; negative = selling pressure.
    mfm          = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv          = mfm * volume
    df["cmf_20"] = mfv.rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)

    # ── Price acceleration (momentum second derivative) ───────────────────────
    # Positive = short-term momentum faster than medium-term → early breakout
    # Negative = momentum decelerating → possible trend fade
    df["price_accel"] = (df["ret_5d"] - df["ret_20d"] * 0.25).clip(-0.10, 0.10)

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


def make_labels_rl(
    df: pd.DataFrame,
    forward_days: int = 5,
    threshold: float = 0.01,
    commission_rate: float = 0.0003,
) -> pd.Series:
    """
    Cost-aware ternary labels for RL-style LGBM training.

    Identical to make_labels() but the threshold is raised by the estimated
    round-trip commission (buy + sell).  This prevents the model from labelling
    marginal-edge bars as BUY/SELL — trades that commission would immediately
    put underwater.

    Example: threshold=0.015, commission_rate=0.0003
      → effective threshold = 0.015 + 2×0.0003 = 0.0156
      → model only labels as BUY if forward_return > 1.56% (net of costs)

    Parameters
    ----------
    df              : DataFrame with a 'close' column, indexed by timestamp.
    forward_days    : Label horizon in bars.
    threshold       : Minimum *net* forward return to label as directional.
    commission_rate : One-way commission fraction (e.g. 0.0003 = 0.03%).
                      Round-trip cost = 2 × commission_rate.

    Returns
    -------
    pd.Series of {-1, 0, +1} aligned to df.index.
    """
    round_trip_cost = 2.0 * commission_rate
    effective_thresh = threshold + round_trip_cost
    fwd_ret = df["close"].pct_change(forward_days).shift(-forward_days)
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[fwd_ret >  effective_thresh] =  1
    labels[fwd_ret < -effective_thresh] = -1
    return labels


def make_labels_adaptive(
    df: pd.DataFrame,
    forward_days: int = 5,
    threshold: float = 0.015,
    commission_rate: float = 0.0003,
    vol_window: int = 20,
    vol_multiplier: float = 1.2,
) -> pd.Series:
    """
    Volatility-adaptive ternary labels.

    The effective threshold scales with the stock's own recent daily volatility:
        eff_thresh(t) = max(base_thresh,
                            vol_multiplier × rolling_vol(t) × √forward_days)

    High-vol stocks require a bigger forward move to be labelled BUY/SELL,
    preventing the model from learning noisy marginal-edge patterns that are
    indistinguishable from random noise.  Low-vol stocks are not over-penalised.

    Parameters
    ----------
    df              : DataFrame with a 'close' column, indexed by timestamp.
    forward_days    : Label horizon in bars.
    threshold       : Minimum base threshold (floor when vol is very low).
    commission_rate : One-way commission — adds to base threshold.
    vol_window      : Rolling window for daily vol estimation (default 20).
    vol_multiplier  : 1-sigma-×-sqrt(T) scale factor (default 1.2).

    Returns
    -------
    pd.Series of {-1, 0, +1} aligned to df.index.
    """
    round_trip  = 2.0 * commission_rate
    base_thresh = threshold + round_trip
    fwd_ret     = df["close"].pct_change(forward_days).shift(-forward_days)
    daily_vol   = df["close"].pct_change().rolling(vol_window, min_periods=10).std()
    vol_thresh  = (vol_multiplier * daily_vol * (forward_days ** 0.5)).clip(base_thresh, 0.10)
    eff_thresh  = vol_thresh.fillna(base_thresh)

    labels = pd.Series(0, index=df.index, dtype=int)
    labels[fwd_ret >  eff_thresh] =  1
    labels[fwd_ret < -eff_thresh] = -1
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
    """
    Wilder's RSI using exponential smoothing (alpha = 1/period).

    More faithful to Wilder's original formulation than a simple rolling mean;
    produces smoother readings and better converges to the classic 70/30 levels.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)

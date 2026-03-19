"""
api/advisor.py — Investment advisor: model training, stock analysis, recommendations.

Capabilities
────────────
1. train_for_symbol(symbol, force=False)
   Train (or load from DB) a LightGBM model on 1 year of daily price data.
   Staleness heuristics prevent unnecessary retraining.

2. analyze_stock(symbol, force_retrain=False)
   Full real-time analysis: LGBM signal + probabilities, fundamentals, recent news,
   feature importance, current price snapshot.

3. get_recommendations(sector=None, top_n=10)
   Score and rank the whole universe (or a sector subset) using
   fundamentals + momentum + stored LGBM signal.  Does NOT trigger training
   for symbols that have no stored model — callers do that explicitly.

Staleness rules
───────────────
• STALE_MODEL_DAYS  (30): retrain if model was trained > 30 calendar days ago.
• MIN_NEW_BARS       (5): need at least 5 more bars than the stored bar_count
                          before a retrain is considered worthwhile.
• If new_bar_count ≤ stored_bar_count + MIN_NEW_BARS → skip ("insufficient_new_data").
• If trained_at < now - 30 days                       → retrain ("model_outdated").
• Otherwise                                           → skip ("fresh").
"""
from __future__ import annotations

import logging
import pickle
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from api import db as _db

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

STALE_MODEL_DAYS: int  = 30    # calendar days before a model is considered outdated
MIN_NEW_BARS:     int  = 5     # minimum extra bars needed to justify a retrain
LOOKBACK_DAYS:    int  = 400   # fetch this many calendar days for training (≈ 1 yr of trading days + buffer)
FORWARD_DAYS:     int  = 5     # label horizon (predict 5-day forward return)
THRESHOLD:        float = 0.015 # ±1.5% threshold for BUY/SELL labels
TRAIN_RATIO:      float = 0.70
STRATEGY_ID:      str  = "lgbm_core"

# LightGBM hyper-parameters — kept in sync with LGBMStrategy defaults
_LGB_PARAMS = dict(
    num_leaves        = 63,      # was 31; more capacity for 27-feature input
    n_estimators      = 500,     # upper bound — early stopping decides actual count
    learning_rate     = 0.03,    # slower LR pairs well with more trees
    feature_fraction  = 0.75,    # was 0.80
    bagging_fraction  = 0.75,    # was 0.80
    bagging_freq      = 5,
    min_child_samples = 25,      # was 20
    reg_alpha         = 0.05,    # L1 regularisation (new)
    reg_lambda        = 0.10,    # L2 regularisation (new)
    min_split_gain    = 0.001,   # minimum gain to create a split (new)
    verbose           = -1,
    class_weight      = "balanced",
)

# ── Sector map ────────────────────────────────────────────────────────────────
# Universe restricted to Shanghai (sh) and Shenzhen (sz) A-share markets only.

SECTOR_MAP: dict[str, str] = {
    # ── Shanghai A-shares ──────────────────────────────────────────────────
    "sh600519": "consumer",   # Kweichow Moutai
    "sh600036": "finance",    # CMB
    "sh601318": "finance",    # Ping An Insurance
    "sh601398": "finance",    # ICBC
    "sh601166": "finance",    # Industrial Bank
    "sh600900": "energy",     # Yangtze Power
    "sh601088": "energy",     # China Shenhua
    "sh601899": "materials",  # Zijin Mining
    "sh600276": "healthcare", # Hengrui Medicine
    "sh600009": "transport",  # Shanghai Airport
    "sh600887": "consumer",   # Inner Mongolia Yili
    "sh601012": "materials",  # LONGi Green Energy
    "sh600585": "materials",  # Anhui Conch Cement
    "sh600031": "industrial", # SANY Heavy Industry
    "sh601888": "consumer",   # China Intl Travel
    # ── Shenzhen A-shares ──────────────────────────────────────────────────
    "sz300750": "ev",         # CATL
    "sz002594": "ev",         # BYD A
    "sz000858": "consumer",   # Wuliangye
    "sz000333": "consumer",   # Midea Group
    "sz002415": "tech",       # Hikvision
    "sz000100": "tech",       # TCL Technology
    "sz000001": "finance",    # Ping An Bank
    "sz300059": "finance",    # East Money
    "sz002352": "transport",  # S.F. Holding
    "sz000568": "consumer",   # Luzhou Laojiao
    "sz002714": "consumer",   # Muyuan Foods
    "sz300015": "healthcare", # Aier Eye Hospital
    "sz002304": "consumer",   # Haitian Flavouring
}

KNOWN_SECTORS = sorted(set(SECTOR_MAP.values()))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_bars_df(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch a DataFrame of daily OHLCV bars for ``symbol``.

    Falls back to the yfinance-backed get_price_sync helper so we reuse
    the existing SQLite cache layer (avoids double network calls).
    """
    from api.runner import get_price_sync
    result = get_price_sync(symbol, days=days)
    bars = result.get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Rename columns to lowercase for feature_engineer compatibility
    df = df.rename(columns=str.lower)
    # Ensure turnover column exists
    if "turnover" not in df.columns:
        df["turnover"] = df["volume"] * df["close"]
    return df


def _df_to_bars(symbol: str, df: pd.DataFrame) -> list:
    """Convert a price DataFrame to a list of Bar objects."""
    from myquant.models.bar import Bar, BarInterval
    bars = []
    for ts, row in df.iterrows():
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.fromisoformat(str(ts))
        bars.append(Bar(
            symbol   = symbol,
            ts       = dt,
            interval = BarInterval.D1,
            open     = float(row["open"]),
            high     = float(row["high"]),
            low      = float(row["low"]),
            close    = float(row["close"]),
            volume   = int(row.get("volume", 0)),
            turnover = float(row.get("turnover", 0.0)),
        ))
    return bars


def _train_lgbm(
    symbol: str,
    df: pd.DataFrame,
) -> tuple[Any, float, list[str], str]:
    """
    Train a LightGBM classifier on price DataFrame ``df``, with:
      • Early stopping (50 rounds) on a held-out calibration/validation set.
      • Sigmoid calibration (Platt scaling) on the same holdout for reliable
        probability estimates at inference time.
      • 3-way temporal split: 60% train | 15% calib/early-stop | 25% OOS.

    Returns
    -------
    model         : calibrated LGBMClassifier (picklable)
    oos_accuracy  : float  (0–1)  measured on the 25% OOS slice
    feature_cols  : list[str]
    last_bar_date : str  (ISO date of the last bar in ``df``)
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed") from exc

    from myquant.strategy.ml.feature_engineer import FEATURE_COLS, bars_to_features, make_labels

    bars = _df_to_bars(symbol, df)
    if len(bars) < 60:
        raise ValueError(f"Not enough bars for {symbol}: {len(bars)}")

    feat_df = bars_to_features(bars)
    if feat_df.empty:
        raise ValueError(f"Feature engineering produced empty DataFrame for {symbol}")

    labels  = make_labels(feat_df, forward_days=FORWARD_DAYS, threshold=THRESHOLD)
    aligned = feat_df.join(labels.rename("label")).dropna()

    if len(aligned) < 60:
        raise ValueError(f"Too few labeled rows for {symbol}: {len(aligned)}")

    feat_cols = list(FEATURE_COLS)

    # ── 3-way temporal split ──────────────────────────────────
    # │ 60% LGBM train │ 15% early-stop + calib │ 25% OOS accuracy │
    n = len(aligned)
    n_lgbm  = int(n * 0.60)
    n_calib = int(n * 0.15)

    if n_lgbm < 40 or n_calib < 10:
        # Scarce data fallback: simple 70 / 30 split, no early stopping
        n_train = int(n * TRAIN_RATIO)
        X_train = aligned.iloc[:n_train][feat_cols]
        y_train = aligned.iloc[:n_train]["label"] + 1
        X_val   = aligned.iloc[n_train:][feat_cols]
        y_val   = aligned.iloc[n_train:]["label"] + 1
        X_oos, y_oos = X_val, y_val
        use_early_stop = False
    else:
        X_train = aligned.iloc[:n_lgbm][feat_cols]
        y_train = aligned.iloc[:n_lgbm]["label"] + 1
        X_val   = aligned.iloc[n_lgbm : n_lgbm + n_calib][feat_cols]
        y_val   = aligned.iloc[n_lgbm : n_lgbm + n_calib]["label"] + 1
        X_oos   = aligned.iloc[n_lgbm + n_calib:][feat_cols]
        y_oos   = aligned.iloc[n_lgbm + n_calib:]["label"] + 1
        use_early_stop = True

    model = lgb.LGBMClassifier(**_LGB_PARAMS)

    if use_early_stop:
        try:
            from lightgbm import early_stopping as _es, log_evaluation as _le
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[_es(50, verbose=False), _le(period=0)],
            )
        except Exception:
            # Older LightGBM (< 4.0) fallback
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=50,
                      verbose=False)
    else:
        model.fit(X_train, y_train)

    # ── Probability calibration ────────────────────────────────
    # Sigmoid calibration makes confidence scores more trustworthy for the
    # trading threshold gate.  Fit on the calibration holdout (not OOS).
    cal_model = model  # default: uncalibrated
    if use_early_stop and len(X_val) >= 20:
        try:
            from sklearn.calibration import CalibratedClassifierCV
            calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrator.fit(X_val, y_val)
            cal_model = calibrator
        except Exception as e:
            _log.debug("_train_lgbm: calibration failed (%s) — using raw probas", e)

    # Out-of-sample accuracy (measured on held-out 25% OOS slice)
    oos_acc = 0.0
    if len(X_oos) > 0:
        oos_acc = float((cal_model.predict(X_oos) == y_oos.values).mean())

    last_bar_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])[:10]
    return cal_model, oos_acc, feat_cols, last_bar_date


def _is_stale(meta: dict, new_bar_count: int) -> tuple[bool, str]:
    """
    Decide whether a stored model should be retrained.

    Returns ``(should_retrain, reason)`` where reason is one of:
      "fresh"                — model is up to date, skip
      "insufficient_new_data"— not enough new bars to warrant a retrain
      "model_outdated"       — trained_at > STALE_MODEL_DAYS ago
      "new_data_available"   — enough new bars present
    """
    now = time.time()

    # Rule 1: model too old by calendar time
    if now - meta["trained_at"] > STALE_MODEL_DAYS * 86400:
        return True, "model_outdated"

    extra_bars = new_bar_count - meta["bar_count"]

    # Rule 2: not enough new data
    if extra_bars < MIN_NEW_BARS:
        return False, "insufficient_new_data"

    # Rule 3: enough new bars → retrain
    return True, "new_data_available"


def _predict_signal(model: Any, feat_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """
    Run inference on the latest row of ``feat_df``.

    Returns a dict with keys: signal, confidence, p_buy, p_hold, p_sell.
    """
    if feat_df.empty or not all(c in feat_df.columns for c in feat_cols):
        return {"signal": "HOLD", "confidence": 0.0, "p_buy": 0.0, "p_hold": 1.0, "p_sell": 0.0}

    latest = feat_df.iloc[[-1]][feat_cols]
    proba  = model.predict_proba(latest)[0]   # [p_sell, p_hold, p_buy]
    pred   = int(model.predict(latest)[0])    # 0=SELL, 1=HOLD, 2=BUY
    conf   = float(proba[pred])

    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return {
        "signal":     label_map[pred],
        "confidence": round(conf, 4),
        "p_buy":      round(float(proba[2]), 4),
        "p_hold":     round(float(proba[1]), 4),
        "p_sell":     round(float(proba[0]), 4),
    }


def _feature_importance(model: Any, feat_cols: list[str]) -> list[dict]:
    """Return feature importances sorted by gain, top-15.
    Unwraps CalibratedClassifierCV wrappers to reach the raw LGBMClassifier.
    """
    try:
        import numpy as np
        # CalibratedClassifierCV stores the base estimator as .estimator
        estimator = getattr(model, "estimator", model)
        imp = estimator.feature_importances_
        ranked = sorted(zip(feat_cols, imp), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 2)} for f, v in ranked[:15]]
    except Exception:
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def train_for_symbol(symbol: str, force: bool = False) -> dict:
    """
    Train (or load) a LightGBM model for ``symbol``.

    Parameters
    ----------
    symbol : str   Tencent-style symbol, e.g. "hk00700"
    force  : bool  If True, always retrain regardless of staleness.

    Returns
    -------
    dict with keys:
        model_id, symbol, strategy_id, trained_at, bar_count,
        last_bar_date, oos_accuracy, feature_cols,
        train_status  ("trained" | "loaded" | "skipped"),
        skip_reason   (set when train_status == "skipped")
    """
    # Load price data first — we need bar_count regardless of whether we retrain
    df = _load_bars_df(symbol, days=LOOKBACK_DAYS)
    if df.empty:
        raise ValueError(f"No price data available for {symbol}")

    new_bar_count = len(df)

    # Check existing model
    existing_meta = _db.get_model_meta(symbol, STRATEGY_ID)

    if existing_meta is not None and not force:
        should_retrain, reason = _is_stale(existing_meta, new_bar_count)
        if not should_retrain:
            _log.info("train_for_symbol %s: skipped (%s)", symbol, reason)
            return {**existing_meta, "train_status": "skipped", "skip_reason": reason}

    # ── Train ─────────────────────────────────────────────────────────────────────
    _log.info("train_for_symbol %s: training on %d bars", symbol, new_bar_count)
    model, oos_acc, feat_cols, last_bar_date = _train_lgbm(symbol, df)
    model_blob = pickle.dumps(model)

    mid = _db.save_model(
        symbol        = symbol,
        strategy_id   = STRATEGY_ID,
        bar_count     = new_bar_count,
        last_bar_date = last_bar_date,
        oos_accuracy  = oos_acc,
        model_blob    = model_blob,
        feature_cols  = feat_cols,
    )

    return {
        "model_id":      mid,
        "symbol":        symbol,
        "strategy_id":   STRATEGY_ID,
        "trained_at":    time.time(),
        "bar_count":     new_bar_count,
        "last_bar_date": last_bar_date,
        "oos_accuracy":  round(oos_acc, 4),
        "feature_cols":  feat_cols,
        "train_status":  "trained",
        "skip_reason":   None,
    }


def analyze_stock(symbol: str, force_retrain: bool = False) -> dict:
    """
    Full real-time analysis for a single stock.

    1. Train / load LGBM model (respects staleness rules).
    2. Compute current signal + class probabilities.
    3. Fetch fundamentals and 5 recent news headlines.
    4. Return combined analysis dict.
    """
    from myquant.strategy.ml.feature_engineer import FEATURE_COLS, bars_to_features
    from api.runner import get_fundamentals_sync, get_news_sync

    # ── Model ─────────────────────────────────────────────────────────────
    train_meta = train_for_symbol(symbol, force=force_retrain)

    # Load the persisted model blob
    loaded = _db.load_model(symbol, STRATEGY_ID)
    if loaded is None:
        raise RuntimeError(f"Model not found in DB after training for {symbol}")
    model_blob, meta = loaded
    model = pickle.loads(model_blob)
    feat_cols = meta["feature_cols"]

    # ── Signal ────────────────────────────────────────────────────────────
    df = _load_bars_df(symbol, days=LOOKBACK_DAYS)
    bars     = _df_to_bars(symbol, df) if not df.empty else []
    feat_df  = bars_to_features(bars) if bars else pd.DataFrame()
    signal   = _predict_signal(model, feat_df, feat_cols)
    feat_imp = _feature_importance(model, feat_cols)

    # ── Recent price snapshot ─────────────────────────────────────────────
    recent_bars: list[dict] = []
    if not df.empty:
        tail = df.tail(30)
        for ts, row in tail.iterrows():
            recent_bars.append({
                "date":   str(ts)[:10],
                "open":   round(float(row["open"]),  4),
                "high":   round(float(row["high"]),  4),
                "low":    round(float(row["low"]),   4),
                "close":  round(float(row["close"]), 4),
                "volume": float(row.get("volume", 0)),
            })

    # ── Momentum metrics ──────────────────────────────────────────────────
    momentum: dict = {}
    if not df.empty and "close" in df.columns:
        closes = df["close"].dropna()
        if len(closes) >= 2:
            momentum["ret_1d"] = round(float(closes.iloc[-1] / closes.iloc[-2] - 1), 6)
        if len(closes) >= 5:
            momentum["ret_5d"] = round(float(closes.iloc[-1] / closes.iloc[-5] - 1), 6)
        if len(closes) >= 20:
            momentum["ret_1m"] = round(float(closes.iloc[-1] / closes.iloc[-20] - 1), 6)
        if len(closes) >= 60:
            momentum["ret_3m"] = round(float(closes.iloc[-1] / closes.iloc[-60] - 1), 6)
        if len(closes) >= 252:
            momentum["ret_1y"] = round(float(closes.iloc[-1] / closes.iloc[-252] - 1), 6)
        momentum["current_price"] = round(float(closes.iloc[-1]), 4)

    # ── Fundamentals ──────────────────────────────────────────────────────
    fundamentals = get_fundamentals_sync(symbol)

    # ── News ──────────────────────────────────────────────────────────────
    news_result = get_news_sync(symbol, limit=5)
    news = news_result.get("items", [])

    return {
        "symbol":        symbol,
        "sector":        SECTOR_MAP.get(symbol, "unknown"),
        "signal":        signal["signal"],
        "confidence":    signal["confidence"],
        "p_buy":         signal["p_buy"],
        "p_hold":        signal["p_hold"],
        "p_sell":        signal["p_sell"],
        "momentum":      momentum,
        "fundamentals":  fundamentals,
        "news":          news,
        "recent_bars":   recent_bars,
        "feature_importance": feat_imp,
        "model_meta": {
            "model_id":      meta["model_id"],
            "trained_at":    meta["trained_at"],
            "bar_count":     meta["bar_count"],
            "last_bar_date": meta["last_bar_date"],
            "oos_accuracy":  meta["oos_accuracy"],
            "train_status":  train_meta.get("train_status", "loaded"),
            "skip_reason":   train_meta.get("skip_reason"),
        },
    }


def get_recommendations(sector: str | None = None, top_n: int = 10) -> list[dict]:
    """
    Score and rank the symbol universe (optionally filtered by sector).

    Scoring formula (all components normalised 0–1):
      score = 0.35 × fundamentals_composite
            + 0.35 × momentum_score
            + 0.30 × model_signal_score   (if model exists; else 0)

    ``model_signal_score`` uses the *stored* LGBM signal — no live training
    is triggered here so recommendations are always fast.

    Returns a list of dicts sorted by descending score.
    """
    from api.runner import get_fundamentals_sync, get_price_sync
    from myquant.strategy.ml.feature_engineer import FEATURE_COLS, bars_to_features

    # Build universe from the curated SECTOR_MAP so every hand-picked stock is
    # scored regardless of the current CSI300/CSI500 constituent snapshot.
    # We fetch both indices only for name / yf_ticker resolution; the actual
    # set of symbols scored is always exactly SECTOR_MAP (optionally filtered).
    from myquant.data.fetchers.universe_fetcher import fetch_universe
    from myquant.data.fetchers.historical_loader import _symbol_to_yfinance
    _raw = fetch_universe(indices=["000300", "000905"])
    _index_lookup: dict[str, tuple[str, str]] = {
        r["sym"]: (r["yf_ticker"], r["name"]) for r in _raw
    }
    universe: dict[str, tuple[str, str]] = {}
    for _sym, _sym_sector in SECTOR_MAP.items():
        if sector is not None and _sym_sector != sector:
            continue
        if _sym in _index_lookup:
            universe[_sym] = _index_lookup[_sym]
        else:
            # Fallback for stocks absent from both index snapshots
            _yf = _symbol_to_yfinance(_sym)
            universe[_sym] = (_yf or _sym, _sym)

    rows: list[dict] = []

    for sym, (yf_ticker, name) in universe.items():
        try:
            # ── Price / momentum ──────────────────────────────────────────
            price_data = get_price_sync(sym, days=365)
            bars_list  = price_data.get("bars", [])
            if len(bars_list) < 30:
                continue

            closes = [b["close"] for b in bars_list]
            ret_1y = (closes[-1] / closes[0] - 1) if closes[0] else 0.0
            ret_3m = (closes[-1] / closes[max(0, len(closes)-63)] - 1)
            ret_1m = (closes[-1] / closes[max(0, len(closes)-20)] - 1)
            mom_raw = 0.4 * ret_3m + 0.4 * ret_1y + 0.2 * ret_1m

            # ── Fundamentals ──────────────────────────────────────────────
            fund = get_fundamentals_sync(sym)
            f_score = (
                fund.get("value_score", 0)
                + fund.get("growth_score", 0)
                + fund.get("quality_score", 0)
            ) / 3.0  # already 0–100 scale

            # ── Model signal (if available) ────────────────────────────────
            model_sig = 0.5   # neutral baseline
            model_confidence = 0.0
            model_signal_str = "N/A"
            p_buy_snap, p_hold_snap, p_sell_snap = 0.0, 1.0, 0.0

            stored = _db.load_model(sym, STRATEGY_ID)
            if stored is not None:
                try:
                    blob, meta = stored
                    m     = pickle.loads(blob)
                    df    = _load_bars_df(sym, days=LOOKBACK_DAYS)
                    bl    = _df_to_bars(sym, df) if not df.empty else []
                    fdf   = bars_to_features(bl) if bl else pd.DataFrame()
                    sig   = _predict_signal(m, fdf, meta["feature_cols"])
                    # Map signal → 0–1 score: BUY=1, HOLD=0.5, SELL=0
                    sig_map = {"BUY": 1.0, "HOLD": 0.5, "SELL": 0.0}
                    model_sig         = sig_map[sig["signal"]]
                    model_confidence  = sig["confidence"]
                    model_signal_str  = sig["signal"]
                    p_buy_snap        = sig["p_buy"]
                    p_hold_snap       = sig["p_hold"]
                    p_sell_snap       = sig["p_sell"]
                except Exception:
                    pass

            # ── Composite score ────────────────────────────────────────────
            # Normalise fundamentals to 0–1 (was 0–100)
            f_norm   = min(f_score / 100.0, 1.0)
            # Clip momentum to ±50% range, normalise to 0–1
            m_norm   = (min(max(mom_raw, -0.5), 0.5) + 0.5)

            composite = 0.35 * f_norm + 0.35 * m_norm + 0.30 * model_sig

            # ── Data scope (price window used) ────────────────────────────
            _ret_scope = (closes[-1] - closes[0]) / closes[0] if closes[0] else 0
            data_scope = {
                "start_date":  bars_list[0]["date"] if bars_list else "",
                "end_date":    bars_list[-1]["date"] if bars_list else "",
                "bars":        len(bars_list),
                "price_start": round(float(closes[0]), 2)   if closes else 0,
                "price_end":   round(float(closes[-1]), 2)  if closes else 0,
                "price_min":   round(float(min(closes)), 2) if closes else 0,
                "price_max":   round(float(max(closes)), 2) if closes else 0,
                "trend": (
                    "UPTREND"   if _ret_scope > 0.05 else
                    "DOWNTREND" if _ret_scope < -0.05 else
                    "SIDEWAYS"
                ),
            }

            # ── Causal trace ───────────────────────────────────────────────
            causal_nodes = [
                {
                    "factor":       "fundamentals",
                    "label":        "Fundamentals",
                    "description":  (
                        f"Value {fund.get('value_score', 0):.0f} · "
                        f"Growth {fund.get('growth_score', 0):.0f} · "
                        f"Quality {fund.get('quality_score', 0):.0f}  "
                        f"(composite {f_score:.1f}/100)"
                    ),
                    "raw_value":    round(f_score, 2),
                    "norm_value":   round(f_norm, 4),
                    "weight":       0.35,
                    "contribution": round(0.35 * f_norm, 4),
                    "direction":    "positive" if f_norm >= 0.55 else "negative" if f_norm < 0.35 else "neutral",
                    "percentile":   "",
                    "extras": {
                        "pe_ttm":        fund.get("pe_ttm", 0),
                        "pb":            fund.get("pb", 0),
                        "roe":           fund.get("roe", 0),
                        "value_score":   fund.get("value_score", 0),
                        "growth_score":  fund.get("growth_score", 0),
                        "quality_score": fund.get("quality_score", 0),
                    },
                },
                {
                    "factor":       "momentum",
                    "label":        "Price Momentum",
                    "description":  f"1Y {ret_1y:+.1%} · 3M {ret_3m:+.1%} · 1M {ret_1m:+.1%}",
                    "raw_value":    round(mom_raw, 4),
                    "norm_value":   round(m_norm, 4),
                    "weight":       0.35,
                    "contribution": round(0.35 * m_norm, 4),
                    "direction":    "positive" if m_norm >= 0.55 else "negative" if m_norm < 0.45 else "neutral",
                    "percentile":   "",
                    "extras": {
                        "ret_1y": round(ret_1y, 4),
                        "ret_3m": round(ret_3m, 4),
                        "ret_1m": round(ret_1m, 4),
                    },
                },
                {
                    "factor":       "model_signal",
                    "label":        "ML Signal (LightGBM)",
                    "description":  (
                        f"Signal: {model_signal_str} with {model_confidence:.0%} confidence"
                        if model_signal_str != "N/A"
                        else "No trained model — using neutral baseline (0.5)"
                    ),
                    "raw_value":    model_signal_str,
                    "norm_value":   round(model_sig, 4),
                    "weight":       0.30,
                    "contribution": round(0.30 * model_sig, 4),
                    "direction":    "positive" if model_sig >= 0.7 else "negative" if model_sig < 0.3 else "neutral",
                    "percentile":   "",
                    "extras": {
                        "signal":        model_signal_str,
                        "confidence":    round(model_confidence, 4),
                        "model_trained": stored is not None,
                        "p_buy":         p_buy_snap,
                        "p_hold":        p_hold_snap,
                        "p_sell":        p_sell_snap,
                    },
                },
            ]

            rows.append({
                "symbol":           sym,
                "yf_ticker":        yf_ticker,
                "name":             name,
                "sector":           SECTOR_MAP.get(sym, "unknown"),
                "score":            round(composite, 4),
                "model_signal":     model_signal_str,
                "model_confidence": round(model_confidence, 4),
                "model_trained":    stored is not None,
                "ret_1y":           round(ret_1y, 4),
                "ret_3m":           round(ret_3m, 4),
                "ret_1m":           round(ret_1m, 4),
                "fundamentals": {
                    "pe_ttm":         fund.get("pe_ttm", 0),
                    "pb":             fund.get("pb", 0),
                    "roe":            fund.get("roe", 0),
                    "revenue_growth": fund.get("revenue_growth", 0),
                    "net_margin":     fund.get("net_margin", 0),
                    "value_score":    fund.get("value_score", 0),
                    "growth_score":   fund.get("growth_score", 0),
                    "quality_score":  fund.get("quality_score", 0),
                },
                "causal_nodes": causal_nodes,
                "data_scope":   data_scope,
            })
        except Exception:
            # Skip symbols that fail — don't let one bad fetch break the whole list
            continue

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]

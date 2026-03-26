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
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
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

_OVERRIDES_PATH = Path(__file__).parent.parent / "model_overrides.json"

def _effective_lgb_params() -> dict:
    """
    Return _LGB_PARAMS merged with any overrides written by auto_tune.py.
    Only recognised LightGBM param keys are merged to prevent injection.
    """
    if not _OVERRIDES_PATH.exists():
        return dict(_LGB_PARAMS)
    try:
        overrides = json.loads(_OVERRIDES_PATH.read_text())
    except Exception:
        return dict(_LGB_PARAMS)
    allowed = set(_LGB_PARAMS.keys())
    merged  = dict(_LGB_PARAMS)
    merged.update({k: v for k, v in overrides.items() if k in allowed})
    _log.info("_effective_lgb_params: applied overrides %s",
              {k: v for k, v in overrides.items() if k in allowed})
    return merged

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


# ── Ensemble model wrapper ────────────────────────────────────────────────────

class _EnsembleModel:
    """
    Picklable wrapper that averages ``predict_proba`` across K sub-models.

    Each sub-model is an independently trained (and calibrated) LGBMClassifier
    over a different temporal window slice of the training data.  Averaging their
    probability outputs reduces sensitivity to any single training window and
    yields more robust uncertainty estimates.

    The class presents a minimal sklearn-style classifier interface so it is a
    drop-in replacement wherever a plain LGBMClassifier is expected.
    """

    def __init__(self, models: list) -> None:
        self.models = models  # list[CalibratedClassifierCV | LGBMClassifier]

    # ── Classifier interface ──────────────────────────────────────────────

    def predict_proba(self, X) -> "np.ndarray":
        """Return averaged probability matrix, shape (n_samples, n_classes)."""
        import numpy as np
        arrays = np.array([m.predict_proba(X) for m in self.models])  # (K, n, 3)
        return arrays.mean(axis=0)                                      # (n, 3)

    def predict(self, X) -> "np.ndarray":
        """Return predicted class (argmax of averaged probas), shape (n_samples,)."""
        import numpy as np
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Introspection helpers ─────────────────────────────────────────────

    @property
    def estimator(self):
        """
        Unwrap to the raw LGBMClassifier inside the first sub-model.
        Used by ``_feature_importance()`` to reach ``feature_importances_``.
        ``CalibratedClassifierCV`` exposes its base via ``.estimator``.
        """
        first = self.models[0]
        return getattr(first, "estimator", first)


def _train_single_window(
    feat_cols: list[str],
    aligned: "pd.DataFrame",
    lgb_params: dict,
    train_ratio: float,
) -> Any:
    """
    Train a single calibrated sub-model on one temporal slice.

    Parameters
    ----------
    feat_cols   : Feature column names.
    aligned     : DataFrame with feature columns + ``label`` column.
    lgb_params  : LightGBM hyperparameters dict.
    train_ratio : Fallback train/val split fraction when data is scarce.

    Returns
    -------
    Calibrated model, or ``None`` if the slice is too small.
    """
    try:
        import lightgbm as lgb
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        return None

    n       = len(aligned)
    n_lgbm  = int(n * 0.60)
    n_calib = int(n * 0.15)

    if n_lgbm < 40 or n_calib < 10:
        n_train = int(n * train_ratio)
        if n_train < 20:
            return None
        X_train = aligned.iloc[:n_train][feat_cols]
        y_train = aligned.iloc[:n_train]["label"] + 1
        X_val   = aligned.iloc[n_train:][feat_cols]
        y_val   = aligned.iloc[n_train:]["label"] + 1
        use_early_stop = False
    else:
        X_train = aligned.iloc[:n_lgbm][feat_cols]
        y_train = aligned.iloc[:n_lgbm]["label"] + 1
        X_val   = aligned.iloc[n_lgbm : n_lgbm + n_calib][feat_cols]
        y_val   = aligned.iloc[n_lgbm : n_lgbm + n_calib]["label"] + 1
        use_early_stop = True

    model = lgb.LGBMClassifier(**lgb_params)

    if use_early_stop:
        try:
            from lightgbm import early_stopping as _es
            from lightgbm import log_evaluation as _le
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[_es(50, verbose=False), _le(period=0)],
            )
        except Exception:
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=50,
                      verbose=False)
    else:
        model.fit(X_train, y_train)

    cal_model = model
    if use_early_stop and len(X_val) >= 20:
        try:
            calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrator.fit(X_val, y_val)
            cal_model = calibrator
        except Exception:
            pass

    return cal_model


def _train_lgbm(
    symbol: str,
    df: pd.DataFrame,
) -> tuple[Any, float, list[str], str]:
    """
    Train a walk-forward ensemble of LightGBM classifiers on price DataFrame ``df``.

    Three sub-models are trained on temporal slices of the labeled data:
      • Window 0 (full,   100%): long-term regime patterns
      • Window 1 (medium,  75%): medium-term emphasis
      • Window 2 (recent,  50%): current market regime focus

    Each sub-model uses 3-way temporal split (60% train / 15% calib / 25% OOS),
    early stopping, and sigmoid calibration.  Inference averages their probabilities.

    Returns
    -------
    model         : _EnsembleModel wrapping K calibrated sub-models (picklable)
    oos_accuracy  : float  measured on the full-window 25% OOS slice
    feature_cols  : list[str]
    last_bar_date : str  ISO date of the last bar in ``df``
    """
    try:
        import lightgbm as lgb  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed") from exc

    from myquant.strategy.ml.feature_engineer import FEATURE_COLS, bars_to_features, make_labels

    bars = _df_to_bars(symbol, df)
    if len(bars) < 60:
        raise ValueError(f"Not enough bars for {symbol}: {len(bars)}")

    feat_df = bars_to_features(bars)
    if feat_df.empty:
        raise ValueError(f"Feature engineering produced empty DataFrame for {symbol}")

    # Read overrides written by auto_tune.py (e.g. adjusted num_leaves, forward_days)
    params  = _effective_lgb_params()
    ov_path = Path(__file__).parent.parent / "model_overrides.json"
    ov      = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    eff_forward_days = int(ov.get("forward_days", FORWARD_DAYS))
    eff_threshold    = float(ov.get("threshold",    THRESHOLD))

    labels  = make_labels(feat_df, forward_days=eff_forward_days, threshold=eff_threshold)
    aligned = feat_df.join(labels.rename("label")).dropna()

    if len(aligned) < 60:
        raise ValueError(f"Too few labeled rows for {symbol}: {len(aligned)}")

    feat_cols = list(FEATURE_COLS)
    n         = len(aligned)

    # ── Walk-forward ensemble ─────────────────────────────────────────────
    # Train 3 sub-models on different temporal slices of the aligned data.
    # Averaging their probabilities at inference time yields more robust
    # uncertainty estimates than any single model.
    sub_models: list[Any] = []
    for frac in [1.0, 0.75, 0.50]:
        n_win     = max(60, int(n * frac))
        win_slice = aligned.iloc[-n_win:]
        sub       = _train_single_window(feat_cols, win_slice, params, TRAIN_RATIO)
        if sub is not None:
            sub_models.append(sub)

    if not sub_models:
        raise ValueError(f"No sub-models trained for {symbol}")

    ensemble_model = _EnsembleModel(sub_models)

    # ── OOS accuracy on the full-window 25% held-out slice ────────────────
    n_lgbm_full  = int(n * 0.60)
    n_calib_full = int(n * 0.15)
    X_oos = aligned.iloc[n_lgbm_full + n_calib_full:][feat_cols]
    y_oos = aligned.iloc[n_lgbm_full + n_calib_full:]["label"] + 1
    oos_acc = 0.0
    if len(X_oos) > 0:
        oos_acc = float((ensemble_model.predict(X_oos) == y_oos.values).mean())

    last_bar_date = (
        str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])[:10]
    )
    return ensemble_model, oos_acc, feat_cols, last_bar_date


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
    from api.runner import get_fundamentals_sync, get_news_sync
    from myquant.strategy.ml.feature_engineer import bars_to_features

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
    Score and rank the live A-share universe using value-biased,
    anti-hot-stock principles that mirror the stock screener.

    Universe
    ────────
    Full live CSI300 + CSI500 from fetch_universe (no hardcoded list).
    SECTOR_MAP provides sector labels for known symbols; others get "other".

    Hard filters (same as stock screener — applied before scoring):
      • 20-day gain > 25%    → excluded (hot-stock surge)
      • price_52w_pct > 90%  → excluded (near 52-week peak)

    Scoring formula (all components cross-sectionally normalised 0–1):
      composite = 0.40 × fundamentals_composite
                + 0.30 × model_signal_score   (stored LGBM; 0.5 if absent)
                + 0.30 × quality_position

      quality_position = 0.45 × dist_52w_high   (contrarian: beaten-down beats hot)
                       + 0.35 × low_vol          (quality / stability)
                       + 0.20 × ret_6m_norm      (small 6-month momentum, capped ±30%)

    ``model_signal_score`` uses the *stored* LGBM signal — no live training
    is triggered here so recommendations are always fast.

    Returns a list of dicts sorted by descending composite score.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from api.runner import get_fundamentals_sync, get_price_sync
    from myquant.data.fetchers.universe_fetcher import fetch_universe
    from myquant.strategy.ml.feature_engineer import bars_to_features

    # ── Universe: full live CSI300 + CSI500 (sector from SECTOR_MAP or "other") ─
    _raw = fetch_universe(indices=["000300", "000905"])
    universe: list[dict] = []
    for r in _raw:
        sym = r["sym"]
        sym_sector = SECTOR_MAP.get(sym, "other")
        if sector is not None and sym_sector != sector:
            continue
        universe.append({
            "sym":       sym,
            "yf_ticker": r["yf_ticker"],
            "name":      r["name"],
            "sector":    sym_sector,
        })

    # ── First pass: collect raw metrics concurrently ───────────────────────────
    def _fetch_one(entry: dict) -> dict | None:
        """Fetch price + fundamentals + stored model signal for one symbol."""
        sym = entry["sym"]
        try:
            price_data = get_price_sync(sym, days=400)
            bars_list  = price_data.get("bars", [])
            if len(bars_list) < 30:
                return None

            closes = [b["close"] for b in bars_list]
            n = len(closes)

            # ── Hot-stock filter metrics ───────────────────────────────────
            ret_20d = (closes[-1] / closes[max(0, n - 20)] - 1) if n >= 20 else 0.0
            hi52    = max(closes[max(0, n - 252):])
            lo52    = min(closes[max(0, n - 252):])
            price_52w_pct = (closes[-1] - lo52) / (hi52 - lo52 + 1e-10)
            dist_52w_high = max(0.0, (hi52 - closes[-1]) / (hi52 + 1e-10))

            # ── Low-volatility metric (annualised daily vol, inverted) ──────
            window      = closes[max(0, n - 252):]
            daily_rets  = [window[i] / window[i - 1] - 1 for i in range(1, len(window))]
            vol_annual  = float(np.std(daily_rets)) * (252 ** 0.5) if len(daily_rets) >= 20 else 0.20
            low_vol_raw = 1.0 / (1.0 + vol_annual)   # higher = less volatile = better

            # ── Return metrics ─────────────────────────────────────────────
            ret_6m = (closes[-1] / closes[max(0, n - 126)] - 1) if n >= 30 else 0.0
            ret_1y = (closes[-1] / closes[0] - 1)               if closes[0] else 0.0
            ret_3m = (closes[-1] / closes[max(0, n - 63)]  - 1)
            ret_1m = (closes[-1] / closes[max(0, n - 20)]  - 1)

            # ── Fundamentals ───────────────────────────────────────────────
            fund    = get_fundamentals_sync(sym)
            f_score = (
                fund.get("value_score", 0)
                + fund.get("growth_score", 0)
                + fund.get("quality_score", 0)
            ) / 3.0

            # ── Stored model signal (no live training) ─────────────────────
            model_sig        = 0.5
            model_confidence = 0.0
            model_signal_str = "N/A"
            p_buy_snap, p_hold_snap, p_sell_snap = 0.0, 1.0, 0.0
            stored = _db.load_model(sym, STRATEGY_ID)
            if stored is not None:
                try:
                    blob, meta = stored
                    m   = pickle.loads(blob)
                    df  = _load_bars_df(sym, days=LOOKBACK_DAYS)
                    bl  = _df_to_bars(sym, df) if not df.empty else []
                    fdf = bars_to_features(bl) if bl else pd.DataFrame()
                    sig = _predict_signal(m, fdf, meta["feature_cols"])
                    sig_map          = {"BUY": 1.0, "HOLD": 0.5, "SELL": 0.0}
                    model_sig        = sig_map[sig["signal"]]
                    model_confidence = sig["confidence"]
                    model_signal_str = sig["signal"]
                    p_buy_snap       = sig["p_buy"]
                    p_hold_snap      = sig["p_hold"]
                    p_sell_snap      = sig["p_sell"]
                except Exception:
                    pass

            return {
                "sym":             sym,
                "yf_ticker":       entry["yf_ticker"],
                "name":            entry["name"],
                "sector":          entry["sector"],
                "closes":          closes,
                "bars_list":       bars_list,
                "ret_20d":         ret_20d,
                "price_52w_pct":   price_52w_pct,
                "dist_52w_high":   dist_52w_high,
                "low_vol_raw":     low_vol_raw,
                "ret_6m":          ret_6m,
                "ret_1y":          ret_1y,
                "ret_3m":          ret_3m,
                "ret_1m":          ret_1m,
                "f_score":         f_score,
                "fund":            fund,
                "model_sig":       model_sig,
                "model_confidence":model_confidence,
                "model_signal_str":model_signal_str,
                "p_buy_snap":      p_buy_snap,
                "p_hold_snap":     p_hold_snap,
                "p_sell_snap":     p_sell_snap,
                "stored":          stored,
            }
        except Exception:
            return None

    raw_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_one, e): e for e in universe}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                raw_rows.append(result)

    # ── Hard filters: block hot stocks & near-peak stocks (mirrors screener) ───
    hot_excluded: list[str] = []
    filtered: list[dict] = []
    for r in raw_rows:
        if r["ret_20d"] > 0.25:
            hot_excluded.append(f"{r['sym']} (20d +{r['ret_20d']:.0%})")
            continue
        if r["price_52w_pct"] > 0.90:
            hot_excluded.append(f"{r['sym']} (52w pos {r['price_52w_pct']:.0%})")
            continue
        filtered.append(r)

    if hot_excluded:
        _log.info(
            "Recommendations: hot-stock filter excluded %d: %s",
            len(hot_excluded), ", ".join(hot_excluded[:8]),
        )
    if not filtered:
        filtered = raw_rows  # safety fallback — score everything if all excluded
    if not filtered:
        return []  # nothing to score (e.g. unknown sector, or all data failures)

    # ── Cross-sectional min-max normalisation ─────────────────────────────────
    def _minmax(vals: list[float]) -> list[float]:
        lo, hi = min(vals), max(vals)
        return [0.5] * len(vals) if hi == lo else [(v - lo) / (hi - lo) for v in vals]

    if len(filtered) == 1:
        norm_52w_dist = [0.5]
        norm_low_vol  = [0.5]
        norm_6m       = [0.5]
    else:
        norm_52w_dist = _minmax([r["dist_52w_high"]                          for r in filtered])  # further from peak = better
        norm_low_vol  = _minmax([r["low_vol_raw"]                            for r in filtered])  # less volatile = better
        norm_6m       = _minmax([min(max(r["ret_6m"], -0.30), 0.30)         for r in filtered])  # 6M momentum, capped

    # ── Build final scored rows ────────────────────────────────────────────────
    rows: list[dict] = []
    for i, r in enumerate(filtered):
        sym   = r["sym"]
        fund  = r["fund"]
        closes = r["closes"]
        bars_list = r["bars_list"]

        f_norm = min(r["f_score"] / 100.0, 1.0)

        # Quality/contrarian position score (anti-hot-stock):
        #   45% distance from 52w high (contrarian: beaten-down beats hot)
        #   35% low volatility (quality / stability)
        #   20% small 6M momentum (capped, cross-sectionally normalised)
        q_score = (
            0.45 * norm_52w_dist[i]
            + 0.35 * norm_low_vol[i]
            + 0.20 * norm_6m[i]
        )

        composite = 0.40 * f_norm + 0.30 * r["model_sig"] + 0.30 * q_score

        ret_1y = r["ret_1y"]
        ret_3m = r["ret_3m"]
        ret_1m = r["ret_1m"]

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

        causal_nodes = [
            {
                "factor":       "fundamentals",
                "label":        "Fundamentals",
                "description":  (
                    f"Value {fund.get('value_score', 0):.0f} · "
                    f"Growth {fund.get('growth_score', 0):.0f} · "
                    f"Quality {fund.get('quality_score', 0):.0f}  "
                    f"(composite {r['f_score']:.1f}/100)"
                ),
                "raw_value":    round(r["f_score"], 2),
                "norm_value":   round(f_norm, 4),
                "weight":       0.40,
                "contribution": round(0.40 * f_norm, 4),
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
                "factor":       "model_signal",
                "label":        "ML Signal (LightGBM)",
                "description":  (
                    f"Signal: {r['model_signal_str']} with {r['model_confidence']:.0%} confidence"
                    if r["model_signal_str"] != "N/A"
                    else "No trained model — using neutral baseline (0.5)"
                ),
                "raw_value":    r["model_signal_str"],
                "norm_value":   round(r["model_sig"], 4),
                "weight":       0.30,
                "contribution": round(0.30 * r["model_sig"], 4),
                "direction":    "positive" if r["model_sig"] >= 0.7 else "negative" if r["model_sig"] < 0.3 else "neutral",
                "percentile":   "",
                "extras": {
                    "signal":        r["model_signal_str"],
                    "confidence":    round(r["model_confidence"], 4),
                    "model_trained": r["stored"] is not None,
                    "p_buy":         r["p_buy_snap"],
                    "p_hold":        r["p_hold_snap"],
                    "p_sell":        r["p_sell_snap"],
                },
            },
            {
                "factor":       "quality_position",
                "label":        "Quality Positioning",
                "description":  (
                    f"Dist from 52w high: {r['dist_52w_high']:.0%} · "
                    f"6M return: {r['ret_6m']:+.1%} · "
                    f"52w position: {r['price_52w_pct']:.0%}"
                ),
                "raw_value":    round(q_score, 4),
                "norm_value":   round(q_score, 4),
                "weight":       0.30,
                "contribution": round(0.30 * q_score, 4),
                "direction":    "positive" if q_score >= 0.55 else "negative" if q_score < 0.35 else "neutral",
                "percentile":   "",
                "extras": {
                    "dist_52w_high": round(r["dist_52w_high"], 4),
                    "price_52w_pct": round(r["price_52w_pct"], 4),
                    "ret_6m":        round(r["ret_6m"], 4),
                    "ret_20d":       round(r["ret_20d"], 4),
                    "low_vol_raw":   round(r["low_vol_raw"], 4),
                    "norm_52w_dist": round(norm_52w_dist[i], 4),
                    "norm_low_vol":  round(norm_low_vol[i], 4),
                    "norm_6m":       round(norm_6m[i], 4),
                },
            },
        ]

        rows.append({
            "symbol":           sym,
            "yf_ticker":        r["yf_ticker"],
            "name":             r["name"],
            "sector":           r["sector"],
            "score":            round(composite, 4),
            "model_signal":     r["model_signal_str"],
            "model_confidence": round(r["model_confidence"], 4),
            "model_trained":    r["stored"] is not None,
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

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]

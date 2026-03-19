"""
LightGBM Strategy — gradient-boosted tree classifier trained on historical bars.

Training:
  - warm_bars() is called by the Backtester with N years of pre-period bars.
  - on_start() triggers _train() per symbol: fits a LGBMClassifier with ternary
    labels (+1=BUY, 0=HOLD, -1=SELL) based on forward N-day return.
  - Train/validation split: first `train_ratio` fraction for training,
    remainder for out-of-sample accuracy reporting.

Inference:
  - on_bar() recomputes features for the rolling bar buffer and calls predict().
  - Signals are emitted only when predicted probability exceeds `min_confidence`.
  - Optionally re-trains on a rolling window every `retrain_every_n_bars` bars.

Optional macro features:
  - If a MacroFetcher is injected, macro indicators are appended to every
    feature row so the model can learn regime-conditional patterns.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from myquant.config.logging_config import get_logger
from myquant.models.bar import Bar
from myquant.models.signal import Signal, SignalStrength, SignalType
from myquant.strategy.base import BaseStrategy
from myquant.strategy.ml.feature_engineer import (
    FEATURE_COLS,
    bars_to_features,
    make_labels,
    add_macro_features,
)

logger = get_logger(__name__)

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning(
        "lightgbm not installed — LGBMStrategy disabled. "
        "Run: pip install lightgbm"
    )

# Feature columns used by models that include macro features
_MACRO_EXTRA_COLS = [
    "macro_pmi", "macro_cpi", "macro_usdcny",
    "macro_us10y", "macro_vix", "macro_risk_on", "macro_risk_off",
]


class LGBMStrategy(BaseStrategy):
    """
    Gradient-boosted classifier strategy.

    Parameters
    ----------
    forward_days    : Label horizon — forward N-day return for classification.
    threshold       : Minimum absolute forward return to trigger directional label.
    train_ratio     : Fraction of labeled rows used for training + calibration
                      (rest = OOS validation).  Within this training portion,
                      the last 25% is reserved as the early-stopping / calibration
                      holdout; LightGBM itself trains on the first 75%.
    min_confidence  : Minimum predicted class probability (post-calibration) to emit a signal.
    retrain_every   : Re-train every N on_bar() calls (0 = never re-train after start).
    use_macro       : If True, attempt to inject macro features (requires MacroFetcher).
    calibrate       : If True, wrap the trained LightGBM model in an isotonic-regression
                      calibrator for more reliable probability estimates.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        forward_days: int = 5,
        threshold: float = 0.015,
        train_ratio: float = 0.70,
        min_confidence: float = 0.52,
        retrain_every: int = 63,      # quarterly rolling retrain (0 = disabled)
        max_train_bars: int = 504,    # rolling window: ~2 years of daily bars
        use_macro: bool = False,
        calibrate: bool = True,
        # ── LightGBM hyperparameters ────────────────────────────────────
        num_leaves: int = 63,         # was 31; deeper trees capture more non-linearity
        n_estimators: int = 500,      # upper bound — early stopping cuts this in practice
        learning_rate: float = 0.03,  # slower LR pairs well with more trees
    ) -> None:
        super().__init__(strategy_id, symbols)
        self.forward_days   = forward_days
        self.threshold      = threshold
        self.train_ratio    = train_ratio
        self.min_confidence = min_confidence
        self.retrain_every  = retrain_every
        self.max_train_bars = max_train_bars  # rolling window cap
        self.use_macro      = use_macro
        self.calibrate      = calibrate

        self._lgb_params = dict(
            num_leaves        = num_leaves,
            n_estimators      = n_estimators,
            learning_rate     = learning_rate,
            feature_fraction  = 0.75,    # was 0.80; slight reduction reduces correlation
            bagging_fraction  = 0.75,    # was 0.80
            bagging_freq      = 5,
            min_child_samples = 25,      # was 20; slightly higher for smoother splits
            reg_alpha         = 0.05,    # L1 regularisation (new)
            reg_lambda        = 0.10,    # L2 regularisation (new)
            min_split_gain    = 0.001,   # minimum gain to create a split (new)
            verbose           = -1,
            class_weight      = "balanced",
        )

        # Per-symbol state
        self._models:      dict[str, "lgb.LGBMClassifier"] = {}
        self._cal_models:  dict[str, Any]                  = {}  # calibrated wrappers
        self._is_trained:  dict[str, bool]                 = {}
        self._bar_counter: dict[str, int]                  = {}
        self._feat_cols:   dict[str, list[str]]            = {}  # actual cols used

        # Optional: injected MacroFetcher
        self._macro_fetcher = None

    def set_macro_fetcher(self, fetcher) -> None:
        """Inject a MacroFetcher for macro-augmented features."""
        self._macro_fetcher = fetcher

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def on_start(self) -> None:
        if not _HAS_LGB:
            return
        logger.info("LGBMStrategy [%s]: training on warm data…", self.strategy_id)
        for symbol in self.symbols:
            self._train(symbol)

    # ── Bar handler ───────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        super().on_bar(bar)  # appends to _bar_buffer
        if not _HAS_LGB:
            return None

        symbol = bar.symbol

        # Lazy train if model not ready
        if not self._is_trained.get(symbol):
            bars = list(self._bar_buffer.get(symbol, []))
            if len(bars) >= max(100, self.forward_days + 30):
                self._train(symbol)
            return None

        # Optional periodic re-training
        if self.retrain_every > 0:
            cnt = self._bar_counter.get(symbol, 0) + 1
            self._bar_counter[symbol] = cnt
            if cnt % self.retrain_every == 0:
                logger.debug("LGBMStrategy [%s]: re-training %s", self.strategy_id, symbol)
                self._train(symbol)

        return self._predict_signal(symbol, bar)

    # ── Training ──────────────────────────────────────────────────────────

    def _train(self, symbol: str) -> None:
        if not _HAS_LGB:
            return

        bars = list(self._bar_buffer.get(symbol, []))

        # Walk-forward: train only on the most recent rolling window.
        # This prevents the model from being anchored to stale market regimes
        # and reduces the risk of overfitting to the distant past.
        if self.max_train_bars > 0 and len(bars) > self.max_train_bars:
            bars = bars[-self.max_train_bars:]

        if len(bars) < max(100, self.forward_days * 5 + 30):
            logger.debug(
                "LGBMStrategy: not enough bars for %s (%d)", symbol, len(bars)
            )
            return

        df = bars_to_features(bars)
        if df.empty:
            return

        # Optionally enrich with macro
        feat_cols = list(FEATURE_COLS)
        if self.use_macro and self._macro_fetcher is not None:
            try:
                snap = self._macro_fetcher.fetch()
                df   = add_macro_features(df, snap)
                feat_cols = feat_cols + _MACRO_EXTRA_COLS
            except Exception as e:
                logger.debug("LGBMStrategy: macro enrichment failed: %s", e)

        labels = make_labels(df, self.forward_days, self.threshold)
        aligned = df.join(labels.rename("label")).dropna()

        if len(aligned) < 60:
            logger.debug("LGBMStrategy: too few labeled rows for %s (%d)", symbol, len(aligned))
            return

        # ── 3-way temporal split ──────────────────────────────────
        # │ 60% LGBM train │ 15% early-stop + calib │ 25% OOS accuracy │
        # └─────────────────────────────────────────────────────────┘
        n = len(aligned)
        n_lgbm  = int(n * 0.60)            # pure training rows
        n_calib = int(n * 0.15)            # early-stopping + calibration rows

        if n_lgbm < 40 or n_calib < 10:
            # Fall back to simple 70 / 30 split when data is scarce
            n_train = int(n * self.train_ratio)
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

        model = lgb.LGBMClassifier(**self._lgb_params)

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

        # ── Optional probability calibration ──────────────────────────
        # Sigmoid (Platt scaling) maps raw LGBM probas through a 2-param
        # sigmoid fit on the calibration holdout — produces more reliable
        # confidence scores for use as the trading threshold gate.
        cal_model = model  # fallback: uncalibrated
        if self.calibrate and use_early_stop and len(X_val) >= 20:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                calibrator = CalibratedClassifierCV(
                    model, method="sigmoid", cv="prefit"
                )
                calibrator.fit(X_val, y_val)
                cal_model = calibrator
            except Exception as e:
                logger.debug("LGBMStrategy: calibration failed (%s) — using raw probas", e)

        self._models[symbol]     = model       # raw model (for feature_importances_)
        self._cal_models[symbol] = cal_model   # calibrated model (for predict_proba)
        self._is_trained[symbol] = True
        self._feat_cols[symbol]  = feat_cols

        # OOS accuracy
        if len(X_oos) > 0:
            acc  = (cal_model.predict(X_oos) == y_oos.values).mean()
            dist = dict(zip(*np.unique(y_train.values, return_counts=True)))
            logger.info(
                "LGBMStrategy [%s] %s | lgbm_train=%d calib=%d OOS=%d acc=%.1f%% | "
                "label dist SELL:%d HOLD:%d BUY:%d",
                self.strategy_id, symbol,
                len(X_train), len(X_val), len(X_oos), acc * 100,
                dist.get(0, 0), dist.get(1, 0), dist.get(2, 0),
            )

    # ── Inference ─────────────────────────────────────────────────────────

    def _predict_signal(self, symbol: str, bar: Bar) -> Optional[Signal]:
        model = self._models.get(symbol)
        if model is None:
            return None

        bars = list(self._bar_buffer.get(symbol, []))
        df   = bars_to_features(bars)
        if df.empty:
            return None

        feat_cols = self._feat_cols.get(symbol, FEATURE_COLS)

        if self.use_macro and self._macro_fetcher is not None:
            try:
                snap = self._macro_fetcher.fetch()
                df   = add_macro_features(df, snap)
            except Exception:
                pass

        if not all(c in df.columns for c in feat_cols):
            return None

        latest = df.iloc[[-1]][feat_cols]

        # Use calibrated model for probabilities if available
        infer_model = self._cal_models.get(symbol, model)
        proba  = infer_model.predict_proba(latest)[0]  # [p_sell, p_hold, p_buy]
        pred   = int(infer_model.predict(latest)[0])   # 0=SELL, 1=HOLD, 2=BUY
        conf   = float(proba[pred])

        if conf < self.min_confidence:
            return None

        if pred == 2:  # BUY
            strength = SignalStrength.STRONG if conf > 0.70 else SignalStrength.NORMAL
            return self.make_signal(
                symbol, SignalType.BUY, bar.close,
                strength=strength,
                metadata={
                    "confidence": round(conf, 4),
                    "p_buy":  round(float(proba[2]), 4),
                    "p_hold": round(float(proba[1]), 4),
                    "p_sell": round(float(proba[0]), 4),
                    "model":  "lgbm",
                },
            )

        if pred == 0:  # SELL
            strength = SignalStrength.STRONG if conf > 0.70 else SignalStrength.NORMAL
            return self.make_signal(
                symbol, SignalType.SELL, bar.close,
                strength=strength,
                metadata={
                    "confidence": round(conf, 4),
                    "p_buy":  round(float(proba[2]), 4),
                    "p_hold": round(float(proba[1]), 4),
                    "p_sell": round(float(proba[0]), 4),
                    "model":  "lgbm",
                },
            )

        return None  # HOLD

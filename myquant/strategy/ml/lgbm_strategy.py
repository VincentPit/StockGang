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
    make_labels_rl,
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
    forward_days       : Label horizon — forward N-day return for classification.
    threshold          : Minimum absolute forward return to trigger directional label.
    train_ratio        : Fraction of labeled rows used for training + calibration
                         (rest = OOS validation).
    min_confidence     : Minimum predicted class probability to emit a signal.
    retrain_every      : Re-train every N on_bar() calls (0 = never re-train after start).
    n_ensemble_windows : Number of walk-forward temporal windows for ensemble training.
                         Each sub-model is trained on a different rolling window slice:
                         window 0 = full, window 1 = 75%, window 2 = 50%, …
                         Predicted probabilities are averaged across all sub-models.
    use_macro          : If True, attempt to inject macro features (requires MacroFetcher).
    calibrate          : If True, apply sigmoid (Platt) calibration to each sub-model.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        forward_days: int = 5,
        threshold: float = 0.015,
        train_ratio: float = 0.70,
        min_confidence: float = 0.52,
        retrain_every: int = 63,         # quarterly rolling retrain (0 = disabled)
        max_train_bars: int = 504,       # rolling window: ~2 years of daily bars
        n_ensemble_windows: int = 3,     # walk-forward ensemble: K temporal slices
        use_macro: bool = False,
        calibrate: bool = True,
        min_hold_bars: int = 5,          # bars of cooldown between consecutive signals
        commission_rate: float = 0.0003, # one-way commission — raises label threshold
        # ── LightGBM hyperparameters ────────────────────────────────────
        num_leaves: int = 63,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
    ) -> None:
        super().__init__(strategy_id, symbols)
        self.forward_days        = forward_days
        self.threshold           = threshold
        self.train_ratio         = train_ratio
        self.min_confidence      = min_confidence
        self.retrain_every       = retrain_every
        self.max_train_bars      = max_train_bars
        self.n_ensemble_windows  = max(1, n_ensemble_windows)
        self.use_macro           = use_macro
        self.calibrate           = calibrate
        self.min_hold_bars       = max(0, min_hold_bars)
        self.commission_rate     = commission_rate

        self._lgb_params = dict(
            num_leaves        = num_leaves,
            n_estimators      = n_estimators,
            learning_rate     = learning_rate,
            feature_fraction  = 0.75,
            bagging_fraction  = 0.75,
            bagging_freq      = 5,
            min_child_samples = 25,
            reg_alpha         = 0.05,
            reg_lambda        = 0.10,
            min_split_gain    = 0.001,
            verbose           = -1,
            class_weight      = "balanced",
        )

        # Per-symbol state
        self._models:      dict[str, Any]        = {}   # raw model for feature_importances_
        self._ensemble:    dict[str, list[Any]]  = {}   # K calibrated models per symbol
        self._is_trained:  dict[str, bool]       = {}
        self._bar_counter: dict[str, int]        = {}
        self._feat_cols:   dict[str, list[str]]  = {}

        # Cooldown tracking: last bar index on which a signal was emitted per symbol
        self._bar_index:        dict[str, int] = {}   # global bar counter per symbol
        self._last_signal_bar:  dict[str, int] = {}   # bar index when last signal fired

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

        # Increment global bar counter for this symbol
        idx = self._bar_index.get(symbol, 0) + 1
        self._bar_index[symbol] = idx

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

        # Cooldown: do not emit a signal within min_hold_bars bars of the last one
        if self.min_hold_bars > 0:
            last = self._last_signal_bar.get(symbol, -9999)
            if (idx - last) < self.min_hold_bars:
                return None

        sig = self._predict_signal(symbol, bar)
        if sig is not None:
            self._last_signal_bar[symbol] = idx
        return sig

    # ── Training helpers ──────────────────────────────────────────────────────

    def _train_one_window(
        self,
        aligned: pd.DataFrame,
        feat_cols: list[str],
    ) -> tuple[Any, Any]:
        """
        Fit a single (raw, calibrated) LightGBM model on a pre-labeled slice.

        Parameters
        ----------
        aligned   : DataFrame with feature columns and a ``label`` column (−1/0/+1).
        feat_cols : Feature column names to use as model input.

        Returns
        -------
        (raw_model, cal_model) — cal_model may equal raw_model when calibration skipped.
        Returns (None, None) if the slice is too small to train reliably.
        """
        n       = len(aligned)
        n_lgbm  = int(n * 0.60)
        n_calib = int(n * 0.15)

        if n_lgbm < 40 or n_calib < 10:
            n_train = int(n * self.train_ratio)
            if n_train < 20:
                return None, None
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
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          early_stopping_rounds=50,
                          verbose=False)
        else:
            model.fit(X_train, y_train)

        cal_model = model
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

        return model, cal_model

    # ── Training ──────────────────────────────────────────────────────────

    def _train(self, symbol: str) -> None:
        if not _HAS_LGB:
            return

        all_bars = list(self._bar_buffer.get(symbol, []))
        if self.max_train_bars > 0 and len(all_bars) > self.max_train_bars:
            all_bars = all_bars[-self.max_train_bars:]

        if len(all_bars) < max(100, self.forward_days * 5 + 30):
            logger.debug("LGBMStrategy: not enough bars for %s (%d)", symbol, len(all_bars))
            return

        # ── Feature engineering on the full capped window ────────────────
        df = bars_to_features(all_bars)
        if df.empty:
            return

        feat_cols = list(FEATURE_COLS)
        if self.use_macro and self._macro_fetcher is not None:
            try:
                snap = self._macro_fetcher.fetch()
                df   = add_macro_features(df, snap)
                feat_cols = feat_cols + _MACRO_EXTRA_COLS
            except Exception as e:
                logger.debug("LGBMStrategy: macro enrichment failed: %s", e)

        labels  = make_labels_rl(df, self.forward_days, self.threshold, self.commission_rate)
        aligned = df.join(labels.rename("label")).dropna()

        if len(aligned) < 60:
            logger.debug("LGBMStrategy: too few labeled rows for %s (%d)", symbol, len(aligned))
            return

        # ── Walk-forward ensemble ─────────────────────────────────────────
        # Train K sub-models on K temporal slices of the aligned data.
        # Slice widths decrease from 100% → 50% in equal steps.
        # Final probability = element-wise mean across K calibrated models.
        n = len(aligned)
        window_fracs = [
            max(0.5, 1.0 - i * (0.5 / max(1, self.n_ensemble_windows - 1)))
            for i in range(self.n_ensemble_windows)
        ]
        window_sizes = sorted(
            {max(60, int(n * f)) for f in window_fracs}, reverse=True
        )

        ensemble:          list[Any] = []
        raw_model_primary: Any      = None

        for idx, win_size in enumerate(window_sizes):
            aligned_slice    = aligned.iloc[-win_size:]
            raw_m, cal_m     = self._train_one_window(aligned_slice, feat_cols)
            if cal_m is None:
                continue
            ensemble.append(cal_m)
            if idx == 0:
                raw_model_primary = raw_m

        if not ensemble:
            logger.debug("LGBMStrategy: all sub-models failed for %s", symbol)
            return

        self._models[symbol]     = raw_model_primary or ensemble[0]
        self._ensemble[symbol]   = ensemble
        self._is_trained[symbol] = True
        self._feat_cols[symbol]  = feat_cols

        dist = dict(zip(*np.unique(
            (aligned.iloc[:int(n * 0.60)]["label"] + 1).values, return_counts=True
        )))
        logger.info(
            "LGBMStrategy [%s] %s | ensemble=%d models | windows=%s | "
            "label dist SELL:%d HOLD:%d BUY:%d",
            self.strategy_id, symbol, len(ensemble), window_sizes[:len(ensemble)],
            dist.get(0, 0), dist.get(1, 0), dist.get(2, 0),
        )

    # ── Inference ─────────────────────────────────────────────────────────

    def _predict_signal(self, symbol: str, bar: Bar) -> Optional[Signal]:
        ensemble = self._ensemble.get(symbol)
        if not ensemble:
            return None

        bars = list(self._bar_buffer.get(symbol, []))
        df   = bars_to_features(bars)
        if df.empty:
            return None

        feat_cols = self._feat_cols.get(symbol, list(FEATURE_COLS))

        if self.use_macro and self._macro_fetcher is not None:
            try:
                snap = self._macro_fetcher.fetch()
                df   = add_macro_features(df, snap)
            except Exception:
                pass

        if not all(c in df.columns for c in feat_cols):
            return None

        latest = df.iloc[[-1]][feat_cols]

        # Average predicted probabilities across all K ensemble members.
        # Each member returns shape (1, 3) → [p_sell, p_hold, p_buy].
        all_probas = np.array([m.predict_proba(latest)[0] for m in ensemble])
        avg_proba  = all_probas.mean(axis=0)    # (3,)
        pred       = int(np.argmax(avg_proba))  # 0=SELL  1=HOLD  2=BUY
        conf       = float(avg_proba[pred])

        if conf < self.min_confidence:
            return None

        if pred == 2:  # BUY
            strength = SignalStrength.STRONG if conf > 0.70 else SignalStrength.NORMAL
            return self.make_signal(
                symbol, SignalType.BUY, bar.close,
                strength=strength,
                metadata={
                    "confidence": round(conf, 4),
                    "p_buy":    round(float(avg_proba[2]), 4),
                    "p_hold":   round(float(avg_proba[1]), 4),
                    "p_sell":   round(float(avg_proba[0]), 4),
                    "model":    "lgbm_ensemble",
                    "n_models": len(ensemble),
                },
            )

        if pred == 0:  # SELL
            strength = SignalStrength.STRONG if conf > 0.70 else SignalStrength.NORMAL
            return self.make_signal(
                symbol, SignalType.SELL, bar.close,
                strength=strength,
                metadata={
                    "confidence": round(conf, 4),
                    "p_buy":    round(float(avg_proba[2]), 4),
                    "p_hold":   round(float(avg_proba[1]), 4),
                    "p_sell":   round(float(avg_proba[0]), 4),
                    "model":    "lgbm_ensemble",
                    "n_models": len(ensemble),
                },
            )

        return None  # HOLD

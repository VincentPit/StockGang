"""
Multi-Factor Alpha Model — combines multiple signal sources into a unified score.

This is the core alpha-generation engine. Instead of relying on a single
strategy, it synthesises signals from:
  1. Technical factors (momentum, mean-reversion, trend)
  2. ML model predictions (LightGBM confidence)
  3. Fundamental factors (value, quality, growth)
  4. Sentiment (news, social media)
  5. Regime context (bull/bear/sideways)

The combination uses an adaptive weighting scheme:
  - Base weights from historical factor Sharpe ratios
  - Dynamic weights from regime detection
  - Regularisation to prevent overfitting to recent regimes

Usage:
    alpha_model = MultiFactorAlpha()
    alpha_model.add_factor("momentum", momentum_scores)
    alpha_model.add_factor("ml_signal", lgbm_scores)
    combined = alpha_model.compute_alpha(regime=MarketRegime.BULL)
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from myquant.config.logging_config import get_logger
from myquant.strategy.regime_detector import MarketRegime

logger = get_logger(__name__)


class FactorCategory(str, Enum):
    TECHNICAL = "technical"
    ML_MODEL = "ml_model"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    FLOW = "flow"          # order flow, institutional activity


@dataclass
class FactorScore:
    """A single factor's score for one symbol."""
    symbol: str
    factor_name: str
    category: FactorCategory
    raw_score: float        # original score (any range)
    z_score: float = 0.0   # standardised score (mean=0, std=1)
    percentile: float = 0.5 # rank percentile (0 to 1)
    confidence: float = 1.0 # data quality / freshness


@dataclass
class AlphaScore:
    """Combined alpha score for a symbol."""
    symbol: str
    composite_score: float      # final combined alpha (higher = more attractive)
    composite_confidence: float # combined confidence
    factor_contributions: dict[str, float] = field(default_factory=dict)
    signal: str = "HOLD"  # BUY / SELL / HOLD based on thresholds
    rank: int = 0          # rank among universe (1 = best)

    def summary_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": round(self.composite_score, 4),
            "confidence": round(self.composite_confidence, 3),
            "signal": self.signal,
            "rank": self.rank,
            "factors": {k: round(v, 4) for k, v in self.factor_contributions.items()},
        }


class MultiFactorAlpha:
    """
    Adaptive multi-factor alpha combination engine.

    The weighting scheme:
      1. Start with equal weights across factors.
      2. As each factor's predictive power is observed (via realised P&L),
         increase its weight using exponential moving average of IC (Information Coefficient).
      3. In different regimes, apply regime-specific weight adjustments:
         - BULL: boost momentum, reduce mean-reversion
         - BEAR: boost mean-reversion / defensive, reduce momentum
         - SIDEWAYS: boost mean-reversion, reduce trend-following
    """

    def __init__(
        self,
        ic_window: int = 60,
        ic_ema_alpha: float = 0.10,
        min_weight: float = 0.05,
        max_weight: float = 0.50,
    ) -> None:
        self._ic_window = ic_window
        self._ic_alpha = ic_ema_alpha
        self._min_weight = min_weight
        self._max_weight = max_weight

        # Factor data: {factor_name: {symbol: FactorScore}}
        self._factors: dict[str, dict[str, FactorScore]] = {}
        self._factor_categories: dict[str, FactorCategory] = {}

        # Adaptive weights
        self._base_weights: dict[str, float] = {}
        self._ic_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=ic_window))
        self._smoothed_ic: dict[str, float] = {}

        # Regime weight adjustments
        self._regime_adjustments: dict[MarketRegime, dict[FactorCategory, float]] = {
            MarketRegime.BULL: {
                FactorCategory.TECHNICAL: 1.3,   # momentum works in bull
                FactorCategory.ML_MODEL: 1.2,
                FactorCategory.FUNDAMENTAL: 0.8,
                FactorCategory.SENTIMENT: 1.1,
                FactorCategory.FLOW: 1.2,
            },
            MarketRegime.BEAR: {
                FactorCategory.TECHNICAL: 0.7,   # momentum fails in bear
                FactorCategory.ML_MODEL: 1.0,
                FactorCategory.FUNDAMENTAL: 1.3, # value shines in bear
                FactorCategory.SENTIMENT: 1.2,   # sentiment matters more
                FactorCategory.FLOW: 0.8,
            },
            MarketRegime.SIDEWAYS: {
                FactorCategory.TECHNICAL: 0.9,
                FactorCategory.ML_MODEL: 1.1,
                FactorCategory.FUNDAMENTAL: 1.1,
                FactorCategory.SENTIMENT: 0.9,
                FactorCategory.FLOW: 1.0,
            },
        }

    def add_factor(
        self,
        factor_name: str,
        scores: dict[str, float],
        category: FactorCategory = FactorCategory.TECHNICAL,
        confidence: float = 1.0,
    ) -> None:
        """
        Add or update a factor's scores for the universe.

        Args:
            factor_name: Unique factor identifier.
            scores: symbol → raw score mapping.
            category: Factor type (for regime-dependent weighting).
            confidence: Data quality score (0=stale, 1=fresh).
        """
        if not scores:
            return

        self._factor_categories[factor_name] = category

        # Standardise (z-score) across the universe
        values = list(scores.values())
        mean_v = np.mean(values)
        std_v = np.std(values) if len(values) > 1 else 1.0
        if std_v < 1e-10:
            std_v = 1.0

        sorted_vals = sorted(values)
        factor_scores = {}

        for sym, raw in scores.items():
            z = (raw - mean_v) / std_v
            rank_idx = sorted_vals.index(raw) if raw in sorted_vals else len(sorted_vals) // 2
            pctile = rank_idx / max(len(sorted_vals) - 1, 1)

            factor_scores[sym] = FactorScore(
                symbol=sym,
                factor_name=factor_name,
                category=category,
                raw_score=raw,
                z_score=z,
                percentile=pctile,
                confidence=confidence,
            )

        self._factors[factor_name] = factor_scores

        # Init base weight if new factor
        if factor_name not in self._base_weights:
            n = len(self._factors)
            for f in self._factors:
                self._base_weights[f] = 1.0 / n

    def update_ic(self, factor_name: str, realised_returns: dict[str, float]) -> None:
        """
        Update a factor's Information Coefficient based on realised returns.

        IC = rank_correlation(factor_scores, realised_returns)
        This drives the adaptive weighting: factors with higher IC get more weight.
        """
        factor_scores = self._factors.get(factor_name, {})
        if not factor_scores:
            return

        # Get common symbols
        common = set(factor_scores.keys()) & set(realised_returns.keys())
        if len(common) < 5:
            return

        f_vals = [factor_scores[s].z_score for s in common]
        r_vals = [realised_returns[s] for s in common]

        # Rank correlation (Spearman IC)
        f_ranks = np.argsort(np.argsort(f_vals)).astype(float)
        r_ranks = np.argsort(np.argsort(r_vals)).astype(float)
        n = len(common)
        d_sq = np.sum((f_ranks - r_ranks) ** 2)
        ic = 1 - 6 * d_sq / (n * (n ** 2 - 1)) if n > 1 else 0.0

        self._ic_history[factor_name].append(ic)

        # EMA smoothing
        prev = self._smoothed_ic.get(factor_name, 0.0)
        self._smoothed_ic[factor_name] = self._ic_alpha * ic + (1 - self._ic_alpha) * prev

        # Update base weights proportional to |smoothed IC|
        self._update_weights_from_ic()

    def compute_alpha(
        self,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        buy_threshold: float = 0.5,
        sell_threshold: float = -0.5,
    ) -> list[AlphaScore]:
        """
        Compute combined alpha scores for all symbols in the universe.

        Args:
            regime: Current market regime for weight adjustment.
            buy_threshold: Z-score threshold for BUY signal.
            sell_threshold: Z-score threshold for SELL signal.

        Returns:
            List of AlphaScore objects, sorted by composite_score descending.
        """
        if not self._factors:
            return []

        # Get effective weights (base × regime adjustment)
        effective_weights = self._get_effective_weights(regime)

        # Collect all symbols
        all_symbols: set[str] = set()
        for factor_scores in self._factors.values():
            all_symbols.update(factor_scores.keys())

        results: list[AlphaScore] = []

        for sym in all_symbols:
            composite = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            contributions: dict[str, float] = {}

            for factor_name, factor_scores in self._factors.items():
                if sym not in factor_scores:
                    continue

                fs = factor_scores[sym]
                w = effective_weights.get(factor_name, 0.0)

                # Weight × z-score × confidence
                contribution = w * fs.z_score * fs.confidence
                composite += contribution
                total_weight += w * fs.confidence
                confidence_sum += w * fs.confidence
                contributions[factor_name] = contribution

            # Normalise
            if total_weight > 0:
                composite /= total_weight

            # Signal classification
            signal = "HOLD"
            if composite > buy_threshold:
                signal = "BUY"
            elif composite < sell_threshold:
                signal = "SELL"

            results.append(AlphaScore(
                symbol=sym,
                composite_score=composite,
                composite_confidence=min(1.0, confidence_sum),
                factor_contributions=contributions,
                signal=signal,
            ))

        # Rank
        results.sort(key=lambda a: a.composite_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def get_top_picks(
        self,
        regime: MarketRegime = MarketRegime.UNKNOWN,
        n: int = 10,
    ) -> list[AlphaScore]:
        """Get the top N buy candidates."""
        alphas = self.compute_alpha(regime)
        return [a for a in alphas if a.signal == "BUY"][:n]

    def _update_weights_from_ic(self) -> None:
        """Recompute base weights proportional to smoothed |IC|."""
        if not self._smoothed_ic:
            return

        total_ic = sum(abs(v) for v in self._smoothed_ic.values())
        if total_ic < 1e-10:
            return

        for factor_name in self._factors:
            ic = abs(self._smoothed_ic.get(factor_name, 0.0))
            raw_weight = ic / total_ic
            self._base_weights[factor_name] = np.clip(
                raw_weight, self._min_weight, self._max_weight
            )

        # Renormalise
        total = sum(self._base_weights.values())
        if total > 0:
            for f in self._base_weights:
                self._base_weights[f] /= total

    def _get_effective_weights(self, regime: MarketRegime) -> dict[str, float]:
        """Apply regime adjustments to base weights."""
        adjustments = self._regime_adjustments.get(regime, {})
        effective = {}

        for factor_name, base_w in self._base_weights.items():
            category = self._factor_categories.get(factor_name, FactorCategory.TECHNICAL)
            regime_mult = adjustments.get(category, 1.0)
            effective[factor_name] = base_w * regime_mult

        # Renormalise
        total = sum(effective.values())
        if total > 0:
            for f in effective:
                effective[f] /= total

        return effective

    @property
    def factor_summary(self) -> dict:
        """Summary of current factor weights and ICs."""
        return {
            "factors": {
                name: {
                    "weight": round(self._base_weights.get(name, 0), 3),
                    "category": self._factor_categories.get(name, "unknown").value
                        if hasattr(self._factor_categories.get(name, "unknown"), "value")
                        else str(self._factor_categories.get(name, "unknown")),
                    "ic": round(self._smoothed_ic.get(name, 0), 4),
                    "num_symbols": len(self._factors.get(name, {})),
                }
                for name in self._factors
            }
        }

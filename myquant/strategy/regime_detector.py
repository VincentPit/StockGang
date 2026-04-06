"""
Regime Detector — Hidden Markov Model for market regime identification.

Detects three market regimes:
  1. BULL (risk-on):  trending up, low volatility
  2. BEAR (risk-off): trending down, high volatility
  3. SIDEWAYS:        range-bound, moderate volatility

Used for:
  - Dynamic strategy selection (momentum in BULL, mean-revert in SIDEWAYS)
  - Dynamic risk parameters (tighter stops in BEAR, wider in BULL)
  - Signal confidence scaling (boost in BULL, suppress in BEAR)
  - Position sizing (full size in BULL, half in SIDEWAYS, minimal in BEAR)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from myquant.config.logging_config import get_logger

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeState:
    """Current regime classification with metadata."""
    regime: MarketRegime
    confidence: float          # 0 to 1 — how confident we are in the classification
    volatility_percentile: float  # where current vol sits in history (0=low, 1=high)
    trend_strength: float      # absolute trend signal (0=flat, 1=strong trend)
    regime_duration: int       # how many bars we've been in this regime
    transition_prob: dict[str, float]  # probability of transitioning to each regime

    # Dynamic parameters based on regime
    position_scale: float      # multiplier for position sizing (0.3 to 1.5)
    confidence_threshold: float # minimum model confidence to act on signals
    stop_loss_adjust: float    # multiplier for stop-loss width


class RegimeDetector:
    """
    Multi-signal regime detector using a simplified HMM-like approach.

    Instead of a full HMM (which requires hmmlearn and can be unstable),
    we use a robust feature-based classifier:
      1. Rolling return momentum (20d, 60d)
      2. Realised volatility (20d) percentile vs 252d history
      3. Trend strength (return / vol ratio)
      4. Breadth (fraction of symbols above their MA50)
      5. Volatility regime shift (vol acceleration)

    These features are combined with exponential smoothing to produce
    a regime classification with transition probabilities.
    """

    def __init__(
        self,
        lookback: int = 252,
        vol_window: int = 20,
        momentum_window: int = 60,
        smoothing_alpha: float = 0.15,
    ) -> None:
        self._lookback = lookback
        self._vol_window = vol_window
        self._momentum_window = momentum_window
        self._alpha = smoothing_alpha

        # State
        self._returns_history: deque[float] = deque(maxlen=lookback)
        self._vol_history: deque[float] = deque(maxlen=lookback)
        self._regime_history: deque[MarketRegime] = deque(maxlen=lookback)

        self._current_regime = MarketRegime.UNKNOWN
        self._regime_duration = 0
        self._smoothed_bull_score = 0.5
        self._smoothed_bear_score = 0.5

        # Per-symbol tracking for breadth
        self._symbol_above_ma50: dict[str, bool] = {}
        self._symbol_prices: dict[str, deque[float]] = {}

    @property
    def regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def state(self) -> RegimeState:
        """Get full regime state with dynamic parameters."""
        vol_pctile = self._vol_percentile()
        trend = self._trend_strength()

        # Dynamic parameters
        if self._current_regime == MarketRegime.BULL:
            pos_scale = 1.0 + 0.3 * (1 - vol_pctile)  # up to 1.3x in low-vol bull
            conf_threshold = 0.50
            sl_adjust = 1.2  # wider stops in bull (allow more room)
        elif self._current_regime == MarketRegime.BEAR:
            pos_scale = max(0.3, 0.5 - 0.2 * vol_pctile)  # 0.3-0.5x in bear
            conf_threshold = 0.65  # higher bar to enter
            sl_adjust = 0.7  # tighter stops in bear
        else:  # SIDEWAYS
            pos_scale = 0.6 + 0.2 * trend  # 0.6-0.8x depending on trend
            conf_threshold = 0.55
            sl_adjust = 1.0

        transition = self._transition_probs()

        return RegimeState(
            regime=self._current_regime,
            confidence=self._regime_confidence(),
            volatility_percentile=vol_pctile,
            trend_strength=trend,
            regime_duration=self._regime_duration,
            transition_prob=transition,
            position_scale=pos_scale,
            confidence_threshold=conf_threshold,
            stop_loss_adjust=sl_adjust,
        )

    def update(self, market_return: float, symbol_prices: Optional[dict[str, float]] = None) -> MarketRegime:
        """
        Update regime detection with new data point.

        Args:
            market_return: Daily return of the market index (or portfolio).
            symbol_prices: Optional dict of symbol → current price for breadth.
        """
        self._returns_history.append(market_return)

        # Update symbol breadth tracking
        if symbol_prices:
            for sym, price in symbol_prices.items():
                buf = self._symbol_prices.setdefault(sym, deque(maxlen=50))
                buf.append(price)
                if len(buf) >= 50:
                    ma50 = sum(buf) / len(buf)
                    self._symbol_above_ma50[sym] = price > ma50

        # Compute features
        if len(self._returns_history) < self._vol_window:
            return self._current_regime

        recent_returns = list(self._returns_history)

        # 1. Volatility
        vol_20d = np.std(recent_returns[-self._vol_window:]) * np.sqrt(252)
        self._vol_history.append(vol_20d)

        # 2. Momentum
        momentum_20d = sum(recent_returns[-20:]) if len(recent_returns) >= 20 else 0
        momentum_60d = sum(recent_returns[-60:]) if len(recent_returns) >= 60 else 0

        # 3. Trend strength
        vol = np.std(recent_returns[-20:])
        trend_strength = abs(momentum_20d) / (vol * np.sqrt(20) + 1e-10)

        # 4. Market breadth
        breadth = self._market_breadth()

        # ── Classify regime ──
        bull_score = 0.0
        bear_score = 0.0

        # Momentum signal
        if momentum_20d > 0.02:
            bull_score += 0.3
        elif momentum_20d < -0.02:
            bear_score += 0.3

        if momentum_60d > 0.05:
            bull_score += 0.2
        elif momentum_60d < -0.05:
            bear_score += 0.2

        # Volatility signal
        vol_pctile = self._vol_percentile()
        if vol_pctile < 0.3:
            bull_score += 0.15  # low vol is bullish
        elif vol_pctile > 0.7:
            bear_score += 0.15  # high vol is bearish

        # Trend strength
        if trend_strength > 1.5 and momentum_20d > 0:
            bull_score += 0.2
        elif trend_strength > 1.5 and momentum_20d < 0:
            bear_score += 0.2

        # Breadth
        if breadth > 0.6:
            bull_score += 0.15
        elif breadth < 0.4:
            bear_score += 0.15

        # Exponential smoothing
        self._smoothed_bull_score = (
            self._alpha * bull_score + (1 - self._alpha) * self._smoothed_bull_score
        )
        self._smoothed_bear_score = (
            self._alpha * bear_score + (1 - self._alpha) * self._smoothed_bear_score
        )

        # Classification
        prev_regime = self._current_regime

        if self._smoothed_bull_score > 0.45 and self._smoothed_bull_score > self._smoothed_bear_score * 1.3:
            new_regime = MarketRegime.BULL
        elif self._smoothed_bear_score > 0.45 and self._smoothed_bear_score > self._smoothed_bull_score * 1.3:
            new_regime = MarketRegime.BEAR
        else:
            new_regime = MarketRegime.SIDEWAYS

        # Regime persistence: require 3 consecutive signals to switch
        if new_regime != prev_regime:
            if self._regime_duration < 3:
                # Don't switch yet — could be noise
                new_regime = prev_regime

        if new_regime != prev_regime:
            logger.info(
                "Regime change: %s → %s (bull=%.2f, bear=%.2f, dur=%d)",
                prev_regime.value, new_regime.value,
                self._smoothed_bull_score, self._smoothed_bear_score,
                self._regime_duration,
            )
            self._regime_duration = 0
        else:
            self._regime_duration += 1

        self._current_regime = new_regime
        self._regime_history.append(new_regime)

        return new_regime

    def _vol_percentile(self) -> float:
        """Where current vol sits in history (0=lowest, 1=highest)."""
        if len(self._vol_history) < 2:
            return 0.5
        vols = sorted(self._vol_history)
        current = self._vol_history[-1]
        rank = sum(1 for v in vols if v <= current)
        return rank / len(vols)

    def _trend_strength(self) -> float:
        """Absolute trend strength (0=flat, 1+=strong)."""
        if len(self._returns_history) < 20:
            return 0.0
        ret_20d = sum(list(self._returns_history)[-20:])
        vol_20d = np.std(list(self._returns_history)[-20:])
        return abs(ret_20d) / (vol_20d * np.sqrt(20) + 1e-10)

    def _market_breadth(self) -> float:
        """Fraction of symbols above their MA50."""
        if not self._symbol_above_ma50:
            return 0.5  # neutral default
        above = sum(1 for v in self._symbol_above_ma50.values() if v)
        return above / len(self._symbol_above_ma50)

    def _regime_confidence(self) -> float:
        """Confidence in current regime classification."""
        spread = abs(self._smoothed_bull_score - self._smoothed_bear_score)
        # Higher spread = higher confidence; longer duration = higher confidence
        dur_boost = min(0.2, self._regime_duration * 0.02)
        return min(1.0, spread * 2.0 + dur_boost)

    def _transition_probs(self) -> dict[str, float]:
        """Estimate regime transition probabilities from history."""
        if len(self._regime_history) < 10:
            return {r.value: 0.33 for r in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]}

        transitions: dict[str, int] = {
            MarketRegime.BULL.value: 0,
            MarketRegime.BEAR.value: 0,
            MarketRegime.SIDEWAYS.value: 0,
        }
        history = list(self._regime_history)
        current = self._current_regime

        # Count transitions from current regime
        count = 0
        for i in range(len(history) - 1):
            if history[i] == current:
                transitions[history[i + 1].value] = transitions.get(history[i + 1].value, 0) + 1
                count += 1

        if count == 0:
            return {r.value: 0.33 for r in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]}

        return {k: v / count for k, v in transitions.items()}

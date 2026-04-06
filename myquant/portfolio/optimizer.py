"""
Portfolio Optimizer — Markowitz mean-variance, risk parity, and Kelly criterion.

Provides portfolio-level allocation decisions that go beyond per-signal sizing:
  - Mean-Variance Optimization (MVO): maximise Sharpe-optimal weights
  - Risk Parity: equalise risk contribution from each asset
  - Kelly Criterion: information-theoretic optimal sizing
  - Correlation-aware position limits: reduce size when adding correlated assets
  - Black-Litterman: combine market equilibrium with strategy views (signals)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from myquant.config.logging_config import get_logger

logger = get_logger(__name__)


class OptimizationMethod(str, Enum):
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"
    EQUAL_WEIGHT = "equal_weight"
    MIN_VARIANCE = "min_variance"


@dataclass
class AllocationResult:
    """Recommended portfolio weights."""
    weights: dict[str, float]           # symbol → target weight (0 to 1)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    expected_sharpe: float = 0.0
    method: OptimizationMethod = OptimizationMethod.EQUAL_WEIGHT
    correlation_matrix: Optional[pd.DataFrame] = None
    risk_contributions: dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> dict:
        return {
            "method": self.method.value,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "expected_return": round(self.expected_return, 4),
            "expected_volatility": round(self.expected_volatility, 4),
            "expected_sharpe": round(self.expected_sharpe, 3),
            "risk_contributions": {k: round(v, 4) for k, v in self.risk_contributions.items()},
        }


class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer.

    Usage:
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        result = optimizer.optimize(
            returns_df=historical_returns,  # DatetimeIndex × symbols
            method=OptimizationMethod.RISK_PARITY,
            signal_confidence={"sh600519": 0.7, "sh601318": 0.55},
        )
        # result.weights → {"sh600519": 0.35, "sh601318": 0.25, ...}
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.20,
        min_weight: float = 0.0,
        max_correlation_penalty: float = 0.5,
    ) -> None:
        self._rf = risk_free_rate
        self._max_w = max_weight
        self._min_w = min_weight
        self._corr_penalty = max_correlation_penalty

    def optimize(
        self,
        returns_df: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
        signal_confidence: Optional[dict[str, float]] = None,
        target_volatility: float = 0.15,
    ) -> AllocationResult:
        """
        Compute optimal portfolio weights.

        Args:
            returns_df: Daily returns DataFrame, columns = symbols.
            method: Optimisation method.
            signal_confidence: Per-symbol model confidence for view-tilting.
            target_volatility: Target annualised portfolio vol for vol-targeting.
        """
        symbols = list(returns_df.columns)
        n = len(symbols)

        if n == 0:
            return AllocationResult(weights={})
        if n == 1:
            return AllocationResult(weights={symbols[0]: 1.0}, method=method)

        # Clean data
        returns_clean = returns_df.dropna(how="any")
        if len(returns_clean) < 30:
            # Fall back to equal weight
            w = 1.0 / n
            return AllocationResult(
                weights={s: w for s in symbols},
                method=OptimizationMethod.EQUAL_WEIGHT,
            )

        mu = returns_clean.mean().values * 252          # annualised expected returns
        cov = returns_clean.cov().values * 252           # annualised covariance
        corr = returns_clean.corr()

        # Apply confidence tilting to expected returns
        if signal_confidence:
            for i, sym in enumerate(symbols):
                conf = signal_confidence.get(sym, 0.5)
                # Scale expected return by confidence (center at 0.5)
                mu[i] *= max(0.3, min(2.0, 2.0 * conf))

        # Choose optimisation method
        if method == OptimizationMethod.MEAN_VARIANCE:
            raw_weights = self._mean_variance(mu, cov, n)
        elif method == OptimizationMethod.RISK_PARITY:
            raw_weights = self._risk_parity(cov, n)
        elif method == OptimizationMethod.KELLY:
            raw_weights = self._kelly(mu, cov, n)
        elif method == OptimizationMethod.MIN_VARIANCE:
            raw_weights = self._min_variance(cov, n)
        else:
            raw_weights = np.ones(n) / n

        # Apply constraints
        weights = self._apply_constraints(raw_weights, symbols, corr)

        # Vol targeting: scale weights so portfolio vol ≈ target
        port_vol = np.sqrt(weights @ cov @ weights)
        if port_vol > 0:
            scale = target_volatility / port_vol
            weights = weights * min(scale, 2.0)  # cap at 2x leverage
            weights = np.minimum(weights, self._max_w)
            # Re-normalise if sum > 1 (no leverage)
            if weights.sum() > 1.0:
                weights /= weights.sum()

        # Compute expected portfolio stats
        exp_ret = float(weights @ mu)
        exp_vol = float(np.sqrt(weights @ cov @ weights))
        exp_sharpe = (exp_ret - self._rf) / exp_vol if exp_vol > 0 else 0.0

        # Risk contributions
        risk_contribs = self._risk_contribution(weights, cov, symbols)

        weight_dict = {sym: float(w) for sym, w in zip(symbols, weights)}

        return AllocationResult(
            weights=weight_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=exp_sharpe,
            method=method,
            correlation_matrix=corr,
            risk_contributions=risk_contribs,
        )

    def correlation_penalty(
        self, symbol: str, existing_symbols: list[str], corr_matrix: pd.DataFrame
    ) -> float:
        """
        Compute a sizing penalty for a new position based on correlation with existing.

        Returns a multiplier in [1 - max_penalty, 1.0].
        High correlation → smaller position.
        """
        if not existing_symbols or symbol not in corr_matrix.index:
            return 1.0

        valid = [s for s in existing_symbols if s in corr_matrix.index]
        if not valid:
            return 1.0

        avg_corr = float(corr_matrix.loc[symbol, valid].abs().mean())
        penalty = 1.0 - self._corr_penalty * avg_corr
        return max(1.0 - self._corr_penalty, min(1.0, penalty))

    # ── Optimisation methods ──

    def _mean_variance(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Maximum Sharpe ratio portfolio (analytical tangent portfolio)."""
        try:
            cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
            rf_excess = mu - self._rf
            raw = cov_inv @ rf_excess
            # Make all weights non-negative (long-only)
            raw = np.maximum(raw, 0)
            total = raw.sum()
            return raw / total if total > 0 else np.ones(n) / n
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def _risk_parity(self, cov: np.ndarray, n: int) -> np.ndarray:
        """
        Risk parity: equal risk contribution from each asset.
        Uses Newton-Raphson iterative approach.
        """
        # Initial equal weight
        w = np.ones(n) / n
        target_risk = 1.0 / n

        for _ in range(100):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                break
            # Marginal risk contribution
            mrc = (cov @ w) / port_vol
            # Risk contribution
            rc = w * mrc
            rc_pct = rc / rc.sum() if rc.sum() > 0 else np.ones(n) / n

            # Gradient step
            gradient = rc_pct - target_risk
            w -= 0.1 * gradient * w
            w = np.maximum(w, 1e-6)
            w /= w.sum()

        return w

    def _kelly(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """
        Kelly criterion: f* = Σ^{-1} × (μ - rf).
        Scaled to half-Kelly for safety (reduce variance of outcomes).
        """
        try:
            cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
            kelly_full = cov_inv @ (mu - self._rf)
            kelly_half = kelly_full * 0.5  # Half-Kelly
            kelly_half = np.maximum(kelly_half, 0)  # Long-only
            total = kelly_half.sum()
            return kelly_half / total if total > 0 else np.ones(n) / n
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def _min_variance(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Global minimum variance portfolio."""
        try:
            cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
            ones = np.ones(n)
            w = cov_inv @ ones
            w = np.maximum(w, 0)
            total = w.sum()
            return w / total if total > 0 else np.ones(n) / n
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def _apply_constraints(
        self, weights: np.ndarray, symbols: list[str], corr: pd.DataFrame
    ) -> np.ndarray:
        """Apply position limits and correlation penalties."""
        n = len(weights)
        w = weights.copy()

        # Cap individual weights
        w = np.minimum(w, self._max_w)
        w = np.maximum(w, self._min_w)

        # Apply correlation penalty: if two assets are highly correlated,
        # reduce the smaller one's weight
        for i in range(n):
            for j in range(i + 1, n):
                c = abs(float(corr.iloc[i, j]))
                if c > 0.7:
                    penalty = 1.0 - self._corr_penalty * (c - 0.7) / 0.3
                    # Reduce the one with lower weight
                    if w[i] < w[j]:
                        w[i] *= max(0.3, penalty)
                    else:
                        w[j] *= max(0.3, penalty)

        # Re-normalise
        total = w.sum()
        if total > 0:
            w /= total

        return w

    def _risk_contribution(
        self, weights: np.ndarray, cov: np.ndarray, symbols: list[str]
    ) -> dict[str, float]:
        """Compute each asset's risk contribution to portfolio variance."""
        port_vol = np.sqrt(weights @ cov @ weights)
        if port_vol < 1e-12:
            return {s: 1.0 / len(symbols) for s in symbols}

        mrc = (cov @ weights) / port_vol
        rc = weights * mrc
        total_rc = rc.sum()
        if total_rc > 0:
            rc_pct = rc / total_rc
        else:
            rc_pct = np.ones(len(symbols)) / len(symbols)

        return {sym: float(rc_pct[i]) for i, sym in enumerate(symbols)}

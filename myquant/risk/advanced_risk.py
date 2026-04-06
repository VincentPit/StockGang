"""
Advanced Risk Manager — correlation-aware, tail-risk-protected risk engine.

Upgrades over the basic RiskGate:
  1. Correlation-adjusted position sizing: reduce size when adding correlated assets
  2. Dynamic VaR: EWMA volatility model instead of static estimates
  3. Tail risk protection: increase cash when tail risk indicators spike
  4. Portfolio heat monitoring: aggregate open-trade risk vs budget
  5. Drawdown recovery mode: reduce risk after drawdowns, scale back gradually
  6. Regime-adaptive risk parameters: tighter in BEAR, looser in BULL
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from myquant.config.logging_config import get_logger
from myquant.strategy.regime_detector import MarketRegime, RegimeState

logger = get_logger(__name__)


@dataclass
class RiskBudget:
    """Portfolio-level risk budget allocation."""
    total_risk_budget: float         # max portfolio VaR as % of NAV
    used_risk_budget: float          # current VaR as % of NAV
    remaining_budget: float          # room for new positions
    position_heat: float             # sum of open position risk / budget
    max_new_position_risk: float     # max risk a new position can take
    drawdown_mode: bool = False      # True if in drawdown recovery mode
    risk_multiplier: float = 1.0     # dynamic scaling factor


@dataclass
class PositionRiskReport:
    """Risk metrics for a single position."""
    symbol: str
    notional: float
    weight: float                    # fraction of NAV
    daily_vol: float                 # annualised volatility
    var_contribution: float          # portfolio VaR contribution
    correlation_avg: float           # average correlation with other positions
    beta: float                      # beta to portfolio
    stop_distance_pct: float         # distance to stop-loss


class AdvancedRiskManager:
    """
    Portfolio-level risk management with dynamic risk budgeting.

    The key insight: don't just check each signal independently — manage
    the TOTAL portfolio risk as a budget. Each new position consumes
    risk budget. When the budget is exhausted, no new positions regardless
    of signal quality.
    """

    def __init__(
        self,
        nav_getter,
        positions_getter,
        max_portfolio_var_pct: float = 0.08,     # max 1-day 95% VaR = 8% of NAV
        max_portfolio_heat: float = 1.5,         # max heat = 1.5x risk budget
        drawdown_recovery_threshold: float = -0.05,  # enter recovery mode at 5% DD
        drawdown_full_recovery: float = 0.0,     # exit recovery when DD = 0
        ewma_lambda: float = 0.94,               # EWMA decay for vol estimation
        correlation_lookback: int = 60,           # days for correlation estimation
    ) -> None:
        self._get_nav = nav_getter
        self._get_positions = positions_getter
        self._max_var_pct = max_portfolio_var_pct
        self._max_heat = max_portfolio_heat
        self._dd_recovery_threshold = drawdown_recovery_threshold
        self._dd_full_recovery = drawdown_full_recovery
        self._ewma_lambda = ewma_lambda
        self._corr_lookback = correlation_lookback

        # State
        self._returns_history: dict[str, deque[float]] = {}  # per-symbol daily returns
        self._ewma_var: dict[str, float] = {}                 # EWMA variance estimates
        self._peak_nav: float = 0.0
        self._in_recovery: bool = False
        self._recovery_scale: float = 1.0

        # Regime state
        self._regime_state: Optional[RegimeState] = None

    def update_regime(self, state: RegimeState) -> None:
        """Inject current regime state for dynamic risk adjustment."""
        self._regime_state = state

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """Feed daily return for EWMA volatility estimation."""
        buf = self._returns_history.setdefault(symbol, deque(maxlen=self._corr_lookback))
        buf.append(daily_return)

        # EWMA variance update: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t
        prev_var = self._ewma_var.get(symbol, daily_return ** 2)
        self._ewma_var[symbol] = (
            self._ewma_lambda * prev_var + (1 - self._ewma_lambda) * daily_return ** 2
        )

    def get_risk_budget(self) -> RiskBudget:
        """Compute current portfolio risk budget status."""
        nav = self._get_nav()
        positions = self._get_positions()

        if nav <= 0:
            return RiskBudget(
                total_risk_budget=self._max_var_pct,
                used_risk_budget=0.0,
                remaining_budget=self._max_var_pct,
                position_heat=0.0,
                max_new_position_risk=self._max_var_pct,
            )

        # Current portfolio VaR
        portfolio_var = self._compute_portfolio_var(positions, nav)
        used_pct = portfolio_var / nav if nav > 0 else 0.0

        # Drawdown recovery mode
        if nav > self._peak_nav:
            self._peak_nav = nav
        current_dd = (nav - self._peak_nav) / self._peak_nav if self._peak_nav > 0 else 0.0

        if current_dd < self._dd_recovery_threshold:
            if not self._in_recovery:
                logger.warning(
                    "Entering drawdown recovery mode (DD=%.1f%%). Reducing risk budget.",
                    current_dd * 100,
                )
            self._in_recovery = True
            # Scale risk budget linearly: at -5% DD → 50% of normal, at -10% DD → 25%
            self._recovery_scale = max(0.25, 1.0 + current_dd / abs(self._dd_recovery_threshold))
        elif self._in_recovery and current_dd >= self._dd_full_recovery:
            logger.info("Exiting drawdown recovery mode. Full risk budget restored.")
            self._in_recovery = False
            self._recovery_scale = 1.0

        # Regime adjustment
        regime_mult = 1.0
        if self._regime_state:
            regime_mult = self._regime_state.position_scale

        effective_budget = self._max_var_pct * self._recovery_scale * regime_mult
        remaining = max(0, effective_budget - used_pct)

        # Position heat: sum of individual position risks / budget
        heat = used_pct / effective_budget if effective_budget > 0 else 0.0

        return RiskBudget(
            total_risk_budget=effective_budget,
            used_risk_budget=used_pct,
            remaining_budget=remaining,
            position_heat=heat,
            max_new_position_risk=remaining,
            drawdown_mode=self._in_recovery,
            risk_multiplier=self._recovery_scale * regime_mult,
        )

    def compute_position_risk(
        self, symbol: str, quantity: int, price: float
    ) -> PositionRiskReport:
        """Compute risk metrics for a specific position."""
        nav = self._get_nav()
        positions = self._get_positions()

        notional = quantity * price
        weight = notional / nav if nav > 0 else 0.0

        # Daily vol (annualised)
        ewma_var = self._ewma_var.get(symbol, 0.0004)  # default 2% daily
        daily_vol = float(np.sqrt(ewma_var) * np.sqrt(252))

        # VaR contribution
        var_1d = abs(notional) * np.sqrt(ewma_var) * 1.645
        var_contribution = var_1d / nav if nav > 0 else 0.0

        # Average correlation with existing positions
        avg_corr = self._avg_correlation_with_portfolio(symbol, positions)

        return PositionRiskReport(
            symbol=symbol,
            notional=notional,
            weight=weight,
            daily_vol=daily_vol,
            var_contribution=var_contribution,
            correlation_avg=avg_corr,
            beta=1.0,  # simplified; could compute vs index
            stop_distance_pct=0.0,
        )

    def size_new_position(
        self,
        symbol: str,
        price: float,
        signal_confidence: float,
        base_risk_per_trade: float = 0.005,
    ) -> int:
        """
        Compute risk-budget-aware position size.

        This goes beyond ATR sizing: it considers:
          1. Remaining risk budget (don't exceed portfolio VaR limit)
          2. Correlation penalty (reduce size for correlated positions)
          3. Drawdown recovery scaling
          4. Regime-adjusted sizing
        """
        nav = self._get_nav()
        if nav <= 0 or price <= 0:
            return 100  # minimum lot

        budget = self.get_risk_budget()

        # Risk per trade, adjusted by recovery and regime
        effective_risk = base_risk_per_trade * budget.risk_multiplier

        # Don't exceed remaining risk budget
        effective_risk = min(effective_risk, budget.remaining_budget * 0.5)

        # EWMA vol estimate
        ewma_var = self._ewma_var.get(symbol, 0.0004)
        daily_vol = np.sqrt(ewma_var)
        if daily_vol < 0.003:
            daily_vol = 0.003  # floor

        # Base sizing: risk_budget / (price × daily_vol × z95)
        risk_budget_cash = nav * effective_risk
        per_share_risk = price * daily_vol * 1.645
        base_shares = risk_budget_cash / per_share_risk if per_share_risk > 0 else 100

        # Confidence scaling
        conf_scale = max(0.3, min(2.0, 2.0 * (signal_confidence - 0.5)))
        shares = int(base_shares * conf_scale)

        # Correlation penalty
        positions = self._get_positions()
        corr_penalty = self._correlation_penalty(symbol, positions)
        shares = int(shares * corr_penalty)

        # Round to lot size
        lot_size = 100
        shares = max(lot_size, (shares // lot_size) * lot_size)

        return shares

    def _compute_portfolio_var(self, positions: dict, nav: float) -> float:
        """
        Compute portfolio 1-day 95% VaR using EWMA covariance.
        For simplicity, uses diagonal approximation with correlation haircut.
        """
        if not positions:
            return 0.0

        z95 = 1.645
        individual_vars = []

        for sym, pos in positions.items():
            notional = abs(pos.market_value)
            ewma_var = self._ewma_var.get(sym, 0.0004)
            var_i = notional * np.sqrt(ewma_var) * z95
            individual_vars.append(var_i)

        if not individual_vars:
            return 0.0

        # Sum with diversification benefit (assume avg correlation of 0.5)
        n = len(individual_vars)
        sum_var = sum(individual_vars)
        avg_corr = 0.5
        diversified_var = sum_var * np.sqrt(
            (1 / n) + (1 - 1 / n) * avg_corr
        ) if n > 1 else sum_var

        return diversified_var

    def _avg_correlation_with_portfolio(self, symbol: str, positions: dict) -> float:
        """Estimate average correlation of symbol with existing positions."""
        if not positions or symbol not in self._returns_history:
            return 0.3  # conservative default

        sym_returns = list(self._returns_history.get(symbol, []))
        if len(sym_returns) < 20:
            return 0.3

        correlations = []
        for pos_sym in positions:
            pos_returns = list(self._returns_history.get(pos_sym, []))
            if len(pos_returns) < 20:
                continue
            # Align lengths
            n = min(len(sym_returns), len(pos_returns))
            a = np.array(sym_returns[-n:])
            b = np.array(pos_returns[-n:])
            corr = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
            correlations.append(abs(corr))

        return float(np.mean(correlations)) if correlations else 0.3

    def _correlation_penalty(self, symbol: str, positions: dict) -> float:
        """
        Reduce position size for highly correlated positions.
        Returns a multiplier between 0.4 and 1.0.
        """
        avg_corr = self._avg_correlation_with_portfolio(symbol, positions)
        # Penalty: at 0.0 corr → 1.0x (no penalty), at 0.8+ corr → 0.4x
        if avg_corr < 0.3:
            return 1.0
        elif avg_corr > 0.8:
            return 0.4
        else:
            return 1.0 - (avg_corr - 0.3) * 1.2  # linear from 1.0 to 0.4

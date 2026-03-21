"""
UCB1 Multi-Armed Bandit for adaptive strategy weighting.

Reinforcement Learning framing
-------------------------------
Each trading strategy (LGBM, RSI, MACD, MA-cross) is an "arm" of the bandit.

  Pull   = allow that strategy's signal to enter a trade
  Reward = realized P&L of the resulting closed round-trip

The bandit balances:
  Exploit — favour strategies that have recently been profitable
  Explore — occasionally give underperforming strategies a chance to recover

Algorithm: UCB1 (Upper Confidence Bound)
  score_i  = ema_pnl_i / scale  +  C × sqrt(ln(T) / n_i)
  weight_i = sigmoid(score_i) mapped to [W_MIN, W_MAX]

The EMA decay (alpha ≈ 0.15) means the bandit adapts to regime shifts:
old profitable strategies can fall out of favour when conditions change,
and recovering strategies earn back weight naturally.

Usage in the backtester
-----------------------
1. Instantiate once: ``bandit = StrategyBandit(strategy_ids=[...])``.
2. After each BUY fill, note which strategy opened the position.
3. After each SELL/risk-exit fill, call ``bandit.update(strategy_id, pnl)``.
4. When processing a new signal, multiply confidence by
   ``bandit.get_weight(signal.strategy_id)``.
"""
from __future__ import annotations

import math
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# Weight range: losing strategy still fires at 60% confidence, winner gets 40% boost
_WEIGHT_MIN: float = 0.60
_WEIGHT_MAX: float = 1.40

# Default P&L scale: map ¥5 000 realized gain → score +1.0
# (tune higher for larger portfolio sizes)
_DEFAULT_PNL_SCALE: float = 5_000.0

# UCB1 exploration strength (higher = more exploration)
_DEFAULT_UCB_C: float = 1.0


class StrategyBandit:
    """
    UCB1 multi-armed bandit that adapts strategy confidence weights
    based on realized trading P&L.

    Parameters
    ----------
    strategy_ids : Strategy identifiers to track.
    ema_alpha    : EMA smoothing factor for P&L history (0–1).
                   Smaller = slower to forget old results.
                   Larger  = adapts faster but noisier.
    ucb_c        : UCB1 exploration coefficient.
                   0 = pure exploitation (no exploration bonus).
                   Higher values favour underused strategies.
    pnl_scale    : P&L normalisation scale (¥).
                   Realized P&L is divided by this before entering the score,
                   so the UCB1 exploration term and the exploitation term
                   are on the same numerical scale.
    """

    def __init__(
        self,
        strategy_ids: Sequence[str],
        ema_alpha: float = 0.15,
        ucb_c: float = _DEFAULT_UCB_C,
        pnl_scale: float = _DEFAULT_PNL_SCALE,
    ) -> None:
        self._ids = list(strategy_ids)
        self._alpha = ema_alpha
        self._ucb_c = ucb_c
        self._scale = max(1.0, pnl_scale)

        # Per-strategy running state
        self._ema_pnl:    dict[str, float] = {s: 0.0 for s in self._ids}
        self._n_updates:  dict[str, int]   = {s: 0   for s in self._ids}
        self._total_updates: int = 0

    # ── Reward signal ─────────────────────────────────────────────────────

    def update(self, strategy_id: str, realized_pnl: float) -> None:
        """
        Update the bandit after a closed trade.

        Parameters
        ----------
        strategy_id  : Strategy that *opened* the position.
        realized_pnl : Gross realized P&L of the round trip (¥, before commission).
                       Positive = profit, negative = loss.
        """
        # Auto-register unknown strategies gracefully
        if strategy_id not in self._ema_pnl:
            self._ema_pnl[strategy_id]   = 0.0
            self._n_updates[strategy_id] = 0
            self._ids.append(strategy_id)

        alpha = self._alpha
        self._ema_pnl[strategy_id] = (
            alpha * realized_pnl + (1.0 - alpha) * self._ema_pnl[strategy_id]
        )
        self._n_updates[strategy_id] += 1
        self._total_updates += 1

        logger.debug(
            "Bandit.update [%s] pnl=%.0f ema=%.0f n=%d weight=%.3f",
            strategy_id,
            realized_pnl,
            self._ema_pnl[strategy_id],
            self._n_updates[strategy_id],
            self.get_weight(strategy_id),
        )

    # ── Weight query ──────────────────────────────────────────────────────

    def get_weight(self, strategy_id: str) -> float:
        """
        Return the UCB1 confidence multiplier for a strategy.

        Before any updates (cold start) returns 1.0 for all strategies.
        After updates, profitable strategies get > 1.0, losing ones < 1.0.

        Returns
        -------
        float in [_WEIGHT_MIN, _WEIGHT_MAX].
        """
        if strategy_id not in self._ema_pnl or self._total_updates == 0:
            return 1.0

        T   = max(1, self._total_updates)
        n   = max(1, self._n_updates[strategy_id])

        # Exploitation: normalised EMA P&L (dimensionless, ≈ [-3, +3] range)
        exploit = self._ema_pnl[strategy_id] / self._scale

        # Exploration: UCB1 bonus — rewards under-utilised strategies
        explore = self._ucb_c * math.sqrt(math.log(T) / n)

        # Combined UCB1 score
        ucb_score = exploit + explore

        # Map to [W_MIN, W_MAX] via sigmoid
        # sigmoid(0)  → 0.5 → weight = W_MIN + 0.5*(W_MAX-W_MIN) = 1.0 (neutral)
        # sigmoid(+∞) → 1.0 → weight = W_MAX
        # sigmoid(-∞) → 0.0 → weight = W_MIN
        clamped = max(-500.0, min(500.0, ucb_score))   # prevent exp() overflow
        sig = 1.0 / (1.0 + math.exp(-clamped))
        weight = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * sig
        return float(weight)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def weights_summary(self) -> dict[str, dict]:
        """Return a full diagnostic snapshot for all tracked strategies."""
        out: dict[str, dict] = {}
        for s in self._ids:
            out[s] = {
                "weight":   round(self.get_weight(s), 4),
                "ema_pnl":  round(self._ema_pnl.get(s, 0.0), 1),
                "n_trades": self._n_updates.get(s, 0),
            }
        return out

    def __repr__(self) -> str:
        parts = [f"{s}={self.get_weight(s):.3f}" for s in self._ids]
        return f"StrategyBandit(total_updates={self._total_updates}, {', '.join(parts)})"

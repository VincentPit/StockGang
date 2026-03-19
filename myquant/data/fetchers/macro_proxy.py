"""
HistoricalRegimeDetector — derives market regime purely from replayed price bars.

Unlike MacroFetcher (which calls live economic APIs), this class reads only the
price history that has already been replayed — making it completely free of
look-ahead bias and correct for backtesting.

Regime logic (per symbol, then aggregated by majority vote)
-----------------------------------------------------------
  RISK_ON  : close > MA50 > MA200  AND  realised-vol20 < 65th historical percentile
  RISK_OFF : close < MA200  OR  realised-vol20 > 80th historical percentile
  NEUTRAL  : anything in-between

Signal multiplier
-----------------
  RISK_ON  → 1.20  (allow slightly larger confidence / positions)
  NEUTRAL  → 1.00
  RISK_OFF → 0.70  (shrink confidence; new longs are additionally screened
                    in the simulator's signal loop)

Usage
-----
    detector = HistoricalRegimeDetector()

    for bar in replay_bars:           # feed bars as they arrive
        detector.update(bar.symbol, bar.close)

    print(detector.regime)            # "RISK_ON" / "NEUTRAL" / "RISK_OFF"
    print(detector.signal_multiplier) # 1.20 / 1.00 / 0.70
"""
from __future__ import annotations

import numpy as np

from myquant.config.logging_config import get_logger

logger = get_logger(__name__)


class HistoricalRegimeDetector:
    """
    Maintains a rolling close-price history per tracked symbol and derives
    a consensus portfolio regime on every bar update.

    All computation uses only historical data up to (and including) the current
    bar, so there is no look-ahead contamination.
    """

    def __init__(self, min_bars: int = 50) -> None:
        """
        Parameters
        ----------
        min_bars : Minimum number of bars required before a symbol can vote.
                   Symbols with fewer bars are ignored in the vote.
        """
        self.min_bars = min_bars
        self._closes: dict[str, list[float]] = {}
        self._regime = "NEUTRAL"
        self._multiplier = 1.0

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, symbol: str, close: float) -> None:
        """Feed the latest bar's close price; regime is recomputed immediately."""
        hist = self._closes.setdefault(symbol, [])
        hist.append(close)
        self._recompute()

    @property
    def regime(self) -> str:
        """Current regime string: 'RISK_ON', 'NEUTRAL', or 'RISK_OFF'."""
        return self._regime

    @property
    def signal_multiplier(self) -> float:
        """Confidence multiplier derived from current regime."""
        return self._multiplier

    def is_risk_on(self) -> bool:
        return self._regime == "RISK_ON"

    def is_risk_off(self) -> bool:
        return self._regime == "RISK_OFF"

    # ── Internal ─────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        votes_on  = 0
        votes_off = 0
        n_eligible = 0

        for closes in self._closes.values():
            if len(closes) < self.min_bars:
                continue

            n_eligible += 1
            price = closes[-1]

            # Moving averages
            ma50  = float(np.mean(closes[-50:]))
            ma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else ma50

            # 20-day realised volatility
            rets  = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
            vol20 = float(np.std(rets[-20:])) if len(rets) >= 20 else 0.02

            # Historical vol percentile (sample every 5 bars to keep it fast)
            if len(rets) >= 40:
                vol_samples = [
                    float(np.std(rets[max(0, i - 20):i]))
                    for i in range(20, len(rets), 5)
                ]
                vol_pct = (
                    sum(1 for v in vol_samples if v < vol20)
                    / (len(vol_samples) + 1e-10)
                )
            else:
                vol_pct = 0.5  # assume neutral until enough history

            # Vote
            if price > ma50 > ma200 and vol_pct < 0.65:
                votes_on  += 1
            elif price < ma200 or vol_pct > 0.80:
                votes_off += 1

        if n_eligible == 0:
            return  # keep current regime until enough data accumulates

        frac_on  = votes_on  / n_eligible
        frac_off = votes_off / n_eligible

        prev = self._regime

        if frac_on > 0.50:
            self._regime     = "RISK_ON"
            self._multiplier = 1.20
        elif frac_off > 0.40:
            self._regime     = "RISK_OFF"
            self._multiplier = 0.70
        else:
            self._regime     = "NEUTRAL"
            self._multiplier = 1.00

        if self._regime != prev:
            logger.info(
                "Regime shift: %s → %s  (on=%.0f%% off=%.0f%% of %d symbols)",
                prev, self._regime, frac_on * 100, frac_off * 100, n_eligible,
            )

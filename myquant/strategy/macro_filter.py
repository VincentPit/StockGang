"""
Macro Filter — overlays macroeconomic regime logic onto trading signals.

Layered filtering:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. RISK_OFF regime → suppress all new long entries                  │
  │ 2. Fundamental quality gate → suppress buys on poor-quality stocks  │
  │ 3. Confidence scaling → multiply signal.confidence by regime factor │
  │ 4. Strength upgrade → RISK_ON lifts NORMAL → STRONG                │
  └─────────────────────────────────────────────────────────────────────┘

Usage (standalone):
    from myquant.strategy.macro_filter import MacroFilter
    mf = MacroFilter()
    filtered = mf.filter_signal(signal)   # returns None if suppressed

Usage (inside Backtester via macro_filter= kwarg):
    backtester = Backtester(config, macro_filter=MacroFilter())
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.data.fetchers.fundamental_fetcher import FundamentalFetcher
from myquant.data.fetchers.macro_fetcher import MacroFetcher, MacroSnapshot
from myquant.models.signal import Signal, SignalStrength, SignalType

logger = get_logger(__name__)


class MacroFilter:
    """
    Signal post-processor that applies macro and fundamental overlays.

    Parameters
    ----------
    min_value_score  : Minimum FundamentalSnapshot.value_score to pass a buy.
                       Set to 0 to disable fundamental screening.
    use_fundamentals : Whether to run per-symbol fundamental checks.
    cache_ttl_hours  : TTL for macro data cache (default: 24h).
    """

    def __init__(
        self,
        min_value_score: float = 15.0,
        use_fundamentals: bool = True,
        cache_ttl_hours:  int  = 24,
    ) -> None:
        self._macro_fetcher = MacroFetcher(cache_ttl_hours=cache_ttl_hours)
        self._fund_fetcher  = (
            FundamentalFetcher(cache_ttl_hours=cache_ttl_hours)
            if use_fundamentals
            else None
        )
        self._min_value_score = min_value_score
        self._snapshot: Optional[MacroSnapshot] = None

    # ── Public API ────────────────────────────────────────────────────────

    def refresh(self) -> MacroSnapshot:
        """Force macro data refresh (bypass cache)."""
        self._snapshot = self._macro_fetcher.fetch(use_cache=False)
        return self._snapshot

    @property
    def snapshot(self) -> MacroSnapshot:
        if self._snapshot is None:
            self._snapshot = self._macro_fetcher.fetch()
        return self._snapshot

    def filter_signal(self, signal: Signal) -> Optional[Signal]:
        """
        Apply macro and fundamental overlays to a signal.

        Returns:
            Adjusted Signal, or None if suppressed.
        """
        snap   = self.snapshot
        regime = snap.regime

        is_entry = signal.signal_type in (SignalType.BUY, SignalType.SHORT)
        is_exit  = signal.signal_type in (
            SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT
        )

        # ── Layer 1: regime gate ──────────────────────────────
        if regime == "RISK_OFF" and is_entry:
            logger.info(
                "MacroFilter SUPPRESS %s %s — RISK_OFF (VIX=%.1f PMI=%.1f US10Y=%.1f%%)",
                signal.signal_type.value, signal.symbol,
                snap.vix, snap.china_pmi_mfg, snap.us_10y_yield,
            )
            return None

        # ── Layer 2: fundamental quality gate ────────────────
        if is_entry and self._fund_fetcher is not None:
            fund = self._fund_fetcher.fetch(signal.symbol)
            if fund.value_score < self._min_value_score:
                logger.info(
                    "MacroFilter SUPPRESS %s %s — poor fundamentals "
                    "(value_score=%.1f PE=%.1f PB=%.1f ROE=%.1f%%)",
                    signal.signal_type.value, signal.symbol,
                    fund.value_score, fund.pe_ttm, fund.pb, fund.roe,
                )
                return None

        # ── Layer 3: confidence scaling ───────────────────────
        multiplier = snap.signal_multiplier
        new_conf   = min(1.0, max(0.0, signal.confidence * multiplier))

        # ── Layer 4: strength adjustment ──────────────────────
        if regime == "RISK_OFF":
            new_strength = SignalStrength.WEAK
        elif regime == "RISK_ON" and signal.strength == SignalStrength.NORMAL:
            new_strength = SignalStrength.STRONG
        else:
            new_strength = signal.strength

        filtered = dataclasses.replace(
            signal,
            confidence = new_conf,
            strength   = new_strength,
            metadata   = {
                **signal.metadata,
                "macro_regime":     regime,
                "macro_pmi":        snap.china_pmi_mfg,
                "macro_vix":        snap.vix,
                "macro_us10y":      snap.us_10y_yield,
                "macro_usdcny":     snap.usdcny,
                "conf_multiplier":  multiplier,
            },
        )

        logger.debug(
            "MacroFilter PASS %s %s | regime=%s conf %.2f→%.2f strength=%s",
            signal.signal_type.value, signal.symbol,
            regime, signal.confidence, new_conf, new_strength.value,
        )
        return filtered

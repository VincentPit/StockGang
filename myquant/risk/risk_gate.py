"""
Risk Gate — every signal passes through here before reaching execution.
All checks must pass, or the signal is rejected with a reason.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.config.settings import settings
from myquant.models.order import OrderSide
from myquant.models.position import Position
from myquant.models.signal import Signal, SignalType

logger = get_logger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""
    adjusted_quantity: Optional[int] = None  # Gate may reduce qty instead of rejecting


class RiskGate:
    """
    Multi-layer risk gate. Evaluated in order — first failure rejects the signal.

    Layers:
        1. Market state filter   — no trading during halt / auction / off-hours
        2. Throttle limiter      — max orders per minute
        3. Daily drawdown circuit breaker — new entries halted at daily loss limit
        4. Cooldown timer        — no repeated entry for same symbol within N seconds
        5. Position limit check  — max single position % of NAV
        6. Sector exposure check — max sector % of NAV
        7. VaR check             — 1-day 95% parametric VaR must stay under limit
    """

    def __init__(
        self,
        nav_getter,       # callable: () -> float — returns current NAV
        positions_getter, # callable: () -> dict[str, Position]
        var_limit_pct: float = 0.08,       # max 1-day 95% VaR as fraction of NAV
        default_daily_vol: float = 0.020,  # default daily vol estimate (2% for A-shares)
    ) -> None:
        self._get_nav = nav_getter
        self._get_positions = positions_getter

        # Throttle state
        self._order_timestamps: deque[float] = deque()

        # Cooldown per symbol (last order time)
        self._last_order_time: dict[str, float] = {}
        self.cooldown_seconds: int = 60

        # Daily P&L tracking
        self._daily_start_nav: Optional[float] = None
        self._current_date: Optional[datetime] = None

        # Sector mapping (symbol → sector)
        self._sector_map: dict[str, str] = {}

        # VaR parameters
        self._var_limit_pct: float = var_limit_pct
        self._default_vol:   float = default_daily_vol
        self._vol_estimates: dict[str, float] = {}   # per-symbol daily vol, injected externally

    # ── Public ────────────────────────────────────────────────

    def set_sector_map(self, sector_map: dict[str, str]) -> None:
        self._sector_map = sector_map

    def set_vol_estimate(self, symbol: str, daily_vol_pct: float) -> None:
        """Inject a per-symbol daily-return volatility estimate for VaR calculation."""
        self._vol_estimates[symbol] = daily_vol_pct

    def set_vol_estimates(self, estimates: dict[str, float]) -> None:
        """
        Bulk-update per-symbol daily volatility estimates.

        Args:
            estimates: Mapping of symbol → daily vol fraction (e.g. 0.018 = 1.8%).
        """
        self._vol_estimates.update(estimates)

    def evaluate(
        self,
        signal: Signal,
        sim_time: Optional[datetime] = None,
    ) -> RiskDecision:
        """
        Evaluate a signal through all risk layers.

        Args:
            signal  : The trading signal to evaluate.
            sim_time: When provided (backtest mode) all time-dependent checks
                      use this timestamp instead of wall-clock time, so market-
                      hours gates and cooldowns work correctly during replay.
        """
        sim_epoch = sim_time.timestamp() if sim_time else None
        checks = [
            lambda s: self._check_active(s, sim_time),
            lambda s: self._check_throttle(s, sim_epoch),
            self._check_drawdown,
            lambda s: self._check_cooldown(s, sim_epoch),
            self._check_position_limit,
            self._check_sector_limit,
            self._check_var,          # 7. VaR check
        ]
        final = RiskDecision(approved=True)
        for check in checks:
            decision = check(signal)
            if not decision.approved:
                logger.warning(
                    "RISK REJECTED [%s] %s %s: %s",
                    check.__name__,
                    signal.signal_type.value,
                    signal.symbol,
                    decision.reason,
                )
                return decision
            # Propagate quantity adjustments from intermediate checks (e.g. position cap)
            if decision.adjusted_quantity is not None:
                final = decision

        # All checks passed — record state
        self._record_order(signal.symbol, sim_epoch)
        return final

    def record_nav(self, nav: float, sim_date=None) -> None:
        """Call once per day at market open to set baseline for drawdown."""
        today = sim_date or datetime.now().date()
        if self._current_date != today:
            self._daily_start_nav = nav
            self._current_date = today
            logger.info("Daily NAV baseline set: %.2f", nav)

    # ── Individual checks ────────────────────────────────────

    def _check_active(
        self,
        signal: Signal,
        sim_time: Optional[datetime] = None,
    ) -> RiskDecision:
        now = sim_time or datetime.now()

        # In backtest (sim_time supplied) we process end-of-day bars, so the
        # market was unambiguously open that day — skip the intraday hours check.
        # The check only makes sense for live / intraday trading.
        if sim_time is not None:
            return RiskDecision(approved=True)

        hour, minute = now.hour, now.minute
        current_time = hour * 100 + minute

        # A-shares trading hours (CST): 0930–1130, 1300–1500
        # Allow exits (SELL/CLOSE) even slightly outside regular hours
        is_exit = signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)

        if signal.symbol.startswith(("sh", "sz")):
            in_morning   = 930  <= current_time <= 1130
            in_afternoon = 1300 <= current_time <= 1500
            if not (in_morning or in_afternoon):
                if is_exit:
                    return RiskDecision(approved=True)  # Allow exits
                return RiskDecision(
                    approved=False,
                    reason=f"A-share market closed at {now.strftime('%H:%M')}",
                )
        elif signal.symbol.startswith("hk"):
            # HK: 0930–1600 (with lunch 1200–1300, but continuous since 2011)
            if not (930 <= current_time <= 1600):
                if is_exit:
                    return RiskDecision(approved=True)
                return RiskDecision(
                    approved=False,
                    reason=f"HK market closed at {now.strftime('%H:%M')}",
                )

        return RiskDecision(approved=True)

    def _check_throttle(
        self,
        signal: Signal,
        sim_epoch: Optional[float] = None,
    ) -> RiskDecision:
        # In backtest (sim_epoch provided) skip throttle — all bars replay instantly
        if sim_epoch is not None:
            return RiskDecision(approved=True)
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        while self._order_timestamps and now - self._order_timestamps[0] > 60:
            self._order_timestamps.popleft()

        if len(self._order_timestamps) >= settings.MAX_ORDERS_PER_MINUTE:
            return RiskDecision(
                approved=False,
                reason=f"Order rate limit reached ({settings.MAX_ORDERS_PER_MINUTE}/min)",
            )
        return RiskDecision(approved=True)

    def _check_drawdown(self, signal: Signal) -> RiskDecision:
        # Allow exits (SELL/CLOSE) even when drawdown limit is hit.
        # Only new entries (BUY) should be halted to stop digging the hole deeper.
        is_exit = signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)
        if is_exit:
            return RiskDecision(approved=True)

        if self._daily_start_nav is None or self._daily_start_nav == 0:
            return RiskDecision(approved=True)

        current_nav = self._get_nav()
        drawdown = (current_nav - self._daily_start_nav) / self._daily_start_nav

        if drawdown < settings.DAILY_DRAWDOWN_LIMIT:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Daily drawdown limit hit: {drawdown:.2%} < "
                    f"{settings.DAILY_DRAWDOWN_LIMIT:.2%}. New entries halted."
                ),
            )
        return RiskDecision(approved=True)

    def _check_cooldown(
        self,
        signal: Signal,
        sim_epoch: Optional[float] = None,
    ) -> RiskDecision:
        # Exits (SELL/CLOSE) must never be delayed by cooldown — allow immediately.
        is_exit = signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)
        if is_exit:
            return RiskDecision(approved=True)

        last = self._last_order_time.get(signal.symbol, 0)
        now = sim_epoch if sim_epoch is not None else time.monotonic()
        elapsed = now - last
        if elapsed < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - elapsed)
            return RiskDecision(
                approved=False,
                reason=f"Cooldown active for {signal.symbol}: {remaining}s remaining",
            )
        return RiskDecision(approved=True)

    def _check_position_limit(self, signal: Signal) -> RiskDecision:
        positions = self._get_positions()
        is_exit = signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)

        if is_exit:
            # ── No naked shorts ──────────────────────────────────────────
            # Only SELL shares we actually own (retail-style, no margin shorts).
            pos = positions.get(signal.symbol)
            long_qty = pos.quantity if pos else 0
            if long_qty <= 0:
                return RiskDecision(
                    approved=False,
                    reason=f"No long position to sell for {signal.symbol}",
                )
            # Cap SELL quantity to the shares currently held
            sell_qty = min(signal.quantity or long_qty, long_qty)
            return RiskDecision(approved=True, adjusted_quantity=sell_qty)

        # ── BUY: enforce per-symbol position limit ────────────────────
        nav = self._get_nav()
        if nav <= 0:
            return RiskDecision(approved=True)

        pos = positions.get(signal.symbol)
        current_exposure = abs(pos.market_value) if pos else 0.0
        new_exposure = signal.price * (signal.quantity or 100)
        total_exposure = current_exposure + new_exposure
        pct = total_exposure / nav

        if pct > settings.MAX_POSITION_PCT:
            # Try to reduce quantity to fit within limit
            max_notional = settings.MAX_POSITION_PCT * nav - current_exposure
            adjusted_qty = int(max_notional / signal.price) if signal.price > 0 else 0
            if adjusted_qty <= 0:
                return RiskDecision(
                    approved=False,
                    reason=f"Position limit exceeded for {signal.symbol}: {pct:.1%} > {settings.MAX_POSITION_PCT:.0%}",
                )
            return RiskDecision(
                approved=True,
                reason=f"Quantity adjusted down from {signal.quantity} to {adjusted_qty}",
                adjusted_quantity=adjusted_qty,
            )
        return RiskDecision(approved=True)

    def _check_sector_limit(self, signal: Signal) -> RiskDecision:
        if signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
            return RiskDecision(approved=True)

        sector = self._sector_map.get(signal.symbol)
        if not sector:
            return RiskDecision(approved=True)  # Unknown sector — skip

        nav = self._get_nav()
        if nav <= 0:
            return RiskDecision(approved=True)

        positions = self._get_positions()
        sector_exposure = sum(
            abs(p.market_value)
            for sym, p in positions.items()
            if self._sector_map.get(sym) == sector
        )
        pct = sector_exposure / nav

        if pct > settings.MAX_SECTOR_PCT:
            return RiskDecision(
                approved=False,
                reason=f"Sector '{sector}' exposure at {pct:.1%} > limit {settings.MAX_SECTOR_PCT:.0%}",
            )
        return RiskDecision(approved=True)

    # ── Internal ──────────────────────────────────────────────

    def _check_var(self, signal: Signal) -> RiskDecision:
        """
        Parametric 1-day 95% Value-at-Risk check.

        Rejects new BUY entries when the combined portfolio VaR (existing
        positions + proposed trade) would exceed ``_var_limit_pct`` × NAV.

            VaR_i = abs(market_value_i) × daily_vol_i × 1.645

        Assumes independent, normally distributed daily returns.  Inject
        accurate per-symbol vol estimates via ``set_vol_estimate()`` for
        better accuracy; defaults to ``_default_vol`` (2% for A-shares).
        """
        is_exit = signal.signal_type in (
            SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT
        )
        if is_exit:
            return RiskDecision(approved=True)

        nav = self._get_nav()
        if nav <= 0:
            return RiskDecision(approved=True)

        z95       = 1.645
        positions = self._get_positions()

        # VaR contribution from existing open positions
        portfolio_var = sum(
            abs(pos.market_value) * self._vol_estimates.get(sym, self._default_vol)
            for sym, pos in positions.items()
        ) * z95

        # VaR contribution from the proposed new position.
        # Use the position-limit-capped notional so VaR and position-limit
        # don't double-count the same risk: the effective exposure is
        # min(raw_notional, remaining room to MAX_POSITION_PCT limit).
        pos               = positions.get(signal.symbol)
        current_exposure  = abs(pos.market_value) if pos else 0.0
        max_new_notional  = max(0.0, settings.MAX_POSITION_PCT * nav - current_exposure)
        raw_notional      = signal.price * (signal.quantity or 100)
        proposed_notional = min(raw_notional, max_new_notional)
        proposed_var      = proposed_notional * self._vol_estimates.get(
            signal.symbol, self._default_vol
        ) * z95

        total_var = portfolio_var + proposed_var
        var_pct   = total_var / nav

        if var_pct > self._var_limit_pct:
            return RiskDecision(
                approved=False,
                reason=(
                    f"VaR limit exceeded: {var_pct:.1%} > {self._var_limit_pct:.1%} of NAV "
                    f"(portfolio={portfolio_var:,.0f}, proposed={proposed_var:,.0f})"
                ),
            )
        return RiskDecision(approved=True)

    def _record_order(
        self,
        symbol: str,
        sim_epoch: Optional[float] = None,
    ) -> None:
        now = sim_epoch if sim_epoch is not None else time.monotonic()
        self._order_timestamps.append(now)
        self._last_order_time[symbol] = now

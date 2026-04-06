"""
Execution Algorithms — VWAP and TWAP smart order execution.

In real trading, HOW you execute matters as much as WHAT you trade.
Large orders move the market. These algorithms split orders into
smaller slices to minimise market impact and slippage.

Implements:
  - VWAP (Volume-Weighted Average Price): execute proportional to historical volume
  - TWAP (Time-Weighted Average Price): execute evenly across time
  - Adaptive: starts as TWAP, switches to VWAP when volume picks up
  - Iceberg: expose only a fraction of the total order at a time
  - Execution Analytics: measure slippage, market impact, execution quality
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from myquant.config.logging_config import get_logger
from myquant.models.order import Order, OrderSide

logger = get_logger(__name__)


class AlgoType(str, Enum):
    VWAP = "VWAP"
    TWAP = "TWAP"
    ADAPTIVE = "ADAPTIVE"
    ICEBERG = "ICEBERG"


@dataclass
class ExecutionSlice:
    """A single child order in an algo execution."""
    slice_id: int
    target_qty: int
    filled_qty: int = 0
    target_time: Optional[datetime] = None
    fill_price: float = 0.0
    arrival_price: float = 0.0  # market price when slice was created
    slippage_bps: float = 0.0   # basis points of slippage


@dataclass
class ExecutionPlan:
    """Full execution plan for an algo order."""
    algo_type: AlgoType
    symbol: str
    side: OrderSide
    total_qty: int
    slices: list[ExecutionSlice] = field(default_factory=list)
    # Volume profile (for VWAP): fractional volume per slice
    volume_profile: list[float] = field(default_factory=list)
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # Urgency: 0.0 = patient, 1.0 = aggressive
    urgency: float = 0.5
    # Max participation rate (fraction of market volume to consume)
    max_participation: float = 0.10


@dataclass
class ExecutionReport:
    """Post-execution analysis."""
    symbol: str
    side: str
    total_qty: int
    total_filled: int
    avg_fill_price: float
    arrival_price: float       # price when order started
    vwap_benchmark: float      # market VWAP over execution period

    # Quality metrics
    slippage_bps: float = 0.0           # vs arrival price
    implementation_shortfall_bps: float = 0.0  # vs decision price
    market_impact_bps: float = 0.0      # estimated permanent impact
    participation_rate: float = 0.0     # our volume / market volume
    execution_time_seconds: float = 0.0

    def summary_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.total_qty,
            "filled": self.total_filled,
            "avg_price": round(self.avg_fill_price, 4),
            "arrival_price": round(self.arrival_price, 4),
            "vwap_benchmark": round(self.vwap_benchmark, 4),
            "slippage_bps": round(self.slippage_bps, 1),
            "impl_shortfall_bps": round(self.implementation_shortfall_bps, 1),
            "market_impact_bps": round(self.market_impact_bps, 1),
            "participation_rate": round(self.participation_rate, 3),
        }


class VWAPEngine:
    """
    VWAP execution engine: splits orders according to historical volume profile.

    For A-shares, the typical intraday volume curve is:
      - 09:30-10:00: 15% (opening rush)
      - 10:00-11:30: 25% (morning session)
      - 13:00-14:00: 20% (afternoon open)
      - 14:00-14:30: 15% (afternoon)
      - 14:30-15:00: 25% (closing rush)
    """

    # Default A-share intraday volume profile (5 bins)
    DEFAULT_VOLUME_PROFILE = [0.15, 0.25, 0.20, 0.15, 0.25]

    def create_plan(
        self,
        symbol: str,
        side: OrderSide,
        total_qty: int,
        volume_profile: Optional[list[float]] = None,
        urgency: float = 0.5,
        max_participation: float = 0.10,
        num_slices: int = 10,
    ) -> ExecutionPlan:
        """
        Create a VWAP execution plan.

        Args:
            symbol: Instrument to trade.
            side: BUY or SELL.
            total_qty: Total shares to execute.
            volume_profile: Normalised volume weights per time bucket.
            urgency: 0=patient (spread across full session), 1=aggressive (front-load).
            max_participation: Max fraction of market volume per slice.
            num_slices: Number of child orders to split into.
        """
        profile = volume_profile or self.DEFAULT_VOLUME_PROFILE

        # Adjust profile for urgency
        if urgency > 0.5:
            # Front-load: increase early slices
            adjusted = []
            for i, v in enumerate(profile):
                weight = 1 + (urgency - 0.5) * 2 * (1 - i / len(profile))
                adjusted.append(v * weight)
            total = sum(adjusted)
            profile = [v / total for v in adjusted]

        # Distribute quantity across slices according to profile
        # Interpolate profile to num_slices
        slice_weights = np.interp(
            np.linspace(0, len(profile) - 1, num_slices),
            np.arange(len(profile)),
            profile,
        ).tolist()
        total_weight = sum(slice_weights)
        slice_weights = [w / total_weight for w in slice_weights]

        slices = []
        remaining = total_qty
        lot_size = 100  # A-share lot

        for i, weight in enumerate(slice_weights):
            target = int(total_qty * weight)
            target = max(lot_size, (target // lot_size) * lot_size)
            target = min(target, remaining)

            if target > 0:
                slices.append(ExecutionSlice(
                    slice_id=i,
                    target_qty=target,
                ))
                remaining -= target

        # Distribute remainder to last slice
        if remaining > 0 and slices:
            slices[-1].target_qty += remaining

        return ExecutionPlan(
            algo_type=AlgoType.VWAP,
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            slices=slices,
            volume_profile=slice_weights,
            urgency=urgency,
            max_participation=max_participation,
        )


class TWAPEngine:
    """
    TWAP execution engine: splits orders evenly across time.
    Simpler than VWAP but effective when volume profile is unknown.
    """

    def create_plan(
        self,
        symbol: str,
        side: OrderSide,
        total_qty: int,
        duration_minutes: int = 60,
        num_slices: int = 10,
    ) -> ExecutionPlan:
        """Create a TWAP plan with equal-sized slices."""
        lot_size = 100
        base_qty = total_qty // num_slices
        base_qty = max(lot_size, (base_qty // lot_size) * lot_size)

        slices = []
        remaining = total_qty
        interval = duration_minutes / num_slices

        for i in range(num_slices):
            qty = min(base_qty, remaining)
            if qty < lot_size:
                break
            slices.append(ExecutionSlice(
                slice_id=i,
                target_qty=qty,
            ))
            remaining -= qty

        if remaining >= lot_size and slices:
            slices[-1].target_qty += (remaining // lot_size) * lot_size

        return ExecutionPlan(
            algo_type=AlgoType.TWAP,
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            slices=slices,
        )


class IcebergEngine:
    """
    Iceberg order: only show a fraction of total size to the market.
    When the visible portion fills, refresh with another visible chunk.
    """

    def create_plan(
        self,
        symbol: str,
        side: OrderSide,
        total_qty: int,
        visible_pct: float = 0.10,
        randomize: bool = True,
    ) -> ExecutionPlan:
        """
        Create an iceberg plan.

        Args:
            visible_pct: Fraction of total to show at a time.
            randomize: Add ±20% random variation to visible size.
        """
        lot_size = 100
        base_visible = max(lot_size, int(total_qty * visible_pct))
        base_visible = (base_visible // lot_size) * lot_size

        slices = []
        remaining = total_qty
        i = 0

        while remaining >= lot_size:
            if randomize:
                import random
                variation = random.uniform(0.8, 1.2)
                visible = int(base_visible * variation)
                visible = max(lot_size, (visible // lot_size) * lot_size)
            else:
                visible = base_visible

            qty = min(visible, remaining)
            qty = (qty // lot_size) * lot_size
            if qty < lot_size:
                break

            slices.append(ExecutionSlice(slice_id=i, target_qty=qty))
            remaining -= qty
            i += 1

        return ExecutionPlan(
            algo_type=AlgoType.ICEBERG,
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            slices=slices,
        )


class ExecutionAnalytics:
    """
    Measures execution quality: slippage, market impact, implementation shortfall.
    """

    @staticmethod
    def analyze(
        fills: list[tuple[int, float]],  # (qty, price) pairs
        arrival_price: float,
        decision_price: float,
        market_vwap: float,
        market_volume: int = 0,
    ) -> ExecutionReport:
        """
        Compute execution quality metrics.

        Args:
            fills: List of (quantity, fill_price) for each child fill.
            arrival_price: Market mid-price when algo started.
            decision_price: Price when signal was generated (for impl shortfall).
            market_vwap: Market VWAP during execution window.
            market_volume: Total market volume during execution window.
        """
        total_qty = sum(q for q, _ in fills)
        total_cost = sum(q * p for q, p in fills)
        avg_price = total_cost / total_qty if total_qty > 0 else 0.0

        # Slippage vs arrival price
        if arrival_price > 0:
            slippage_bps = (avg_price - arrival_price) / arrival_price * 10_000
        else:
            slippage_bps = 0.0

        # For BUY: positive slippage = we paid more than arrival (bad)
        # For SELL: negative slippage = we sold lower than arrival (bad)

        # Implementation shortfall vs decision price
        if decision_price > 0:
            impl_shortfall = (avg_price - decision_price) / decision_price * 10_000
        else:
            impl_shortfall = 0.0

        # Estimated market impact (simplified)
        participation = total_qty / max(market_volume, 1)
        market_impact = slippage_bps * 0.5  # rough: half of slippage is permanent

        return ExecutionReport(
            symbol=fills[0][1] if fills else "",
            side="BUY",  # caller should set
            total_qty=total_qty,
            total_filled=total_qty,
            avg_fill_price=avg_price,
            arrival_price=arrival_price,
            vwap_benchmark=market_vwap,
            slippage_bps=slippage_bps,
            implementation_shortfall_bps=impl_shortfall,
            market_impact_bps=market_impact,
            participation_rate=participation,
        )


# Needed for VWAPEngine
import numpy as np

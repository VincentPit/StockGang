"""
ATR-based Position Sizer — risk-adjusted lot calculator.

Replaces the fixed 100-share default with a quantity derived from:

  1. Risk budget per trade: ``risk_pct × NAV``  (default: 0.5% of portfolio)
  2. Volatility unit: 1 ATR — the expected single-bar adverse move
  3. Confidence scaling: higher model confidence → proportionally larger position

Formula
-------
    base_shares  = risk_budget / atr_per_share
    conf_scale   = 2 × (confidence − 0.5)   [clamped to 0.3 – 2.0]
    final_shares = round_down_to_lot(base_shares × conf_scale)
    final_shares = min(final_shares, max_pos_pct × NAV / price)

Usage
-----
    from myquant.strategy.sizing import atr_position_size

    qty = atr_position_size(
        nav        = 1_000_000,
        price      = 350.0,       # HKD close
        atr_pct    = 0.012,       # 1.2% of price = 1 ATR
        confidence = 0.65,        # model confidence
    )
    # → e.g. 400 shares (4 lots of 100)
"""
from __future__ import annotations


def atr_position_size(
    nav: float,
    price: float,
    atr_pct: float,
    confidence: float = 0.55,
    risk_pct: float = 0.005,
    lot_size: int = 100,
    max_pos_pct: float = 0.15,
) -> int:
    """
    Return position size (in shares) so a 1-ATR adverse move costs at most
    ``risk_pct × NAV``.

    Parameters
    ----------
    nav         : Current portfolio NAV.
    price       : Current close price of the instrument.
    atr_pct     : ATR expressed as a fraction of price (e.g. ``0.012`` = 1.2%).
                  Values below 0.003 are floored to avoid extreme over-sizing.
    confidence  : Model confidence score in [0.5, 1.0].
    risk_pct    : Fraction of NAV to risk per trade (default 0.5%).
    lot_size    : Minimum trading lot for rounding (100 for HK/A-shares).
    max_pos_pct : Hard cap — single-instrument position may not exceed this
                  fraction of NAV.

    Returns
    -------
    int : Recommended quantity (multiple of ``lot_size``, at least ``lot_size``).
    """
    if price <= 0 or nav <= 0:
        return lot_size

    # Floor ATR to avoid absurdly large positions on near-zero-vol instruments
    effective_atr = max(atr_pct, 0.003)
    atr_per_share = price * effective_atr

    risk_budget = nav * risk_pct
    base_shares = risk_budget / atr_per_share

    # Confidence scaling: 50% conf → 0.5× size, 80% conf → 1.6× size
    # Linear: scale = 2*(conf − 0.5), clamped [0.3, 2.0]
    conf_scale = max(0.3, min(2.0, 2.0 * (confidence - 0.5)))
    shares = int(base_shares * conf_scale)

    # Round down to the nearest whole lot, minimum 1 lot
    shares = max(lot_size, (shares // lot_size) * lot_size)

    # Hard cap: no single position > max_pos_pct of NAV
    max_shares = int(nav * max_pos_pct / price)
    max_shares = max(lot_size, (max_shares // lot_size) * lot_size)

    return min(shares, max_shares)

"""
Tick — a single real-time quote snapshot from Tencent's quote feed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Tick:
    """
    Represents one tick of market data.

    symbol  : Tencent-format code, e.g. "sh600036", "hk00700", "usTSLA"
    ts      : Server timestamp (exchange time)
    price   : Last traded price
    bid1    : Best bid price
    ask1    : Best ask price
    bid_vol1: Best bid volume (lots)
    ask_vol1: Best ask volume (lots)
    volume  : Cumulative volume (lots) since market open
    turnover: Cumulative turnover (CNY/HKD/USD) since market open
    open    : Opening price of the day
    high    : Intra-day high
    low     : Intra-day low
    prev_close: Previous day's closing price
    pct_chg : Percentage change from prev_close (decimal, e.g. 0.023 = +2.3%)
    """

    symbol: str
    ts: datetime
    price: float
    bid1: float = 0.0
    ask1: float = 0.0
    bid_vol1: int = 0
    ask_vol1: int = 0
    volume: int = 0
    turnover: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0
    pct_chg: float = 0.0

    # Five-level order book (optional, populated when available)
    bids: list[tuple[float, int]] = field(default_factory=list)
    asks: list[tuple[float, int]] = field(default_factory=list)

    # Extra metadata
    exchange: str = ""      # "SH" | "SZ" | "HK" | "US"
    name: str = ""          # Chinese short name, e.g. "招商银行"

    @property
    def mid_price(self) -> float:
        if self.bid1 > 0 and self.ask1 > 0:
            return (self.bid1 + self.ask1) / 2
        return self.price

    @property
    def spread(self) -> float:
        if self.bid1 > 0 and self.ask1 > 0:
            return self.ask1 - self.bid1
        return 0.0

    @classmethod
    def from_tencent_raw(cls, raw: str) -> "Tick":
        """
        Parse a raw Tencent quote string.
        Format: v_sh600036="1~招商银行~600036~price~prev_close~open~..."
        """
        try:
            inner = raw.split("=", 1)[1].strip().strip('"')
            parts = inner.split("~")

            def _f(idx: int, default: float = 0.0) -> float:
                try:
                    return float(parts[idx]) if parts[idx] else default
                except (IndexError, ValueError):
                    return default

            def _i(idx: int, default: int = 0) -> int:
                try:
                    return int(float(parts[idx])) if parts[idx] else default
                except (IndexError, ValueError):
                    return default

            # Tencent field positions (v_sh type)
            name = parts[1] if len(parts) > 1 else ""
            price = _f(3)
            prev_close = _f(4)
            open_ = _f(5)
            volume = _i(6)
            turnover = _f(37)
            high = _f(33)
            low = _f(34)

            bid1 = _f(9)
            bid_vol1 = _i(10)
            ask1 = _f(19)
            ask_vol1 = _i(20)

            pct_chg = (price - prev_close) / prev_close if prev_close else 0.0

            # Parse symbol from key prefix
            key_part = raw.split("=", 1)[0].split("v_", 1)[-1]
            exchange = key_part[:2].upper()

            # Parse bids / asks (levels 1-5)
            bids = [(_f(9 + i * 2), _i(10 + i * 2)) for i in range(5)]
            asks = [(_f(19 + i * 2), _i(20 + i * 2)) for i in range(5)]

            return cls(
                symbol=key_part,
                ts=datetime.now(),
                price=price,
                bid1=bid1,
                ask1=ask1,
                bid_vol1=bid_vol1,
                ask_vol1=ask_vol1,
                volume=volume,
                turnover=turnover,
                open=open_,
                high=high,
                low=low,
                prev_close=prev_close,
                pct_chg=pct_chg,
                bids=bids,
                asks=asks,
                exchange=exchange,
                name=name,
            )
        except Exception:
            raise ValueError(f"Failed to parse Tencent raw tick: {raw[:100]}")

    def __repr__(self) -> str:
        return (
            f"Tick({self.symbol} {self.price:.4f} "
            f"[{self.pct_chg:+.2%}] @ {self.ts.strftime('%H:%M:%S')})"
        )

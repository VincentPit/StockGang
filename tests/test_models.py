"""
Tests for core data models.
"""
from datetime import datetime

import pytest

from myquant.models.bar import Bar, BarInterval
from myquant.models.order import Order, OrderSide, OrderStatus, OrderType
from myquant.models.position import Position
from myquant.models.tick import Tick


class TestTick:
    def test_mid_price(self):
        tick = Tick(symbol="hk00700", ts=datetime.now(), price=340.0, bid1=339.8, ask1=340.2)
        assert tick.mid_price == 340.0

    def test_spread(self):
        tick = Tick(symbol="hk00700", ts=datetime.now(), price=340.0, bid1=339.8, ask1=340.2)
        assert abs(tick.spread - 0.4) < 0.001

    def test_pct_chg(self):
        tick = Tick(symbol="sh600036", ts=datetime.now(), price=44.0, prev_close=40.0, pct_chg=0.1)
        assert tick.pct_chg == pytest.approx(0.1)


class TestBar:
    def test_bullish(self):
        bar = Bar("hk00700", datetime.now(), BarInterval.D1, 330, 345, 328, 342)
        assert bar.is_bullish

    def test_bearish(self):
        bar = Bar("hk00700", datetime.now(), BarInterval.D1, 342, 345, 328, 330)
        assert not bar.is_bullish

    def test_pct_chg(self):
        bar = Bar("hk00700", datetime.now(), BarInterval.D1, 300, 310, 295, 306)
        assert bar.pct_chg == pytest.approx(0.02)


class TestPosition:
    def test_open_long(self):
        pos = Position("hk00700")
        pos.add_fill(200, 330.0)
        assert pos.quantity == 200
        assert pos.avg_cost == pytest.approx(330.0)
        assert pos.is_long

    def test_add_to_position(self):
        pos = Position("hk00700")
        pos.add_fill(200, 330.0)
        pos.add_fill(100, 336.0)
        assert pos.quantity == 300
        assert pos.avg_cost == pytest.approx(332.0)  # (200*330 + 100*336) / 300

    def test_partial_close(self):
        pos = Position("sh600036")
        pos.add_fill(1000, 42.0)
        realized = pos.add_fill(-500, 45.0)
        assert pos.quantity == 500
        assert realized == pytest.approx(1500.0)  # 500 * (45 - 42)
        assert pos.realized_pnl == pytest.approx(1500.0)

    def test_full_close(self):
        pos = Position("sh600036")
        pos.add_fill(1000, 42.0)
        pos.add_fill(-1000, 45.0)
        assert pos.is_flat
        assert pos.avg_cost == 0.0

    def test_unrealized_pnl(self):
        pos = Position("usTSLA")
        pos.add_fill(50, 195.0)
        pos.update_price(205.0)
        assert pos.unrealized_pnl == pytest.approx(500.0)


class TestOrder:
    def test_fill(self):
        order = Order("hk00700", OrderSide.BUY, OrderType.LIMIT, 200, limit_price=330.0)
        order.apply_fill(200, 330.5, commission=19.83)
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(330.5)
        assert order.commission == pytest.approx(19.83)

    def test_partial_fill(self):
        order = Order("hk00700", OrderSide.BUY, OrderType.LIMIT, 400, limit_price=330.0)
        order.apply_fill(200, 330.0)
        assert order.status == OrderStatus.PARTIAL
        assert order.remaining_quantity == 200

    def test_notional(self):
        order = Order("sh600036", OrderSide.BUY, OrderType.LIMIT, 1000, limit_price=42.5)
        assert order.notional == pytest.approx(42500.0)

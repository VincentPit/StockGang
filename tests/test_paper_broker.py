"""
Tests for PaperBroker — order fill simulation.
"""
import asyncio

import pytest

from myquant.execution.brokers.paper_broker import PaperBroker
from myquant.models.order import Order, OrderSide, OrderStatus, OrderType


def _run(coro):
    """Run a coroutine synchronously without touching the deprecated event-loop API."""
    return asyncio.run(coro)


@pytest.fixture
def broker():
    b = PaperBroker(
        initial_cash=100_000.0,
        commission_rate=0.0003,
        slippage=0.0,      # Zero slippage for deterministic tests
        apply_stamp_duty=False,
        price_getter=lambda sym: 100.0,
    )
    _run(b.connect())
    return b


class TestPaperBroker:
    def test_market_buy_fills_immediately(self, broker):
        order = Order("hk00700", OrderSide.BUY, OrderType.MARKET, 100, limit_price=100.0)
        _run(broker.submit_order(order))
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100

    def test_market_buy_deducts_cash(self, broker):
        initial_cash = _run(broker.get_cash())
        order = Order("hk00700", OrderSide.BUY, OrderType.MARKET, 100, limit_price=100.0)
        _run(broker.submit_order(order))
        new_cash = _run(broker.get_cash())
        cost = 100 * 100.0 + 100 * 100.0 * 0.0003
        assert abs((initial_cash - new_cash) - cost) < 0.01

    def test_insufficient_cash_rejected(self, broker):
        # Try to buy 10,000 shares at 100 = 1,000,000 but only have 100,000
        order = Order("hk00700", OrderSide.BUY, OrderType.MARKET, 10_000, limit_price=100.0)
        _run(broker.submit_order(order))
        assert order.status == OrderStatus.REJECTED

    def test_limit_buy_not_filled_immediately(self, broker):
        # Limit buy at 90, price is 100 — should NOT fill
        order = Order("hk00700", OrderSide.BUY, OrderType.LIMIT, 100, limit_price=90.0)
        _run(broker.submit_order(order))
        assert order.status == OrderStatus.SUBMITTED

    def test_limit_buy_fills_on_price_drop(self, broker):
        order = Order("hk00700", OrderSide.BUY, OrderType.LIMIT, 100, limit_price=95.0)
        _run(broker.submit_order(order))
        # Simulate price dropping to 94
        broker.simulate_tick("hk00700", 94.0)
        assert order.status == OrderStatus.FILLED

    def test_market_sell(self, broker):
        # Buy first
        buy_order = Order("hk00700", OrderSide.BUY, OrderType.MARKET, 100, limit_price=100.0)
        _run(broker.submit_order(buy_order))
        # Then sell
        sell_order = Order("hk00700", OrderSide.SELL, OrderType.MARKET, 100, limit_price=100.0)
        _run(broker.submit_order(sell_order))
        assert sell_order.status == OrderStatus.FILLED
        positions = _run(broker.get_positions())
        assert "hk00700" not in positions

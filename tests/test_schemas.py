"""
tests/test_schemas.py — Pydantic schema validation tests.

Covers every validation rule in api/schemas.py: required fields,
sh/sz enforcement, deduplication, length cap, and numeric bounds.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    AnalyzeRequest,
    BacktestRequest,
    OrderRequest,
    ScreenRequest,
    TrainRequest,
    WorkflowRequest,
)


# ═══════════════════════════════════════════════════════════════════
# BacktestRequest
# ═══════════════════════════════════════════════════════════════════

class TestBacktestRequestSymbols:
    """symbols is now a required field with sh/sz enforcement."""

    def test_valid_symbols_accepted(self):
        req = BacktestRequest(symbols=["sh600519", "sz300750"])
        assert req.symbols == ["sh600519", "sz300750"]

    def test_symbols_are_lowercased(self):
        req = BacktestRequest(symbols=["SH600519", "SZ300750"])
        assert req.symbols == ["sh600519", "sz300750"]

    def test_symbols_are_stripped(self):
        req = BacktestRequest(symbols=["  sh600519  ", " sz300750"])
        assert req.symbols == ["sh600519", "sz300750"]

    def test_symbols_are_deduplicated_preserving_order(self):
        req = BacktestRequest(symbols=["sh600519", "sz300750", "sh600519"])
        assert req.symbols == ["sh600519", "sz300750"]

    def test_blank_entries_are_filtered_out(self):
        req = BacktestRequest(symbols=["sh600519", "  ", "", "sz300750"])
        assert req.symbols == ["sh600519", "sz300750"]

    def test_symbols_required_raises_when_missing(self):
        with pytest.raises(ValidationError, match="symbols"):
            BacktestRequest()

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError, match="at least one"):
            BacktestRequest(symbols=[])

    def test_all_blank_raises(self):
        with pytest.raises(ValidationError, match="at least one"):
            BacktestRequest(symbols=["  ", ""])

    def test_non_sh_sz_raises(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            BacktestRequest(symbols=["hk00700"])

    def test_mixed_valid_invalid_raises(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            BacktestRequest(symbols=["sh600519", "AAPL"])

    def test_21_symbols_raises_cap_error(self):
        symbols = [f"sh{600000 + i}" for i in range(21)]
        with pytest.raises(ValidationError, match="At most 20"):
            BacktestRequest(symbols=symbols)

    def test_exactly_20_symbols_accepted(self):
        symbols = [f"sh{600000 + i}" for i in range(20)]
        req = BacktestRequest(symbols=symbols)
        assert len(req.symbols) == 20

    def test_single_symbol_accepted(self):
        req = BacktestRequest(symbols=["sh600519"])
        assert req.symbols == ["sh600519"]


class TestBacktestRequestNumericFields:
    def _valid(self, **kwargs) -> BacktestRequest:
        return BacktestRequest(symbols=["sh600519"], **kwargs)

    def test_default_lookback_days(self):
        assert self._valid().lookback_days == 365

    def test_lookback_days_minimum(self):
        assert self._valid(lookback_days=30).lookback_days == 30

    def test_lookback_days_too_small_raises(self):
        with pytest.raises(ValidationError):
            self._valid(lookback_days=29)

    def test_lookback_days_maximum(self):
        assert self._valid(lookback_days=730).lookback_days == 730

    def test_lookback_days_too_large_raises(self):
        with pytest.raises(ValidationError):
            self._valid(lookback_days=731)

    def test_initial_cash_minimum(self):
        assert self._valid(initial_cash=10_000.0).initial_cash == 10_000.0

    def test_initial_cash_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            self._valid(initial_cash=9_999.0)

    def test_commission_rate_zero_accepted(self):
        assert self._valid(commission_rate=0.0).commission_rate == 0.0

    def test_commission_rate_maximum(self):
        assert self._valid(commission_rate=0.01).commission_rate == 0.01

    def test_commission_rate_above_max_raises(self):
        with pytest.raises(ValidationError):
            self._valid(commission_rate=0.011)

    def test_commission_rate_negative_raises(self):
        with pytest.raises(ValidationError):
            self._valid(commission_rate=-0.001)

    def test_stop_loss_must_be_le_zero(self):
        with pytest.raises(ValidationError):
            self._valid(stop_loss_pct=0.01)

    def test_stop_loss_zero_accepted(self):
        assert self._valid(stop_loss_pct=0.0).stop_loss_pct == 0.0

    def test_trailing_stop_negative_raises(self):
        with pytest.raises(ValidationError):
            self._valid(trailing_stop_pct=-0.01)

    def test_take_profit_negative_raises(self):
        with pytest.raises(ValidationError):
            self._valid(take_profit_pct=-0.01)


# ═══════════════════════════════════════════════════════════════════
# ScreenRequest
# ═══════════════════════════════════════════════════════════════════

class TestScreenRequest:
    def test_defaults(self):
        req = ScreenRequest()
        assert req.top_n == 6
        assert req.indices == ["000300"]

    def test_valid_index(self):
        req = ScreenRequest(indices=["000905"])
        assert req.indices == ["000905"]

    def test_multiple_valid_indices(self):
        req = ScreenRequest(indices=["000300", "000905", "000852"])
        assert len(req.indices) == 3

    def test_unknown_index_raises(self):
        with pytest.raises(ValidationError, match="Unknown index"):
            ScreenRequest(indices=["999999"])

    def test_empty_indices_raises(self):
        with pytest.raises(ValidationError, match="at least one"):
            ScreenRequest(indices=[])

    def test_top_n_minimum(self):
        assert ScreenRequest(top_n=1).top_n == 1

    def test_top_n_maximum(self):
        assert ScreenRequest(top_n=20).top_n == 20

    def test_top_n_zero_raises(self):
        with pytest.raises(ValidationError):
            ScreenRequest(top_n=0)

    def test_top_n_above_max_raises(self):
        with pytest.raises(ValidationError):
            ScreenRequest(top_n=21)


# ═══════════════════════════════════════════════════════════════════
# TrainRequest / AnalyzeRequest (share symbol validator)
# ═══════════════════════════════════════════════════════════════════

class TestTrainRequest:
    def test_valid_sh_symbol(self):
        req = TrainRequest(symbol="sh600519")
        assert req.symbol == "sh600519"

    def test_valid_sz_symbol(self):
        req = TrainRequest(symbol="sz300750")
        assert req.symbol == "sz300750"

    def test_symbol_uppercased_input_lowercased(self):
        req = TrainRequest(symbol="SH600519")
        assert req.symbol == "sh600519"

    def test_blank_symbol_raises(self):
        with pytest.raises(ValidationError, match="blank"):
            TrainRequest(symbol="   ")

    def test_non_sh_sz_raises(self):
        with pytest.raises(ValidationError, match="SH/SZ"):
            TrainRequest(symbol="AAPL")


class TestAnalyzeRequest:
    def test_valid(self):
        from api.schemas import AnalyzeRequest
        req = AnalyzeRequest(symbol="sz000858")
        assert req.symbol == "sz000858"

    def test_invalid_raises(self):
        from api.schemas import AnalyzeRequest
        with pytest.raises(ValidationError):
            AnalyzeRequest(symbol="TSLA")


# ═══════════════════════════════════════════════════════════════════
# WorkflowRequest
# ═══════════════════════════════════════════════════════════════════

class TestWorkflowRequest:
    def test_defaults(self):
        req = WorkflowRequest()
        assert req.top_n == 6
        assert req.backtest_days == 365
        assert req.trailing_stop_pct == 0.0
        assert req.take_profit_pct == 0.0

    def test_trailing_stop_accepted(self):
        req = WorkflowRequest(trailing_stop_pct=0.07)
        assert req.trailing_stop_pct == pytest.approx(0.07)

    def test_take_profit_accepted(self):
        req = WorkflowRequest(take_profit_pct=0.15)
        assert req.take_profit_pct == pytest.approx(0.15)

    def test_negative_trailing_stop_raises(self):
        with pytest.raises(ValidationError):
            WorkflowRequest(trailing_stop_pct=-0.05)

    def test_backtest_days_too_small_raises(self):
        with pytest.raises(ValidationError):
            WorkflowRequest(backtest_days=10)

    def test_commission_rate_max_accepted(self):
        req = WorkflowRequest(commission_rate=0.01)
        assert req.commission_rate == pytest.approx(0.01)

    def test_commission_rate_above_max_raises(self):
        with pytest.raises(ValidationError):
            WorkflowRequest(commission_rate=0.02)

    def test_commission_rate_negative_raises(self):
        with pytest.raises(ValidationError):
            WorkflowRequest(commission_rate=-0.001)

    def test_empty_indices_raises(self):
        with pytest.raises(ValidationError, match="at least one"):
            WorkflowRequest(indices=[])

    def test_unknown_index_raises(self):
        with pytest.raises(ValidationError, match="Unknown index"):
            WorkflowRequest(indices=["999999"])


# ═══════════════════════════════════════════════════════════════════
# OrderRequest
# ═══════════════════════════════════════════════════════════════════

class TestOrderRequest:
    def _valid(self, **kwargs) -> OrderRequest:
        defaults = dict(symbol="sh600519", side="BUY", quantity=100)
        defaults.update(kwargs)
        return OrderRequest(**defaults)

    def test_valid_buy(self):
        req = self._valid()
        assert req.side == "BUY"

    def test_valid_sell(self):
        req = self._valid(side="SELL")
        assert req.side == "SELL"

    def test_invalid_side_raises(self):
        with pytest.raises(ValidationError):
            self._valid(side="HOLD")

    def test_market_order_type(self):
        assert self._valid(order_type="MARKET").order_type == "MARKET"

    def test_limit_order_type(self):
        assert self._valid(order_type="LIMIT").order_type == "LIMIT"

    def test_invalid_order_type_raises(self):
        with pytest.raises(ValidationError):
            self._valid(order_type="STOP")

    def test_zero_quantity_raises(self):
        with pytest.raises(ValidationError):
            self._valid(quantity=0)

    def test_negative_quantity_raises(self):
        with pytest.raises(ValidationError):
            self._valid(quantity=-1)

    def test_blank_symbol_raises(self):
        with pytest.raises(ValidationError):
            self._valid(symbol="  ")

    def test_symbol_lowercased(self):
        req = self._valid(symbol="SH600519")
        assert req.symbol == "sh600519"

    def test_limit_price_positive_accepted(self):
        req = self._valid(order_type="LIMIT", limit_price=50.0)
        assert req.limit_price == 50.0

    def test_limit_price_zero_raises(self):
        with pytest.raises(ValidationError):
            self._valid(order_type="LIMIT", limit_price=0.0)

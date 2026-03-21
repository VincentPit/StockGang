"""
Backtesting Simulator — event-driven replay of historical bars.

Workflow:
    1. Load historical bars via HistoricalLoader
    2. Register strategies in StrategyRegistry
    3. Replay bars chronologically → strategies emit signals
    4. Signals pass through RiskGate
    5. Orders fill via PaperBroker with realistic cost model
    6. PortfolioEngine tracks PnL
    7. BacktestReporter generates tearsheet
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from myquant.config.logging_config import get_logger
from myquant.data.fetchers.historical_loader import HistoricalLoader
from myquant.data.fetchers.macro_proxy import HistoricalRegimeDetector
from myquant.execution.brokers.paper_broker import PaperBroker
from myquant.execution.order_manager import OrderManager
from myquant.models.bar import Bar, BarInterval
from myquant.models.order import Order, OrderSide
from myquant.models.signal import Signal, SignalType
from myquant.models.tick import Tick
from myquant.portfolio.portfolio_engine import PortfolioEngine
from myquant.risk.risk_gate import RiskDecision, RiskGate
from myquant.strategy.registry import StrategyRegistry
from myquant.strategy.rl.bandit import StrategyBandit
from myquant.strategy.sizing import atr_position_size

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    symbols: list[str]
    start_date: datetime
    end_date: datetime
    initial_cash: float = 1_000_000.0
    interval: BarInterval = BarInterval.D1
    commission_rate: float = 0.0003
    slippage: float = 0.0002
    apply_stamp_duty: bool = True
    train_years: int = 0   # if > 0, warm strategies on [start_date - train_years, start_date] data
    stop_loss_pct: float = -0.08       # force-close any position down more than this from cost
    trailing_stop_pct: float = 0.0     # if > 0, trail stop this % below running price peak; 0 = off
    take_profit_pct:   float = 0.0     # if > 0, lock in gains when PnL exceeds this %; 0 = off
    symbol_loss_cap: float = -20_000.0 # stop all new BUYs on symbol once cumulative realized P&L < this


@dataclass
class BacktestResult:
    config: BacktestConfig
    nav_series: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "        BACKTEST RESULTS",
            "=" * 50,
            f"  Period     : {self.config.start_date.date()} → {self.config.end_date.date()}",
            f"  Symbols    : {', '.join(self.config.symbols)}",
            f"  Initial NAV: {self.config.initial_cash:>12,.2f}",
            f"  Final NAV  : {self.config.initial_cash + self.total_pnl:>12,.2f}",
            f"  Total PnL  : {self.total_pnl:>+12,.2f}  ({self.total_pnl_pct:+.2%})",
            f"  Sharpe     : {self.sharpe_ratio:>12.3f}",
            f"  Max DD     : {self.max_drawdown:>12.2%}",
            f"  # Trades   : {self.num_trades:>12}",
            f"  Win Rate   : {self.win_rate:>12.2%}",
            f"  Avg Win    : {self.avg_win:>+12.2f}",
            f"  Avg Loss   : {self.avg_loss:>+12.2f}",
            f"  Profit Fac : {self.profit_factor:>12.3f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Event-driven backtester.
    """

    def __init__(self, config: BacktestConfig, macro_filter=None) -> None:
        self.config = config
        self._macro_filter = macro_filter  # Optional[MacroFilter]
        self._loader = HistoricalLoader()
        self._portfolio = PortfolioEngine(initial_cash=config.initial_cash)
        self._broker = PaperBroker(
            initial_cash=config.initial_cash,
            commission_rate=config.commission_rate,
            slippage=config.slippage,
            apply_stamp_duty=config.apply_stamp_duty,
            price_getter=self._get_current_price,
        )
        self._order_manager = OrderManager(broker=self._broker)
        # _track_sym_pnl must be registered BEFORE portfolio.on_fill so the
        # position still exists when we read avg_cost to compute realized P&L.
        self._symbol_cum_pnl: dict[str, float] = {}   # cumulative gross realized P&L per symbol
        self._order_manager.on_fill(self._track_sym_pnl)
        self._order_manager.on_fill(self._portfolio.on_fill)

        self._registry = StrategyRegistry()
        self._risk_gate = RiskGate(
            nav_getter=lambda: self._portfolio.nav,
            positions_getter=lambda: self._portfolio.positions,
        )

        # ── Regime & ATR tracking (no look-ahead) ─────────────────────
        self._regime_detector = HistoricalRegimeDetector()
        self._atr_history:  dict[str, list[float]] = {}   # rolling 14-bar TR values
        self._prev_close:   dict[str, float]       = {}   # for TR calculation
        self._bar_atr_pct:  dict[str, float]       = {}   # latest ATR fraction
        self._ma50_buf:     dict[str, list[float]] = {}   # rolling 50-bar close buffer
        self._ma50:         dict[str, float]       = {}   # latest MA50 value
        # Per-symbol running price peak for trailing stop (reset on each new entry)
        self._peak_price: dict[str, float] = {}

        # ── RL bandit: adaptive strategy weighting ──────────────────────
        # Tracks realized P&L per strategy and assigns UCB1 confidence weights.
        # Weights start neutral (1.0) and adapt as trades close.
        self._bandit = StrategyBandit(
            strategy_ids=[],   # populated lazily as strategies register signals
            ema_alpha=0.15,
            ucb_c=1.0,
            pnl_scale=config.initial_cash * 0.005,  # 0.5% of initial capital
        )
        # Maps symbol → strategy_id that opened the current long position
        self._position_open_strat: dict[str, str] = {}
        # Current bar prices (used by broker for slippage)
        self._current_prices: dict[str, float] = {}

    # ── Setup ─────────────────────────────────────────────────

    def add_strategy(self, strategy) -> "Backtester":
        self._registry.register(strategy)
        return self

    # ── Run ───────────────────────────────────────────────────

    async def run(self) -> BacktestResult:
        logger.info("Loading historical data for %d symbols...", len(self.config.symbols))

        # ── Warm-up / training pass ──────────────────────────────
        if self.config.train_years > 0:
            from datetime import timedelta
            train_end = self.config.start_date
            train_start = train_end - timedelta(days=365 * self.config.train_years)
            logger.info(
                "Loading %d-year training window: %s → %s",
                self.config.train_years, train_start.date(), train_end.date(),
            )
            for symbol in self.config.symbols:
                warm_bars = self._loader.load_bars(
                    symbol,
                    start=train_start.date(),
                    end=train_end.date(),
                    interval=self.config.interval,
                )
                if warm_bars:
                    logger.info("  Warm-up: %d bars for %s", len(warm_bars), symbol)
                    for strategy in self._registry._strategies.values():
                        if symbol in strategy.symbols:
                            strategy.warm_bars(symbol, warm_bars)
                    # Pre-seed ATR and MA50 from warm-up data so they're ready at bar 1
                    for wb in warm_bars:
                        self._bar_atr_pct[wb.symbol] = self._update_atr(wb)
                        self._update_ma50(wb)

        # ── Main backtest data load ───────────────────────────────
        all_bars: list[Bar] = []
        for symbol in self.config.symbols:
            bars = self._loader.load_bars(
                symbol,
                start=self.config.start_date.date(),
                end=self.config.end_date.date(),
                interval=self.config.interval,
            )
            logger.info("Loaded %d bars for %s", len(bars), symbol)
            all_bars.extend(bars)

        # Sort all bars chronologically
        all_bars.sort(key=lambda b: b.ts)

        # Start strategies (triggers LGBMStrategy training on warm data)
        await self._registry.start_all()

        # Set daily nav baseline
        self._risk_gate.record_nav(self._portfolio.nav)

        logger.info("Replaying %d bars...", len(all_bars))

        # Replay
        nav_records: list[tuple[datetime, float]] = []
        prev_date = None

        for bar in all_bars:
            self._current_prices[bar.symbol] = bar.close

            # Simulate price movement for limit order checks
            self._broker.simulate_tick(bar.symbol, bar.low)
            self._broker.simulate_tick(bar.symbol, bar.high)
            self._broker.simulate_tick(bar.symbol, bar.close)

            # Update regime detector, ATR, and MA50 (no look-ahead)
            self._regime_detector.update(bar.symbol, bar.close)
            self._bar_atr_pct[bar.symbol] = self._update_atr(bar)
            self._update_ma50(bar)

            # Update portfolio mark-to-market
            fake_tick = _bar_to_tick(bar)
            self._portfolio.on_tick(fake_tick)

            # ── Risk exits: stop-loss / trailing-stop / take-profit ────────
            pos = self._portfolio.positions.get(bar.symbol)
            if pos is not None and pos.is_long:
                # Track running price peak for trailing stop (reset when position opened)
                peak = self._peak_price.get(bar.symbol, pos.avg_cost)
                if bar.close > peak:
                    self._peak_price[bar.symbol] = bar.close
                    peak = bar.close

                exit_reason: str | None = None

                # 1. Fixed stop-loss: position unrealised PnL below threshold from cost
                if pos.pnl_pct < self.config.stop_loss_pct:
                    exit_reason = f"stop_loss pnl={pos.pnl_pct:.1%}"

                # 2. Trailing stop: close fell more than trailing_stop_pct from peak
                elif self.config.trailing_stop_pct > 0:
                    drop_from_peak = (bar.close - peak) / (peak + 1e-10)
                    if drop_from_peak < -self.config.trailing_stop_pct:
                        exit_reason = (
                            f"trailing_stop drop={drop_from_peak:.1%} "
                            f"from peak={peak:.2f}"
                        )

                # 3. Take-profit: unrealised PnL has hit the target
                elif self.config.take_profit_pct > 0 and pos.pnl_pct >= self.config.take_profit_pct:
                    exit_reason = f"take_profit pnl={pos.pnl_pct:.1%}"

                if exit_reason:
                    stop_sig = Signal(
                        symbol      = bar.symbol,
                        signal_type = SignalType.SELL,
                        price       = bar.close,
                        confidence  = 1.0,
                        strategy_id = "risk_exit",
                        metadata    = {"reason": exit_reason},
                    )
                    qty = pos.quantity
                    # Bypass risk gate — risk exits must always execute
                    await self._order_manager.process_signal(
                        stop_sig, nav=self._portfolio.nav, adjusted_qty=qty
                    )
                    self._peak_price.pop(bar.symbol, None)  # reset for next entry
                    logger.info(
                        "RISK EXIT [%s] %s  qty=%d  pnl_pct=%.1f%%",
                        exit_reason.split()[0], bar.symbol, qty, pos.pnl_pct * 100,
                    )

            # Dispatch bar to strategies
            signals = self._registry.dispatch_bar(bar)

            # Optionally filter signals through macro layer
            if self._macro_filter is not None:
                filtered = []
                for sig in signals:
                    filtered_sig = self._macro_filter.filter_signal(sig)
                    if filtered_sig is not None:
                        filtered.append(filtered_sig)
                signals = filtered

            # Process signals through regime gate, risk gate and order manager
            multiplier = self._regime_detector.signal_multiplier
            for signal in signals:
                # Scale confidence by market regime
                adj_conf = min(1.0, signal.confidence * multiplier)

                # Apply RL bandit weight: strategies with better recent P&L get
                # a confidence boost; under-performers get a small penalty.
                # Weight is 1.0 at cold-start so it has no effect until trades close.
                bandit_weight = self._bandit.get_weight(signal.strategy_id, symbol=signal.symbol)
                adj_conf = min(1.0, adj_conf * bandit_weight)

                # Suppress new longs in RISK_OFF unless model is very confident
                if (
                    self._regime_detector.is_risk_off()
                    and signal.signal_type == SignalType.BUY
                    and adj_conf < 0.70
                ):
                    logger.debug(
                        "REGIME suppressed BUY %s (RISK_OFF, conf=%.2f)",
                        signal.symbol, adj_conf,
                    )
                    continue

                # Suppress BUY when price is below symbol's MA50 + 1% buffer (trend filter)
                if signal.signal_type == SignalType.BUY:
                    ma50 = self._ma50.get(signal.symbol, 0.0)
                    cur_price = self._current_prices.get(signal.symbol, 0.0)
                    if ma50 > 0 and cur_price < ma50 * 1.01:
                        logger.debug(
                            "MA50 suppressed BUY %s (price=%.2f < MA50*1.01=%.2f)",
                            signal.symbol, cur_price, ma50 * 1.01,
                        )
                        continue

                # Suppress BUY when cumulative realized P&L on this symbol is below cap.
                # Uses _symbol_cum_pnl (not positions dict) because closed positions
                # are deleted from _portfolio.positions and their history would be lost.
                if signal.signal_type == SignalType.BUY:
                    sym_realized = self._symbol_cum_pnl.get(signal.symbol, 0.0)
                    if sym_realized < self.config.symbol_loss_cap:
                        logger.debug(
                            "Loss-cap suppressed BUY %s (cum_realized=%.0f < cap=%.0f)",
                            signal.symbol, sym_realized, self.config.symbol_loss_cap,
                        )
                        continue

                decision: RiskDecision = self._risk_gate.evaluate(
                    signal, sim_time=bar.ts
                )
                if decision.approved:
                    # ATR-based position sizing
                    atr_pct = self._bar_atr_pct.get(bar.symbol, 0.01)
                    qty = atr_position_size(
                        nav        = self._portfolio.nav,
                        price      = bar.close,
                        atr_pct    = atr_pct,
                        confidence = adj_conf,
                    )
                    final_qty = decision.adjusted_quantity or qty
                    await self._order_manager.process_signal(
                        signal,
                        nav=self._portfolio.nav,
                        adjusted_qty=final_qty,
                    )

            # Record NAV once per day
            bar_date = bar.ts.date()
            if bar_date != prev_date:
                nav_records.append((bar.ts, self._portfolio.nav))
                self._risk_gate.record_nav(self._portfolio.nav, sim_date=bar_date)
                prev_date = bar_date

        await self._registry.stop_all()

        result = self._build_result(nav_records)
        logger.info("\n%s", result.summary())
        return result

    # ── Helpers ───────────────────────────────────────────────

    def _get_current_price(self, symbol: str) -> float:
        return self._current_prices.get(symbol, 0.0)

    def _update_atr(self, bar: Bar) -> float:
        """
        Maintain a 14-period rolling ATR (as fraction of close price) per symbol.
        Uses True Range = max(H-L, |H-prev_close|, |L-prev_close|).
        """
        prev = self._prev_close.get(bar.symbol, bar.close)
        tr_abs = max(
            bar.high - bar.low,
            abs(bar.high - prev),
            abs(bar.low  - prev),
        )
        tr_pct = tr_abs / (bar.close + 1e-10)

        buf = self._atr_history.setdefault(bar.symbol, [])
        buf.append(tr_pct)
        if len(buf) > 14:
            buf.pop(0)

        self._prev_close[bar.symbol] = bar.close
        return sum(buf) / len(buf)

    def _update_ma50(self, bar: Bar) -> None:
        """Maintain a 50-period simple moving average of close prices per symbol."""
        buf = self._ma50_buf.setdefault(bar.symbol, [])
        buf.append(bar.close)
        if len(buf) > 50:
            buf.pop(0)
        self._ma50[bar.symbol] = sum(buf) / len(buf)

    def _track_sym_pnl(self, order: Order) -> None:
        """
        Accumulate gross realized P&L per symbol from SELL fills.
        Also drives the RL bandit: records which strategy opened each position,
        then rewards/penalises that strategy when the position closes.
        Must be registered as a fill callback BEFORE portfolio.on_fill so that
        the Position object still exists and avg_cost is readable.
        """
        if order.side == OrderSide.BUY and order.filled_quantity > 0:
            # Record which strategy opened this position (first BUY wins)
            if order.symbol not in self._position_open_strat:
                self._position_open_strat[order.symbol] = order.strategy_id

        if order.side == OrderSide.SELL and order.filled_quantity > 0:
            pos = self._portfolio.positions.get(order.symbol)
            if pos is not None and pos.is_long:
                closing_qty = min(order.filled_quantity, pos.quantity)
                realized = (order.avg_fill_price - pos.avg_cost) * closing_qty
                self._symbol_cum_pnl[order.symbol] = (
                    self._symbol_cum_pnl.get(order.symbol, 0.0) + realized
                )
                # Update bandit with realized P&L for the opening strategy
                open_strat = self._position_open_strat.pop(order.symbol, order.strategy_id)
                self._bandit.update(open_strat, realized, symbol=order.symbol)
                logger.debug(
                    "Bandit ← [%s] realized=%.0f | weights=%s",
                    open_strat, realized,
                    {k: v["weight"] for k, v in self._bandit.weights_summary().items()},
                )

    def _build_result(self, nav_records: list) -> BacktestResult:
        nav_series = pd.Series(
            {ts: nav for ts, nav in nav_records}
        )

        filled = self._broker.filled_orders
        trades_data = [
            {
                "symbol":     o.symbol,
                "side":       o.side.value,
                "qty":        o.filled_quantity,
                "price":      o.avg_fill_price,
                "commission": o.commission,
                "strategy":   o.strategy_id,
                "time":       o.updated_at,
            }
            for o in filled
        ]
        trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

        # ── Sharpe ratio from daily NAV returns ──────────────
        sharpe = 0.0
        if len(nav_records) >= 2:
            navs = [n for _, n in nav_records]
            daily_rets = [(navs[i] - navs[i-1]) / navs[i-1] for i in range(1, len(navs))]
            if daily_rets:
                avg_r = sum(daily_rets) / len(daily_rets)
                var   = sum((r - avg_r) ** 2 for r in daily_rets) / len(daily_rets)
                std   = var ** 0.5
                if std > 0:
                    sharpe = (avg_r / std) * (252 ** 0.5)

        # ── Max drawdown from daily NAV ───────────────────────
        max_dd = 0.0
        if nav_records:
            navs = [n for _, n in nav_records]
            peak = navs[0]
            for n in navs:
                if n > peak:
                    peak = n
                dd = (n - peak) / peak
                if dd < max_dd:
                    max_dd = dd

        # ── Win / Loss analysis from matched round trips ──────
        # Group fills per symbol; match consecutive buys → sells (FIFO)
        win_pnls: list[float] = []
        loss_pnls: list[float] = []
        if not trades_df.empty:
            for sym, grp in trades_df.groupby("symbol"):
                grp = grp.sort_values("time").reset_index(drop=True)
                buy_queue: list[tuple[float, float]] = []   # (qty, price)
                for _, row in grp.iterrows():
                    qty_r   = float(row["qty"])
                    price_r = float(row["price"])
                    comm_r  = float(row["commission"])
                    if row["side"] == "BUY":
                        buy_queue.append((qty_r, price_r + comm_r / qty_r))
                    else:  # SELL
                        remaining = qty_r
                        pnl = -comm_r
                        while remaining > 0 and buy_queue:
                            bqty, bprice = buy_queue[0]
                            matched = min(remaining, bqty)
                            pnl     += matched * (price_r - bprice)
                            remaining -= matched
                            if matched < bqty:
                                buy_queue[0] = (bqty - matched, bprice)
                            else:
                                buy_queue.pop(0)
                        if pnl > 0:
                            win_pnls.append(pnl)
                        elif pnl < 0:
                            loss_pnls.append(pnl)

        num_wins  = len(win_pnls)
        num_loss  = len(loss_pnls)
        total_closed = num_wins + num_loss
        win_rate      = num_wins / total_closed if total_closed > 0 else 0.0
        avg_win       = sum(win_pnls)  / num_wins  if num_wins  > 0 else 0.0
        avg_loss      = sum(loss_pnls) / num_loss  if num_loss  > 0 else 0.0
        gross_profit  = sum(win_pnls)
        gross_loss    = abs(sum(loss_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        total_pnl = self._portfolio.total_pnl

        result = BacktestResult(
            config=self.config,
            nav_series=nav_series,
            trades=trades_df,
            total_pnl=total_pnl,
            total_pnl_pct=self._portfolio.total_pnl_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            num_trades=len(filled),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
        )
        return result


def _bar_to_tick(bar: Bar) -> Tick:
    return Tick(
        symbol=bar.symbol,
        ts=bar.ts,
        price=bar.close,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        prev_close=bar.open,
        volume=bar.volume,
        turnover=bar.turnover,
    )

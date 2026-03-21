"""
TradingEngine — the main orchestrator that ties everything together.

Flow:
    WatchlistSyncer → symbols
    TencentQuoteFetcher → Ticks → StrategyRegistry → Signals
                                                        ↓
                                                     RiskGate
                                                        ↓
                                                   OrderManager → Broker
                                                        ↓
                                                 PortfolioEngine
                                                        ↓
                                              AlertManager + Redis snapshot
"""
from __future__ import annotations

import asyncio
import json
import signal as _signal
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from myquant.config.logging_config import get_logger, setup_logging
from myquant.config.settings import settings
from myquant.data.fetchers.tencent_quote import TencentQuoteFetcher
from myquant.data.fetchers.watchlist_syncer import WatchlistSyncer
from myquant.data.store.redis_client import redis_client
from myquant.execution.brokers.base_broker import BaseBroker
from myquant.execution.brokers.futu_broker import FutuBroker
from myquant.execution.brokers.paper_broker import PaperBroker
from myquant.execution.brokers.web_broker import WebBroker
from myquant.execution.order_manager import OrderManager
from myquant.models.bar import Bar, BarInterval
from myquant.models.signal import Signal
from myquant.models.tick import Tick
from myquant.monitoring.alerts import alert_manager
from myquant.portfolio.portfolio_engine import PortfolioEngine
from myquant.risk.risk_gate import RiskGate
from myquant.strategy.registry import StrategyRegistry

logger = get_logger(__name__)

_SNAPSHOT_INTERVAL = 30   # seconds between portfolio snapshots to Redis
_HEARTBEAT_INTERVAL = 300  # seconds between heartbeat alerts
_DAILY_SUMMARY_HOUR = 15  # 3pm — send daily summary after market close


class TradingEngine:
    """
    The main event loop that orchestrates all subsystems.

    Usage:
        engine = TradingEngine()
        engine.add_strategy(MACrossoverStrategy(...))
        await engine.start()
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        broker: Optional[BaseBroker] = None,
        watchlist_cookie: Optional[str] = None,
    ) -> None:
        self._initial_cash = initial_cash

        # ── Subsystems ────────────────────────────────────────
        self._watchlist = WatchlistSyncer(cookie=watchlist_cookie)

        # Choose broker based on environment
        if broker is not None:
            self._broker = broker
        elif settings.IS_LIVE and settings.WEB_BROKER_USERNAME:
            self._broker = WebBroker()
            logger.info("LIVE mode: using WebBroker (browser automation)")
        elif settings.IS_LIVE:
            self._broker = FutuBroker()
            logger.info("LIVE mode: using FutuBroker")
        else:
            self._broker = PaperBroker(
                initial_cash=initial_cash,
                price_getter=self._get_current_price,
            )
            logger.info("PAPER mode: using PaperBroker")

        self._portfolio = PortfolioEngine(initial_cash=initial_cash)
        self._order_manager = OrderManager(broker=self._broker)
        self._order_manager.on_fill(self._on_fill)

        self._registry = StrategyRegistry()
        self._risk_gate = RiskGate(
            nav_getter=lambda: self._portfolio.nav,
            positions_getter=lambda: self._portfolio.positions,
        )

        self._fetcher: Optional[TencentQuoteFetcher] = None

        # Bar aggregation: {symbol → {interval → partial_bar}}
        self._bar_state: dict[str, dict] = defaultdict(dict)

        # Running state
        self._running = False
        self._last_snapshot = 0.0
        self._last_heartbeat = 0.0
        self._daily_summary_sent = False

        # Current prices
        self._current_prices: dict[str, float] = {}

    # ── Setup ─────────────────────────────────────────────────

    def add_strategy(self, strategy) -> "TradingEngine":
        self._registry.register(strategy)
        return self

    def set_sector_map(self, sector_map: dict[str, str]) -> "TradingEngine":
        self._risk_gate.set_sector_map(sector_map)
        return self

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self) -> None:
        setup_logging()
        logger.info("=" * 60)
        logger.info("  MyQuant TradingEngine starting [%s]", settings.ENV.upper())
        logger.info("=" * 60)

        # Connect infrastructure
        await redis_client.connect()
        await self._broker.connect()

        # Sync watchlist
        symbols = await self._watchlist.sync()
        logger.info("Trading universe: %d symbols", len(symbols))

        # Set daily nav baseline for drawdown
        self._risk_gate.record_nav(self._portfolio.nav)

        # Start strategies
        await self._registry.start_all()

        # Setup quote fetcher
        self._fetcher = TencentQuoteFetcher(symbols=symbols)
        self._fetcher.on_tick(self._on_tick)

        # Register graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (_signal.SIGINT, _signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        self._running = True
        await alert_manager.info(
            "Engine Started",
            f"Mode: {settings.ENV.upper()} | Universe: {len(symbols)} symbols",
        )

        # Run main loop
        await asyncio.gather(
            self._fetcher.start(),
            self._maintenance_loop(),
        )

    async def stop(self) -> None:
        logger.info("Shutting down TradingEngine...")
        self._running = False

        if self._fetcher:
            await self._fetcher.stop()

        # Cancel all open orders before disconnect
        await self._order_manager.cancel_all()

        await self._registry.stop_all()
        await self._broker.disconnect()
        await redis_client.disconnect()

        # Send final summary
        await alert_manager.daily_summary(self._portfolio.summary())
        logger.info("TradingEngine stopped cleanly.")

    # ── Event handlers ────────────────────────────────────────

    def _on_tick(self, tick: Tick) -> None:
        """Called on every incoming tick."""
        self._current_prices[tick.symbol] = tick.price
        self._portfolio.on_tick(tick)

        # Dispatch to strategies
        signals = self._registry.dispatch_tick(tick)
        for signal in signals:
            asyncio.create_task(self._process_signal(signal))

        # Aggregate tick into 1-min bar (simplified)
        self._aggregate_tick_to_bar(tick)

    def _on_fill(self, order) -> None:
        """Called when an order is filled."""
        self._portfolio.on_fill(order)
        asyncio.create_task(
            alert_manager.trade_fill(
                symbol=order.symbol,
                side=order.side.value,
                qty=order.filled_quantity,
                price=order.avg_fill_price,
                strategy=order.strategy_id,
                commission=order.commission,
            )
        )

    async def _process_signal(self, signal: Signal) -> None:
        """Route signal through risk gate then order manager."""
        decision = self._risk_gate.evaluate(signal)
        if not decision.approved:
            if "limit" in decision.reason.lower() or "breach" in decision.reason.lower():
                await alert_manager.risk_breach(
                    rule="Position/Sector Limit",
                    detail=f"{signal.symbol}: {decision.reason}",
                )
            return

        if "DAILY_DRAWDOWN" in decision.reason.upper():
            await alert_manager.risk_breach("Daily Drawdown Halt", decision.reason)
            return

        await self._order_manager.process_signal(
            signal,
            nav=self._portfolio.nav,
            adjusted_qty=decision.adjusted_quantity,
        )

    # ── Bar aggregation (1-min from ticks) ───────────────────

    def _aggregate_tick_to_bar(self, tick: Tick) -> None:
        """Build 1-minute bars from ticks and dispatch to strategies."""
        minute_key = tick.ts.replace(second=0, microsecond=0)
        sym = tick.symbol

        state = self._bar_state[sym]
        current_key = state.get("key")

        if current_key != minute_key:
            # Flush completed bar
            if current_key is not None:
                bar = Bar(
                    symbol=sym,
                    ts=current_key,
                    interval=BarInterval.M1,
                    open=state["open"],
                    high=state["high"],
                    low=state["low"],
                    close=state["close"],
                    volume=state["volume"],
                    is_complete=True,
                )
                signals = self._registry.dispatch_bar(bar)
                for signal in signals:
                    asyncio.create_task(self._process_signal(signal))

            # Start new bar
            self._bar_state[sym] = {
                "key":    minute_key,
                "open":   tick.price,
                "high":   tick.price,
                "low":    tick.price,
                "close":  tick.price,
                "volume": tick.volume,
            }
        else:
            # Update running bar
            state["high"]   = max(state["high"],  tick.price)
            state["low"]    = min(state["low"],   tick.price)
            state["close"]  = tick.price
            state["volume"] = tick.volume

    # ── Maintenance loop ──────────────────────────────────────

    async def _maintenance_loop(self) -> None:
        """Background tasks: snapshots, heartbeats, daily summary."""
        import time

        while self._running:
            now = time.monotonic()
            real_now = datetime.now()

            # Portfolio snapshot to Redis
            if now - self._last_snapshot > _SNAPSHOT_INTERVAL:
                try:
                    summary = self._portfolio.summary()
                    await redis_client.set_state("portfolio_snapshot", summary)
                    self._portfolio.snapshot()
                    self._last_snapshot = now
                except Exception as exc:
                    logger.warning("Snapshot failed: %s", exc)

            # Heartbeat alert
            if now - self._last_heartbeat > _HEARTBEAT_INTERVAL:
                await alert_manager.heartbeat("TradingEngine")
                self._last_heartbeat = now

            # Daily summary (after market close)
            if (
                real_now.hour == _DAILY_SUMMARY_HOUR
                and real_now.minute >= 5
                and not self._daily_summary_sent
            ):
                await alert_manager.daily_summary(self._portfolio.summary())
                self._daily_summary_sent = True
            elif real_now.hour < _DAILY_SUMMARY_HOUR:
                self._daily_summary_sent = False

            await asyncio.sleep(5)

    # ── Helpers ───────────────────────────────────────────────

    def _get_current_price(self, symbol: str) -> float:
        return self._current_prices.get(symbol, 0.0)

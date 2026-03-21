"""
Monitoring & Alerting — WeChat Work (企业微信) webhook + log alerts.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional

import aiohttp

from myquant.config.logging_config import get_logger
from myquant.config.settings import settings

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


_LEVEL_EMOJI = {
    AlertLevel.INFO:     "ℹ️",
    AlertLevel.WARNING:  "⚠️",
    AlertLevel.CRITICAL: "🚨",
}


class AlertManager:
    """
    Sends structured alerts to:
        - WeChat Work (企业微信) webhook
        - Python logger (always)
    """

    def __init__(self, webhook_url: Optional[str] = None) -> None:
        self._webhook = webhook_url or settings.WECHAT_WORK_WEBHOOK
        self._enabled = bool(self._webhook)
        if not self._enabled:
            logger.warning("AlertManager: No webhook configured — alerts will only log.")

    # ── Public methods ────────────────────────────────────────

    async def info(self, title: str, message: str = "") -> None:
        await self._send(AlertLevel.INFO, title, message)

    async def warning(self, title: str, message: str = "") -> None:
        await self._send(AlertLevel.WARNING, title, message)

    async def critical(self, title: str, message: str = "") -> None:
        await self._send(AlertLevel.CRITICAL, title, message)

    async def trade_fill(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        strategy: str,
        commission: float,
    ) -> None:
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        await self._send(
            AlertLevel.INFO,
            f"{emoji} FILL: {side.upper()} {qty}×{symbol}",
            (
                f"Price: {price:.4f}\n"
                f"Notional: {price * qty:,.2f}\n"
                f"Commission: {commission:.2f}\n"
                f"Strategy: {strategy}"
            ),
        )

    async def risk_breach(self, rule: str, detail: str) -> None:
        await self._send(
            AlertLevel.CRITICAL,
            f"🛑 RISK BREACH: {rule}",
            detail,
        )

    async def daily_summary(self, portfolio_summary: dict) -> None:
        nav  = portfolio_summary.get("nav", 0)
        pnl  = portfolio_summary.get("total_pnl", 0)
        pct  = portfolio_summary.get("total_pnl_pct", "0%")
        dd   = portfolio_summary.get("current_drawdown", "0%")
        ntrd = portfolio_summary.get("total_trades", 0)
        npos = portfolio_summary.get("open_positions", 0)

        direction = "📈" if pnl >= 0 else "📉"
        await self._send(
            AlertLevel.INFO,
            f"{direction} Daily Summary — {datetime.now().strftime('%Y-%m-%d')}",
            (
                f"NAV: {nav:,.2f}\n"
                f"PnL: {pnl:+,.2f} ({pct})\n"
                f"Drawdown: {dd}\n"
                f"Trades: {ntrd} | Positions: {npos}"
            ),
        )

    async def heartbeat(self, engine_name: str) -> None:
        await self._send(
            AlertLevel.INFO,
            f"💓 Heartbeat: {engine_name}",
            f"Engine running @ {datetime.now().strftime('%H:%M:%S')}",
        )

    # ── Internal ─────────────────────────────────────────────

    async def _send(self, level: AlertLevel, title: str, message: str) -> None:
        emoji = _LEVEL_EMOJI[level]
        full_msg = f"{emoji} **{title}**"
        if message:
            full_msg += f"\n{message}"

        # Always log
        log_fn = logger.warning if level == AlertLevel.WARNING else (
            logger.critical if level == AlertLevel.CRITICAL else logger.info
        )
        log_fn("ALERT [%s] %s: %s", level.value.upper(), title, message)

        # Send webhook
        if self._enabled and self._webhook:
            payload = {
                "msgtype": "markdown",
                "markdown": {"content": full_msg},
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._webhook,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("Webhook returned HTTP %d", resp.status)
            except Exception as exc:
                logger.warning("Failed to send webhook alert: %s", exc)


# Singleton
alert_manager = AlertManager()

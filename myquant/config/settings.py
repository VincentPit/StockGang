"""
Central settings — loaded once at startup from environment variables / .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


def _required(key: str) -> str:
    val = os.getenv(key)
    if val is None:
        raise RuntimeError(f"Required environment variable '{key}' is not set.")
    return val


def _get(key: str, default: str) -> str:
    return os.getenv(key, default)


class Settings:
    # ── General ──────────────────────────────────────────────
    ENV: str = _get("ENV", "development")
    IS_LIVE: bool = ENV == "live"
    IS_PAPER: bool = ENV == "paper"

    # ── Redis ─────────────────────────────────────────────────
    REDIS_HOST: str = _get("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(_get("REDIS_PORT", "6379"))
    REDIS_DB: int = int(_get("REDIS_DB", "0"))

    # ── PostgreSQL ────────────────────────────────────────────
    POSTGRES_HOST: str = _get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(_get("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = _get("POSTGRES_DB", "myquant")
    POSTGRES_USER: str = _get("POSTGRES_USER", "myquant")
    POSTGRES_PASSWORD: str = _get("POSTGRES_PASSWORD", "changeme")

    @property
    def POSTGRES_DSN(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Data Sources ──────────────────────────────────────────
    TUSHARE_TOKEN: str = _get("TUSHARE_TOKEN", "")

    # ── Futu OpenAPI ─────────────────────────────────────────
    FUTU_HOST: str = _get("FUTU_HOST", "127.0.0.1")
    FUTU_PORT: int = int(_get("FUTU_PORT", "11111"))
    FUTU_TRADE_PASSWORD: str = _get("FUTU_TRADE_PASSWORD", "")
    FUTU_TRADE_PASSWORD_MD5: str = _get("FUTU_TRADE_PASSWORD_MD5", "")

    # ── Web Broker (Playwright) ───────────────────────────────
    WEB_BROKER_URL: str      = _get("WEB_BROKER_URL", "https://trade.zszq.com/h5/")
    WEB_BROKER_USERNAME: str = _get("WEB_BROKER_USERNAME", "")
    WEB_BROKER_PASSWORD: str = _get("WEB_BROKER_PASSWORD", "")
    WEB_BROKER_HEADLESS: bool = _get("WEB_BROKER_HEADLESS", "true").lower() == "true"

    # ── IBKR ─────────────────────────────────────────────────
    IBKR_HOST: str = _get("IBKR_HOST", "127.0.0.1")
    IBKR_PORT: int = int(_get("IBKR_PORT", "7497"))
    IBKR_CLIENT_ID: int = int(_get("IBKR_CLIENT_ID", "1"))

    # ── Alerts ───────────────────────────────────────────────
    WECHAT_WORK_WEBHOOK: str = _get("WECHAT_WORK_WEBHOOK", "")

    # ── Risk ─────────────────────────────────────────────────
    MAX_POSITION_PCT: float = float(_get("MAX_POSITION_PCT", "0.20"))
    MAX_SECTOR_PCT: float = float(_get("MAX_SECTOR_PCT", "0.40"))
    DAILY_DRAWDOWN_LIMIT: float = float(_get("DAILY_DRAWDOWN_LIMIT", "-0.03"))
    VAR_LIMIT: float = float(_get("VAR_LIMIT", "0.02"))
    MAX_ORDERS_PER_MINUTE: int = int(_get("MAX_ORDERS_PER_MINUTE", "10"))

    # ── Market Hours (CST = UTC+8) ───────────────────────────
    MARKET_OPEN_MORNING: str = "09:30"
    MARKET_CLOSE_MORNING: str = "11:30"
    MARKET_OPEN_AFTERNOON: str = "13:00"
    MARKET_CLOSE_AFTERNOON: str = "15:00"
    PRE_MARKET_AUCTION: str = "09:15"


settings = Settings()

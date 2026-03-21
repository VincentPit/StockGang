"""
main.py — Entry point for the live / paper trading engine.

Usage:
    # Paper trading (default)
    python main.py

    # Live trading (requires LIVE broker credentials in .env)
    ENV=live python main.py
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).parent))

from myquant.config.logging_config import setup_logging
from myquant.config.settings import settings
from myquant.engine.trading_engine import TradingEngine
from myquant.strategy.technical.ma_crossover import MACrossoverStrategy
from myquant.strategy.technical.macd_strategy import MACDStrategy
from myquant.strategy.technical.rsi_strategy import RSIStrategy


async def main() -> None:
    setup_logging()

    # ── Define the trading universe ───────────────────────────
    # These will be overridden by 腾讯自选股 watchlist if cookie is set.
    # Used as fallback for development.
    universe = [
        # HK Blue Chips
        "hk00700",   # 腾讯
        "hk09988",   # 阿里巴巴
        "hk03690",   # 美团
        "hk02318",   # 中国平安
        # A-Share Blue Chips
        "sh600036",  # 招商银行
        "sh600519",  # 贵州茅台
        "sh601318",  # 中国平安A
        "sz000858",  # 五粮液
        # US Tech
        "usTSLA",
        "usAAPL",
        "usNVDA",
    ]

    # ── Build engine ─────────────────────────────────────────
    engine = TradingEngine(
        initial_cash=1_000_000.0,
        # watchlist_cookie="YOUR_QQ_COOKIE_HERE",  # Optional: pull real 自选股
    )

    # ── Sector map for risk management ───────────────────────
    engine.set_sector_map({
        "hk00700": "科技", "hk09988": "科技", "hk03690": "科技",
        "hk02318": "金融", "sh600036": "金融", "sh601318": "金融",
        "sh600519": "消费", "sz000858": "消费",
        "usTSLA": "美股", "usAAPL": "美股", "usNVDA": "美股",
    })

    # ── Register strategies ───────────────────────────────────

    # Strategy 1: MA Crossover on HK blue chips (daily-ish, 10/30 EMA)
    engine.add_strategy(
        MACrossoverStrategy(
            strategy_id="ma_cross_hk",
            symbols=["hk00700", "hk09988", "hk03690"],
            fast_period=10,
            slow_period=30,
        )
    )

    # Strategy 2: RSI mean reversion on A-shares
    engine.add_strategy(
        RSIStrategy(
            strategy_id="rsi_ashare",
            symbols=["sh600036", "sh600519", "sz000858"],
            period=14,
            oversold=30,
            overbought=70,
        )
    )

    # Strategy 3: MACD on US tech
    engine.add_strategy(
        MACDStrategy(
            strategy_id="macd_us",
            symbols=["usTSLA", "usAAPL", "usNVDA"],
            fast=12,
            slow=26,
            signal=9,
        )
    )

    # ── Start ─────────────────────────────────────────────────
    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())

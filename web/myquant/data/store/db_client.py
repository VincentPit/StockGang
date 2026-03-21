"""
Async PostgreSQL client using SQLAlchemy + asyncpg.
Stores orders, positions, fills, and performance records.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    UniqueConstraint, text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from myquant.config.logging_config import get_logger
from myquant.config.settings import settings

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class OrderRecord(Base):
    __tablename__ = "orders"

    id              = Column(String(36), primary_key=True)
    symbol          = Column(String(20), nullable=False, index=True)
    side            = Column(String(4), nullable=False)
    order_type      = Column(String(12), nullable=False)
    quantity        = Column(Integer, nullable=False)
    limit_price     = Column(Float, default=0.0)
    filled_quantity = Column(Integer, default=0)
    avg_fill_price  = Column(Float, default=0.0)
    commission      = Column(Float, default=0.0)
    status          = Column(String(12), nullable=False, index=True)
    strategy_id     = Column(String(64), index=True)
    signal_id       = Column(String(36))
    broker_order_id = Column(String(64))
    notes           = Column(Text, default="")
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PositionRecord(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("symbol", name="uq_position_symbol"),)

    id             = Column(Integer, primary_key=True, autoincrement=True)
    symbol         = Column(String(20), nullable=False)
    quantity       = Column(Integer, default=0)
    avg_cost       = Column(Float, default=0.0)
    realized_pnl   = Column(Float, default=0.0)
    market_price   = Column(Float, default=0.0)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PerformanceRecord(Base):
    __tablename__ = "performance"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    date       = Column(DateTime, nullable=False, index=True)
    nav        = Column(Float, nullable=False)
    daily_pnl  = Column(Float, default=0.0)
    cum_pnl    = Column(Float, default=0.0)
    drawdown   = Column(Float, default=0.0)
    num_trades = Column(Integer, default=0)


# ── DB Client ─────────────────────────────────────────────────────────────────

class DBClient:
    def __init__(self) -> None:
        self._engine = None
        self._session_factory = None

    async def connect(self) -> None:
        self._engine = create_async_engine(
            settings.POSTGRES_DSN,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database connected and tables verified.")

    async def disconnect(self) -> None:
        if self._engine:
            await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    async def health_check(self) -> bool:
        try:
            async with self.session() as sess:
                await sess.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


# Singleton
db_client = DBClient()

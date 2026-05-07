"""
myquant/db/session.py — Sessionmaker factories.

Use ``AsyncSessionLocal`` from FastAPI handlers and async workers; use
``SessionLocal`` from synchronous thread-pool callers (api/db.py shim).
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker

from .engine import async_engine, sync_engine

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    autoflush=False,
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
    autoflush=False,
    future=True,
)

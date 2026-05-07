"""
myquant/db/models.py — SQLAlchemy declarative ORM models.

Mirrors the original SQLite schema in api/db.py 1:1 so the migration is
behaviour-preserving. Storage decisions worth noting:

  * ``payload``, ``value``, ``feature_cols`` stay as TEXT (JSON-encoded by
    callers) rather than JSONB — the cutover stays semantically identical to
    the SQLite layer. A later migration can promote these to JSONB.
  * ``trained_at`` / ``created_at`` / ``updated_at`` / ``expires_at`` stay as
    DOUBLE PRECISION (Unix epoch seconds) — every caller passes ``time.time()``.
    Promoting to TIMESTAMPTZ is a follow-up.
  * ``model_blob`` uses ``LargeBinary`` → BYTEA on Postgres.
"""
from __future__ import annotations

from sqlalchemy import (
    Float,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base — its metadata is the Alembic target."""


class Job(Base):
    __tablename__ = "jobs"

    id:         Mapped[str]   = mapped_column(String, primary_key=True)
    kind:       Mapped[str]   = mapped_column(String, nullable=False)
    status:     Mapped[str]   = mapped_column(String, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)
    payload:    Mapped[str]   = mapped_column(Text,  nullable=False, default="{}")

    __table_args__ = (
        Index("idx_jobs_kind",    "kind"),
        Index("idx_jobs_status",  "status"),
        Index("idx_jobs_updated", "updated_at"),
    )


class Cache(Base):
    __tablename__ = "cache"

    cache_key:  Mapped[str]   = mapped_column(String, primary_key=True)
    value:      Mapped[str]   = mapped_column(Text,  nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("idx_cache_exp", "expires_at"),
    )


class PaperAccount(Base):
    __tablename__ = "paper_account"

    id:   Mapped[int]   = mapped_column(Integer, primary_key=True, default=1)
    cash: Mapped[float] = mapped_column(Float,   nullable=False)


class PaperPosition(Base):
    __tablename__ = "paper_positions"

    symbol:    Mapped[str]   = mapped_column(String,  primary_key=True)
    qty:       Mapped[int]   = mapped_column(Integer, nullable=False)
    avg_price: Mapped[float] = mapped_column(Float,   nullable=False, default=0.0)


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id:            Mapped[str]           = mapped_column(String,      primary_key=True)
    symbol:        Mapped[str]           = mapped_column(String,      nullable=False)
    strategy_id:   Mapped[str]           = mapped_column(String,      nullable=False, default="lgbm_core")
    trained_at:    Mapped[float]         = mapped_column(Float,       nullable=False)
    bar_count:     Mapped[int]           = mapped_column(Integer,     nullable=False)
    last_bar_date: Mapped[str]           = mapped_column(String,      nullable=False)
    oos_accuracy:  Mapped[float | None]  = mapped_column(Float,       nullable=True)
    model_blob:    Mapped[bytes]         = mapped_column(LargeBinary, nullable=False)
    feature_cols:  Mapped[str]           = mapped_column(Text,        nullable=False)

    __table_args__ = (
        Index("idx_models_symbol",  "symbol"),
        Index("idx_models_trained", "trained_at"),
    )

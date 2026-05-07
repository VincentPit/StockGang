"""initial schema — jobs, cache, paper_account, paper_positions, trained_models

Revision ID: 0001
Revises:
Create Date: 2026-05-06

Captures the schema previously created by api/db.py's ``init_db()`` against
SQLite, now in Postgres. Column types map straight across (REAL → DOUBLE
PRECISION, BLOB → BYTEA, TEXT → TEXT) so existing callers see no semantic
difference. ``payload`` / ``value`` / ``feature_cols`` stay as TEXT (callers
JSON-encode); promoting to JSONB is a follow-up.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id",         sa.String(),  nullable=False),
        sa.Column("kind",       sa.String(),  nullable=False),
        sa.Column("status",     sa.String(),  nullable=False),
        sa.Column("created_at", sa.Float(),   nullable=False),
        sa.Column("updated_at", sa.Float(),   nullable=False),
        sa.Column("payload",    sa.Text(),    nullable=False, server_default=sa.text("'{}'")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_jobs_kind",    "jobs", ["kind"])
    op.create_index("idx_jobs_status",  "jobs", ["status"])
    op.create_index("idx_jobs_updated", "jobs", ["updated_at"])

    op.create_table(
        "cache",
        sa.Column("cache_key",  sa.String(), nullable=False),
        sa.Column("value",      sa.Text(),   nullable=False),
        sa.Column("expires_at", sa.Float(),  nullable=False),
        sa.PrimaryKeyConstraint("cache_key"),
    )
    op.create_index("idx_cache_exp", "cache", ["expires_at"])

    op.create_table(
        "paper_account",
        sa.Column("id",   sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("cash", sa.Float(),   nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "paper_positions",
        sa.Column("symbol",    sa.String(),  nullable=False),
        sa.Column("qty",       sa.Integer(), nullable=False),
        sa.Column("avg_price", sa.Float(),   nullable=False, server_default=sa.text("0.0")),
        sa.PrimaryKeyConstraint("symbol"),
    )

    op.create_table(
        "trained_models",
        sa.Column("id",            sa.String(),      nullable=False),
        sa.Column("symbol",        sa.String(),      nullable=False),
        sa.Column("strategy_id",   sa.String(),      nullable=False, server_default=sa.text("'lgbm_core'")),
        sa.Column("trained_at",    sa.Float(),       nullable=False),
        sa.Column("bar_count",     sa.Integer(),     nullable=False),
        sa.Column("last_bar_date", sa.String(),      nullable=False),
        sa.Column("oos_accuracy",  sa.Float(),       nullable=True),
        sa.Column("model_blob",    sa.LargeBinary(), nullable=False),
        sa.Column("feature_cols",  sa.Text(),        nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_models_symbol",  "trained_models", ["symbol"])
    op.create_index("idx_models_trained", "trained_models", ["trained_at"])


def downgrade() -> None:
    op.drop_index("idx_models_trained", table_name="trained_models")
    op.drop_index("idx_models_symbol",  table_name="trained_models")
    op.drop_table("trained_models")

    op.drop_table("paper_positions")
    op.drop_table("paper_account")

    op.drop_index("idx_cache_exp", table_name="cache")
    op.drop_table("cache")

    op.drop_index("idx_jobs_updated", table_name="jobs")
    op.drop_index("idx_jobs_status",  table_name="jobs")
    op.drop_index("idx_jobs_kind",    table_name="jobs")
    op.drop_table("jobs")

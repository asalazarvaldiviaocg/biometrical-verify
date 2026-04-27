"""per-user KEK + replay-protected challenge nonces

Adds:
  * user_keks  — per-user wrapped KEK for crypto-shred (GDPR Art. 17).
  * issued_nonces  — server-issued challenge nonces with TTL + one-time use,
    fixing the anti-replay gap where /submit accepted any nonce string.

Revision ID: 0002_user_keks_and_nonces
Revises:    0001_initial
Create Date: 2026-04-26
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002_user_keks_and_nonces"
down_revision: str | None = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_keks",
        sa.Column("user_id", sa.String(64), primary_key=True),
        sa.Column("wrapped_kek", sa.LargeBinary, nullable=False),
        sa.Column("kms_key_id", sa.String(128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "issued_nonces",
        sa.Column("nonce", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False, index=True),
        sa.Column("challenge", sa.String(32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_issued_nonces_expires_at", "issued_nonces", ["expires_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_issued_nonces_expires_at", table_name="issued_nonces")
    op.drop_table("issued_nonces")
    op.drop_table("user_keks")

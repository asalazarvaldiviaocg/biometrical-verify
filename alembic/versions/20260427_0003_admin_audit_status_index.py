"""admin audit trail + verification status/created index

Adds:
  * admin_audit_events  — privileged-operation audit log (actor / action /
    target / detail / at). LFPDPPP Art. 22 trace of crypto-shred + future
    admin actions.
  * ix_verifications_status_created  — index needed by the stuck-queued
    sweeper task to scan only `status='queued' AND created_at < cutoff`
    without a sequential scan.

Revision ID: 0003_admin_audit_status_index
Revises:    0002_user_keks_and_nonces
Create Date: 2026-04-27
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003_admin_audit_status_index"
down_revision: str | None = "0002_user_keks_and_nonces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "admin_audit_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("actor", sa.String(64), nullable=False, index=True),
        sa.Column("action", sa.String(64), nullable=False, index=True),
        sa.Column("target", sa.String(128), nullable=True),
        sa.Column("detail", sa.JSON, nullable=True),
        sa.Column(
            "at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            index=True,
        ),
    )

    # Sweeper-friendly composite index: the requeue-stuck-verifications task
    # filters by (status='queued', created_at < cutoff). PG can use this even
    # without statistics on freshly seeded data.
    op.create_index(
        "ix_verifications_status_created",
        "verifications",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_verifications_status_created", table_name="verifications")
    op.drop_table("admin_audit_events")

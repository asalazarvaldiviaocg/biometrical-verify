"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-21
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "verifications",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False, index=True),
        sa.Column("contract_id", sa.String(64), nullable=False, index=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("decision", sa.String(16), nullable=True),
        sa.Column("challenge", sa.String(32), nullable=False, server_default="blink_twice"),
        sa.Column("nonce", sa.String(64), nullable=True),
        sa.Column("sha_id", sa.String(64), nullable=True),
        sa.Column("sha_video", sa.String(64), nullable=True),
        sa.Column("similarity", sa.Float, nullable=True),
        sa.Column("distance", sa.Float, nullable=True),
        sa.Column("threshold", sa.Float, nullable=True),
        sa.Column("liveness_score", sa.Float, nullable=True),
        sa.Column("challenge_passed", sa.Boolean, nullable=True),
        sa.Column("deepfake_suspicious", sa.Boolean, nullable=True),
        sa.Column("model", sa.String(32), nullable=True),
        sa.Column("reason", sa.String(256), nullable=True),
        sa.Column("id_blob_ref", postgresql.JSONB, nullable=True),
        sa.Column("video_blob_ref", postgresql.JSONB, nullable=True),
        sa.Column("receipt", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_verifications_user_status", "verifications", ["user_id", "status"]
    )

    op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "verification_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("verifications.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event", sa.String(64), nullable=False),
        sa.Column("detail", sa.JSON, nullable=True),
        sa.Column(
            "at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
    )
    op.create_index("ix_audit_events_verification_id", "audit_events", ["verification_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_verification_id", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_verifications_user_status", table_name="verifications")
    op.drop_table("verifications")

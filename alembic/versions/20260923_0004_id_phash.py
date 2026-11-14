"""ID perceptual hash for duplicate-AI-ID detection

Adds:
  * verifications.id_phash  — 64-bit DCT perceptual hash (hex) of the ID
    image, written by the pipeline's forensics stage. The duplicate check
    queries same-hash rows under different accounts to flag reused
    AI-generated outputs and shared fraud-ring photos.
  * ix_verifications_id_phash — equality lookup for that check.

Revision ID: 0004_id_phash
Revises:    0003_admin_audit_status_index
Create Date: 2026-09-23
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.alembic_idem import add_column, create_index, create_table

revision: str = "0004_id_phash"
down_revision: str | None = "0003_admin_audit_status_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    add_column(
        "verifications", sa.Column("id_phash", sa.String(16), nullable=True)
    )
    create_index("ix_verifications_id_phash", "verifications", ["id_phash"])


def downgrade() -> None:
    op.drop_index("ix_verifications_id_phash", table_name="verifications")
    op.drop_column("verifications", "id_phash")

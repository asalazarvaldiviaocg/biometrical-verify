"""Idempotent DDL helpers for the migration series.

Alembic's op.create_table / op.add_column / op.create_index emit plain DDL,
which fails when the target already exists. That matters here because
alembic_version is the only record of what has run: lose that one row - a
restored snapshot, a partially-applied deploy, a wiped bookkeeping table -
and `alembic upgrade head` replays the whole series against a populated
schema and dies on the first CREATE TABLE. It happened: the schema was
there, the version row was not, and recovery meant hand-crafting the row.

Every DDL in this series goes through these helpers, so a replay is a no-op
rather than an error. The Laundex API's migrations are idempotent for the
same reason; this brings the engine to the same standard.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


def _inspector():
    return sa.inspect(op.get_bind())


def create_table(name: str, *columns, **kw) -> None:
    if _inspector().has_table(name):
        return
    op.create_table(name, *columns, **kw)


def add_column(table: str, column: sa.Column) -> None:
    if column.name in {c["name"] for c in _inspector().get_columns(table)}:
        return
    op.add_column(table, column)


def create_index(index_name: str, table: str, columns, **kw) -> None:
    if index_name in {i["name"] for i in _inspector().get_indexes(table)}:
        return
    op.create_index(index_name, table, columns, **kw)
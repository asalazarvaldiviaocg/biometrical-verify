from __future__ import annotations

import os
import sys
from logging.config import fileConfig

# The idempotent-DDL helpers live beside this file, and the version scripts
# import them by module name. Alembic imports version scripts itself, with
# no package context, so their directory has to be importable for the
# duration of a migration run - which is exactly the only place they run.
sys.path.insert(0, os.path.dirname(__file__))

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.models.base import Base
import app.models.verification  # noqa: F401  (register tables)

config = context.config

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

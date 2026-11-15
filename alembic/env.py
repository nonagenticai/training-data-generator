"""Alembic environment configuration for Training Data Generator.

The migrate init-container runs ``alembic upgrade head`` as the schema source
of truth; the app's runtime ``create_all`` is best-effort belt-and-suspenders.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add the service's src/ to the path so ``from models...`` resolves both in the
# image (PYTHONPATH=/app/src) and for local ``alembic`` invocations.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import models — Base.metadata is the autogenerate target.
from models.db_models import Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate
target_metadata = Base.metadata

# Resolve the database URL from the environment. The migrate init-container sets
# a bare ``DATABASE_URL``; the service itself uses the prefixed
# ``TRAINING_DATA_GEN_DATABASE_URL`` — accept either so both paths work.
# alembic.ini carries no DSN: a localhost default silently hits whatever runs on
# that port on a dev box.
database_url = os.getenv("DATABASE_URL") or os.getenv("TRAINING_DATA_GEN_DATABASE_URL")
if not database_url:
    raise RuntimeError(
        "alembic: set DATABASE_URL (or TRAINING_DATA_GEN_DATABASE_URL); there is no default "
        "DSN. For a local run start an ephemeral docker postgres:18 and point it there."
    )
# Normalize async/bare DSNs to the sync psycopg3 driver alembic runs under —
# mirrors settings.model_post_init so the init-container's DSN works verbatim.
if database_url.startswith("postgresql+asyncpg://"):
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
elif database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DBAPI connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

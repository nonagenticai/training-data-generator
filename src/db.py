"""Async database setup using SQLAlchemy async engine.

Provides init_db / dispose_db lifecycle functions and a get_session
async context manager for obtaining async sessions.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from logging_config import get_logger

logger = get_logger("db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(database_url: str) -> None:
    """Create the async engine, session factory, and ensure all tables exist."""
    global _engine, _session_factory  # noqa: PLW0603

    # NullPool: pgbouncer (transaction-mode, default_pool_size=2) is the
    # connection pool. App-side pooling on top would multiplex sessions
    # through a shared backend and break session-scoped state. Every service
    # that fronts via pgbouncer.example.com:6432 uses
    # NullPool — see ingestion deployment for the reference pattern.
    # pool_recycle dropped: NullPool closes connections per-checkout so
    # there's nothing to recycle.
    _engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    from models.db_models import Base

    # The alembic `migrate` init-container (running `alembic upgrade head`) is
    # the schema source of truth in the cluster. This create_all is best-effort
    # belt-and-suspenders so local dev without a manual `alembic upgrade` still
    # self-schemas; create_all is checkfirst=True, so it no-ops when the table
    # already exists. Under a restricted (no-DDL) runtime role it would raise
    # permission-denied — downgrade that to a warning rather than crashlooping
    # startup, since the init-container has already created the table.
    try:
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except (SQLAlchemyError, OSError) as exc:
        logger.warning(
            "create_all skipped — the alembic migrate init-container is the "
            "schema source of truth; create_all is best-effort: %s",
            exc,
        )


async def dispose_db() -> None:
    """Dispose of the async engine, releasing all connections."""
    global _engine, _session_factory  # noqa: PLW0603

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def check_db_connectivity() -> bool:
    """Execute SELECT 1 to verify the database connection is alive."""
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except (SQLAlchemyError, OSError) as exc:
        # SQLAlchemyError covers DBAPI/driver errors (psycopg, pool exhaustion,
        # operational errors); OSError covers raw socket failures that escape
        # before psycopg wraps them (e.g. ECONNREFUSED on a fresh pool).
        logger.warning("db_connectivity_check_failed", error=str(exc), exc_info=True)
        return False


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async SQLAlchemy session, committing on success or rolling back on error."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            logger.error("db_session_rollback", exc_info=True)
            await session.rollback()
            raise

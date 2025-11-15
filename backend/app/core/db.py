"""
T.A.R.S. Database Configuration
PostgreSQL connection and session management
Phase 6 - Production Scaling & Monitoring
"""

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base for models
Base = declarative_base()

# Global engine and session maker
async_engine: AsyncEngine = None
async_session_maker: async_sessionmaker = None


def get_database_url() -> str:
    """
    Get PostgreSQL database URL from settings.

    Returns:
        Database connection URL
    """
    if not getattr(settings, 'POSTGRES_ENABLED', False):
        return None

    return (
        f"postgresql+asyncpg://"
        f"{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
        f"/{settings.POSTGRES_DB}"
    )


async def init_db() -> bool:
    """
    Initialize database connection and create tables.

    Returns:
        True if successful, False otherwise
    """
    global async_engine, async_session_maker

    if not getattr(settings, 'POSTGRES_ENABLED', False):
        logger.info("PostgreSQL disabled, skipping database initialization")
        return False

    try:
        db_url = get_database_url()
        logger.info(f"Connecting to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")

        # Create async engine
        async_engine = create_async_engine(
            db_url,
            echo=settings.FASTAPI_DEBUG,
            pool_size=getattr(settings, 'POSTGRES_POOL_SIZE', 20),
            max_overflow=getattr(settings, 'POSTGRES_MAX_OVERFLOW', 10),
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
        )

        # Create session maker
        async_session_maker = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )

        # Import models to register them
        from ..models.analytics_model import (
            QueryLog,
            DocumentAccess,
            ErrorLog
        )

        # Create tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


async def close_db():
    """Close database connections"""
    global async_engine

    if async_engine:
        await async_engine.dispose()
        logger.info("Database connections closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session (dependency injection).

    Yields:
        AsyncSession instance
    """
    if not async_session_maker:
        raise RuntimeError("Database not initialized")

    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session():
    """
    Get database session as context manager.

    Usage:
        async with get_db_session() as session:
            # Use session
    """
    if not async_session_maker:
        raise RuntimeError("Database not initialized")

    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def health_check() -> bool:
    """
    Check database health.

    Returns:
        True if database is healthy, False otherwise
    """
    if not async_engine:
        return False

    try:
        async with async_session_maker() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def get_sync_database_url() -> str:
    """
    Get synchronous PostgreSQL database URL (for Alembic migrations).

    Returns:
        Synchronous database connection URL
    """
    if not getattr(settings, 'POSTGRES_ENABLED', False):
        return None

    return (
        f"postgresql://"
        f"{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
        f"/{settings.POSTGRES_DB}"
    )

"""
Storage backends for coordination hub state persistence
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, List
import asyncpg
import asyncio

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend"""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to storage"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from storage"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        """Set a key-value pair"""
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key"""
        pass

    @abstractmethod
    async def list_keys(self, prefix: str) -> List[str]:
        """List keys with prefix"""
        pass


class PostgresStorage(StorageBackend):
    """PostgreSQL-backed storage"""

    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Connect to PostgreSQL"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )

            # Create table if not exists
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS federation_kv (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Create index on key prefix for fast prefix queries
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_federation_kv_prefix
                    ON federation_kv (key text_pattern_ops)
                """)

            logger.info("Connected to PostgreSQL storage backend")

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from PostgreSQL")

    async def set(self, key: str, value: Any) -> None:
        """Set a key-value pair"""
        if not self.pool:
            raise RuntimeError("Not connected to PostgreSQL")

        try:
            # Serialize value to JSON
            value_json = json.dumps(value)

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO federation_kv (key, value, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (key)
                    DO UPDATE SET value = $2, updated_at = NOW()
                """, key, value_json)

            logger.debug(f"Set {key} in PostgreSQL")

        except Exception as e:
            logger.error(f"Failed to set {key}: {e}")
            raise

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        if not self.pool:
            raise RuntimeError("Not connected to PostgreSQL")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT value FROM federation_kv WHERE key = $1
                """, key)

                if row:
                    return json.loads(row["value"])
                return None

        except Exception as e:
            logger.error(f"Failed to get {key}: {e}")
            raise

    async def delete(self, key: str) -> None:
        """Delete a key"""
        if not self.pool:
            raise RuntimeError("Not connected to PostgreSQL")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM federation_kv WHERE key = $1
                """, key)

            logger.debug(f"Deleted {key} from PostgreSQL")

        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            raise

    async def list_keys(self, prefix: str) -> List[str]:
        """List keys with prefix"""
        if not self.pool:
            raise RuntimeError("Not connected to PostgreSQL")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT key FROM federation_kv
                    WHERE key LIKE $1
                    ORDER BY key
                """, f"{prefix}%")

                return [row["key"] for row in rows]

        except Exception as e:
            logger.error(f"Failed to list keys with prefix {prefix}: {e}")
            raise


class EtcdStorage(StorageBackend):
    """Etcd-backed storage (stub implementation)"""

    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.client = None

    async def connect(self) -> None:
        """Connect to etcd"""
        # TODO: Implement with python-etcd3 or aioetcd
        logger.warning("Etcd storage backend is a stub implementation")
        logger.info(f"Would connect to etcd at {self.endpoints}")

    async def disconnect(self) -> None:
        """Disconnect from etcd"""
        logger.info("Etcd disconnect (stub)")

    async def set(self, key: str, value: Any) -> None:
        """Set a key-value pair"""
        # Stub: log only
        logger.debug(f"Etcd set (stub): {key}={value}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        # Stub: return None
        return None

    async def delete(self, key: str) -> None:
        """Delete a key"""
        logger.debug(f"Etcd delete (stub): {key}")

    async def list_keys(self, prefix: str) -> List[str]:
        """List keys with prefix"""
        # Stub: return empty list
        return []

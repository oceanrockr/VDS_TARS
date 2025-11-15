"""
T.A.R.S. API Key Persistent Storage
Redis-backed persistent API key storage with hot-rotation support
Phase 12 Part 2
"""

import os
import json
import redis
import logging
from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram
    METRICS_AVAILABLE = True

    # API key operation metrics
    api_key_created_total = Counter(
        'api_key_created_total',
        'Total number of API keys created',
        ['service_name']
    )

    api_key_revoked_total = Counter(
        'api_key_revoked_total',
        'Total number of API keys revoked',
        ['service_name']
    )

    api_key_deleted_total = Counter(
        'api_key_deleted_total',
        'Total number of API keys deleted',
        ['service_name']
    )

    api_key_verification_total = Counter(
        'api_key_verification_total',
        'Total API key verification attempts',
        ['status']  # success, failed
    )

    api_key_active_count = Gauge(
        'api_key_active_count',
        'Number of active API keys'
    )

    api_key_revoked_count = Gauge(
        'api_key_revoked_count',
        'Number of revoked API keys'
    )

    api_key_store_operation_duration = Histogram(
        'api_key_store_operation_duration_seconds',
        'Duration of API key store operations',
        ['operation']  # create, get, update, revoke, delete
    )

except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class APIKeyRecord(BaseModel):
    """Persistent API key record"""
    key_id: str
    service_name: str
    key_hash: str  # SHA-256 hash of the key
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    metadata: Optional[Dict] = None


class APIKeyStore:
    """
    Persistent API key storage using Redis

    Features:
    - Persistent storage across service restarts
    - Fast key lookups using Redis hash operations
    - Hot-rotation without downtime
    - Usage tracking with last_used_at timestamps
    - Revocation support
    - Prometheus metrics integration

    Redis Schema:
    - Hash: api_keys:{key_id} -> JSON serialized APIKeyRecord
    - Hash: api_keys:by_hash:{key_hash} -> key_id (for fast reverse lookup)
    - Set: api_keys:active -> set of active key_ids
    - Set: api_keys:revoked -> set of revoked key_ids
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        key_prefix: str = "api_keys"
    ):
        """
        Initialize API key store

        Args:
            redis_host: Redis host (default: REDIS_HOST env or localhost)
            redis_port: Redis port (default: REDIS_PORT env or 6379)
            redis_db: Redis database number (default: 0)
            redis_password: Redis password (default: REDIS_PASSWORD env or None)
            key_prefix: Prefix for Redis keys (default: api_keys)
        """
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = redis_db
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD")
        self.key_prefix = key_prefix

        # Initialize Redis client
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"API Key Store connected to Redis at {self.redis_host}:{self.redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("API Key Store will operate in degraded mode (no persistence)")
            self.redis_client = None

    def _key_path(self, key_id: str) -> str:
        """Generate Redis key path for API key record"""
        return f"{self.key_prefix}:{key_id}"

    def _hash_path(self, key_hash: str) -> str:
        """Generate Redis key path for hash-to-key_id mapping"""
        return f"{self.key_prefix}:by_hash:{key_hash}"

    def _active_set_key(self) -> str:
        """Redis set key for active API keys"""
        return f"{self.key_prefix}:active"

    def _revoked_set_key(self) -> str:
        """Redis set key for revoked API keys"""
        return f"{self.key_prefix}:revoked"

    def create(self, record: APIKeyRecord) -> bool:
        """
        Create a new API key record

        Args:
            record: APIKeyRecord to store

        Returns:
            True if created successfully, False otherwise
        """
        if self.redis_client is None:
            logger.error("Redis client not available, cannot create API key")
            return False

        start_time = datetime.utcnow()
        try:
            # Serialize record
            record_json = record.json()

            # Store record
            self.redis_client.set(self._key_path(record.key_id), record_json)

            # Create hash-to-key_id mapping for fast reverse lookup
            self.redis_client.set(self._hash_path(record.key_hash), record.key_id)

            # Add to active set
            self.redis_client.sadd(self._active_set_key(), record.key_id)

            logger.info(f"Created API key record: {record.key_id} ({record.service_name})")

            # Metrics
            if METRICS_AVAILABLE:
                api_key_created_total.labels(service_name=record.service_name).inc()
                api_key_active_count.set(self.count_active())
                duration = (datetime.utcnow() - start_time).total_seconds()
                api_key_store_operation_duration.labels(operation='create').observe(duration)

            return True

        except Exception as e:
            logger.error(f"Failed to create API key record {record.key_id}: {e}")
            return False

    def get_by_id(self, key_id: str) -> Optional[APIKeyRecord]:
        """
        Retrieve API key record by key_id

        Args:
            key_id: Unique key identifier

        Returns:
            APIKeyRecord if found, None otherwise
        """
        if self.redis_client is None:
            return None

        try:
            record_json = self.redis_client.get(self._key_path(key_id))

            if record_json is None:
                return None

            return APIKeyRecord.parse_raw(record_json)

        except Exception as e:
            logger.error(f"Failed to retrieve API key {key_id}: {e}")
            return None

    def get_by_hash(self, key_hash: str) -> Optional[APIKeyRecord]:
        """
        Retrieve API key record by hash (fast reverse lookup)

        Args:
            key_hash: SHA-256 hash of the API key

        Returns:
            APIKeyRecord if found, None otherwise
        """
        if self.redis_client is None:
            return None

        try:
            # Reverse lookup: hash -> key_id
            key_id = self.redis_client.get(self._hash_path(key_hash))

            if key_id is None:
                return None

            # Retrieve record by key_id
            return self.get_by_id(key_id)

        except Exception as e:
            logger.error(f"Failed to retrieve API key by hash: {e}")
            return None

    def update_last_used(self, key_id: str) -> bool:
        """
        Update last_used_at timestamp for an API key

        Args:
            key_id: Unique key identifier

        Returns:
            True if updated successfully, False otherwise
        """
        if self.redis_client is None:
            return False

        try:
            record = self.get_by_id(key_id)

            if record is None:
                return False

            # Update timestamp
            record.last_used_at = datetime.utcnow()

            # Save back to Redis
            self.redis_client.set(self._key_path(key_id), record.json())

            return True

        except Exception as e:
            logger.error(f"Failed to update last_used for {key_id}: {e}")
            return False

    def revoke(self, key_id: str) -> bool:
        """
        Revoke an API key

        Args:
            key_id: Unique key identifier

        Returns:
            True if revoked successfully, False otherwise
        """
        if self.redis_client is None:
            return False

        start_time = datetime.utcnow()
        try:
            record = self.get_by_id(key_id)

            if record is None:
                logger.warning(f"Cannot revoke non-existent key: {key_id}")
                return False

            # Update record
            record.revoked = True
            record.revoked_at = datetime.utcnow()

            # Save to Redis
            self.redis_client.set(self._key_path(key_id), record.json())

            # Move from active to revoked set
            self.redis_client.srem(self._active_set_key(), key_id)
            self.redis_client.sadd(self._revoked_set_key(), key_id)

            logger.info(f"Revoked API key: {key_id}")

            # Metrics
            if METRICS_AVAILABLE:
                api_key_revoked_total.labels(service_name=record.service_name).inc()
                api_key_active_count.set(self.count_active())
                api_key_revoked_count.set(self.count_revoked())
                duration = (datetime.utcnow() - start_time).total_seconds()
                api_key_store_operation_duration.labels(operation='revoke').observe(duration)

            return True

        except Exception as e:
            logger.error(f"Failed to revoke API key {key_id}: {e}")
            return False

    def delete(self, key_id: str) -> bool:
        """
        Permanently delete an API key record

        Args:
            key_id: Unique key identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        if self.redis_client is None:
            return False

        try:
            record = self.get_by_id(key_id)

            if record is None:
                return False

            # Delete hash-to-key_id mapping
            self.redis_client.delete(self._hash_path(record.key_hash))

            # Delete record
            self.redis_client.delete(self._key_path(key_id))

            # Remove from sets
            self.redis_client.srem(self._active_set_key(), key_id)
            self.redis_client.srem(self._revoked_set_key(), key_id)

            logger.info(f"Deleted API key: {key_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete API key {key_id}: {e}")
            return False

    def list_active(self) -> List[APIKeyRecord]:
        """
        List all active (non-revoked) API keys

        Returns:
            List of active APIKeyRecords
        """
        if self.redis_client is None:
            return []

        try:
            active_key_ids = self.redis_client.smembers(self._active_set_key())

            records = []
            for key_id in active_key_ids:
                record = self.get_by_id(key_id)
                if record and not record.revoked:
                    records.append(record)

            return records

        except Exception as e:
            logger.error(f"Failed to list active API keys: {e}")
            return []

    def list_revoked(self) -> List[APIKeyRecord]:
        """
        List all revoked API keys

        Returns:
            List of revoked APIKeyRecords
        """
        if self.redis_client is None:
            return []

        try:
            revoked_key_ids = self.redis_client.smembers(self._revoked_set_key())

            records = []
            for key_id in revoked_key_ids:
                record = self.get_by_id(key_id)
                if record and record.revoked:
                    records.append(record)

            return records

        except Exception as e:
            logger.error(f"Failed to list revoked API keys: {e}")
            return []

    def list_all(self) -> List[APIKeyRecord]:
        """
        List all API keys (active + revoked)

        Returns:
            List of all APIKeyRecords
        """
        return self.list_active() + self.list_revoked()

    def count_active(self) -> int:
        """Count active API keys"""
        if self.redis_client is None:
            return 0

        try:
            return self.redis_client.scard(self._active_set_key())
        except Exception as e:
            logger.error(f"Failed to count active API keys: {e}")
            return 0

    def count_revoked(self) -> int:
        """Count revoked API keys"""
        if self.redis_client is None:
            return 0

        try:
            return self.redis_client.scard(self._revoked_set_key())
        except Exception as e:
            logger.error(f"Failed to count revoked API keys: {e}")
            return 0

    def health_check(self) -> Dict[str, any]:
        """
        Health check for API key store

        Returns:
            Dict with health status and stats
        """
        if self.redis_client is None:
            return {
                "status": "degraded",
                "redis_connected": False,
                "error": "Redis client not initialized"
            }

        try:
            # Test Redis connection
            self.redis_client.ping()

            return {
                "status": "healthy",
                "redis_connected": True,
                "active_keys": self.count_active(),
                "revoked_keys": self.count_revoked(),
                "total_keys": self.count_active() + self.count_revoked()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e)
            }

    def migrate_from_memory(self, memory_keys: Dict[str, any]) -> int:
        """
        Migrate in-memory API keys to Redis

        Args:
            memory_keys: Dictionary of in-memory API keys

        Returns:
            Number of keys migrated
        """
        if self.redis_client is None:
            logger.error("Cannot migrate: Redis client not available")
            return 0

        migrated = 0

        for key_id, key_obj in memory_keys.items():
            try:
                # Convert to APIKeyRecord
                record = APIKeyRecord(
                    key_id=key_obj.key_id,
                    service_name=key_obj.service_name,
                    key_hash=key_obj.key_hash,
                    created_at=key_obj.created_at,
                    last_used_at=key_obj.last_used,
                    revoked=not key_obj.is_active
                )

                if self.create(record):
                    migrated += 1

            except Exception as e:
                logger.error(f"Failed to migrate key {key_id}: {e}")

        logger.info(f"Migrated {migrated}/{len(memory_keys)} API keys to Redis")
        return migrated


# Global API key store instance
api_key_store = APIKeyStore()

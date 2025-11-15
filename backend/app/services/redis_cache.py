"""
T.A.R.S. Redis Caching Service
Provides caching layer for embeddings and reranker results
Phase 6 - Production Scaling & Monitoring
"""

import logging
import json
import hashlib
from typing import Optional, Any, List
import asyncio
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool

from ..core.config import settings

logger = logging.getLogger(__name__)


class RedisCacheService:
    """
    Redis-based caching service for T.A.R.S.

    Provides TTL-based caching for:
    - Query embeddings (60 min TTL)
    - Cross-encoder reranking scores (60 min TTL)
    - Document embeddings (persistent)

    Features:
    - Connection pooling for performance
    - Automatic key generation from content hashing
    - JSON serialization for complex types
    - Health monitoring
    - Cache statistics
    """

    def __init__(self):
        self.host = getattr(settings, 'REDIS_HOST', 'localhost')
        self.port = getattr(settings, 'REDIS_PORT', 6379)
        self.db = getattr(settings, 'REDIS_DB', 0)
        self.password = getattr(settings, 'REDIS_PASSWORD', None)
        self.max_connections = getattr(settings, 'REDIS_MAX_CONNECTIONS', 50)

        # TTL configurations (in seconds)
        self.embedding_ttl = getattr(settings, 'REDIS_EMBEDDING_TTL', 3600)  # 60 minutes
        self.reranker_ttl = getattr(settings, 'REDIS_RERANKER_TTL', 3600)  # 60 minutes
        self.default_ttl = getattr(settings, 'REDIS_DEFAULT_TTL', 3600)  # 60 minutes

        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None
        self.is_connected = False

        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }

        logger.info(
            f"RedisCacheService initialized "
            f"(host: {self.host}:{self.port}, db: {self.db})"
        )

    async def connect(self) -> bool:
        """
        Connect to Redis server with connection pooling.

        Returns:
            True if successful, False otherwise
        """
        if self.is_connected:
            logger.info("Redis already connected")
            return True

        try:
            logger.info(f"Connecting to Redis at {self.host}:{self.port}")

            # Create connection pool
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )

            # Create client from pool
            self.client = Redis(connection_pool=self.pool)

            # Test connection
            await self.client.ping()

            self.is_connected = True
            logger.info(
                f"Connected to Redis successfully "
                f"(pool size: {self.max_connections})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup resources."""
        if self.client:
            await self.client.close()
            self.client = None

        if self.pool:
            await self.pool.disconnect()
            self.pool = None

        self.is_connected = False
        logger.info("Disconnected from Redis")

    def _generate_key(self, prefix: str, content: str) -> str:
        """
        Generate cache key from content hash.

        Args:
            prefix: Key prefix (e.g., 'embedding', 'rerank')
            content: Content to hash

        Returns:
            Cache key string
        """
        # Create SHA256 hash of content
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"tars:{prefix}:{content_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.is_connected or not self.client:
            logger.warning("Redis not connected, cache get skipped")
            return None

        try:
            value = await self.client.get(key)

            if value is not None:
                self.stats['hits'] += 1
                logger.debug(f"Cache HIT: {key}")

                # Deserialize JSON if needed
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            else:
                self.stats['misses'] += 1
                logger.debug(f"Cache MISS: {key}")
                return None

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if complex type)
            ttl: Time-to-live in seconds (None for no expiration)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or not self.client:
            logger.warning("Redis not connected, cache set skipped")
            return False

        try:
            # Serialize complex types to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value)

            # Set with TTL if specified
            if ttl is not None:
                await self.client.setex(key, ttl, value)
            else:
                await self.client.set(key, value)

            self.stats['sets'] += 1
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected or not self.client:
            return False

        try:
            result = await self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None if not cached
        """
        key = self._generate_key('embedding', text)
        return await self.get(key)

    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache embedding for text.

        Args:
            text: Input text
            embedding: Embedding vector
            ttl: TTL in seconds (defaults to REDIS_EMBEDDING_TTL)

        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.embedding_ttl

        key = self._generate_key('embedding', text)
        return await self.set(key, embedding, ttl)

    async def get_reranker_scores(
        self,
        query: str,
        document_ids: List[str]
    ) -> Optional[List[float]]:
        """
        Get cached reranker scores for query-document pairs.

        Args:
            query: Search query
            document_ids: List of document/chunk IDs

        Returns:
            List of scores or None if not cached
        """
        # Create compound key from query + sorted doc IDs
        compound_key = f"{query}::{','.join(sorted(document_ids))}"
        key = self._generate_key('rerank', compound_key)
        return await self.get(key)

    async def set_reranker_scores(
        self,
        query: str,
        document_ids: List[str],
        scores: List[float],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache reranker scores for query-document pairs.

        Args:
            query: Search query
            document_ids: List of document/chunk IDs
            scores: Reranking scores
            ttl: TTL in seconds (defaults to REDIS_RERANKER_TTL)

        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.reranker_ttl

        compound_key = f"{query}::{','.join(sorted(document_ids))}"
        key = self._generate_key('rerank', compound_key)
        return await self.set(key, scores, ttl)

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Redis key pattern (e.g., 'tars:embedding:*')

        Returns:
            Number of keys deleted
        """
        if not self.is_connected or not self.client:
            return 0

        try:
            # Scan for matching keys
            deleted = 0
            async for key in self.client.scan_iter(match=pattern):
                await self.client.delete(key)
                deleted += 1

            logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
            return deleted

        except Exception as e:
            logger.error(f"Error clearing pattern: {e}")
            return 0

    async def clear_all(self) -> bool:
        """
        Clear all T.A.R.S. cache entries.

        Returns:
            True if successful
        """
        try:
            deleted = await self.clear_pattern('tars:*')
            logger.info(f"Cleared all cache entries ({deleted} keys)")
            return True
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False

    async def get_info(self) -> dict:
        """
        Get Redis server info and cache statistics.

        Returns:
            Dictionary with Redis info and stats
        """
        if not self.is_connected or not self.client:
            return {
                'connected': False,
                'error': 'Not connected to Redis'
            }

        try:
            # Get Redis info
            info = await self.client.info()

            # Count T.A.R.S. keys
            tars_keys = 0
            async for _ in self.client.scan_iter(match='tars:*'):
                tars_keys += 1

            # Calculate hit rate
            total_reads = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_reads * 100) if total_reads > 0 else 0.0

            return {
                'connected': True,
                'host': self.host,
                'port': self.port,
                'db': self.db,
                'redis_version': info.get('redis_version', 'unknown'),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'total_keys': info.get('db0', {}).get('keys', 0) if isinstance(info.get('db0'), dict) else 0,
                'tars_keys': tars_keys,
                'cache_stats': {
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'sets': self.stats['sets'],
                    'errors': self.stats['errors'],
                    'hit_rate_percent': round(hit_rate, 2)
                },
                'ttl_config': {
                    'embedding_ttl_seconds': self.embedding_ttl,
                    'reranker_ttl_seconds': self.reranker_ttl,
                    'default_ttl_seconds': self.default_ttl
                }
            }

        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {
                'connected': self.is_connected,
                'error': str(e)
            }

    async def health_check(self) -> bool:
        """
        Check if Redis is healthy and responsive.

        Returns:
            True if healthy, False otherwise
        """
        if not self.is_connected or not self.client:
            return False

        try:
            response = await self.client.ping()
            return response is True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_reads = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_reads * 100) if total_reads > 0 else 0.0

        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'errors': self.stats['errors'],
            'hit_rate_percent': round(hit_rate, 2),
            'total_reads': total_reads
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
        logger.info("Cache statistics reset")


# Global service instance
redis_cache = RedisCacheService()

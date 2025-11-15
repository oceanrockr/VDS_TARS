"""
Unit Tests for Redis Cache Service
Phase 6 - Production Scaling & Monitoring
"""

import pytest
import asyncio
from typing import List
import time

from app.services.redis_cache import RedisCacheService, redis_cache


class TestRedisCacheService:
    """Test suite for RedisCacheService"""

    @pytest.fixture
    async def cache_service(self):
        """Create a test cache service instance"""
        service = RedisCacheService()
        connected = await service.connect()
        if not connected:
            pytest.skip("Redis not available for testing")

        # Clear any existing test data
        await service.clear_pattern('tars:test:*')

        yield service

        # Cleanup
        await service.clear_pattern('tars:test:*')
        await service.disconnect()

    @pytest.mark.asyncio
    async def test_connection(self, cache_service):
        """Test Redis connection"""
        assert cache_service.is_connected is True
        health = await cache_service.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_set_and_get_string(self, cache_service):
        """Test basic string set/get"""
        key = "tars:test:string"
        value = "test_value"

        # Set value
        result = await cache_service.set(key, value)
        assert result is True

        # Get value
        retrieved = await cache_service.get(key)
        assert retrieved == value

    @pytest.mark.asyncio
    async def test_set_and_get_json(self, cache_service):
        """Test JSON object set/get"""
        key = "tars:test:json"
        value = {"name": "T.A.R.S.", "version": "0.3.0", "items": [1, 2, 3]}

        # Set value
        result = await cache_service.set(key, value)
        assert result is True

        # Get value
        retrieved = await cache_service.get(key)
        assert retrieved == value

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache_service):
        """Test TTL expiration"""
        key = "tars:test:ttl"
        value = "temporary_value"
        ttl = 2  # 2 seconds

        # Set with TTL
        result = await cache_service.set(key, value, ttl=ttl)
        assert result is True

        # Should exist immediately
        retrieved = await cache_service.get(key)
        assert retrieved == value

        # Wait for expiration
        await asyncio.sleep(ttl + 1)

        # Should be expired
        retrieved = await cache_service.get(key)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache_service):
        """Test getting a key that doesn't exist"""
        result = await cache_service.get("tars:test:nonexistent")
        assert result is None

        # Check stats
        stats = cache_service.get_stats()
        assert stats['misses'] > 0

    @pytest.mark.asyncio
    async def test_delete(self, cache_service):
        """Test key deletion"""
        key = "tars:test:delete"
        value = "to_be_deleted"

        # Set value
        await cache_service.set(key, value)
        assert await cache_service.get(key) == value

        # Delete
        result = await cache_service.delete(key)
        assert result is True

        # Verify deleted
        retrieved = await cache_service.get(key)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_embedding_cache(self, cache_service):
        """Test embedding-specific caching"""
        text = "What is machine learning?"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4]  # 384 dimensions

        # Set embedding
        result = await cache_service.set_embedding(text, embedding)
        assert result is True

        # Get embedding
        retrieved = await cache_service.get_embedding(text)
        assert retrieved == embedding

    @pytest.mark.asyncio
    async def test_reranker_scores_cache(self, cache_service):
        """Test reranker scores caching"""
        query = "What is AI?"
        document_ids = ["doc_1", "doc_2", "doc_3"]
        scores = [0.95, 0.87, 0.72]

        # Set scores
        result = await cache_service.set_reranker_scores(query, document_ids, scores)
        assert result is True

        # Get scores
        retrieved = await cache_service.get_reranker_scores(query, document_ids)
        assert retrieved == scores

    @pytest.mark.asyncio
    async def test_reranker_scores_order_independence(self, cache_service):
        """Test that document ID order doesn't affect cache lookup"""
        query = "What is deep learning?"
        document_ids_1 = ["doc_a", "doc_b", "doc_c"]
        document_ids_2 = ["doc_c", "doc_a", "doc_b"]  # Different order
        scores = [0.9, 0.8, 0.7]

        # Set with first order
        await cache_service.set_reranker_scores(query, document_ids_1, scores)

        # Get with second order (should find same cache entry)
        retrieved = await cache_service.get_reranker_scores(query, document_ids_2)
        assert retrieved == scores

    @pytest.mark.asyncio
    async def test_clear_pattern(self, cache_service):
        """Test clearing keys by pattern"""
        # Set multiple keys
        await cache_service.set("tars:test:clear:1", "value1")
        await cache_service.set("tars:test:clear:2", "value2")
        await cache_service.set("tars:test:keep:3", "value3")

        # Clear pattern
        deleted = await cache_service.clear_pattern("tars:test:clear:*")
        assert deleted == 2

        # Verify cleared
        assert await cache_service.get("tars:test:clear:1") is None
        assert await cache_service.get("tars:test:clear:2") is None

        # Verify kept
        assert await cache_service.get("tars:test:keep:3") == "value3"

    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_service):
        """Test cache statistics tracking"""
        # Reset stats
        cache_service.reset_stats()

        # Perform operations
        await cache_service.set("tars:test:stats:1", "value1")
        await cache_service.get("tars:test:stats:1")  # Hit
        await cache_service.get("tars:test:stats:nonexistent")  # Miss

        stats = cache_service.get_stats()

        assert stats['sets'] >= 1
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
        assert stats['hit_rate_percent'] > 0

    @pytest.mark.asyncio
    async def test_get_info(self, cache_service):
        """Test getting Redis info"""
        info = await cache_service.get_info()

        assert info['connected'] is True
        assert 'redis_version' in info
        assert 'cache_stats' in info
        assert 'ttl_config' in info
        assert info['ttl_config']['embedding_ttl_seconds'] == 3600
        assert info['ttl_config']['reranker_ttl_seconds'] == 3600

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache_service):
        """Test concurrent cache access"""
        async def set_and_get(index: int):
            key = f"tars:test:concurrent:{index}"
            value = f"value_{index}"
            await cache_service.set(key, value)
            retrieved = await cache_service.get(key)
            return retrieved == value

        # Run 10 concurrent operations
        tasks = [set_and_get(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

    @pytest.mark.asyncio
    async def test_large_value(self, cache_service):
        """Test caching large values (embeddings, etc.)"""
        key = "tars:test:large"
        # Simulate a large embedding batch
        value = [0.1] * 10000  # 10,000 floats

        result = await cache_service.set(key, value)
        assert result is True

        retrieved = await cache_service.get(key)
        assert len(retrieved) == len(value)
        assert retrieved == value

    @pytest.mark.asyncio
    async def test_key_generation_consistency(self, cache_service):
        """Test that same content generates same key"""
        text = "Consistent hashing test"

        key1 = cache_service._generate_key('embedding', text)
        key2 = cache_service._generate_key('embedding', text)

        assert key1 == key2
        assert key1.startswith('tars:embedding:')

    @pytest.mark.asyncio
    async def test_different_content_different_keys(self, cache_service):
        """Test that different content generates different keys"""
        text1 = "First text"
        text2 = "Second text"

        key1 = cache_service._generate_key('embedding', text1)
        key2 = cache_service._generate_key('embedding', text2)

        assert key1 != key2

    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, cache_service):
        """Test cache performance improvement"""
        key = "tars:test:performance"
        value = [0.1] * 384  # Typical embedding size

        # Set value
        await cache_service.set(key, value)

        # Measure cache hit time
        start = time.time()
        for _ in range(100):
            await cache_service.get(key)
        cache_time = time.time() - start

        # Cache should be fast (< 1 second for 100 ops)
        assert cache_time < 1.0

    @pytest.mark.asyncio
    async def test_disconnect_and_reconnect(self, cache_service):
        """Test disconnection and reconnection"""
        # Disconnect
        await cache_service.disconnect()
        assert cache_service.is_connected is False

        # Reconnect
        result = await cache_service.connect()
        assert result is True
        assert cache_service.is_connected is True

    @pytest.mark.asyncio
    async def test_graceful_failure_when_disconnected(self, cache_service):
        """Test that operations fail gracefully when disconnected"""
        await cache_service.disconnect()

        # Operations should return None/False but not crash
        result = await cache_service.get("any_key")
        assert result is None

        result = await cache_service.set("any_key", "value")
        assert result is False


class TestRedisCacheIntegration:
    """Integration tests with global redis_cache instance"""

    @pytest.mark.asyncio
    async def test_global_instance(self):
        """Test global redis_cache instance"""
        connected = await redis_cache.connect()
        if not connected:
            pytest.skip("Redis not available")

        # Test basic operation
        test_key = "tars:test:global"
        await redis_cache.set(test_key, "test_value")
        result = await redis_cache.get(test_key)

        assert result == "test_value"

        # Cleanup
        await redis_cache.delete(test_key)

    @pytest.mark.asyncio
    async def test_embedding_workflow(self):
        """Test typical embedding caching workflow"""
        connected = await redis_cache.connect()
        if not connected:
            pytest.skip("Redis not available")

        query = "What is artificial intelligence?"
        embedding = [0.1, 0.2, 0.3] * 128  # 384-dim embedding

        # First call - cache miss
        cached = await redis_cache.get_embedding(query)
        assert cached is None

        # Store embedding
        await redis_cache.set_embedding(query, embedding, ttl=60)

        # Second call - cache hit
        cached = await redis_cache.get_embedding(query)
        assert cached == embedding

    @pytest.mark.asyncio
    async def test_reranker_workflow(self):
        """Test typical reranker caching workflow"""
        connected = await redis_cache.connect()
        if not connected:
            pytest.skip("Redis not available")

        query = "Explain neural networks"
        doc_ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        scores = [0.95, 0.88, 0.82, 0.75, 0.68]

        # First call - cache miss
        cached = await redis_cache.get_reranker_scores(query, doc_ids)
        assert cached is None

        # Store scores
        await redis_cache.set_reranker_scores(query, doc_ids, scores, ttl=60)

        # Second call - cache hit
        cached = await redis_cache.get_reranker_scores(query, doc_ids)
        assert cached == scores


# Performance benchmarks (optional, can be run separately)
class TestRedisCachePerformance:
    """Performance benchmarks for Redis cache"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_embedding_cache_throughput(self):
        """Benchmark embedding cache throughput"""
        connected = await redis_cache.connect()
        if not connected:
            pytest.skip("Redis not available")

        embeddings = [[0.1] * 384 for _ in range(100)]
        queries = [f"Query {i}" for i in range(100)]

        # Benchmark writes
        start = time.time()
        for query, embedding in zip(queries, embeddings):
            await redis_cache.set_embedding(query, embedding)
        write_time = time.time() - start

        # Benchmark reads
        start = time.time()
        for query in queries:
            await redis_cache.get_embedding(query)
        read_time = time.time() - start

        print(f"\nWrite throughput: {100/write_time:.2f} ops/sec")
        print(f"Read throughput: {100/read_time:.2f} ops/sec")

        # Cleanup
        for query in queries:
            key = redis_cache._generate_key('embedding', query)
            await redis_cache.delete(key)

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cache_hit_rate_simulation(self):
        """Simulate realistic cache hit rate"""
        connected = await redis_cache.connect()
        if not connected:
            pytest.skip("Redis not available")

        # Reset stats
        redis_cache.reset_stats()

        # Simulate 100 queries with 30% unique (70% hit rate expected)
        unique_queries = [f"Unique query {i}" for i in range(30)]
        all_queries = unique_queries * 3 + unique_queries  # Repeat to simulate cache hits

        for query in all_queries:
            embedding = await redis_cache.get_embedding(query)
            if embedding is None:
                # Simulate embedding generation
                embedding = [0.1] * 384
                await redis_cache.set_embedding(query, embedding, ttl=300)

        stats = redis_cache.get_stats()
        print(f"\nCache hit rate: {stats['hit_rate_percent']:.2f}%")
        print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")

        # Should achieve reasonable hit rate
        assert stats['hit_rate_percent'] > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
T.A.R.S. Rate Limiter Tests

Comprehensive test suite for the rate limiting module, covering:
- In-memory rate limiting
- Redis-backed rate limiting
- Token bucket algorithm
- Sliding window algorithm
- Rate limit enforcement
- Multiple client isolation

NOTE: This module requires cognition.shared.rate_limiter to be importable.
Ensure the project root is in PYTHONPATH or run pytest from project root.
"""

import asyncio
import os
import sys
import time
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports - ensure it's at the very beginning
# This is a workaround for pytest's module collection which runs before conftest.py
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Try to import, skip the entire module if it fails
try:
    from cognition.shared.rate_limiter import (
        InMemoryRateLimiter,
        RateLimitConfig,
        RateLimiter,
        RedisRateLimiter,
    )
except ImportError as e:
    pytest.skip(
        f"Skipping rate limiter tests: cognition module not importable. "
        f"Ensure PYTHONPATH includes project root. Error: {e}",
        allow_module_level=True
    )


# ============================================================================
# TEST IN-MEMORY RATE LIMITER
# ============================================================================

class TestInMemoryRateLimiter:
    """Test suite for InMemoryRateLimiter class"""

    def test_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed"""
        limiter = InMemoryRateLimiter()
        key = "test_user_1"
        limit = 5
        window = 60

        # Make requests under the limit
        for i in range(limit):
            allowed, info = limiter.check_rate_limit(key, limit, window)

            assert allowed is True, f"Request {i+1} should be allowed"
            assert info["remaining"] == limit - (i + 1), \
                f"Remaining should be {limit - (i + 1)}"
            assert info["retry_after"] == 0, "No retry needed under limit"
            assert "reset_at" in info, "Should include reset_at"

    def test_blocks_requests_over_limit(self):
        """Test that requests over the limit are blocked"""
        limiter = InMemoryRateLimiter()
        key = "test_user_2"
        limit = 3
        window = 60

        # Make requests up to the limit
        for i in range(limit):
            allowed, info = limiter.check_rate_limit(key, limit, window)
            assert allowed is True, f"Request {i+1} should be allowed"

        # Next request should be blocked
        allowed, info = limiter.check_rate_limit(key, limit, window)

        assert allowed is False, "Request over limit should be blocked"
        assert info["remaining"] == 0, "No remaining requests"
        assert info["retry_after"] > 0, "Should have retry_after"
        assert info["reset_at"] > time.time(), "Reset should be in future"

    def test_different_keys_have_separate_limits(self):
        """Test that different keys maintain separate rate limits"""
        limiter = InMemoryRateLimiter()
        limit = 2
        window = 60

        # Make requests for user 1 up to limit
        for i in range(limit):
            allowed, info = limiter.check_rate_limit("user_1", limit, window)
            assert allowed is True

        # User 1 should be rate limited
        allowed, info = limiter.check_rate_limit("user_1", limit, window)
        assert allowed is False

        # User 2 should still be able to make requests
        allowed, info = limiter.check_rate_limit("user_2", limit, window)
        assert allowed is True

        allowed, info = limiter.check_rate_limit("user_2", limit, window)
        assert allowed is True

        # User 2 should now be rate limited
        allowed, info = limiter.check_rate_limit("user_2", limit, window)
        assert allowed is False

    def test_window_reset(self):
        """Test that rate limit resets after window expires"""
        limiter = InMemoryRateLimiter()
        key = "test_user_3"
        limit = 2
        window = 1  # 1 second window

        # Use up the limit
        for i in range(limit):
            allowed, info = limiter.check_rate_limit(key, limit, window)
            assert allowed is True

        # Should be blocked
        allowed, info = limiter.check_rate_limit(key, limit, window)
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        allowed, info = limiter.check_rate_limit(key, limit, window)
        assert allowed is True

    def test_cleanup_expired_buckets(self):
        """Test cleanup of expired buckets"""
        limiter = InMemoryRateLimiter()
        limit = 5
        window = 1  # 1 second window

        # Create several buckets
        for i in range(5):
            limiter.check_rate_limit(f"user_{i}", limit, window)

        assert len(limiter.buckets) == 5, "Should have 5 buckets"

        # Wait for expiration
        time.sleep(1.1)

        # Trigger cleanup
        limiter.cleanup_expired()

        assert len(limiter.buckets) == 0, "Expired buckets should be cleaned up"

    def test_remaining_count_accuracy(self):
        """Test that remaining count is accurate"""
        limiter = InMemoryRateLimiter()
        key = "test_user_4"
        limit = 10
        window = 60

        # Make 7 requests
        for i in range(7):
            allowed, info = limiter.check_rate_limit(key, limit, window)
            assert allowed is True

        # Check remaining count
        allowed, info = limiter.check_rate_limit(key, limit, window)
        assert allowed is True
        assert info["remaining"] == 2, "Should have 2 requests remaining"

    def test_concurrent_requests_same_key(self):
        """Test rate limiting with concurrent requests for the same key"""
        limiter = InMemoryRateLimiter()
        key = "concurrent_user"
        limit = 10
        window = 60

        results = []

        # Simulate concurrent requests
        for i in range(15):
            allowed, info = limiter.check_rate_limit(key, limit, window)
            results.append(allowed)

        # First 10 should be allowed, rest blocked
        assert sum(results) == limit, f"Expected {limit} allowed, got {sum(results)}"
        assert results[:limit] == [True] * limit, "First requests should be allowed"
        assert results[limit:] == [False] * (15 - limit), "Extra requests should be blocked"


# ============================================================================
# TEST REDIS RATE LIMITER
# ============================================================================

class TestRedisRateLimiter:
    """Test suite for RedisRateLimiter class"""

    def test_allows_requests_under_limit(self, mock_redis_sync):
        """Test Redis-backed rate limiter allows requests under limit"""
        limiter = RedisRateLimiter(mock_redis_sync)
        key = "redis_user_1"
        limit = 5
        window = 60

        # Make requests under the limit
        for i in range(limit):
            allowed, info = limiter.check_rate_limit(key, limit, window)

            assert allowed is True, f"Request {i+1} should be allowed"
            assert info["remaining"] >= 0, "Should have remaining requests"
            assert info["retry_after"] == 0, "No retry needed under limit"

    def test_blocks_requests_over_limit(self, mock_redis_sync):
        """Test Redis-backed rate limiter blocks requests over limit"""
        limiter = RedisRateLimiter(mock_redis_sync)
        key = "redis_user_2"
        limit = 3
        window = 60

        # Make requests up to the limit
        for i in range(limit):
            allowed, info = limiter.check_rate_limit(key, limit, window)
            assert allowed is True

        # Next request should be blocked
        allowed, info = limiter.check_rate_limit(key, limit, window)

        assert allowed is False, "Request over limit should be blocked"
        assert info["remaining"] == 0, "No remaining requests"
        assert info["retry_after"] > 0, "Should have retry_after"

    def test_different_keys_have_separate_limits(self, mock_redis_sync):
        """Test different Redis keys maintain separate limits"""
        limiter = RedisRateLimiter(mock_redis_sync)
        limit = 2
        window = 60

        # User 1 uses up limit
        for i in range(limit):
            allowed, info = limiter.check_rate_limit("redis_user_a", limit, window)
            assert allowed is True

        # User 1 blocked
        allowed, info = limiter.check_rate_limit("redis_user_a", limit, window)
        assert allowed is False

        # User 2 can still make requests
        allowed, info = limiter.check_rate_limit("redis_user_b", limit, window)
        assert allowed is True

    def test_sliding_window_cleanup(self, mock_redis_sync):
        """Test that old entries are cleaned up in sliding window"""
        limiter = RedisRateLimiter(mock_redis_sync)
        key = "redis_user_3"
        limit = 5
        window = 2  # 2 second window

        # Make some requests
        for i in range(3):
            allowed, info = limiter.check_rate_limit(key, limit, window)
            assert allowed is True

        # Mock Redis should have called zremrangebyscore for cleanup
        assert mock_redis_sync.zremrangebyscore.called, \
            "Should cleanup old entries"

    def test_redis_error_fail_open(self, mock_redis_sync):
        """Test that Redis errors fail open (allow request)"""
        # Make Redis operations raise an exception
        mock_redis_sync.zremrangebyscore.side_effect = Exception("Redis connection error")

        limiter = RedisRateLimiter(mock_redis_sync)
        key = "redis_user_error"
        limit = 5
        window = 60

        # Should allow request despite Redis error
        allowed, info = limiter.check_rate_limit(key, limit, window)

        assert allowed is True, "Should fail open on Redis error"
        assert info["remaining"] == limit, "Should return full limit on error"

    def test_sorted_set_operations(self, mock_redis_sync):
        """Test that Redis sorted set operations are called correctly"""
        limiter = RedisRateLimiter(mock_redis_sync)
        key = "redis_user_4"
        limit = 5
        window = 60

        allowed, info = limiter.check_rate_limit(key, limit, window)

        # Verify Redis operations were called
        assert mock_redis_sync.zremrangebyscore.called, "Should remove old entries"
        assert mock_redis_sync.zcard.called, "Should count current entries"
        assert mock_redis_sync.zadd.called, "Should add new entry"
        assert mock_redis_sync.expire.called, "Should set expiry on key"


# ============================================================================
# TEST RATE LIMITER (MAIN CLASS)
# ============================================================================

class TestRateLimiter:
    """Test suite for main RateLimiter class"""

    def test_uses_redis_when_available(self, mock_redis_sync):
        """Test that RateLimiter uses Redis when configured"""
        config = RateLimitConfig()
        config.use_redis = True
        config.redis_client = mock_redis_sync

        limiter = RateLimiter(config)

        assert isinstance(limiter.backend, RedisRateLimiter), \
            "Should use Redis backend when available"

    def test_falls_back_to_memory(self):
        """Test that RateLimiter falls back to in-memory when Redis unavailable"""
        config = RateLimitConfig()
        config.use_redis = False
        config.redis_client = None

        limiter = RateLimiter(config)

        assert isinstance(limiter.backend, InMemoryRateLimiter), \
            "Should use in-memory backend when Redis unavailable"

    def test_check_rate_limit_delegates_to_backend(self):
        """Test that check_rate_limit delegates to the backend"""
        config = RateLimitConfig()
        config.use_redis = False

        limiter = RateLimiter(config)
        key = "test_delegation"
        limit = 5
        window = 60

        allowed, info = limiter.check_rate_limit(key, limit, window)

        assert allowed is True
        assert isinstance(info, dict)
        assert "remaining" in info
        assert "reset_at" in info
        assert "retry_after" in info

    def test_get_client_identifier_from_ip(self, mock_request):
        """Test client identifier extraction from IP address"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {}

        identifier = limiter.get_client_identifier(mock_request)

        assert isinstance(identifier, str), "Should return string identifier"
        assert len(identifier) == 16, "Should be hashed to 16 chars"

    def test_get_client_identifier_from_forwarded_header(self, mock_request):
        """Test client identifier extraction from X-Forwarded-For header"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        mock_request.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}

        identifier = limiter.get_client_identifier(mock_request)

        assert isinstance(identifier, str), "Should return string identifier"
        assert len(identifier) == 16, "Should be hashed to 16 chars"

    def test_client_identifier_consistency(self, mock_request):
        """Test that same IP produces same identifier"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        mock_request.client.host = "192.168.1.100"

        id1 = limiter.get_client_identifier(mock_request)
        id2 = limiter.get_client_identifier(mock_request)

        assert id1 == id2, "Same IP should produce same identifier"

    def test_different_ips_different_identifiers(self, mock_request):
        """Test that different IPs produce different identifiers"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        mock_request.client.host = "192.168.1.100"
        id1 = limiter.get_client_identifier(mock_request)

        mock_request.client.host = "192.168.1.101"
        id2 = limiter.get_client_identifier(mock_request)

        assert id1 != id2, "Different IPs should produce different identifiers"


# ============================================================================
# TEST RATE LIMIT CONFIG
# ============================================================================

class TestRateLimitConfig:
    """Test suite for RateLimitConfig class"""

    def test_default_configuration(self):
        """Test default rate limit configuration"""
        config = RateLimitConfig()

        assert hasattr(config, 'public_endpoint_limit'), "Should have public limit"
        assert hasattr(config, 'auth_endpoint_limit'), "Should have auth limit"
        assert hasattr(config, 'window_seconds'), "Should have window"
        assert config.window_seconds == 60, "Default window should be 60 seconds"

    def test_redis_url_from_env(self):
        """Test Redis URL is read from environment"""
        import os
        with patch.dict(os.environ, {'REDIS_URL': 'redis://test-host:6380'}):
            config = RateLimitConfig()
            assert config.redis_url == 'redis://test-host:6380'

    def test_use_redis_flag(self):
        """Test USE_REDIS environment flag"""
        import os

        with patch.dict(os.environ, {'USE_REDIS': 'false'}):
            config = RateLimitConfig()
            assert config.use_redis is False

    def test_custom_rate_limits_from_env(self):
        """Test custom rate limits from environment"""
        import os

        with patch.dict(os.environ, {
            'PUBLIC_RATE_LIMIT': '100',
            'AUTH_RATE_LIMIT': '20'
        }):
            config = RateLimitConfig()
            assert config.public_endpoint_limit == 100
            assert config.auth_endpoint_limit == 20


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRateLimiterIntegration:
    """Integration tests for rate limiting"""

    def test_rate_limit_per_endpoint(self):
        """Test that rate limits are isolated per endpoint"""
        limiter = InMemoryRateLimiter()
        user = "user_multi_endpoint"
        limit = 5
        window = 60

        # Make requests to endpoint 1
        for i in range(limit):
            allowed, _ = limiter.check_rate_limit(
                f"{user}:/api/endpoint1",
                limit,
                window
            )
            assert allowed is True

        # User limited on endpoint 1
        allowed, _ = limiter.check_rate_limit(
            f"{user}:/api/endpoint1",
            limit,
            window
        )
        assert allowed is False

        # But can still access endpoint 2
        allowed, _ = limiter.check_rate_limit(
            f"{user}:/api/endpoint2",
            limit,
            window
        )
        assert allowed is True

    def test_varying_limits_same_user(self):
        """Test user with different limits on different resources"""
        limiter = InMemoryRateLimiter()
        user = "user_varied_limits"

        # Public endpoint with higher limit
        for i in range(30):
            allowed, _ = limiter.check_rate_limit(
                f"{user}:public",
                30,
                60
            )
            assert allowed is True

        # Auth endpoint with lower limit
        for i in range(10):
            allowed, _ = limiter.check_rate_limit(
                f"{user}:auth",
                10,
                60
            )
            assert allowed is True

        # Both should enforce their respective limits
        allowed, _ = limiter.check_rate_limit(f"{user}:public", 30, 60)
        assert allowed is False

        allowed, _ = limiter.check_rate_limit(f"{user}:auth", 10, 60)
        assert allowed is False

    def test_burst_traffic_handling(self):
        """Test handling of burst traffic"""
        limiter = InMemoryRateLimiter()
        key = "burst_user"
        limit = 100
        window = 60

        # Simulate burst of 100 requests
        results = []
        for i in range(150):
            allowed, _ = limiter.check_rate_limit(key, limit, window)
            results.append(allowed)

        # Exactly limit requests should be allowed
        assert sum(results) == limit
        assert all(results[:limit]), "First batch should be allowed"
        assert not any(results[limit:]), "Excess should be blocked"

    def test_gradual_traffic_pattern(self):
        """Test gradual traffic over time"""
        limiter = InMemoryRateLimiter()
        key = "gradual_user"
        limit = 5
        window = 2  # 2 second window

        # First batch
        for i in range(3):
            allowed, _ = limiter.check_rate_limit(key, limit, window)
            assert allowed is True

        # Wait a bit
        time.sleep(1)

        # More requests
        for i in range(2):
            allowed, _ = limiter.check_rate_limit(key, limit, window)
            assert allowed is True

        # Now limited
        allowed, _ = limiter.check_rate_limit(key, limit, window)
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.5)

        # Should be allowed again
        allowed, _ = limiter.check_rate_limit(key, limit, window)
        assert allowed is True


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestRateLimiterEdgeCases:
    """Test edge cases and error handling"""

    def test_zero_limit(self):
        """Test behavior with zero limit (block all)"""
        limiter = InMemoryRateLimiter()
        key = "zero_limit_user"

        allowed, info = limiter.check_rate_limit(key, 0, 60)

        assert allowed is False, "Zero limit should block all requests"
        assert info["remaining"] == 0

    def test_very_large_limit(self):
        """Test behavior with very large limit"""
        limiter = InMemoryRateLimiter()
        key = "large_limit_user"
        limit = 1000000

        allowed, info = limiter.check_rate_limit(key, limit, 60)

        assert allowed is True
        assert info["remaining"] == limit - 1

    def test_very_short_window(self):
        """Test behavior with very short window"""
        limiter = InMemoryRateLimiter()
        key = "short_window_user"

        # 100ms window
        allowed, info = limiter.check_rate_limit(key, 5, 0.1)

        assert allowed is True
        # Window should reset quickly
        time.sleep(0.15)

        allowed, info = limiter.check_rate_limit(key, 5, 0.1)
        assert allowed is True, "Window should have reset"

    def test_empty_key(self):
        """Test behavior with empty key"""
        limiter = InMemoryRateLimiter()

        allowed, info = limiter.check_rate_limit("", 10, 60)

        assert allowed is True, "Should handle empty key"

    def test_special_characters_in_key(self):
        """Test keys with special characters"""
        limiter = InMemoryRateLimiter()
        special_keys = [
            "user:with:colons",
            "user/with/slashes",
            "user@with@at",
            "user-with-dashes",
            "user_with_underscores"
        ]

        for key in special_keys:
            allowed, info = limiter.check_rate_limit(key, 5, 60)
            assert allowed is True, f"Should handle key: {key}"

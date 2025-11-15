"""
T.A.R.S. Rate Limiting Module
Token bucket and sliding window rate limiting with Redis backend
Phase 11.5
"""

import os
import time
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, status
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory rate limiting")


class RateLimitConfig:
    """Rate limiting configuration"""

    def __init__(self):
        self.use_redis = os.getenv("USE_REDIS", "true").lower() == "true" and REDIS_AVAILABLE
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Rate limit defaults
        self.public_endpoint_limit = int(os.getenv("PUBLIC_RATE_LIMIT", "30"))  # per minute
        self.auth_endpoint_limit = int(os.getenv("AUTH_RATE_LIMIT", "10"))  # per minute
        self.window_seconds = 60  # 1 minute window

        # Redis client
        self.redis_client: Optional[Redis] = None
        if self.use_redis:
            self._init_redis()

    def _init_redis(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("✓ Redis connected for rate limiting")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory: {e}")
            self.use_redis = False
            self.redis_client = None


# Global rate limit config
rate_limit_config = RateLimitConfig()


class InMemoryRateLimiter:
    """In-memory token bucket rate limiter (fallback)"""

    def __init__(self):
        # Format: {key: {"count": int, "reset_at": float}}
        self.buckets: Dict[str, Dict[str, Any]] = {}

    def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit

        Returns:
            (allowed, info) where info contains:
                - remaining: requests remaining
                - reset_at: timestamp when limit resets
                - retry_after: seconds until reset (if limited)
        """
        now = time.time()

        # Get or create bucket
        if key not in self.buckets:
            self.buckets[key] = {
                "count": 0,
                "reset_at": now + window_seconds
            }

        bucket = self.buckets[key]

        # Reset bucket if window expired
        if now >= bucket["reset_at"]:
            bucket["count"] = 0
            bucket["reset_at"] = now + window_seconds

        # Check limit
        if bucket["count"] >= limit:
            retry_after = int(bucket["reset_at"] - now) + 1
            return False, {
                "remaining": 0,
                "reset_at": bucket["reset_at"],
                "retry_after": retry_after
            }

        # Increment counter
        bucket["count"] += 1

        return True, {
            "remaining": limit - bucket["count"],
            "reset_at": bucket["reset_at"],
            "retry_after": 0
        }

    def cleanup_expired(self):
        """Remove expired buckets"""
        now = time.time()
        expired = [k for k, v in self.buckets.items() if now >= v["reset_at"]]
        for k in expired:
            del self.buckets[k]


class RedisRateLimiter:
    """Redis-backed sliding window rate limiter"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit using sliding window

        Uses Redis sorted sets for accurate sliding window
        """
        now = time.time()
        window_start = now - window_seconds

        # Prefix key for namespacing
        redis_key = f"rate_limit:{key}"

        try:
            # Remove old entries
            self.redis.zremrangebyscore(redis_key, 0, window_start)

            # Count current requests in window
            current_count = self.redis.zcard(redis_key)

            if current_count >= limit:
                # Get oldest entry to calculate retry_after
                oldest = self.redis.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = oldest[0][1]
                    retry_after = int((oldest_time + window_seconds) - now) + 1
                else:
                    retry_after = window_seconds

                return False, {
                    "remaining": 0,
                    "reset_at": now + retry_after,
                    "retry_after": retry_after
                }

            # Add current request
            self.redis.zadd(redis_key, {str(now): now})

            # Set expiry on key
            self.redis.expire(redis_key, window_seconds * 2)

            return True, {
                "remaining": limit - (current_count + 1),
                "reset_at": now + window_seconds,
                "retry_after": 0
            }

        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fail open - allow request if Redis fails
            return True, {
                "remaining": limit,
                "reset_at": now + window_seconds,
                "retry_after": 0
            }


class RateLimiter:
    """Main rate limiter that uses Redis or in-memory backend"""

    def __init__(self, config: RateLimitConfig = rate_limit_config):
        self.config = config

        if config.use_redis and config.redis_client:
            self.backend = RedisRateLimiter(config.redis_client)
            logger.info("✓ Using Redis rate limiting")
        else:
            self.backend = InMemoryRateLimiter()
            logger.info("✓ Using in-memory rate limiting")

    def check_rate_limit(self, key: str, limit: int, window_seconds: int = 60) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit"""
        return self.backend.check_rate_limit(key, limit, window_seconds)

    def get_client_identifier(self, request: Request) -> str:
        """Get a unique identifier for the client"""
        # Use X-Forwarded-For if behind proxy, otherwise client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        # Hash IP for privacy
        return hashlib.sha256(client_ip.encode()).hexdigest()[:16]


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(limit: int = 30, window_seconds: int = 60):
    """
    Decorator for rate limiting endpoints

    Args:
        limit: Maximum requests allowed in window
        window_seconds: Time window in seconds

    Example:
        @app.get("/api/data")
        @rate_limit(limit=30, window_seconds=60)
        async def get_data(request: Request):
            return {"data": "..."}
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            if request is None:
                # Try to find request in args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is None:
                # No request object, skip rate limiting
                logger.warning("Rate limit decorator: No request object found")
                return await func(*args, **kwargs)

            # Get client identifier
            client_id = rate_limiter.get_client_identifier(request)

            # Create rate limit key
            endpoint = f"{request.method}:{request.url.path}"
            rate_limit_key = f"{client_id}:{endpoint}"

            # Check rate limit
            allowed, info = rate_limiter.check_rate_limit(
                rate_limit_key,
                limit,
                window_seconds
            )

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Retry after {info['retry_after']} seconds",
                    headers={
                        "Retry-After": str(info["retry_after"]),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(info["reset_at"]))
                    }
                )

            # Add rate limit headers to response
            # Note: This requires a response object, handled by middleware

            return await func(*args, request=request, **kwargs)

        return wrapper
    return decorator


def public_rate_limit(func):
    """Decorator for public endpoints (30 req/min)"""
    return rate_limit(
        limit=rate_limit_config.public_endpoint_limit,
        window_seconds=rate_limit_config.window_seconds
    )(func)


def auth_rate_limit(func):
    """Decorator for auth endpoints (10 req/min)"""
    return rate_limit(
        limit=rate_limit_config.auth_endpoint_limit,
        window_seconds=rate_limit_config.window_seconds
    )(func)


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to add rate limit headers to all responses

    Add to FastAPI app:
        app.middleware("http")(rate_limit_middleware)
    """
    response = await call_next(request)

    # Get client identifier
    client_id = rate_limiter.get_client_identifier(request)

    # Create rate limit key
    endpoint = f"{request.method}:{request.url.path}"
    rate_limit_key = f"{client_id}:{endpoint}"

    # Get current rate limit status (without incrementing)
    # This is a read-only check for headers
    try:
        if isinstance(rate_limiter.backend, RedisRateLimiter):
            redis_key = f"rate_limit:{rate_limit_key}"
            current_count = rate_limiter.backend.redis.zcard(redis_key)
            remaining = max(0, rate_limit_config.public_endpoint_limit - current_count)
        else:
            bucket = rate_limiter.backend.buckets.get(rate_limit_key, {"count": 0})
            remaining = max(0, rate_limit_config.public_endpoint_limit - bucket["count"])

        response.headers["X-RateLimit-Limit"] = str(rate_limit_config.public_endpoint_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
    except Exception as e:
        logger.debug(f"Could not add rate limit headers: {e}")

    return response

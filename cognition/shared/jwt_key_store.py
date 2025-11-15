"""
T.A.R.S. JWT Key Store with Multi-Key Support
JWKS-style key rotation with kid (key ID) headers
Phase 12 Part 2
"""

import os
import json
import redis
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pydantic import BaseModel

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge
    METRICS_AVAILABLE = True

    jwt_issued_total = Counter(
        'jwt_issued_total',
        'Total JWT tokens issued',
        ['kid']  # key ID
    )

    jwt_verified_total = Counter(
        'jwt_verified_total',
        'Total JWT verification attempts',
        ['kid', 'status']  # success, expired, invalid_kid, invalid
    )

    jwt_rotation_total = Counter(
        'jwt_rotation_total',
        'Total JWT key rotations'
    )

    jwt_active_keys_count = Gauge(
        'jwt_active_keys_count',
        'Number of active JWT signing keys'
    )

except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class JWTKey(BaseModel):
    """JWT signing key"""
    kid: str  # Key ID (unique identifier)
    secret: str  # Secret key for HMAC signing
    algorithm: str = "HS256"  # Signing algorithm
    created_at: datetime
    expires_at: Optional[datetime] = None  # Optional expiration for old keys
    is_active: bool = True  # Whether this key can sign new tokens
    is_valid: bool = True  # Whether this key can verify tokens


class JWTKeyStore:
    """
    JWT key store with multi-key support and rotation

    Features:
    - Multiple active signing keys (JWKS-style)
    - kid (key ID) in JWT headers
    - Graceful key rotation with overlap period
    - Old keys remain valid for verification during grace period
    - Redis persistence for distributed systems
    - Prometheus metrics

    Rotation Strategy:
    1. New key is created with new kid
    2. New tokens use new kid
    3. Old keys remain valid for verification (grace period)
    4. After grace period, old keys are marked invalid
    5. Invalid keys are eventually deleted

    Redis Schema:
    - Hash: jwt_keys:{kid} -> JSON serialized JWTKey
    - String: jwt_keys:current_kid -> kid of current signing key
    - Set: jwt_keys:active -> set of active key IDs
    - Set: jwt_keys:valid -> set of valid (can verify) key IDs
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        key_prefix: str = "jwt_keys",
        grace_period_hours: int = 24
    ):
        """
        Initialize JWT key store

        Args:
            redis_host: Redis host (default: REDIS_HOST env or localhost)
            redis_port: Redis port (default: REDIS_PORT env or 6379)
            redis_db: Redis database number (default: 0)
            redis_password: Redis password (default: REDIS_PASSWORD env)
            key_prefix: Prefix for Redis keys (default: jwt_keys)
            grace_period_hours: Hours to keep old keys valid after rotation (default: 24)
        """
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = redis_db
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD")
        self.key_prefix = key_prefix
        self.grace_period_hours = grace_period_hours

        # In-memory fallback
        self.memory_keys: Dict[str, JWTKey] = {}
        self.current_kid: Optional[str] = None

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
            logger.info(f"JWT Key Store connected to Redis at {self.redis_host}:{self.redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("JWT Key Store will operate in-memory only (not suitable for multi-node)")
            self.redis_client = None

        # Initialize with current key if not exists
        self._initialize_default_key()

    def _key_path(self, kid: str) -> str:
        """Generate Redis key path for JWT key"""
        return f"{self.key_prefix}:{kid}"

    def _current_kid_key(self) -> str:
        """Redis key for current signing key ID"""
        return f"{self.key_prefix}:current_kid"

    def _active_set_key(self) -> str:
        """Redis set key for active signing keys"""
        return f"{self.key_prefix}:active"

    def _valid_set_key(self) -> str:
        """Redis set key for valid verification keys"""
        return f"{self.key_prefix}:valid"

    def _initialize_default_key(self):
        """Initialize with a default key if none exists"""
        current = self.get_current_key()
        if current is None:
            logger.info("No current JWT key found, creating default key")
            self.create_key()

    def create_key(self, secret: Optional[str] = None) -> JWTKey:
        """
        Create a new JWT signing key

        Args:
            secret: Optional secret (if None, generates random secret)

        Returns:
            New JWTKey
        """
        # Generate kid
        kid = f"key-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"

        # Generate or use provided secret
        if secret is None:
            secret = secrets.token_urlsafe(32)

        # Create key
        jwt_key = JWTKey(
            kid=kid,
            secret=secret,
            created_at=datetime.utcnow(),
            is_active=True,
            is_valid=True
        )

        # Store key
        if self.redis_client is not None:
            try:
                # Store key
                self.redis_client.set(self._key_path(kid), jwt_key.json())

                # Add to active and valid sets
                self.redis_client.sadd(self._active_set_key(), kid)
                self.redis_client.sadd(self._valid_set_key(), kid)

                logger.info(f"Created JWT key {kid} in Redis")

            except Exception as e:
                logger.error(f"Failed to store JWT key in Redis: {e}")
                # Fall through to in-memory

        # Store in memory (always)
        self.memory_keys[kid] = jwt_key

        # Set as current if no current key
        if self.get_current_kid() is None:
            self.set_current_kid(kid)

        # Metrics
        if METRICS_AVAILABLE:
            jwt_active_keys_count.set(len(self.list_active()))

        logger.info(f"Created new JWT signing key: {kid}")
        return jwt_key

    def get_key(self, kid: str) -> Optional[JWTKey]:
        """
        Get JWT key by kid

        Args:
            kid: Key ID

        Returns:
            JWTKey if found, None otherwise
        """
        # Try Redis first
        if self.redis_client is not None:
            try:
                key_json = self.redis_client.get(self._key_path(kid))
                if key_json:
                    return JWTKey.parse_raw(key_json)
            except Exception as e:
                logger.error(f"Failed to retrieve JWT key from Redis: {e}")

        # Fallback to memory
        return self.memory_keys.get(kid)

    def get_current_kid(self) -> Optional[str]:
        """Get current signing key ID"""
        # Try Redis first
        if self.redis_client is not None:
            try:
                kid = self.redis_client.get(self._current_kid_key())
                if kid:
                    return kid
            except Exception as e:
                logger.error(f"Failed to get current kid from Redis: {e}")

        # Fallback to memory
        return self.current_kid

    def set_current_kid(self, kid: str):
        """Set current signing key ID"""
        # Store in Redis
        if self.redis_client is not None:
            try:
                self.redis_client.set(self._current_kid_key(), kid)
            except Exception as e:
                logger.error(f"Failed to set current kid in Redis: {e}")

        # Store in memory
        self.current_kid = kid
        logger.info(f"Set current signing key to: {kid}")

    def get_current_key(self) -> Optional[JWTKey]:
        """Get current signing key"""
        kid = self.get_current_kid()
        if kid is None:
            return None
        return self.get_key(kid)

    def rotate_key(self) -> JWTKey:
        """
        Rotate JWT signing key

        Process:
        1. Create new key
        2. Set as current signing key
        3. Mark old key as inactive for signing (but valid for verification)
        4. Set expiration on old key

        Returns:
            New JWTKey
        """
        # Get old key
        old_kid = self.get_current_kid()
        old_key = self.get_key(old_kid) if old_kid else None

        # Create new key
        new_key = self.create_key()

        # Set new key as current
        self.set_current_kid(new_key.kid)

        # Deactivate old key (but keep valid for verification)
        if old_key:
            old_key.is_active = False
            old_key.expires_at = datetime.utcnow() + timedelta(hours=self.grace_period_hours)

            # Update in Redis
            if self.redis_client is not None:
                try:
                    self.redis_client.set(self._key_path(old_key.kid), old_key.json())
                    self.redis_client.srem(self._active_set_key(), old_key.kid)
                except Exception as e:
                    logger.error(f"Failed to update old key in Redis: {e}")

            # Update in memory
            self.memory_keys[old_key.kid] = old_key

            logger.info(f"Deactivated old key {old_key.kid}, expires at {old_key.expires_at}")

        # Metrics
        if METRICS_AVAILABLE:
            jwt_rotation_total.inc()
            jwt_active_keys_count.set(len(self.list_active()))

        logger.info(f"Rotated JWT key: {old_kid} -> {new_key.kid}")
        return new_key

    def list_active(self) -> List[JWTKey]:
        """List all active signing keys"""
        keys = []

        # Try Redis first
        if self.redis_client is not None:
            try:
                active_kids = self.redis_client.smembers(self._active_set_key())
                for kid in active_kids:
                    key = self.get_key(kid)
                    if key and key.is_active:
                        keys.append(key)
                return keys
            except Exception as e:
                logger.error(f"Failed to list active keys from Redis: {e}")

        # Fallback to memory
        return [k for k in self.memory_keys.values() if k.is_active]

    def list_valid(self) -> List[JWTKey]:
        """List all valid verification keys"""
        keys = []

        # Try Redis first
        if self.redis_client is not None:
            try:
                valid_kids = self.redis_client.smembers(self._valid_set_key())
                for kid in valid_kids:
                    key = self.get_key(kid)
                    if key and key.is_valid:
                        keys.append(key)
                return keys
            except Exception as e:
                logger.error(f"Failed to list valid keys from Redis: {e}")

        # Fallback to memory
        return [k for k in self.memory_keys.values() if k.is_valid]

    def cleanup_expired(self) -> int:
        """
        Clean up expired keys

        Returns:
            Number of keys deleted
        """
        deleted = 0
        now = datetime.utcnow()

        # Get all keys
        all_keys = self.list_valid()

        for key in all_keys:
            if key.expires_at and now > key.expires_at:
                # Mark as invalid
                key.is_valid = False

                # Update in Redis
                if self.redis_client is not None:
                    try:
                        self.redis_client.srem(self._valid_set_key(), key.kid)
                        # Optionally delete the key entirely
                        # self.redis_client.delete(self._key_path(key.kid))
                    except Exception as e:
                        logger.error(f"Failed to invalidate expired key {key.kid}: {e}")

                # Update in memory
                if key.kid in self.memory_keys:
                    self.memory_keys[key.kid] = key

                logger.info(f"Invalidated expired JWT key: {key.kid}")
                deleted += 1

        return deleted

    def health_check(self) -> Dict[str, any]:
        """Health check for JWT key store"""
        if self.redis_client is None:
            return {
                "status": "degraded",
                "redis_connected": False,
                "current_kid": self.get_current_kid(),
                "active_keys": len(self.list_active()),
                "valid_keys": len(self.list_valid())
            }

        try:
            self.redis_client.ping()
            return {
                "status": "healthy",
                "redis_connected": True,
                "current_kid": self.get_current_kid(),
                "active_keys": len(self.list_active()),
                "valid_keys": len(self.list_valid())
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e)
            }


# Global JWT key store instance
jwt_key_store = JWTKeyStore()

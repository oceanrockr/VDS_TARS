"""
T.A.R.S. Cognition Module Test Fixtures

Provides shared fixtures for cognition module testing:
- Mock Redis clients (async and sync)
- Mock database fixtures
- JWT manager fixtures
- Authentication fixtures
- Service mock fixtures
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add project root to path (two levels up from tests/cognition)
# This ensures cognition module is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Mock environment variables before imports
os.environ.setdefault('REDIS_HOST', 'localhost')
os.environ.setdefault('REDIS_PORT', '6379')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379')
os.environ.setdefault('REDIS_PASSWORD', 'test-password')
os.environ.setdefault('JWT_SECRET', 'test-secret-key-for-testing-only')
os.environ.setdefault('JWT_ALGORITHM', 'HS256')
os.environ.setdefault('JWT_EXPIRY_MINUTES', '60')
os.environ.setdefault('REFRESH_EXPIRY_DAYS', '7')
os.environ.setdefault('USE_REDIS', 'false')


# ============================================================================
# ASYNC REDIS FIXTURES
# ============================================================================

class AsyncMockRedis:
    """Mock async Redis client for testing"""

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.expire_times: Dict[str, datetime] = {}
        self.sorted_sets: Dict[str, Dict[str, float]] = {}

    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        if key in self.expire_times and datetime.utcnow() > self.expire_times[key]:
            del self.store[key]
            del self.expire_times[key]
            return None
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key-value with optional expiration"""
        self.store[key] = value
        if ex:
            self.expire_times[key] = datetime.utcnow() + timedelta(seconds=ex)
        return True

    async def delete(self, *keys: str) -> int:
        """Delete keys"""
        count = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                count += 1
            if key in self.expire_times:
                del self.expire_times[key]
            if key in self.sorted_sets:
                del self.sorted_sets[key]
        return count

    async def exists(self, *keys: str) -> int:
        """Check if keys exist"""
        count = 0
        for key in keys:
            if await self.get(key) is not None:
                count += 1
        return count

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members to sorted set"""
        if key not in self.sorted_sets:
            self.sorted_sets[key] = {}
        count = 0
        for member, score in mapping.items():
            if member not in self.sorted_sets[key]:
                count += 1
            self.sorted_sets[key][member] = score
        return count

    async def zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        withscores: bool = False
    ) -> list:
        """Get members by score range"""
        if key not in self.sorted_sets:
            return []

        members = [
            (member, score)
            for member, score in self.sorted_sets[key].items()
            if min_score <= score <= max_score
        ]

        if withscores:
            return members
        return [member for member, _ in members]

    async def zremrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float
    ) -> int:
        """Remove members by score range"""
        if key not in self.sorted_sets:
            return 0

        to_remove = [
            member
            for member, score in self.sorted_sets[key].items()
            if min_score <= score <= max_score
        ]

        for member in to_remove:
            del self.sorted_sets[key][member]

        return len(to_remove)

    async def zcard(self, key: str) -> int:
        """Get sorted set cardinality"""
        if key not in self.sorted_sets:
            return 0
        return len(self.sorted_sets[key])

    async def zrange(
        self,
        key: str,
        start: int,
        stop: int,
        withscores: bool = False
    ) -> list:
        """Get members by index range"""
        if key not in self.sorted_sets:
            return []

        # Sort by score
        sorted_members = sorted(
            self.sorted_sets[key].items(),
            key=lambda x: x[1]
        )

        # Apply range
        selected = sorted_members[start:stop+1] if stop >= 0 else sorted_members[start:]

        if withscores:
            return selected
        return [member for member, _ in selected]

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key"""
        if key in self.store or key in self.sorted_sets:
            self.expire_times[key] = datetime.utcnow() + timedelta(seconds=seconds)
            return True
        return False

    async def ping(self) -> bool:
        """Ping Redis"""
        return True

    async def flushdb(self) -> bool:
        """Flush database"""
        self.store.clear()
        self.expire_times.clear()
        self.sorted_sets.clear()
        return True


@pytest.fixture
async def mock_redis() -> AsyncMockRedis:
    """Provide a mock async Redis client"""
    redis = AsyncMockRedis()
    await redis.flushdb()
    return redis


@pytest.fixture
def mock_redis_sync() -> MagicMock:
    """
    Provide a synchronous mock Redis client with common methods.

    This is useful for testing synchronous code that uses Redis.
    For testing the rate limiter which uses sync Redis, this fixture
    provides the necessary methods.
    """
    mock = MagicMock()

    # Storage
    store: Dict[str, Any] = {}
    sorted_sets: Dict[str, Dict[str, float]] = {}
    expire_times: Dict[str, datetime] = {}

    def mock_get(key: str) -> Optional[str]:
        if key in expire_times and datetime.utcnow() > expire_times[key]:
            if key in store:
                del store[key]
            del expire_times[key]
            return None
        return store.get(key)

    def mock_set(key: str, value: str, ex: Optional[int] = None) -> bool:
        store[key] = value
        if ex:
            expire_times[key] = datetime.utcnow() + timedelta(seconds=ex)
        return True

    def mock_delete(*keys: str) -> int:
        count = 0
        for key in keys:
            if key in store:
                del store[key]
                count += 1
            if key in expire_times:
                del expire_times[key]
            if key in sorted_sets:
                del sorted_sets[key]
        return count

    def mock_zadd(key: str, mapping: Dict[str, float]) -> int:
        if key not in sorted_sets:
            sorted_sets[key] = {}
        count = 0
        for member, score in mapping.items():
            if member not in sorted_sets[key]:
                count += 1
            sorted_sets[key][member] = score
        return count

    def mock_zremrangebyscore(key: str, min_score: float, max_score: float) -> int:
        if key not in sorted_sets:
            return 0
        to_remove = [
            member
            for member, score in sorted_sets[key].items()
            if min_score <= score <= max_score
        ]
        for member in to_remove:
            del sorted_sets[key][member]
        return len(to_remove)

    def mock_zcard(key: str) -> int:
        if key not in sorted_sets:
            return 0
        return len(sorted_sets[key])

    def mock_zrange(key: str, start: int, stop: int, withscores: bool = False) -> list:
        if key not in sorted_sets:
            return []
        sorted_members = sorted(sorted_sets[key].items(), key=lambda x: x[1])
        selected = sorted_members[start:stop+1] if stop >= 0 else sorted_members[start:]
        if withscores:
            return selected
        return [member for member, _ in selected]

    def mock_expire(key: str, seconds: int) -> bool:
        if key in store or key in sorted_sets:
            expire_times[key] = datetime.utcnow() + timedelta(seconds=seconds)
            return True
        return False

    def mock_ping() -> bool:
        return True

    # Configure mock
    mock.get.side_effect = mock_get
    mock.set.side_effect = mock_set
    mock.delete.side_effect = mock_delete
    mock.zadd.side_effect = mock_zadd
    mock.zremrangebyscore.side_effect = mock_zremrangebyscore
    mock.zcard.side_effect = mock_zcard
    mock.zrange.side_effect = mock_zrange
    mock.expire.side_effect = mock_expire
    mock.ping.side_effect = mock_ping

    return mock


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

class MockDatabase:
    """Mock database for testing"""

    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.audit_logs: list[Dict[str, Any]] = []

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        return self.users.get(user_id)

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user"""
        user_id = user_data.get('user_id')
        self.users[user_id] = user_data
        return user_data

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user data"""
        if user_id in self.users:
            self.users[user_id].update(updates)
            return True
        return False

    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False

    async def create_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a session"""
        session_id = session_data.get('session_id')
        self.sessions[session_id] = session_data
        return session_data

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    async def add_audit_log(self, log_data: Dict[str, Any]) -> None:
        """Add audit log entry"""
        log_data['timestamp'] = datetime.utcnow()
        self.audit_logs.append(log_data)

    async def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> list[Dict[str, Any]]:
        """Get audit logs"""
        logs = self.audit_logs
        if user_id:
            logs = [log for log in logs if log.get('user_id') == user_id]
        return logs[-limit:]

    async def clear(self) -> None:
        """Clear all data"""
        self.users.clear()
        self.sessions.clear()
        self.audit_logs.clear()


@pytest.fixture
async def mock_database() -> AsyncGenerator[MockDatabase, None]:
    """Provide a mock database"""
    db = MockDatabase()
    yield db
    await db.clear()


# ============================================================================
# JWT MANAGER FIXTURES
# ============================================================================

@pytest.fixture
def jwt_manager() -> MagicMock:
    """
    Provide a mock JWT manager for testing authentication.

    This fixture provides methods for:
    - Creating access and refresh tokens
    - Verifying tokens
    - Managing JWT keys
    """
    from cognition.shared.auth import AuthService, User, Role

    # Use real AuthService for JWT operations
    auth_service = AuthService()

    # Create a mock that wraps the real service
    mock = MagicMock(spec=AuthService)
    mock.create_access_token = auth_service.create_access_token
    mock.create_refresh_token = auth_service.create_refresh_token
    mock.verify_token = auth_service.verify_token

    return mock


@pytest.fixture
def test_user():
    """Provide a test user"""
    from cognition.shared.auth import User, Role

    return User(
        user_id="test-user-001",
        username="testuser",
        roles=[Role.DEVELOPER],
        email="test@tars.ai"
    )


@pytest.fixture
def test_admin_user():
    """Provide a test admin user"""
    from cognition.shared.auth import User, Role

    return User(
        user_id="admin-001",
        username="admin",
        roles=[Role.ADMIN],
        email="admin@tars.ai"
    )


@pytest.fixture
def test_viewer_user():
    """Provide a test viewer user"""
    from cognition.shared.auth import User, Role

    return User(
        user_id="viewer-001",
        username="viewer",
        roles=[Role.VIEWER],
        email="viewer@tars.ai"
    )


@pytest.fixture
def access_token(jwt_manager, test_user):
    """Generate an access token for testing"""
    return jwt_manager.create_access_token(test_user)


@pytest.fixture
def admin_token(jwt_manager, test_admin_user):
    """Generate an admin access token for testing"""
    return jwt_manager.create_access_token(test_admin_user)


# ============================================================================
# FASTAPI REQUEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_request():
    """Provide a mock FastAPI Request object"""
    from fastapi import Request

    mock = MagicMock(spec=Request)
    mock.method = "GET"
    mock.url.path = "/api/test"
    mock.client.host = "127.0.0.1"
    mock.headers = {}

    return mock


# ============================================================================
# TIME FIXTURES
# ============================================================================

@pytest.fixture
def freeze_time():
    """Freeze time for testing"""
    import time

    frozen_time = 1700000000.0  # Fixed timestamp

    with patch('time.time', return_value=frozen_time):
        yield frozen_time


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state between tests"""
    yield
    # Clean up any rate limiter state
    try:
        from cognition.shared.rate_limiter import rate_limiter
        if hasattr(rate_limiter.backend, 'buckets'):
            rate_limiter.backend.buckets.clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def cleanup_env():
    """Cleanup environment variables after each test"""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

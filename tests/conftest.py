"""
Global pytest fixtures for T.A.R.S. Phase 12 QA Suite

Provides:
- Mock Redis clients
- Mock service clients
- Test FastAPI app instances
- Authentication fixtures
- Database fixtures
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Generator, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis import Redis

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock environment variables before imports
os.environ.setdefault('REDIS_HOST', 'localhost')
os.environ.setdefault('REDIS_PORT', '6379')
os.environ.setdefault('REDIS_PASSWORD', 'test-password')
os.environ.setdefault('JWT_SECRET', 'test-secret-key-for-testing-only')
os.environ.setdefault('JWT_ALGORITHM', 'HS256')
os.environ.setdefault('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '60')
os.environ.setdefault('JWT_REFRESH_TOKEN_EXPIRE_DAYS', '7')
os.environ.setdefault('ORCHESTRATION_SERVICE_URL', 'http://localhost:8094')
os.environ.setdefault('AUTOML_SERVICE_URL', 'http://localhost:8096')
os.environ.setdefault('HYPERSYNC_SERVICE_URL', 'http://localhost:8098')


# ============================================================================
# REDIS FIXTURES
# ============================================================================

class MockRedis:
    """Mock Redis client for testing"""

    def __init__(self):
        self.store: Dict[str, bytes] = {}
        self.expire_times: Dict[str, datetime] = {}
        self.hstore: Dict[str, Dict[str, bytes]] = {}

    def get(self, key: str) -> Optional[bytes]:
        """Get value by key"""
        if key in self.expire_times and datetime.utcnow() > self.expire_times[key]:
            del self.store[key]
            del self.expire_times[key]
            return None
        return self.store.get(key)

    def set(self, key: str, value: bytes, ex: Optional[int] = None) -> bool:
        """Set key-value with optional expiration"""
        self.store[key] = value
        if ex:
            self.expire_times[key] = datetime.utcnow() + timedelta(seconds=ex)
        return True

    def setex(self, key: str, time: int, value: bytes) -> bool:
        """Set key-value with expiration"""
        return self.set(key, value, ex=time)

    def delete(self, *keys: str) -> int:
        """Delete keys"""
        count = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                count += 1
            if key in self.expire_times:
                del self.expire_times[key]
        return count

    def exists(self, *keys: str) -> int:
        """Check if keys exist"""
        count = 0
        for key in keys:
            if self.get(key) is not None:
                count += 1
        return count

    def keys(self, pattern: str = '*') -> list:
        """Get keys matching pattern"""
        import re
        # Convert Redis pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        regex = re.compile(f'^{regex_pattern}$')
        return [k.encode() for k in self.store.keys() if regex.match(k)]

    def hset(self, name: str, key: str, value: bytes) -> int:
        """Set hash field"""
        if name not in self.hstore:
            self.hstore[name] = {}
        is_new = key not in self.hstore[name]
        self.hstore[name][key] = value
        return 1 if is_new else 0

    def hget(self, name: str, key: str) -> Optional[bytes]:
        """Get hash field"""
        return self.hstore.get(name, {}).get(key)

    def hgetall(self, name: str) -> Dict[bytes, bytes]:
        """Get all hash fields"""
        result = self.hstore.get(name, {})
        return {k.encode() if isinstance(k, str) else k: v
                for k, v in result.items()}

    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        if name not in self.hstore:
            return 0
        count = 0
        for key in keys:
            if key in self.hstore[name]:
                del self.hstore[name][key]
                count += 1
        return count

    def hkeys(self, name: str) -> list:
        """Get hash keys"""
        return [k.encode() if isinstance(k, str) else k
                for k in self.hstore.get(name, {}).keys()]

    def ping(self) -> bool:
        """Ping Redis"""
        return True

    def flushdb(self) -> bool:
        """Flush database"""
        self.store.clear()
        self.expire_times.clear()
        self.hstore.clear()
        return True


@pytest.fixture
def mock_redis() -> MockRedis:
    """Provide a mock Redis client"""
    return MockRedis()


@pytest.fixture
def mock_redis_client(mock_redis: MockRedis) -> Generator[MockRedis, None, None]:
    """Mock Redis client with automatic cleanup"""
    with patch('redis.Redis', return_value=mock_redis):
        with patch('redis.from_url', return_value=mock_redis):
            yield mock_redis
    # Cleanup
    mock_redis.flushdb()


# ============================================================================
# AUTHENTICATION FIXTURES
# ============================================================================

@pytest.fixture
def test_user_viewer() -> Dict:
    """Test user with viewer role"""
    return {
        "user_id": "test-viewer",
        "username": "viewer",
        "email": "viewer@test.com",
        "roles": ["viewer"]
    }


@pytest.fixture
def test_user_developer() -> Dict:
    """Test user with developer role"""
    return {
        "user_id": "test-developer",
        "username": "developer",
        "email": "developer@test.com",
        "roles": ["viewer", "developer"]
    }


@pytest.fixture
def test_user_admin() -> Dict:
    """Test user with admin role"""
    return {
        "user_id": "test-admin",
        "username": "admin",
        "email": "admin@test.com",
        "roles": ["viewer", "developer", "admin"]
    }


@pytest.fixture
def admin_token(test_user_admin: Dict) -> str:
    """Generate admin JWT token"""
    from cognition.shared.auth import create_access_token
    return create_access_token(test_user_admin)


@pytest.fixture
def developer_token(test_user_developer: Dict) -> str:
    """Generate developer JWT token"""
    from cognition.shared.auth import create_access_token
    return create_access_token(test_user_developer)


@pytest.fixture
def viewer_token(test_user_viewer: Dict) -> str:
    """Generate viewer JWT token"""
    from cognition.shared.auth import create_access_token
    return create_access_token(test_user_viewer)


# ============================================================================
# FASTAPI APP FIXTURES
# ============================================================================

@pytest.fixture
def admin_app(mock_redis_client: MockRedis) -> FastAPI:
    """Create test FastAPI app with admin routes"""
    from fastapi import FastAPI
    from dashboard.api.admin_routes import router as admin_router

    app = FastAPI(title="Test Admin API")
    app.include_router(admin_router, prefix="/admin")

    return app


@pytest.fixture
def test_client(admin_app: FastAPI) -> TestClient:
    """Create test client for admin app"""
    return TestClient(admin_app)


@pytest.fixture
def authenticated_client(test_client: TestClient, admin_token: str) -> TestClient:
    """Test client with admin authentication"""
    test_client.headers = {"Authorization": f"Bearer {admin_token}"}
    return test_client


# ============================================================================
# SERVICE MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_orchestration_service():
    """Mock orchestration service client"""
    mock = MagicMock()
    mock.get_agents.return_value = {
        "agents": [
            {
                "id": "dqn_agent",
                "name": "DQN Agent",
                "state": "active",
                "algorithm": "DQN",
                "reward": 0.75,
                "loss": 0.23,
                "entropy": 0.45
            },
            {
                "id": "ppo_agent",
                "name": "PPO Agent",
                "state": "training",
                "algorithm": "PPO",
                "reward": 0.82,
                "loss": 0.18,
                "entropy": 0.52
            }
        ]
    }
    mock.get_agent.return_value = {
        "id": "dqn_agent",
        "name": "DQN Agent",
        "state": "active",
        "algorithm": "DQN",
        "reward": 0.75,
        "loss": 0.23,
        "entropy": 0.45,
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 0.1
        }
    }
    mock.reload_agent.return_value = {"success": True, "message": "Agent reloaded"}
    mock.promote_model.return_value = {"success": True, "message": "Model promoted"}
    return mock


@pytest.fixture
def mock_automl_service():
    """Mock AutoML service client"""
    mock = MagicMock()
    mock.get_trials.return_value = {
        "trials": [
            {
                "trial_id": 1,
                "params": {"lr": 0.001, "gamma": 0.99},
                "value": 0.85,
                "state": "COMPLETE"
            }
        ]
    }
    mock.get_search_status.return_value = {
        "status": "running",
        "best_trial": 1,
        "best_value": 0.85
    }
    return mock


@pytest.fixture
def mock_hypersync_service():
    """Mock HyperSync service client"""
    mock = MagicMock()
    mock.get_proposals.return_value = {
        "proposals": [
            {
                "proposal_id": "prop-1",
                "agent_id": "dqn_agent",
                "params": {"lr": 0.002},
                "reward_improvement": 0.15,
                "status": "pending"
            }
        ]
    }
    mock.approve_proposal.return_value = {"success": True, "message": "Proposal approved"}
    return mock


# ============================================================================
# TIME FIXTURES
# ============================================================================

@pytest.fixture
def freeze_time():
    """Freeze time for testing"""
    with patch('dashboard.api.admin_routes.datetime') as mock_dt:
        mock_dt.utcnow.return_value = datetime(2025, 11, 15, 12, 0, 0)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_dt


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_metrics():
    """Cleanup Prometheus metrics after each test"""
    yield
    # Reset all Prometheus collectors
    from prometheus_client import REGISTRY
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances after each test"""
    yield
    # Reset JWT key store
    try:
        from cognition.shared.jwt_key_store import JWTKeyStore
        if hasattr(JWTKeyStore, '_instance'):
            JWTKeyStore._instance = None
    except ImportError:
        pass

    # Reset API key store
    try:
        from cognition.shared.api_key_store import APIKeyStore
        if hasattr(APIKeyStore, '_instance'):
            APIKeyStore._instance = None
    except ImportError:
        pass

    # Reset audit logger
    try:
        from cognition.shared.audit_logger import AuditLogger
        if hasattr(AuditLogger, '_instance'):
            AuditLogger._instance = None
    except ImportError:
        pass

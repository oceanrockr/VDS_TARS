"""
T.A.R.S. Test Configuration
Pytest fixtures and configuration for tests
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio

from app.main import app
from app.core.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for synchronous tests"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client():
    """Create an async test client for async tests"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_client_id():
    """Test client ID"""
    return "test-client-123"


@pytest.fixture
def test_token(client, test_client_id):
    """Generate a test JWT token"""
    response = client.post(
        "/auth/token",
        json={
            "client_id": test_client_id,
            "device_name": "pytest",
            "device_type": "test",
        },
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(test_token):
    """Create authorization headers with test token"""
    return {"Authorization": f"Bearer {test_token}"}

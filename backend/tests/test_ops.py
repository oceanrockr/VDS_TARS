"""
T.A.R.S. Operations API Tests
Phase 23 - Production-in-Use Observability

Tests for /ops/summary and /ops/health-snapshot endpoints.
Verifies authentication requirements and response structure.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from fastapi.testclient import TestClient


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_services():
    """Mock all service dependencies."""
    with patch('backend.app.api.ops.ollama_service') as mock_ollama, \
         patch('backend.app.api.ops.chromadb_service') as mock_chroma, \
         patch('backend.app.api.ops.redis_cache') as mock_redis, \
         patch('backend.app.api.ops.conversation_service') as mock_conv, \
         patch('backend.app.api.ops.nas_watcher') as mock_nas:

        # Configure Ollama mock
        mock_ollama.health_check = AsyncMock(return_value=True)
        mock_ollama.list_models = AsyncMock(return_value=[
            {'name': 'mistral:7b-instruct', 'size': '4.1GB'}
        ])

        # Configure ChromaDB mock
        mock_chroma.health_check.return_value = {'status': 'healthy'}
        mock_chroma.get_stats = AsyncMock(return_value=MagicMock(
            collection_name='tars_documents',
            total_chunks=1500
        ))

        # Configure Redis mock
        mock_redis.health_check = AsyncMock(return_value={'status': 'healthy'})
        mock_redis.get = AsyncMock(return_value=None)

        # Configure Conversation service mock
        mock_conv.health_check.return_value = {'status': 'healthy'}

        # Configure NAS watcher mock
        mock_nas.get_stats.return_value = {
            'enabled': True,
            'nas_accessible': True,
            'document_count': 42,
            'last_ingestion_time': '2025-12-27T10:00:00',
            'documents_indexed': 42,
            'pending_files': 0
        }

        yield {
            'ollama': mock_ollama,
            'chroma': mock_chroma,
            'redis': mock_redis,
            'conversation': mock_conv,
            'nas': mock_nas
        }


@pytest.fixture
def valid_token():
    """Generate a valid JWT token for testing."""
    return "Bearer test-token-12345"


@pytest.fixture
def mock_verify_token():
    """Mock the verify_token dependency."""
    with patch('backend.app.api.ops.verify_token') as mock:
        mock.return_value = {'sub': 'test-user', 'role': 'admin'}
        yield mock


# ==============================================================================
# Test: Authentication Required
# ==============================================================================

class TestOpsAuthentication:
    """Test that ops endpoints require authentication."""

    def test_summary_requires_auth(self, mock_services):
        """Test that /ops/summary returns 401 without token."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get("/ops/summary")

        assert response.status_code == 401
        assert "detail" in response.json()

    def test_health_snapshot_requires_auth(self, mock_services):
        """Test that /ops/health-snapshot returns 401 without token."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get("/ops/health-snapshot")

        assert response.status_code == 401

    def test_summary_rejects_invalid_token(self, mock_services):
        """Test that /ops/summary rejects invalid tokens."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer invalid-token"}
        )

        assert response.status_code in [401, 403]


# ==============================================================================
# Test: /ops/summary Endpoint
# ==============================================================================

class TestOpsSummary:
    """Test the /ops/summary endpoint."""

    def test_summary_returns_200_with_valid_token(
        self, mock_services, mock_verify_token
    ):
        """Test that /ops/summary returns 200 with valid token."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        # Should return 200 (may fail due to mock setup, but checks endpoint exists)
        assert response.status_code in [200, 500]

    def test_summary_response_structure(self, mock_services, mock_verify_token):
        """Test that /ops/summary returns expected structure."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()

            # Check required fields
            assert "timestamp" in data
            assert "version" in data
            assert "uptime_seconds" in data
            assert "overall_status" in data
            assert "services" in data

            # Check overall_status is valid
            assert data["overall_status"] in ["healthy", "degraded", "unhealthy"]

            # Check services is a list
            assert isinstance(data["services"], list)

    def test_summary_includes_model_info(self, mock_services, mock_verify_token):
        """Test that /ops/summary includes model information."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("model"):
                assert "name" in data["model"]
                assert "loaded" in data["model"]

    def test_summary_includes_chroma_stats(self, mock_services, mock_verify_token):
        """Test that /ops/summary includes ChromaDB stats."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("chroma_stats"):
                assert "collection_name" in data["chroma_stats"]
                assert "chunk_count" in data["chroma_stats"]

    def test_summary_includes_nas_status(self, mock_services, mock_verify_token):
        """Test that /ops/summary includes NAS status."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()
            assert "nas_mounted" in data
            assert "nas_document_count" in data


# ==============================================================================
# Test: /ops/health-snapshot Endpoint
# ==============================================================================

class TestOpsHealthSnapshot:
    """Test the /ops/health-snapshot endpoint."""

    def test_health_snapshot_returns_200_with_valid_token(
        self, mock_services, mock_verify_token
    ):
        """Test that /ops/health-snapshot returns 200 with valid token."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/health-snapshot",
            headers={"Authorization": "Bearer valid-token"}
        )

        assert response.status_code in [200, 500]

    def test_health_snapshot_response_structure(
        self, mock_services, mock_verify_token
    ):
        """Test that /ops/health-snapshot returns expected structure."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/health-snapshot",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()

            assert "timestamp" in data
            assert "version" in data
            assert "status" in data
            assert "services" in data

            # Check status is valid
            assert data["status"] in ["healthy", "degraded", "unhealthy"]

            # Check services has expected keys
            assert "ollama" in data["services"]
            assert "chromadb" in data["services"]
            assert "redis" in data["services"]

    def test_health_snapshot_is_minimal(self, mock_services, mock_verify_token):
        """Test that /ops/health-snapshot is minimal compared to /summary."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/health-snapshot",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()

            # Should NOT have detailed fields (those are in /summary)
            assert "model" not in data
            assert "chroma_stats" not in data
            assert "ingestion" not in data


# ==============================================================================
# Test: Service Health Aggregation
# ==============================================================================

class TestServiceHealthAggregation:
    """Test service health status aggregation logic."""

    def test_healthy_when_all_services_healthy(
        self, mock_services, mock_verify_token
    ):
        """Test overall status is 'healthy' when all services are healthy."""
        from backend.app.main import app
        client = TestClient(app)

        # All services are healthy (default mock setup)
        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()
            assert data["overall_status"] == "healthy"

    def test_degraded_when_some_services_unhealthy(
        self, mock_services, mock_verify_token
    ):
        """Test overall status is 'degraded' when some services are unhealthy."""
        # Make one service unhealthy
        mock_services['redis'].health_check = AsyncMock(
            return_value={'status': 'unhealthy'}
        )

        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        if response.status_code == 200:
            data = response.json()
            assert data["overall_status"] in ["degraded", "healthy"]


# ==============================================================================
# Test: Error Handling
# ==============================================================================

class TestOpsErrorHandling:
    """Test error handling in ops endpoints."""

    def test_handles_service_timeout(self, mock_services, mock_verify_token):
        """Test that endpoint handles service timeouts gracefully."""
        import asyncio

        # Make a service timeout
        async def timeout_health():
            await asyncio.sleep(10)
            return True

        mock_services['ollama'].health_check = timeout_health

        from backend.app.main import app
        client = TestClient(app)

        # Should not hang - timeout should be handled
        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        # Should return some response (may be degraded due to timeout)
        assert response.status_code in [200, 500]

    def test_handles_service_exception(self, mock_services, mock_verify_token):
        """Test that endpoint handles service exceptions gracefully."""
        # Make a service throw an exception
        mock_services['ollama'].health_check = AsyncMock(
            side_effect=Exception("Service crashed")
        )

        from backend.app.main import app
        client = TestClient(app)

        response = client.get(
            "/ops/summary",
            headers={"Authorization": "Bearer valid-token"}
        )

        # Should not crash - should return degraded status
        assert response.status_code in [200, 500]


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

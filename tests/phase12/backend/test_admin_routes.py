"""
Test suite for Admin API routes (dashboard/api/admin_routes.py)

Coverage targets:
- All admin endpoints (agents, API keys, JWT, system health, audit logs)
- RBAC enforcement (viewer, developer, admin)
- Request/response validation
- Error handling
- State mutations
- Audit logging integration
"""

import json
from datetime import datetime, timedelta
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ============================================================================
# AGENT MANAGEMENT ENDPOINTS
# ============================================================================

@pytest.mark.admin
@pytest.mark.integration
class TestAgentManagement:
    """Test agent management endpoints"""

    @patch('dashboard.api.admin_routes.httpx.get')
    def test_get_all_agents_success(
        self,
        mock_get: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/agents returns all agents"""
        # Mock orchestration service response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agents": [
                {
                    "id": "dqn_agent",
                    "name": "DQN Agent",
                    "state": "active",
                    "algorithm": "DQN",
                    "reward": 0.75
                },
                {
                    "id": "ppo_agent",
                    "name": "PPO Agent",
                    "state": "training",
                    "algorithm": "PPO",
                    "reward": 0.82
                }
            ]
        }
        mock_get.return_value = mock_response

        response = authenticated_client.get("/admin/agents")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) == 2
        assert data["agents"][0]["id"] == "dqn_agent"
        assert data["agents"][1]["id"] == "ppo_agent"

    def test_get_all_agents_requires_auth(self, test_client: TestClient):
        """Test GET /admin/agents requires authentication"""
        response = test_client.get("/admin/agents")
        assert response.status_code == 401

    @patch('dashboard.api.admin_routes.httpx.get')
    def test_get_all_agents_requires_admin_role(
        self,
        mock_get: MagicMock,
        test_client: TestClient,
        viewer_token: str
    ):
        """Test GET /admin/agents requires admin role"""
        test_client.headers = {"Authorization": f"Bearer {viewer_token}"}
        response = test_client.get("/admin/agents")
        assert response.status_code == 403

    @patch('dashboard.api.admin_routes.httpx.get')
    def test_get_agent_by_id_success(
        self,
        mock_get: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/agents/{id} returns specific agent"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "dqn_agent",
            "name": "DQN Agent",
            "state": "active",
            "algorithm": "DQN",
            "reward": 0.75,
            "hyperparameters": {
                "learning_rate": 0.001,
                "gamma": 0.99
            }
        }
        mock_get.return_value = mock_response

        response = authenticated_client.get("/admin/agents/dqn_agent")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "dqn_agent"
        assert "hyperparameters" in data

    @patch('dashboard.api.admin_routes.httpx.get')
    def test_get_agent_not_found(
        self,
        mock_get: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/agents/{id} with non-existent agent"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        response = authenticated_client.get("/admin/agents/nonexistent")
        assert response.status_code == 404

    @patch('dashboard.api.admin_routes.httpx.post')
    @patch('dashboard.api.admin_routes.audit_logger')
    def test_reload_agent_success(
        self,
        mock_audit: MagicMock,
        mock_post: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/agents/{id}/reload"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Agent reloaded successfully"
        }
        mock_post.return_value = mock_response

        response = authenticated_client.post(
            "/admin/agents/dqn_agent/reload",
            json={"reason": "Performance degradation"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify audit log was created
        mock_audit.log.assert_called_once()
        log_call = mock_audit.log.call_args
        assert log_call[1]["event_type"] == "AGENT_RELOAD"
        assert log_call[1]["severity"] == "MEDIUM"

    @patch('dashboard.api.admin_routes.httpx.post')
    @patch('dashboard.api.admin_routes.audit_logger')
    def test_promote_model_success(
        self,
        mock_audit: MagicMock,
        mock_post: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/agents/{id}/promote"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Model promoted to production"
        }
        mock_post.return_value = mock_response

        response = authenticated_client.post(
            "/admin/agents/dqn_agent/promote",
            json={
                "version": "v1.2.0",
                "reason": "Reward improvement +15%"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify audit log
        mock_audit.log.assert_called_once()
        log_call = mock_audit.log.call_args
        assert log_call[1]["event_type"] == "MODEL_PROMOTION"
        assert log_call[1]["severity"] == "HIGH"


# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================

@pytest.mark.admin
@pytest.mark.integration
@pytest.mark.apikey
class TestAPIKeyManagement:
    """Test API key management endpoints"""

    @patch('dashboard.api.admin_routes.api_key_store')
    def test_list_api_keys_success(
        self,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/api-keys returns all keys"""
        mock_store.list_keys.return_value = [
            {
                "id": "key-1",
                "service_name": "Test Service",
                "created_at": "2025-11-15T12:00:00Z",
                "expires_at": None,
                "is_active": True
            },
            {
                "id": "key-2",
                "service_name": "Another Service",
                "created_at": "2025-11-14T10:00:00Z",
                "expires_at": "2026-11-14T10:00:00Z",
                "is_active": True
            }
        ]

        response = authenticated_client.get("/admin/api-keys")

        assert response.status_code == 200
        data = response.json()
        assert "keys" in data
        assert len(data["keys"]) == 2
        assert data["total"] == 2

    @patch('dashboard.api.admin_routes.api_key_store')
    @patch('dashboard.api.admin_routes.audit_logger')
    def test_create_api_key_success(
        self,
        mock_audit: MagicMock,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/api-keys creates new key"""
        mock_store.create_key.return_value = {
            "id": "new-key-id",
            "key": "tars_1234567890abcdef",
            "service_name": "New Service",
            "created_at": "2025-11-15T12:00:00Z",
            "expires_at": None
        }

        response = authenticated_client.post(
            "/admin/api-keys",
            json={
                "service_name": "New Service",
                "expires_in_days": None
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "new-key-id"
        assert data["key"] == "tars_1234567890abcdef"
        assert "WARNING" in data["message"]

        # Verify audit log
        mock_audit.log.assert_called_once()
        log_call = mock_audit.log.call_args
        assert log_call[1]["event_type"] == "API_KEY_CREATED"

    @patch('dashboard.api.admin_routes.api_key_store')
    def test_create_api_key_with_expiration(
        self,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/api-keys with expiration"""
        expires_at = (datetime.utcnow() + timedelta(days=365)).isoformat() + "Z"
        mock_store.create_key.return_value = {
            "id": "new-key-id",
            "key": "tars_abcdef123456",
            "service_name": "Expiring Service",
            "created_at": "2025-11-15T12:00:00Z",
            "expires_at": expires_at
        }

        response = authenticated_client.post(
            "/admin/api-keys",
            json={
                "service_name": "Expiring Service",
                "expires_in_days": 365
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["expires_at"] == expires_at

    @patch('dashboard.api.admin_routes.api_key_store')
    @patch('dashboard.api.admin_routes.audit_logger')
    def test_rotate_api_key_success(
        self,
        mock_audit: MagicMock,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/api-keys/{id}/rotate"""
        mock_store.rotate_key.return_value = {
            "id": "key-1",
            "key": "tars_newkey123456",
            "service_name": "Test Service",
            "rotated_at": "2025-11-15T12:00:00Z"
        }

        response = authenticated_client.post("/admin/api-keys/key-1/rotate")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "key-1"
        assert data["key"] == "tars_newkey123456"

        # Verify audit log
        mock_audit.log.assert_called_once()
        log_call = mock_audit.log.call_args
        assert log_call[1]["event_type"] == "API_KEY_ROTATED"

    @patch('dashboard.api.admin_routes.api_key_store')
    def test_rotate_api_key_not_found(
        self,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/api-keys/{id}/rotate with non-existent key"""
        mock_store.rotate_key.side_effect = ValueError("API key not found")

        response = authenticated_client.post("/admin/api-keys/nonexistent/rotate")
        assert response.status_code == 404

    @patch('dashboard.api.admin_routes.api_key_store')
    @patch('dashboard.api.admin_routes.audit_logger')
    def test_revoke_api_key_success(
        self,
        mock_audit: MagicMock,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/api-keys/{id}/revoke"""
        mock_store.revoke_key.return_value = True

        response = authenticated_client.post(
            "/admin/api-keys/key-1/revoke",
            json={"reason": "Security incident"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify audit log with reason
        mock_audit.log.assert_called_once()
        log_call = mock_audit.log.call_args
        assert log_call[1]["event_type"] == "API_KEY_REVOKED"
        assert log_call[1]["metadata"]["reason"] == "Security incident"


# ============================================================================
# JWT KEY MANAGEMENT ENDPOINTS
# ============================================================================

@pytest.mark.admin
@pytest.mark.integration
@pytest.mark.jwt
class TestJWTKeyManagement:
    """Test JWT key management endpoints"""

    @patch('dashboard.api.admin_routes.jwt_key_store')
    def test_get_jwt_status_success(
        self,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/jwt/status"""
        mock_store.get_current_key.return_value = MagicMock(
            kid="key-current-123",
            algorithm="HS256",
            created_at=datetime(2025, 11, 15, 12, 0, 0),
            is_active=True,
            is_valid=True
        )
        mock_store.list_keys.return_value = [
            MagicMock(
                kid="key-current-123",
                is_active=True,
                is_valid=True
            ),
            MagicMock(
                kid="key-old-456",
                is_active=False,
                is_valid=True
            )
        ]

        response = authenticated_client.get("/admin/jwt/status")

        assert response.status_code == 200
        data = response.json()
        assert data["current_kid"] == "key-current-123"
        assert data["total_active"] == 1
        assert data["total_valid"] == 2

    @patch('dashboard.api.admin_routes.jwt_key_store')
    @patch('dashboard.api.admin_routes.audit_logger')
    def test_rotate_jwt_key_success(
        self,
        mock_audit: MagicMock,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/jwt/rotate"""
        old_key = MagicMock(kid="key-old-123")
        new_key = MagicMock(
            kid="key-new-456",
            algorithm="HS256",
            created_at=datetime(2025, 11, 15, 12, 0, 0)
        )
        mock_store.get_current_key.return_value = old_key
        mock_store.rotate_key.return_value = new_key

        response = authenticated_client.post("/admin/jwt/rotate")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["old_kid"] == "key-old-123"
        assert data["new_kid"] == "key-new-456"
        assert data["grace_period_hours"] == 24

        # Verify audit log
        mock_audit.log.assert_called_once()
        log_call = mock_audit.log.call_args
        assert log_call[1]["event_type"] == "JWT_ROTATION"
        assert log_call[1]["severity"] == "HIGH"

    @patch('dashboard.api.admin_routes.jwt_key_store')
    def test_list_jwt_keys_success(
        self,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/jwt/keys"""
        mock_store.list_keys.return_value = [
            MagicMock(
                kid="key-1",
                algorithm="HS256",
                created_at=datetime(2025, 11, 15, 12, 0, 0),
                expires_at=None,
                is_active=True,
                is_valid=True
            ),
            MagicMock(
                kid="key-2",
                algorithm="HS256",
                created_at=datetime(2025, 11, 14, 12, 0, 0),
                expires_at=datetime(2025, 11, 14, 13, 0, 0),
                is_active=False,
                is_valid=False
            )
        ]
        mock_store.get_current_key.return_value = mock_store.list_keys.return_value[0]

        response = authenticated_client.get("/admin/jwt/keys")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["current_kid"] == "key-1"
        assert len(data["keys"]) == 2

    @patch('dashboard.api.admin_routes.jwt_key_store')
    @patch('dashboard.api.admin_routes.audit_logger')
    def test_invalidate_jwt_key_success(
        self,
        mock_audit: MagicMock,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/jwt/keys/{kid}/invalidate"""
        mock_store.invalidate_key.return_value = True

        response = authenticated_client.post(
            "/admin/jwt/keys/key-123/invalidate",
            json={"reason": "Key compromised"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify audit log with CRITICAL severity
        mock_audit.log.assert_called_once()
        log_call = mock_audit.log.call_args
        assert log_call[1]["event_type"] == "JWT_KEY_INVALIDATED"
        assert log_call[1]["severity"] == "CRITICAL"
        assert log_call[1]["metadata"]["reason"] == "Key compromised"

    @patch('dashboard.api.admin_routes.jwt_key_store')
    def test_invalidate_jwt_key_not_found(
        self,
        mock_store: MagicMock,
        authenticated_client: TestClient
    ):
        """Test POST /admin/jwt/keys/{kid}/invalidate with non-existent key"""
        mock_store.invalidate_key.side_effect = ValueError("JWT key not found")

        response = authenticated_client.post(
            "/admin/jwt/keys/nonexistent/invalidate",
            json={"reason": "Test"}
        )
        assert response.status_code == 404


# ============================================================================
# SYSTEM HEALTH ENDPOINTS
# ============================================================================

@pytest.mark.admin
@pytest.mark.integration
class TestSystemHealth:
    """Test system health endpoints"""

    @patch('dashboard.api.admin_routes.httpx.get')
    @patch('dashboard.api.admin_routes.redis_client')
    def test_get_system_health_all_healthy(
        self,
        mock_redis: MagicMock,
        mock_httpx: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/system/health with all services healthy"""
        # Mock Redis ping
        mock_redis.ping.return_value = True

        # Mock service health checks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_httpx.return_value = mock_response

        response = authenticated_client.get("/admin/system/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["redis"] == "healthy"
        assert data["services"]["orchestration"] == "healthy"

    @patch('dashboard.api.admin_routes.httpx.get')
    @patch('dashboard.api.admin_routes.redis_client')
    def test_get_system_health_redis_down(
        self,
        mock_redis: MagicMock,
        mock_httpx: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/system/health with Redis down"""
        # Mock Redis failure
        mock_redis.ping.side_effect = Exception("Connection refused")

        response = authenticated_client.get("/admin/system/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["redis"] == "unhealthy"


# ============================================================================
# AUDIT LOG ENDPOINTS
# ============================================================================

@pytest.mark.admin
@pytest.mark.integration
@pytest.mark.audit
class TestAuditLogs:
    """Test audit log endpoints"""

    @patch('dashboard.api.admin_routes.audit_logger')
    def test_get_audit_logs_success(
        self,
        mock_audit: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/audit-logs"""
        mock_audit.get_logs.return_value = [
            {
                "id": "log-1",
                "event_type": "API_KEY_CREATED",
                "user_id": "admin",
                "timestamp": "2025-11-15T12:00:00Z",
                "severity": "MEDIUM"
            },
            {
                "id": "log-2",
                "event_type": "JWT_ROTATION",
                "user_id": "admin",
                "timestamp": "2025-11-15T11:00:00Z",
                "severity": "HIGH"
            }
        ]

        response = authenticated_client.get("/admin/audit-logs")

        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert len(data["logs"]) == 2

    @patch('dashboard.api.admin_routes.audit_logger')
    def test_get_audit_logs_with_filters(
        self,
        mock_audit: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/audit-logs with filters"""
        mock_audit.get_logs.return_value = [
            {
                "id": "log-1",
                "event_type": "JWT_ROTATION",
                "user_id": "admin",
                "timestamp": "2025-11-15T12:00:00Z",
                "severity": "HIGH"
            }
        ]

        response = authenticated_client.get(
            "/admin/audit-logs",
            params={
                "event_type": "JWT_ROTATION",
                "limit": 10
            }
        )

        assert response.status_code == 200
        mock_audit.get_logs.assert_called_once()

    @patch('dashboard.api.admin_routes.audit_logger')
    def test_get_audit_stats_success(
        self,
        mock_audit: MagicMock,
        authenticated_client: TestClient
    ):
        """Test GET /admin/audit-logs/stats"""
        mock_audit.get_stats.return_value = {
            "total_events": 150,
            "events_by_type": {
                "API_KEY_CREATED": 45,
                "JWT_ROTATION": 12,
                "AGENT_RELOAD": 32
            },
            "events_by_severity": {
                "LOW": 50,
                "MEDIUM": 60,
                "HIGH": 30,
                "CRITICAL": 10
            }
        }

        response = authenticated_client.get("/admin/audit-logs/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 150
        assert "events_by_type" in data
        assert "events_by_severity" in data

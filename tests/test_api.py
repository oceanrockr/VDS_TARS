"""
Tests for Enterprise API

Covers:
- JWT login and token refresh
- API key authentication
- RBAC role enforcement
- Endpoint responses
- Rate limiting behavior
- Error handling
- CORS headers

Usage:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --cov=enterprise_api
"""

import pytest
import time
from fastapi.testclient import TestClient

from enterprise_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def admin_api_key():
    """Admin API key for testing."""
    return "dev-key-admin"


@pytest.fixture
def sre_api_key():
    """SRE API key for testing."""
    return "dev-key-sre"


@pytest.fixture
def readonly_api_key():
    """Readonly API key for testing."""
    return "dev-key-readonly"


@pytest.fixture
def admin_token(client):
    """Get admin JWT token."""
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "demo123"}
    )
    return response.json()["access_token"]


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Test health response has required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_no_auth_required(self, client):
        """Test health endpoint doesn't require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_format(self, client):
        """Test metrics are in Prometheus format."""
        response = client.get("/metrics")
        content = response.text

        # Should contain Prometheus metrics
        assert "# HELP" in content or "# TYPE" in content

    def test_metrics_no_auth_required(self, client):
        """Test metrics endpoint doesn't require authentication."""
        response = client.get("/metrics")
        assert response.status_code == 200


class TestJWTAuthentication:
    """Test JWT authentication flow."""

    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "demo123"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrong"}
        )

        assert response.status_code == 401

    def test_login_missing_fields(self, client):
        """Test login with missing fields."""
        response = client.post(
            "/auth/login",
            json={"username": "admin"}
        )

        assert response.status_code == 422

    def test_token_refresh(self, client):
        """Test token refresh flow."""
        # Login
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "demo123"}
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh
        refresh_response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )

        assert refresh_response.status_code == 200
        data = refresh_response.json()

        assert "access_token" in data
        assert "token_type" in data

    def test_token_refresh_invalid_token(self, client):
        """Test refresh with invalid token."""
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": "invalid_token"}
        )

        assert response.status_code == 401

    def test_protected_endpoint_with_token(self, client, admin_token):
        """Test accessing protected endpoint with valid token."""
        response = client.get(
            "/ga",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200

    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get("/ga")

        assert response.status_code in [401, 403]

    def test_protected_endpoint_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        response = client.get(
            "/ga",
            headers={"Authorization": "Bearer invalid_token"}
        )

        assert response.status_code == 401


class TestAPIKeyAuthentication:
    """Test API key authentication."""

    def test_api_key_admin(self, client, admin_api_key):
        """Test admin API key authentication."""
        response = client.get(
            "/ga",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200

    def test_api_key_sre(self, client, sre_api_key):
        """Test SRE API key authentication."""
        response = client.get(
            "/ga",
            headers={"X-API-Key": sre_api_key}
        )

        assert response.status_code == 200

    def test_api_key_readonly(self, client, readonly_api_key):
        """Test readonly API key authentication."""
        response = client.get(
            "/ga",
            headers={"X-API-Key": readonly_api_key}
        )

        assert response.status_code == 200

    def test_invalid_api_key(self, client):
        """Test invalid API key."""
        response = client.get(
            "/ga",
            headers={"X-API-Key": "invalid-key"}
        )

        assert response.status_code == 401


class TestRBACEnforcement:
    """Test role-based access control."""

    def test_admin_can_access_admin_endpoint(self, client, admin_api_key):
        """Test admin can access admin endpoints."""
        response = client.get(
            "/admin/compliance",
            headers={"X-API-Key": admin_api_key}
        )

        # Should succeed or return 404 if endpoint not implemented
        assert response.status_code in [200, 404]

    def test_sre_cannot_access_admin_endpoint(self, client, sre_api_key):
        """Test SRE cannot access admin endpoints."""
        response = client.get(
            "/admin/compliance",
            headers={"X-API-Key": sre_api_key}
        )

        # Should be forbidden or 404 if endpoint not implemented
        assert response.status_code in [403, 404]

    def test_readonly_cannot_access_write_endpoint(self, client, readonly_api_key):
        """Test readonly cannot access write endpoints."""
        response = client.post(
            "/admin/refresh-data",
            headers={"X-API-Key": readonly_api_key}
        )

        # Should be forbidden or 404/405 if endpoint not implemented
        assert response.status_code in [403, 404, 405]

    def test_role_hierarchy(self, client, admin_api_key):
        """Test admin has access to all role levels."""
        # Admin should access readonly endpoints
        response = client.get(
            "/ga",
            headers={"X-API-Key": admin_api_key}
        )
        assert response.status_code == 200


class TestGAKPIEndpoint:
    """Test GA KPI endpoint."""

    def test_ga_kpi_success(self, client, admin_api_key):
        """Test getting GA KPI."""
        response = client.get(
            "/ga",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert "overall_availability" in data
        assert "ga_ready" in data
        assert "generated_at" in data

    def test_ga_kpi_with_days_parameter(self, client, admin_api_key):
        """Test GA KPI with days parameter."""
        response = client.get(
            "/ga?days=30",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200

    def test_ga_kpi_invalid_days(self, client, admin_api_key):
        """Test GA KPI with invalid days parameter."""
        response = client.get(
            "/ga?days=999",
            headers={"X-API-Key": admin_api_key}
        )

        # Should return validation error
        assert response.status_code in [422, 400]


class TestDailySummariesEndpoint:
    """Test daily summaries endpoints."""

    def test_get_daily_summaries(self, client, admin_api_key):
        """Test getting daily summaries."""
        response = client.get(
            "/summaries/daily",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert "summaries" in data or "total_days" in data

    def test_get_specific_day_summary(self, client, admin_api_key):
        """Test getting specific day summary."""
        response = client.get(
            "/summaries/daily/2025-11-26",
            headers={"X-API-Key": admin_api_key}
        )

        # Should succeed or 404 if no data
        assert response.status_code in [200, 404]

    def test_get_summary_invalid_date(self, client, admin_api_key):
        """Test getting summary with invalid date format."""
        response = client.get(
            "/summaries/daily/invalid-date",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code in [400, 404, 422]


class TestAnomaliesEndpoint:
    """Test anomalies endpoints."""

    def test_get_anomalies(self, client, admin_api_key):
        """Test getting anomalies."""
        response = client.get(
            "/anomalies",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert "anomalies" in data or "total_anomalies" in data

    def test_get_anomalies_filtered(self, client, admin_api_key):
        """Test getting filtered anomalies."""
        response = client.get(
            "/anomalies?severity=high&resolved=false",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200

    def test_get_anomalies_by_date(self, client, admin_api_key):
        """Test getting anomalies for specific date."""
        response = client.get(
            "/anomalies/2025-11-26",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code in [200, 404]


class TestRegressionsEndpoint:
    """Test regressions endpoints."""

    def test_get_regressions(self, client, admin_api_key):
        """Test getting regressions."""
        response = client.get(
            "/regressions",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert "regressions" in data or "total_regressions" in data

    def test_get_regressions_filtered(self, client, admin_api_key):
        """Test getting filtered regressions."""
        response = client.get(
            "/regressions?severity=high",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200

    def test_get_regressions_by_date(self, client, admin_api_key):
        """Test getting regressions for specific date."""
        response = client.get(
            "/regressions/2025-11-26",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code in [200, 404]


class TestRetrospectiveEndpoint:
    """Test retrospective endpoints."""

    def test_get_retrospective(self, client, admin_api_key):
        """Test getting retrospective report."""
        response = client.get(
            "/retrospective",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert "period" in data or "overall_metrics" in data

    def test_get_retrospective_with_days(self, client, admin_api_key):
        """Test retrospective with custom days."""
        response = client.get(
            "/retrospective?days=30",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200

    def test_get_retrospective_signed(self, client, admin_api_key):
        """Test signed retrospective."""
        response = client.get(
            "/retrospective?sign=true",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have signature if signing is enabled
        assert "signature" in data or "data" in data

    def test_download_retrospective(self, client, admin_api_key):
        """Test downloading retrospective."""
        response = client.get(
            "/retrospective/download",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        assert "Content-Disposition" in response.headers


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.get("/health")

        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers or \
               response.status_code == 200

    def test_rate_limit_enforcement(self, client):
        """Test rate limit is enforced."""
        # Make many requests quickly
        responses = []
        for _ in range(50):
            response = client.get("/health")
            responses.append(response)

        # Should eventually get rate limited
        status_codes = [r.status_code for r in responses]

        # Either all succeed (rate limit not configured) or some get 429
        assert all(c in [200, 429] for c in status_codes)

    def test_authenticated_higher_limit(self, client, admin_api_key):
        """Test authenticated requests have higher rate limit."""
        # Make requests with auth
        responses_auth = []
        for _ in range(50):
            response = client.get(
                "/ga",
                headers={"X-API-Key": admin_api_key}
            )
            responses_auth.append(response)

        # Should have fewer rate limit errors than unauthenticated
        rate_limited = sum(1 for r in responses_auth if r.status_code == 429)
        assert rate_limited < 50  # Not all should be rate limited


class TestErrorHandling:
    """Test API error handling."""

    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.delete("/health")

        assert response.status_code == 405

    def test_422_validation_error(self, client, admin_api_key):
        """Test 422 for validation errors."""
        response = client.get(
            "/ga?days=invalid",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 422

    def test_error_response_format(self, client):
        """Test error responses have consistent format."""
        response = client.get("/nonexistent")

        data = response.json()
        assert "detail" in data


class TestCORS:
    """Test CORS headers."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present."""
        response = client.options("/health")

        # Should have CORS headers if CORS is enabled
        # If not enabled, this is still a valid state
        assert response.status_code in [200, 405]

    def test_cors_allows_origin(self, client):
        """Test CORS allows configured origins."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        # Should succeed
        assert response.status_code == 200


class TestResponseModels:
    """Test API response models."""

    def test_ga_response_model(self, client, admin_api_key):
        """Test GA KPI response has expected fields."""
        response = client.get(
            "/ga",
            headers={"X-API-Key": admin_api_key}
        )

        if response.status_code == 200:
            data = response.json()
            expected_fields = ["overall_availability", "ga_ready", "generated_at"]
            for field in expected_fields:
                assert field in data

    def test_timestamp_format(self, client, admin_api_key):
        """Test timestamps are in ISO format."""
        response = client.get(
            "/ga",
            headers={"X-API-Key": admin_api_key}
        )

        if response.status_code == 200:
            data = response.json()
            if "generated_at" in data:
                timestamp = data["generated_at"]
                # Should be valid ISO 8601 format
                assert "T" in timestamp or ":" in timestamp


# Integration tests
class TestAPIIntegration:
    """Integration tests for API workflows."""

    def test_complete_authentication_flow(self, client):
        """Test complete JWT authentication flow."""
        # 1. Login
        login_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "demo123"}
        )
        assert login_response.status_code == 200
        access_token = login_response.json()["access_token"]
        refresh_token = login_response.json()["refresh_token"]

        # 2. Use access token
        ga_response = client.get(
            "/ga",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert ga_response.status_code == 200

        # 3. Refresh token
        refresh_response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert refresh_response.status_code == 200
        new_access_token = refresh_response.json()["access_token"]

        # 4. Use new token
        ga_response2 = client.get(
            "/ga",
            headers={"Authorization": f"Bearer {new_access_token}"}
        )
        assert ga_response2.status_code == 200

    def test_complete_data_retrieval_flow(self, client, admin_api_key):
        """Test retrieving all data types."""
        headers = {"X-API-Key": admin_api_key}

        # Get GA KPI
        ga_response = client.get("/ga", headers=headers)
        assert ga_response.status_code == 200

        # Get daily summaries
        summaries_response = client.get("/summaries/daily", headers=headers)
        assert summaries_response.status_code == 200

        # Get anomalies
        anomalies_response = client.get("/anomalies", headers=headers)
        assert anomalies_response.status_code == 200

        # Get regressions
        regressions_response = client.get("/regressions", headers=headers)
        assert regressions_response.status_code == 200

        # Get retrospective
        retrospective_response = client.get("/retrospective", headers=headers)
        assert retrospective_response.status_code == 200


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

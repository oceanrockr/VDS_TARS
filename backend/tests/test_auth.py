"""
T.A.R.S. Authentication Tests
Unit tests for JWT authentication endpoints and utilities
"""

import pytest
from datetime import datetime, timedelta
from fastapi import status

from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    blacklist_token,
)


class TestJWTUtilities:
    """Test JWT utility functions"""

    def test_create_access_token(self):
        """Test access token creation"""
        data = {"sub": "test-user"}
        token = create_access_token(data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test refresh token creation"""
        data = {"sub": "test-user"}
        token = create_refresh_token(data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token(self):
        """Test token verification"""
        data = {"sub": "test-user", "email": "test@example.com"}
        token = create_access_token(data)

        payload = verify_token(token)

        assert payload is not None
        assert payload["sub"] == "test-user"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_verify_expired_token(self):
        """Test that expired tokens are rejected"""
        data = {"sub": "test-user"}
        expires_delta = timedelta(seconds=-1)  # Already expired
        token = create_access_token(data, expires_delta)

        with pytest.raises(Exception):
            verify_token(token)

    def test_verify_invalid_token(self):
        """Test that invalid tokens are rejected"""
        invalid_token = "invalid.token.here"

        with pytest.raises(Exception):
            verify_token(invalid_token)

    def test_blacklist_token(self):
        """Test token blacklisting"""
        data = {"sub": "test-user"}
        token = create_access_token(data)

        # Token should be valid initially
        payload = verify_token(token)
        assert payload is not None

        # Blacklist the token
        blacklist_token(token)

        # Token should now be invalid
        with pytest.raises(Exception):
            verify_token(token)


class TestAuthEndpoints:
    """Test authentication REST endpoints"""

    def test_generate_token_success(self, client):
        """Test successful token generation"""
        response = client.post(
            "/auth/token",
            json={
                "client_id": "test-client-123",
                "device_name": "Test Device",
                "device_type": "windows",
            },
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_generate_token_minimal(self, client):
        """Test token generation with minimal data"""
        response = client.post(
            "/auth/token",
            json={"client_id": "test-client-456"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_generate_token_invalid_request(self, client):
        """Test token generation with invalid request"""
        response = client.post(
            "/auth/token",
            json={},  # Missing required client_id
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_refresh_token_success(self, client, test_client_id):
        """Test successful token refresh"""
        # Generate initial tokens
        response = client.post(
            "/auth/token",
            json={"client_id": test_client_id},
        )
        refresh_token = response.json()["refresh_token"]

        # Refresh the token
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["access_token"] != refresh_token

    def test_refresh_with_access_token_fails(self, client, test_token):
        """Test that access tokens cannot be used for refresh"""
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": test_token},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_with_invalid_token(self, client):
        """Test token refresh with invalid token"""
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_validate_token_success(self, client, test_token):
        """Test successful token validation"""
        response = client.post(
            "/auth/validate",
            headers={"Authorization": f"Bearer {test_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["valid"] is True
        assert "client_id" in data
        assert "expires_at" in data

    def test_validate_token_invalid(self, client):
        """Test validation of invalid token"""
        response = client.post(
            "/auth/validate",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["valid"] is False

    def test_validate_token_missing(self, client):
        """Test validation without token"""
        response = client.post("/auth/validate")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_revoke_token_success(self, client, test_client_id):
        """Test successful token revocation"""
        # Generate tokens
        response = client.post(
            "/auth/token",
            json={"client_id": test_client_id},
        )
        access_token = response.json()["access_token"]
        refresh_token = response.json()["refresh_token"]

        # Revoke the refresh token
        response = client.post(
            "/auth/revoke",
            json={"token": refresh_token},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

        # Try to use the revoked token
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_revoke_token_unauthorized(self, client, test_client_id):
        """Test token revocation without authentication"""
        response = client.post(
            "/auth/token",
            json={"client_id": test_client_id},
        )
        refresh_token = response.json()["refresh_token"]

        response = client.post(
            "/auth/revoke",
            json={"token": refresh_token},
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_auth_health(self, client):
        """Test authentication service health endpoint"""
        response = client.get("/auth/health")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "authentication"
        assert "timestamp" in data


class TestJWTExpiration:
    """Test JWT token expiration handling"""

    def test_access_token_expiration_time(self):
        """Test that access token has correct expiration"""
        from app.core.config import settings

        data = {"sub": "test-user"}
        token = create_access_token(data)
        payload = verify_token(token)

        # Calculate expected expiration
        iat = datetime.fromtimestamp(payload["iat"])
        exp = datetime.fromtimestamp(payload["exp"])
        duration = exp - iat

        expected_hours = settings.JWT_EXPIRATION_HOURS
        assert duration.total_seconds() == pytest.approx(
            expected_hours * 3600, rel=1
        )

    def test_refresh_token_expiration_time(self):
        """Test that refresh token has correct expiration"""
        from app.core.config import settings

        data = {"sub": "test-user"}
        token = create_refresh_token(data)
        payload = verify_token(token)

        # Calculate expected expiration
        iat = datetime.fromtimestamp(payload["iat"])
        exp = datetime.fromtimestamp(payload["exp"])
        duration = exp - iat

        expected_days = settings.JWT_REFRESH_EXPIRATION_DAYS
        assert duration.total_seconds() == pytest.approx(
            expected_days * 86400, rel=1
        )


class TestTokenPayload:
    """Test JWT token payload structure"""

    def test_access_token_payload_structure(self):
        """Test access token contains required fields"""
        data = {
            "sub": "test-user",
            "device_name": "Test Device",
            "device_type": "test",
        }
        token = create_access_token(data)
        payload = verify_token(token)

        # Required fields
        assert "sub" in payload
        assert "exp" in payload
        assert "iat" in payload
        assert "type" in payload
        assert payload["type"] == "access"

        # Custom fields
        assert payload["device_name"] == "Test Device"
        assert payload["device_type"] == "test"

    def test_refresh_token_payload_structure(self):
        """Test refresh token contains required fields"""
        data = {"sub": "test-user"}
        token = create_refresh_token(data)
        payload = verify_token(token)

        # Required fields
        assert "sub" in payload
        assert "exp" in payload
        assert "iat" in payload
        assert "type" in payload
        assert payload["type"] == "refresh"

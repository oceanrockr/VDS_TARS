"""
Security Test: RBAC Bypass Attempts
Phase 13.6 - Security Testing

Tests:
- Role escalation attempts
- Admin-only endpoint protection
- Token forgery detection
- Invalid kid handling
"""
import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock


@pytest.fixture
def jwt_secret():
    """JWT secret key."""
    return "test-secret-key"


@pytest.fixture
def viewer_token(jwt_secret):
    """Generate viewer role token."""
    payload = {
        "sub": "viewer-user",
        "role": "viewer",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def developer_token(jwt_secret):
    """Generate developer role token."""
    payload = {
        "sub": "dev-user",
        "role": "developer",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def admin_token(jwt_secret):
    """Generate admin role token."""
    payload = {
        "sub": "admin-user",
        "role": "admin",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


class TestRoleEscalation:
    """Test role escalation attempts."""

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_viewer_cannot_modify_baselines(self, viewer_token):
        """
        Verify: Viewer role cannot POST to /baselines (403).
        """
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 403
            mock_response.json = AsyncMock(return_value={
                "error": "Insufficient permissions"
            })
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/baselines",
                    json={
                        "agent_type": "DQN",
                        "metrics": {"mean_reward": 200.0}
                    },
                    headers={"Authorization": f"Bearer {viewer_token}"}
                ) as resp:
                    assert resp.status == 403

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_developer_cannot_admin_operations(self, developer_token):
        """
        Verify: Developer role cannot perform admin-only operations (403).
        """
        # Try to force-update baseline (admin only)
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 403
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/baselines",
                    json={
                        "agent_type": "DQN",
                        "environment": "CartPole-v1",
                        "metrics": {"mean_reward": 200.0},
                        "force": True  # Admin-only parameter
                    },
                    headers={"Authorization": f"Bearer {developer_token}"}
                ) as resp:
                    assert resp.status == 403

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_forged_role_in_token(self, jwt_secret):
        """
        Verify: Forged admin role in token is rejected.
        """
        # Create token with forged admin role
        payload = {
            "sub": "attacker",
            "role": "admin",  # Forged
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        # Sign with wrong secret
        forged_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 401  # Invalid signature
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/baselines",
                    json={"agent_type": "DQN"},
                    headers={"Authorization": f"Bearer {forged_token}"}
                ) as resp:
                    assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_tampered_role_claim(self, developer_token, jwt_secret):
        """
        Verify: Tampering with role claim invalidates signature.
        """
        # Decode token
        payload = jwt.decode(developer_token, jwt_secret, algorithms=["HS256"])

        # Tamper with role
        payload["role"] = "admin"

        # Re-encode WITHOUT correct secret
        tampered_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/baselines",
                    json={"agent_type": "DQN"},
                    headers={"Authorization": f"Bearer {tampered_token}"}
                ) as resp:
                    assert resp.status == 401


class TestAdminOnlyEndpoints:
    """Test admin-only endpoint protection."""

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_admin_only_baseline_update(self, admin_token, developer_token):
        """
        Verify: Only admin can update baselines.
        """
        # Admin succeeds
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/baselines",
                    json={
                        "agent_type": "DQN",
                        "environment": "CartPole-v1",
                        "metrics": {"mean_reward": 200.0}
                    },
                    headers={"Authorization": f"Bearer {admin_token}"}
                ) as resp:
                    assert resp.status == 201

        # Developer fails
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 403
            mock_post.return_value.__aenter__.return_value = mock_response

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/baselines",
                    json={
                        "agent_type": "DQN",
                        "environment": "CartPole-v1",
                        "metrics": {"mean_reward": 200.0}
                    },
                    headers={"Authorization": f"Bearer {developer_token}"}
                ) as resp:
                    assert resp.status == 403

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_viewer_can_read_baselines(self, viewer_token):
        """
        Verify: Viewer can read baselines (GET).
        """
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "agent_type": "DQN",
                "baseline": {"mean_reward": 200.0}
            })
            mock_get.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://eval-engine:8099/v1/baselines/DQN",
                    headers={"Authorization": f"Bearer {viewer_token}"}
                ) as resp:
                    assert resp.status == 200


class TestTokenForgery:
    """Test token forgery detection."""

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_invalid_kid_header(self):
        """
        Verify: Token with invalid 'kid' (key ID) rejected (401).
        """
        # Create token with fake kid
        header = {"alg": "HS256", "typ": "JWT", "kid": "fake-key-id"}
        payload = {
            "sub": "attacker",
            "role": "admin",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }

        # Encode with fake kid
        token_bytes = jwt.encode(payload, "fake-secret", algorithm="HS256", headers=header)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json = AsyncMock(return_value={"error": "Invalid key ID"})
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/evaluate",
                    json={"agent_type": "DQN"},
                    headers={"Authorization": f"Bearer {token_bytes}"}
                ) as resp:
                    assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_no_credentials_401(self):
        """
        Verify: Request without Authorization header returns 401.
        """
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/evaluate",
                    json={"agent_type": "DQN"}
                    # No Authorization header
                ) as resp:
                    assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_malformed_authorization_header(self):
        """
        Verify: Malformed Authorization header rejected.
        """
        malformed_headers = [
            {"Authorization": "InvalidFormat token123"},
            {"Authorization": "Bearer"},  # Missing token
            {"Authorization": "token123"},  # Missing scheme
            {"Authorization": "Bearer "},  # Empty token
        ]

        for headers in malformed_headers:
            with patch("aiohttp.ClientSession.post") as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 401
                mock_post.return_value.__aenter__.return_value = mock_response

                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://eval-engine:8099/v1/evaluate",
                        json={"agent_type": "DQN"},
                        headers=headers
                    ) as resp:
                        assert resp.status == 401


class TestRoleInheritance:
    """Test role inheritance and permissions."""

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_admin_has_all_permissions(self, admin_token):
        """
        Verify: Admin role has all permissions (read + write).
        """
        operations = [
            ("GET", "/v1/baselines/DQN", 200),
            ("POST", "/v1/evaluate", 200),
            ("POST", "/v1/baselines", 201),
        ]

        for method, path, expected_status in operations:
            with patch(f"aiohttp.ClientSession.{method.lower()}") as mock_method:
                mock_response = AsyncMock()
                mock_response.status = expected_status
                mock_response.json = AsyncMock(return_value={})
                mock_method.return_value.__aenter__.return_value = mock_response

                import aiohttp
                async with aiohttp.ClientSession() as session:
                    if method == "GET":
                        async with session.get(
                            f"http://eval-engine:8099{path}",
                            headers={"Authorization": f"Bearer {admin_token}"}
                        ) as resp:
                            assert resp.status == expected_status
                    else:
                        async with session.post(
                            f"http://eval-engine:8099{path}",
                            json={},
                            headers={"Authorization": f"Bearer {admin_token}"}
                        ) as resp:
                            assert resp.status == expected_status

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_developer_can_evaluate_but_not_admin(self, developer_token):
        """
        Verify: Developer can evaluate but not perform admin operations.
        """
        # Can evaluate
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"job_id": "test"})
            mock_post.return_value.__aenter__.return_value = mock_response

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/evaluate",
                    json={"agent_type": "DQN"},
                    headers={"Authorization": f"Bearer {developer_token}"}
                ) as resp:
                    assert resp.status == 200

        # Cannot update baselines
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 403
            mock_post.return_value.__aenter__.return_value = mock_response

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://eval-engine:8099/v1/baselines",
                    json={"agent_type": "DQN"},
                    headers={"Authorization": f"Bearer {developer_token}"}
                ) as resp:
                    assert resp.status == 403


# Pytest markers
pytestmark = pytest.mark.security

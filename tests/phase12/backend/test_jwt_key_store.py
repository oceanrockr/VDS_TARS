"""
Test suite for JWT Key Store (cognition/shared/jwt_key_store.py)

Coverage targets:
- Key creation and rotation
- Multi-key support (JWKS-style)
- Grace period handling
- Key expiration
- Key invalidation
- Redis persistence
- Fallback mode
- Token signing and verification
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from tests.conftest import MockRedis


# Import after fixtures are set up
@pytest.fixture
def jwt_key_store(mock_redis_client: MockRedis):
    """Create JWT key store instance with mocked Redis"""
    from cognition.shared.jwt_key_store import JWTKeyStore
    # Reset singleton
    JWTKeyStore._instance = None
    store = JWTKeyStore(redis_client=mock_redis_client)
    return store


# ============================================================================
# KEY CREATION AND ROTATION
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestJWTKeyCreation:
    """Test JWT key creation"""

    def test_create_key_generates_valid_key(self, jwt_key_store):
        """Test create_key() generates a valid JWT key"""
        key = jwt_key_store.create_key()

        assert key is not None
        assert key.kid.startswith("key-")
        assert key.algorithm == "HS256"
        assert key.secret is not None
        assert len(key.secret) >= 32  # Minimum secret length
        assert key.created_at is not None
        assert key.is_active is True
        assert key.is_valid is True

    def test_create_key_with_custom_algorithm(self, jwt_key_store):
        """Test create_key() with custom algorithm"""
        key = jwt_key_store.create_key(algorithm="HS512")

        assert key.algorithm == "HS512"

    def test_create_key_persists_to_redis(self, jwt_key_store, mock_redis_client):
        """Test create_key() persists to Redis"""
        key = jwt_key_store.create_key()

        # Verify key was stored in Redis
        stored_data = mock_redis_client.hget("jwt_keys", key.kid)
        assert stored_data is not None

        # Verify stored data is valid JSON
        stored_key = json.loads(stored_data)
        assert stored_key["kid"] == key.kid
        assert stored_key["algorithm"] == key.algorithm

    def test_create_key_increments_prometheus_counter(self, jwt_key_store):
        """Test create_key() increments Prometheus metrics"""
        with patch('cognition.shared.jwt_key_store.jwt_keys_created_total') as mock_counter:
            jwt_key_store.create_key()
            mock_counter.labels.assert_called_once()


@pytest.mark.unit
@pytest.mark.jwt
class TestJWTKeyRotation:
    """Test JWT key rotation"""

    def test_rotate_key_creates_new_active_key(self, jwt_key_store):
        """Test rotate_key() creates new active key"""
        old_key = jwt_key_store.get_current_key()
        new_key = jwt_key_store.rotate_key()

        assert new_key.kid != old_key.kid
        assert new_key.is_active is True
        assert old_key.is_active is False

    def test_rotate_key_sets_grace_period(self, jwt_key_store):
        """Test rotate_key() sets grace period for old key"""
        old_key = jwt_key_store.get_current_key()
        old_kid = old_key.kid

        new_key = jwt_key_store.rotate_key(grace_period_hours=24)

        # Retrieve old key from store
        old_key_updated = jwt_key_store.get_key(old_kid)
        assert old_key_updated.expires_at is not None

        # Grace period should be ~24 hours from now
        expected_expiry = datetime.utcnow() + timedelta(hours=24)
        time_diff = abs((old_key_updated.expires_at - expected_expiry).total_seconds())
        assert time_diff < 5  # Within 5 seconds

    def test_rotate_key_maintains_old_key_validity(self, jwt_key_store):
        """Test rotate_key() keeps old key valid during grace period"""
        old_key = jwt_key_store.get_current_key()
        old_kid = old_key.kid

        jwt_key_store.rotate_key(grace_period_hours=24)

        # Old key should still be valid
        old_key_updated = jwt_key_store.get_key(old_kid)
        assert old_key_updated.is_valid is True

    def test_rotate_key_updates_current_key_pointer(self, jwt_key_store):
        """Test rotate_key() updates current key pointer"""
        old_key = jwt_key_store.get_current_key()
        new_key = jwt_key_store.rotate_key()

        current_key = jwt_key_store.get_current_key()
        assert current_key.kid == new_key.kid
        assert current_key.kid != old_key.kid

    def test_rotate_key_persists_both_keys(self, jwt_key_store, mock_redis_client):
        """Test rotate_key() persists both old and new keys to Redis"""
        old_key = jwt_key_store.get_current_key()
        new_key = jwt_key_store.rotate_key()

        # Both keys should exist in Redis
        old_data = mock_redis_client.hget("jwt_keys", old_key.kid)
        new_data = mock_redis_client.hget("jwt_keys", new_key.kid)

        assert old_data is not None
        assert new_data is not None


# ============================================================================
# KEY RETRIEVAL
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestJWTKeyRetrieval:
    """Test JWT key retrieval"""

    def test_get_current_key_returns_active_key(self, jwt_key_store):
        """Test get_current_key() returns the active key"""
        current = jwt_key_store.get_current_key()

        assert current is not None
        assert current.is_active is True

    def test_get_key_by_kid_returns_correct_key(self, jwt_key_store):
        """Test get_key() retrieves key by kid"""
        key1 = jwt_key_store.create_key()
        key2 = jwt_key_store.create_key()

        retrieved = jwt_key_store.get_key(key1.kid)
        assert retrieved.kid == key1.kid

    def test_get_key_nonexistent_returns_none(self, jwt_key_store):
        """Test get_key() with non-existent kid returns None"""
        result = jwt_key_store.get_key("nonexistent-kid")
        assert result is None

    def test_list_keys_returns_all_keys(self, jwt_key_store):
        """Test list_keys() returns all keys"""
        initial_count = len(jwt_key_store.list_keys())

        jwt_key_store.create_key()
        jwt_key_store.create_key()

        keys = jwt_key_store.list_keys()
        assert len(keys) == initial_count + 2

    def test_list_keys_includes_inactive_keys(self, jwt_key_store):
        """Test list_keys() includes inactive keys"""
        jwt_key_store.create_key()
        jwt_key_store.rotate_key()  # Creates inactive key

        keys = jwt_key_store.list_keys()
        active_count = sum(1 for k in keys if k.is_active)
        inactive_count = sum(1 for k in keys if not k.is_active)

        assert active_count >= 1
        assert inactive_count >= 1


# ============================================================================
# KEY EXPIRATION
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestJWTKeyExpiration:
    """Test JWT key expiration"""

    def test_expired_key_is_invalid(self, jwt_key_store, mock_redis_client):
        """Test expired key is marked as invalid"""
        key = jwt_key_store.create_key()

        # Manually set expiration to past
        key.expires_at = datetime.utcnow() - timedelta(hours=1)
        key_data = {
            "kid": key.kid,
            "algorithm": key.algorithm,
            "secret": key.secret,
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat(),
            "is_active": key.is_active
        }
        mock_redis_client.hset("jwt_keys", key.kid, json.dumps(key_data).encode())

        # Retrieve and check validity
        retrieved = jwt_key_store.get_key(key.kid)
        assert retrieved.is_valid is False

    def test_non_expired_key_is_valid(self, jwt_key_store, mock_redis_client):
        """Test non-expired key is marked as valid"""
        key = jwt_key_store.create_key()

        # Set expiration to future
        key.expires_at = datetime.utcnow() + timedelta(hours=24)
        key_data = {
            "kid": key.kid,
            "algorithm": key.algorithm,
            "secret": key.secret,
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat(),
            "is_active": key.is_active
        }
        mock_redis_client.hset("jwt_keys", key.kid, json.dumps(key_data).encode())

        retrieved = jwt_key_store.get_key(key.kid)
        assert retrieved.is_valid is True

    def test_cleanup_removes_expired_keys(self, jwt_key_store, mock_redis_client):
        """Test cleanup_expired_keys() removes expired keys"""
        # Create expired key
        key = jwt_key_store.create_key()
        key.expires_at = datetime.utcnow() - timedelta(hours=1)
        key_data = {
            "kid": key.kid,
            "algorithm": key.algorithm,
            "secret": key.secret,
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat(),
            "is_active": False
        }
        mock_redis_client.hset("jwt_keys", key.kid, json.dumps(key_data).encode())

        # Run cleanup
        removed_count = jwt_key_store.cleanup_expired_keys()

        assert removed_count >= 1
        assert jwt_key_store.get_key(key.kid) is None


# ============================================================================
# KEY INVALIDATION
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestJWTKeyInvalidation:
    """Test JWT key invalidation"""

    def test_invalidate_key_marks_as_invalid(self, jwt_key_store):
        """Test invalidate_key() marks key as invalid immediately"""
        key = jwt_key_store.create_key()

        result = jwt_key_store.invalidate_key(key.kid)

        assert result is True
        retrieved = jwt_key_store.get_key(key.kid)
        assert retrieved.is_valid is False

    def test_invalidate_key_sets_expiration_to_now(self, jwt_key_store):
        """Test invalidate_key() sets expiration to now"""
        key = jwt_key_store.create_key()

        jwt_key_store.invalidate_key(key.kid)

        retrieved = jwt_key_store.get_key(key.kid)
        assert retrieved.expires_at is not None
        time_diff = abs((retrieved.expires_at - datetime.utcnow()).total_seconds())
        assert time_diff < 5  # Within 5 seconds

    def test_invalidate_nonexistent_key_raises_error(self, jwt_key_store):
        """Test invalidate_key() with non-existent kid raises ValueError"""
        with pytest.raises(ValueError, match="JWT key not found"):
            jwt_key_store.invalidate_key("nonexistent-kid")

    def test_invalidate_key_increments_prometheus_counter(self, jwt_key_store):
        """Test invalidate_key() increments Prometheus metrics"""
        key = jwt_key_store.create_key()

        with patch('cognition.shared.jwt_key_store.jwt_keys_invalidated_total') as mock_counter:
            jwt_key_store.invalidate_key(key.kid)
            mock_counter.labels.assert_called_once()


# ============================================================================
# REDIS PERSISTENCE
# ============================================================================

@pytest.mark.integration
@pytest.mark.jwt
@pytest.mark.redis
class TestJWTKeyRedisPersistence:
    """Test JWT key Redis persistence"""

    def test_keys_persist_across_store_instances(self, mock_redis_client):
        """Test keys persist across JWTKeyStore instances"""
        from cognition.shared.jwt_key_store import JWTKeyStore

        # Create key with first instance
        JWTKeyStore._instance = None
        store1 = JWTKeyStore(redis_client=mock_redis_client)
        key1 = store1.create_key()
        kid = key1.kid

        # Retrieve with second instance
        JWTKeyStore._instance = None
        store2 = JWTKeyStore(redis_client=mock_redis_client)
        key2 = store2.get_key(kid)

        assert key2 is not None
        assert key2.kid == kid
        assert key2.secret == key1.secret

    def test_current_key_pointer_persists(self, mock_redis_client):
        """Test current key pointer persists across instances"""
        from cognition.shared.jwt_key_store import JWTKeyStore

        # Set current key with first instance
        JWTKeyStore._instance = None
        store1 = JWTKeyStore(redis_client=mock_redis_client)
        key1 = store1.get_current_key()

        # Retrieve with second instance
        JWTKeyStore._instance = None
        store2 = JWTKeyStore(redis_client=mock_redis_client)
        key2 = store2.get_current_key()

        assert key2.kid == key1.kid

    def test_rotation_state_persists(self, mock_redis_client):
        """Test rotation state persists across instances"""
        from cognition.shared.jwt_key_store import JWTKeyStore

        # Rotate key with first instance
        JWTKeyStore._instance = None
        store1 = JWTKeyStore(redis_client=mock_redis_client)
        old_key = store1.get_current_key()
        new_key = store1.rotate_key()

        # Verify with second instance
        JWTKeyStore._instance = None
        store2 = JWTKeyStore(redis_client=mock_redis_client)
        current = store2.get_current_key()
        old_retrieved = store2.get_key(old_key.kid)

        assert current.kid == new_key.kid
        assert old_retrieved.is_active is False


# ============================================================================
# FALLBACK MODE
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestJWTKeyStoreFallback:
    """Test JWT key store fallback mode"""

    def test_fallback_when_redis_unavailable(self):
        """Test fallback to in-memory store when Redis unavailable"""
        from cognition.shared.jwt_key_store import JWTKeyStore

        # Create mock that fails
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        JWTKeyStore._instance = None
        store = JWTKeyStore(redis_client=mock_redis)

        # Should still work with in-memory fallback
        key = store.get_current_key()
        assert key is not None

    def test_fallback_mode_supports_rotation(self):
        """Test fallback mode supports key rotation"""
        from cognition.shared.jwt_key_store import JWTKeyStore

        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        JWTKeyStore._instance = None
        store = JWTKeyStore(redis_client=mock_redis)

        old_key = store.get_current_key()
        new_key = store.rotate_key()

        assert new_key.kid != old_key.kid
        assert new_key.is_active is True


# ============================================================================
# TOKEN SIGNING AND VERIFICATION
# ============================================================================

@pytest.mark.integration
@pytest.mark.jwt
class TestJWTTokenOperations:
    """Test JWT token signing and verification"""

    def test_create_token_with_kid_header(self, jwt_key_store):
        """Test creating token with kid header"""
        import jwt

        current_key = jwt_key_store.get_current_key()
        payload = {"user_id": "test-user", "exp": datetime.utcnow() + timedelta(hours=1)}

        token = jwt.encode(
            payload,
            current_key.secret,
            algorithm=current_key.algorithm,
            headers={"kid": current_key.kid}
        )

        # Decode without verification to check header
        unverified = jwt.decode(token, options={"verify_signature": False})
        header = jwt.get_unverified_header(token)

        assert header["kid"] == current_key.kid

    def test_verify_token_with_correct_key(self, jwt_key_store):
        """Test verifying token with correct key"""
        import jwt

        current_key = jwt_key_store.get_current_key()
        payload = {"user_id": "test-user", "exp": datetime.utcnow() + timedelta(hours=1)}

        token = jwt.encode(
            payload,
            current_key.secret,
            algorithm=current_key.algorithm,
            headers={"kid": current_key.kid}
        )

        # Verify token
        decoded = jwt.decode(
            token,
            current_key.secret,
            algorithms=[current_key.algorithm]
        )

        assert decoded["user_id"] == "test-user"

    def test_verify_token_after_rotation_within_grace_period(self, jwt_key_store):
        """Test old tokens work during grace period after rotation"""
        import jwt

        old_key = jwt_key_store.get_current_key()
        payload = {"user_id": "test-user", "exp": datetime.utcnow() + timedelta(hours=1)}

        # Create token with old key
        token = jwt.encode(
            payload,
            old_key.secret,
            algorithm=old_key.algorithm,
            headers={"kid": old_key.kid}
        )

        # Rotate key
        jwt_key_store.rotate_key(grace_period_hours=24)

        # Old token should still verify
        old_key_retrieved = jwt_key_store.get_key(old_key.kid)
        decoded = jwt.decode(
            token,
            old_key_retrieved.secret,
            algorithms=[old_key_retrieved.algorithm]
        )

        assert decoded["user_id"] == "test-user"

    def test_verify_token_fails_with_invalidated_key(self, jwt_key_store):
        """Test token verification fails with invalidated key"""
        import jwt

        key = jwt_key_store.create_key()
        payload = {"user_id": "test-user", "exp": datetime.utcnow() + timedelta(hours=1)}

        # Create token
        token = jwt.encode(
            payload,
            key.secret,
            algorithm=key.algorithm,
            headers={"kid": key.kid}
        )

        # Invalidate key
        jwt_key_store.invalidate_key(key.kid)

        # Verification should fail (key is invalid)
        invalidated_key = jwt_key_store.get_key(key.kid)
        assert invalidated_key.is_valid is False

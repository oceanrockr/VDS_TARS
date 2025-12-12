"""
Test suite for API Key Store (cognition/shared/api_key_store.py)

Coverage targets:
- Key creation with prefix "tars_"
- Key verification (hash-based)
- Key rotation (hot-swap)
- Key revocation
- Redis persistence
- Reverse hash lookup
- Fallback mode
- Migration from in-memory keys
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from tests.conftest import MockRedis


@pytest.fixture
def api_key_store(mock_redis_client: MockRedis):
    """Create API key store instance with mocked Redis"""
    from cognition.shared.api_key_store import APIKeyStore
    # Reset singleton
    APIKeyStore._instance = None
    store = APIKeyStore(redis_client=mock_redis_client)
    return store


# ============================================================================
# KEY CREATION
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestAPIKeyCreation:
    """Test API key creation"""

    def test_create_key_generates_valid_key(self, api_key_store):
        """Test create_key() generates a valid API key"""
        result = api_key_store.create_key(service_name="Test Service")

        assert result is not None
        assert "id" in result
        assert "key" in result
        assert result["key"].startswith("tars_")
        assert result["service_name"] == "Test Service"
        assert "created_at" in result

    def test_create_key_has_minimum_length(self, api_key_store):
        """Test create_key() generates key with minimum length"""
        result = api_key_store.create_key(service_name="Test Service")

        # Key should be: "tars_" + 32 hex chars = 37 chars minimum
        assert len(result["key"]) >= 37

    def test_create_key_with_expiration(self, api_key_store):
        """Test create_key() with expiration date"""
        expires_in_days = 365
        result = api_key_store.create_key(
            service_name="Expiring Service",
            expires_in_days=expires_in_days
        )

        assert "expires_at" in result
        assert result["expires_at"] is not None

        # Parse and verify expiration is ~365 days from now
        expires_at = datetime.fromisoformat(result["expires_at"].replace("Z", ""))
        expected = datetime.utcnow() + timedelta(days=expires_in_days)
        time_diff = abs((expires_at - expected).total_seconds())
        assert time_diff < 10  # Within 10 seconds

    def test_create_key_without_expiration(self, api_key_store):
        """Test create_key() without expiration"""
        result = api_key_store.create_key(service_name="Permanent Service")

        assert result["expires_at"] is None

    def test_create_key_persists_to_redis(self, api_key_store, mock_redis_client):
        """Test create_key() persists to Redis"""
        result = api_key_store.create_key(service_name="Test Service")
        key_id = result["id"]

        # Verify key was stored in Redis
        stored_data = mock_redis_client.hget("api_keys", key_id)
        assert stored_data is not None

        # Verify stored data
        stored_key = json.loads(stored_data)
        assert stored_key["service_name"] == "Test Service"
        assert "key_hash" in stored_key

    def test_create_key_stores_hash_not_plaintext(self, api_key_store, mock_redis_client):
        """Test create_key() stores hash, not plaintext key"""
        result = api_key_store.create_key(service_name="Test Service")
        key_id = result["id"]

        stored_data = mock_redis_client.hget("api_keys", key_id)
        stored_key = json.loads(stored_data)

        # Stored hash should not match plaintext key
        assert stored_key["key_hash"] != result["key"]
        # Hash should be 64 hex chars (SHA256)
        assert len(stored_key["key_hash"]) == 64


# ============================================================================
# KEY VERIFICATION
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestAPIKeyVerification:
    """Test API key verification"""

    def test_verify_key_accepts_valid_key(self, api_key_store):
        """Test verify_key() accepts valid key"""
        result = api_key_store.create_key(service_name="Test Service")
        api_key = result["key"]

        is_valid = api_key_store.verify_key(api_key)
        assert is_valid is True

    def test_verify_key_rejects_invalid_key(self, api_key_store):
        """Test verify_key() rejects invalid key"""
        is_valid = api_key_store.verify_key("tars_invalidkey123")
        assert is_valid is False

    def test_verify_key_rejects_revoked_key(self, api_key_store):
        """Test verify_key() rejects revoked key"""
        result = api_key_store.create_key(service_name="Test Service")
        api_key = result["key"]
        key_id = result["id"]

        # Revoke the key
        api_key_store.revoke_key(key_id)

        # Verification should fail
        is_valid = api_key_store.verify_key(api_key)
        assert is_valid is False

    def test_verify_key_rejects_expired_key(self, api_key_store, mock_redis_client):
        """Test verify_key() rejects expired key"""
        result = api_key_store.create_key(
            service_name="Expired Service",
            expires_in_days=1
        )
        api_key = result["key"]
        key_id = result["id"]

        # Manually set expiration to past
        stored_data = mock_redis_client.hget("api_keys", key_id)
        stored_key = json.loads(stored_data)
        stored_key["expires_at"] = (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z"
        mock_redis_client.hset("api_keys", key_id, json.dumps(stored_key).encode())

        # Verification should fail
        is_valid = api_key_store.verify_key(api_key)
        assert is_valid is False

    def test_verify_key_uses_hash_lookup(self, api_key_store):
        """Test verify_key() uses reverse hash lookup"""
        result = api_key_store.create_key(service_name="Test Service")
        api_key = result["key"]

        # Verify uses hash, not plaintext
        with patch.object(api_key_store, '_hash_key') as mock_hash:
            mock_hash.return_value = "dummy_hash"
            api_key_store.verify_key(api_key)
            mock_hash.assert_called_once_with(api_key)


# ============================================================================
# KEY ROTATION
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestAPIKeyRotation:
    """Test API key rotation"""

    def test_rotate_key_generates_new_key(self, api_key_store):
        """Test rotate_key() generates new key"""
        original = api_key_store.create_key(service_name="Test Service")
        original_key = original["key"]
        key_id = original["id"]

        rotated = api_key_store.rotate_key(key_id)

        assert rotated["key"] != original_key
        assert rotated["key"].startswith("tars_")
        assert rotated["id"] == key_id

    def test_rotate_key_invalidates_old_key(self, api_key_store):
        """Test rotate_key() invalidates old key"""
        original = api_key_store.create_key(service_name="Test Service")
        original_key = original["key"]
        key_id = original["id"]

        api_key_store.rotate_key(key_id)

        # Old key should no longer verify
        is_valid = api_key_store.verify_key(original_key)
        assert is_valid is False

    def test_rotate_key_new_key_verifies(self, api_key_store):
        """Test rotate_key() new key verifies successfully"""
        original = api_key_store.create_key(service_name="Test Service")
        key_id = original["id"]

        rotated = api_key_store.rotate_key(key_id)
        new_key = rotated["key"]

        # New key should verify
        is_valid = api_key_store.verify_key(new_key)
        assert is_valid is True

    def test_rotate_key_preserves_metadata(self, api_key_store, mock_redis_client):
        """Test rotate_key() preserves service metadata"""
        original = api_key_store.create_key(service_name="Test Service")
        key_id = original["id"]

        api_key_store.rotate_key(key_id)

        # Retrieve and verify metadata preserved
        stored_data = mock_redis_client.hget("api_keys", key_id)
        stored_key = json.loads(stored_data)
        assert stored_key["service_name"] == "Test Service"

    def test_rotate_nonexistent_key_raises_error(self, api_key_store):
        """Test rotate_key() with non-existent key raises ValueError"""
        with pytest.raises(ValueError, match="API key not found"):
            api_key_store.rotate_key("nonexistent-key-id")


# ============================================================================
# KEY REVOCATION
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestAPIKeyRevocation:
    """Test API key revocation"""

    def test_revoke_key_marks_as_revoked(self, api_key_store, mock_redis_client):
        """Test revoke_key() marks key as revoked"""
        result = api_key_store.create_key(service_name="Test Service")
        key_id = result["id"]

        success = api_key_store.revoke_key(key_id)

        assert success is True

        # Verify revoked in storage
        stored_data = mock_redis_client.hget("api_keys", key_id)
        stored_key = json.loads(stored_data)
        assert stored_key["is_active"] is False

    def test_revoke_key_with_reason(self, api_key_store, mock_redis_client):
        """Test revoke_key() with reason"""
        result = api_key_store.create_key(service_name="Test Service")
        key_id = result["id"]

        api_key_store.revoke_key(key_id, reason="Security incident")

        # Verify reason stored
        stored_data = mock_redis_client.hget("api_keys", key_id)
        stored_key = json.loads(stored_data)
        assert stored_key.get("revocation_reason") == "Security incident"

    def test_revoke_key_prevents_verification(self, api_key_store):
        """Test revoked key fails verification"""
        result = api_key_store.create_key(service_name="Test Service")
        api_key = result["key"]
        key_id = result["id"]

        api_key_store.revoke_key(key_id)

        is_valid = api_key_store.verify_key(api_key)
        assert is_valid is False

    def test_revoke_nonexistent_key_raises_error(self, api_key_store):
        """Test revoke_key() with non-existent key raises ValueError"""
        with pytest.raises(ValueError, match="API key not found"):
            api_key_store.revoke_key("nonexistent-key-id")


# ============================================================================
# KEY LISTING
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestAPIKeyListing:
    """Test API key listing"""

    def test_list_keys_returns_all_keys(self, api_key_store):
        """Test list_keys() returns all keys"""
        api_key_store.create_key(service_name="Service 1")
        api_key_store.create_key(service_name="Service 2")

        keys = api_key_store.list_keys()

        assert len(keys) >= 2
        service_names = [k["service_name"] for k in keys]
        assert "Service 1" in service_names
        assert "Service 2" in service_names

    def test_list_keys_excludes_plaintext_key(self, api_key_store):
        """Test list_keys() does not include plaintext key"""
        api_key_store.create_key(service_name="Test Service")

        keys = api_key_store.list_keys()

        for key_info in keys:
            assert "key" not in key_info
            assert "key_hash" not in key_info  # Also exclude hash

    def test_list_keys_includes_active_status(self, api_key_store):
        """Test list_keys() includes active status"""
        result = api_key_store.create_key(service_name="Test Service")
        key_id = result["id"]

        # Create one active, one revoked
        api_key_store.create_key(service_name="Active Service")
        api_key_store.revoke_key(key_id)

        keys = api_key_store.list_keys()

        # Find our keys
        test_key = next(k for k in keys if k["id"] == key_id)
        assert test_key["is_active"] is False

    def test_list_active_keys_only(self, api_key_store):
        """Test list_keys(active_only=True) returns only active keys"""
        result1 = api_key_store.create_key(service_name="Active Service")
        result2 = api_key_store.create_key(service_name="Revoked Service")

        api_key_store.revoke_key(result2["id"])

        active_keys = api_key_store.list_keys(active_only=True)

        ids = [k["id"] for k in active_keys]
        assert result1["id"] in ids
        assert result2["id"] not in ids


# ============================================================================
# REDIS PERSISTENCE
# ============================================================================

@pytest.mark.integration
@pytest.mark.apikey
@pytest.mark.redis
class TestAPIKeyRedisPersistence:
    """Test API key Redis persistence"""

    def test_keys_persist_across_store_instances(self, mock_redis_client):
        """Test keys persist across APIKeyStore instances"""
        from cognition.shared.api_key_store import APIKeyStore

        # Create key with first instance
        APIKeyStore._instance = None
        store1 = APIKeyStore(redis_client=mock_redis_client)
        result = store1.create_key(service_name="Test Service")
        api_key = result["key"]

        # Verify with second instance
        APIKeyStore._instance = None
        store2 = APIKeyStore(redis_client=mock_redis_client)
        is_valid = store2.verify_key(api_key)

        assert is_valid is True

    def test_revocation_persists(self, mock_redis_client):
        """Test revocation state persists across instances"""
        from cognition.shared.api_key_store import APIKeyStore

        # Revoke with first instance
        APIKeyStore._instance = None
        store1 = APIKeyStore(redis_client=mock_redis_client)
        result = store1.create_key(service_name="Test Service")
        api_key = result["key"]
        key_id = result["id"]
        store1.revoke_key(key_id)

        # Verify with second instance
        APIKeyStore._instance = None
        store2 = APIKeyStore(redis_client=mock_redis_client)
        is_valid = store2.verify_key(api_key)

        assert is_valid is False

    def test_rotation_persists(self, mock_redis_client):
        """Test rotation state persists across instances"""
        from cognition.shared.api_key_store import APIKeyStore

        # Rotate with first instance
        APIKeyStore._instance = None
        store1 = APIKeyStore(redis_client=mock_redis_client)
        original = store1.create_key(service_name="Test Service")
        rotated = store1.rotate_key(original["id"])
        new_key = rotated["key"]

        # Verify with second instance
        APIKeyStore._instance = None
        store2 = APIKeyStore(redis_client=mock_redis_client)
        is_valid = store2.verify_key(new_key)

        assert is_valid is True


# ============================================================================
# FALLBACK MODE
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestAPIKeyStoreFallback:
    """Test API key store fallback mode"""

    def test_fallback_when_redis_unavailable(self):
        """Test fallback to in-memory store when Redis unavailable"""
        from cognition.shared.api_key_store import APIKeyStore

        # Create mock that fails
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        APIKeyStore._instance = None
        store = APIKeyStore(redis_client=mock_redis)

        # Should still work with in-memory fallback
        result = store.create_key(service_name="Test Service")
        assert result is not None

    def test_fallback_mode_supports_verification(self):
        """Test fallback mode supports key verification"""
        from cognition.shared.api_key_store import APIKeyStore

        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        APIKeyStore._instance = None
        store = APIKeyStore(redis_client=mock_redis)

        result = store.create_key(service_name="Test Service")
        api_key = result["key"]

        is_valid = store.verify_key(api_key)
        assert is_valid is True

    def test_fallback_mode_supports_rotation(self):
        """Test fallback mode supports key rotation"""
        from cognition.shared.api_key_store import APIKeyStore

        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        APIKeyStore._instance = None
        store = APIKeyStore(redis_client=mock_redis)

        original = store.create_key(service_name="Test Service")
        rotated = store.rotate_key(original["id"])

        assert rotated["key"] != original["key"]


# ============================================================================
# REVERSE HASH LOOKUP
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestReverseHashLookup:
    """Test reverse hash lookup optimization"""

    def test_reverse_hash_index_created_on_key_creation(self, api_key_store, mock_redis_client):
        """Test reverse hash index is created when key is created"""
        result = api_key_store.create_key(service_name="Test Service")
        api_key = result["key"]

        # Compute hash
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Verify reverse index exists
        reverse_data = mock_redis_client.hget("api_keys_reverse", key_hash)
        assert reverse_data is not None

        # Verify it points to correct key ID
        assert reverse_data.decode() == result["id"]

    def test_reverse_hash_index_updated_on_rotation(self, api_key_store, mock_redis_client):
        """Test reverse hash index is updated on rotation"""
        original = api_key_store.create_key(service_name="Test Service")
        key_id = original["id"]

        rotated = api_key_store.rotate_key(key_id)
        new_key = rotated["key"]

        # Compute new hash
        import hashlib
        new_hash = hashlib.sha256(new_key.encode()).hexdigest()

        # Verify reverse index points to same key ID
        reverse_data = mock_redis_client.hget("api_keys_reverse", new_hash)
        assert reverse_data.decode() == key_id

    def test_verify_key_uses_reverse_index(self, api_key_store):
        """Test verify_key() uses reverse hash index for fast lookup"""
        result = api_key_store.create_key(service_name="Test Service")
        api_key = result["key"]

        # Mock to verify reverse index is used
        with patch.object(api_key_store, '_get_key_id_by_hash') as mock_lookup:
            mock_lookup.return_value = result["id"]
            api_key_store.verify_key(api_key)
            mock_lookup.assert_called_once()


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.unit
@pytest.mark.apikey
class TestAPIKeyEdgeCases:
    """Test edge cases and error handling"""

    def test_create_key_with_empty_service_name(self, api_key_store):
        """Test create_key() with empty service name"""
        with pytest.raises(ValueError, match="Service name cannot be empty"):
            api_key_store.create_key(service_name="")

    def test_create_key_with_negative_expiration(self, api_key_store):
        """Test create_key() with negative expiration"""
        with pytest.raises(ValueError, match="Expiration must be positive"):
            api_key_store.create_key(service_name="Test", expires_in_days=-1)

    def test_verify_key_with_malformed_key(self, api_key_store):
        """Test verify_key() with malformed key"""
        is_valid = api_key_store.verify_key("not-a-valid-key")
        assert is_valid is False

    def test_verify_key_with_wrong_prefix(self, api_key_store):
        """Test verify_key() with wrong prefix"""
        is_valid = api_key_store.verify_key("wrong_prefix_123456")
        assert is_valid is False

    def test_double_revocation_idempotent(self, api_key_store):
        """Test revoking key twice is idempotent"""
        result = api_key_store.create_key(service_name="Test Service")
        key_id = result["id"]

        api_key_store.revoke_key(key_id)
        success = api_key_store.revoke_key(key_id)

        assert success is True

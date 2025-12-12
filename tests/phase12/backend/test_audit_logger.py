"""
Test suite for Audit Logger (cognition/shared/audit_logger.py)

Coverage targets:
- All 20 event types
- Severity levels (LOW, MEDIUM, HIGH, CRITICAL)
- Filtering (by type, date range, user, severity)
- Structured JSON output
- Prometheus metrics integration
- Redis persistence
- Statistics aggregation
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from tests.conftest import MockRedis


@pytest.fixture
def audit_logger(mock_redis_client: MockRedis):
    """Create audit logger instance with mocked Redis"""
    from cognition.shared.audit_logger import AuditLogger
    # Reset singleton
    AuditLogger._instance = None
    logger = AuditLogger(redis_client=mock_redis_client)
    return logger


# ============================================================================
# EVENT LOGGING
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditEventLogging:
    """Test audit event logging"""

    def test_log_event_creates_entry(self, audit_logger, mock_redis_client):
        """Test log() creates audit entry"""
        event_id = audit_logger.log(
            event_type="API_KEY_CREATED",
            user_id="test-admin",
            severity="MEDIUM",
            metadata={"service_name": "Test Service"}
        )

        assert event_id is not None

        # Verify stored in Redis
        log_data = mock_redis_client.hget("audit_logs", event_id)
        assert log_data is not None

    def test_log_event_includes_timestamp(self, audit_logger, mock_redis_client):
        """Test log() includes timestamp"""
        event_id = audit_logger.log(
            event_type="API_KEY_CREATED",
            user_id="test-admin",
            severity="MEDIUM"
        )

        log_data = mock_redis_client.hget("audit_logs", event_id)
        log_entry = json.loads(log_data)

        assert "timestamp" in log_entry
        # Verify timestamp is recent (within last 5 seconds)
        timestamp = datetime.fromisoformat(log_entry["timestamp"].replace("Z", ""))
        time_diff = abs((datetime.utcnow() - timestamp).total_seconds())
        assert time_diff < 5

    def test_log_event_stores_metadata(self, audit_logger, mock_redis_client):
        """Test log() stores metadata"""
        metadata = {
            "service_name": "Test Service",
            "key_id": "key-123",
            "reason": "Security incident"
        }

        event_id = audit_logger.log(
            event_type="API_KEY_REVOKED",
            user_id="test-admin",
            severity="HIGH",
            metadata=metadata
        )

        log_data = mock_redis_client.hget("audit_logs", event_id)
        log_entry = json.loads(log_data)

        assert log_entry["metadata"] == metadata

    def test_log_event_increments_prometheus_counter(self, audit_logger):
        """Test log() increments Prometheus metrics"""
        with patch('cognition.shared.audit_logger.audit_events_total') as mock_counter:
            audit_logger.log(
                event_type="API_KEY_CREATED",
                user_id="test-admin",
                severity="MEDIUM"
            )
            mock_counter.labels.assert_called_once_with(
                event_type="API_KEY_CREATED",
                severity="MEDIUM"
            )


# ============================================================================
# EVENT TYPES
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditEventTypes:
    """Test all supported event types"""

    @pytest.mark.parametrize("event_type,severity", [
        ("API_KEY_CREATED", "MEDIUM"),
        ("API_KEY_ROTATED", "MEDIUM"),
        ("API_KEY_REVOKED", "HIGH"),
        ("JWT_ROTATION", "HIGH"),
        ("JWT_KEY_INVALIDATED", "CRITICAL"),
        ("AGENT_RELOAD", "MEDIUM"),
        ("MODEL_PROMOTION", "HIGH"),
        ("HYPERSYNC_APPROVAL", "MEDIUM"),
        ("HYPERSYNC_REJECTION", "LOW"),
        ("AUTOML_TRIAL_COMPLETE", "LOW"),
        ("USER_LOGIN", "LOW"),
        ("USER_LOGOUT", "LOW"),
        ("RBAC_PERMISSION_DENIED", "MEDIUM"),
        ("SYSTEM_HEALTH_DEGRADED", "HIGH"),
        ("SYSTEM_HEALTH_RECOVERED", "MEDIUM"),
        ("CONFIG_CHANGE", "HIGH"),
        ("DATA_EXPORT", "MEDIUM"),
        ("SECURITY_INCIDENT", "CRITICAL"),
        ("RATE_LIMIT_EXCEEDED", "MEDIUM"),
        ("UNKNOWN_ERROR", "HIGH"),
    ])
    def test_log_all_event_types(self, audit_logger, event_type, severity):
        """Test logging all supported event types"""
        event_id = audit_logger.log(
            event_type=event_type,
            user_id="test-user",
            severity=severity
        )

        assert event_id is not None

    def test_log_invalid_event_type_raises_error(self, audit_logger):
        """Test log() with invalid event type raises ValueError"""
        with pytest.raises(ValueError, match="Invalid event type"):
            audit_logger.log(
                event_type="INVALID_EVENT",
                user_id="test-user",
                severity="MEDIUM"
            )

    def test_log_invalid_severity_raises_error(self, audit_logger):
        """Test log() with invalid severity raises ValueError"""
        with pytest.raises(ValueError, match="Invalid severity"):
            audit_logger.log(
                event_type="API_KEY_CREATED",
                user_id="test-user",
                severity="INVALID_SEVERITY"
            )


# ============================================================================
# EVENT RETRIEVAL
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditEventRetrieval:
    """Test audit event retrieval"""

    def test_get_logs_returns_all_logs(self, audit_logger):
        """Test get_logs() returns all logs"""
        # Create multiple events
        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user2", "HIGH")
        audit_logger.log("AGENT_RELOAD", "user3", "MEDIUM")

        logs = audit_logger.get_logs()

        assert len(logs) >= 3

    def test_get_logs_with_limit(self, audit_logger):
        """Test get_logs() with limit parameter"""
        # Create 5 events
        for i in range(5):
            audit_logger.log("API_KEY_CREATED", f"user{i}", "MEDIUM")

        logs = audit_logger.get_logs(limit=3)

        assert len(logs) <= 3

    def test_get_logs_sorted_by_timestamp_desc(self, audit_logger):
        """Test get_logs() returns logs sorted by timestamp (newest first)"""
        # Create events with slight delay
        id1 = audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        id2 = audit_logger.log("JWT_ROTATION", "user2", "HIGH")

        logs = audit_logger.get_logs()

        # Find our events
        log1 = next(l for l in logs if l["id"] == id1)
        log2 = next(l for l in logs if l["id"] == id2)

        # id2 should come before id1 (newer first)
        idx1 = logs.index(log1)
        idx2 = logs.index(log2)
        assert idx2 < idx1

    def test_get_log_by_id(self, audit_logger):
        """Test get_log() retrieves specific log by ID"""
        event_id = audit_logger.log(
            "API_KEY_CREATED",
            "test-user",
            "MEDIUM",
            metadata={"key": "value"}
        )

        log = audit_logger.get_log(event_id)

        assert log is not None
        assert log["id"] == event_id
        assert log["event_type"] == "API_KEY_CREATED"
        assert log["metadata"]["key"] == "value"

    def test_get_log_nonexistent_returns_none(self, audit_logger):
        """Test get_log() with non-existent ID returns None"""
        log = audit_logger.get_log("nonexistent-id")
        assert log is None


# ============================================================================
# FILTERING
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditEventFiltering:
    """Test audit event filtering"""

    def test_filter_by_event_type(self, audit_logger):
        """Test get_logs() filter by event type"""
        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user2", "HIGH")
        audit_logger.log("API_KEY_CREATED", "user3", "MEDIUM")

        logs = audit_logger.get_logs(event_type="API_KEY_CREATED")

        assert len(logs) >= 2
        for log in logs:
            assert log["event_type"] == "API_KEY_CREATED"

    def test_filter_by_user_id(self, audit_logger):
        """Test get_logs() filter by user ID"""
        audit_logger.log("API_KEY_CREATED", "target-user", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "other-user", "HIGH")
        audit_logger.log("AGENT_RELOAD", "target-user", "MEDIUM")

        logs = audit_logger.get_logs(user_id="target-user")

        assert len(logs) >= 2
        for log in logs:
            assert log["user_id"] == "target-user"

    def test_filter_by_severity(self, audit_logger):
        """Test get_logs() filter by severity"""
        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user2", "HIGH")
        audit_logger.log("JWT_KEY_INVALIDATED", "user3", "CRITICAL")

        logs = audit_logger.get_logs(severity="HIGH")

        for log in logs:
            assert log["severity"] == "HIGH"

    def test_filter_by_date_range(self, audit_logger):
        """Test get_logs() filter by date range"""
        start_time = datetime.utcnow()

        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user2", "HIGH")

        end_time = datetime.utcnow()

        logs = audit_logger.get_logs(
            start_date=start_time.isoformat(),
            end_date=end_time.isoformat()
        )

        # All logs should be within date range
        for log in logs:
            log_time = datetime.fromisoformat(log["timestamp"].replace("Z", ""))
            assert start_time <= log_time <= end_time + timedelta(seconds=1)

    def test_filter_multiple_criteria(self, audit_logger):
        """Test get_logs() with multiple filters"""
        audit_logger.log("API_KEY_CREATED", "target-user", "MEDIUM")
        audit_logger.log("API_KEY_CREATED", "other-user", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "target-user", "HIGH")

        logs = audit_logger.get_logs(
            event_type="API_KEY_CREATED",
            user_id="target-user",
            severity="MEDIUM"
        )

        assert len(logs) >= 1
        for log in logs:
            assert log["event_type"] == "API_KEY_CREATED"
            assert log["user_id"] == "target-user"
            assert log["severity"] == "MEDIUM"


# ============================================================================
# STATISTICS
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditStatistics:
    """Test audit statistics aggregation"""

    def test_get_stats_returns_total_events(self, audit_logger):
        """Test get_stats() returns total event count"""
        initial_count = audit_logger.get_stats()["total_events"]

        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user2", "HIGH")

        stats = audit_logger.get_stats()
        assert stats["total_events"] >= initial_count + 2

    def test_get_stats_groups_by_event_type(self, audit_logger):
        """Test get_stats() groups events by type"""
        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("API_KEY_CREATED", "user2", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user3", "HIGH")

        stats = audit_logger.get_stats()

        assert "events_by_type" in stats
        assert stats["events_by_type"]["API_KEY_CREATED"] >= 2
        assert stats["events_by_type"]["JWT_ROTATION"] >= 1

    def test_get_stats_groups_by_severity(self, audit_logger):
        """Test get_stats() groups events by severity"""
        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user2", "HIGH")
        audit_logger.log("JWT_KEY_INVALIDATED", "user3", "CRITICAL")

        stats = audit_logger.get_stats()

        assert "events_by_severity" in stats
        assert stats["events_by_severity"]["MEDIUM"] >= 1
        assert stats["events_by_severity"]["HIGH"] >= 1
        assert stats["events_by_severity"]["CRITICAL"] >= 1

    def test_get_stats_groups_by_user(self, audit_logger):
        """Test get_stats() groups events by user"""
        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user1", "HIGH")
        audit_logger.log("AGENT_RELOAD", "user2", "MEDIUM")

        stats = audit_logger.get_stats()

        assert "events_by_user" in stats
        assert stats["events_by_user"]["user1"] >= 2
        assert stats["events_by_user"]["user2"] >= 1

    def test_get_stats_with_date_range(self, audit_logger):
        """Test get_stats() with date range filter"""
        start_time = datetime.utcnow()

        audit_logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        audit_logger.log("JWT_ROTATION", "user2", "HIGH")

        end_time = datetime.utcnow()

        stats = audit_logger.get_stats(
            start_date=start_time.isoformat(),
            end_date=end_time.isoformat()
        )

        # Should have at least the 2 events we created
        assert stats["total_events"] >= 2


# ============================================================================
# EVENT TYPES LIST
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditEventTypesList:
    """Test event types listing"""

    def test_get_event_types_returns_all_types(self, audit_logger):
        """Test get_event_types() returns all supported event types"""
        event_types = audit_logger.get_event_types()

        expected_types = [
            "API_KEY_CREATED",
            "API_KEY_ROTATED",
            "API_KEY_REVOKED",
            "JWT_ROTATION",
            "JWT_KEY_INVALIDATED",
            "AGENT_RELOAD",
            "MODEL_PROMOTION",
            "HYPERSYNC_APPROVAL",
            "HYPERSYNC_REJECTION",
            "AUTOML_TRIAL_COMPLETE",
            "USER_LOGIN",
            "USER_LOGOUT",
            "RBAC_PERMISSION_DENIED",
            "SYSTEM_HEALTH_DEGRADED",
            "SYSTEM_HEALTH_RECOVERED",
            "CONFIG_CHANGE",
            "DATA_EXPORT",
            "SECURITY_INCIDENT",
            "RATE_LIMIT_EXCEEDED",
            "UNKNOWN_ERROR",
        ]

        for event_type in expected_types:
            assert event_type in event_types


# ============================================================================
# REDIS PERSISTENCE
# ============================================================================

@pytest.mark.integration
@pytest.mark.audit
@pytest.mark.redis
class TestAuditLoggerRedisPersistence:
    """Test audit logger Redis persistence"""

    def test_logs_persist_across_logger_instances(self, mock_redis_client):
        """Test logs persist across AuditLogger instances"""
        from cognition.shared.audit_logger import AuditLogger

        # Create log with first instance
        AuditLogger._instance = None
        logger1 = AuditLogger(redis_client=mock_redis_client)
        event_id = logger1.log("API_KEY_CREATED", "user1", "MEDIUM")

        # Retrieve with second instance
        AuditLogger._instance = None
        logger2 = AuditLogger(redis_client=mock_redis_client)
        log = logger2.get_log(event_id)

        assert log is not None
        assert log["id"] == event_id

    def test_stats_reflect_persisted_data(self, mock_redis_client):
        """Test statistics reflect persisted data"""
        from cognition.shared.audit_logger import AuditLogger

        # Create logs with first instance
        AuditLogger._instance = None
        logger1 = AuditLogger(redis_client=mock_redis_client)
        logger1.log("API_KEY_CREATED", "user1", "MEDIUM")
        logger1.log("JWT_ROTATION", "user2", "HIGH")

        # Get stats with second instance
        AuditLogger._instance = None
        logger2 = AuditLogger(redis_client=mock_redis_client)
        stats = logger2.get_stats()

        assert stats["total_events"] >= 2


# ============================================================================
# STRUCTURED JSON OUTPUT
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditStructuredOutput:
    """Test structured JSON output"""

    def test_log_entry_is_valid_json(self, audit_logger, mock_redis_client):
        """Test log entry is valid JSON"""
        event_id = audit_logger.log(
            "API_KEY_CREATED",
            "test-user",
            "MEDIUM",
            metadata={"key": "value"}
        )

        log_data = mock_redis_client.hget("audit_logs", event_id)

        # Should parse without error
        log_entry = json.loads(log_data)
        assert isinstance(log_entry, dict)

    def test_log_entry_includes_required_fields(self, audit_logger, mock_redis_client):
        """Test log entry includes all required fields"""
        event_id = audit_logger.log(
            "API_KEY_CREATED",
            "test-user",
            "MEDIUM"
        )

        log_data = mock_redis_client.hget("audit_logs", event_id)
        log_entry = json.loads(log_data)

        required_fields = ["id", "event_type", "user_id", "severity", "timestamp"]
        for field in required_fields:
            assert field in log_entry

    def test_log_entry_metadata_is_json_serializable(self, audit_logger, mock_redis_client):
        """Test log entry metadata is JSON serializable"""
        metadata = {
            "string": "value",
            "number": 123,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": "value"}
        }

        event_id = audit_logger.log(
            "API_KEY_CREATED",
            "test-user",
            "MEDIUM",
            metadata=metadata
        )

        log_data = mock_redis_client.hget("audit_logs", event_id)
        log_entry = json.loads(log_data)

        assert log_entry["metadata"] == metadata


# ============================================================================
# FALLBACK MODE
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditLoggerFallback:
    """Test audit logger fallback mode"""

    def test_fallback_when_redis_unavailable(self):
        """Test fallback to in-memory storage when Redis unavailable"""
        from cognition.shared.audit_logger import AuditLogger

        # Create mock that fails
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        AuditLogger._instance = None
        logger = AuditLogger(redis_client=mock_redis)

        # Should still work with in-memory fallback
        event_id = logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        assert event_id is not None

    def test_fallback_mode_supports_retrieval(self):
        """Test fallback mode supports log retrieval"""
        from cognition.shared.audit_logger import AuditLogger

        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        AuditLogger._instance = None
        logger = AuditLogger(redis_client=mock_redis)

        event_id = logger.log("API_KEY_CREATED", "user1", "MEDIUM")
        log = logger.get_log(event_id)

        assert log is not None
        assert log["id"] == event_id


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.unit
@pytest.mark.audit
class TestAuditLoggerEdgeCases:
    """Test edge cases and error handling"""

    def test_log_with_empty_user_id(self, audit_logger):
        """Test log() with empty user ID"""
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            audit_logger.log("API_KEY_CREATED", "", "MEDIUM")

    def test_log_with_none_user_id(self, audit_logger):
        """Test log() with None user ID"""
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            audit_logger.log("API_KEY_CREATED", None, "MEDIUM")

    def test_log_with_invalid_metadata_type(self, audit_logger):
        """Test log() with non-serializable metadata"""
        class NonSerializable:
            pass

        with pytest.raises(TypeError):
            audit_logger.log(
                "API_KEY_CREATED",
                "user1",
                "MEDIUM",
                metadata={"obj": NonSerializable()}
            )

    def test_get_logs_with_invalid_date_format(self, audit_logger):
        """Test get_logs() with invalid date format"""
        with pytest.raises(ValueError):
            audit_logger.get_logs(start_date="not-a-date")

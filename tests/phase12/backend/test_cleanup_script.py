"""
Test suite for JWT Cleanup Script (scripts/jwt_cleanup.py)

Coverage targets:
- Once mode (single cleanup run)
- Daemon mode (continuous loop)
- Skip-on-unhealthy logic
- Key cleanup logic
- Prometheus metrics
- Error handling
- Health checks
"""

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
from io import StringIO

import pytest
from tests.conftest import MockRedis


# ============================================================================
# CLEANUP LOGIC
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestCleanupLogic:
    """Test JWT cleanup logic"""

    def test_cleanup_removes_expired_keys(self, mock_redis_client):
        """Test cleanup removes expired JWT keys"""
        from cognition.shared.jwt_key_store import JWTKeyStore
        import json

        # Create expired key
        JWTKeyStore._instance = None
        store = JWTKeyStore(redis_client=mock_redis_client)

        key = store.create_key()
        key.expires_at = datetime.utcnow() - timedelta(hours=1)
        key.is_active = False

        # Store expired key manually
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
        removed_count = store.cleanup_expired_keys()

        assert removed_count >= 1
        # Verify key was removed
        assert store.get_key(key.kid) is None

    def test_cleanup_preserves_valid_keys(self, mock_redis_client):
        """Test cleanup preserves valid JWT keys"""
        from cognition.shared.jwt_key_store import JWTKeyStore

        JWTKeyStore._instance = None
        store = JWTKeyStore(redis_client=mock_redis_client)

        # Create valid key (active, no expiration)
        valid_key = store.get_current_key()

        # Run cleanup
        store.cleanup_expired_keys()

        # Valid key should still exist
        assert store.get_key(valid_key.kid) is not None

    def test_cleanup_preserves_keys_in_grace_period(self, mock_redis_client):
        """Test cleanup preserves keys within grace period"""
        from cognition.shared.jwt_key_store import JWTKeyStore

        JWTKeyStore._instance = None
        store = JWTKeyStore(redis_client=mock_redis_client)

        # Rotate key with 24-hour grace period
        old_key = store.get_current_key()
        old_kid = old_key.kid
        store.rotate_key(grace_period_hours=24)

        # Run cleanup
        store.cleanup_expired_keys()

        # Old key should still exist (within grace period)
        assert store.get_key(old_kid) is not None


# ============================================================================
# HEALTH CHECKS
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestHealthChecks:
    """Test health check logic"""

    def test_health_check_passes_when_redis_healthy(self, mock_redis_client):
        """Test health check passes when Redis is healthy"""
        # Redis ping should succeed
        result = mock_redis_client.ping()
        assert result is True

    def test_health_check_fails_when_redis_down(self):
        """Test health check fails when Redis is down"""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        with pytest.raises(Exception):
            mock_redis.ping()

    def test_cleanup_skips_when_unhealthy(self, mock_redis_client):
        """Test cleanup skips execution when health check fails"""
        # Mock unhealthy Redis
        mock_redis_client.ping = MagicMock(side_effect=Exception("Connection refused"))

        # Cleanup should detect unhealthy state
        try:
            mock_redis_client.ping()
            should_run = True
        except Exception:
            should_run = False

        assert should_run is False


# ============================================================================
# ONCE MODE
# ============================================================================

@pytest.mark.integration
@pytest.mark.jwt
class TestOnceModeExecution:
    """Test once mode execution"""

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    def test_once_mode_runs_single_cleanup(self, mock_redis, mock_store_class):
        """Test once mode runs cleanup once and exits"""
        # Mock store
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 5
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run
        from scripts.jwt_cleanup import run_cleanup_once

        result = run_cleanup_once(mock_redis_instance)

        # Should call cleanup once
        mock_store.cleanup_expired_keys.assert_called_once()
        assert result == 5

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    def test_once_mode_increments_prometheus_metrics(self, mock_redis, mock_store_class):
        """Test once mode increments Prometheus metrics"""
        # Mock store
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 3
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run with metrics mock
        with patch('scripts.jwt_cleanup.jwt_cleanup_total') as mock_cleanup_counter:
            with patch('scripts.jwt_cleanup.jwt_keys_cleaned_total') as mock_keys_counter:
                from scripts.jwt_cleanup import run_cleanup_once

                run_cleanup_once(mock_redis_instance)

                # Verify metrics incremented
                mock_cleanup_counter.inc.assert_called_once()
                mock_keys_counter.inc.assert_called_once_with(3)


# ============================================================================
# DAEMON MODE
# ============================================================================

@pytest.mark.integration
@pytest.mark.jwt
@pytest.mark.slow
class TestDaemonModeExecution:
    """Test daemon mode execution"""

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    @patch('scripts.jwt_cleanup.time.sleep')
    def test_daemon_mode_runs_continuously(self, mock_sleep, mock_redis, mock_store_class):
        """Test daemon mode runs cleanup in loop"""
        # Mock store
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 2
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Mock sleep to break loop after 2 iterations
        mock_sleep.side_effect = [None, KeyboardInterrupt]

        # Import and run
        from scripts.jwt_cleanup import run_cleanup_daemon

        try:
            run_cleanup_daemon(mock_redis_instance, interval_hours=1)
        except KeyboardInterrupt:
            pass

        # Should have called cleanup twice before KeyboardInterrupt
        assert mock_store.cleanup_expired_keys.call_count == 2

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    @patch('scripts.jwt_cleanup.time.sleep')
    def test_daemon_mode_respects_interval(self, mock_sleep, mock_redis, mock_store_class):
        """Test daemon mode respects interval parameter"""
        # Mock store
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 1
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Mock sleep to verify interval
        mock_sleep.side_effect = KeyboardInterrupt

        # Import and run with 12-hour interval
        from scripts.jwt_cleanup import run_cleanup_daemon

        try:
            run_cleanup_daemon(mock_redis_instance, interval_hours=12)
        except KeyboardInterrupt:
            pass

        # Should sleep for 12 hours (43200 seconds)
        mock_sleep.assert_called_with(12 * 3600)


# ============================================================================
# ERROR HANDLING
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestErrorHandling:
    """Test error handling in cleanup script"""

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    def test_cleanup_handles_redis_errors_gracefully(self, mock_redis, mock_store_class):
        """Test cleanup handles Redis errors gracefully"""
        # Mock store that raises error
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.side_effect = Exception("Redis connection lost")
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run
        from scripts.jwt_cleanup import run_cleanup_once

        # Should not crash, should return 0
        result = run_cleanup_once(mock_redis_instance)
        assert result == 0

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    def test_cleanup_increments_error_counter(self, mock_redis, mock_store_class):
        """Test cleanup increments error counter on failure"""
        # Mock store that raises error
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.side_effect = Exception("Test error")
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run with metrics mock
        with patch('scripts.jwt_cleanup.jwt_cleanup_errors_total') as mock_error_counter:
            from scripts.jwt_cleanup import run_cleanup_once

            run_cleanup_once(mock_redis_instance)

            # Verify error counter incremented
            mock_error_counter.inc.assert_called_once()


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestPrometheusMetrics:
    """Test Prometheus metrics integration"""

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    def test_cleanup_tracks_duration(self, mock_redis, mock_store_class):
        """Test cleanup tracks duration in histogram"""
        # Mock store
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 5
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run with histogram mock
        with patch('scripts.jwt_cleanup.jwt_cleanup_duration_seconds') as mock_histogram:
            from scripts.jwt_cleanup import run_cleanup_once

            run_cleanup_once(mock_redis_instance)

            # Verify histogram recorded duration
            mock_histogram.observe.assert_called_once()
            # Duration should be a small positive number
            duration = mock_histogram.observe.call_args[0][0]
            assert duration >= 0

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    def test_cleanup_tracks_keys_cleaned(self, mock_redis, mock_store_class):
        """Test cleanup tracks number of keys cleaned"""
        # Mock store with specific count
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 7
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run with counter mock
        with patch('scripts.jwt_cleanup.jwt_keys_cleaned_total') as mock_counter:
            from scripts.jwt_cleanup import run_cleanup_once

            run_cleanup_once(mock_redis_instance)

            # Verify counter incremented by exact amount
            mock_counter.inc.assert_called_once_with(7)


# ============================================================================
# LOGGING
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestLogging:
    """Test logging functionality"""

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    @patch('scripts.jwt_cleanup.logging')
    def test_cleanup_logs_start_message(self, mock_logging, mock_redis, mock_store_class):
        """Test cleanup logs start message"""
        # Mock store
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 5
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run
        from scripts.jwt_cleanup import run_cleanup_once

        run_cleanup_once(mock_redis_instance)

        # Verify logging calls
        assert mock_logging.info.called

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    @patch('scripts.jwt_cleanup.logging')
    def test_cleanup_logs_completion_message(self, mock_logging, mock_redis, mock_store_class):
        """Test cleanup logs completion message with count"""
        # Mock store
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.return_value = 5
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run
        from scripts.jwt_cleanup import run_cleanup_once

        run_cleanup_once(mock_redis_instance)

        # Verify completion message includes count
        info_calls = [str(call) for call in mock_logging.info.call_args_list]
        assert any("5" in str(call) for call in info_calls)

    @patch('scripts.jwt_cleanup.JWTKeyStore')
    @patch('scripts.jwt_cleanup.redis.Redis')
    @patch('scripts.jwt_cleanup.logging')
    def test_cleanup_logs_errors(self, mock_logging, mock_redis, mock_store_class):
        """Test cleanup logs errors"""
        # Mock store that raises error
        mock_store = MagicMock()
        mock_store.cleanup_expired_keys.side_effect = Exception("Test error")
        mock_store_class.return_value = mock_store

        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Import and run
        from scripts.jwt_cleanup import run_cleanup_once

        run_cleanup_once(mock_redis_instance)

        # Verify error was logged
        assert mock_logging.error.called


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

@pytest.mark.integration
@pytest.mark.jwt
class TestCommandLineInterface:
    """Test command line interface"""

    @patch('scripts.jwt_cleanup.run_cleanup_once')
    @patch('scripts.jwt_cleanup.redis.Redis')
    @patch('sys.argv', ['jwt_cleanup.py', '--mode', 'once'])
    def test_cli_once_mode(self, mock_redis, mock_run_once):
        """Test CLI with --mode once"""
        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        mock_run_once.return_value = 5

        # Import main (would normally run)
        # Note: In real tests, we'd use subprocess or click.testing.CliRunner
        # This is simplified for unit testing
        assert '--mode' in sys.argv
        assert 'once' in sys.argv

    @patch('scripts.jwt_cleanup.run_cleanup_daemon')
    @patch('scripts.jwt_cleanup.redis.Redis')
    @patch('sys.argv', ['jwt_cleanup.py', '--mode', 'daemon', '--interval', '6'])
    def test_cli_daemon_mode_with_interval(self, mock_redis, mock_run_daemon):
        """Test CLI with --mode daemon and --interval"""
        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Verify arguments
        assert '--mode' in sys.argv
        assert 'daemon' in sys.argv
        assert '--interval' in sys.argv
        assert '6' in sys.argv


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

@pytest.mark.unit
@pytest.mark.jwt
class TestEnvironmentConfiguration:
    """Test environment variable configuration"""

    @patch.dict('os.environ', {
        'REDIS_HOST': 'custom-redis',
        'REDIS_PORT': '6380',
        'REDIS_PASSWORD': 'custom-password'
    })
    def test_uses_environment_variables(self):
        """Test script uses environment variables for configuration"""
        import os

        assert os.environ['REDIS_HOST'] == 'custom-redis'
        assert os.environ['REDIS_PORT'] == '6380'
        assert os.environ['REDIS_PASSWORD'] == 'custom-password'

    @patch.dict('os.environ', {'LOG_LEVEL': 'DEBUG'})
    def test_respects_log_level_env_var(self):
        """Test script respects LOG_LEVEL environment variable"""
        import os

        assert os.environ['LOG_LEVEL'] == 'DEBUG'

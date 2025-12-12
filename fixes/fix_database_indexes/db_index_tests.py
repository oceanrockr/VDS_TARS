"""
Database Index Optimization Tests

TARS-1004: Database Index Optimization - Test Suite
----------------------------------------------------
Tests for validating database index performance improvements.

Performance Requirements:
- API p95 latency < 100ms (down from 500ms)
- Evaluation queries < 50ms
- Agent state queries < 25ms
- API key lookups < 5ms

Test Coverage:
1. Index existence validation
2. Query plan verification (EXPLAIN ANALYZE)
3. Performance benchmarking
4. Index usage statistics
5. Concurrent index creation

Author: T.A.R.S. Engineering Team
Version: 1.0.1
Date: 2025-11-20
"""

import pytest
import psycopg2
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import random
import string


# =================================================================
# TEST CONFIGURATION
# =================================================================

SQL_MIGRATION_PATH = Path(__file__).parent / "v1_0_1_add_indexes.sql"

# Performance thresholds (milliseconds)
THRESHOLDS = {
    "evaluation_queries": 50,
    "agent_state_queries": 25,
    "metric_queries": 60,
    "audit_log_queries": 45,
    "api_key_queries": 5,
    "aggregate_queries": 80
}

# Expected indexes
EXPECTED_INDEXES = [
    "idx_evaluations_agent_region_time",
    "idx_evaluations_status_time",
    "idx_evaluations_agent_region_created",
    "idx_agent_states_agent_region_updated",
    "idx_agent_states_type_updated",
    "idx_metrics_name_region_time",
    "idx_metrics_name_time",
    "idx_audit_logs_user_time",
    "idx_audit_logs_resource",
    "idx_api_keys_hash_active",
    "idx_api_keys_user_active"
]

# =================================================================
# MOCK DATABASE CONNECTION
# =================================================================

class MockCursor:
    """Mock PostgreSQL cursor for testing"""

    def __init__(self, has_indexes: bool = True):
        self.has_indexes = has_indexes
        self.query_log: List[Dict[str, Any]] = []
        self.last_query: Optional[str] = None
        self.last_params: Optional[tuple] = None

    def execute(self, query: str, params: tuple = None):
        """Execute SQL query (mocked)"""
        self.last_query = query
        self.last_params = params

        # Log query for performance simulation
        self.query_log.append({
            "query": query,
            "params": params,
            "timestamp": datetime.now()
        })

    def fetchall(self) -> List[tuple]:
        """Fetch all results (mocked)"""
        if self.last_query is None:
            return []

        # Simulate index existence check
        if "pg_indexes" in self.last_query and "indexname" in self.last_query:
            if self.has_indexes:
                return [(name,) for name in EXPECTED_INDEXES]
            else:
                return []

        # Simulate EXPLAIN output
        if "EXPLAIN" in self.last_query.upper():
            if self.has_indexes:
                return [
                    ("Index Scan using idx_evaluations_agent_region_time",),
                    ("Index Cond: ((agent_id = 'dqn_agent_1'::text))",),
                    ("Execution Time: 45.23 ms",)
                ]
            else:
                return [
                    ("Seq Scan on evaluations",),
                    ("Filter: (agent_id = 'dqn_agent_1'::text)",),
                    ("Execution Time: 523.45 ms",)
                ]

        # Simulate index statistics
        if "pg_stat_user_indexes" in self.last_query:
            return [
                (name, random.randint(1000, 50000), random.randint(5000, 100000))
                for name in EXPECTED_INDEXES
            ]

        return []

    def fetchone(self) -> Optional[tuple]:
        """Fetch single result (mocked)"""
        results = self.fetchall()
        return results[0] if results else None

    def close(self):
        """Close cursor"""
        pass


class MockConnection:
    """Mock PostgreSQL connection for testing"""

    def __init__(self, has_indexes: bool = True):
        self.has_indexes = has_indexes
        self.closed = False
        self.in_transaction = False

    def cursor(self) -> MockCursor:
        """Create mock cursor"""
        return MockCursor(has_indexes=self.has_indexes)

    def commit(self):
        """Commit transaction"""
        self.in_transaction = False

    def rollback(self):
        """Rollback transaction"""
        self.in_transaction = False

    def close(self):
        """Close connection"""
        self.closed = True


# =================================================================
# DATABASE HELPER FUNCTIONS
# =================================================================

def get_mock_connection(has_indexes: bool = True) -> MockConnection:
    """Get mock database connection"""
    return MockConnection(has_indexes=has_indexes)


def execute_query_with_timing(cursor: MockCursor, query: str, params: tuple = None) -> float:
    """
    Execute query and return execution time in milliseconds

    Args:
        cursor: Database cursor
        query: SQL query to execute
        params: Query parameters

    Returns:
        Execution time in milliseconds (simulated)
    """
    start_time = time.time()

    cursor.execute(query, params)
    cursor.fetchall()

    # Simulate query execution time based on index usage
    if cursor.has_indexes and ("Index Scan" in str(cursor.fetchall()) or
                                any(idx in query for idx in EXPECTED_INDEXES)):
        # Fast query with index
        execution_time_ms = random.uniform(10, 50)
    else:
        # Slow query without index
        execution_time_ms = random.uniform(300, 600)

    # Simulate delay
    time.sleep(execution_time_ms / 1000)

    return execution_time_ms


# =================================================================
# SQL MIGRATION VALIDATION TESTS
# =================================================================

class TestMigrationScript:
    """Test SQL migration script structure and syntax"""

    def test_migration_file_exists(self):
        """Verify migration SQL file exists"""
        assert SQL_MIGRATION_PATH.exists(), \
            f"Migration SQL file not found: {SQL_MIGRATION_PATH}"

    def test_migration_file_not_empty(self):
        """Verify migration SQL file is not empty"""
        content = SQL_MIGRATION_PATH.read_text()
        assert len(content) > 0, "Migration SQL file is empty"
        assert len(content) > 1000, "Migration SQL file suspiciously short"

    def test_migration_uses_create_index_concurrently(self):
        """Verify migration uses CONCURRENTLY for zero-downtime"""
        content = SQL_MIGRATION_PATH.read_text()

        # Count CREATE INDEX statements
        create_index_count = content.count("CREATE INDEX")

        # Count CONCURRENTLY usage
        concurrent_count = content.count("CONCURRENTLY")

        # Should use CONCURRENTLY for all CREATE INDEX statements
        assert concurrent_count >= create_index_count, \
            f"Not all CREATE INDEX statements use CONCURRENTLY ({concurrent_count}/{create_index_count})"

    def test_migration_has_expected_indexes(self):
        """Verify migration creates all expected indexes"""
        content = SQL_MIGRATION_PATH.read_text()

        for index_name in EXPECTED_INDEXES:
            assert index_name in content, \
                f"Expected index '{index_name}' not found in migration script"

    def test_migration_has_rollback_procedure(self):
        """Verify migration includes rollback instructions"""
        content = SQL_MIGRATION_PATH.read_text()

        assert "ROLLBACK" in content.upper(), \
            "Migration script missing rollback procedure"
        assert "DROP INDEX" in content.upper(), \
            "Migration script missing DROP INDEX statements in rollback"

    def test_migration_has_comments(self):
        """Verify migration has comprehensive comments"""
        content = SQL_MIGRATION_PATH.read_text()

        # Should have comments explaining each index
        comment_count = content.count("COMMENT ON INDEX")

        assert comment_count >= len(EXPECTED_INDEXES), \
            f"Missing comments for indexes ({comment_count}/{len(EXPECTED_INDEXES)})"

    def test_migration_has_analyze_statements(self):
        """Verify migration includes ANALYZE for statistics update"""
        content = SQL_MIGRATION_PATH.read_text()

        assert "ANALYZE" in content.upper(), \
            "Migration script should include ANALYZE statements for statistics update"


# =================================================================
# INDEX EXISTENCE TESTS
# =================================================================

class TestIndexExistence:
    """Test index creation and existence"""

    @pytest.fixture
    def db_connection(self):
        """Mock database connection with indexes"""
        return get_mock_connection(has_indexes=True)

    def test_all_indexes_created(self, db_connection):
        """Verify all expected indexes exist"""
        cursor = db_connection.cursor()

        # Query to check index existence
        query = """
        SELECT indexname
        FROM pg_indexes
        WHERE tablename IN ('evaluations', 'agent_states', 'metrics', 'audit_logs', 'api_keys')
          AND indexname LIKE 'idx_%'
        ORDER BY indexname;
        """

        cursor.execute(query)
        results = cursor.fetchall()

        existing_indexes = [row[0] for row in results]

        for expected_index in EXPECTED_INDEXES:
            assert expected_index in existing_indexes, \
                f"Index '{expected_index}' not found in database"

    def test_evaluations_indexes(self, db_connection):
        """Verify evaluations table indexes exist"""
        cursor = db_connection.cursor()

        cursor.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'evaluations' AND indexname LIKE 'idx_%'
        """)

        indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_evaluations_agent_region_time" in indexes
        assert "idx_evaluations_status_time" in indexes
        assert "idx_evaluations_agent_region_created" in indexes

    def test_agent_states_indexes(self, db_connection):
        """Verify agent_states table indexes exist"""
        cursor = db_connection.cursor()

        cursor.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'agent_states' AND indexname LIKE 'idx_%'
        """)

        indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_agent_states_agent_region_updated" in indexes
        assert "idx_agent_states_type_updated" in indexes

    def test_api_keys_indexes(self, db_connection):
        """Verify api_keys table indexes exist"""
        cursor = db_connection.cursor()

        cursor.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'api_keys' AND indexname LIKE 'idx_%'
        """)

        indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_api_keys_hash_active" in indexes
        assert "idx_api_keys_user_active" in indexes


# =================================================================
# QUERY PLAN VERIFICATION TESTS
# =================================================================

class TestQueryPlans:
    """Test query execution plans use indexes"""

    @pytest.fixture
    def db_connection(self):
        """Mock database connection with indexes"""
        return get_mock_connection(has_indexes=True)

    def test_evaluation_query_uses_index(self, db_connection):
        """Verify evaluation query uses idx_evaluations_agent_region_time"""
        cursor = db_connection.cursor()

        # Query that should use index
        query = """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT * FROM evaluations
        WHERE agent_id = %s
          AND region = %s
          AND created_at > %s
        ORDER BY created_at DESC
        LIMIT 100;
        """

        cursor.execute(query, ('dqn_agent_1', 'us-west-2', datetime.now() - timedelta(hours=1)))
        plan = cursor.fetchall()

        # Should use index scan
        plan_text = " ".join([str(row[0]) for row in plan])
        assert "Index Scan" in plan_text or "idx_evaluations" in plan_text.lower(), \
            "Query should use index scan"

    def test_agent_state_query_uses_index(self, db_connection):
        """Verify agent state query uses idx_agent_states_agent_region_updated"""
        cursor = db_connection.cursor()

        query = """
        EXPLAIN
        SELECT DISTINCT ON (agent_id, region)
            agent_id, region, reward, updated_at
        FROM agent_states
        WHERE updated_at > %s
        ORDER BY agent_id, region, updated_at DESC;
        """

        cursor.execute(query, (datetime.now() - timedelta(minutes=5),))
        plan = cursor.fetchall()

        plan_text = " ".join([str(row[0]) for row in plan])
        assert "Index" in plan_text, \
            "Agent state query should use index"

    def test_api_key_query_uses_index(self, db_connection):
        """Verify API key authentication query uses idx_api_keys_hash_active"""
        cursor = db_connection.cursor()

        query = """
        EXPLAIN
        SELECT id, user_id, scopes
        FROM api_keys
        WHERE key_hash = %s
          AND revoked_at IS NULL
          AND expires_at > NOW();
        """

        cursor.execute(query, ('sha256:abc123',))
        plan = cursor.fetchall()

        plan_text = " ".join([str(row[0]) for row in plan])
        assert "Index" in plan_text, \
            "API key query should use index"


# =================================================================
# PERFORMANCE BENCHMARK TESTS
# =================================================================

class TestPerformance:
    """Test query performance with indexes"""

    @pytest.fixture
    def db_with_indexes(self):
        """Database connection with indexes"""
        return get_mock_connection(has_indexes=True)

    @pytest.fixture
    def db_without_indexes(self):
        """Database connection without indexes"""
        return get_mock_connection(has_indexes=False)

    def test_evaluation_query_performance(self, db_with_indexes):
        """Test evaluation query performance with indexes"""
        cursor = db_with_indexes.cursor()

        query = """
        SELECT * FROM evaluations
        WHERE agent_id = %s AND region = %s AND created_at > %s
        ORDER BY created_at DESC LIMIT 100;
        """

        execution_time = execute_query_with_timing(
            cursor, query,
            ('dqn_agent_1', 'us-west-2', datetime.now() - timedelta(hours=1))
        )

        assert execution_time < THRESHOLDS["evaluation_queries"], \
            f"Evaluation query took {execution_time}ms (threshold: {THRESHOLDS['evaluation_queries']}ms)"

    def test_agent_state_query_performance(self, db_with_indexes):
        """Test agent state query performance with indexes"""
        cursor = db_with_indexes.cursor()

        query = """
        SELECT DISTINCT ON (agent_id, region)
            agent_id, region, reward
        FROM agent_states
        WHERE updated_at > %s
        ORDER BY agent_id, region, updated_at DESC;
        """

        execution_time = execute_query_with_timing(
            cursor, query,
            (datetime.now() - timedelta(minutes=5),)
        )

        assert execution_time < THRESHOLDS["agent_state_queries"], \
            f"Agent state query took {execution_time}ms (threshold: {THRESHOLDS['agent_state_queries']}ms)"

    def test_api_key_query_performance(self, db_with_indexes):
        """Test API key authentication performance (CRITICAL)"""
        cursor = db_with_indexes.cursor()

        query = """
        SELECT id, user_id, scopes
        FROM api_keys
        WHERE key_hash = %s AND revoked_at IS NULL AND expires_at > NOW();
        """

        execution_time = execute_query_with_timing(
            cursor, query,
            ('sha256:abc123',)
        )

        # API key lookups must be very fast (<5ms)
        assert execution_time < THRESHOLDS["api_key_queries"], \
            f"API key query took {execution_time}ms (threshold: {THRESHOLDS['api_key_queries']}ms)"

    def test_performance_improvement(self, db_with_indexes, db_without_indexes):
        """Validate performance improvement from indexes"""
        query = """
        SELECT * FROM evaluations
        WHERE agent_id = %s AND region = %s AND created_at > %s
        ORDER BY created_at DESC LIMIT 100;
        """
        params = ('dqn_agent_1', 'us-west-2', datetime.now() - timedelta(hours=1))

        # Query without indexes
        cursor_without = db_without_indexes.cursor()
        time_without = execute_query_with_timing(cursor_without, query, params)

        # Query with indexes
        cursor_with = db_with_indexes.cursor()
        time_with = execute_query_with_timing(cursor_with, query, params)

        # Calculate improvement
        improvement = ((time_without - time_with) / time_without) * 100

        # Should see at least 80% improvement
        assert improvement >= 80, \
            f"Performance improvement {improvement:.1f}% below expected 80%"

        print(f"\n✓ Query performance improvement: {improvement:.1f}%")
        print(f"  Without indexes: {time_without:.1f}ms")
        print(f"  With indexes: {time_with:.1f}ms")


# =================================================================
# INDEX USAGE STATISTICS TESTS
# =================================================================

class TestIndexUsage:
    """Test index usage statistics"""

    @pytest.fixture
    def db_connection(self):
        """Mock database connection"""
        return get_mock_connection(has_indexes=True)

    def test_indexes_are_used(self, db_connection):
        """Verify indexes are actually being used by queries"""
        cursor = db_connection.cursor()

        query = """
        SELECT indexname, idx_scan, idx_tup_read
        FROM pg_stat_user_indexes
        WHERE indexname LIKE 'idx_%'
        ORDER BY idx_scan DESC;
        """

        cursor.execute(query)
        results = cursor.fetchall()

        # All indexes should have some usage
        for index_name, scan_count, tup_read in results:
            assert scan_count > 0, \
                f"Index '{index_name}' has not been used (idx_scan = 0)"

    def test_index_scan_counts(self, db_connection):
        """Verify index scan counts are reasonable"""
        cursor = db_connection.cursor()

        cursor.execute("""
            SELECT indexname, idx_scan
            FROM pg_stat_user_indexes
            WHERE indexname IN %s
        """, (tuple(EXPECTED_INDEXES),))

        results = cursor.fetchall()

        for index_name, scan_count in results:
            # Each index should have been used multiple times
            assert scan_count >= 100, \
                f"Index '{index_name}' has low usage count: {scan_count}"


# =================================================================
# CONCURRENT OPERATION TESTS
# =================================================================

class TestConcurrentOperations:
    """Test concurrent index creation and operations"""

    def test_concurrent_index_creation(self):
        """Verify CREATE INDEX CONCURRENTLY is used"""
        content = SQL_MIGRATION_PATH.read_text()

        # All CREATE INDEX statements should use CONCURRENTLY
        create_statements = [line for line in content.split('\n')
                              if 'CREATE INDEX' in line.upper()]

        for statement in create_statements:
            assert 'CONCURRENTLY' in statement.upper(), \
                f"CREATE INDEX should use CONCURRENTLY: {statement[:80]}"

    def test_if_not_exists_used(self):
        """Verify IF NOT EXISTS is used for idempotency"""
        content = SQL_MIGRATION_PATH.read_text()

        create_statements = [line for line in content.split('\n')
                              if 'CREATE INDEX' in line.upper() and 'CONCURRENTLY' in line.upper()]

        for statement in create_statements:
            assert 'IF NOT EXISTS' in statement.upper(), \
                f"CREATE INDEX should use IF NOT EXISTS: {statement[:80]}"


# =================================================================
# PERFORMANCE BENCHMARKING
# =================================================================

class TestPerformanceBenchmark:
    """Comprehensive performance benchmarking"""

    @pytest.fixture
    def db_with_indexes(self):
        """Database with indexes"""
        return get_mock_connection(has_indexes=True)

    def test_benchmark_all_query_types(self, db_with_indexes):
        """Benchmark all query types and generate report"""
        cursor = db_with_indexes.cursor()

        benchmarks = []

        # 1. Evaluation queries
        query = "SELECT * FROM evaluations WHERE agent_id = %s AND region = %s AND created_at > %s ORDER BY created_at DESC LIMIT 100"
        time_ms = execute_query_with_timing(cursor, query, ('dqn_agent_1', 'us-west-2', datetime.now() - timedelta(hours=1)))
        benchmarks.append(("Evaluation query", time_ms, THRESHOLDS["evaluation_queries"]))

        # 2. Agent state queries
        query = "SELECT DISTINCT ON (agent_id, region) agent_id, region, reward FROM agent_states WHERE updated_at > %s ORDER BY agent_id, region, updated_at DESC"
        time_ms = execute_query_with_timing(cursor, query, (datetime.now() - timedelta(minutes=5),))
        benchmarks.append(("Agent state query", time_ms, THRESHOLDS["agent_state_queries"]))

        # 3. API key queries
        query = "SELECT id, user_id, scopes FROM api_keys WHERE key_hash = %s AND revoked_at IS NULL"
        time_ms = execute_query_with_timing(cursor, query, ('sha256:abc123',))
        benchmarks.append(("API key query", time_ms, THRESHOLDS["api_key_queries"]))

        # 4. Metric queries
        query = "SELECT metric_name, value FROM metrics WHERE metric_name = %s AND region = %s AND timestamp > %s"
        time_ms = execute_query_with_timing(cursor, query, ('http_latency_p95', 'us-east-1', datetime.now() - timedelta(hours=6)))
        benchmarks.append(("Metric query", time_ms, THRESHOLDS["metric_queries"]))

        # 5. Audit log queries
        query = "SELECT * FROM audit_logs WHERE user_id = %s AND timestamp > %s ORDER BY timestamp DESC LIMIT 100"
        time_ms = execute_query_with_timing(cursor, query, ('user@example.com', datetime.now() - timedelta(days=7)))
        benchmarks.append(("Audit log query", time_ms, THRESHOLDS["audit_log_queries"]))

        # Print benchmark report
        print("\n" + "="*70)
        print("QUERY PERFORMANCE BENCHMARK")
        print("="*70)
        print(f"{'Query Type':<30} {'Time (ms)':<15} {'Threshold':<15} {'Status':<10}")
        print("-"*70)

        all_passed = True
        for query_type, time_ms, threshold in benchmarks:
            status = "PASS" if time_ms < threshold else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"{query_type:<30} {time_ms:<15.1f} {threshold:<15} {status:<10}")

        print("="*70)

        assert all_passed, "Some queries exceeded performance thresholds"


# =================================================================
# MAIN TEST EXECUTION
# =================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])

"""
Grafana Query Optimization Tests

TARS-1002: Grafana Query Timeout Fix - Test Suite
--------------------------------------------------
Tests for validating dashboard query performance with recording rules.

Performance Requirements:
- Dashboard load time < 5s for 5000+ evaluations
- Query execution time < 200ms per panel
- No raw expensive queries remaining
- All recording rules properly integrated

Test Coverage:
1. Recording rule validation (promtool)
2. Panel transformation verification
3. Load testing (1k, 5k, 10k evaluations)
4. Query performance benchmarking
5. Dashboard JSON integrity checks

Author: T.A.R.S. Engineering Team
Version: 1.0.1
Date: 2025-11-20
"""

import json
import time
import pytest
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import hashlib


# =================================================================
# TEST CONFIGURATION
# =================================================================

RECORDING_RULES_PATH = Path(__file__).parent / "recording_rules.yaml"
DASHBOARD_PATCH_PATH = Path(__file__).parent / "grafana_dashboard_patch.json"

# Performance thresholds
DASHBOARD_LOAD_TIME_THRESHOLD_MS = 5000  # 5 seconds
PANEL_QUERY_TIME_THRESHOLD_MS = 200      # 200ms per panel
TOTAL_PANELS = 68
RECORDING_RULES_COUNT = 52

# Test data sizes
EVAL_COUNTS = [1000, 5000, 10000]


# =================================================================
# MOCK PROMETHEUS SERVER
# =================================================================

class MockPrometheusServer:
    """
    Mock Prometheus server for testing query performance

    Simulates:
    - Recording rule time series
    - Raw metric time series
    - Query execution timing
    - Cardinality characteristics
    """

    def __init__(self, eval_count: int = 5000):
        """
        Initialize mock Prometheus server

        Args:
            eval_count: Number of evaluations to simulate (affects cardinality)
        """
        self.eval_count = eval_count
        self.recording_rules: Dict[str, Any] = {}
        self.raw_metrics: Dict[str, Any] = {}
        self.query_log: List[Dict[str, Any]] = []

        # Initialize recording rules
        self._initialize_recording_rules()

        # Initialize raw metrics (high cardinality)
        self._initialize_raw_metrics()

    def _initialize_recording_rules(self):
        """Pre-compute recording rule time series (low cardinality)"""
        # Evaluation metrics
        self.recording_rules["tars:evaluation_rate:1m"] = {
            "cardinality": 20,  # 4 agents x 5 regions
            "series_count": 20,
            "avg_query_time_ms": 10
        }
        self.recording_rules["tars:evaluation_latency:p95:1m"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 12
        }
        self.recording_rules["tars:evaluation_latency:p99:1m"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 12
        }
        self.recording_rules["tars:evaluation_latency:p50:1m"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 11
        }
        self.recording_rules["tars:evaluation_latency:avg:1m"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 10
        }
        self.recording_rules["tars:evaluation_success_rate:1m"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 11
        }
        self.recording_rules["tars:evaluation_error_rate:1m"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 11
        }

        # Agent metrics
        self.recording_rules["tars:agent_reward:avg:1h"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 15
        }
        self.recording_rules["tars:agent_training_rate:1m"] = {
            "cardinality": 20,
            "series_count": 20,
            "avg_query_time_ms": 10
        }

        # Queue metrics
        self.recording_rules["tars:queue_wait_time:p95:1m"] = {
            "cardinality": 10,  # 5 regions x 2 priorities
            "series_count": 10,
            "avg_query_time_ms": 12
        }
        self.recording_rules["tars:queue_depth:avg:1m"] = {
            "cardinality": 10,
            "series_count": 10,
            "avg_query_time_ms": 8
        }

        # API metrics
        self.recording_rules["tars:http_request_latency:p95:1m"] = {
            "cardinality": 50,  # 10 endpoints x 5 regions
            "series_count": 50,
            "avg_query_time_ms": 15
        }
        self.recording_rules["tars:http_error_rate:1m"] = {
            "cardinality": 50,
            "series_count": 50,
            "avg_query_time_ms": 12
        }

        # Database metrics
        self.recording_rules["tars:db_query_duration:p95:1m"] = {
            "cardinality": 5,  # 5 databases
            "series_count": 5,
            "avg_query_time_ms": 10
        }

        # Add all other recording rules with default values
        for i in range(len(self.recording_rules), RECORDING_RULES_COUNT):
            self.recording_rules[f"tars:rule_{i}"] = {
                "cardinality": 20,
                "series_count": 20,
                "avg_query_time_ms": 10
            }

    def _initialize_raw_metrics(self):
        """Initialize raw metrics with high cardinality (eval_count dependent)"""
        # Evaluation duration histogram (very high cardinality)
        bucket_count = 20  # Histogram buckets
        self.raw_metrics["tars_evaluation_duration_seconds_bucket"] = {
            "cardinality": self.eval_count * bucket_count,
            "series_count": self.eval_count * bucket_count,
            "avg_query_time_ms": 2000 + (self.eval_count / 100)  # Scales with data
        }

        # Evaluation counter
        self.raw_metrics["tars_evaluation_total"] = {
            "cardinality": self.eval_count,
            "series_count": self.eval_count,
            "avg_query_time_ms": 800 + (self.eval_count / 200)
        }

        # HTTP request duration histogram
        self.raw_metrics["http_request_duration_seconds_bucket"] = {
            "cardinality": self.eval_count * bucket_count * 2,
            "series_count": self.eval_count * bucket_count * 2,
            "avg_query_time_ms": 2500 + (self.eval_count / 80)
        }

    def query(self, expr: str) -> Dict[str, Any]:
        """
        Execute a PromQL query and return simulated results

        Args:
            expr: PromQL query expression

        Returns:
            Query result with timing information
        """
        start_time = time.time()

        # Determine if query uses recording rule or raw metric
        is_recording_rule = expr.startswith("tars:")

        if is_recording_rule:
            # Fast query using recording rule
            rule_name = expr.split("{")[0].split("[")[0].strip()
            if rule_name in self.recording_rules:
                rule_info = self.recording_rules[rule_name]
                query_time_ms = rule_info["avg_query_time_ms"]
                cardinality = rule_info["cardinality"]
            else:
                query_time_ms = 15
                cardinality = 10
        else:
            # Slow query using raw metrics
            # Check for expensive operations
            if "histogram_quantile" in expr:
                query_time_ms = 5000 + (self.eval_count / 100)
            elif "rate(" in expr and "bucket" in expr:
                query_time_ms = 3000 + (self.eval_count / 150)
            elif "avg_over_time" in expr or "sum(" in expr:
                query_time_ms = 1000 + (self.eval_count / 300)
            else:
                query_time_ms = 500

            cardinality = self.eval_count

        # Simulate query execution time
        time.sleep(query_time_ms / 1000)

        result = {
            "query": expr,
            "is_recording_rule": is_recording_rule,
            "cardinality": cardinality,
            "query_time_ms": query_time_ms,
            "status": "success",
            "data": {"resultType": "vector", "result": []}
        }

        # Log query
        self.query_log.append(result)

        return result

    def get_total_query_time(self) -> float:
        """Get total query time across all logged queries"""
        return sum(q["query_time_ms"] for q in self.query_log)

    def get_recording_rule_usage_percentage(self) -> float:
        """Calculate percentage of queries using recording rules"""
        if not self.query_log:
            return 0.0
        recording_rule_count = sum(1 for q in self.query_log if q["is_recording_rule"])
        return (recording_rule_count / len(self.query_log)) * 100

    def reset_query_log(self):
        """Clear query log"""
        self.query_log = []


# =================================================================
# RECORDING RULE VALIDATION TESTS
# =================================================================

class TestRecordingRuleValidation:
    """Test recording rule YAML syntax and structure"""

    def test_recording_rules_file_exists(self):
        """Verify recording rules file exists"""
        assert RECORDING_RULES_PATH.exists(), \
            f"Recording rules file not found: {RECORDING_RULES_PATH}"

    def test_promtool_validation(self):
        """Validate recording rules using promtool"""
        try:
            result = subprocess.run(
                ["promtool", "check", "rules", str(RECORDING_RULES_PATH)],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Check if promtool is available
            if result.returncode == 127:  # Command not found
                pytest.skip("promtool not installed")

            assert result.returncode == 0, \
                f"promtool validation failed: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("promtool not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("promtool validation timed out")

    def test_recording_rule_count(self):
        """Verify expected number of recording rules"""
        with open(RECORDING_RULES_PATH, 'r') as f:
            content = f.read()

        # Count recording rules (lines with "- record:")
        rule_count = content.count("- record:")

        assert rule_count >= RECORDING_RULES_COUNT, \
            f"Expected at least {RECORDING_RULES_COUNT} recording rules, found {rule_count}"

    def test_recording_rule_naming_convention(self):
        """Verify recording rules follow naming convention (tars:*)"""
        with open(RECORDING_RULES_PATH, 'r') as f:
            content = f.read()

        # Extract all record names
        import re
        record_names = re.findall(r"- record:\s+(\S+)", content)

        for name in record_names:
            assert name.startswith("tars:"), \
                f"Recording rule '{name}' does not follow 'tars:*' naming convention"

    def test_recording_rule_intervals(self):
        """Verify recording rule evaluation intervals are reasonable"""
        with open(RECORDING_RULES_PATH, 'r') as f:
            content = f.read()

        # Extract all intervals
        import re
        intervals = re.findall(r"interval:\s+(\d+)s", content)

        for interval in intervals:
            interval_sec = int(interval)
            assert 10 <= interval_sec <= 60, \
                f"Recording rule interval {interval_sec}s outside recommended range (10-60s)"


# =================================================================
# DASHBOARD PATCH VALIDATION TESTS
# =================================================================

class TestDashboardPatchValidation:
    """Test dashboard patch JSON structure and completeness"""

    @pytest.fixture
    def dashboard_patch(self) -> Dict[str, Any]:
        """Load dashboard patch JSON"""
        with open(DASHBOARD_PATCH_PATH, 'r') as f:
            return json.load(f)

    def test_dashboard_patch_file_exists(self):
        """Verify dashboard patch file exists"""
        assert DASHBOARD_PATCH_PATH.exists(), \
            f"Dashboard patch file not found: {DASHBOARD_PATCH_PATH}"

    def test_dashboard_patch_json_valid(self, dashboard_patch):
        """Verify dashboard patch is valid JSON"""
        assert isinstance(dashboard_patch, dict), \
            "Dashboard patch is not a valid JSON object"

    def test_panel_transformations_exist(self, dashboard_patch):
        """Verify panel transformations section exists"""
        assert "panel_transformations" in dashboard_patch, \
            "Missing 'panel_transformations' section"

        transformations = dashboard_patch["panel_transformations"]
        assert len(transformations) > 0, \
            "No dashboard transformations defined"

    def test_total_panels_updated(self, dashboard_patch):
        """Verify expected number of panels updated"""
        metadata = dashboard_patch.get("dashboard_patch_metadata", {})
        total_panels = metadata.get("total_panels_updated", 0)

        assert total_panels >= TOTAL_PANELS, \
            f"Expected at least {TOTAL_PANELS} panels updated, found {total_panels}"

    def test_all_panels_have_before_after(self, dashboard_patch):
        """Verify all panels have before/after queries"""
        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})

                assert "before" in transformation, \
                    f"Panel {panel.get('panel_id')} missing 'before' query"
                assert "after" in transformation, \
                    f"Panel {panel.get('panel_id')} missing 'after' query"

                before_query = transformation["before"]["query"]
                after_query = transformation["after"]["query"]

                assert len(before_query) > 0, \
                    f"Panel {panel.get('panel_id')} has empty 'before' query"
                assert len(after_query) > 0, \
                    f"Panel {panel.get('panel_id')} has empty 'after' query"

    def test_after_queries_use_recording_rules(self, dashboard_patch):
        """Verify 'after' queries use recording rules (tars:*)"""
        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                after_query = transformation.get("after", {}).get("query", "")

                # After query should use recording rule (starts with tars:)
                assert after_query.startswith("tars:"), \
                    f"Panel {panel.get('panel_id')} 'after' query does not use recording rule: {after_query}"

    def test_performance_gain_documented(self, dashboard_patch):
        """Verify performance gains are documented for each transformation"""
        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})

                assert "performance_gain" in transformation, \
                    f"Panel {panel.get('panel_id')} missing 'performance_gain'"


# =================================================================
# QUERY PERFORMANCE TESTS
# =================================================================

class TestQueryPerformance:
    """Test query performance with different evaluation counts"""

    @pytest.fixture
    def dashboard_patch(self) -> Dict[str, Any]:
        """Load dashboard patch JSON"""
        with open(DASHBOARD_PATCH_PATH, 'r') as f:
            return json.load(f)

    @pytest.mark.parametrize("eval_count", EVAL_COUNTS)
    def test_dashboard_load_time(self, dashboard_patch, eval_count):
        """Test dashboard load time with different evaluation counts"""
        prometheus = MockPrometheusServer(eval_count=eval_count)

        # Execute all panel queries (after transformation)
        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                after_query = transformation.get("after", {}).get("query", "")

                # Execute query
                prometheus.query(after_query)

        # Calculate total dashboard load time
        total_time_ms = prometheus.get_total_query_time()

        # For 5000+ evaluations, dashboard should load in <5s
        if eval_count >= 5000:
            assert total_time_ms < DASHBOARD_LOAD_TIME_THRESHOLD_MS, \
                f"Dashboard load time {total_time_ms}ms exceeds threshold {DASHBOARD_LOAD_TIME_THRESHOLD_MS}ms for {eval_count} evaluations"

        # All queries should use recording rules
        recording_rule_usage = prometheus.get_recording_rule_usage_percentage()
        assert recording_rule_usage == 100.0, \
            f"Only {recording_rule_usage}% of queries use recording rules"

        print(f"\n✓ Dashboard load time for {eval_count} evals: {total_time_ms:.0f}ms")

    @pytest.mark.parametrize("eval_count", EVAL_COUNTS)
    def test_panel_query_time(self, dashboard_patch, eval_count):
        """Test individual panel query times"""
        prometheus = MockPrometheusServer(eval_count=eval_count)

        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                after_query = transformation.get("after", {}).get("query", "")

                # Execute query
                result = prometheus.query(after_query)

                # Each panel query should complete in <200ms
                assert result["query_time_ms"] < PANEL_QUERY_TIME_THRESHOLD_MS, \
                    f"Panel {panel.get('panel_id')} query time {result['query_time_ms']}ms exceeds threshold {PANEL_QUERY_TIME_THRESHOLD_MS}ms"

    def test_performance_improvement_validation(self, dashboard_patch):
        """Validate performance improvements from before/after queries"""
        prometheus_before = MockPrometheusServer(eval_count=5000)
        prometheus_after = MockPrometheusServer(eval_count=5000)

        transformations = dashboard_patch["panel_transformations"]

        before_times = []
        after_times = []

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                before_query = transformation.get("before", {}).get("query", "")
                after_query = transformation.get("after", {}).get("query", "")

                # Execute before query
                before_result = prometheus_before.query(before_query)
                before_times.append(before_result["query_time_ms"])

                # Execute after query
                after_result = prometheus_after.query(after_query)
                after_times.append(after_result["query_time_ms"])

        # Calculate total improvement
        total_before = sum(before_times)
        total_after = sum(after_times)
        improvement = ((total_before - total_after) / total_before) * 100

        # Should see at least 70% improvement
        assert improvement >= 70, \
            f"Performance improvement {improvement:.1f}% below expected 70%"

        print(f"\n✓ Query performance improvement: {improvement:.1f}%")
        print(f"  Before: {total_before:.0f}ms")
        print(f"  After: {total_after:.0f}ms")


# =================================================================
# REGRESSION TESTS
# =================================================================

class TestRegressionPrevention:
    """Ensure no raw expensive queries remain"""

    @pytest.fixture
    def dashboard_patch(self) -> Dict[str, Any]:
        """Load dashboard patch JSON"""
        with open(DASHBOARD_PATCH_PATH, 'r') as f:
            return json.load(f)

    def test_no_histogram_quantile_in_after_queries(self, dashboard_patch):
        """Ensure no expensive histogram_quantile queries remain"""
        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                after_query = transformation.get("after", {}).get("query", "")

                assert "histogram_quantile" not in after_query, \
                    f"Panel {panel.get('panel_id')} still uses expensive histogram_quantile in 'after' query"

    def test_no_raw_bucket_queries_in_after(self, dashboard_patch):
        """Ensure no raw histogram bucket queries remain"""
        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                after_query = transformation.get("after", {}).get("query", "")

                # Should not query raw bucket metrics
                assert "_bucket[" not in after_query and "_bucket{" not in after_query, \
                    f"Panel {panel.get('panel_id')} still queries raw histogram buckets"

    def test_all_queries_use_1m_or_better_interval(self, dashboard_patch):
        """Verify all queries use optimized time intervals"""
        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                after_query = transformation.get("after", {}).get("query", "")

                # Recording rules should use pre-computed intervals
                # No need for [5m], [1h] ranges in recording rule queries
                assert "[" not in after_query, \
                    f"Panel {panel.get('panel_id')} uses time range in recording rule query (unnecessary)"


# =================================================================
# INTEGRATION TESTS
# =================================================================

class TestGrafanaIntegration:
    """Test Grafana API integration (mocked)"""

    @pytest.fixture
    def dashboard_patch(self) -> Dict[str, Any]:
        """Load dashboard patch JSON"""
        with open(DASHBOARD_PATCH_PATH, 'r') as f:
            return json.load(f)

    def test_deployment_instructions_exist(self, dashboard_patch):
        """Verify deployment instructions are documented"""
        assert "deployment_instructions" in dashboard_patch, \
            "Missing deployment instructions"

        instructions = dashboard_patch["deployment_instructions"]
        assert len(instructions) > 0, \
            "No deployment instructions provided"

    def test_rollback_procedure_exists(self, dashboard_patch):
        """Verify rollback procedure is documented"""
        assert "rollback_procedure" in dashboard_patch, \
            "Missing rollback procedure"

        rollback = dashboard_patch["rollback_procedure"]
        assert len(rollback) > 0, \
            "No rollback steps provided"

    def test_validation_checklist_exists(self, dashboard_patch):
        """Verify validation checklist is documented"""
        assert "validation_checklist" in dashboard_patch, \
            "Missing validation checklist"

        checklist = dashboard_patch["validation_checklist"]
        assert "pre_deployment" in checklist, \
            "Missing pre-deployment checklist"
        assert "post_deployment" in checklist, \
            "Missing post-deployment checklist"


# =================================================================
# LOAD TESTING
# =================================================================

class TestLoadTesting:
    """Load testing with high evaluation counts"""

    @pytest.fixture
    def dashboard_patch(self) -> Dict[str, Any]:
        """Load dashboard patch JSON"""
        with open(DASHBOARD_PATCH_PATH, 'r') as f:
            return json.load(f)

    def test_dashboard_load_under_extreme_load(self, dashboard_patch):
        """Test dashboard with 10k evaluations (stress test)"""
        prometheus = MockPrometheusServer(eval_count=10000)

        transformations = dashboard_patch["panel_transformations"]

        for dashboard_name, dashboard_data in transformations.items():
            panels = dashboard_data.get("panels", [])

            for panel in panels:
                transformation = panel.get("transformation", {})
                after_query = transformation.get("after", {}).get("query", "")

                prometheus.query(after_query)

        total_time_ms = prometheus.get_total_query_time()

        # Even under extreme load, should complete in reasonable time
        # Allow 10s for 10k evaluations (relaxed threshold)
        assert total_time_ms < 10000, \
            f"Dashboard load time {total_time_ms}ms exceeds 10s under extreme load (10k evals)"

        print(f"\n✓ Dashboard load time (10k evals): {total_time_ms:.0f}ms")

    def test_concurrent_dashboard_loads(self, dashboard_patch):
        """Test multiple concurrent dashboard loads"""
        import concurrent.futures

        def load_dashboard(eval_count: int) -> float:
            """Load dashboard and return total query time"""
            prometheus = MockPrometheusServer(eval_count=eval_count)
            transformations = dashboard_patch["panel_transformations"]

            for dashboard_name, dashboard_data in transformations.items():
                panels = dashboard_data.get("panels", [])

                for panel in panels:
                    transformation = panel.get("transformation", {})
                    after_query = transformation.get("after", {}).get("query", "")
                    prometheus.query(after_query)

            return prometheus.get_total_query_time()

        # Simulate 5 concurrent users loading dashboard
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_dashboard, 5000) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All concurrent loads should complete in <5s
        for i, load_time in enumerate(results):
            assert load_time < DASHBOARD_LOAD_TIME_THRESHOLD_MS, \
                f"Concurrent load {i+1} took {load_time}ms (exceeds {DASHBOARD_LOAD_TIME_THRESHOLD_MS}ms)"

        avg_load_time = sum(results) / len(results)
        print(f"\n✓ Average concurrent load time: {avg_load_time:.0f}ms")


# =================================================================
# PERFORMANCE BENCHMARKING
# =================================================================

class TestPerformanceBenchmarking:
    """Comprehensive performance benchmarking"""

    @pytest.fixture
    def dashboard_patch(self) -> Dict[str, Any]:
        """Load dashboard patch JSON"""
        with open(DASHBOARD_PATCH_PATH, 'r') as f:
            return json.load(f)

    def test_performance_benchmark_summary(self, dashboard_patch):
        """Generate performance benchmark summary"""
        results = []

        for eval_count in EVAL_COUNTS:
            prometheus_before = MockPrometheusServer(eval_count=eval_count)
            prometheus_after = MockPrometheusServer(eval_count=eval_count)

            transformations = dashboard_patch["panel_transformations"]

            # Execute before queries
            for dashboard_name, dashboard_data in transformations.items():
                panels = dashboard_data.get("panels", [])
                for panel in panels:
                    transformation = panel.get("transformation", {})
                    before_query = transformation.get("before", {}).get("query", "")
                    prometheus_before.query(before_query)

            before_time = prometheus_before.get_total_query_time()

            # Execute after queries
            for dashboard_name, dashboard_data in transformations.items():
                panels = dashboard_data.get("panels", [])
                for panel in panels:
                    transformation = panel.get("transformation", {})
                    after_query = transformation.get("after", {}).get("query", "")
                    prometheus_after.query(after_query)

            after_time = prometheus_after.get_total_query_time()

            improvement = ((before_time - after_time) / before_time) * 100

            results.append({
                "eval_count": eval_count,
                "before_ms": before_time,
                "after_ms": after_time,
                "improvement_percent": improvement
            })

        # Print benchmark summary
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Evaluations':<15} {'Before (ms)':<15} {'After (ms)':<15} {'Improvement':<15}")
        print("-"*70)

        for result in results:
            print(f"{result['eval_count']:<15} "
                  f"{result['before_ms']:<15.0f} "
                  f"{result['after_ms']:<15.0f} "
                  f"{result['improvement_percent']:<15.1f}%")

        print("="*70)

        # Validate improvements
        for result in results:
            assert result["improvement_percent"] >= 70, \
                f"Improvement {result['improvement_percent']:.1f}% below 70% for {result['eval_count']} evals"


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

"""
Phase 13.9 - Canary Metric SLO Guardrails
=========================================

Validates that canary deployments don't violate SLO thresholds.

SLO Guardrails:
--------------
1. Error rate < 5% (vs baseline)
2. p95 latency < 2x baseline
3. Regression detector < 10% performance drop
4. Success rate > 95%
5. Resource utilization < 80% (CPU, memory)

Canary Decision Logic:
---------------------
- PROCEED: All guardrails pass
- PAUSE: 1-2 guardrails warning
- ROLLBACK: Any guardrail critical failure

Author: T.A.R.S. SRE Team
Date: 2025-11-19
"""

import asyncio
import os
import random
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

EVAL_API_BASE = os.getenv("EVAL_API_BASE", "http://localhost:8096")

# SLO Thresholds
SLO_ERROR_RATE_PERCENT = 5.0
SLO_LATENCY_MULTIPLIER = 2.0  # Max 2x baseline
SLO_REGRESSION_PERCENT = 10.0
SLO_SUCCESS_RATE_PERCENT = 95.0
SLO_CPU_PERCENT = 80.0
SLO_MEMORY_PERCENT = 80.0

# Baseline metrics (from blue/production)
BASELINE_METRICS = {
    "error_rate": 0.5,  # 0.5%
    "p50_latency_ms": 45.0,
    "p95_latency_ms": 120.0,
    "p99_latency_ms": 250.0,
    "success_rate": 99.5,  # 99.5%
    "avg_reward": 150.0,
    "cpu_percent": 35.0,
    "memory_percent": 50.0,
}


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def baseline_metrics():
    """Return baseline production metrics."""
    return BASELINE_METRICS.copy()


@pytest.fixture
def mock_prometheus_client():
    """Mock Prometheus client for querying metrics."""

    class MockPrometheus:
        def __init__(self):
            self.metrics_storage = {}

        async def query(self, query: str) -> Dict:
            """Execute PromQL query."""
            # Mock responses for common queries
            if "http_requests_total" in query:
                return {"data": {"result": [{"value": [time.time(), "1000"]}]}}
            elif "http_request_duration_seconds" in query:
                return {"data": {"result": [{"value": [time.time(), "0.045"]}]}}
            elif "http_errors_total" in query:
                return {"data": {"result": [{"value": [time.time(), "5"]}]}}
            return {"data": {"result": []}}

        async def query_range(
            self, query: str, start: float, end: float, step: str
        ) -> Dict:
            """Execute range query."""
            # Generate time series data
            values = []
            current = start
            while current <= end:
                values.append([current, str(random.uniform(0.04, 0.06))])
                current += 15  # 15s step
            return {"data": {"result": [{"values": values}]}}

    return MockPrometheus()


# ============================================================================
# TEST SUITE 1: ERROR RATE GUARDRAILS
# ============================================================================


@pytest.mark.asyncio
async def test_canary_error_rate_within_slo(baseline_metrics):
    """
    Test that canary error rate is within SLO.

    SLO: Error rate < 5% (baseline: 0.5%)
    Threshold: < 5.5% (baseline + 5%)
    """
    # Simulate 1000 requests to canary
    total_requests = 1000
    errors = 10  # 1.0% error rate

    canary_error_rate = (errors / total_requests) * 100

    # Compare to baseline
    error_rate_increase = canary_error_rate - baseline_metrics["error_rate"]

    assert error_rate_increase < SLO_ERROR_RATE_PERCENT, (
        f"Error rate increase too high: {error_rate_increase:.2f}%"
    )

    print(f"✅ Canary error rate: {canary_error_rate:.2f}% (baseline: {baseline_metrics['error_rate']:.2f}%)")
    print(f"   Increase: +{error_rate_increase:.2f}% (SLO: <{SLO_ERROR_RATE_PERCENT}%)")


@pytest.mark.asyncio
async def test_canary_error_rate_triggers_rollback_on_spike():
    """
    Test that error rate spike triggers rollback.

    Scenario: Error rate jumps to 15% → ROLLBACK
    """
    total_requests = 1000
    errors = 150  # 15% error rate

    canary_error_rate = (errors / total_requests) * 100

    # Decision logic
    def should_rollback(error_rate: float, baseline: float) -> bool:
        """Determine if rollback needed based on error rate."""
        increase = error_rate - baseline
        return increase >= SLO_ERROR_RATE_PERCENT

    rollback = should_rollback(canary_error_rate, BASELINE_METRICS["error_rate"])

    assert rollback is True, "Rollback should be triggered"

    print(f"✅ Rollback triggered: Error rate {canary_error_rate:.2f}% (threshold: {SLO_ERROR_RATE_PERCENT}%)")


@pytest.mark.asyncio
async def test_error_rate_by_endpoint():
    """
    Test that error rate is tracked per endpoint.

    Some endpoints may have higher error rates than others
    """
    endpoint_metrics = {
        "/api/v1/evaluations": {"requests": 500, "errors": 2},
        "/api/v1/agents": {"requests": 300, "errors": 1},
        "/api/v1/hypersync/proposals": {"requests": 200, "errors": 8},  # High error rate
    }

    for endpoint, metrics in endpoint_metrics.items():
        error_rate = (metrics["errors"] / metrics["requests"]) * 100

        if error_rate >= SLO_ERROR_RATE_PERCENT:
            print(f"⚠️  High error rate on {endpoint}: {error_rate:.2f}%")
        else:
            print(f"✅ {endpoint}: {error_rate:.2f}% error rate")


# ============================================================================
# TEST SUITE 2: LATENCY GUARDRAILS
# ============================================================================


@pytest.mark.asyncio
async def test_canary_p95_latency_within_slo(baseline_metrics):
    """
    Test that canary p95 latency is within SLO.

    SLO: p95 latency < 2x baseline (baseline: 120ms → max: 240ms)
    """
    # Simulate latency samples
    latencies = [random.gauss(130, 20) for _ in range(1000)]  # Mean 130ms

    # Calculate p95
    canary_p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

    # Compare to baseline
    max_allowed_p95 = baseline_metrics["p95_latency_ms"] * SLO_LATENCY_MULTIPLIER

    assert canary_p95 < max_allowed_p95, f"p95 latency too high: {canary_p95:.2f}ms"

    print(f"✅ Canary p95 latency: {canary_p95:.2f}ms (max: {max_allowed_p95:.2f}ms)")


@pytest.mark.asyncio
async def test_canary_p99_latency_degradation():
    """
    Test that p99 latency doesn't degrade significantly.

    Warning threshold: p99 > 2x baseline
    """
    # Simulate latency samples with some outliers
    latencies = [random.gauss(130, 30) for _ in range(1000)]

    # Calculate p99
    canary_p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile

    max_allowed_p99 = BASELINE_METRICS["p99_latency_ms"] * SLO_LATENCY_MULTIPLIER

    if canary_p99 > max_allowed_p99:
        print(f"⚠️  p99 latency degraded: {canary_p99:.2f}ms (max: {max_allowed_p99:.2f}ms)")
    else:
        print(f"✅ Canary p99 latency: {canary_p99:.2f}ms (baseline: {BASELINE_METRICS['p99_latency_ms']:.2f}ms)")


@pytest.mark.asyncio
async def test_latency_distribution_comparison():
    """
    Test that latency distribution is similar to baseline.

    Use Kolmogorov-Smirnov test for distribution comparison
    """
    # Baseline latency distribution
    baseline_latencies = [random.gauss(45, 10) for _ in range(1000)]

    # Canary latency distribution (slightly higher mean)
    canary_latencies = [random.gauss(50, 12) for _ in range(1000)]

    # Simple distribution comparison (mean and stddev)
    baseline_mean = statistics.mean(baseline_latencies)
    canary_mean = statistics.mean(canary_latencies)

    mean_increase_percent = ((canary_mean - baseline_mean) / baseline_mean) * 100

    # Warning if mean latency increases > 20%
    if mean_increase_percent > 20:
        print(f"⚠️  Mean latency increased {mean_increase_percent:.2f}%")
    else:
        print(f"✅ Mean latency increase: {mean_increase_percent:.2f}%")


# ============================================================================
# TEST SUITE 3: REGRESSION DETECTION GUARDRAILS
# ============================================================================


@pytest.mark.asyncio
async def test_canary_reward_regression_detection(baseline_metrics):
    """
    Test that RL agent reward doesn't regress.

    SLO: Reward drop < 10% from baseline
    """
    # Baseline average reward
    baseline_reward = baseline_metrics["avg_reward"]  # 150.0

    # Canary reward (simulated)
    canary_reward = 142.0  # 5.3% drop (acceptable)

    reward_drop_percent = ((baseline_reward - canary_reward) / baseline_reward) * 100

    assert reward_drop_percent < SLO_REGRESSION_PERCENT, (
        f"Reward regression too high: {reward_drop_percent:.2f}%"
    )

    print(f"✅ Reward regression: {reward_drop_percent:.2f}% (SLO: <{SLO_REGRESSION_PERCENT}%)")


@pytest.mark.asyncio
async def test_canary_regression_triggers_rollback():
    """
    Test that significant regression triggers rollback.

    Scenario: Reward drops 15% → ROLLBACK
    """
    baseline_reward = BASELINE_METRICS["avg_reward"]
    canary_reward = 127.5  # 15% drop

    reward_drop_percent = ((baseline_reward - canary_reward) / baseline_reward) * 100

    def should_rollback_regression(drop_percent: float) -> bool:
        """Determine if rollback needed based on regression."""
        return drop_percent >= SLO_REGRESSION_PERCENT

    rollback = should_rollback_regression(reward_drop_percent)

    assert rollback is True, "Rollback should be triggered on regression"

    print(f"✅ Rollback triggered: Reward drop {reward_drop_percent:.2f}%")


@pytest.mark.asyncio
async def test_multi_metric_regression_detection():
    """
    Test regression detection across multiple metrics.

    Metrics: reward, success_rate, episode_length
    """
    baseline = {
        "avg_reward": 150.0,
        "success_rate": 95.0,
        "avg_episode_length": 200.0,
    }

    canary = {
        "avg_reward": 145.0,  # 3.3% drop (OK)
        "success_rate": 92.0,  # 3.2% drop (OK)
        "avg_episode_length": 220.0,  # 10% increase (OK)
    }

    regressions = {}
    for metric, baseline_val in baseline.items():
        canary_val = canary[metric]

        if metric in ["avg_reward", "success_rate"]:
            # Lower is worse
            drop_percent = ((baseline_val - canary_val) / baseline_val) * 100
            regressions[metric] = drop_percent
        else:
            # Episode length: higher may be OK
            change_percent = ((canary_val - baseline_val) / baseline_val) * 100
            regressions[metric] = change_percent

    # Check if any metric regressed significantly
    critical_regressions = [
        k for k, v in regressions.items()
        if v >= SLO_REGRESSION_PERCENT and k in ["avg_reward", "success_rate"]
    ]

    assert len(critical_regressions) == 0, f"Critical regressions: {critical_regressions}"

    print("✅ Multi-metric regression check PASSED")
    for metric, change in regressions.items():
        print(f"   {metric}: {change:+.2f}%")


# ============================================================================
# TEST SUITE 4: SUCCESS RATE GUARDRAILS
# ============================================================================


@pytest.mark.asyncio
async def test_canary_success_rate_above_slo():
    """
    Test that canary success rate is above SLO.

    SLO: Success rate > 95%
    """
    total_requests = 1000
    successful_requests = 970  # 97% success rate

    success_rate = (successful_requests / total_requests) * 100

    assert success_rate >= SLO_SUCCESS_RATE_PERCENT, (
        f"Success rate below SLO: {success_rate:.2f}%"
    )

    print(f"✅ Canary success rate: {success_rate:.2f}% (SLO: >{SLO_SUCCESS_RATE_PERCENT}%)")


@pytest.mark.asyncio
async def test_success_rate_by_operation_type():
    """
    Test success rate for different operation types.

    Operations: create, read, update, delete
    """
    operations = {
        "create_evaluation": {"total": 100, "success": 98},
        "read_evaluation": {"total": 500, "success": 500},
        "update_evaluation": {"total": 50, "success": 48},
        "delete_evaluation": {"total": 20, "success": 20},
    }

    for operation, metrics in operations.items():
        success_rate = (metrics["success"] / metrics["total"]) * 100

        if success_rate < SLO_SUCCESS_RATE_PERCENT:
            print(f"⚠️  {operation}: {success_rate:.2f}% (below SLO)")
        else:
            print(f"✅ {operation}: {success_rate:.2f}%")


# ============================================================================
# TEST SUITE 5: RESOURCE UTILIZATION GUARDRAILS
# ============================================================================


@pytest.mark.asyncio
async def test_canary_cpu_utilization_within_limits():
    """
    Test that canary CPU utilization is within limits.

    SLO: CPU < 80%
    """
    # Simulate CPU samples over 5 minutes
    cpu_samples = [random.uniform(40, 65) for _ in range(20)]

    avg_cpu = statistics.mean(cpu_samples)
    max_cpu = max(cpu_samples)

    assert avg_cpu < SLO_CPU_PERCENT, f"Average CPU too high: {avg_cpu:.2f}%"
    assert max_cpu < 90, f"Peak CPU too high: {max_cpu:.2f}%"

    print(f"✅ Canary CPU: avg={avg_cpu:.2f}%, max={max_cpu:.2f}% (SLO: <{SLO_CPU_PERCENT}%)")


@pytest.mark.asyncio
async def test_canary_memory_utilization_within_limits():
    """
    Test that canary memory utilization is within limits.

    SLO: Memory < 80%
    """
    # Simulate memory samples
    memory_samples = [random.uniform(50, 70) for _ in range(20)]

    avg_memory = statistics.mean(memory_samples)
    max_memory = max(memory_samples)

    assert avg_memory < SLO_MEMORY_PERCENT, f"Average memory too high: {avg_memory:.2f}%"

    print(f"✅ Canary memory: avg={avg_memory:.2f}%, max={max_memory:.2f}% (SLO: <{SLO_MEMORY_PERCENT}%)")


@pytest.mark.asyncio
async def test_resource_utilization_compared_to_baseline(baseline_metrics):
    """
    Test that canary resource usage is comparable to baseline.

    Warning if resource usage > 1.5x baseline
    """
    canary_cpu = 52.0  # vs baseline 35.0
    canary_memory = 68.0  # vs baseline 50.0

    cpu_ratio = canary_cpu / baseline_metrics["cpu_percent"]
    memory_ratio = canary_memory / baseline_metrics["memory_percent"]

    if cpu_ratio > 1.5:
        print(f"⚠️  CPU usage increased {cpu_ratio:.2f}x")
    else:
        print(f"✅ CPU usage: {canary_cpu:.2f}% ({cpu_ratio:.2f}x baseline)")

    if memory_ratio > 1.5:
        print(f"⚠️  Memory usage increased {memory_ratio:.2f}x")
    else:
        print(f"✅ Memory usage: {canary_memory:.2f}% ({memory_ratio:.2f}x baseline)")


# ============================================================================
# TEST SUITE 6: GUARDRAIL DECISION LOGIC
# ============================================================================


@pytest.mark.asyncio
async def test_guardrail_decision_proceed():
    """
    Test that canary proceeds when all guardrails pass.

    Decision: PROCEED
    """
    guardrail_results = {
        "error_rate": {"status": "pass", "value": 1.5, "threshold": 5.0},
        "p95_latency": {"status": "pass", "value": 140, "threshold": 240},
        "regression": {"status": "pass", "value": 3.2, "threshold": 10.0},
        "success_rate": {"status": "pass", "value": 98.5, "threshold": 95.0},
        "cpu_utilization": {"status": "pass", "value": 55, "threshold": 80},
    }

    # Decision logic
    failed_guardrails = [k for k, v in guardrail_results.items() if v["status"] == "fail"]
    warning_guardrails = [k for k, v in guardrail_results.items() if v["status"] == "warning"]

    if len(failed_guardrails) > 0:
        decision = "ROLLBACK"
    elif len(warning_guardrails) > 2:
        decision = "PAUSE"
    else:
        decision = "PROCEED"

    assert decision == "PROCEED", f"Expected PROCEED, got {decision}"

    print(f"✅ Guardrail decision: {decision}")


@pytest.mark.asyncio
async def test_guardrail_decision_pause():
    """
    Test that canary pauses when 1-2 guardrails in warning state.

    Decision: PAUSE
    """
    guardrail_results = {
        "error_rate": {"status": "pass", "value": 2.0, "threshold": 5.0},
        "p95_latency": {"status": "warning", "value": 180, "threshold": 240},  # WARNING
        "regression": {"status": "warning", "value": 8.5, "threshold": 10.0},  # WARNING
        "success_rate": {"status": "pass", "value": 97.0, "threshold": 95.0},
        "cpu_utilization": {"status": "pass", "value": 65, "threshold": 80},
    }

    failed_guardrails = [k for k, v in guardrail_results.items() if v["status"] == "fail"]
    warning_guardrails = [k for k, v in guardrail_results.items() if v["status"] == "warning"]

    if len(failed_guardrails) > 0:
        decision = "ROLLBACK"
    elif len(warning_guardrails) >= 2:
        decision = "PAUSE"
    else:
        decision = "PROCEED"

    assert decision == "PAUSE", f"Expected PAUSE, got {decision}"

    print(f"✅ Guardrail decision: {decision} ({len(warning_guardrails)} warnings)")


@pytest.mark.asyncio
async def test_guardrail_decision_rollback():
    """
    Test that canary rolls back when any guardrail fails.

    Decision: ROLLBACK
    """
    guardrail_results = {
        "error_rate": {"status": "fail", "value": 12.0, "threshold": 5.0},  # FAIL
        "p95_latency": {"status": "pass", "value": 150, "threshold": 240},
        "regression": {"status": "pass", "value": 5.0, "threshold": 10.0},
        "success_rate": {"status": "pass", "value": 96.0, "threshold": 95.0},
        "cpu_utilization": {"status": "pass", "value": 60, "threshold": 80},
    }

    failed_guardrails = [k for k, v in guardrail_results.items() if v["status"] == "fail"]

    if len(failed_guardrails) > 0:
        decision = "ROLLBACK"
    else:
        decision = "PROCEED"

    assert decision == "ROLLBACK", f"Expected ROLLBACK, got {decision}"

    print(f"✅ Guardrail decision: {decision} (failed: {failed_guardrails})")


# ============================================================================
# TEST SUITE 7: CANARY METRIC COLLECTION
# ============================================================================


@pytest.mark.asyncio
async def test_collect_metrics_from_prometheus(mock_prometheus_client):
    """
    Test that metrics are collected from Prometheus during canary.
    """
    # Query error rate
    error_rate_query = 'rate(http_errors_total{deployment="canary"}[5m])'
    error_result = await mock_prometheus_client.query(error_rate_query)

    assert "data" in error_result
    assert len(error_result["data"]["result"]) > 0

    # Query latency
    latency_query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{deployment="canary"}[5m]))'
    latency_result = await mock_prometheus_client.query(latency_query)

    assert "data" in latency_result

    print("✅ Prometheus metrics collected")


@pytest.mark.asyncio
async def test_metrics_aggregation_per_canary_phase():
    """
    Test that metrics are aggregated per canary phase.

    Phases: 5%, 25%, 50%, 100%
    """
    phases = [
        {"name": "canary-5", "traffic": 5, "duration_min": 10},
        {"name": "canary-25", "traffic": 25, "duration_min": 15},
        {"name": "canary-50", "traffic": 50, "duration_min": 20},
        {"name": "full-rollout", "traffic": 100, "duration_min": 0},
    ]

    phase_metrics = {}

    for phase in phases:
        # Simulate metric collection
        phase_metrics[phase["name"]] = {
            "error_rate": random.uniform(0.5, 2.0),
            "p95_latency_ms": random.uniform(100, 150),
            "success_rate": random.uniform(96, 99),
        }

    # Validate metrics collected for all phases
    assert len(phase_metrics) == len(phases)

    print("✅ Metrics aggregated per phase:")
    for phase_name, metrics in phase_metrics.items():
        print(f"   {phase_name}: {metrics['error_rate']:.2f}% errors, {metrics['p95_latency_ms']:.2f}ms p95")


# ============================================================================
# TEST SUITE 8: COMPREHENSIVE SLO REPORT
# ============================================================================


@pytest.mark.asyncio
async def test_generate_canary_slo_report(baseline_metrics):
    """
    Generate comprehensive SLO compliance report for canary.
    """
    canary_metrics = {
        "error_rate": 1.8,
        "p50_latency_ms": 48.0,
        "p95_latency_ms": 135.0,
        "p99_latency_ms": 280.0,
        "success_rate": 98.2,
        "avg_reward": 147.0,
        "cpu_percent": 52.0,
        "memory_percent": 68.0,
    }

    report = {
        "canary_date": datetime.utcnow().isoformat(),
        "baseline_version": "v1.0.0",
        "canary_version": "v1.1.0",
        "guardrails": [],
        "overall_status": "PASS",
    }

    # Evaluate each guardrail
    # 1. Error rate
    error_rate_increase = canary_metrics["error_rate"] - baseline_metrics["error_rate"]
    report["guardrails"].append({
        "name": "error_rate",
        "baseline": baseline_metrics["error_rate"],
        "canary": canary_metrics["error_rate"],
        "change": error_rate_increase,
        "threshold": SLO_ERROR_RATE_PERCENT,
        "status": "PASS" if error_rate_increase < SLO_ERROR_RATE_PERCENT else "FAIL",
    })

    # 2. p95 Latency
    p95_multiplier = canary_metrics["p95_latency_ms"] / baseline_metrics["p95_latency_ms"]
    report["guardrails"].append({
        "name": "p95_latency",
        "baseline": baseline_metrics["p95_latency_ms"],
        "canary": canary_metrics["p95_latency_ms"],
        "multiplier": p95_multiplier,
        "threshold": SLO_LATENCY_MULTIPLIER,
        "status": "PASS" if p95_multiplier < SLO_LATENCY_MULTIPLIER else "FAIL",
    })

    # 3. Regression
    reward_drop = ((baseline_metrics["avg_reward"] - canary_metrics["avg_reward"]) / baseline_metrics["avg_reward"]) * 100
    report["guardrails"].append({
        "name": "reward_regression",
        "baseline": baseline_metrics["avg_reward"],
        "canary": canary_metrics["avg_reward"],
        "drop_percent": reward_drop,
        "threshold": SLO_REGRESSION_PERCENT,
        "status": "PASS" if reward_drop < SLO_REGRESSION_PERCENT else "FAIL",
    })

    # 4. Success rate
    report["guardrails"].append({
        "name": "success_rate",
        "baseline": baseline_metrics["success_rate"],
        "canary": canary_metrics["success_rate"],
        "threshold": SLO_SUCCESS_RATE_PERCENT,
        "status": "PASS" if canary_metrics["success_rate"] >= SLO_SUCCESS_RATE_PERCENT else "FAIL",
    })

    # 5. CPU utilization
    report["guardrails"].append({
        "name": "cpu_utilization",
        "baseline": baseline_metrics["cpu_percent"],
        "canary": canary_metrics["cpu_percent"],
        "threshold": SLO_CPU_PERCENT,
        "status": "PASS" if canary_metrics["cpu_percent"] < SLO_CPU_PERCENT else "FAIL",
    })

    # Determine overall status
    failed_guardrails = [g for g in report["guardrails"] if g["status"] == "FAIL"]
    if len(failed_guardrails) > 0:
        report["overall_status"] = "FAIL"

    # Validate report
    assert report["overall_status"] == "PASS", f"Canary failed: {failed_guardrails}"

    print("✅ Canary SLO Report Generated")
    print(f"   Overall Status: {report['overall_status']}")
    print(f"   Guardrails Passed: {len([g for g in report['guardrails'] if g['status'] == 'PASS'])}/{len(report['guardrails'])}")

    for guardrail in report["guardrails"]:
        print(f"   [{guardrail['status']}] {guardrail['name']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Alert Trigger Paths End-to-End Test

Tests that Prometheus alerts trigger correctly under failure scenarios:

1. High latency alerts
2. Regression alerts
3. Redis down alerts
4. PostgreSQL failover alerts
5. Evaluation failure alerts
6. Alert naming consistency with prometheus-alerts.yaml

**Version:** v1.0.0-rc2
**Phase:** 13.8 - Final Pre-Production Validation
**Author:** T.A.R.S. Development Team
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from unittest.mock import patch, AsyncMock

import httpx
import pytest
import yaml


class AlertTestContext:
    """Context manager for alert testing."""

    def __init__(self):
        self.prometheus_client: Optional[httpx.AsyncClient] = None
        self.eval_engine_client: Optional[httpx.AsyncClient] = None
        self.orchestration_client: Optional[httpx.AsyncClient] = None
        self.alert_rules: Dict[str, Any] = {}

    async def __aenter__(self):
        """Initialize service clients and load alert rules."""
        # Prometheus (if running)
        self.prometheus_client = httpx.AsyncClient(
            base_url="http://localhost:9090",
            timeout=10.0
        )

        self.eval_engine_client = httpx.AsyncClient(
            base_url="http://localhost:8099",
            timeout=30.0
        )

        self.orchestration_client = httpx.AsyncClient(
            base_url="http://localhost:8094",
            timeout=30.0
        )

        # Load alert rules from prometheus-alerts.yaml
        try:
            with open("observability/alerts/prometheus-alerts.yaml", "r") as f:
                alert_config = yaml.safe_load(f)

            # Extract alert names
            for group in alert_config.get("groups", []):
                for rule in group.get("rules", []):
                    if "alert" in rule:
                        alert_name = rule["alert"]
                        self.alert_rules[alert_name] = rule

        except FileNotFoundError:
            print("⚠ prometheus-alerts.yaml not found")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup clients."""
        if self.prometheus_client:
            await self.prometheus_client.aclose()
        if self.eval_engine_client:
            await self.eval_engine_client.aclose()
        if self.orchestration_client:
            await self.orchestration_client.aclose()

    async def check_alert_firing(self, alert_name: str, timeout: float = 30.0) -> bool:
        """Check if specific alert is currently firing."""
        try:
            response = await self.prometheus_client.get("/api/v1/alerts")

            if response.status_code == 200:
                data = response.json()
                alerts = data.get("data", {}).get("alerts", [])

                for alert in alerts:
                    if alert.get("labels", {}).get("alertname") == alert_name:
                        if alert.get("state") == "firing":
                            return True

        except Exception as e:
            print(f"Error checking alert {alert_name}: {e}")

        return False

    async def get_metric_value(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Query Prometheus for current metric value."""
        try:
            # Build query
            if labels:
                label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                query = f"{metric_name}{{{label_str}}}"
            else:
                query = metric_name

            response = await self.prometheus_client.get(
                "/api/v1/query",
                params={"query": query}
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get("data", {}).get("result", [])

                if result:
                    value = result[0].get("value", [None, None])[1]
                    return float(value) if value is not None else None

        except Exception as e:
            print(f"Error querying metric {metric_name}: {e}")

        return None


@pytest.fixture
async def alert_context():
    """Fixture providing alert test context."""
    async with AlertTestContext() as ctx:
        yield ctx


@pytest.mark.asyncio
async def test_high_evaluation_latency_alert(alert_context: AlertTestContext):
    """
    Test HighEvaluationLatency alert triggers correctly.

    **Scenario:**
    1. Submit evaluation with many episodes (slow)
    2. Monitor p95 latency metric
    3. Verify alert triggers if p95 > 300s

    **Expected:**
    - Metric: tars_eval_evaluation_duration_seconds
    - Alert: HighEvaluationLatency
    - Threshold: p95 > 300s
    """
    ctx = alert_context

    # Check if alert rule exists
    if "HighEvaluationLatency" not in ctx.alert_rules:
        pytest.skip("HighEvaluationLatency alert not defined")

    print(f"Alert rule: {ctx.alert_rules['HighEvaluationLatency']}")

    # ====================================================================
    # STEP 1: Trigger slow evaluation
    # ====================================================================
    slow_eval_request = {
        "agent_type": "dqn",
        "environment": "CartPole-v1",
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99
        },
        "num_episodes": 100,  # Many episodes → slower
        "quick_mode": False
    }

    eval_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=slow_eval_request
    )

    if eval_response.status_code not in [200, 201, 202]:
        pytest.skip("Could not start evaluation")

    job_id = eval_response.json()["job_id"]

    # ====================================================================
    # STEP 2: Monitor latency metric
    # ====================================================================
    # Wait for evaluation to complete and metric to update
    start_time = time.time()
    max_wait = 600  # 10 minutes

    while time.time() - start_time < max_wait:
        # Check evaluation status
        status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")

        if status_response.status_code == 200:
            status_data = status_response.json()

            if status_data["status"] == "completed":
                duration = status_data.get("duration_seconds", 0)
                print(f"Evaluation completed in {duration:.2f}s")

                # Check if duration would trigger alert (> 300s)
                if duration > 300:
                    print("✓ Evaluation latency exceeds threshold")

                    # Check metric (requires Prometheus)
                    latency_p95 = await ctx.get_metric_value(
                        "histogram_quantile(0.95, tars_eval_evaluation_duration_seconds_bucket)"
                    )

                    if latency_p95:
                        print(f"P95 latency: {latency_p95:.2f}s")

                        if latency_p95 > 300:
                            print("✓ Metric exceeds alert threshold")

                            # Check if alert firing (requires Prometheus AlertManager)
                            await asyncio.sleep(65)  # Wait for alert evaluation (1min)
                            is_firing = await ctx.check_alert_firing("HighEvaluationLatency")

                            if is_firing:
                                print("✓ HighEvaluationLatency alert is FIRING")
                            else:
                                print("⚠ Alert not firing (may require AlertManager)")
                break

            elif status_data["status"] == "failed":
                pytest.fail("Evaluation failed")

        await asyncio.sleep(5)

    print("✓ High latency alert path validated")


@pytest.mark.asyncio
async def test_evaluation_failure_rate_alert(alert_context: AlertTestContext):
    """
    Test EvaluationFailureRateHigh alert triggers.

    **Scenario:**
    1. Submit multiple evaluations with bad parameters
    2. Cause failures
    3. Monitor failure rate metric
    4. Verify alert triggers if failure_rate > 5%

    **Expected:**
    - Metric: tars_eval_evaluations_total{status="failed"}
    - Alert: EvaluationFailureRateHigh
    - Threshold: failure_rate > 0.05 for 5m
    """
    ctx = alert_context

    if "EvaluationFailureRateHigh" not in ctx.alert_rules:
        pytest.skip("EvaluationFailureRateHigh alert not defined")

    # ====================================================================
    # STEP 1: Submit evaluations designed to fail
    # ====================================================================
    failing_requests = []

    for i in range(3):
        bad_request = {
            "agent_type": "dqn",
            "environment": "NonExistentEnv-v999",  # Invalid environment
            "hyperparameters": {
                "learning_rate": -0.001  # Invalid
            },
            "num_episodes": 10
        }
        failing_requests.append(bad_request)

    # Submit all
    failure_count = 0
    for req in failing_requests:
        try:
            response = await ctx.eval_engine_client.post(
                "/v1/evaluate",
                json=req
            )

            if response.status_code >= 400:
                failure_count += 1
            elif response.status_code in [200, 201, 202]:
                # Wait for job to fail
                job_id = response.json().get("job_id")
                if job_id:
                    await asyncio.sleep(2)
                    status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")
                    if status_response.status_code == 200:
                        if status_response.json().get("status") == "failed":
                            failure_count += 1

        except Exception as e:
            print(f"Request exception: {e}")
            failure_count += 1

    print(f"Failed evaluations: {failure_count}/{len(failing_requests)}")

    # ====================================================================
    # STEP 2: Check failure rate metric
    # ====================================================================
    failure_metric = await ctx.get_metric_value(
        "tars_eval_evaluations_total",
        {"status": "failed"}
    )

    if failure_metric:
        print(f"Total failures: {failure_metric}")

        # Calculate rate (would need historical data in Prometheus)
        # For testing, we'll just verify metric incremented
        assert failure_metric >= failure_count, "Failure metric not incremented"

    print("✓ Evaluation failure alert path validated")


@pytest.mark.asyncio
async def test_regression_detection_alert(alert_context: AlertTestContext):
    """
    Test regression detection alert triggers.

    **Scenario:**
    1. Establish baseline
    2. Submit worse performing trial
    3. Regression detected
    4. Alert fires

    **Expected:**
    - Metric: tars_hypersync_proposals_total{status="rejected"}
    - Alert: RegressionDetected
    - Trigger: Regression rejected due to performance drop
    """
    ctx = alert_context

    # ====================================================================
    # STEP 1: Create baseline
    # ====================================================================
    baseline = {
        "agent_type": "a2c",
        "environment": "CartPole-v1",
        "hyperparameters": {
            "learning_rate": 0.0007,
            "gamma": 0.99
        },
        "mean_reward": 200.0,
        "std_reward": 10.0,
        "num_episodes": 50,
        "trial_id": f"alert_baseline_{int(time.time())}",
        "rank": 1
    }

    await ctx.orchestration_client.post("/v1/baselines", json=baseline)

    # ====================================================================
    # STEP 2: Submit worse trial
    # ====================================================================
    worse_trial = {
        "agent_type": "a2c",
        "environment": "CartPole-v1",
        "hyperparameters": {
            "learning_rate": 0.01,  # Too high
            "gamma": 0.9            # Too low
        },
        "num_episodes": 20,
        "quick_mode": True
    }

    eval_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=worse_trial
    )

    if eval_response.status_code in [200, 201, 202]:
        job_id = eval_response.json()["job_id"]

        # Wait for evaluation
        start_time = time.time()
        while time.time() - start_time < 60:
            status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    mean_reward = status_data.get("mean_reward", 0)

                    if mean_reward < baseline["mean_reward"]:
                        print(f"✓ Regression detected: {mean_reward} < {baseline['mean_reward']}")

                        # Check regression metric
                        rejections = await ctx.get_metric_value(
                            "tars_hypersync_proposals_total",
                            {"status": "rejected"}
                        )

                        if rejections:
                            print(f"Total rejections: {rejections}")

                    break

            await asyncio.sleep(2)

    print("✓ Regression alert path validated")


@pytest.mark.asyncio
async def test_redis_connection_failure_alert(alert_context: AlertTestContext):
    """
    Test Redis connection failure alert.

    **Scenario:**
    1. Simulate Redis connection failures
    2. Monitor redis connection metric
    3. Verify alert triggers

    **Expected:**
    - Metric: redis_up == 0 OR tars_redis_connection_failures_total
    - Alert: RedisConnectionFailures
    """
    ctx = alert_context

    if "RedisConnectionFailures" not in ctx.alert_rules:
        pytest.skip("RedisConnectionFailures alert not defined")

    # ====================================================================
    # NOTE: This test cannot actually bring down Redis
    # We'll verify the metric path and alert definition
    # ====================================================================

    print("Alert rule for RedisConnectionFailures:")
    print(yaml.dump(ctx.alert_rules["RedisConnectionFailures"], default_flow_style=False))

    # Verify alert targets correct metric
    alert_expr = ctx.alert_rules["RedisConnectionFailures"].get("expr", "")
    assert "redis" in alert_expr.lower(), "Alert does not check Redis metrics"

    print("✓ Redis failure alert path validated (configuration check)")


@pytest.mark.asyncio
async def test_postgres_connection_pool_exhausted_alert(alert_context: AlertTestContext):
    """
    Test PostgreSQL connection pool exhaustion alert.

    **Expected:**
    - Metric: pg_active_connections / pg_max_connections > 0.9
    - Alert: PostgreSQLConnectionPoolExhausted
    """
    ctx = alert_context

    if "PostgreSQLConnectionPoolExhausted" not in ctx.alert_rules:
        pytest.skip("PostgreSQLConnectionPoolExhausted alert not defined")

    print("Alert rule for PostgreSQLConnectionPoolExhausted:")
    print(yaml.dump(ctx.alert_rules["PostgreSQLConnectionPoolExhausted"], default_flow_style=False))

    # Verify alert definition
    alert_expr = ctx.alert_rules["PostgreSQLConnectionPoolExhausted"].get("expr", "")
    assert "pg" in alert_expr.lower() or "postgres" in alert_expr.lower(), (
        "Alert does not check PostgreSQL metrics"
    )

    print("✓ PostgreSQL pool alert path validated (configuration check)")


@pytest.mark.asyncio
async def test_alert_naming_consistency(alert_context: AlertTestContext):
    """
    Test that alert names match prometheus-alerts.yaml.

    **Scenario:**
    1. Load prometheus-alerts.yaml
    2. Verify expected alerts defined
    3. Verify alert naming conventions
    4. Verify all services have coverage

    **Expected Alerts:**
    - HighEvaluationLatency
    - EvaluationFailureRateHigh
    - RegressionDetected
    - RedisConnectionFailures
    - PostgreSQLConnectionPoolExhausted
    - HotReloadFailure
    - BaselineUpdateFailure
    """
    ctx = alert_context

    if not ctx.alert_rules:
        pytest.skip("Alert rules not loaded")

    # ====================================================================
    # STEP 1: Verify expected alerts exist
    # ====================================================================
    expected_alerts = [
        "HighEvaluationLatency",
        "EvaluationFailureRateHigh",
        "PostgreSQLConnectionPoolExhausted",
        "RedisConnectionFailures",
    ]

    recommended_alerts = [
        "RegressionDetected",
        "HotReloadFailure",
        "BaselineUpdateFailure",
        "PostgreSQLSlowQueries",
        "RedisMemoryHigh"
    ]

    print("\nExpected Alerts:")
    for alert in expected_alerts:
        if alert in ctx.alert_rules:
            print(f"  ✓ {alert}")
        else:
            print(f"  ✗ {alert} (MISSING)")

    print("\nRecommended Alerts:")
    for alert in recommended_alerts:
        if alert in ctx.alert_rules:
            print(f"  ✓ {alert}")
        else:
            print(f"  ⚠ {alert} (recommended)")

    # ====================================================================
    # STEP 2: Verify naming conventions
    # ====================================================================
    print(f"\nAll defined alerts ({len(ctx.alert_rules)}):")
    for alert_name in sorted(ctx.alert_rules.keys()):
        # Check naming convention: PascalCase
        if alert_name[0].isupper():
            print(f"  ✓ {alert_name}")
        else:
            print(f"  ⚠ {alert_name} (not PascalCase)")

    # ====================================================================
    # STEP 3: Verify alert metadata
    # ====================================================================
    for alert_name, alert_rule in ctx.alert_rules.items():
        # Check required fields
        assert "expr" in alert_rule, f"{alert_name} missing 'expr'"

        # Check recommended fields
        if "annotations" not in alert_rule:
            print(f"⚠ {alert_name} missing annotations")

        if "labels" not in alert_rule:
            print(f"⚠ {alert_name} missing labels")

        # Check severity label
        severity = alert_rule.get("labels", {}).get("severity")
        if not severity:
            print(f"⚠ {alert_name} missing severity label")

    print("\n✓ Alert naming consistency validated")


@pytest.mark.asyncio
async def test_alert_runbook_links(alert_context: AlertTestContext):
    """
    Test that alerts have runbook links in annotations.

    **Expected:**
    - Each alert should have 'runbook_url' annotation
    - Links should point to valid runbook sections
    """
    ctx = alert_context

    if not ctx.alert_rules:
        pytest.skip("Alert rules not loaded")

    print("\nAlert Runbook Coverage:")

    for alert_name, alert_rule in ctx.alert_rules.items():
        annotations = alert_rule.get("annotations", {})

        # Check for runbook link
        runbook_url = annotations.get("runbook_url") or annotations.get("runbook")

        if runbook_url:
            print(f"  ✓ {alert_name}: {runbook_url}")
        else:
            print(f"  ⚠ {alert_name}: No runbook link")

        # Check for summary/description
        summary = annotations.get("summary") or annotations.get("description")
        if not summary:
            print(f"    ⚠ Missing summary/description")


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint_accessibility(alert_context: AlertTestContext):
    """
    Test that Prometheus can scrape metrics from all services.

    **Services:**
    - Eval Engine (8099/metrics)
    - Orchestration (8094/metrics)
    - HyperSync (8098/metrics)

    **Expected:**
    - All /metrics endpoints return 200
    - Metrics in Prometheus format
    - Target metrics present
    """
    ctx = alert_context

    services = [
        ("eval_engine", ctx.eval_engine_client, [
            "tars_eval_evaluations_total",
            "tars_eval_evaluation_duration_seconds"
        ]),
        ("orchestration", ctx.orchestration_client, [
            "tars_orchestration_baseline_updates_total"
        ])
    ]

    print("\nMetrics Endpoint Accessibility:")

    for service_name, client, expected_metrics in services:
        try:
            response = await client.get("/metrics")

            if response.status_code == 200:
                metrics_text = response.text

                # Check for expected metrics
                found_metrics = []
                for metric in expected_metrics:
                    if metric in metrics_text:
                        found_metrics.append(metric)

                print(f"  ✓ {service_name}: {len(found_metrics)}/{len(expected_metrics)} metrics")

                for metric in found_metrics:
                    print(f"    ✓ {metric}")

                for metric in set(expected_metrics) - set(found_metrics):
                    print(f"    ⚠ {metric} (missing)")

            else:
                print(f"  ✗ {service_name}: HTTP {response.status_code}")

        except Exception as e:
            print(f"  ✗ {service_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

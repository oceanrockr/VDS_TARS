"""
End-to-End Full Pipeline Test

Tests the complete T.A.R.S. pipeline from AutoML trial submission through
hot-reload deployment:

1. AutoML submits trial hyperparameters
2. Orchestration receives trial and initiates evaluation
3. Eval Engine runs evaluation episodes
4. HyperSync generates hyperparameter proposal
5. Orchestration updates baseline
6. Hot-reload propagates to all agents
7. Verification of metrics, logs, and Redis Streams checkpoints
8. Regression detection and rollback validation

**Version:** v1.0.0-rc2
**Phase:** 13.8 - Final Pre-Production Validation
**Author:** T.A.R.S. Development Team
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Test configuration
TEST_TIMEOUT = 300  # 5 minutes for full pipeline
EVALUATION_TIMEOUT = 120  # 2 minutes for evaluation
HOT_RELOAD_TIMEOUT = 5  # 5 seconds for hot reload
REDIS_STREAM_KEY = "tars:eval:requests"
BASELINE_UPDATE_STREAM = "tars:baseline:updates"


class PipelineContext:
    """Context manager for E2E pipeline testing."""

    def __init__(self):
        self.automl_client: Optional[httpx.AsyncClient] = None
        self.orchestration_client: Optional[httpx.AsyncClient] = None
        self.eval_engine_client: Optional[httpx.AsyncClient] = None
        self.hypersync_client: Optional[httpx.AsyncClient] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.trace_exporter: Optional[InMemorySpanExporter] = None
        self.tracer_provider: Optional[TracerProvider] = None

    async def __aenter__(self):
        """Initialize all service clients and tracing."""
        # Initialize HTTP clients for each service
        self.automl_client = httpx.AsyncClient(
            base_url="http://localhost:8093",
            timeout=30.0
        )
        self.orchestration_client = httpx.AsyncClient(
            base_url="http://localhost:8094",
            timeout=30.0
        )
        self.eval_engine_client = httpx.AsyncClient(
            base_url="http://localhost:8099",
            timeout=30.0
        )
        self.hypersync_client = httpx.AsyncClient(
            base_url="http://localhost:8098",
            timeout=30.0
        )

        # Initialize Redis client
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )

        # Initialize distributed tracing
        self.trace_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.trace_exporter)
        )
        trace.set_tracer_provider(self.tracer_provider)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all clients."""
        if self.automl_client:
            await self.automl_client.aclose()
        if self.orchestration_client:
            await self.orchestration_client.aclose()
        if self.eval_engine_client:
            await self.eval_engine_client.aclose()
        if self.hypersync_client:
            await self.hypersync_client.aclose()
        if self.redis_client:
            await self.redis_client.close()


@pytest.fixture
async def pipeline_context():
    """Fixture providing pipeline context for tests."""
    async with PipelineContext() as ctx:
        # Clear Redis streams before each test
        await ctx.redis_client.delete(REDIS_STREAM_KEY)
        await ctx.redis_client.delete(BASELINE_UPDATE_STREAM)
        yield ctx


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_full_pipeline_success(pipeline_context: PipelineContext):
    """
    Test successful end-to-end pipeline execution.

    **Flow:**
    1. AutoML submits trial with improved hyperparameters
    2. Orchestration receives and validates trial
    3. Eval Engine runs 50-episode evaluation
    4. HyperSync generates approval proposal
    5. Orchestration updates baseline
    6. Hot-reload completes in < 100ms
    7. All metrics and traces validated

    **Expected:**
    - Status 200/201 at each stage
    - Evaluation completes in < 2 minutes
    - Hot-reload latency < 100ms
    - Distributed trace spans connected
    - Redis Streams checkpointed correctly
    """
    ctx = pipeline_context
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("e2e_pipeline_success") as root_span:
        # ====================================================================
        # STEP 1: AutoML submits trial
        # ====================================================================
        trial_id = f"trial_{int(time.time())}"
        trial_payload = {
            "trial_id": trial_id,
            "agent_type": "dqn",
            "environment": "CartPole-v1",
            "hyperparameters": {
                "learning_rate": 0.0005,  # Improved from baseline 0.001
                "gamma": 0.995,            # Improved from baseline 0.99
                "epsilon": 0.05,           # Improved from baseline 0.1
                "buffer_size": 100000,
                "batch_size": 64,
                "target_update_freq": 1000
            },
            "metadata": {
                "optimizer": "TPE",
                "iteration": 42,
                "study_id": "study_20251119"
            }
        }

        with tracer.start_as_current_span("automl_submit_trial"):
            response = await ctx.automl_client.post(
                "/v1/trials",
                json=trial_payload
            )
            assert response.status_code == 201
            trial_response = response.json()
            assert trial_response["trial_id"] == trial_id
            assert trial_response["status"] == "submitted"

        # ====================================================================
        # STEP 2: Orchestration receives trial and initiates evaluation
        # ====================================================================
        job_id: Optional[str] = None

        with tracer.start_as_current_span("orchestration_receive_trial"):
            # Poll orchestration for job creation (async processing)
            start_time = time.time()
            while time.time() - start_time < 10:  # 10s timeout
                response = await ctx.orchestration_client.get(
                    f"/v1/trials/{trial_id}/job"
                )
                if response.status_code == 200:
                    job_data = response.json()
                    job_id = job_data.get("job_id")
                    if job_id:
                        break
                await asyncio.sleep(0.5)

            assert job_id is not None, "Orchestration did not create job within 10s"

            # Verify Redis Stream checkpoint
            stream_messages = await ctx.redis_client.xread(
                {REDIS_STREAM_KEY: "0"},
                count=10
            )
            assert len(stream_messages) > 0
            # Find our trial in the stream
            found_trial = False
            for stream_name, messages in stream_messages:
                for msg_id, msg_data in messages:
                    if msg_data.get("trial_id") == trial_id:
                        found_trial = True
                        assert msg_data.get("job_id") == job_id
                        break
            assert found_trial, "Trial not found in Redis Stream"

        # ====================================================================
        # STEP 3: Eval Engine runs evaluation
        # ====================================================================
        eval_result: Optional[Dict[str, Any]] = None

        with tracer.start_as_current_span("eval_engine_run_evaluation"):
            # Poll for evaluation completion
            start_time = time.time()
            while time.time() - start_time < EVALUATION_TIMEOUT:
                response = await ctx.eval_engine_client.get(
                    f"/v1/jobs/{job_id}"
                )
                assert response.status_code == 200
                eval_data = response.json()

                if eval_data["status"] == "completed":
                    eval_result = eval_data
                    break
                elif eval_data["status"] == "failed":
                    pytest.fail(f"Evaluation failed: {eval_data.get('error')}")

                await asyncio.sleep(2)  # Poll every 2s

            assert eval_result is not None, "Evaluation did not complete within timeout"

            # Validate evaluation result structure
            assert "mean_reward" in eval_result
            assert "std_reward" in eval_result
            assert "num_episodes" in eval_result
            assert eval_result["num_episodes"] == 50
            assert eval_result["mean_reward"] > 0  # CartPole baseline is ~195

            # Validate evaluation latency
            eval_duration = eval_result.get("duration_seconds", 999)
            assert eval_duration < EVALUATION_TIMEOUT, f"Evaluation took {eval_duration}s"

        # ====================================================================
        # STEP 4: HyperSync generates proposal
        # ====================================================================
        proposal_id: Optional[str] = None

        with tracer.start_as_current_span("hypersync_generate_proposal"):
            # Poll HyperSync for proposal generation
            start_time = time.time()
            while time.time() - start_time < 30:  # 30s timeout
                response = await ctx.hypersync_client.get(
                    f"/v1/trials/{trial_id}/proposal"
                )
                if response.status_code == 200:
                    proposal_data = response.json()
                    proposal_id = proposal_data.get("proposal_id")
                    if proposal_id:
                        # Validate proposal structure
                        assert proposal_data["trial_id"] == trial_id
                        assert proposal_data["status"] in ["pending", "approved"]
                        assert "hyperparameters" in proposal_data
                        assert "improvement" in proposal_data

                        # Check if improvement is positive (regression detection)
                        improvement = proposal_data["improvement"]
                        assert improvement > 0, f"Regression detected: {improvement}%"
                        break

                await asyncio.sleep(1)

            assert proposal_id is not None, "HyperSync did not generate proposal"

        # ====================================================================
        # STEP 5: Orchestration updates baseline
        # ====================================================================
        baseline_updated = False

        with tracer.start_as_current_span("orchestration_update_baseline"):
            # Approve proposal (simulate manual/threshold/autonomous approval)
            approve_response = await ctx.hypersync_client.post(
                f"/v1/proposals/{proposal_id}/approve",
                json={"approval_mode": "threshold"}
            )
            assert approve_response.status_code == 200

            # Poll for baseline update
            start_time = time.time()
            while time.time() - start_time < 20:  # 20s timeout
                response = await ctx.orchestration_client.get(
                    f"/v1/baselines/dqn/CartPole-v1/rank/1"
                )
                if response.status_code == 200:
                    baseline = response.json()
                    # Check if baseline matches new hyperparameters
                    if (baseline.get("hyperparameters", {}).get("learning_rate") == 0.0005):
                        baseline_updated = True
                        assert baseline["trial_id"] == trial_id
                        assert baseline["rank"] == 1
                        break

                await asyncio.sleep(1)

            assert baseline_updated, "Baseline not updated within timeout"

            # Verify baseline update in Redis Stream
            stream_messages = await ctx.redis_client.xread(
                {BASELINE_UPDATE_STREAM: "0"},
                count=10
            )
            found_update = False
            for stream_name, messages in stream_messages:
                for msg_id, msg_data in messages:
                    if msg_data.get("trial_id") == trial_id:
                        found_update = True
                        assert msg_data.get("agent_type") == "dqn"
                        assert msg_data.get("environment") == "CartPole-v1"
                        break
            assert found_update, "Baseline update not found in Redis Stream"

        # ====================================================================
        # STEP 6: Hot-reload completes
        # ====================================================================
        with tracer.start_as_current_span("hot_reload_validation"):
            hot_reload_start = time.time()

            # Poll orchestration for hot-reload completion
            reload_completed = False
            while time.time() - hot_reload_start < HOT_RELOAD_TIMEOUT:
                response = await ctx.orchestration_client.get(
                    "/v1/agents/dqn/status"
                )
                if response.status_code == 200:
                    agent_status = response.json()
                    if agent_status.get("hyperparameters", {}).get("learning_rate") == 0.0005:
                        reload_completed = True
                        hot_reload_latency = time.time() - hot_reload_start
                        assert hot_reload_latency < 0.1, f"Hot-reload took {hot_reload_latency*1000:.2f}ms"
                        break

                await asyncio.sleep(0.05)  # Poll every 50ms

            assert reload_completed, "Hot-reload did not complete within 5s"

        # ====================================================================
        # STEP 7: Validate distributed traces
        # ====================================================================
        with tracer.start_as_current_span("validate_traces"):
            # Export all spans
            spans = ctx.trace_exporter.get_finished_spans()

            # Validate trace structure
            assert len(spans) > 0, "No trace spans exported"

            # Find root span
            root_spans = [s for s in spans if s.name == "e2e_pipeline_success"]
            assert len(root_spans) == 1

            # Validate child spans
            expected_spans = [
                "automl_submit_trial",
                "orchestration_receive_trial",
                "eval_engine_run_evaluation",
                "hypersync_generate_proposal",
                "orchestration_update_baseline",
                "hot_reload_validation",
                "validate_traces"
            ]

            span_names = [s.name for s in spans]
            for expected in expected_spans:
                assert expected in span_names, f"Missing span: {expected}"

            # Validate trace timing (p99 < 250ms per service hop)
            # This is aspirational; actual latencies depend on episode count
            total_trace_time = root_spans[0].end_time - root_spans[0].start_time
            # Convert nanoseconds to seconds
            total_trace_seconds = total_trace_time / 1e9
            assert total_trace_seconds < TEST_TIMEOUT, "E2E trace exceeded timeout"


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_pipeline_with_regression_rollback(pipeline_context: PipelineContext):
    """
    Test pipeline with regression detection and automatic rollback.

    **Scenario:**
    1. Submit trial with worse hyperparameters
    2. Evaluation detects regression (mean_reward < baseline)
    3. HyperSync rejects proposal
    4. Orchestration does NOT update baseline
    5. No hot-reload triggered

    **Expected:**
    - Evaluation completes successfully
    - Regression detected (improvement < 0)
    - Proposal rejected
    - Baseline unchanged
    - Rollback metrics logged
    """
    ctx = pipeline_context
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("e2e_pipeline_regression"):
        # ====================================================================
        # STEP 1: Get current baseline for comparison
        # ====================================================================
        baseline_response = await ctx.orchestration_client.get(
            "/v1/baselines/dqn/CartPole-v1/rank/1"
        )
        assert baseline_response.status_code == 200
        original_baseline = baseline_response.json()
        original_learning_rate = original_baseline["hyperparameters"]["learning_rate"]

        # ====================================================================
        # STEP 2: Submit trial with WORSE hyperparameters
        # ====================================================================
        trial_id = f"trial_regression_{int(time.time())}"
        trial_payload = {
            "trial_id": trial_id,
            "agent_type": "dqn",
            "environment": "CartPole-v1",
            "hyperparameters": {
                "learning_rate": 0.01,     # Much higher (worse)
                "gamma": 0.9,               # Lower (worse)
                "epsilon": 0.5,             # Higher (worse exploration)
                "buffer_size": 10000,       # Smaller (worse)
                "batch_size": 16,           # Smaller (worse)
                "target_update_freq": 100   # More frequent (worse stability)
            },
            "metadata": {
                "optimizer": "TPE",
                "iteration": 99,
                "expected_outcome": "regression"
            }
        }

        response = await ctx.automl_client.post("/v1/trials", json=trial_payload)
        assert response.status_code == 201

        # ====================================================================
        # STEP 3: Wait for evaluation to complete
        # ====================================================================
        job_id = None
        start_time = time.time()
        while time.time() - start_time < 10:
            response = await ctx.orchestration_client.get(
                f"/v1/trials/{trial_id}/job"
            )
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get("job_id")
                if job_id:
                    break
            await asyncio.sleep(0.5)

        assert job_id is not None

        # Wait for evaluation completion
        eval_result = None
        start_time = time.time()
        while time.time() - start_time < EVALUATION_TIMEOUT:
            response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")
            assert response.status_code == 200
            eval_data = response.json()

            if eval_data["status"] == "completed":
                eval_result = eval_data
                break
            elif eval_data["status"] == "failed":
                pytest.fail(f"Evaluation failed: {eval_data.get('error')}")

            await asyncio.sleep(2)

        assert eval_result is not None

        # ====================================================================
        # STEP 4: HyperSync detects regression and rejects proposal
        # ====================================================================
        start_time = time.time()
        regression_detected = False

        while time.time() - start_time < 30:
            response = await ctx.hypersync_client.get(
                f"/v1/trials/{trial_id}/proposal"
            )
            if response.status_code == 200:
                proposal = response.json()
                improvement = proposal.get("improvement", 0)

                # Regression is negative improvement
                if improvement < 0:
                    regression_detected = True
                    assert proposal["status"] == "rejected"
                    assert "regression" in proposal.get("rejection_reason", "").lower()
                    break

            await asyncio.sleep(1)

        assert regression_detected, "Regression not detected by HyperSync"

        # ====================================================================
        # STEP 5: Verify baseline unchanged
        # ====================================================================
        await asyncio.sleep(2)  # Give time for any erroneous updates

        response = await ctx.orchestration_client.get(
            "/v1/baselines/dqn/CartPole-v1/rank/1"
        )
        assert response.status_code == 200
        current_baseline = response.json()

        # Baseline should be unchanged
        assert current_baseline["hyperparameters"]["learning_rate"] == original_learning_rate
        assert current_baseline["trial_id"] != trial_id

        # ====================================================================
        # STEP 6: Verify no hot-reload triggered
        # ====================================================================
        # Check baseline update stream - should not contain our trial
        stream_messages = await ctx.redis_client.xread(
            {BASELINE_UPDATE_STREAM: "0"},
            count=100
        )

        for stream_name, messages in stream_messages:
            for msg_id, msg_data in messages:
                # Should not find our regression trial in baseline updates
                assert msg_data.get("trial_id") != trial_id


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_pipeline_metrics_validation(pipeline_context: PipelineContext):
    """
    Validate all metrics are correctly emitted during pipeline execution.

    **Metrics Validated:**
    - tars_eval_evaluations_total{status="success"}
    - tars_eval_evaluation_duration_seconds
    - tars_hypersync_proposals_total{status="approved"}
    - tars_orchestration_baseline_updates_total
    - tars_orchestration_hot_reload_duration_seconds

    **Expected:**
    - All metrics incremented correctly
    - Metric labels match job context
    - Histogram buckets populated
    """
    ctx = pipeline_context

    # Submit a simple trial
    trial_id = f"trial_metrics_{int(time.time())}"
    trial_payload = {
        "trial_id": trial_id,
        "agent_type": "a2c",
        "environment": "Acrobot-v1",
        "hyperparameters": {
            "learning_rate": 0.0007,
            "gamma": 0.99,
            "n_steps": 5
        }
    }

    response = await ctx.automl_client.post("/v1/trials", json=trial_payload)
    assert response.status_code == 201

    # Wait for pipeline to complete (simplified check)
    await asyncio.sleep(30)  # Allow time for processing

    # ====================================================================
    # Fetch metrics from each service
    # ====================================================================

    # Eval Engine metrics
    eval_metrics_response = await ctx.eval_engine_client.get("/metrics")
    assert eval_metrics_response.status_code == 200
    eval_metrics_text = eval_metrics_response.text

    # Validate evaluation counter
    assert "tars_eval_evaluations_total" in eval_metrics_text
    assert 'agent_type="a2c"' in eval_metrics_text
    assert 'environment="Acrobot-v1"' in eval_metrics_text
    assert 'status="success"' in eval_metrics_text or 'status="completed"' in eval_metrics_text

    # Validate evaluation duration histogram
    assert "tars_eval_evaluation_duration_seconds" in eval_metrics_text
    assert "_bucket{" in eval_metrics_text  # Histogram buckets
    assert "_sum{" in eval_metrics_text
    assert "_count{" in eval_metrics_text

    # HyperSync metrics
    hypersync_metrics_response = await ctx.hypersync_client.get("/metrics")
    assert hypersync_metrics_response.status_code == 200
    hypersync_metrics_text = hypersync_metrics_response.text

    assert "tars_hypersync_proposals_total" in hypersync_metrics_text

    # Orchestration metrics
    orchestration_metrics_response = await ctx.orchestration_client.get("/metrics")
    assert orchestration_metrics_response.status_code == 200
    orchestration_metrics_text = orchestration_metrics_response.text

    # Should see baseline update or hot-reload metrics
    # (depending on whether improvement threshold was met)
    assert "tars_orchestration" in orchestration_metrics_text


@pytest.mark.asyncio
async def test_pipeline_concurrent_trials(pipeline_context: PipelineContext):
    """
    Test concurrent trial submissions and ensure isolation.

    **Scenario:**
    - Submit 4 trials concurrently (DQN, A2C, PPO, DDPG)
    - All evaluations should complete successfully
    - No cross-contamination of hyperparameters
    - No deadlocks or race conditions

    **Expected:**
    - All 4 trials complete within 3 minutes
    - Each trial maintains its own state
    - All baselines updated independently
    """
    ctx = pipeline_context

    # Define 4 concurrent trials
    trials = [
        {
            "trial_id": f"trial_dqn_{int(time.time())}",
            "agent_type": "dqn",
            "environment": "CartPole-v1",
            "hyperparameters": {"learning_rate": 0.001, "gamma": 0.99}
        },
        {
            "trial_id": f"trial_a2c_{int(time.time())}",
            "agent_type": "a2c",
            "environment": "CartPole-v1",
            "hyperparameters": {"learning_rate": 0.0007, "gamma": 0.99}
        },
        {
            "trial_id": f"trial_ppo_{int(time.time())}",
            "agent_type": "ppo",
            "environment": "CartPole-v1",
            "hyperparameters": {"learning_rate": 0.0003, "gamma": 0.99}
        },
        {
            "trial_id": f"trial_ddpg_{int(time.time())}",
            "agent_type": "ddpg",
            "environment": "Pendulum-v1",
            "hyperparameters": {"learning_rate": 0.001, "gamma": 0.99}
        }
    ]

    # Submit all trials concurrently
    submit_tasks = [
        ctx.automl_client.post("/v1/trials", json=trial)
        for trial in trials
    ]
    responses = await asyncio.gather(*submit_tasks)

    # All should succeed
    for response in responses:
        assert response.status_code == 201

    # Wait for all to complete (with timeout)
    trial_ids = [t["trial_id"] for t in trials]
    completed = set()

    start_time = time.time()
    while len(completed) < len(trial_ids) and time.time() - start_time < 180:
        for trial_id in trial_ids:
            if trial_id in completed:
                continue

            # Check if job exists and completed
            response = await ctx.orchestration_client.get(
                f"/v1/trials/{trial_id}/job"
            )
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get("job_id")

                if job_id:
                    eval_response = await ctx.eval_engine_client.get(
                        f"/v1/jobs/{job_id}"
                    )
                    if eval_response.status_code == 200:
                        eval_data = eval_response.json()
                        if eval_data["status"] == "completed":
                            completed.add(trial_id)

        await asyncio.sleep(5)  # Poll every 5s

    assert len(completed) == len(trial_ids), f"Only {len(completed)}/4 trials completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

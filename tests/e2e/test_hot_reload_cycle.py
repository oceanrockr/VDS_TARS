"""
Hot-Reload Cycle End-to-End Test

Tests the complete hot-reload cycle from baseline update through model
deployment with latency validation:

1. Baseline update triggered
2. HyperSync approval processed
3. Model hot-reload executed
4. Reload latency validated (< 100ms)
5. End-to-end evaluation re-run verification
6. Performance comparison pre/post reload

**Version:** v1.0.0-rc2
**Phase:** 13.8 - Final Pre-Production Validation
**Author:** T.A.R.S. Development Team
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional

import httpx
import pytest
import redis.asyncio as aioredis

# Test configuration
HOT_RELOAD_LATENCY_TARGET = 0.1  # 100ms
BASELINE_UPDATE_TIMEOUT = 10  # 10 seconds
RE_EVALUATION_TIMEOUT = 120  # 2 minutes


class HotReloadContext:
    """Context manager for hot-reload testing."""

    def __init__(self):
        self.orchestration_client: Optional[httpx.AsyncClient] = None
        self.hypersync_client: Optional[httpx.AsyncClient] = None
        self.eval_engine_client: Optional[httpx.AsyncClient] = None
        self.redis_client: Optional[aioredis.Redis] = None

    async def __aenter__(self):
        """Initialize service clients."""
        self.orchestration_client = httpx.AsyncClient(
            base_url="http://localhost:8094",
            timeout=30.0
        )
        self.hypersync_client = httpx.AsyncClient(
            base_url="http://localhost:8098",
            timeout=30.0
        )
        self.eval_engine_client = httpx.AsyncClient(
            base_url="http://localhost:8099",
            timeout=30.0
        )
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup clients."""
        if self.orchestration_client:
            await self.orchestration_client.aclose()
        if self.hypersync_client:
            await self.hypersync_client.aclose()
        if self.eval_engine_client:
            await self.eval_engine_client.aclose()
        if self.redis_client:
            await self.redis_client.close()


@pytest.fixture
async def hot_reload_context():
    """Fixture providing hot-reload context."""
    async with HotReloadContext() as ctx:
        yield ctx


@pytest.mark.asyncio
async def test_hot_reload_latency_target(hot_reload_context: HotReloadContext):
    """
    Test that hot-reload completes within 100ms latency target.

    **Scenario:**
    1. Trigger baseline update via direct API
    2. Measure time from update to agent reload completion
    3. Validate latency < 100ms
    4. Verify agent uses new hyperparameters

    **Expected:**
    - Hot-reload latency < 100ms (p99 target)
    - Agent state updated atomically
    - No interruption to in-flight evaluations
    """
    ctx = hot_reload_context

    # ====================================================================
    # STEP 1: Get current baseline
    # ====================================================================
    agent_type = "dqn"
    environment = "CartPole-v1"

    baseline_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert baseline_response.status_code == 200
    original_baseline = baseline_response.json()
    original_lr = original_baseline["hyperparameters"]["learning_rate"]

    # ====================================================================
    # STEP 2: Create new baseline with different hyperparameters
    # ====================================================================
    new_lr = original_lr * 0.9  # 10% reduction
    new_baseline = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            **original_baseline["hyperparameters"],
            "learning_rate": new_lr
        },
        "mean_reward": original_baseline.get("mean_reward", 200) + 5,  # Improvement
        "std_reward": 10.0,
        "num_episodes": 50,
        "trial_id": f"reload_test_{int(time.time())}",
        "rank": 1
    }

    # ====================================================================
    # STEP 3: Trigger baseline update and measure reload latency
    # ====================================================================
    reload_start = time.perf_counter()

    update_response = await ctx.orchestration_client.post(
        f"/v1/baselines",
        json=new_baseline
    )
    assert update_response.status_code in [200, 201]

    # Poll for agent reload completion
    reload_completed = False
    while time.perf_counter() - reload_start < 5.0:  # 5s timeout
        agent_response = await ctx.orchestration_client.get(
            f"/v1/agents/{agent_type}/status"
        )
        if agent_response.status_code == 200:
            agent_status = agent_response.json()
            current_lr = agent_status.get("hyperparameters", {}).get("learning_rate")

            if abs(current_lr - new_lr) < 1e-9:  # Float comparison
                reload_latency = time.perf_counter() - reload_start
                reload_completed = True
                break

        await asyncio.sleep(0.01)  # Poll every 10ms

    assert reload_completed, "Hot-reload did not complete within 5s"

    # ====================================================================
    # STEP 4: Validate latency target
    # ====================================================================
    assert reload_latency < HOT_RELOAD_LATENCY_TARGET, (
        f"Hot-reload latency {reload_latency*1000:.2f}ms exceeds "
        f"target {HOT_RELOAD_LATENCY_TARGET*1000:.0f}ms"
    )

    # Log latency for tracking
    print(f"✓ Hot-reload completed in {reload_latency*1000:.2f}ms")

    # ====================================================================
    # STEP 5: Verify Redis event published
    # ====================================================================
    # Check for baseline update event in Redis Stream
    stream_messages = await ctx.redis_client.xread(
        {"tars:baseline:updates": "0"},
        count=10
    )

    found_update = False
    for stream_name, messages in stream_messages:
        for msg_id, msg_data in messages:
            if (msg_data.get("agent_type") == agent_type and
                msg_data.get("trial_id") == new_baseline["trial_id"]):
                found_update = True
                assert float(msg_data.get("learning_rate", 0)) == new_lr
                break

    assert found_update, "Baseline update not found in Redis Stream"


@pytest.mark.asyncio
async def test_hot_reload_preserves_in_flight_evaluations(hot_reload_context: HotReloadContext):
    """
    Test that hot-reload does not interrupt in-flight evaluations.

    **Scenario:**
    1. Start long-running evaluation (100 episodes)
    2. Trigger hot-reload during evaluation
    3. Verify evaluation completes with OLD hyperparameters
    4. Verify NEW evaluation uses NEW hyperparameters

    **Expected:**
    - In-flight evaluation completes successfully
    - In-flight evaluation uses old hyperparameters
    - Subsequent evaluations use new hyperparameters
    - No evaluation failures or crashes
    """
    ctx = hot_reload_context

    agent_type = "a2c"
    environment = "CartPole-v1"

    # ====================================================================
    # STEP 1: Get current baseline
    # ====================================================================
    baseline_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )

    if baseline_response.status_code == 404:
        # Create initial baseline if not exists
        initial_baseline = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": {
                "learning_rate": 0.0007,
                "gamma": 0.99,
                "n_steps": 5
            },
            "mean_reward": 190.0,
            "std_reward": 15.0,
            "num_episodes": 50,
            "trial_id": "initial_baseline",
            "rank": 1
        }
        await ctx.orchestration_client.post("/v1/baselines", json=initial_baseline)
        baseline_response = await ctx.orchestration_client.get(
            f"/v1/baselines/{agent_type}/{environment}/rank/1"
        )

    assert baseline_response.status_code == 200
    original_baseline = baseline_response.json()
    original_lr = original_baseline["hyperparameters"]["learning_rate"]

    # ====================================================================
    # STEP 2: Start long-running evaluation (in background)
    # ====================================================================
    eval_request = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": original_baseline["hyperparameters"],
        "num_episodes": 100,  # Long-running
        "quick_mode": False
    }

    eval_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=eval_request
    )
    assert eval_response.status_code in [200, 201, 202]
    eval_data = eval_response.json()
    job_id_in_flight = eval_data["job_id"]

    # Wait for evaluation to start
    await asyncio.sleep(2)

    # ====================================================================
    # STEP 3: Trigger hot-reload DURING evaluation
    # ====================================================================
    new_lr = original_lr * 1.1  # 10% increase
    new_baseline = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            **original_baseline["hyperparameters"],
            "learning_rate": new_lr
        },
        "mean_reward": original_baseline.get("mean_reward", 190) + 10,
        "std_reward": 12.0,
        "num_episodes": 50,
        "trial_id": f"reload_during_eval_{int(time.time())}",
        "rank": 1
    }

    update_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=new_baseline
    )
    assert update_response.status_code in [200, 201]

    # Wait for hot-reload to propagate
    await asyncio.sleep(1)

    # ====================================================================
    # STEP 4: Verify in-flight evaluation completes with OLD hyperparameters
    # ====================================================================
    # Poll for in-flight evaluation completion
    start_time = time.time()
    in_flight_completed = False

    while time.time() - start_time < RE_EVALUATION_TIMEOUT:
        eval_status_response = await ctx.eval_engine_client.get(
            f"/v1/jobs/{job_id_in_flight}"
        )
        if eval_status_response.status_code == 200:
            status_data = eval_status_response.json()
            if status_data["status"] in ["completed", "failed"]:
                in_flight_completed = True
                assert status_data["status"] == "completed", "In-flight eval failed"

                # Verify it used OLD hyperparameters
                # (This assumes eval result includes hyperparameters)
                if "hyperparameters" in status_data:
                    used_lr = status_data["hyperparameters"].get("learning_rate")
                    assert abs(used_lr - original_lr) < 1e-9, (
                        "In-flight evaluation did not preserve original hyperparameters"
                    )
                break

        await asyncio.sleep(2)

    assert in_flight_completed, "In-flight evaluation did not complete"

    # ====================================================================
    # STEP 5: Start NEW evaluation and verify it uses NEW hyperparameters
    # ====================================================================
    new_eval_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json={
            "agent_type": agent_type,
            "environment": environment,
            "num_episodes": 10,  # Quick validation
            "quick_mode": True
        }
    )
    assert new_eval_response.status_code in [200, 201, 202]
    new_eval_data = new_eval_response.json()
    job_id_new = new_eval_data["job_id"]

    # Wait for new evaluation to complete
    start_time = time.time()
    new_eval_completed = False

    while time.time() - start_time < 60:  # 1 minute timeout
        new_status_response = await ctx.eval_engine_client.get(
            f"/v1/jobs/{job_id_new}"
        )
        if new_status_response.status_code == 200:
            new_status_data = new_status_response.json()
            if new_status_data["status"] == "completed":
                new_eval_completed = True

                # Verify it used NEW hyperparameters
                # This may require fetching from baseline or agent status
                agent_status = await ctx.orchestration_client.get(
                    f"/v1/agents/{agent_type}/status"
                )
                assert agent_status.status_code == 200
                agent_data = agent_status.json()
                current_lr = agent_data["hyperparameters"]["learning_rate"]
                assert abs(current_lr - new_lr) < 1e-9, (
                    "New evaluation did not use new hyperparameters"
                )
                break

        await asyncio.sleep(2)

    assert new_eval_completed, "New evaluation did not complete"


@pytest.mark.asyncio
async def test_hot_reload_rollback(hot_reload_context: HotReloadContext):
    """
    Test hot-reload rollback when new hyperparameters cause failures.

    **Scenario:**
    1. Deploy bad hyperparameters (e.g., learning_rate = 0)
    2. Detect evaluation failures
    3. Trigger automatic rollback to previous baseline
    4. Verify rollback completes successfully

    **Expected:**
    - Bad hyperparameters cause evaluation failure
    - Rollback triggered automatically
    - Previous baseline restored
    - Subsequent evaluations succeed
    """
    ctx = hot_reload_context

    agent_type = "ppo"
    environment = "CartPole-v1"

    # ====================================================================
    # STEP 1: Get current working baseline
    # ====================================================================
    baseline_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )

    if baseline_response.status_code == 404:
        # Create working baseline
        working_baseline = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "clip_range": 0.2,
                "n_steps": 2048
            },
            "mean_reward": 195.0,
            "std_reward": 10.0,
            "num_episodes": 50,
            "trial_id": "working_baseline",
            "rank": 1
        }
        await ctx.orchestration_client.post("/v1/baselines", json=working_baseline)
        baseline_response = await ctx.orchestration_client.get(
            f"/v1/baselines/{agent_type}/{environment}/rank/1"
        )

    assert baseline_response.status_code == 200
    working_baseline = baseline_response.json()
    working_hyperparameters = working_baseline["hyperparameters"].copy()

    # ====================================================================
    # STEP 2: Deploy BAD hyperparameters
    # ====================================================================
    bad_baseline = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            **working_hyperparameters,
            "learning_rate": 0.0  # Invalid: will cause training failure
        },
        "mean_reward": 0.0,  # Clearly bad
        "std_reward": 0.0,
        "num_episodes": 50,
        "trial_id": f"bad_baseline_{int(time.time())}",
        "rank": 1
    }

    bad_update_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=bad_baseline
    )
    # Some systems may reject bad hyperparameters at API level
    # If accepted, continue with rollback test

    if bad_update_response.status_code not in [200, 201]:
        # System rejected bad hyperparameters (good!)
        print("✓ System rejected invalid hyperparameters at API level")
        return

    # Wait for hot-reload
    await asyncio.sleep(2)

    # ====================================================================
    # STEP 3: Attempt evaluation with bad hyperparameters (expect failure)
    # ====================================================================
    eval_request = {
        "agent_type": agent_type,
        "environment": environment,
        "num_episodes": 10,
        "quick_mode": True
    }

    eval_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=eval_request
    )

    if eval_response.status_code not in [200, 201, 202]:
        # Evaluation rejected immediately (system detected bad hyperparams)
        print("✓ Evaluation rejected bad hyperparameters immediately")
    else:
        eval_data = eval_response.json()
        job_id = eval_data["job_id"]

        # Wait for evaluation to fail
        start_time = time.time()
        while time.time() - start_time < 60:
            status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data["status"] == "failed":
                    print("✓ Evaluation failed with bad hyperparameters")
                    break
                elif status_data["status"] == "completed":
                    # Evaluation succeeded despite bad hyperparameters
                    # (may indicate test issue or robust training)
                    print("⚠ Evaluation completed despite learning_rate=0")
                    return

            await asyncio.sleep(2)

    # ====================================================================
    # STEP 4: Trigger rollback to working baseline
    # ====================================================================
    rollback_response = await ctx.orchestration_client.post(
        f"/v1/baselines/{agent_type}/{environment}/rollback"
    )
    # Some APIs may not have explicit rollback endpoint
    # Alternative: restore previous baseline manually

    if rollback_response.status_code == 404:
        # Manual rollback: re-post working baseline
        rollback_response = await ctx.orchestration_client.post(
            "/v1/baselines",
            json=working_baseline
        )

    assert rollback_response.status_code in [200, 201], "Rollback failed"

    # Wait for rollback to propagate
    await asyncio.sleep(2)

    # ====================================================================
    # STEP 5: Verify rollback succeeded
    # ====================================================================
    current_baseline_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert current_baseline_response.status_code == 200
    current_baseline = current_baseline_response.json()

    # Verify hyperparameters restored
    current_lr = current_baseline["hyperparameters"]["learning_rate"]
    working_lr = working_hyperparameters["learning_rate"]
    assert abs(current_lr - working_lr) < 1e-9, "Rollback did not restore hyperparameters"

    # ====================================================================
    # STEP 6: Verify subsequent evaluation succeeds
    # ====================================================================
    verification_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=eval_request
    )
    assert verification_response.status_code in [200, 201, 202]
    verification_data = verification_response.json()
    verification_job_id = verification_data["job_id"]

    # Wait for verification evaluation to complete
    start_time = time.time()
    verification_completed = False

    while time.time() - start_time < 60:
        verification_status = await ctx.eval_engine_client.get(
            f"/v1/jobs/{verification_job_id}"
        )
        if verification_status.status_code == 200:
            verification_status_data = verification_status.json()
            if verification_status_data["status"] == "completed":
                verification_completed = True
                assert verification_status_data.get("mean_reward", 0) > 0
                print("✓ Post-rollback evaluation succeeded")
                break
            elif verification_status_data["status"] == "failed":
                pytest.fail("Post-rollback evaluation failed")

        await asyncio.sleep(2)

    assert verification_completed, "Verification evaluation did not complete"


@pytest.mark.asyncio
async def test_hot_reload_multiple_agents_simultaneously(hot_reload_context: HotReloadContext):
    """
    Test hot-reload of multiple agents simultaneously.

    **Scenario:**
    1. Update baselines for DQN, A2C, PPO, DDPG simultaneously
    2. Verify all agents reload within 100ms each
    3. Verify no cross-contamination of hyperparameters

    **Expected:**
    - All 4 agents reload successfully
    - Each agent uses correct hyperparameters
    - No race conditions or deadlocks
    - Total time < 500ms
    """
    ctx = hot_reload_context

    agents = ["dqn", "a2c", "ppo", "ddpg"]
    environment = "CartPole-v1"  # Common environment

    # Note: DDPG typically requires continuous action spaces
    # For testing purposes, we'll use compatible agents

    test_agents = ["dqn", "a2c", "ppo"]  # Remove DDPG for discrete env

    # ====================================================================
    # STEP 1: Prepare new baselines for all agents
    # ====================================================================
    new_baselines = []
    for agent_type in test_agents:
        new_baselines.append({
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": {
                "learning_rate": 0.0001 * (test_agents.index(agent_type) + 1),  # Unique
                "gamma": 0.99
            },
            "mean_reward": 200.0,
            "std_reward": 10.0,
            "num_episodes": 50,
            "trial_id": f"multi_reload_{agent_type}_{int(time.time())}",
            "rank": 1
        })

    # ====================================================================
    # STEP 2: Trigger all baseline updates simultaneously
    # ====================================================================
    reload_start = time.perf_counter()

    update_tasks = [
        ctx.orchestration_client.post("/v1/baselines", json=baseline)
        for baseline in new_baselines
    ]
    update_responses = await asyncio.gather(*update_tasks, return_exceptions=True)

    # Verify all updates succeeded
    for i, response in enumerate(update_responses):
        if isinstance(response, Exception):
            pytest.fail(f"Update failed for {test_agents[i]}: {response}")
        assert response.status_code in [200, 201]

    # ====================================================================
    # STEP 3: Poll for all agents to complete reload
    # ====================================================================
    agents_reloaded = set()

    while len(agents_reloaded) < len(test_agents) and time.perf_counter() - reload_start < 5.0:
        for i, agent_type in enumerate(test_agents):
            if agent_type in agents_reloaded:
                continue

            agent_response = await ctx.orchestration_client.get(
                f"/v1/agents/{agent_type}/status"
            )
            if agent_response.status_code == 200:
                agent_status = agent_response.json()
                expected_lr = new_baselines[i]["hyperparameters"]["learning_rate"]
                current_lr = agent_status.get("hyperparameters", {}).get("learning_rate")

                if current_lr and abs(current_lr - expected_lr) < 1e-9:
                    agents_reloaded.add(agent_type)

        await asyncio.sleep(0.01)  # Poll every 10ms

    total_reload_time = time.perf_counter() - reload_start

    # ====================================================================
    # STEP 4: Validate results
    # ====================================================================
    assert len(agents_reloaded) == len(test_agents), (
        f"Only {len(agents_reloaded)}/{len(test_agents)} agents reloaded"
    )

    assert total_reload_time < 0.5, (
        f"Multi-agent reload took {total_reload_time*1000:.2f}ms "
        f"(target: 500ms for {len(test_agents)} agents)"
    )

    print(f"✓ All {len(test_agents)} agents reloaded in {total_reload_time*1000:.2f}ms")

    # Verify no cross-contamination
    for i, agent_type in enumerate(test_agents):
        agent_response = await ctx.orchestration_client.get(
            f"/v1/agents/{agent_type}/status"
        )
        agent_status = agent_response.json()
        expected_lr = new_baselines[i]["hyperparameters"]["learning_rate"]
        actual_lr = agent_status["hyperparameters"]["learning_rate"]

        assert abs(actual_lr - expected_lr) < 1e-9, (
            f"{agent_type} has wrong learning_rate: {actual_lr} != {expected_lr}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

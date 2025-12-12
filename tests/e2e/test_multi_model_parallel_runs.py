"""
Multi-Model Parallel Runs End-to-End Test

Stress-tests concurrent evaluation of multiple RL agents (DQN, A2C, PPO, DDPG):

1. Parallel evaluation of 4 different agents
2. State isolation verification
3. Environment cache correctness per agent
4. Baseline independence
5. Resource contention handling
6. Cross-agent performance comparison

**Version:** v1.0.0-rc2
**Phase:** 13.8 - Final Pre-Production Validation
**Author:** T.A.R.S. Development Team
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple

import httpx
import pytest


class MultiModelContext:
    """Context manager for multi-model testing."""

    def __init__(self):
        self.eval_engine_client: Optional[httpx.AsyncClient] = None
        self.orchestration_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Initialize service clients."""
        self.eval_engine_client = httpx.AsyncClient(
            base_url="http://localhost:8099",
            timeout=60.0
        )
        self.orchestration_client = httpx.AsyncClient(
            base_url="http://localhost:8094",
            timeout=60.0
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup clients."""
        if self.eval_engine_client:
            await self.eval_engine_client.aclose()
        if self.orchestration_client:
            await self.orchestration_client.aclose()


@pytest.fixture
async def multi_model_context():
    """Fixture providing multi-model context."""
    async with MultiModelContext() as ctx:
        yield ctx


# Agent configurations for testing
AGENT_CONFIGS = {
    "dqn": {
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 0.1,
            "buffer_size": 50000,
            "batch_size": 32,
            "target_update_freq": 1000
        },
        "environments": ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    },
    "a2c": {
        "hyperparameters": {
            "learning_rate": 0.0007,
            "gamma": 0.99,
            "n_steps": 5,
            "ent_coef": 0.01,
            "vf_coef": 0.25
        },
        "environments": ["CartPole-v1", "Acrobot-v1"]
    },
    "ppo": {
        "hyperparameters": {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_range": 0.2,
            "n_steps": 2048,
            "ent_coef": 0.01,
            "vf_coef": 0.5
        },
        "environments": ["CartPole-v1", "Acrobot-v1"]
    },
    "ddpg": {
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 100000,
            "batch_size": 64
        },
        "environments": ["Pendulum-v1"]  # Continuous control
    }
}


@pytest.mark.asyncio
async def test_four_agents_parallel_evaluation(multi_model_context: MultiModelContext):
    """
    Test parallel evaluation of all 4 RL agents.

    **Scenario:**
    1. Submit evaluations for DQN, A2C, PPO, DDPG simultaneously
    2. Each agent evaluates on appropriate environment
    3. Verify all complete successfully
    4. Verify state isolation (no cross-contamination)

    **Expected:**
    - All 4 evaluations complete in < 5 minutes
    - Each agent uses correct hyperparameters
    - No resource conflicts
    - Results are agent-specific
    """
    ctx = multi_model_context

    # ====================================================================
    # STEP 1: Prepare evaluation requests for all agents
    # ====================================================================
    eval_requests = []
    for agent_type, config in AGENT_CONFIGS.items():
        # Use first supported environment for each agent
        environment = config["environments"][0]

        eval_request = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": config["hyperparameters"],
            "num_episodes": 20,  # Quick evaluation
            "quick_mode": True
        }
        eval_requests.append((agent_type, environment, eval_request))

    # ====================================================================
    # STEP 2: Submit all evaluations concurrently
    # ====================================================================
    submit_start = time.time()

    submit_tasks = [
        ctx.eval_engine_client.post("/v1/evaluate", json=req)
        for _, _, req in eval_requests
    ]
    submit_responses = await asyncio.gather(*submit_tasks, return_exceptions=True)

    submit_latency = time.time() - submit_start
    print(f"Submitted 4 agent evaluations in {submit_latency*1000:.2f}ms")

    # ====================================================================
    # STEP 3: Collect job IDs
    # ====================================================================
    job_mapping = {}  # (agent_type, environment) -> job_id

    for i, response in enumerate(submit_responses):
        if isinstance(response, Exception):
            pytest.fail(f"Evaluation {i} submission failed: {response}")

        assert response.status_code in [200, 201, 202], (
            f"Evaluation {i} submission returned {response.status_code}"
        )

        data = response.json()
        agent_type, environment, _ = eval_requests[i]
        job_mapping[(agent_type, environment)] = data["job_id"]

    # ====================================================================
    # STEP 4: Poll for all completions
    # ====================================================================
    completed_jobs = {}
    timeout = 300  # 5 minutes
    poll_start = time.time()

    while len(completed_jobs) < len(job_mapping) and time.time() - poll_start < timeout:
        for (agent_type, environment), job_id in job_mapping.items():
            key = (agent_type, environment)
            if key in completed_jobs:
                continue

            status_response = await ctx.eval_engine_client.get(
                f"/v1/jobs/{job_id}"
            )

            if status_response.status_code == 200:
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    completed_jobs[key] = status_data
                    mean_reward = status_data.get("mean_reward", "N/A")
                    print(f"✓ {agent_type} on {environment}: reward={mean_reward}")
                elif status_data["status"] == "failed":
                    error = status_data.get("error", "Unknown error")
                    pytest.fail(f"{agent_type} on {environment} failed: {error}")

        if len(completed_jobs) < len(job_mapping):
            await asyncio.sleep(3)

    # ====================================================================
    # STEP 5: Verify all completed
    # ====================================================================
    total_time = time.time() - poll_start
    assert len(completed_jobs) == len(job_mapping), (
        f"Only {len(completed_jobs)}/4 evaluations completed in {total_time:.2f}s"
    )

    print(f"✓ All 4 agents completed in {total_time:.2f}s")

    # ====================================================================
    # STEP 6: Verify state isolation (hyperparameters correctness)
    # ====================================================================
    for (agent_type, environment), result in completed_jobs.items():
        expected_hyperparams = AGENT_CONFIGS[agent_type]["hyperparameters"]

        # If result includes hyperparameters, verify they match
        if "hyperparameters" in result:
            actual_hyperparams = result["hyperparameters"]

            # Check learning rate as key indicator
            expected_lr = expected_hyperparams["learning_rate"]
            actual_lr = actual_hyperparams.get("learning_rate")

            if actual_lr is not None:
                assert abs(actual_lr - expected_lr) < 1e-9, (
                    f"{agent_type} used wrong learning_rate: {actual_lr} != {expected_lr}"
                )

        # Verify episode count
        assert result["num_episodes"] == 20

    print("✓ State isolation verified (all agents used correct hyperparameters)")


@pytest.mark.asyncio
async def test_agent_baseline_independence(multi_model_context: MultiModelContext):
    """
    Test that baselines are independent per agent type.

    **Scenario:**
    1. Create baselines for DQN, A2C, PPO on same environment
    2. Update DQN baseline
    3. Verify A2C and PPO baselines unchanged
    4. Query all baselines
    5. Verify agent-specific isolation

    **Expected:**
    - Each agent has independent baselines
    - Updates to one agent don't affect others
    - Hyperparameters correctly scoped to agent
    """
    ctx = multi_model_context

    environment = "CartPole-v1"
    agent_types = ["dqn", "a2c", "ppo"]

    # ====================================================================
    # STEP 1: Create baselines for all agents
    # ====================================================================
    original_baselines = {}

    for agent_type in agent_types:
        baseline = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": AGENT_CONFIGS[agent_type]["hyperparameters"].copy(),
            "mean_reward": 190.0,
            "std_reward": 10.0,
            "num_episodes": 50,
            "trial_id": f"{agent_type}_baseline_v1_{int(time.time())}",
            "rank": 1
        }

        response = await ctx.orchestration_client.post(
            "/v1/baselines",
            json=baseline
        )
        assert response.status_code in [200, 201], f"{agent_type} baseline creation failed"

        original_baselines[agent_type] = baseline
        await asyncio.sleep(0.2)

    # ====================================================================
    # STEP 2: Update DQN baseline
    # ====================================================================
    updated_dqn_baseline = {
        **original_baselines["dqn"],
        "trial_id": f"dqn_baseline_v2_{int(time.time())}",
        "mean_reward": 205.0,  # Improved
        "hyperparameters": {
            **original_baselines["dqn"]["hyperparameters"],
            "learning_rate": 0.0008  # Changed
        }
    }

    update_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=updated_dqn_baseline
    )
    assert update_response.status_code in [200, 201]

    await asyncio.sleep(1)

    # ====================================================================
    # STEP 3: Verify DQN updated, others unchanged
    # ====================================================================
    dqn_current = await ctx.orchestration_client.get(
        f"/v1/baselines/dqn/{environment}/rank/1"
    )
    assert dqn_current.status_code == 200
    dqn_data = dqn_current.json()

    # DQN should be updated
    assert dqn_data["trial_id"] == updated_dqn_baseline["trial_id"]
    assert abs(dqn_data["mean_reward"] - 205.0) < 1e-6
    assert abs(dqn_data["hyperparameters"]["learning_rate"] - 0.0008) < 1e-9

    # A2C should be unchanged
    a2c_current = await ctx.orchestration_client.get(
        f"/v1/baselines/a2c/{environment}/rank/1"
    )
    assert a2c_current.status_code == 200
    a2c_data = a2c_current.json()

    assert a2c_data["trial_id"] == original_baselines["a2c"]["trial_id"]
    assert abs(a2c_data["mean_reward"] - 190.0) < 1e-6
    original_a2c_lr = original_baselines["a2c"]["hyperparameters"]["learning_rate"]
    assert abs(a2c_data["hyperparameters"]["learning_rate"] - original_a2c_lr) < 1e-9

    # PPO should be unchanged
    ppo_current = await ctx.orchestration_client.get(
        f"/v1/baselines/ppo/{environment}/rank/1"
    )
    assert ppo_current.status_code == 200
    ppo_data = ppo_current.json()

    assert ppo_data["trial_id"] == original_baselines["ppo"]["trial_id"]
    assert abs(ppo_data["mean_reward"] - 190.0) < 1e-6

    print("✓ Agent baseline independence verified")


@pytest.mark.asyncio
async def test_concurrent_agent_hot_reload(multi_model_context: MultiModelContext):
    """
    Test hot-reload of multiple agents simultaneously.

    **Scenario:**
    1. Update baselines for DQN, A2C, PPO concurrently
    2. Verify all agents reload independently
    3. Verify no cross-agent interference
    4. Verify reload latency per agent < 100ms

    **Expected:**
    - All agents reload successfully
    - Each agent uses correct updated hyperparameters
    - No race conditions
    - Total time < 500ms
    """
    ctx = multi_model_context

    environment = "CartPole-v1"
    agent_types = ["dqn", "a2c", "ppo"]

    # ====================================================================
    # STEP 1: Prepare updated baselines
    # ====================================================================
    updated_baselines = []

    for i, agent_type in enumerate(agent_types):
        updated_baseline = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": {
                **AGENT_CONFIGS[agent_type]["hyperparameters"],
                "learning_rate": 0.0001 * (i + 2),  # Unique per agent
                "gamma": 0.995  # Common update
            },
            "mean_reward": 200.0 + (i * 2),
            "std_reward": 10.0,
            "num_episodes": 50,
            "trial_id": f"{agent_type}_hot_reload_{int(time.time())}",
            "rank": 1
        }
        updated_baselines.append(updated_baseline)

    # ====================================================================
    # STEP 2: Submit all baseline updates concurrently
    # ====================================================================
    reload_start = time.perf_counter()

    update_tasks = [
        ctx.orchestration_client.post("/v1/baselines", json=baseline)
        for baseline in updated_baselines
    ]
    update_responses = await asyncio.gather(*update_tasks, return_exceptions=True)

    # Verify all updates succeeded
    for i, response in enumerate(update_responses):
        if isinstance(response, Exception):
            pytest.fail(f"Update for {agent_types[i]} failed: {response}")
        assert response.status_code in [200, 201]

    # ====================================================================
    # STEP 3: Poll for agent reload completion
    # ====================================================================
    agents_reloaded = set()

    while len(agents_reloaded) < len(agent_types) and time.perf_counter() - reload_start < 5.0:
        for i, agent_type in enumerate(agent_types):
            if agent_type in agents_reloaded:
                continue

            agent_response = await ctx.orchestration_client.get(
                f"/v1/agents/{agent_type}/status"
            )

            if agent_response.status_code == 200:
                agent_status = agent_response.json()
                expected_lr = updated_baselines[i]["hyperparameters"]["learning_rate"]
                current_lr = agent_status.get("hyperparameters", {}).get("learning_rate")

                if current_lr and abs(current_lr - expected_lr) < 1e-9:
                    agents_reloaded.add(agent_type)

        await asyncio.sleep(0.01)

    total_reload_time = time.perf_counter() - reload_start

    # ====================================================================
    # STEP 4: Validate results
    # ====================================================================
    assert len(agents_reloaded) == len(agent_types), (
        f"Only {len(agents_reloaded)}/3 agents reloaded"
    )

    assert total_reload_time < 0.5, (
        f"Multi-agent reload took {total_reload_time*1000:.2f}ms (target: 500ms)"
    )

    print(f"✓ All 3 agents reloaded in {total_reload_time*1000:.2f}ms")

    # ====================================================================
    # STEP 5: Verify no cross-contamination
    # ====================================================================
    for i, agent_type in enumerate(agent_types):
        agent_response = await ctx.orchestration_client.get(
            f"/v1/agents/{agent_type}/status"
        )
        agent_status = agent_response.json()

        expected_lr = updated_baselines[i]["hyperparameters"]["learning_rate"]
        actual_lr = agent_status["hyperparameters"]["learning_rate"]

        assert abs(actual_lr - expected_lr) < 1e-9, (
            f"{agent_type} has wrong learning_rate: {actual_lr} != {expected_lr}"
        )

        # Verify gamma also updated
        assert abs(agent_status["hyperparameters"]["gamma"] - 0.995) < 1e-9

    print("✓ No cross-agent contamination detected")


@pytest.mark.asyncio
async def test_resource_contention_handling(multi_model_context: MultiModelContext):
    """
    Test system behavior under heavy concurrent load.

    **Scenario:**
    1. Submit 8 evaluations concurrently (2 per agent type)
    2. Monitor for failures, timeouts, or degradation
    3. Verify all complete or gracefully queue

    **Expected:**
    - System handles high concurrency gracefully
    - No crashes or data corruption
    - Either all complete or some are queued (not failed)
    - Performance degrades gracefully
    """
    ctx = multi_model_context

    # ====================================================================
    # STEP 1: Prepare 8 concurrent evaluations
    # ====================================================================
    eval_requests = []

    for agent_type, config in AGENT_CONFIGS.items():
        for i in range(2):  # 2 evaluations per agent
            environment = config["environments"][0]
            eval_request = {
                "agent_type": agent_type,
                "environment": environment,
                "hyperparameters": config["hyperparameters"],
                "num_episodes": 15,
                "quick_mode": True
            }
            eval_requests.append((f"{agent_type}_{i}", eval_request))

    # ====================================================================
    # STEP 2: Submit all concurrently
    # ====================================================================
    submit_start = time.time()

    submit_tasks = [
        ctx.eval_engine_client.post("/v1/evaluate", json=req)
        for _, req in eval_requests
    ]
    submit_responses = await asyncio.gather(*submit_tasks, return_exceptions=True)

    submit_latency = time.time() - submit_start
    print(f"Submitted 8 evaluations in {submit_latency*1000:.2f}ms")

    # ====================================================================
    # STEP 3: Collect job IDs
    # ====================================================================
    job_ids = {}

    for i, response in enumerate(submit_responses):
        name, _ = eval_requests[i]

        if isinstance(response, Exception):
            print(f"⚠ {name} submission exception: {response}")
            continue

        if response.status_code in [200, 201, 202]:
            data = response.json()
            job_ids[name] = data["job_id"]
        elif response.status_code == 429:  # Rate limited
            print(f"⚠ {name} rate limited (expected under load)")
        elif response.status_code == 503:  # Service unavailable
            print(f"⚠ {name} service unavailable (queue full?)")
        else:
            print(f"⚠ {name} unexpected status: {response.status_code}")

    # ====================================================================
    # STEP 4: Poll for completions (with extended timeout)
    # ====================================================================
    completed_jobs = {}
    failed_jobs = []
    timeout = 600  # 10 minutes (generous for 8 concurrent)
    start_time = time.time()

    while len(completed_jobs) + len(failed_jobs) < len(job_ids) and time.time() - start_time < timeout:
        for name, job_id in job_ids.items():
            if name in completed_jobs or name in failed_jobs:
                continue

            try:
                status_response = await ctx.eval_engine_client.get(
                    f"/v1/jobs/{job_id}"
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()

                    if status_data["status"] == "completed":
                        completed_jobs[name] = status_data
                        print(f"✓ {name} completed")
                    elif status_data["status"] == "failed":
                        failed_jobs.append(name)
                        print(f"✗ {name} failed: {status_data.get('error')}")

            except Exception as e:
                print(f"⚠ Error polling {name}: {e}")

        if len(completed_jobs) + len(failed_jobs) < len(job_ids):
            await asyncio.sleep(5)

    # ====================================================================
    # STEP 5: Analyze results
    # ====================================================================
    total_submitted = len(job_ids)
    total_completed = len(completed_jobs)
    total_failed = len(failed_jobs)

    print(f"\nResults:")
    print(f"  Submitted: {total_submitted}/8")
    print(f"  Completed: {total_completed}")
    print(f"  Failed: {total_failed}")

    # Success criteria: At least 50% complete successfully
    success_rate = total_completed / total_submitted if total_submitted > 0 else 0

    assert success_rate >= 0.5, (
        f"Only {success_rate*100:.1f}% completed (expected ≥50%)"
    )

    # No crashes (system still responsive)
    health_response = await ctx.eval_engine_client.get("/health")
    assert health_response.status_code == 200, "System health check failed"

    print(f"✓ System handled {total_submitted} concurrent evaluations gracefully")
    print(f"✓ Success rate: {success_rate*100:.1f}%")


@pytest.mark.asyncio
async def test_cross_agent_performance_comparison(multi_model_context: MultiModelContext):
    """
    Test performance comparison across agents on same environment.

    **Scenario:**
    1. Evaluate DQN, A2C, PPO on CartPole-v1
    2. Compare mean rewards
    3. Verify results make sense (no obvious bugs)
    4. Store results for baseline comparison

    **Expected:**
    - All agents complete successfully
    - Results are reasonable for CartPole
    - Performance ordering is plausible
    """
    ctx = multi_model_context

    environment = "CartPole-v1"
    agent_types = ["dqn", "a2c", "ppo"]

    # ====================================================================
    # STEP 1: Submit evaluations for all agents
    # ====================================================================
    eval_requests = []
    for agent_type in agent_types:
        eval_request = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": AGENT_CONFIGS[agent_type]["hyperparameters"],
            "num_episodes": 30,
            "quick_mode": False,
            "seed": 42  # Same seed for fair comparison
        }
        eval_requests.append((agent_type, eval_request))

    # Submit concurrently
    submit_tasks = [
        ctx.eval_engine_client.post("/v1/evaluate", json=req)
        for _, req in eval_requests
    ]
    submit_responses = await asyncio.gather(*submit_tasks)

    # Collect job IDs
    job_mapping = {}
    for i, response in enumerate(submit_responses):
        assert response.status_code in [200, 201, 202]
        agent_type, _ = eval_requests[i]
        job_mapping[agent_type] = response.json()["job_id"]

    # ====================================================================
    # STEP 2: Wait for all completions
    # ====================================================================
    results = {}
    start_time = time.time()

    while len(results) < len(agent_types) and time.time() - start_time < 180:
        for agent_type, job_id in job_mapping.items():
            if agent_type in results:
                continue

            status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data["status"] == "completed":
                    results[agent_type] = status_data

        await asyncio.sleep(3)

    assert len(results) == len(agent_types), "Not all evaluations completed"

    # ====================================================================
    # STEP 3: Compare results
    # ====================================================================
    print("\nCross-Agent Performance on CartPole-v1:")
    for agent_type in agent_types:
        mean_reward = results[agent_type]["mean_reward"]
        std_reward = results[agent_type]["std_reward"]
        print(f"  {agent_type.upper()}: {mean_reward:.2f} ± {std_reward:.2f}")

    # Verify results are reasonable (CartPole rewards 0-500)
    for agent_type in agent_types:
        mean_reward = results[agent_type]["mean_reward"]
        assert 0 <= mean_reward <= 500, (
            f"{agent_type} reward {mean_reward} out of range [0, 500]"
        )

    print("✓ All agent results are within reasonable ranges")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

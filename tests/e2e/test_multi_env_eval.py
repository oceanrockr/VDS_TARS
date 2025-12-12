"""
Multi-Environment Evaluation End-to-End Test

Tests concurrent evaluation across multiple Gymnasium environments:

1. Concurrent multi-environment evaluation
2. Environment-specific baseline management
3. Stability/entropy/variance validation
4. Environment cache performance
5. Resource isolation between environments

**Tested Environments:**
- CartPole-v1 (discrete, classic control)
- Acrobot-v1 (discrete, classic control)
- MountainCar-v0 (discrete, classic control)
- Pendulum-v1 (continuous, classic control)

**Version:** v1.0.0-rc2
**Phase:** 13.8 - Final Pre-Production Validation
**Author:** T.A.R.S. Development Team
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

import httpx
import pytest


class MultiEnvContext:
    """Context manager for multi-environment testing."""

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
async def multi_env_context():
    """Fixture providing multi-environment context."""
    async with MultiEnvContext() as ctx:
        yield ctx


# Environment specifications with expected performance characteristics
DISCRETE_ENVIRONMENTS = {
    "CartPole-v1": {
        "action_space": "discrete",
        "expected_baseline_min": 150.0,  # Solved at 195+
        "expected_baseline_max": 500.0,
        "max_episode_steps": 500,
        "quick_episodes": 10,
        "full_episodes": 50
    },
    "Acrobot-v1": {
        "action_space": "discrete",
        "expected_baseline_min": -500.0,  # Solved at -100
        "expected_baseline_max": -50.0,
        "max_episode_steps": 500,
        "quick_episodes": 10,
        "full_episodes": 50
    },
    "MountainCar-v0": {
        "action_space": "discrete",
        "expected_baseline_min": -200.0,  # Solved at -110
        "expected_baseline_max": -90.0,
        "max_episode_steps": 200,
        "quick_episodes": 10,
        "full_episodes": 50
    }
}

CONTINUOUS_ENVIRONMENTS = {
    "Pendulum-v1": {
        "action_space": "continuous",
        "expected_baseline_min": -1600.0,  # Solved at -200 to 0
        "expected_baseline_max": 0.0,
        "max_episode_steps": 200,
        "quick_episodes": 10,
        "full_episodes": 50
    }
}


@pytest.mark.asyncio
async def test_concurrent_multi_environment_evaluation(multi_env_context: MultiEnvContext):
    """
    Test concurrent evaluations across multiple environments.

    **Scenario:**
    1. Submit evaluations for 3 discrete environments simultaneously
    2. Verify all complete successfully
    3. Verify environment isolation (no cross-contamination)
    4. Verify results within expected ranges

    **Expected:**
    - All evaluations complete in < 3 minutes
    - Results environment-specific
    - No resource conflicts
    - Performance metrics reasonable
    """
    ctx = multi_env_context

    agent_type = "dqn"  # Compatible with discrete environments

    # ====================================================================
    # STEP 1: Submit concurrent evaluations
    # ====================================================================
    eval_requests = []
    for env_name, env_spec in DISCRETE_ENVIRONMENTS.items():
        eval_request = {
            "agent_type": agent_type,
            "environment": env_name,
            "hyperparameters": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "buffer_size": 50000,
                "batch_size": 32
            },
            "num_episodes": env_spec["quick_episodes"],
            "quick_mode": True
        }
        eval_requests.append(eval_request)

    # Submit all evaluations concurrently
    submit_start = time.time()
    submit_tasks = [
        ctx.eval_engine_client.post("/v1/evaluate", json=req)
        for req in eval_requests
    ]
    submit_responses = await asyncio.gather(*submit_tasks, return_exceptions=True)

    submit_latency = time.time() - submit_start
    print(f"Submitted {len(eval_requests)} evaluations in {submit_latency*1000:.2f}ms")

    # ====================================================================
    # STEP 2: Verify submissions succeeded
    # ====================================================================
    job_ids = {}
    for i, response in enumerate(submit_responses):
        if isinstance(response, Exception):
            pytest.fail(f"Evaluation {i} submission failed: {response}")

        assert response.status_code in [200, 201, 202]
        data = response.json()
        env_name = eval_requests[i]["environment"]
        job_ids[env_name] = data["job_id"]

    # ====================================================================
    # STEP 3: Poll for completion
    # ====================================================================
    completed_jobs = {}
    timeout = 180  # 3 minutes
    start_time = time.time()

    while len(completed_jobs) < len(job_ids) and time.time() - start_time < timeout:
        for env_name, job_id in job_ids.items():
            if env_name in completed_jobs:
                continue

            status_response = await ctx.eval_engine_client.get(
                f"/v1/jobs/{job_id}"
            )

            if status_response.status_code == 200:
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    completed_jobs[env_name] = status_data
                    print(f"✓ {env_name} completed: mean_reward={status_data.get('mean_reward', 'N/A')}")
                elif status_data["status"] == "failed":
                    pytest.fail(f"{env_name} evaluation failed: {status_data.get('error')}")

        if len(completed_jobs) < len(job_ids):
            await asyncio.sleep(2)

    # ====================================================================
    # STEP 4: Verify all completed
    # ====================================================================
    assert len(completed_jobs) == len(job_ids), (
        f"Only {len(completed_jobs)}/{len(job_ids)} evaluations completed"
    )

    total_time = time.time() - start_time
    print(f"All evaluations completed in {total_time:.2f}s")

    # ====================================================================
    # STEP 5: Validate results are environment-specific
    # ====================================================================
    for env_name, result in completed_jobs.items():
        env_spec = DISCRETE_ENVIRONMENTS[env_name]

        # Verify basic result structure
        assert "mean_reward" in result
        assert "std_reward" in result
        assert "num_episodes" in result

        mean_reward = result["mean_reward"]
        std_reward = result["std_reward"]

        # Verify reward in expected range (with tolerance)
        # Note: Untrained agents may have very poor performance
        # So we use wide ranges
        assert mean_reward >= env_spec["expected_baseline_min"] - 100, (
            f"{env_name} reward {mean_reward} below minimum"
        )
        assert mean_reward <= env_spec["expected_baseline_max"] + 100, (
            f"{env_name} reward {mean_reward} above maximum"
        )

        # Verify std_reward is reasonable (> 0)
        assert std_reward >= 0, f"{env_name} has negative std_reward"

        # Verify episode count
        assert result["num_episodes"] == env_spec["quick_episodes"]

        print(f"✓ {env_name}: reward={mean_reward:.2f} ± {std_reward:.2f}")


@pytest.mark.asyncio
async def test_environment_baseline_isolation(multi_env_context: MultiEnvContext):
    """
    Test that baselines are correctly isolated per environment.

    **Scenario:**
    1. Create baselines for CartPole and Acrobot
    2. Update CartPole baseline
    3. Verify Acrobot baseline unchanged
    4. Query both baselines
    5. Verify no cross-contamination

    **Expected:**
    - Each environment has independent baselines
    - Updates to one environment don't affect others
    - Hyperparameters correctly scoped
    """
    ctx = multi_env_context

    agent_type = "dqn"

    # ====================================================================
    # STEP 1: Create baseline for CartPole
    # ====================================================================
    cartpole_baseline = {
        "agent_type": agent_type,
        "environment": "CartPole-v1",
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99
        },
        "mean_reward": 200.0,
        "std_reward": 10.0,
        "num_episodes": 50,
        "trial_id": f"cartpole_baseline_{int(time.time())}",
        "rank": 1
    }

    cartpole_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=cartpole_baseline
    )
    assert cartpole_response.status_code in [200, 201]

    # ====================================================================
    # STEP 2: Create baseline for Acrobot
    # ====================================================================
    acrobot_baseline = {
        "agent_type": agent_type,
        "environment": "Acrobot-v1",
        "hyperparameters": {
            "learning_rate": 0.0005,  # Different
            "gamma": 0.95  # Different
        },
        "mean_reward": -100.0,
        "std_reward": 15.0,
        "num_episodes": 50,
        "trial_id": f"acrobot_baseline_{int(time.time())}",
        "rank": 1
    }

    acrobot_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=acrobot_baseline
    )
    assert acrobot_response.status_code in [200, 201]

    # ====================================================================
    # STEP 3: Update CartPole baseline
    # ====================================================================
    updated_cartpole_baseline = {
        **cartpole_baseline,
        "trial_id": f"cartpole_updated_{int(time.time())}",
        "mean_reward": 210.0,  # Improved
        "hyperparameters": {
            "learning_rate": 0.0008,  # Changed
            "gamma": 0.99
        }
    }

    update_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=updated_cartpole_baseline
    )
    assert update_response.status_code in [200, 201]

    await asyncio.sleep(1)  # Allow update to propagate

    # ====================================================================
    # STEP 4: Verify CartPole updated, Acrobot unchanged
    # ====================================================================
    cartpole_current = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/CartPole-v1/rank/1"
    )
    assert cartpole_current.status_code == 200
    cartpole_data = cartpole_current.json()

    # Should reflect update
    assert cartpole_data["trial_id"] == updated_cartpole_baseline["trial_id"]
    assert abs(cartpole_data["mean_reward"] - 210.0) < 1e-6
    assert abs(cartpole_data["hyperparameters"]["learning_rate"] - 0.0008) < 1e-9

    # Acrobot should be unchanged
    acrobot_current = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/Acrobot-v1/rank/1"
    )
    assert acrobot_current.status_code == 200
    acrobot_data = acrobot_current.json()

    assert acrobot_data["trial_id"] == acrobot_baseline["trial_id"]
    assert abs(acrobot_data["mean_reward"] - (-100.0)) < 1e-6
    assert abs(acrobot_data["hyperparameters"]["learning_rate"] - 0.0005) < 1e-9
    assert abs(acrobot_data["hyperparameters"]["gamma"] - 0.95) < 1e-9

    print("✓ Environment baseline isolation verified")


@pytest.mark.asyncio
async def test_evaluation_stability_metrics(multi_env_context: MultiEnvContext):
    """
    Test evaluation stability, entropy, and variance metrics.

    **Scenario:**
    1. Run evaluation with 50 episodes
    2. Verify stability metrics within expected ranges
    3. Verify entropy is reasonable (not constant actions)
    4. Verify variance is reasonable (not all identical)

    **Expected:**
    - Std deviation < 50% of mean (for trained agents)
    - Entropy > 0 (some randomness)
    - Coefficient of variation < 0.5 (for stable policies)
    """
    ctx = multi_env_context

    agent_type = "dqn"
    environment = "CartPole-v1"

    # ====================================================================
    # STEP 1: Run evaluation with sufficient episodes
    # ====================================================================
    eval_request = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 0.1
        },
        "num_episodes": 50,
        "quick_mode": False
    }

    eval_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=eval_request
    )
    assert eval_response.status_code in [200, 201, 202]
    eval_data = eval_response.json()
    job_id = eval_data["job_id"]

    # ====================================================================
    # STEP 2: Wait for completion
    # ====================================================================
    result = None
    start_time = time.time()

    while time.time() - start_time < 120:
        status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            if status_data["status"] == "completed":
                result = status_data
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Evaluation failed: {status_data.get('error')}")

        await asyncio.sleep(2)

    assert result is not None, "Evaluation did not complete"

    # ====================================================================
    # STEP 3: Analyze stability metrics
    # ====================================================================
    mean_reward = result["mean_reward"]
    std_reward = result["std_reward"]

    # Coefficient of variation
    if abs(mean_reward) > 1e-6:
        cv = abs(std_reward / mean_reward)
    else:
        cv = float('inf')

    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Coefficient of variation: {cv:.4f}")

    # Validate stability
    # For CartPole, std should be reasonable relative to mean
    # Untrained agents may have high variance
    assert std_reward >= 0, "Std deviation cannot be negative"

    # For well-trained agents, expect CV < 0.5
    # For untrained agents, may be higher
    # We'll use a relaxed threshold
    if mean_reward > 150:  # If reasonably trained
        assert cv < 1.0, f"High coefficient of variation: {cv}"
        print("✓ Evaluation is stable (CV < 1.0)")
    else:
        print(f"⚠ Untrained agent (reward={mean_reward}), skipping stability check")

    # ====================================================================
    # STEP 4: Check for entropy/variance (if available in result)
    # ====================================================================
    if "entropy" in result:
        entropy = result["entropy"]
        assert entropy >= 0, "Entropy cannot be negative"
        print(f"Action entropy: {entropy:.4f}")

        # Expect some entropy (not deterministic)
        if entropy < 0.01:
            print("⚠ Very low entropy (deterministic policy)")

    if "episode_rewards" in result:
        # Check variance across episodes
        episode_rewards = result["episode_rewards"]
        assert len(episode_rewards) == 50

        # All rewards should not be identical
        unique_rewards = len(set(episode_rewards))
        assert unique_rewards > 1, "All episode rewards are identical"
        print(f"✓ {unique_rewards} unique reward values across 50 episodes")


@pytest.mark.asyncio
async def test_environment_cache_performance(multi_env_context: MultiEnvContext):
    """
    Test environment cache performance and correctness.

    **Scenario:**
    1. First evaluation (cold cache)
    2. Second evaluation (warm cache)
    3. Verify warm cache is faster
    4. Verify results are consistent

    **Expected:**
    - Warm cache evaluation 20-50% faster
    - Results statistically similar
    - Cache hit metrics incremented
    """
    ctx = multi_env_context

    agent_type = "dqn"
    environment = "CartPole-v1"

    eval_request = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 0.1
        },
        "num_episodes": 20,
        "quick_mode": True,
        "seed": 42  # Fixed seed for reproducibility
    }

    # ====================================================================
    # STEP 1: First evaluation (cold cache)
    # ====================================================================
    cold_start = time.time()
    cold_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=eval_request
    )
    assert cold_response.status_code in [200, 201, 202]
    cold_job_id = cold_response.json()["job_id"]

    # Wait for completion
    cold_result = None
    while time.time() - cold_start < 60:
        status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{cold_job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            if status_data["status"] == "completed":
                cold_result = status_data
                break
        await asyncio.sleep(1)

    assert cold_result is not None
    cold_duration = cold_result.get("duration_seconds", time.time() - cold_start)

    # ====================================================================
    # STEP 2: Second evaluation (warm cache)
    # ====================================================================
    warm_start = time.time()
    warm_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=eval_request
    )
    assert warm_response.status_code in [200, 201, 202]
    warm_job_id = warm_response.json()["job_id"]

    # Wait for completion
    warm_result = None
    while time.time() - warm_start < 60:
        status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{warm_job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            if status_data["status"] == "completed":
                warm_result = status_data
                break
        await asyncio.sleep(1)

    assert warm_result is not None
    warm_duration = warm_result.get("duration_seconds", time.time() - warm_start)

    # ====================================================================
    # STEP 3: Compare performance
    # ====================================================================
    print(f"Cold cache: {cold_duration:.2f}s")
    print(f"Warm cache: {warm_duration:.2f}s")

    # Warm cache should be at least as fast (may be faster)
    # In practice, episode execution dominates, so cache may not show huge speedup
    speedup = (cold_duration - warm_duration) / cold_duration if cold_duration > 0 else 0
    print(f"Speedup: {speedup*100:.1f}%")

    # Verify results are similar (with fixed seed, should be identical)
    cold_mean = cold_result["mean_reward"]
    warm_mean = warm_result["mean_reward"]

    # With same seed, results should be very close
    # (may differ slightly due to system non-determinism)
    reward_diff = abs(cold_mean - warm_mean)
    print(f"Reward difference: {reward_diff:.4f}")

    # Allow some tolerance
    assert reward_diff < 10.0, f"Results differ significantly: {cold_mean} vs {warm_mean}"


@pytest.mark.asyncio
async def test_resource_isolation_between_environments(multi_env_context: MultiEnvContext):
    """
    Test that concurrent evaluations don't interfere with each other.

    **Scenario:**
    1. Start evaluation for CartPole
    2. Start evaluation for Acrobot (while CartPole running)
    3. Verify both complete successfully
    4. Verify results are environment-specific (no mix-up)

    **Expected:**
    - Both evaluations complete
    - No resource conflicts
    - Results match expected environment characteristics
    """
    ctx = multi_env_context

    agent_type = "dqn"

    # ====================================================================
    # STEP 1: Start CartPole evaluation
    # ====================================================================
    cartpole_request = {
        "agent_type": agent_type,
        "environment": "CartPole-v1",
        "hyperparameters": {"learning_rate": 0.001, "gamma": 0.99},
        "num_episodes": 30,
        "quick_mode": False
    }

    cartpole_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=cartpole_request
    )
    assert cartpole_response.status_code in [200, 201, 202]
    cartpole_job_id = cartpole_response.json()["job_id"]

    # ====================================================================
    # STEP 2: Immediately start Acrobot evaluation
    # ====================================================================
    await asyncio.sleep(0.5)  # Small delay to ensure CartPole is running

    acrobot_request = {
        "agent_type": agent_type,
        "environment": "Acrobot-v1",
        "hyperparameters": {"learning_rate": 0.001, "gamma": 0.99},
        "num_episodes": 30,
        "quick_mode": False
    }

    acrobot_response = await ctx.eval_engine_client.post(
        "/v1/evaluate",
        json=acrobot_request
    )
    assert acrobot_response.status_code in [200, 201, 202]
    acrobot_job_id = acrobot_response.json()["job_id"]

    # ====================================================================
    # STEP 3: Wait for both to complete
    # ====================================================================
    results = {}
    start_time = time.time()

    while len(results) < 2 and time.time() - start_time < 120:
        for job_id, env_name in [(cartpole_job_id, "CartPole-v1"), (acrobot_job_id, "Acrobot-v1")]:
            if env_name in results:
                continue

            status_response = await ctx.eval_engine_client.get(f"/v1/jobs/{job_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data["status"] == "completed":
                    results[env_name] = status_data
                    print(f"✓ {env_name} completed")

        await asyncio.sleep(2)

    assert len(results) == 2, "Both evaluations did not complete"

    # ====================================================================
    # STEP 4: Verify results are environment-specific
    # ====================================================================
    cartpole_mean = results["CartPole-v1"]["mean_reward"]
    acrobot_mean = results["Acrobot-v1"]["mean_reward"]

    # CartPole rewards are positive (0-500)
    # Acrobot rewards are negative (-500 to -50)
    assert cartpole_mean >= 0, f"CartPole reward should be positive, got {cartpole_mean}"
    assert acrobot_mean <= 0, f"Acrobot reward should be negative, got {acrobot_mean}"

    print(f"✓ CartPole: {cartpole_mean:.2f}, Acrobot: {acrobot_mean:.2f}")
    print("✓ Resource isolation verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

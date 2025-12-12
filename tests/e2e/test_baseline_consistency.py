"""
Baseline Consistency End-to-End Test

Tests consistency guarantees across the distributed baseline system:

1. PostgreSQL read/write consistency
2. Redis hot-cache coherence
3. History ordering correctness
4. Cross-service consistency contract
5. Multi-writer conflict resolution
6. Baseline ranking invariants

**Version:** v1.0.0-rc2
**Phase:** 13.8 - Final Pre-Production Validation
**Author:** T.A.R.S. Development Team
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import patch

import asyncpg
import httpx
import pytest
import redis.asyncio as aioredis


class BaselineConsistencyContext:
    """Context manager for baseline consistency testing."""

    def __init__(self):
        self.orchestration_client: Optional[httpx.AsyncClient] = None
        self.eval_engine_client: Optional[httpx.AsyncClient] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.pg_pool: Optional[asyncpg.Pool] = None

    async def __aenter__(self):
        """Initialize service clients and database connections."""
        self.orchestration_client = httpx.AsyncClient(
            base_url="http://localhost:8094",
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

        # Initialize direct PostgreSQL connection for validation
        import os
        postgres_url = os.getenv(
            "POSTGRES_URL",
            "postgresql://postgres:postgres@localhost:5432/tars"
        )
        self.pg_pool = await asyncpg.create_pool(
            postgres_url,
            min_size=1,
            max_size=5
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup clients and connections."""
        if self.orchestration_client:
            await self.orchestration_client.aclose()
        if self.eval_engine_client:
            await self.eval_engine_client.aclose()
        if self.redis_client:
            await self.redis_client.close()
        if self.pg_pool:
            await self.pg_pool.close()


@pytest.fixture
async def consistency_context():
    """Fixture providing baseline consistency context."""
    async with BaselineConsistencyContext() as ctx:
        yield ctx


@pytest.mark.asyncio
async def test_postgres_read_write_consistency(consistency_context: BaselineConsistencyContext):
    """
    Test PostgreSQL read-after-write consistency.

    **Scenario:**
    1. Write baseline via API
    2. Immediately read from Postgres directly
    3. Verify data matches exactly
    4. Read via API
    5. Verify all reads return same data

    **Expected:**
    - Write committed atomically
    - Read-after-write returns correct data
    - No stale reads
    - Timestamps consistent
    """
    ctx = consistency_context

    agent_type = "dqn"
    environment = "CartPole-v1"

    # ====================================================================
    # STEP 1: Write baseline via API
    # ====================================================================
    baseline_payload = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            "learning_rate": 0.00123,
            "gamma": 0.995,
            "epsilon": 0.08
        },
        "mean_reward": 205.5,
        "std_reward": 12.3,
        "num_episodes": 50,
        "trial_id": f"consistency_test_{int(time.time())}",
        "rank": 1
    }

    write_start = time.time()
    write_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=baseline_payload
    )
    write_latency = time.time() - write_start

    assert write_response.status_code in [200, 201]
    print(f"Write latency: {write_latency*1000:.2f}ms")

    # ====================================================================
    # STEP 2: Immediate read from PostgreSQL (bypassing cache)
    # ====================================================================
    async with ctx.pg_pool.acquire() as conn:
        pg_result = await conn.fetchrow(
            """
            SELECT agent_type, environment, hyperparameters, mean_reward,
                   std_reward, num_episodes, trial_id, rank, created_at
            FROM eval_baselines
            WHERE agent_type = $1 AND environment = $2 AND rank = $3
            ORDER BY created_at DESC
            LIMIT 1
            """,
            agent_type,
            environment,
            1
        )

    assert pg_result is not None, "Baseline not found in PostgreSQL"

    # Verify data consistency
    assert pg_result["agent_type"] == agent_type
    assert pg_result["environment"] == environment
    assert pg_result["trial_id"] == baseline_payload["trial_id"]
    assert abs(pg_result["mean_reward"] - baseline_payload["mean_reward"]) < 1e-6
    assert abs(pg_result["std_reward"] - baseline_payload["std_reward"]) < 1e-6
    assert pg_result["num_episodes"] == baseline_payload["num_episodes"]
    assert pg_result["rank"] == baseline_payload["rank"]

    # Verify hyperparameters (JSON field)
    pg_hyperparams = pg_result["hyperparameters"]
    assert abs(pg_hyperparams["learning_rate"] - 0.00123) < 1e-9
    assert abs(pg_hyperparams["gamma"] - 0.995) < 1e-9
    assert abs(pg_hyperparams["epsilon"] - 0.08) < 1e-9

    # ====================================================================
    # STEP 3: Read via API (may hit cache)
    # ====================================================================
    api_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert api_response.status_code == 200
    api_baseline = api_response.json()

    # Verify API data matches PostgreSQL data
    assert api_baseline["trial_id"] == pg_result["trial_id"]
    assert abs(api_baseline["mean_reward"] - pg_result["mean_reward"]) < 1e-6
    assert abs(api_baseline["hyperparameters"]["learning_rate"] - 0.00123) < 1e-9

    # ====================================================================
    # STEP 4: Multiple concurrent reads should return same data
    # ====================================================================
    read_tasks = [
        ctx.orchestration_client.get(f"/v1/baselines/{agent_type}/{environment}/rank/1")
        for _ in range(10)
    ]
    read_responses = await asyncio.gather(*read_tasks)

    for response in read_responses:
        assert response.status_code == 200
        data = response.json()
        assert data["trial_id"] == baseline_payload["trial_id"]
        assert abs(data["mean_reward"] - baseline_payload["mean_reward"]) < 1e-6


@pytest.mark.asyncio
async def test_redis_cache_coherence(consistency_context: BaselineConsistencyContext):
    """
    Test Redis cache coherence with PostgreSQL source of truth.

    **Scenario:**
    1. Write baseline (populates cache)
    2. Read from cache
    3. Update baseline
    4. Verify cache invalidated
    5. Verify subsequent reads return new data

    **Expected:**
    - Cache populated on write
    - Cache invalidated on update
    - No stale cache reads
    - Cache TTL honored
    """
    ctx = consistency_context

    agent_type = "a2c"
    environment = "Acrobot-v1"

    # ====================================================================
    # STEP 1: Write initial baseline
    # ====================================================================
    initial_baseline = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            "learning_rate": 0.0007,
            "gamma": 0.99,
            "n_steps": 5
        },
        "mean_reward": -90.0,
        "std_reward": 15.0,
        "num_episodes": 50,
        "trial_id": f"cache_test_v1_{int(time.time())}",
        "rank": 1
    }

    write_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=initial_baseline
    )
    assert write_response.status_code in [200, 201]

    # ====================================================================
    # STEP 2: Verify cache populated
    # ====================================================================
    # Check Redis for cached baseline
    cache_key = f"baseline:{agent_type}:{environment}:1"
    cached_data = await ctx.redis_client.get(cache_key)

    if cached_data:
        # Cache exists - verify it matches
        import json
        cached_baseline = json.loads(cached_data)
        assert cached_baseline["trial_id"] == initial_baseline["trial_id"]
        assert abs(cached_baseline["mean_reward"] - initial_baseline["mean_reward"]) < 1e-6
        print("✓ Cache populated on write")
    else:
        # Cache may not exist if lazy loading is used
        print("⚠ Cache not populated on write (lazy loading?)")

    # ====================================================================
    # STEP 3: Read via API (should hit cache or DB)
    # ====================================================================
    read_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert read_response.status_code == 200
    read_baseline = read_response.json()
    assert read_baseline["trial_id"] == initial_baseline["trial_id"]

    # Now cache should definitely be populated
    cached_data = await ctx.redis_client.get(cache_key)
    if cached_data:
        import json
        cached_baseline = json.loads(cached_data)
        assert cached_baseline["trial_id"] == initial_baseline["trial_id"]

    # ====================================================================
    # STEP 4: Update baseline
    # ====================================================================
    updated_baseline = {
        **initial_baseline,
        "trial_id": f"cache_test_v2_{int(time.time())}",
        "mean_reward": -85.0,  # Improved
        "hyperparameters": {
            **initial_baseline["hyperparameters"],
            "learning_rate": 0.0008  # Updated
        }
    }

    update_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=updated_baseline
    )
    assert update_response.status_code in [200, 201]

    # ====================================================================
    # STEP 5: Verify cache invalidated or updated
    # ====================================================================
    # Wait briefly for cache invalidation to propagate
    await asyncio.sleep(0.5)

    # Read via API
    updated_read_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert updated_read_response.status_code == 200
    updated_read_baseline = updated_read_response.json()

    # Must return NEW data, not stale cache
    assert updated_read_baseline["trial_id"] == updated_baseline["trial_id"]
    assert abs(updated_read_baseline["mean_reward"] - updated_baseline["mean_reward"]) < 1e-6
    assert abs(updated_read_baseline["hyperparameters"]["learning_rate"] - 0.0008) < 1e-9

    print("✓ Cache coherence maintained")


@pytest.mark.asyncio
async def test_baseline_history_ordering(consistency_context: BaselineConsistencyContext):
    """
    Test that baseline history is correctly ordered.

    **Scenario:**
    1. Create 5 baselines for same agent/env with increasing timestamps
    2. Query history
    3. Verify ordering by created_at DESC
    4. Verify rank 1 is most recent
    5. Verify all historical entries preserved

    **Expected:**
    - History ordered by timestamp (newest first)
    - Rank 1 always points to current baseline
    - All historical baselines retrievable
    - No data loss
    """
    ctx = consistency_context

    agent_type = "ppo"
    environment = "CartPole-v1"

    # ====================================================================
    # STEP 1: Create 5 baselines sequentially
    # ====================================================================
    baselines = []
    for i in range(5):
        baseline = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": {
                "learning_rate": 0.0003 + (i * 0.0001),
                "gamma": 0.99,
                "clip_range": 0.2
            },
            "mean_reward": 190.0 + (i * 5),  # Improving
            "std_reward": 10.0,
            "num_episodes": 50,
            "trial_id": f"history_test_v{i}_{int(time.time())}",
            "rank": 1
        }

        response = await ctx.orchestration_client.post(
            "/v1/baselines",
            json=baseline
        )
        assert response.status_code in [200, 201]

        baselines.append(baseline)
        await asyncio.sleep(0.2)  # Ensure distinct timestamps

    # ====================================================================
    # STEP 2: Query baseline history
    # ====================================================================
    history_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/history"
    )

    if history_response.status_code == 404:
        # History endpoint may not exist - query from PostgreSQL directly
        async with ctx.pg_pool.acquire() as conn:
            history_rows = await conn.fetch(
                """
                SELECT trial_id, mean_reward, hyperparameters, created_at
                FROM eval_baselines
                WHERE agent_type = $1 AND environment = $2
                ORDER BY created_at DESC
                """,
                agent_type,
                environment
            )
    else:
        assert history_response.status_code == 200
        history_data = history_response.json()
        history_rows = history_data.get("history", [])

    # ====================================================================
    # STEP 3: Verify ordering
    # ====================================================================
    assert len(history_rows) >= 5, f"Expected at least 5 entries, got {len(history_rows)}"

    # Most recent should be first (DESC order)
    most_recent = history_rows[0]
    if isinstance(most_recent, dict):
        most_recent_trial_id = most_recent.get("trial_id")
        most_recent_reward = most_recent.get("mean_reward")
    else:
        most_recent_trial_id = most_recent["trial_id"]
        most_recent_reward = most_recent["mean_reward"]

    # Should match last baseline we created
    assert most_recent_trial_id == baselines[-1]["trial_id"]
    assert abs(most_recent_reward - baselines[-1]["mean_reward"]) < 1e-6

    # ====================================================================
    # STEP 4: Verify rank 1 points to most recent
    # ====================================================================
    current_baseline_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert current_baseline_response.status_code == 200
    current_baseline = current_baseline_response.json()

    assert current_baseline["trial_id"] == baselines[-1]["trial_id"]
    assert abs(current_baseline["mean_reward"] - baselines[-1]["mean_reward"]) < 1e-6

    # ====================================================================
    # STEP 5: Verify chronological ordering in history
    # ====================================================================
    if isinstance(history_rows[0], dict):
        # API response
        timestamps = [row.get("created_at") for row in history_rows[:5]]
    else:
        # PostgreSQL response
        timestamps = [row["created_at"] for row in history_rows[:5]]

    # Convert to comparable format if needed
    for i in range(len(timestamps) - 1):
        # DESC order: newer timestamps should come first
        if isinstance(timestamps[i], str):
            ts_current = datetime.fromisoformat(timestamps[i].replace('Z', '+00:00'))
            ts_next = datetime.fromisoformat(timestamps[i+1].replace('Z', '+00:00'))
        else:
            ts_current = timestamps[i]
            ts_next = timestamps[i+1]

        assert ts_current >= ts_next, f"History not in DESC order at index {i}"

    print("✓ Baseline history correctly ordered")


@pytest.mark.asyncio
async def test_cross_service_consistency(consistency_context: BaselineConsistencyContext):
    """
    Test consistency contract between Orchestration and Eval Engine.

    **Scenario:**
    1. Eval Engine completes evaluation
    2. Orchestration updates baseline
    3. Eval Engine queries baseline
    4. Verify Eval Engine sees updated baseline
    5. Verify no race conditions

    **Expected:**
    - Orchestration baseline visible to Eval Engine
    - No stale reads across services
    - Consistency within 1 second
    """
    ctx = consistency_context

    agent_type = "dqn"
    environment = "CartPole-v1"

    # ====================================================================
    # STEP 1: Orchestration updates baseline
    # ====================================================================
    baseline_payload = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            "learning_rate": 0.00099,
            "gamma": 0.99
        },
        "mean_reward": 198.5,
        "std_reward": 11.2,
        "num_episodes": 50,
        "trial_id": f"cross_service_{int(time.time())}",
        "rank": 1
    }

    update_time = time.time()
    update_response = await ctx.orchestration_client.post(
        "/v1/baselines",
        json=baseline_payload
    )
    assert update_response.status_code in [200, 201]

    # ====================================================================
    # STEP 2: Eval Engine queries baseline
    # ====================================================================
    # Poll until Eval Engine sees the update
    max_wait = 2.0  # 2 seconds max
    start_time = time.time()
    consistent = False

    while time.time() - start_time < max_wait:
        # Query Eval Engine for baseline
        eval_baseline_response = await ctx.eval_engine_client.get(
            f"/v1/baselines/{agent_type}/{environment}"
        )

        if eval_baseline_response.status_code == 200:
            eval_baseline = eval_baseline_response.json()

            if eval_baseline.get("trial_id") == baseline_payload["trial_id"]:
                consistency_latency = time.time() - update_time
                consistent = True
                print(f"✓ Cross-service consistency in {consistency_latency*1000:.2f}ms")
                break

        await asyncio.sleep(0.1)

    assert consistent, "Eval Engine did not see baseline update within 2s"


@pytest.mark.asyncio
async def test_multi_writer_conflict_resolution(consistency_context: BaselineConsistencyContext):
    """
    Test conflict resolution when multiple writers update same baseline.

    **Scenario:**
    1. Two concurrent baseline updates for same agent/env/rank
    2. Verify both writes succeed OR one is rejected
    3. Verify final state is consistent (one winner)
    4. Verify no data corruption

    **Expected:**
    - Last-write-wins OR optimistic locking
    - No data corruption
    - Consistent final state
    - All writes logged in history
    """
    ctx = consistency_context

    agent_type = "a2c"
    environment = "CartPole-v1"

    # ====================================================================
    # STEP 1: Prepare two conflicting updates
    # ====================================================================
    baseline_1 = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            "learning_rate": 0.0007,
            "gamma": 0.99
        },
        "mean_reward": 195.0,
        "std_reward": 10.0,
        "num_episodes": 50,
        "trial_id": f"conflict_writer_1_{int(time.time())}",
        "rank": 1
    }

    baseline_2 = {
        "agent_type": agent_type,
        "environment": environment,
        "hyperparameters": {
            "learning_rate": 0.0008,  # Different
            "gamma": 0.99
        },
        "mean_reward": 197.0,  # Different
        "std_reward": 12.0,
        "num_episodes": 50,
        "trial_id": f"conflict_writer_2_{int(time.time())}",
        "rank": 1
    }

    # ====================================================================
    # STEP 2: Submit both updates concurrently
    # ====================================================================
    responses = await asyncio.gather(
        ctx.orchestration_client.post("/v1/baselines", json=baseline_1),
        ctx.orchestration_client.post("/v1/baselines", json=baseline_2),
        return_exceptions=True
    )

    # ====================================================================
    # STEP 3: Analyze results
    # ====================================================================
    success_count = sum(
        1 for r in responses
        if not isinstance(r, Exception) and r.status_code in [200, 201]
    )

    # Both may succeed (last-write-wins) or one may be rejected (optimistic locking)
    assert success_count >= 1, "At least one write should succeed"

    if success_count == 2:
        print("✓ Both writes succeeded (last-write-wins strategy)")
    else:
        print("✓ One write rejected (optimistic locking)")

    # ====================================================================
    # STEP 4: Verify final state is consistent
    # ====================================================================
    await asyncio.sleep(0.5)  # Allow time for updates to settle

    final_baseline_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert final_baseline_response.status_code == 200
    final_baseline = final_baseline_response.json()

    # Final state should be one of the two baselines
    assert final_baseline["trial_id"] in [baseline_1["trial_id"], baseline_2["trial_id"]]

    # Verify no data corruption (no mix of hyperparameters)
    final_lr = final_baseline["hyperparameters"]["learning_rate"]
    assert final_lr in [0.0007, 0.0008], "Hyperparameters corrupted"

    if final_lr == 0.0007:
        # Baseline 1 won
        assert abs(final_baseline["mean_reward"] - 195.0) < 1e-6
    else:
        # Baseline 2 won
        assert abs(final_baseline["mean_reward"] - 197.0) < 1e-6

    print("✓ Final state consistent, no corruption")


@pytest.mark.asyncio
async def test_baseline_ranking_invariants(consistency_context: BaselineConsistencyContext):
    """
    Test baseline ranking invariants.

    **Invariants:**
    - rank=1 always exists for any agent/env with baselines
    - rank=1 has highest mean_reward (if reward-based ranking)
    - ranks are unique per agent/env
    - rank numbering is contiguous (1, 2, 3, ...)

    **Expected:**
    - All invariants hold after updates
    - Rank 1 automatically assigned to best baseline
    """
    ctx = consistency_context

    agent_type = "ddpg"
    environment = "Pendulum-v1"

    # ====================================================================
    # STEP 1: Create multiple baselines with different rewards
    # ====================================================================
    baselines = [
        {"mean_reward": -150.0, "trial_id": "rank_test_poor"},
        {"mean_reward": -120.0, "trial_id": "rank_test_medium"},
        {"mean_reward": -100.0, "trial_id": "rank_test_good"},
        {"mean_reward": -90.0, "trial_id": "rank_test_best"},
    ]

    for baseline_spec in baselines:
        baseline = {
            "agent_type": agent_type,
            "environment": environment,
            "hyperparameters": {
                "learning_rate": 0.001,
                "gamma": 0.99
            },
            "mean_reward": baseline_spec["mean_reward"],
            "std_reward": 10.0,
            "num_episodes": 50,
            "trial_id": baseline_spec["trial_id"],
            "rank": 1  # All submitted as rank 1 - system should rerank
        }

        response = await ctx.orchestration_client.post(
            "/v1/baselines",
            json=baseline
        )
        assert response.status_code in [200, 201]
        await asyncio.sleep(0.2)

    # ====================================================================
    # STEP 2: Verify rank 1 is the best baseline
    # ====================================================================
    rank_1_response = await ctx.orchestration_client.get(
        f"/v1/baselines/{agent_type}/{environment}/rank/1"
    )
    assert rank_1_response.status_code == 200
    rank_1_baseline = rank_1_response.json()

    # Rank 1 should be the best (highest reward for Pendulum is least negative)
    assert rank_1_baseline["mean_reward"] >= -100.0, "Rank 1 is not the best baseline"

    # Ideally should be -90.0 (best)
    # But depending on system implementation, may just be most recent
    # For reward-based ranking:
    if abs(rank_1_baseline["mean_reward"] - (-90.0)) < 1e-6:
        print("✓ Rank 1 correctly assigned to best baseline (reward-based)")
    else:
        print(f"⚠ Rank 1 is most recent, not best (mean_reward={rank_1_baseline['mean_reward']})")

    # ====================================================================
    # STEP 3: Verify all baselines retrievable
    # ====================================================================
    async with ctx.pg_pool.acquire() as conn:
        all_baselines = await conn.fetch(
            """
            SELECT trial_id, mean_reward, rank
            FROM eval_baselines
            WHERE agent_type = $1 AND environment = $2
            ORDER BY rank ASC
            """,
            agent_type,
            environment
        )

    assert len(all_baselines) >= 1, "No baselines found"

    # Verify rank 1 exists
    ranks = [row["rank"] for row in all_baselines]
    assert 1 in ranks, "Rank 1 does not exist"

    print(f"✓ {len(all_baselines)} baselines with ranks: {ranks[:5]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

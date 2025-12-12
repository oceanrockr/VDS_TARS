"""
Baseline Manager - CRUD operations for performance baselines.
"""
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncpg

from models import BaselineRecord, MetricsResult


class BaselineManager:
    """Manage performance baselines in PostgreSQL."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool

    async def get_baseline(
        self,
        agent_type: str,
        environment: str,
        rank: int = 1
    ) -> Optional[BaselineRecord]:
        """
        Get baseline for agent+environment.

        Args:
            agent_type: Agent type (DQN, A2C, PPO, DDPG)
            environment: Environment ID (e.g., "CartPole-v1")
            rank: 1 = best, 2 = second best, etc.

        Returns:
            BaselineRecord if found, None otherwise
        """
        query = """
            SELECT * FROM eval_baselines
            WHERE agent_type = $1 AND environment = $2 AND rank = $3
        """
        row = await self.db.fetchrow(query, agent_type, environment, rank)

        if not row:
            return None

        return BaselineRecord(
            agent_type=row['agent_type'],
            environment=row['environment'],
            mean_reward=float(row['mean_reward']),
            std_reward=float(row['std_reward']),
            success_rate=float(row['success_rate']),
            hyperparameters=row['hyperparameters'] if isinstance(row['hyperparameters'], dict) else json.loads(row['hyperparameters']),
            version=int(row['version']),
            rank=int(row['rank']),
            created_at=row['created_at'].isoformat()
        )

    async def insert_baseline(
        self,
        baseline: BaselineRecord
    ) -> str:
        """
        Insert new baseline, return ID.

        Args:
            baseline: BaselineRecord to insert

        Returns:
            Baseline ID (UUID as string)
        """
        query = """
            INSERT INTO eval_baselines (
                agent_type, environment, mean_reward, std_reward,
                success_rate, hyperparameters, version, rank
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """
        baseline_id = await self.db.fetchval(
            query,
            baseline.agent_type,
            baseline.environment,
            baseline.mean_reward,
            baseline.std_reward,
            baseline.success_rate,
            json.dumps(baseline.hyperparameters),
            baseline.version,
            baseline.rank
        )

        return str(baseline_id)

    async def update_baseline_if_better(
        self,
        agent_type: str,
        environment: str,
        new_metrics: MetricsResult,
        hyperparameters: Dict[str, Any],
        version: int
    ) -> Optional[str]:
        """
        Update baseline if new metrics are better.

        Args:
            agent_type: Agent type
            environment: Environment ID
            new_metrics: New evaluation metrics
            hyperparameters: Hyperparameters used
            version: Agent version

        Returns:
            baseline_id if updated, None otherwise
        """
        current = await self.get_baseline(agent_type, environment)

        if current is None or new_metrics.mean_reward > current.mean_reward:
            # Demote current baseline
            if current:
                await self.rerank_baselines(agent_type, environment)

            # Insert new baseline as rank 1
            new_baseline = BaselineRecord(
                agent_type=agent_type,
                environment=environment,
                mean_reward=new_metrics.mean_reward,
                std_reward=new_metrics.std_reward,
                success_rate=new_metrics.success_rate,
                hyperparameters=hyperparameters,
                version=version,
                rank=1,
                created_at=datetime.utcnow().isoformat()
            )
            return await self.insert_baseline(new_baseline)

        return None

    async def rerank_baselines(
        self,
        agent_type: str,
        environment: str
    ):
        """
        Increment rank of all existing baselines.

        Called when inserting a new rank-1 baseline.
        """
        query = """
            UPDATE eval_baselines
            SET rank = rank + 1
            WHERE agent_type = $1 AND environment = $2
        """
        await self.db.execute(query, agent_type, environment)

    async def get_baseline_history(
        self,
        agent_type: str,
        environment: str,
        limit: int = 10
    ) -> List[BaselineRecord]:
        """
        Get historical baselines ordered by rank.

        Args:
            agent_type: Agent type
            environment: Environment ID
            limit: Max number of baselines to return

        Returns:
            List of BaselineRecord ordered by rank (best first)
        """
        query = """
            SELECT * FROM eval_baselines
            WHERE agent_type = $1 AND environment = $2
            ORDER BY rank ASC
            LIMIT $3
        """
        rows = await self.db.fetch(query, agent_type, environment, limit)

        return [
            BaselineRecord(
                agent_type=row['agent_type'],
                environment=row['environment'],
                mean_reward=float(row['mean_reward']),
                std_reward=float(row['std_reward']),
                success_rate=float(row['success_rate']),
                hyperparameters=row['hyperparameters'] if isinstance(row['hyperparameters'], dict) else json.loads(row['hyperparameters']),
                version=int(row['version']),
                rank=int(row['rank']),
                created_at=row['created_at'].isoformat()
            )
            for row in rows
        ]

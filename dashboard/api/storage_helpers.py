"""
T.A.R.S. Dashboard Storage Helpers
Unified interface for Redis backend with in-memory fallback

Author: T.A.R.S. Cognitive Team
Version: v0.9.4-alpha
"""

from typing import Dict, Any, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class StorageInterface:
    """
    Unified storage interface that automatically uses Redis if available,
    otherwise falls back to in-memory storage.
    """

    def __init__(
        self,
        redis_backend=None,
        fallback_agent_history: Optional[Dict[str, deque]] = None,
        fallback_conflict_history: Optional[deque] = None,
        fallback_nash_history: Optional[deque] = None
    ):
        """
        Initialize storage interface.

        Args:
            redis_backend: RedisBackend instance (or None for in-memory)
            fallback_agent_history: In-memory agent history
            fallback_conflict_history: In-memory conflict history
            fallback_nash_history: In-memory Nash equilibrium history
        """
        self.redis = redis_backend
        self.agent_history_fallback = fallback_agent_history or {}
        self.conflict_history_fallback = fallback_conflict_history or deque(maxlen=500)
        self.nash_history_fallback = fallback_nash_history or deque(maxlen=500)

    @property
    def using_redis(self) -> bool:
        """Check if using Redis backend."""
        return self.redis is not None

    # ============================================================================
    # Agent State Operations
    # ============================================================================

    async def store_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Store agent state."""
        if self.redis:
            await self.redis.store_agent_state(agent_id, state)
        else:
            # In-memory fallback
            if agent_id in self.agent_history_fallback:
                self.agent_history_fallback[agent_id].append(state)

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current agent state."""
        if self.redis:
            return await self.redis.get_agent_state(agent_id)
        else:
            # In-memory fallback (return last item)
            if agent_id in self.agent_history_fallback and self.agent_history_fallback[agent_id]:
                return self.agent_history_fallback[agent_id][-1]
            return None

    async def get_agent_history(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get agent history."""
        if self.redis:
            return await self.redis.get_agent_history(agent_id, limit, offset)
        else:
            # In-memory fallback
            if agent_id in self.agent_history_fallback:
                history = list(self.agent_history_fallback[agent_id])
                return history[offset:offset + limit]
            return []

    # ============================================================================
    # Conflict Operations
    # ============================================================================

    async def store_conflict(self, conflict: Dict[str, Any]):
        """Store conflict event."""
        if self.redis:
            await self.redis.store_conflict(conflict)
        else:
            # In-memory fallback
            self.conflict_history_fallback.append(conflict)

    async def get_conflicts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent conflicts."""
        if self.redis:
            return await self.redis.get_conflicts(limit)
        else:
            # In-memory fallback
            return list(self.conflict_history_fallback)[-limit:]

    # ============================================================================
    # Nash Equilibrium Operations
    # ============================================================================

    async def store_nash_equilibrium(self, equilibrium: Dict[str, Any]):
        """Store Nash equilibrium."""
        if self.redis:
            await self.redis.store_nash_equilibrium(equilibrium)
        else:
            # In-memory fallback
            self.nash_history_fallback.append(equilibrium)

    async def get_nash_equilibrium_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get Nash equilibrium history."""
        if self.redis:
            return await self.redis.get_nash_equilibrium_history(limit)
        else:
            # In-memory fallback
            return list(self.nash_history_fallback)[-limit:]

    # ============================================================================
    # Metrics Operations
    # ============================================================================

    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics."""
        if self.redis:
            await self.redis.update_metrics(metrics)
        # In-memory: no-op (metrics calculated on-the-fly)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        if self.redis:
            return await self.redis.get_metrics()
        # In-memory: return empty (calculated on-the-fly)
        return {}

    # ============================================================================
    # Statistics
    # ============================================================================

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if self.redis:
            stats = await self.redis.get_statistics()
            stats["backend"] = "redis"
            return stats
        else:
            # In-memory statistics
            return {
                "backend": "in_memory",
                "policy_history_length": len(self.agent_history_fallback.get("policy", [])),
                "consensus_history_length": len(self.agent_history_fallback.get("consensus", [])),
                "ethical_history_length": len(self.agent_history_fallback.get("ethical", [])),
                "resource_history_length": len(self.agent_history_fallback.get("resource", [])),
                "conflicts_count": len(self.conflict_history_fallback),
                "equilibrium_count": len(self.nash_history_fallback),
            }

"""
T.A.R.S. Dashboard Redis Backend
Persistent storage for agent states, metrics, and trajectories

Replaces in-memory storage with Redis for scalability and persistence.

Author: T.A.R.S. Cognitive Team
Version: v0.9.4-alpha
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis package not installed. Install with: pip install redis")

logger = logging.getLogger(__name__)


class RedisBackend:
    """
    Redis backend for dashboard data persistence.

    Data structures:
    - agent_state:{agent_id} -> Hash (current state)
    - agent_history:{agent_id} -> List (recent states, max 1000)
    - conflicts:list -> List (conflict events)
    - equilibrium:list -> List (Nash equilibrium history)
    - metrics:hash -> Hash (aggregated metrics)
    - websocket:subscribers -> Set (active WebSocket connection IDs)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_history_length: int = 1000,
        ttl_seconds: int = 86400,  # 24 hours
    ):
        """
        Initialize Redis backend.

        Args:
            redis_url: Redis connection URL
            max_history_length: Maximum history items to retain per agent
            ttl_seconds: Time-to-live for cached data (seconds)
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis package not installed")

        self.redis_url = redis_url
        self.max_history_length = max_history_length
        self.ttl_seconds = ttl_seconds
        self.client: Optional[redis.Redis] = None

        logger.info(f"RedisBackend initialized: url={redis_url}, max_history={max_history_length}")

    async def connect(self):
        """Establish connection to Redis."""
        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis")

    # ============================================================================
    # Agent State Operations
    # ============================================================================

    async def store_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """
        Store current agent state.

        Args:
            agent_id: Agent identifier
            state: State dictionary
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            # Store current state as hash
            key = f"agent_state:{agent_id}"
            await self.client.hset(key, mapping=state)
            await self.client.expire(key, self.ttl_seconds)

            # Add to history
            await self._add_to_agent_history(agent_id, state)

        except Exception as e:
            logger.error(f"Failed to store agent state for {agent_id}: {e}")

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve current agent state.

        Args:
            agent_id: Agent identifier

        Returns:
            State dictionary or None if not found
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = f"agent_state:{agent_id}"
            state = await self.client.hgetall(key)
            return dict(state) if state else None

        except Exception as e:
            logger.error(f"Failed to get agent state for {agent_id}: {e}")
            return None

    async def _add_to_agent_history(self, agent_id: str, state: Dict[str, Any]):
        """Add state to agent history (capped list)."""
        try:
            key = f"agent_history:{agent_id}"

            # Add timestamp if not present
            if "timestamp" not in state:
                state["timestamp"] = datetime.utcnow().isoformat()

            # Serialize and push to list
            serialized = json.dumps(state)
            await self.client.lpush(key, serialized)

            # Trim list to max length
            await self.client.ltrim(key, 0, self.max_history_length - 1)

            # Set expiry
            await self.client.expire(key, self.ttl_seconds)

        except Exception as e:
            logger.error(f"Failed to add to agent history for {agent_id}: {e}")

    async def get_agent_history(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve agent history.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of items to return
            offset: Number of items to skip

        Returns:
            List of state dictionaries
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = f"agent_history:{agent_id}"

            # Get range from list
            items = await self.client.lrange(key, offset, offset + limit - 1)

            # Deserialize
            history = [json.loads(item) for item in items]

            return history

        except Exception as e:
            logger.error(f"Failed to get agent history for {agent_id}: {e}")
            return []

    # ============================================================================
    # Conflict Operations
    # ============================================================================

    async def store_conflict(self, conflict: Dict[str, Any]):
        """Store a conflict event."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "conflicts:list"

            # Add timestamp
            if "timestamp" not in conflict:
                conflict["timestamp"] = datetime.utcnow().isoformat()

            serialized = json.dumps(conflict)
            await self.client.lpush(key, serialized)

            # Trim to max 500 conflicts
            await self.client.ltrim(key, 0, 499)

            await self.client.expire(key, self.ttl_seconds)

        except Exception as e:
            logger.error(f"Failed to store conflict: {e}")

    async def get_conflicts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent conflicts."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "conflicts:list"
            items = await self.client.lrange(key, 0, limit - 1)
            return [json.loads(item) for item in items]

        except Exception as e:
            logger.error(f"Failed to get conflicts: {e}")
            return []

    # ============================================================================
    # Nash Equilibrium Operations
    # ============================================================================

    async def store_nash_equilibrium(self, equilibrium: Dict[str, Any]):
        """Store Nash equilibrium computation result."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "equilibrium:list"

            if "timestamp" not in equilibrium:
                equilibrium["timestamp"] = datetime.utcnow().isoformat()

            serialized = json.dumps(equilibrium)
            await self.client.lpush(key, serialized)

            # Trim to max 500
            await self.client.ltrim(key, 0, 499)

            await self.client.expire(key, self.ttl_seconds)

        except Exception as e:
            logger.error(f"Failed to store Nash equilibrium: {e}")

    async def get_nash_equilibrium_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve Nash equilibrium history."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "equilibrium:list"
            items = await self.client.lrange(key, 0, limit - 1)
            return [json.loads(item) for item in items]

        except Exception as e:
            logger.error(f"Failed to get Nash equilibrium history: {e}")
            return []

    # ============================================================================
    # Metrics Operations
    # ============================================================================

    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update aggregated metrics."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "metrics:hash"

            # Convert complex types to strings for Redis hash
            flat_metrics = self._flatten_metrics(metrics)

            await self.client.hset(key, mapping=flat_metrics)
            await self.client.expire(key, self.ttl_seconds)

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Retrieve aggregated metrics."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "metrics:hash"
            metrics = await self.client.hgetall(key)

            # Unflatten metrics
            return self._unflatten_metrics(dict(metrics)) if metrics else {}

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Flatten nested metrics dictionary for Redis hash storage."""
        flat = {}

        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(self._flatten_metrics(value, prefix=f"{full_key}:"))
            else:
                flat[full_key] = json.dumps(value)

        return flat

    def _unflatten_metrics(self, flat_metrics: Dict[str, str]) -> Dict[str, Any]:
        """Unflatten metrics from Redis hash."""
        metrics = {}

        for key, value in flat_metrics.items():
            keys = key.split(":")
            current = metrics

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            try:
                current[keys[-1]] = json.loads(value)
            except json.JSONDecodeError:
                current[keys[-1]] = value

        return metrics

    # ============================================================================
    # WebSocket Subscriber Operations
    # ============================================================================

    async def add_websocket_subscriber(self, connection_id: str):
        """Add WebSocket subscriber."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "websocket:subscribers"
            await self.client.sadd(key, connection_id)
            await self.client.expire(key, 3600)  # 1 hour TTL

        except Exception as e:
            logger.error(f"Failed to add WebSocket subscriber: {e}")

    async def remove_websocket_subscriber(self, connection_id: str):
        """Remove WebSocket subscriber."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "websocket:subscribers"
            await self.client.srem(key, connection_id)

        except Exception as e:
            logger.error(f"Failed to remove WebSocket subscriber: {e}")

    async def get_websocket_subscribers(self) -> List[str]:
        """Get all active WebSocket subscribers."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            key = "websocket:subscribers"
            subscribers = await self.client.smembers(key)
            return list(subscribers)

        except Exception as e:
            logger.error(f"Failed to get WebSocket subscribers: {e}")
            return []

    # ============================================================================
    # Utility Operations
    # ============================================================================

    async def clear_all_data(self):
        """Clear all dashboard data (use with caution)."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            patterns = [
                "agent_state:*",
                "agent_history:*",
                "conflicts:list",
                "equilibrium:list",
                "metrics:hash",
                "websocket:subscribers"
            ]

            for pattern in patterns:
                keys = []
                async for key in self.client.scan_iter(match=pattern):
                    keys.append(key)

                if keys:
                    await self.client.delete(*keys)

            logger.info("Cleared all dashboard data from Redis")

        except Exception as e:
            logger.error(f"Failed to clear data: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get Redis storage statistics."""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        try:
            stats = {}

            # Count keys by pattern
            for agent_id in ["policy", "consensus", "ethical", "resource"]:
                state_key = f"agent_state:{agent_id}"
                history_key = f"agent_history:{agent_id}"

                stats[f"{agent_id}_state_exists"] = await self.client.exists(state_key)
                stats[f"{agent_id}_history_length"] = await self.client.llen(history_key)

            stats["conflicts_count"] = await self.client.llen("conflicts:list")
            stats["equilibrium_count"] = await self.client.llen("equilibrium:list")
            stats["websocket_subscribers"] = await self.client.scard("websocket:subscribers")

            # Memory usage
            info = await self.client.info("memory")
            stats["redis_memory_used_mb"] = round(info.get("used_memory", 0) / 1024 / 1024, 2)

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

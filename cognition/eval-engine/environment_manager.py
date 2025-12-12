"""
Environment Manager - Gymnasium environment lifecycle with LRU caching.
"""
import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class CachedEnvironment:
    """Cached environment entry."""
    env_id: str
    env: gym.Env
    last_accessed: datetime
    access_count: int


class EnvironmentCache:
    """LRU cache for Gymnasium environments."""

    def __init__(self, max_size: int = 50):
        self.cache: OrderedDict[str, CachedEnvironment] = OrderedDict()
        self.max_size = max_size
        self.lock = asyncio.Lock()

    async def get_or_create_env(
        self,
        env_id: str,
        apply_wrappers: bool = True
    ) -> gym.Env:
        """
        Get environment from cache or create new one.

        LRU eviction policy: Remove least recently used if cache full.

        Args:
            env_id: Gymnasium environment ID (e.g., "CartPole-v1")
            apply_wrappers: Apply noise and reward delay wrappers

        Returns:
            gym.Env instance
        """
        async with self.lock:
            # Check if env_id in cache
            if env_id in self.cache:
                # Move to end (most recently used)
                cached = self.cache.pop(env_id)
                cached.last_accessed = datetime.utcnow()
                cached.access_count += 1
                self.cache[env_id] = cached
                return cached.env

            # Create new environment
            env = await self._create_env(env_id, apply_wrappers)

            # Evict LRU if cache full
            if len(self.cache) >= self.max_size:
                lru_key, lru_cached = self.cache.popitem(last=False)
                lru_cached.env.close()

            # Add to cache
            self.cache[env_id] = CachedEnvironment(
                env_id=env_id,
                env=env,
                last_accessed=datetime.utcnow(),
                access_count=1
            )

            return env

    async def _create_env(
        self,
        env_id: str,
        apply_wrappers: bool
    ) -> gym.Env:
        """
        Create Gymnasium environment with optional wrappers.

        Wrappers:
        - NoiseWrapper: Add Gaussian noise to observations (sigma=0.01)
        - RewardDelayWrapper: Delay reward by 1-3 steps
        """
        env = gym.make(env_id)

        if apply_wrappers:
            env = NoiseWrapper(env, noise_sigma=0.01)
            env = RewardDelayWrapper(env, delay_range=(1, 3))

        return env

    async def close_all(self):
        """Close all cached environments."""
        async with self.lock:
            for cached in self.cache.values():
                cached.env.close()
            self.cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "environments": [
                {
                    "env_id": cached.env_id,
                    "access_count": cached.access_count,
                    "last_accessed": cached.last_accessed.isoformat()
                }
                for cached in self.cache.values()
            ]
        }


class NoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to observations."""

    def __init__(self, env, noise_sigma: float = 0.01):
        super().__init__(env)
        self.noise_sigma = noise_sigma

    def observation(self, obs):
        """Add Gaussian noise to observation."""
        noise = np.random.normal(0, self.noise_sigma, size=obs.shape)
        return obs + noise


class RewardDelayWrapper(gym.Wrapper):
    """Delay reward by N steps."""

    def __init__(self, env, delay_range: Tuple[int, int] = (1, 3)):
        super().__init__(env)
        self.delay_range = delay_range
        self.reward_buffer = deque()

    def step(self, action):
        """Step with delayed reward."""
        obs, reward, done, truncated, info = self.env.step(action)

        # Add current reward to buffer with random delay
        delay = np.random.randint(self.delay_range[0], self.delay_range[1] + 1)
        self.reward_buffer.append((reward, delay))

        # Decrement delay counters and release rewards with delay <= 0
        released_reward = 0.0
        new_buffer = deque()

        for r, d in self.reward_buffer:
            if d <= 1:  # d=1 means release this step
                released_reward += r
            else:
                new_buffer.append((r, d - 1))

        self.reward_buffer = new_buffer

        return obs, released_reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset environment and clear reward buffer."""
        self.reward_buffer.clear()
        return self.env.reset(**kwargs)

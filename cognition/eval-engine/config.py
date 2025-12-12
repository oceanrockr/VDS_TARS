"""
Evaluation Engine Configuration
Loads settings from environment variables with validation.
"""
import os
from dataclasses import dataclass
from typing import Optional


def env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean from environment variable."""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes"):
        return True
    elif value in ("false", "0", "no"):
        return False
    elif value == "":
        return default
    else:
        return default


def env_int(key: str, default: int) -> int:
    """Parse integer from environment variable."""
    try:
        value = os.getenv(key)
        if value is None or value == "":
            return default
        return int(value)
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    """Parse float from environment variable."""
    try:
        value = os.getenv(key)
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


@dataclass
class RegressionThresholds:
    """Regression detection thresholds."""
    failure_rate: float = 0.15          # Max 15% episode failures
    reward_drop_pct: float = 0.10       # Max 10% reward drop
    loss_trend_window: int = 10         # Check last 10 episodes
    variance_multiplier: float = 2.5    # Max 2.5x variance increase

    @classmethod
    def from_env(cls) -> "RegressionThresholds":
        """Load thresholds from environment variables."""
        return cls(
            failure_rate=env_float("EVAL_FAILURE_RATE", 0.15),
            reward_drop_pct=env_float("EVAL_REWARD_DROP_PCT", 0.10),
            loss_trend_window=env_int("EVAL_LOSS_TREND_WINDOW", 10),
            variance_multiplier=env_float("EVAL_VARIANCE_MULTIPLIER", 2.5)
        )


@dataclass
class EvalEngineConfig:
    """Evaluation Engine configuration."""
    port: int
    postgres_url: str
    redis_url: str
    default_episodes: int = 100
    quick_mode_episodes: int = 50
    max_concurrent_evals: int = 4
    env_cache_size: int = 50
    thresholds: RegressionThresholds

    @classmethod
    def from_env(cls) -> "EvalEngineConfig":
        """Load configuration from environment variables."""
        # Required variables
        postgres_url = os.getenv("POSTGRES_URL")
        redis_url = os.getenv("REDIS_URL")

        if not postgres_url:
            raise ValueError("POSTGRES_URL environment variable is required")
        if not redis_url:
            raise ValueError("REDIS_URL environment variable is required")

        # Optional variables with defaults
        port = env_int("EVAL_ENGINE_PORT", 8099)
        default_episodes = env_int("EVAL_DEFAULT_EPISODES", 100)
        quick_mode_episodes = env_int("EVAL_QUICK_MODE_EPISODES", 50)
        max_concurrent_evals = env_int("EVAL_MAX_CONCURRENT", 4)
        env_cache_size = env_int("EVAL_ENV_CACHE_SIZE", 50)

        # Load regression thresholds
        thresholds = RegressionThresholds.from_env()

        return cls(
            port=port,
            postgres_url=postgres_url,
            redis_url=redis_url,
            default_episodes=default_episodes,
            quick_mode_episodes=quick_mode_episodes,
            max_concurrent_evals=max_concurrent_evals,
            env_cache_size=env_cache_size,
            thresholds=thresholds
        )

"""
Pydantic models for Evaluation Engine API.
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json
import sys


# Allowed Gymnasium environments
ALLOWED_ENVIRONMENTS = {"CartPole-v1", "LunarLander-v2", "MountainCar-v0"}

# Max agent state size: 10MB
MAX_AGENT_STATE_SIZE_BYTES = 10 * 1024 * 1024


class EvaluationRequest(BaseModel):
    """Request to evaluate an agent."""
    agent_type: Literal["DQN", "A2C", "PPO", "DDPG"]
    agent_state: Dict[str, Any]
    hyperparameters: Dict[str, float]
    environments: List[str] = ["CartPole-v1"]
    num_episodes: int = Field(100, ge=1, le=500)
    quick_mode: bool = False
    compare_to_baseline: bool = True

    @validator('agent_state')
    def validate_state_size(cls, v):
        """Validate agent_state size is less than 10MB."""
        # Serialize to JSON and check size
        serialized = json.dumps(v)
        size_bytes = sys.getsizeof(serialized)

        if size_bytes > MAX_AGENT_STATE_SIZE_BYTES:
            raise ValueError(
                f"agent_state size ({size_bytes / 1024 / 1024:.2f} MB) "
                f"exceeds maximum allowed size ({MAX_AGENT_STATE_SIZE_BYTES / 1024 / 1024:.2f} MB)"
            )

        return v

    @validator('environments')
    def validate_environments(cls, v):
        """Validate all environments are in the allowed whitelist."""
        invalid_envs = [env for env in v if env not in ALLOWED_ENVIRONMENTS]

        if invalid_envs:
            raise ValueError(
                f"Invalid environment(s): {invalid_envs}. "
                f"Allowed environments: {sorted(ALLOWED_ENVIRONMENTS)}"
            )

        if len(v) == 0:
            raise ValueError("At least one environment must be specified")

        return v


class EpisodeResult(BaseModel):
    """Result from a single episode."""
    episode_num: int
    total_reward: float
    steps: int
    success: bool
    loss: Optional[float] = None


class MetricsResult(BaseModel):
    """Aggregated metrics from evaluation."""
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    success_rate: float = Field(..., ge=0.0, le=1.0)
    mean_steps: float
    mean_loss: Optional[float] = None
    loss_trend: Optional[Literal["increasing", "decreasing", "stable"]] = None
    action_entropy: float
    reward_variance: float


class RegressionResult(BaseModel):
    """Regression detection result."""
    detected: bool
    regression_score: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None
    should_rollback: bool
    failed_checks: List[str] = []


class EvaluationResult(BaseModel):
    """Complete evaluation result."""
    job_id: str
    agent_type: str
    environment: str
    metrics: MetricsResult
    regression: RegressionResult
    episodes: List[EpisodeResult]
    baseline_comparison: Optional[Dict[str, float]] = None
    nash_score: Optional[float] = None
    duration_seconds: float
    completed_at: str


class BaselineRecord(BaseModel):
    """Baseline performance record."""
    agent_type: str
    environment: str
    mean_reward: float
    std_reward: float
    success_rate: float = Field(..., ge=0.0, le=1.0)
    hyperparameters: Dict[str, Any]
    version: int = Field(..., gt=0)
    rank: int = Field(1, gt=0)
    created_at: str


class BaselineResponse(BaseModel):
    """Response for GET /v1/baselines."""
    agent_type: str
    environment: str
    baseline: BaselineRecord
    history: List[BaselineRecord] = []


class BaselineUpdateRequest(BaseModel):
    """Request to update baseline."""
    agent_type: Literal["DQN", "A2C", "PPO", "DDPG"]
    environment: str
    mean_reward: float
    std_reward: float
    success_rate: float = Field(..., ge=0.0, le=1.0)
    hyperparameters: Dict[str, Any]
    version: int = Field(..., gt=0)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str = "eval-engine"
    version: str = "v1.0.0-rc2"
    postgres: str = "unknown"
    redis: str = "unknown"

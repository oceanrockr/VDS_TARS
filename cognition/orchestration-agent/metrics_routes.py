"""
T.A.R.S. Orchestration Agent - Training Metrics Routes
Advanced metrics endpoints for agent training visualization
Phase 12
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel
import numpy as np

# Import auth components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from auth import get_current_user, require_developer, User

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/metrics/training", tags=["Training Metrics"])

# Training history storage (in production, this would come from agent instances)
# This is a simplified version for demonstration
training_history = {
    "policy": {
        "rewards": deque(maxlen=1000),
        "losses": deque(maxlen=1000),
        "epsilon": deque(maxlen=1000),
        "q_values": deque(maxlen=1000),
        "episodes": deque(maxlen=1000),
        "timestamps": deque(maxlen=1000)
    },
    "consensus": {
        "rewards": deque(maxlen=1000),
        "value_losses": deque(maxlen=1000),
        "policy_losses": deque(maxlen=1000),
        "entropy": deque(maxlen=1000),
        "episodes": deque(maxlen=1000),
        "timestamps": deque(maxlen=1000)
    },
    "ethical": {
        "rewards": deque(maxlen=1000),
        "value_losses": deque(maxlen=1000),
        "policy_losses": deque(maxlen=1000),
        "kl_divergence": deque(maxlen=1000),
        "clip_fraction": deque(maxlen=1000),
        "episodes": deque(maxlen=1000),
        "timestamps": deque(maxlen=1000)
    },
    "resource": {
        "rewards": deque(maxlen=1000),
        "critic_losses": deque(maxlen=1000),
        "actor_losses": deque(maxlen=1000),
        "q_values": deque(maxlen=1000),
        "episodes": deque(maxlen=1000),
        "timestamps": deque(maxlen=1000)
    }
}


# ============================================================================
# Response Models
# ============================================================================

class TrainingMetricsResponse(BaseModel):
    """Training metrics response"""
    agent_id: str
    metrics: Dict[str, List[float]]
    episodes: List[int]
    timestamps: List[str]
    total_datapoints: int


class RewardCurveResponse(BaseModel):
    """Reward curve response"""
    agent_id: str
    episodes: List[int]
    rewards: List[float]
    mean_rewards: List[float]  # Rolling mean
    std_rewards: List[float]  # Rolling std
    trend: str  # "improving", "stable", "declining"


class LossCurveResponse(BaseModel):
    """Loss curve response"""
    agent_id: str
    episodes: List[int]
    losses: Dict[str, List[float]]  # Multiple loss types


class ExplorationMetricsResponse(BaseModel):
    """Exploration metrics response"""
    agent_id: str
    episodes: List[int]
    exploration_values: List[float]  # epsilon for DQN, entropy for A2C/PPO
    exploration_type: str  # "epsilon", "entropy", "noise"


class NashVisualizationResponse(BaseModel):
    """Nash equilibrium visualization data"""
    converged: bool
    iterations: int
    strategy_profile: Dict[str, List[float]]
    payoff_matrix: Dict[str, Dict[str, float]]
    convergence_history: List[Dict[str, Any]]


# ============================================================================
# Training Metrics Endpoints
# ============================================================================

@router.get("/all", response_model=Dict[str, TrainingMetricsResponse])
async def get_all_training_metrics(
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(require_developer)
) -> Dict[str, TrainingMetricsResponse]:
    """
    Get training metrics for all agents

    Returns reward curves, loss curves, and exploration metrics

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested all training metrics")

    result = {}

    for agent_id, history in training_history.items():
        # Get last N datapoints
        metrics = {}
        for metric_name, metric_data in history.items():
            if metric_name in ["episodes", "timestamps"]:
                continue
            metrics[metric_name] = list(metric_data)[-limit:]

        episodes = list(history.get("episodes", []))[-limit:]
        timestamps = [ts.isoformat() if isinstance(ts, datetime) else ts
                     for ts in list(history.get("timestamps", []))[-limit:]]

        result[agent_id] = TrainingMetricsResponse(
            agent_id=agent_id,
            metrics=metrics,
            episodes=episodes,
            timestamps=timestamps,
            total_datapoints=len(episodes)
        )

    return result


@router.get("/{agent_id}/rewards", response_model=RewardCurveResponse)
async def get_reward_curve(
    agent_id: str,
    window: int = Query(10, ge=1, le=100),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(require_developer)
) -> RewardCurveResponse:
    """
    Get reward curve for a specific agent

    Includes rolling mean and std for smoothed visualization

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested reward curve for agent {agent_id}")

    if agent_id not in training_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    history = training_history[agent_id]
    rewards = list(history.get("rewards", []))[-limit:]
    episodes = list(history.get("episodes", []))[-limit:]

    if not rewards:
        # Return empty/mock data
        return RewardCurveResponse(
            agent_id=agent_id,
            episodes=[],
            rewards=[],
            mean_rewards=[],
            std_rewards=[],
            trend="stable"
        )

    # Calculate rolling statistics
    mean_rewards = []
    std_rewards = []

    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        window_data = rewards[start_idx:i+1]
        mean_rewards.append(float(np.mean(window_data)))
        std_rewards.append(float(np.std(window_data)))

    # Determine trend
    if len(mean_rewards) >= 20:
        recent_mean = np.mean(mean_rewards[-10:])
        earlier_mean = np.mean(mean_rewards[-20:-10])
        if recent_mean > earlier_mean * 1.1:
            trend = "improving"
        elif recent_mean < earlier_mean * 0.9:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return RewardCurveResponse(
        agent_id=agent_id,
        episodes=episodes,
        rewards=rewards,
        mean_rewards=mean_rewards,
        std_rewards=std_rewards,
        trend=trend
    )


@router.get("/{agent_id}/losses", response_model=LossCurveResponse)
async def get_loss_curve(
    agent_id: str,
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(require_developer)
) -> LossCurveResponse:
    """
    Get loss curves for a specific agent

    Different agents have different loss types:
    - DQN: TD loss
    - A2C: value_loss, policy_loss
    - PPO: value_loss, policy_loss
    - DDPG: critic_loss, actor_loss

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested loss curve for agent {agent_id}")

    if agent_id not in training_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    history = training_history[agent_id]
    episodes = list(history.get("episodes", []))[-limit:]

    # Extract loss data based on agent type
    losses = {}

    if agent_id == "policy":  # DQN
        if "losses" in history:
            losses["td_loss"] = list(history["losses"])[-limit:]

    elif agent_id == "consensus":  # A2C
        if "value_losses" in history:
            losses["value_loss"] = list(history["value_losses"])[-limit:]
        if "policy_losses" in history:
            losses["policy_loss"] = list(history["policy_losses"])[-limit:]

    elif agent_id == "ethical":  # PPO
        if "value_losses" in history:
            losses["value_loss"] = list(history["value_losses"])[-limit:]
        if "policy_losses" in history:
            losses["policy_loss"] = list(history["policy_losses"])[-limit:]

    elif agent_id == "resource":  # DDPG
        if "critic_losses" in history:
            losses["critic_loss"] = list(history["critic_losses"])[-limit:]
        if "actor_losses" in history:
            losses["actor_loss"] = list(history["actor_losses"])[-limit:]

    return LossCurveResponse(
        agent_id=agent_id,
        episodes=episodes,
        losses=losses
    )


@router.get("/{agent_id}/exploration", response_model=ExplorationMetricsResponse)
async def get_exploration_metrics(
    agent_id: str,
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(require_developer)
) -> ExplorationMetricsResponse:
    """
    Get exploration metrics for a specific agent

    Different agents use different exploration mechanisms:
    - DQN: epsilon-greedy
    - A2C/PPO: entropy regularization
    - DDPG: action noise

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested exploration metrics for agent {agent_id}")

    if agent_id not in training_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    history = training_history[agent_id]
    episodes = list(history.get("episodes", []))[-limit:]

    # Extract exploration data based on agent type
    exploration_values = []
    exploration_type = ""

    if agent_id == "policy":  # DQN
        exploration_values = list(history.get("epsilon", []))[-limit:]
        exploration_type = "epsilon"

    elif agent_id in ["consensus", "ethical"]:  # A2C or PPO
        exploration_values = list(history.get("entropy", []))[-limit:]
        exploration_type = "entropy"

    elif agent_id == "resource":  # DDPG
        # For DDPG, we might track noise sigma if available
        # For now, use a placeholder
        exploration_values = [0.1] * len(episodes)
        exploration_type = "noise"

    return ExplorationMetricsResponse(
        agent_id=agent_id,
        episodes=episodes,
        exploration_values=exploration_values,
        exploration_type=exploration_type
    )


@router.get("/nash/visualization", response_model=NashVisualizationResponse)
async def get_nash_visualization(
    current_user: User = Depends(require_developer)
) -> NashVisualizationResponse:
    """
    Get Nash equilibrium visualization data

    Includes:
    - Strategy profiles for each agent
    - Payoff matrix
    - Convergence history

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested Nash equilibrium visualization")

    # Mock Nash equilibrium data (in production, this would come from NashSolver)
    return NashVisualizationResponse(
        converged=True,
        iterations=32,
        strategy_profile={
            "policy": [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01],
            "consensus": [0.4, 0.35, 0.15, 0.07, 0.03],
            "ethical": [0.25, 0.20, 0.18, 0.12, 0.10, 0.08, 0.05, 0.02],
            "resource": [0.7, 0.3]
        },
        payoff_matrix={
            "policy": {"policy": 0.76, "consensus": 0.72, "ethical": 0.68, "resource": 0.74},
            "consensus": {"policy": 0.78, "consensus": 0.81, "ethical": 0.75, "resource": 0.79},
            "ethical": {"policy": 0.70, "consensus": 0.73, "ethical": 0.72, "resource": 0.71},
            "resource": {"policy": 0.85, "consensus": 0.83, "ethical": 0.80, "resource": 0.87}
        },
        convergence_history=[
            {"iteration": i, "max_deviation": 1.0 / (i + 1), "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat()}
            for i in range(32)
        ]
    )


@router.get("/{agent_id}/hyperparameters")
async def get_agent_hyperparameters(
    agent_id: str,
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get current hyperparameters for a specific agent

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested hyperparameters for agent {agent_id}")

    if agent_id not in training_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    # Mock hyperparameters (in production, fetch from agent instances)
    hyperparameters = {
        "policy": {  # DQN
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 0.15,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "buffer_size": 10000,
            "batch_size": 64,
            "target_update_freq": 1000
        },
        "consensus": {  # A2C
            "learning_rate": 0.0007,
            "gamma": 0.99,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5
        },
        "ethical": {  # PPO
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "vf_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 10
        },
        "resource": {  # DDPG
            "actor_lr": 0.0001,
            "critic_lr": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 100000,
            "batch_size": 128,
            "noise_sigma": 0.1
        }
    }

    return {
        "agent_id": agent_id,
        "hyperparameters": hyperparameters.get(agent_id, {}),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/{agent_id}/performance-summary")
async def get_performance_summary(
    agent_id: str,
    window: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get performance summary statistics for an agent

    Includes:
    - Mean/std/min/max rewards
    - Training stability metrics
    - Convergence indicators

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested performance summary for agent {agent_id}")

    if agent_id not in training_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    history = training_history[agent_id]
    rewards = list(history.get("rewards", []))[-window:]

    if not rewards:
        return {
            "agent_id": agent_id,
            "window": window,
            "message": "No training data available",
            "timestamp": datetime.utcnow().isoformat()
        }

    # Calculate statistics
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))

    # Calculate trend
    if len(rewards) >= 20:
        recent_mean = np.mean(rewards[-10:])
        earlier_mean = np.mean(rewards[-20:-10])
        improvement = ((recent_mean - earlier_mean) / abs(earlier_mean)) * 100 if earlier_mean != 0 else 0
    else:
        improvement = 0

    # Calculate stability (inverse of coefficient of variation)
    stability = 1.0 - min(std_reward / abs(mean_reward), 1.0) if mean_reward != 0 else 0

    return {
        "agent_id": agent_id,
        "window": window,
        "statistics": {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "improvement_percent": round(improvement, 2),
            "stability": round(stability, 3)
        },
        "total_episodes": len(list(history.get("episodes", []))),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Helper Functions (for integration with actual agents)
# ============================================================================

def record_training_step(
    agent_id: str,
    episode: int,
    reward: float,
    loss: Optional[float] = None,
    **kwargs
):
    """
    Record a training step for an agent

    This function should be called from the agent training loop
    """
    if agent_id not in training_history:
        logger.warning(f"Unknown agent_id: {agent_id}")
        return

    history = training_history[agent_id]

    history["episodes"].append(episode)
    history["rewards"].append(reward)
    history["timestamps"].append(datetime.utcnow())

    if loss is not None:
        history["losses"].append(loss)

    # Store agent-specific metrics
    for key, value in kwargs.items():
        if key in history:
            history[key].append(value)

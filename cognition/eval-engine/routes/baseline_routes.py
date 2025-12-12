"""
Baseline Routes - Baseline management endpoints.
"""
import sys
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import BaselineRecord, BaselineResponse, BaselineUpdateRequest
from baseline_manager import BaselineManager

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from auth import get_current_user, User, Role

router = APIRouter(prefix="/v1/baselines", tags=["baselines"])


@router.get("/{agent_type}", response_model=BaselineResponse)
async def get_baseline(
    agent_type: str,
    environment: str = Query("CartPole-v1"),
    top_n: int = Query(1, ge=1, le=10),
    current_user: User = Depends(get_current_user)
):
    """
    Get current performance baseline for agent.

    Requires: viewer or higher role.

    Args:
        agent_type: Agent type (DQN, A2C, PPO, DDPG)
        environment: Environment ID
        top_n: Number of top baselines to return

    Returns:
        BaselineResponse with current baseline and history
    """
    # TODO: Implement GET /baselines/{agent_type}
    # 1. Get baseline_manager from app.state
    #
    # 2. baseline = await baseline_manager.get_baseline(agent_type, environment, rank=1)
    #    if not baseline:
    #        raise HTTPException(status_code=404, detail="Baseline not found")
    #
    # 3. history = await baseline_manager.get_baseline_history(
    #        agent_type, environment, limit=top_n-1
    #    )
    #
    # 4. return BaselineResponse(
    #        agent_type=agent_type,
    #        environment=environment,
    #        baseline=baseline,
    #        history=history[1:]  # Skip rank 1 (already in baseline)
    #    )
    pass


@router.post("", status_code=status.HTTP_201_CREATED)
async def update_baseline(
    request: BaselineUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update baseline for agent (admin only).

    Requires: admin role.

    Args:
        request: Baseline update request with metrics

    Returns:
        Baseline ID and rank information
    """
    # TODO: Implement POST /baselines
    # 1. Check current_user.role == Role.ADMIN
    #    if current_user.role != Role.ADMIN:
    #        raise HTTPException(status_code=403, detail="Admin role required")
    #
    # 2. Get baseline_manager from app.state
    #
    # 3. from models import MetricsResult
    #    metrics = MetricsResult(
    #        mean_reward=request.mean_reward,
    #        std_reward=request.std_reward,
    #        success_rate=request.success_rate,
    #        ... (fill in required fields)
    #    )
    #
    # 4. baseline_id = await baseline_manager.update_baseline_if_better(
    #        request.agent_type,
    #        request.environment,
    #        metrics,
    #        request.hyperparameters,
    #        request.version
    #    )
    #
    # 5. if baseline_id:
    #        return {"baseline_id": baseline_id, "rank": 1, "status": "updated"}
    #    else:
    #        raise HTTPException(
    #            status_code=400,
    #            detail="New baseline not better than current"
    #        )
    pass

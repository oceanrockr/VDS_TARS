"""
T.A.R.S. AutoML Pipeline - Training Metrics Routes
Advanced metrics endpoints for AutoML trial visualization
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
router = APIRouter(prefix="/api/v1/automl", tags=["AutoML Metrics"])

# Mock trial history storage (in production, this would come from Optuna/MLflow)
trial_history = deque(maxlen=5000)

# Best trials by agent
best_trials = {
    "policy": None,
    "consensus": None,
    "ethical": None,
    "resource": None
}


# ============================================================================
# Response Models
# ============================================================================

class Trial(BaseModel):
    """AutoML trial"""
    trial_id: str
    agent_id: str
    hyperparameters: Dict[str, Any]
    score: float
    status: str  # "complete", "running", "failed"
    duration_seconds: float
    timestamp: datetime


class TrialListResponse(BaseModel):
    """Trial list response"""
    trials: List[Trial]
    total: int
    best_score: Optional[float]
    worst_score: Optional[float]


class ParetoFrontierResponse(BaseModel):
    """Pareto frontier response"""
    agent_id: str
    frontier_points: List[Dict[str, Any]]  # Each point has: hyperparameters, objectives
    dominated_points: List[Dict[str, Any]]


class SearchProgressResponse(BaseModel):
    """Search progress response"""
    agent_id: str
    total_trials: int
    completed_trials: int
    running_trials: int
    failed_trials: int
    best_score: Optional[float]
    convergence_status: str  # "converging", "exploring", "stuck"
    estimated_completion: Optional[str]


class HyperparameterImportanceResponse(BaseModel):
    """Hyperparameter importance response"""
    agent_id: str
    importances: Dict[str, float]  # hyperparameter -> importance
    correlations: Dict[str, float]  # hyperparameter -> correlation with score


# ============================================================================
# AutoML Trial Endpoints
# ============================================================================

@router.get("/trials", response_model=TrialListResponse)
async def get_trials(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_developer)
) -> TrialListResponse:
    """
    Get AutoML trial history

    Query parameters:
    - agent_id: Filter by agent ID
    - status: Filter by status (complete, running, failed)
    - limit: Maximum number of trials to return
    - offset: Number of trials to skip

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested AutoML trials")

    # Filter trials
    filtered_trials = list(trial_history)

    if agent_id:
        filtered_trials = [t for t in filtered_trials if t.get("agent_id") == agent_id]

    if status:
        filtered_trials = [t for t in filtered_trials if t.get("status") == status]

    # Calculate stats
    scores = [t.get("score") for t in filtered_trials if t.get("score") is not None]
    best_score = max(scores) if scores else None
    worst_score = min(scores) if scores else None

    # Paginate
    paginated_trials = filtered_trials[offset:offset+limit]

    # Convert to Trial objects
    trials = [
        Trial(
            trial_id=t.get("trial_id", ""),
            agent_id=t.get("agent_id", ""),
            hyperparameters=t.get("hyperparameters", {}),
            score=t.get("score", 0.0),
            status=t.get("status", "unknown"),
            duration_seconds=t.get("duration_seconds", 0.0),
            timestamp=t.get("timestamp", datetime.utcnow())
        )
        for t in paginated_trials
    ]

    return TrialListResponse(
        trials=trials,
        total=len(filtered_trials),
        best_score=best_score,
        worst_score=worst_score
    )


@router.get("/trials/{trial_id}")
async def get_trial(
    trial_id: str,
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific trial

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested AutoML trial {trial_id}")

    # Find trial
    trial = None
    for t in trial_history:
        if t.get("trial_id") == trial_id:
            trial = t
            break

    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found"
        )

    return trial


@router.get("/search/status", response_model=Dict[str, SearchProgressResponse])
async def get_search_status(
    current_user: User = Depends(require_developer)
) -> Dict[str, SearchProgressResponse]:
    """
    Get current AutoML search status for all agents

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested AutoML search status")

    result = {}

    for agent_id in ["policy", "consensus", "ethical", "resource"]:
        # Filter trials for this agent
        agent_trials = [t for t in trial_history if t.get("agent_id") == agent_id]

        total_trials = len(agent_trials)
        completed_trials = len([t for t in agent_trials if t.get("status") == "complete"])
        running_trials = len([t for t in agent_trials if t.get("status") == "running"])
        failed_trials = len([t for t in agent_trials if t.get("status") == "failed"])

        # Get best score
        completed_scores = [t.get("score") for t in agent_trials if t.get("status") == "complete" and t.get("score") is not None]
        best_score = max(completed_scores) if completed_scores else None

        # Determine convergence status
        if len(completed_scores) >= 10:
            recent_scores = completed_scores[-10:]
            earlier_scores = completed_scores[-20:-10] if len(completed_scores) >= 20 else completed_scores[:10]

            recent_improvement = (max(recent_scores) - max(earlier_scores)) if earlier_scores else 0

            if recent_improvement > 0.01:
                convergence_status = "converging"
            elif recent_improvement < -0.01:
                convergence_status = "exploring"
            else:
                convergence_status = "stuck"
        else:
            convergence_status = "exploring"

        # Estimate completion (mock)
        if running_trials > 0:
            avg_duration = np.mean([t.get("duration_seconds", 0) for t in agent_trials if t.get("status") == "complete"])
            estimated_time = avg_duration * running_trials if avg_duration > 0 else 0
            estimated_completion = (datetime.utcnow() + timedelta(seconds=estimated_time)).isoformat()
        else:
            estimated_completion = None

        result[agent_id] = SearchProgressResponse(
            agent_id=agent_id,
            total_trials=total_trials,
            completed_trials=completed_trials,
            running_trials=running_trials,
            failed_trials=failed_trials,
            best_score=best_score,
            convergence_status=convergence_status,
            estimated_completion=estimated_completion
        )

    return result


@router.get("/pareto-frontier/{agent_id}", response_model=ParetoFrontierResponse)
async def get_pareto_frontier(
    agent_id: str,
    current_user: User = Depends(require_developer)
) -> ParetoFrontierResponse:
    """
    Get Pareto frontier for multi-objective optimization

    The Pareto frontier contains the set of non-dominated solutions

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested Pareto frontier for agent {agent_id}")

    # Filter trials for this agent
    agent_trials = [t for t in trial_history if t.get("agent_id") == agent_id]

    if not agent_trials:
        return ParetoFrontierResponse(
            agent_id=agent_id,
            frontier_points=[],
            dominated_points=[]
        )

    # Mock Pareto frontier computation
    # In production, this would use actual multi-objective optimization results

    # Convert trials to points with objectives
    points = []
    for trial in agent_trials:
        if trial.get("status") == "complete" and trial.get("score") is not None:
            # Mock multi-objective: (score, -training_time)
            points.append({
                "trial_id": trial.get("trial_id"),
                "hyperparameters": trial.get("hyperparameters", {}),
                "objectives": {
                    "score": trial.get("score"),
                    "training_time": -trial.get("duration_seconds", 0)  # Negative to maximize both
                }
            })

    # Simple Pareto frontier: find non-dominated points
    frontier_points = []
    dominated_points = []

    for i, point_i in enumerate(points):
        dominated = False
        for j, point_j in enumerate(points):
            if i != j:
                # Check if point_j dominates point_i
                score_i = point_i["objectives"]["score"]
                time_i = point_i["objectives"]["training_time"]
                score_j = point_j["objectives"]["score"]
                time_j = point_j["objectives"]["training_time"]

                if score_j >= score_i and time_j >= time_i and (score_j > score_i or time_j > time_i):
                    dominated = True
                    break

        if dominated:
            dominated_points.append(point_i)
        else:
            frontier_points.append(point_i)

    return ParetoFrontierResponse(
        agent_id=agent_id,
        frontier_points=frontier_points,
        dominated_points=dominated_points
    )


@router.get("/hyperparameter-importance/{agent_id}", response_model=HyperparameterImportanceResponse)
async def get_hyperparameter_importance(
    agent_id: str,
    current_user: User = Depends(require_developer)
) -> HyperparameterImportanceResponse:
    """
    Get hyperparameter importance analysis

    Shows which hyperparameters have the most impact on performance

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested hyperparameter importance for agent {agent_id}")

    # Filter trials for this agent
    agent_trials = [t for t in trial_history if t.get("agent_id") == agent_id and t.get("status") == "complete"]

    if not agent_trials:
        return HyperparameterImportanceResponse(
            agent_id=agent_id,
            importances={},
            correlations={}
        )

    # Mock importance calculation
    # In production, this would use Optuna's importance evaluator or SHAP values

    # Extract hyperparameters and scores
    hyperparams = {}
    scores = []

    for trial in agent_trials:
        trial_hyperparams = trial.get("hyperparameters", {})
        trial_score = trial.get("score")

        if trial_score is not None:
            scores.append(trial_score)

            for key, value in trial_hyperparams.items():
                if key not in hyperparams:
                    hyperparams[key] = []
                hyperparams[key].append(value)

    # Calculate mock importances and correlations
    importances = {}
    correlations = {}

    for key, values in hyperparams.items():
        # Mock importance (variance-based)
        if isinstance(values[0], (int, float)):
            variance = np.var(values) if len(values) > 1 else 0
            importances[key] = float(variance)

            # Mock correlation with score
            if len(values) == len(scores) and len(values) > 1:
                correlation = np.corrcoef(values, scores)[0, 1]
                correlations[key] = float(correlation) if not np.isnan(correlation) else 0.0
            else:
                correlations[key] = 0.0
        else:
            importances[key] = 0.0
            correlations[key] = 0.0

    # Normalize importances to sum to 1
    total_importance = sum(importances.values())
    if total_importance > 0:
        importances = {k: v / total_importance for k, v in importances.items()}

    return HyperparameterImportanceResponse(
        agent_id=agent_id,
        importances=importances,
        correlations=correlations
    )


@router.get("/optimization-history/{agent_id}")
async def get_optimization_history(
    agent_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get optimization history (score over trials)

    Shows how the search progresses over time

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested optimization history for agent {agent_id}")

    # Filter trials for this agent
    agent_trials = [t for t in trial_history if t.get("agent_id") == agent_id and t.get("status") == "complete"][-limit:]

    if not agent_trials:
        return {
            "agent_id": agent_id,
            "trials": [],
            "best_scores_running": [],
            "message": "No completed trials available"
        }

    # Extract trial numbers and scores
    trials = []
    scores = []
    best_scores_running = []

    for i, trial in enumerate(agent_trials):
        score = trial.get("score")
        if score is not None:
            trials.append(i + 1)
            scores.append(score)

            # Running best score
            best_so_far = max(scores)
            best_scores_running.append(best_so_far)

    return {
        "agent_id": agent_id,
        "trials": trials,
        "scores": scores,
        "best_scores_running": best_scores_running,
        "total_trials": len(trials),
        "current_best": max(scores) if scores else None,
        "improvement": (max(scores) - min(scores)) if len(scores) > 1 else 0
    }


@router.get("/best-trials")
async def get_best_trials(
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get best trial for each agent

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested best trials")

    result = {}

    for agent_id in ["policy", "consensus", "ethical", "resource"]:
        # Filter trials for this agent
        agent_trials = [t for t in trial_history if t.get("agent_id") == agent_id and t.get("status") == "complete"]

        if agent_trials:
            # Find best trial
            best_trial = max(agent_trials, key=lambda t: t.get("score", float('-inf')))
            result[agent_id] = best_trial
        else:
            result[agent_id] = None

    return {
        "best_trials": result,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Helper Functions (for integration with actual AutoML pipeline)
# ============================================================================

def record_trial(
    trial_id: str,
    agent_id: str,
    hyperparameters: Dict[str, Any],
    score: float,
    status: str,
    duration_seconds: float
):
    """
    Record an AutoML trial

    This function should be called from the AutoML pipeline
    """
    trial = {
        "trial_id": trial_id,
        "agent_id": agent_id,
        "hyperparameters": hyperparameters,
        "score": score,
        "status": status,
        "duration_seconds": duration_seconds,
        "timestamp": datetime.utcnow()
    }

    trial_history.append(trial)

    # Update best trial
    if status == "complete" and score is not None:
        if best_trials[agent_id] is None or score > best_trials[agent_id].get("score", float('-inf')):
            best_trials[agent_id] = trial

    logger.info(f"Recorded trial {trial_id} for agent {agent_id}: score={score}, status={status}")

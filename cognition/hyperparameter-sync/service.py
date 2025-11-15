"""
T.A.R.S. Hyperparameter Sync Service
FastAPI service for managing hyperparameter updates across multi-agent system

Port: 8098

Author: T.A.R.S. Cognitive Team
Version: v0.9.4-alpha
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from updater import (
    HyperparameterUpdater,
    HyperparameterUpdate,
    ApprovalMode,
    UpdateStatus
)

# Import auth components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from auth import get_current_user, verify_service_key, User, Role
from auth_routes import router as auth_router
from rate_limiter import public_rate_limit, rate_limit_middleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="T.A.R.S. Hyperparameter Sync Service",
    description="Synchronize hyperparameters from AutoML to running agents",
    version="0.9.5-alpha",
)

# Add auth routes
app.include_router(auth_router)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
updates_proposed_total = Counter(
    'tars_hyperparam_updates_proposed_total',
    'Total hyperparameter updates proposed',
    ['agent_type']
)
updates_applied_total = Counter(
    'tars_hyperparam_updates_applied_total',
    'Total hyperparameter updates applied',
    ['agent_type', 'status']
)
update_improvement = Histogram(
    'tars_hyperparam_update_improvement',
    'Improvement from hyperparameter updates',
    ['agent_type']
)
pending_updates_gauge = Gauge(
    'tars_hyperparam_pending_updates',
    'Number of pending hyperparameter updates'
)

# Configuration
AUTOML_SERVICE_URL = os.getenv("AUTOML_SERVICE_URL", "http://localhost:8097")
ORCHESTRATION_SERVICE_URL = os.getenv("ORCHESTRATION_SERVICE_URL", "http://localhost:8094")
APPROVAL_MODE = ApprovalMode(os.getenv("APPROVAL_MODE", "manual"))
AUTONOMOUS_THRESHOLD = float(os.getenv("AUTONOMOUS_THRESHOLD", "0.03"))
VALIDATION_STRICTNESS = os.getenv("VALIDATION_STRICTNESS", "medium")

# Global updater instance
updater: Optional[HyperparameterUpdater] = None


# ============================================================================
# Request/Response Models
# ============================================================================

class ProposeUpdateRequest(BaseModel):
    agent_type: str = Field(..., description="Agent type: dqn, a2c, ppo, ddpg")
    current_params: Dict[str, Any] = Field(..., description="Current hyperparameters")
    current_score: float = Field(..., description="Current performance score")


class ApproveUpdateRequest(BaseModel):
    update_id: str = Field(..., description="Update ID to approve")


class RejectUpdateRequest(BaseModel):
    update_id: str = Field(..., description="Update ID to reject")
    reason: Optional[str] = Field(None, description="Reason for rejection")


class SyncAllAgentsRequest(BaseModel):
    agent_configs: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Dictionary mapping agent_type -> {params, score}"
    )


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize hyperparameter updater on startup."""
    global updater

    logger.info("Initializing Hyperparameter Sync Service...")

    try:
        updater = HyperparameterUpdater(
            automl_service_url=AUTOML_SERVICE_URL,
            orchestration_service_url=ORCHESTRATION_SERVICE_URL,
            approval_mode=APPROVAL_MODE,
            autonomous_threshold=AUTONOMOUS_THRESHOLD,
            validation_strictness=VALIDATION_STRICTNESS,
        )

        logger.info("Hyperparameter Sync Service started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize updater: {e}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hyperparameter-sync",
        "updater_initialized": updater is not None,
        "approval_mode": APPROVAL_MODE.value,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/sync/propose")
async def propose_update(
    request: ProposeUpdateRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Propose a hyperparameter update for an agent.

    Requires: Developer role or higher

    This endpoint fetches optimized hyperparameters from AutoML,
    validates them, and creates an update proposal.
    """
    # Check authorization (developer or admin)
    if Role.DEVELOPER not in current_user.roles and Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=403,
            detail="Requires developer role or higher"
        )
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    try:
        update = await updater.propose_update(
            agent_type=request.agent_type,
            current_params=request.current_params,
            current_score=request.current_score
        )

        if update is None:
            return {
                "message": f"No better hyperparameters found for {request.agent_type}",
                "update": None
            }

        # Update metrics
        updates_proposed_total.labels(agent_type=request.agent_type).inc()
        pending_updates_gauge.set(len(updater.get_pending_updates()))

        return {
            "message": "Update proposed successfully",
            "update": update.dict()
        }

    except Exception as e:
        logger.error(f"Failed to propose update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync/approve")
async def approve_update(
    request: ApproveUpdateRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Manually approve a hyperparameter update.

    Requires: Admin role
    """
    # Check authorization (admin only)
    if Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=403,
            detail="Requires admin role"
        )
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    try:
        success = updater.approve_update(request.update_id)

        if not success:
            raise HTTPException(status_code=404, detail="Update not found or cannot be approved")

        return {
            "message": f"Update {request.update_id} approved",
            "update_id": request.update_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync/reject")
async def reject_update(request: RejectUpdateRequest) -> Dict[str, Any]:
    """Reject a hyperparameter update."""
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    try:
        success = updater.reject_update(request.update_id, reason=request.reason or "")

        if not success:
            raise HTTPException(status_code=404, detail="Update not found")

        pending_updates_gauge.set(len(updater.get_pending_updates()))

        return {
            "message": f"Update {request.update_id} rejected",
            "update_id": request.update_id,
            "reason": request.reason
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync/apply/{update_id}")
async def apply_update(update_id: str) -> Dict[str, Any]:
    """
    Apply a hyperparameter update.

    This endpoint performs a hot-reload of the agent with new hyperparameters.
    If the update is in autonomous mode and meets the threshold, it will be
    automatically approved.
    """
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    try:
        # Get update details before applying
        pending_updates = {u.update_id: u for u in updater.get_pending_updates()}

        if update_id not in pending_updates:
            raise HTTPException(status_code=404, detail="Update not found")

        update = pending_updates[update_id]
        agent_type = update.agent_type

        success = await updater.apply_update(update_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply update")

        # Update metrics
        updates_applied_total.labels(agent_type=agent_type, status="success").inc()
        update_improvement.labels(agent_type=agent_type).observe(update.improvement)
        pending_updates_gauge.set(len(updater.get_pending_updates()))

        return {
            "message": f"Update {update_id} applied successfully",
            "update_id": update_id,
            "agent_type": agent_type,
            "improvement": update.improvement
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply update: {e}")
        updates_applied_total.labels(agent_type="unknown", status="failed").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync/all")
async def sync_all_agents(
    request: SyncAllAgentsRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Synchronize hyperparameters for all agents.

    This endpoint checks for optimized hyperparameters for each agent
    and proposes/applies updates as appropriate.
    """
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    try:
        # Run sync in background
        background_tasks.add_task(
            _sync_all_agents_task,
            updater,
            request.agent_configs
        )

        return {
            "message": "Synchronization started for all agents",
            "agents": list(request.agent_configs.keys())
        }

    except Exception as e:
        logger.error(f"Failed to start sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _sync_all_agents_task(
    updater: HyperparameterUpdater,
    agent_configs: Dict[str, Dict[str, Any]]
):
    """Background task for syncing all agents."""
    try:
        results = await updater.sync_all_agents(agent_configs)

        for agent_type, update in results.items():
            if update:
                updates_proposed_total.labels(agent_type=agent_type).inc()

                if update.status == UpdateStatus.COMPLETED:
                    updates_applied_total.labels(agent_type=agent_type, status="success").inc()
                    update_improvement.labels(agent_type=agent_type).observe(update.improvement)

        pending_updates_gauge.set(len(updater.get_pending_updates()))

        logger.info(f"Sync all agents completed: {len(results)} agents processed")

    except Exception as e:
        logger.error(f"Sync all agents task failed: {e}")


@app.get("/api/v1/sync/pending")
async def get_pending_updates() -> Dict[str, Any]:
    """Get all pending hyperparameter updates."""
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    pending = updater.get_pending_updates()

    return {
        "pending_updates": [u.dict() for u in pending],
        "count": len(pending)
    }


@app.get("/api/v1/sync/history")
async def get_update_history(limit: int = 100) -> Dict[str, Any]:
    """Get hyperparameter update history."""
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    history = updater.get_update_history(limit=limit)

    return {
        "history": [u.dict() for u in history],
        "count": len(history),
        "limit": limit
    }


@app.get("/api/v1/sync/stats")
async def get_statistics() -> Dict[str, Any]:
    """Get hyperparameter sync statistics."""
    if updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    return updater.get_statistics()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("HYPERPARAM_SYNC_PORT", "8098"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

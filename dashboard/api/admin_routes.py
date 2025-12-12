"""
T.A.R.S. Admin Dashboard API Routes
RBAC-enforced admin endpoints for operator dashboard
Phase 12
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
import httpx

# Import auth dependencies
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cognition', 'shared'))
from auth import (
    get_current_user,
    require_admin,
    require_developer,
    User,
    Role,
    auth_service,
    APIKey
)
from rate_limiter import rate_limit
from audit_logger import get_audit_logger, AuditEventType, AuditEventSeverity

# Import JWT key store
try:
    from jwt_key_store import jwt_key_store, JWTKey
    JWT_KEY_STORE_AVAILABLE = True
except ImportError:
    JWT_KEY_STORE_AVAILABLE = False
    logger.warning("JWT key store not available")

logger = logging.getLogger(__name__)

# Service URLs
ORCHESTRATION_SERVICE_URL = os.getenv("ORCHESTRATION_SERVICE_URL", "http://localhost:8094")
AUTOML_SERVICE_URL = os.getenv("AUTOML_SERVICE_URL", "http://localhost:8097")
HYPERSYNC_SERVICE_URL = os.getenv("HYPERSYNC_SERVICE_URL", "http://localhost:8098")

# Create router
router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AgentReloadRequest(BaseModel):
    """Request to reload agent parameters"""
    agent_id: str
    config: Optional[Dict[str, Any]] = None
    reason: str = Field(..., description="Reason for reload")


class AgentReloadResponse(BaseModel):
    """Response from agent reload"""
    success: bool
    agent_id: str
    message: str
    timestamp: datetime


class ModelPromoteRequest(BaseModel):
    """Request to promote a model version"""
    agent_id: str
    model_version: str
    reason: str = Field(..., description="Reason for promotion")


class ModelPromoteResponse(BaseModel):
    """Response from model promotion"""
    success: bool
    agent_id: str
    old_version: Optional[str]
    new_version: str
    message: str
    timestamp: datetime


class HyperSyncApprovalRequest(BaseModel):
    """Request to approve/deny a HyperSync proposal"""
    proposal_id: str
    approved: bool
    reason: str = Field(..., description="Approval/denial reason")


class HyperSyncApprovalResponse(BaseModel):
    """Response from HyperSync approval"""
    success: bool
    proposal_id: str
    approved: bool
    message: str
    timestamp: datetime


class APIKeyCreateRequest(BaseModel):
    """Request to create a new API key"""
    service_name: str
    expires_in_days: Optional[int] = None


class APIKeyRevokeRequest(BaseModel):
    """Request to revoke an API key"""
    key_id: str
    reason: str = Field(..., description="Reason for revocation")


class SystemHealthResponse(BaseModel):
    """Aggregated system health response"""
    status: str
    services: Dict[str, Any]
    timestamp: datetime
    overall_healthy: bool


class JWTKeyResponse(BaseModel):
    """JWT key information (without secret)"""
    kid: str
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    is_valid: bool


class JWTRotationResponse(BaseModel):
    """Response from JWT key rotation"""
    success: bool
    old_kid: Optional[str]
    new_kid: str
    message: str
    timestamp: datetime
    grace_period_hours: int


class JWTStatusResponse(BaseModel):
    """JWT key status"""
    current_kid: str
    active_keys: List[JWTKeyResponse]
    valid_keys: List[JWTKeyResponse]
    total_active: int
    total_valid: int


class JWTInvalidateRequest(BaseModel):
    """Request to invalidate a JWT key"""
    reason: str = Field(..., description="Reason for invalidation")


# ============================================================================
# Agent Management Endpoints
# ============================================================================

@router.get("/agents")
@rate_limit
async def get_all_agents_admin(current_user: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    Get detailed state of all agents (admin view)

    Includes:
    - Current hyperparameters
    - Training statistics
    - Model versions
    - Performance metrics
    - Configuration

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested all agents state")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{ORCHESTRATION_SERVICE_URL}/api/v1/orchestration/agents/state",
                headers={"X-Admin-Request": "true"}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch agents from orchestration service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestration service unavailable"
        )


@router.get("/agents/{agent_id}")
@rate_limit
async def get_agent_admin(
    agent_id: str,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get detailed state of a specific agent (admin view)

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested agent {agent_id} state")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{ORCHESTRATION_SERVICE_URL}/api/v1/orchestration/agents/{agent_id}",
                headers={"X-Admin-Request": "true"}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        logger.error(f"Failed to fetch agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestration service unavailable"
        )


@router.post("/agents/{agent_id}/reload", response_model=AgentReloadResponse)
@rate_limit
async def reload_agent(
    agent_id: str,
    request: AgentReloadRequest,
    current_user: User = Depends(require_admin)
) -> AgentReloadResponse:
    """
    Reload agent parameters and configuration

    This triggers a hot-reload of the agent's:
    - Hyperparameters
    - Neural network weights
    - Configuration

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} reloading agent {agent_id}: {request.reason}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ORCHESTRATION_SERVICE_URL}/api/v1/orchestration/agents/{agent_id}/reload",
                json={
                    "config": request.config,
                    "reason": request.reason,
                    "requested_by": current_user.username
                }
            )
            response.raise_for_status()
            data = response.json()

            return AgentReloadResponse(
                success=True,
                agent_id=agent_id,
                message=data.get("message", "Agent reloaded successfully"),
                timestamp=datetime.utcnow()
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to reload agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to reload agent: {str(e)}"
        )


@router.post("/agents/{agent_id}/promote", response_model=ModelPromoteResponse)
@rate_limit
async def promote_model(
    agent_id: str,
    request: ModelPromoteRequest,
    current_user: User = Depends(require_admin)
) -> ModelPromoteResponse:
    """
    Promote a model version to production

    This updates the agent to use a specific model version from MLflow

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} promoting model for agent {agent_id} to {request.model_version}: {request.reason}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ORCHESTRATION_SERVICE_URL}/api/v1/orchestration/agents/{agent_id}/promote",
                json={
                    "model_version": request.model_version,
                    "reason": request.reason,
                    "requested_by": current_user.username
                }
            )
            response.raise_for_status()
            data = response.json()

            return ModelPromoteResponse(
                success=True,
                agent_id=agent_id,
                old_version=data.get("old_version"),
                new_version=request.model_version,
                message=data.get("message", "Model promoted successfully"),
                timestamp=datetime.utcnow()
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to promote model for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to promote model: {str(e)}"
        )


# ============================================================================
# AutoML Management Endpoints
# ============================================================================

@router.get("/automl/trials")
@rate_limit
async def get_automl_trials(
    agent_id: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get AutoML trial history

    Includes:
    - Trial configurations
    - Scores and metrics
    - Search progress
    - Pareto frontiers

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested AutoML trials")

    try:
        params = {"limit": limit}
        if agent_id:
            params["agent_id"] = agent_id

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{AUTOML_SERVICE_URL}/api/v1/automl/trials",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch AutoML trials: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AutoML service unavailable"
        )


@router.get("/automl/trials/{trial_id}")
@rate_limit
async def get_automl_trial(
    trial_id: str,
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific AutoML trial

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested AutoML trial {trial_id}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{AUTOML_SERVICE_URL}/api/v1/automl/trials/{trial_id}"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trial {trial_id} not found"
            )
        logger.error(f"Failed to fetch AutoML trial {trial_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AutoML service unavailable"
        )


@router.get("/automl/search/status")
@rate_limit
async def get_automl_search_status(
    current_user: User = Depends(require_developer)
) -> Dict[str, Any]:
    """
    Get current AutoML search status for all agents

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested AutoML search status")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{AUTOML_SERVICE_URL}/api/v1/automl/search/status"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch AutoML search status: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AutoML service unavailable"
        )


# ============================================================================
# HyperSync Management Endpoints
# ============================================================================

@router.get("/hypersync/proposals")
@rate_limit
async def get_hypersync_proposals(
    status_filter: Optional[str] = None,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get HyperSync proposals

    Query parameters:
    - status_filter: pending, approved, denied

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested HyperSync proposals")

    try:
        params = {}
        if status_filter:
            params["status"] = status_filter

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{HYPERSYNC_SERVICE_URL}/api/v1/hypersync/proposals",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch HyperSync proposals: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HyperSync service unavailable"
        )


@router.get("/hypersync/proposals/{proposal_id}")
@rate_limit
async def get_hypersync_proposal(
    proposal_id: str,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific HyperSync proposal

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested HyperSync proposal {proposal_id}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{HYPERSYNC_SERVICE_URL}/api/v1/hypersync/proposals/{proposal_id}"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Proposal {proposal_id} not found"
            )
        logger.error(f"Failed to fetch HyperSync proposal {proposal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HyperSync service unavailable"
        )


@router.post("/hypersync/proposals/{proposal_id}/approve", response_model=HyperSyncApprovalResponse)
@rate_limit
async def approve_hypersync_proposal(
    proposal_id: str,
    request: HyperSyncApprovalRequest,
    current_user: User = Depends(require_admin)
) -> HyperSyncApprovalResponse:
    """
    Approve or deny a HyperSync proposal

    This allows/prevents the propagation of hyperparameters across agents

    Requires: Admin role
    """
    action = "approving" if request.approved else "denying"
    logger.info(f"Admin {current_user.username} {action} HyperSync proposal {proposal_id}: {request.reason}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{HYPERSYNC_SERVICE_URL}/api/v1/hypersync/proposals/{proposal_id}/approve",
                json={
                    "approved": request.approved,
                    "reason": request.reason,
                    "approved_by": current_user.username
                }
            )
            response.raise_for_status()
            data = response.json()

            return HyperSyncApprovalResponse(
                success=True,
                proposal_id=proposal_id,
                approved=request.approved,
                message=data.get("message", f"Proposal {action}"),
                timestamp=datetime.utcnow()
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to {action} HyperSync proposal {proposal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to {action} proposal: {str(e)}"
        )


@router.get("/hypersync/history")
@rate_limit
async def get_hypersync_history(
    limit: int = 50,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get HyperSync approval/denial history

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested HyperSync history")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{HYPERSYNC_SERVICE_URL}/api/v1/hypersync/history",
                params={"limit": limit}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch HyperSync history: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HyperSync service unavailable"
        )


# ============================================================================
# API Key Management Endpoints
# ============================================================================

@router.get("/api-keys")
@rate_limit
async def list_api_keys(current_user: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    List all API keys (hashes only, not actual keys)

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested API keys list")

    # Get API keys from auth service
    api_keys = []
    for key_id, api_key in auth_service.config.api_keys.items():
        api_keys.append({
            "key_id": api_key.key_id,
            "service_name": api_key.service_name,
            "created_at": api_key.created_at.isoformat(),
            "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            "is_active": api_key.is_active
        })

    return {
        "api_keys": api_keys,
        "total": len(api_keys)
    }


@router.post("/api-keys")
@rate_limit
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Create a new API key for a service

    WARNING: The API key will only be shown once!

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} creating API key for service {request.service_name}")

    # Generate new API key
    key_id, api_key = auth_service.create_api_key(request.service_name)

    return {
        "key_id": key_id,
        "api_key": api_key,
        "service_name": request.service_name,
        "created_at": datetime.utcnow().isoformat(),
        "message": "API key created successfully. Save this key - it will not be shown again!",
        "created_by": current_user.username
    }


@router.post("/api-keys/{key_id}/rotate")
@rate_limit
async def rotate_api_key(
    key_id: str,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Rotate an API key (generate new key, invalidate old)

    WARNING: The new API key will only be shown once!

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} rotating API key {key_id}")

    try:
        new_key = auth_service.rotate_api_key(key_id)

        return {
            "key_id": key_id,
            "new_api_key": new_key,
            "message": "API key rotated successfully. Update your services with the new key!",
            "rotated_at": datetime.utcnow().isoformat(),
            "rotated_by": current_user.username
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post("/api-keys/{key_id}/revoke")
@rate_limit
async def revoke_api_key(
    key_id: str,
    request: APIKeyRevokeRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Revoke an API key (mark as inactive)

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} revoking API key {key_id}: {request.reason}")

    try:
        auth_service.revoke_api_key(key_id)

        return {
            "success": True,
            "key_id": key_id,
            "message": "API key revoked successfully",
            "reason": request.reason,
            "revoked_at": datetime.utcnow().isoformat(),
            "revoked_by": current_user.username
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# ============================================================================
# System Health Endpoints
# ============================================================================

@router.get("/health/system", response_model=SystemHealthResponse)
@rate_limit
async def get_system_health(current_user: User = Depends(require_admin)) -> SystemHealthResponse:
    """
    Get aggregated health status of all services

    Checks:
    - Orchestration service
    - AutoML service
    - HyperSync service
    - Redis (if available)
    - MLflow (if available)

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested system health")

    services = {}
    all_healthy = True

    # Check Orchestration Service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ORCHESTRATION_SERVICE_URL}/health")
            services["orchestration"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": ORCHESTRATION_SERVICE_URL,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "details": response.json() if response.status_code == 200 else None
            }
            if response.status_code != 200:
                all_healthy = False
    except Exception as e:
        services["orchestration"] = {
            "status": "unreachable",
            "url": ORCHESTRATION_SERVICE_URL,
            "error": str(e)
        }
        all_healthy = False

    # Check AutoML Service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{AUTOML_SERVICE_URL}/health")
            services["automl"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": AUTOML_SERVICE_URL,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "details": response.json() if response.status_code == 200 else None
            }
            if response.status_code != 200:
                all_healthy = False
    except Exception as e:
        services["automl"] = {
            "status": "unreachable",
            "url": AUTOML_SERVICE_URL,
            "error": str(e)
        }
        all_healthy = False

    # Check HyperSync Service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{HYPERSYNC_SERVICE_URL}/health")
            services["hypersync"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": HYPERSYNC_SERVICE_URL,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "details": response.json() if response.status_code == 200 else None
            }
            if response.status_code != 200:
                all_healthy = False
    except Exception as e:
        services["hypersync"] = {
            "status": "unreachable",
            "url": HYPERSYNC_SERVICE_URL,
            "error": str(e)
        }
        all_healthy = False

    return SystemHealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=services,
        timestamp=datetime.utcnow(),
        overall_healthy=all_healthy
    )


@router.get("/health/metrics")
@rate_limit
async def get_health_metrics(current_user: User = Depends(require_developer)) -> Dict[str, Any]:
    """
    Get detailed health metrics from all services

    Includes:
    - CPU/memory usage
    - Request rates
    - Error rates
    - Response times

    Requires: Developer or Admin role
    """
    logger.info(f"User {current_user.username} requested health metrics")

    metrics = {}

    # Fetch metrics from each service
    services = [
        ("orchestration", ORCHESTRATION_SERVICE_URL),
        ("automl", AUTOML_SERVICE_URL),
        ("hypersync", HYPERSYNC_SERVICE_URL)
    ]

    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, service_url in services:
            try:
                response = await client.get(f"{service_url}/metrics")
                if response.status_code == 200:
                    # Parse Prometheus metrics (simplified)
                    metrics[service_name] = {
                        "available": True,
                        "metrics_endpoint": f"{service_url}/metrics",
                        "raw": response.text[:1000]  # First 1000 chars
                    }
                else:
                    metrics[service_name] = {
                        "available": False,
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                metrics[service_name] = {
                    "available": False,
                    "error": str(e)
                }

    return {
        "services": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# JWT Key Management Endpoints
# ============================================================================

@router.get("/jwt/status", response_model=JWTStatusResponse)
@rate_limit
async def get_jwt_status(current_user: User = Depends(require_admin)) -> JWTStatusResponse:
    """
    Get current JWT key status

    Returns information about:
    - Current signing key
    - All active signing keys
    - All valid verification keys

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested JWT key status")

    if not JWT_KEY_STORE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT key store not available"
        )

    try:
        current_kid = jwt_key_store.get_current_kid()
        active_keys = jwt_key_store.list_active()
        valid_keys = jwt_key_store.list_valid()

        # Convert to response models (exclude secrets)
        active_key_responses = [
            JWTKeyResponse(
                kid=key.kid,
                algorithm=key.algorithm,
                created_at=key.created_at,
                expires_at=key.expires_at,
                is_active=key.is_active,
                is_valid=key.is_valid
            )
            for key in active_keys
        ]

        valid_key_responses = [
            JWTKeyResponse(
                kid=key.kid,
                algorithm=key.algorithm,
                created_at=key.created_at,
                expires_at=key.expires_at,
                is_active=key.is_active,
                is_valid=key.is_valid
            )
            for key in valid_keys
        ]

        return JWTStatusResponse(
            current_kid=current_kid or "none",
            active_keys=active_key_responses,
            valid_keys=valid_key_responses,
            total_active=len(active_keys),
            total_valid=len(valid_keys)
        )

    except Exception as e:
        logger.error(f"Failed to get JWT key status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get JWT status: {str(e)}"
        )


@router.post("/jwt/rotate", response_model=JWTRotationResponse)
@rate_limit
async def rotate_jwt_key(current_user: User = Depends(require_admin)) -> JWTRotationResponse:
    """
    Rotate JWT signing key

    This creates a new signing key and marks the old key as inactive.
    Old tokens remain valid during the grace period (default 24h).

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} initiated JWT key rotation")

    if not JWT_KEY_STORE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT key store not available"
        )

    try:
        # Get old kid
        old_kid = jwt_key_store.get_current_kid()

        # Rotate key
        new_key = jwt_key_store.rotate_key()

        # Log audit event
        audit_logger = get_audit_logger()
        await audit_logger.log_event(
            event_type=AuditEventType.JWT_ROTATION,
            username=current_user.username,
            severity=AuditEventSeverity.HIGH,
            metadata={
                "old_kid": old_kid,
                "new_kid": new_key.kid,
                "grace_period_hours": jwt_key_store.grace_period_hours
            },
            ip_address="internal"
        )

        return JWTRotationResponse(
            success=True,
            old_kid=old_kid,
            new_kid=new_key.kid,
            message=f"JWT key rotated successfully. Old tokens valid for {jwt_key_store.grace_period_hours}h.",
            timestamp=datetime.utcnow(),
            grace_period_hours=jwt_key_store.grace_period_hours
        )

    except Exception as e:
        logger.error(f"Failed to rotate JWT key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rotate JWT key: {str(e)}"
        )


@router.get("/jwt/keys")
@rate_limit
async def list_jwt_keys(current_user: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    List all JWT keys (active and valid)

    Returns detailed information about all keys in the system.
    Secrets are NOT included in the response.

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested JWT keys list")

    if not JWT_KEY_STORE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT key store not available"
        )

    try:
        active_keys = jwt_key_store.list_active()
        valid_keys = jwt_key_store.list_valid()

        # Combine and deduplicate
        all_keys_dict = {}
        for key in active_keys + valid_keys:
            if key.kid not in all_keys_dict:
                all_keys_dict[key.kid] = {
                    "kid": key.kid,
                    "algorithm": key.algorithm,
                    "created_at": key.created_at.isoformat(),
                    "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                    "is_active": key.is_active,
                    "is_valid": key.is_valid
                }

        return {
            "keys": list(all_keys_dict.values()),
            "current_kid": jwt_key_store.get_current_kid(),
            "total": len(all_keys_dict)
        }

    except Exception as e:
        logger.error(f"Failed to list JWT keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list JWT keys: {str(e)}"
        )


@router.post("/jwt/keys/{kid}/invalidate")
@rate_limit
async def invalidate_jwt_key(
    kid: str,
    request: JWTInvalidateRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Force invalidate a JWT key

    This immediately marks a key as invalid, preventing it from verifying tokens.
    Use this to revoke compromised keys.

    WARNING: This will invalidate all tokens signed with this key!

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} invalidating JWT key {kid}: {request.reason}")

    if not JWT_KEY_STORE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT key store not available"
        )

    try:
        # Get key
        jwt_key = jwt_key_store.get_key(kid)
        if jwt_key is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"JWT key {kid} not found"
            )

        # Mark as invalid
        jwt_key.is_valid = False
        jwt_key.is_active = False

        # Update in store
        if jwt_key_store.redis_client:
            try:
                jwt_key_store.redis_client.set(
                    jwt_key_store._key_path(kid),
                    jwt_key.json()
                )
                jwt_key_store.redis_client.srem(jwt_key_store._active_set_key(), kid)
                jwt_key_store.redis_client.srem(jwt_key_store._valid_set_key(), kid)
            except Exception as e:
                logger.error(f"Failed to update key in Redis: {e}")

        # Update in memory
        jwt_key_store.memory_keys[kid] = jwt_key

        # Log audit event
        audit_logger = get_audit_logger()
        await audit_logger.log_event(
            event_type=AuditEventType.JWT_KEY_INVALIDATED,
            username=current_user.username,
            severity=AuditEventSeverity.CRITICAL,
            metadata={
                "kid": kid,
                "reason": request.reason
            },
            ip_address="internal"
        )

        return {
            "success": True,
            "kid": kid,
            "message": f"JWT key {kid} invalidated successfully",
            "reason": request.reason,
            "invalidated_at": datetime.utcnow().isoformat(),
            "invalidated_by": current_user.username
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invalidate JWT key {kid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate JWT key: {str(e)}"
        )


# ============================================================================
# Audit Log Endpoints
# ============================================================================

@router.get("/audit/logs")
@rate_limit
async def get_audit_logs(
    event_type: Optional[str] = None,
    username: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get audit logs with optional filters

    Query parameters:
    - event_type: Filter by event type (e.g., "auth.login.success")
    - username: Filter by username
    - start_time: Filter by start time (ISO format)
    - end_time: Filter by end time (ISO format)
    - limit: Maximum number of events to return (default: 100)
    - offset: Number of events to skip (default: 0)

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested audit logs")

    audit_logger = get_audit_logger()

    # Parse timestamps
    start_dt = None
    end_dt = None

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_time format. Use ISO format (e.g., 2025-11-14T10:00:00)"
            )

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_time format. Use ISO format (e.g., 2025-11-14T10:00:00)"
            )

    # Parse event type
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = AuditEventType(event_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid event_type. Valid values: {[e.value for e in AuditEventType]}"
            )

    # Query audit logs
    events = await audit_logger.get_events(
        event_type=event_type_enum,
        username=username,
        start_time=start_dt,
        end_time=end_dt,
        limit=limit,
        offset=offset
    )

    return {
        "events": events,
        "total": len(events),
        "limit": limit,
        "offset": offset,
        "filters": {
            "event_type": event_type,
            "username": username,
            "start_time": start_time,
            "end_time": end_time
        }
    }


@router.get("/audit/stats")
@rate_limit
async def get_audit_stats(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get audit log statistics

    Query parameters:
    - start_time: Filter by start time (ISO format)
    - end_time: Filter by end time (ISO format)

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested audit stats")

    audit_logger = get_audit_logger()

    # Parse timestamps
    start_dt = None
    end_dt = None

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_time format"
            )

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_time format"
            )

    # Get stats
    stats = await audit_logger.get_stats(
        start_time=start_dt,
        end_time=end_dt
    )

    return {
        "stats": stats,
        "period": {
            "start_time": start_time,
            "end_time": end_time
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/audit/event-types")
@rate_limit
async def get_audit_event_types(current_user: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    Get list of all available audit event types

    Requires: Admin role
    """
    logger.info(f"Admin {current_user.username} requested audit event types")

    event_types = [
        {
            "value": event.value,
            "name": event.name,
            "category": event.value.split('.')[0]
        }
        for event in AuditEventType
    ]

    # Group by category
    categories = {}
    for event in event_types:
        category = event["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(event)

    return {
        "event_types": event_types,
        "categories": categories,
        "total": len(event_types)
    }

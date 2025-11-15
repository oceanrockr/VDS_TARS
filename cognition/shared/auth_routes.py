"""
T.A.R.S. Authentication Routes
Common auth endpoints for all services
Phase 11.5
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from .auth import (
    auth_service,
    authenticate_user,
    get_current_user,
    User,
    Role,
    require_admin
)
from .rate_limiter import auth_rate_limit


# Request/Response models
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    user_id: str
    username: str
    roles: List[str]
    email: Optional[str] = None


class APIKeyRequest(BaseModel):
    service_name: str


class APIKeyResponse(BaseModel):
    key_id: str
    api_key: str
    service_name: str
    created_at: datetime
    message: str


class APIKeyRotateResponse(BaseModel):
    key_id: str
    new_api_key: str
    service_name: str
    message: str


# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=LoginResponse)
@auth_rate_limit
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT tokens

    Rate limited to 10 requests per minute per IP
    """
    user = authenticate_user(request.username, request.password)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Generate tokens
    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=auth_service.config.jwt_expiry_minutes * 60,
        user={
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "email": user.email
        }
    )


@router.post("/refresh", response_model=RefreshResponse)
@auth_rate_limit
async def refresh(request: RefreshRequest):
    """
    Refresh access token using refresh token

    Rate limited to 10 requests per minute per IP
    """
    try:
        # Verify refresh token
        token_data = auth_service.verify_token(request.refresh_token)

        # Create new user object from token data
        user = User(
            user_id=token_data.user_id,
            username=token_data.username,
            roles=token_data.roles
        )

        # Generate new access token
        access_token = auth_service.create_access_token(user)

        return RefreshResponse(
            access_token=access_token,
            expires_in=auth_service.config.jwt_expiry_minutes * 60
        )

    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information

    Requires valid JWT token in Authorization header
    """
    return UserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        roles=[role.value for role in current_user.roles],
        email=current_user.email
    )


@router.post("/service-token", response_model=APIKeyResponse)
@require_admin
async def create_service_token(
    request: APIKeyRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new API key for service-to-service authentication

    Requires admin role
    """
    # Generate new API key
    raw_key, api_key = auth_service.generate_api_key(request.service_name)

    return APIKeyResponse(
        key_id=api_key.key_id,
        api_key=raw_key,
        service_name=api_key.service_name,
        created_at=api_key.created_at,
        message=f"API key created for {request.service_name}. Store this key securely - it cannot be retrieved again."
    )


@router.post("/service-token/{key_id}/rotate", response_model=APIKeyRotateResponse)
@require_admin
async def rotate_service_token(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Rotate an existing API key

    Requires admin role
    """
    result = auth_service.rotate_api_key(key_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    raw_key, new_api_key = result

    return APIKeyRotateResponse(
        key_id=new_api_key.key_id,
        new_api_key=raw_key,
        service_name=new_api_key.service_name,
        message=f"API key rotated for {new_api_key.service_name}. Old key has been deactivated."
    )


@router.delete("/service-token/{key_id}")
@require_admin
async def revoke_service_token(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Revoke an API key

    Requires admin role
    """
    success = auth_service.revoke_api_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    return {
        "message": f"API key {key_id} has been revoked",
        "key_id": key_id
    }


@router.get("/service-tokens")
@require_admin
async def list_service_tokens(current_user: User = Depends(get_current_user)):
    """
    List all API keys (without revealing the actual keys)

    Requires admin role
    """
    keys = []

    for key_id, api_key in auth_service.config.api_keys.items():
        keys.append({
            "key_id": api_key.key_id,
            "service_name": api_key.service_name,
            "created_at": api_key.created_at.isoformat(),
            "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            "is_active": api_key.is_active
        })

    return {
        "keys": keys,
        "total": len(keys)
    }

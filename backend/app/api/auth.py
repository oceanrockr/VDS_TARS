"""
T.A.R.S. Authentication Router
JWT token generation, refresh, and revocation endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import Dict

from fastapi import APIRouter, HTTPException, status, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models.auth import (
    TokenRequest,
    TokenResponse,
    TokenRefreshRequest,
    TokenValidationResponse,
    TokenRevokeRequest,
)
from ..core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    blacklist_token,
)
from ..core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security scheme
security = HTTPBearer()


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Access Token",
    description="Generate a new JWT access token and refresh token for a client",
)
async def generate_token(request: TokenRequest) -> TokenResponse:
    """
    Generate a new JWT access token and refresh token.

    Phase 2: Basic client_id based authentication
    Phase 4+: Will integrate with proper user authentication

    Args:
        request: Token request with client identification

    Returns:
        TokenResponse with access_token and refresh_token
    """
    try:
        # Create token payload
        token_data = {
            "sub": request.client_id,
            "device_name": request.device_name,
            "device_type": request.device_type,
            "issued_at": datetime.utcnow().isoformat(),
        }

        # Generate tokens
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token({"sub": request.client_id})

        # Calculate expiration
        expires_in = settings.JWT_EXPIRATION_HOURS * 3600

        logger.info(
            f"Generated tokens for client: {request.client_id} "
            f"({request.device_type}/{request.device_name})"
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
        )

    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate token",
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh Access Token",
    description="Exchange a refresh token for a new access token",
)
async def refresh_token(request: TokenRefreshRequest) -> TokenResponse:
    """
    Refresh an access token using a refresh token.

    Args:
        request: Token refresh request with refresh_token

    Returns:
        TokenResponse with new access_token and refresh_token
    """
    try:
        # Verify refresh token
        payload = verify_token(request.refresh_token)

        # Check token type
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        # Create new tokens
        client_id = payload.get("sub")
        token_data = {
            "sub": client_id,
            "issued_at": datetime.utcnow().isoformat(),
        }

        access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token({"sub": client_id})

        # Blacklist old refresh token
        blacklist_token(request.refresh_token)

        # Calculate expiration
        expires_in = settings.JWT_EXPIRATION_HOURS * 3600

        logger.info(f"Refreshed token for client: {client_id}")

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=expires_in,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token",
        )


@router.post(
    "/validate",
    response_model=TokenValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate Token",
    description="Validate a JWT token and return its status",
)
async def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenValidationResponse:
    """
    Validate a JWT token.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        TokenValidationResponse with validation status
    """
    try:
        token = credentials.credentials
        payload = verify_token(token)

        return TokenValidationResponse(
            valid=True,
            client_id=payload.get("sub"),
            expires_at=datetime.fromtimestamp(payload.get("exp")).isoformat(),
        )

    except HTTPException:
        return TokenValidationResponse(
            valid=False,
            client_id=None,
            expires_at=None,
        )


@router.post(
    "/revoke",
    status_code=status.HTTP_200_OK,
    summary="Revoke Token",
    description="Revoke a JWT token (add to blacklist)",
)
async def revoke_token(
    request: TokenRevokeRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, str]:
    """
    Revoke a JWT token by adding it to the blacklist.

    Args:
        request: Token revocation request
        credentials: Bearer token from Authorization header

    Returns:
        Success message
    """
    try:
        # Verify the requesting token
        verify_token(credentials.credentials)

        # Blacklist the target token
        blacklist_token(request.token)

        logger.info("Token revoked successfully")

        return {"message": "Token revoked successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token revocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke token",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Auth Service Health",
    description="Check authentication service health",
)
async def auth_health() -> Dict[str, str]:
    """Authentication service health check"""
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": datetime.utcnow().isoformat(),
    }

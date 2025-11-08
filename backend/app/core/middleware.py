"""
T.A.R.S. Middleware
Authentication and authorization middleware for REST and WebSocket endpoints
"""

import logging
from typing import Optional, Callable
from fastapi import HTTPException, status, WebSocket, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .security import verify_token

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to get the current authenticated user from JWT token.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        Token payload containing user information

    Raises:
        HTTPException: If token is invalid or missing
    """
    try:
        token = credentials.credentials
        payload = verify_token(token)

        # Check token type (should be access token)
        if payload.get("type") == "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Cannot use refresh token for authentication",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_websocket_token(websocket: WebSocket) -> Optional[dict]:
    """
    Verify JWT token for WebSocket connections.

    The token can be provided in:
    1. Query parameter: ?token=<jwt>
    2. Subprotocol header: Sec-WebSocket-Protocol: bearer, <jwt>

    Args:
        websocket: WebSocket connection

    Returns:
        Token payload if valid, None otherwise
    """
    try:
        # Try to get token from query parameters
        token = websocket.query_params.get("token")

        # If not in query params, try to get from headers/subprotocols
        if not token:
            # Check Sec-WebSocket-Protocol header
            subprotocols = websocket.headers.get("sec-websocket-protocol", "")
            if "bearer" in subprotocols.lower():
                parts = subprotocols.split(",")
                if len(parts) >= 2:
                    token = parts[1].strip()

        if not token:
            logger.warning("WebSocket connection attempted without token")
            return None

        # Verify the token
        payload = verify_token(token)

        # Check token type
        if payload.get("type") == "refresh":
            logger.warning("WebSocket connection attempted with refresh token")
            return None

        logger.info(f"WebSocket authenticated for client: {payload.get('sub')}")
        return payload

    except Exception as e:
        logger.error(f"WebSocket token verification failed: {e}")
        return None


def requires_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication for REST endpoints.

    Usage:
        @router.get("/protected")
        @requires_auth
        async def protected_endpoint(current_user: dict = Depends(get_current_user)):
            return {"user": current_user}
    """
    async def wrapper(*args, **kwargs):
        # The actual authentication is done by the get_current_user dependency
        return await func(*args, **kwargs)

    return wrapper

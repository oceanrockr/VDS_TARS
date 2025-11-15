"""
T.A.R.S. Security Middleware
Phase 6: JWT RBAC, Rate Limiting, HTTPS Enforcement & Security Headers

Features:
- Role-Based Access Control (admin/user scopes)
- Rate limiting per client (30 requests/min default)
- HTTPS enforcement middleware
- CSP and CORS headers
- Request validation for sensitive endpoints
"""

import time
import logging
from typing import Dict, Optional, Set, Callable
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.config import settings

logger = logging.getLogger(__name__)


# ==============================================================================
# RATE LIMITING MIDDLEWARE
# ==============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for per-client request throttling.

    Attributes:
        rate_limit: Maximum requests per minute per client
        buckets: Dict tracking request timestamps per client
    """

    def __init__(self, rate_limit: int = 30):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Maximum requests per minute (default: 30)
        """
        self.rate_limit = rate_limit
        self.buckets: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = 300  # Clean up old entries every 5 minutes
        self.last_cleanup = time.time()

    def is_allowed(self, client_id: str) -> tuple[bool, Optional[int]]:
        """
        Check if a client is allowed to make a request.

        Args:
            client_id: Unique identifier for the client (IP or token)

        Returns:
            Tuple of (allowed: bool, retry_after: Optional[int])
        """
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Periodic cleanup
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(window_start)

        # Get client's request history
        requests = self.buckets[client_id]

        # Remove requests outside the time window
        requests[:] = [req_time for req_time in requests if req_time > window_start]

        # Check if limit exceeded
        if len(requests) >= self.rate_limit:
            oldest_request = min(requests)
            retry_after = int(60 - (now - oldest_request))
            return False, retry_after

        # Add current request
        requests.append(now)
        return True, None

    def _cleanup_old_entries(self, threshold: float):
        """Remove old entries to prevent memory bloat"""
        self.buckets = {
            client_id: [t for t in times if t > threshold]
            for client_id, times in self.buckets.items()
            if any(t > threshold for t in times)
        }
        self.last_cleanup = time.time()
        logger.debug(f"Rate limiter cleanup: {len(self.buckets)} active clients")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting requests.

    Applies rate limiting per client IP or authenticated user.
    """

    def __init__(self, app: ASGIApp, rate_limit: int = None):
        super().__init__(app)
        limit = rate_limit or getattr(settings, 'RATE_LIMIT_PER_MINUTE', 30)
        self.limiter = RateLimiter(rate_limit=limit)
        self.exempt_paths: Set[str] = {
            "/health",
            "/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and apply rate limiting"""

        # Exempt certain paths from rate limiting
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get client identifier (IP or authenticated user)
        client_id = self._get_client_id(request)

        # Check rate limit
        allowed, retry_after = self.limiter.is_allowed(client_id)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for client {client_id} - "
                f"Path: {request.url.path} - Retry after: {retry_after}s"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please retry after {retry_after} seconds.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(
            self.limiter.rate_limit - len(self.limiter.buckets.get(client_id, []))
        )

        return response

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Try to get authenticated user from token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Use token hash as identifier (more stable than IP)
            token = auth_header[7:]
            return f"user:{hash(token)}"

        # Fall back to client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"


# ==============================================================================
# SECURITY HEADERS MIDDLEWARE
# ==============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for security headers and HTTPS enforcement.

    Features:
    - HTTPS redirect enforcement
    - Security headers (CSP, HSTS, X-Frame-Options, etc.)
    - CORS validation
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.enforce_https = getattr(settings, 'ENABLE_HTTPS', False)
        self.allowed_origins = self._parse_allowed_origins()

    def _parse_allowed_origins(self) -> Set[str]:
        """Parse allowed origins from settings"""
        origins_str = getattr(settings, 'ALLOWED_ORIGINS', '')
        if not origins_str:
            origins_str = getattr(settings, 'CORS_ORIGINS', '')

        origins = {origin.strip() for origin in origins_str.split(',') if origin.strip()}
        return origins

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add security headers"""

        # HTTPS enforcement (skip in local development)
        if self.enforce_https and not self._is_secure(request):
            if request.method == "GET":
                https_url = request.url.replace(scheme="https")
                return Response(
                    status_code=status.HTTP_301_MOVED_PERMANENTLY,
                    headers={"Location": str(https_url)},
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "HTTPS Required",
                        "message": "This endpoint requires HTTPS connection",
                    },
                )

        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response, request)

        return response

    def _is_secure(self, request: Request) -> bool:
        """Check if request is over HTTPS"""
        # Check X-Forwarded-Proto header (set by load balancers/proxies)
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        if forwarded_proto == "https":
            return True

        # Check request scheme
        if request.url.scheme == "https":
            return True

        # Allow localhost for development
        host = request.headers.get("Host", "")
        if host.startswith("localhost") or host.startswith("127.0.0.1"):
            return True

        return False

    def _add_security_headers(self, response: Response, request: Request):
        """Add comprehensive security headers"""

        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Relaxed for Swagger UI
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' ws: wss: https:",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS (only for HTTPS)
        if self._is_secure(request):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Permissions policy (restrict features)
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"


# ==============================================================================
# JWT ROLE-BASED ACCESS CONTROL
# ==============================================================================

class RBACMiddleware:
    """
    Role-Based Access Control utilities for JWT tokens.

    Note: This is not a middleware but a utility class for RBAC checks.
    Use these functions in your endpoint dependencies.
    """

    @staticmethod
    def get_admin_client_ids() -> Set[str]:
        """Get set of admin client IDs from config"""
        admin_ids_str = getattr(settings, 'ADMIN_CLIENT_IDS', '')
        admin_ids = {cid.strip() for cid in admin_ids_str.split(',') if cid.strip()}
        return admin_ids

    @staticmethod
    def is_admin(client_id: str) -> bool:
        """Check if a client ID has admin privileges"""
        admin_ids = RBACMiddleware.get_admin_client_ids()
        return client_id in admin_ids

    @staticmethod
    def require_admin(client_id: str):
        """
        Raise HTTPException if client is not an admin.

        Usage in endpoints:
            RBACMiddleware.require_admin(token_data.client_id)
        """
        if not RBACMiddleware.is_admin(client_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for this operation",
            )

    @staticmethod
    def get_user_scopes(client_id: str) -> Set[str]:
        """
        Get user scopes based on client ID.

        Returns:
            Set of scope strings (e.g., {"read", "write"} or {"admin"})
        """
        if RBACMiddleware.is_admin(client_id):
            return {"admin", "read", "write", "delete"}
        else:
            return {"read", "write"}

    @staticmethod
    def has_scope(client_id: str, required_scope: str) -> bool:
        """Check if user has a specific scope"""
        user_scopes = RBACMiddleware.get_user_scopes(client_id)
        return required_scope in user_scopes


# ==============================================================================
# REQUEST VALIDATION
# ==============================================================================

def validate_analytics_export_path(file_path: str) -> bool:
    """
    Validate analytics export file path to prevent path traversal.

    Args:
        file_path: Requested export file path

    Returns:
        True if valid, raises HTTPException otherwise
    """
    import os
    from pathlib import Path

    # Normalize path
    try:
        normalized = os.path.normpath(file_path)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path format",
        )

    # Check for path traversal
    if ".." in normalized or normalized.startswith("/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal not allowed",
        )

    # Ensure file extension is allowed
    allowed_extensions = {".csv", ".json", ".xlsx"}
    if not any(normalized.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {allowed_extensions}",
        )

    return True


# ==============================================================================
# USAGE EXAMPLES (for documentation)
# ==============================================================================

"""
Example 1: Adding middleware to FastAPI app
--------------------------------------------
from app.middleware.security import SecurityMiddleware, RateLimitMiddleware

app = FastAPI()
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware, rate_limit=30)


Example 2: Using RBAC in endpoints
-----------------------------------
from fastapi import Depends
from app.middleware.security import RBACMiddleware
from app.api.auth import get_current_user

@app.delete("/admin/delete-all")
async def admin_only_endpoint(token_data = Depends(get_current_user)):
    RBACMiddleware.require_admin(token_data.client_id)
    # Admin-only logic here
    return {"message": "Admin operation successful"}


Example 3: Checking scopes
---------------------------
@app.post("/sensitive-operation")
async def sensitive_op(token_data = Depends(get_current_user)):
    if not RBACMiddleware.has_scope(token_data.client_id, "write"):
        raise HTTPException(403, detail="Write permission required")
    # Operation logic here
    return {"status": "success"}


Example 4: Validating file paths
---------------------------------
from app.middleware.security import validate_analytics_export_path

@app.post("/analytics/export")
async def export_analytics(file_path: str):
    validate_analytics_export_path(file_path)
    # Export logic here
    return {"export_path": file_path}
"""

"""
T.A.R.S. Security Headers Middleware
Configurable security headers middleware for FastAPI applications

This middleware adds industry-standard security headers to all HTTP responses
to protect against common web vulnerabilities including XSS, clickjacking,
MIME-type confusion, and other attacks.

Features:
- Comprehensive security headers (X-Content-Type-Options, X-XSS-Protection, etc.)
- Configurable Content Security Policy (CSP)
- Strict Transport Security (HSTS) for HTTPS connections
- Optional header customization
- Type-safe implementation with full type hints
"""

from typing import Optional, Dict, Set, Callable
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class SecurityHeadersConfig:
    """
    Configuration class for SecurityHeadersMiddleware.

    Attributes:
        enable_hsts: Enable HTTP Strict Transport Security header
        enable_csp: Enable Content Security Policy header
        enable_x_frame_options: Enable X-Frame-Options header
        enable_x_content_type_options: Enable X-Content-Type-Options header
        enable_x_xss_protection: Enable X-XSS-Protection header
        custom_csp: Custom CSP directives (overrides default)
        csp_directives: Default CSP directives
        x_frame_options: X-Frame-Options value (DENY, SAMEORIGIN, or ALLOW-FROM uri)
        hsts_max_age: HSTS max-age in seconds (default: 1 year)
        hsts_include_subdomains: Include subdomains in HSTS
        hsts_preload: Enable HSTS preload
        referrer_policy: Referrer-Policy header value
        permissions_policy: Permissions-Policy header value
    """

    def __init__(
        self,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        enable_x_frame_options: bool = True,
        enable_x_content_type_options: bool = True,
        enable_x_xss_protection: bool = True,
        custom_csp: Optional[str] = None,
        x_frame_options: str = "DENY",
        hsts_max_age: int = 31536000,  # 1 year
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = False,
        referrer_policy: Optional[str] = "strict-origin-when-cross-origin",
        permissions_policy: Optional[str] = None,
    ):
        """
        Initialize security headers configuration.

        Args:
            enable_hsts: Enable HSTS header (only applied to HTTPS)
            enable_csp: Enable Content Security Policy header
            enable_x_frame_options: Enable X-Frame-Options header
            enable_x_content_type_options: Enable X-Content-Type-Options header
            enable_x_xss_protection: Enable X-XSS-Protection header
            custom_csp: Custom CSP string (if None, uses default directives)
            x_frame_options: X-Frame-Options value (DENY, SAMEORIGIN)
            hsts_max_age: HSTS max-age in seconds
            hsts_include_subdomains: Include subdomains in HSTS
            hsts_preload: Enable HSTS preload directive
            referrer_policy: Referrer-Policy header value
            permissions_policy: Permissions-Policy header value
        """
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.enable_x_frame_options = enable_x_frame_options
        self.enable_x_content_type_options = enable_x_content_type_options
        self.enable_x_xss_protection = enable_x_xss_protection
        self.custom_csp = custom_csp
        self.x_frame_options = x_frame_options
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy

        # Default CSP directives (used when custom_csp is None)
        self.default_csp_directives = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]

    def get_csp_header_value(self) -> str:
        """
        Get the Content Security Policy header value.

        Returns:
            CSP header value string
        """
        if self.custom_csp:
            return self.custom_csp
        return "; ".join(self.default_csp_directives)

    def get_hsts_header_value(self) -> str:
        """
        Get the Strict-Transport-Security header value.

        Returns:
            HSTS header value string
        """
        hsts_parts = [f"max-age={self.hsts_max_age}"]

        if self.hsts_include_subdomains:
            hsts_parts.append("includeSubDomains")

        if self.hsts_preload:
            hsts_parts.append("preload")

        return "; ".join(hsts_parts)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for adding security headers to all HTTP responses.

    This middleware protects against common web vulnerabilities by adding
    security-related HTTP headers to all responses. It is highly configurable
    and can be customized per application requirements.

    Security Headers Added:
        - X-Content-Type-Options: Prevents MIME-type sniffing
        - X-XSS-Protection: Enables browser XSS protection (legacy)
        - X-Frame-Options: Prevents clickjacking attacks
        - Content-Security-Policy: Defines content sources policy
        - Strict-Transport-Security: Enforces HTTPS (for HTTPS only)
        - Referrer-Policy: Controls referrer information
        - Permissions-Policy: Controls browser features

    Example:
        >>> from fastapi import FastAPI
        >>> from backend.app.middleware.security_headers import (
        ...     SecurityHeadersMiddleware,
        ...     SecurityHeadersConfig
        ... )
        >>>
        >>> app = FastAPI()
        >>> config = SecurityHeadersConfig(
        ...     custom_csp="default-src 'self'; script-src 'self' 'unsafe-inline'",
        ...     enable_hsts=True
        ... )
        >>> app.add_middleware(SecurityHeadersMiddleware, config=config)
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[SecurityHeadersConfig] = None
    ):
        """
        Initialize the SecurityHeadersMiddleware.

        Args:
            app: ASGI application instance
            config: SecurityHeadersConfig instance (uses defaults if None)
        """
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()
        logger.info("SecurityHeadersMiddleware initialized with configuration")
        self._log_configuration()

    def _log_configuration(self) -> None:
        """Log the current security headers configuration."""
        logger.debug("Security Headers Configuration:")
        logger.debug(f"  - HSTS: {'Enabled' if self.config.enable_hsts else 'Disabled'}")
        logger.debug(f"  - CSP: {'Enabled' if self.config.enable_csp else 'Disabled'}")
        logger.debug(f"  - X-Frame-Options: {'Enabled' if self.config.enable_x_frame_options else 'Disabled'}")
        logger.debug(f"  - X-Content-Type-Options: {'Enabled' if self.config.enable_x_content_type_options else 'Disabled'}")
        logger.debug(f"  - X-XSS-Protection: {'Enabled' if self.config.enable_x_xss_protection else 'Disabled'}")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request and add security headers to the response.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response with security headers added
        """
        # Process the request through the application
        response = await call_next(request)

        # Add security headers to the response
        self._add_security_headers(response, request)

        return response

    def _add_security_headers(self, response: Response, request: Request) -> None:
        """
        Add security headers to the response.

        Args:
            response: HTTP response object
            request: HTTP request object (used to check if HTTPS)
        """
        # X-Content-Type-Options: Prevent MIME-type sniffing
        if self.config.enable_x_content_type_options:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection: Enable browser XSS filter (legacy, but still useful)
        if self.config.enable_x_xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"

        # X-Frame-Options: Prevent clickjacking
        if self.config.enable_x_frame_options:
            response.headers["X-Frame-Options"] = self.config.x_frame_options

        # Content-Security-Policy: Define allowed content sources
        if self.config.enable_csp:
            response.headers["Content-Security-Policy"] = self.config.get_csp_header_value()

        # Strict-Transport-Security: Enforce HTTPS (only for HTTPS connections)
        if self.config.enable_hsts and self._is_secure_connection(request):
            response.headers["Strict-Transport-Security"] = self.config.get_hsts_header_value()

        # Referrer-Policy: Control referrer information
        if self.config.referrer_policy:
            response.headers["Referrer-Policy"] = self.config.referrer_policy

        # Permissions-Policy: Control browser features
        if self.config.permissions_policy:
            response.headers["Permissions-Policy"] = self.config.permissions_policy

    def _is_secure_connection(self, request: Request) -> bool:
        """
        Check if the request is over a secure HTTPS connection.

        This method checks multiple indicators of a secure connection:
        1. X-Forwarded-Proto header (set by reverse proxies/load balancers)
        2. Request URL scheme
        3. Localhost/127.0.0.1 (considered secure for development)

        Args:
            request: HTTP request object

        Returns:
            True if connection is secure, False otherwise
        """
        # Check X-Forwarded-Proto header (common in production with load balancers)
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        if forwarded_proto.lower() == "https":
            return True

        # Check request URL scheme
        if request.url.scheme == "https":
            return True

        # Consider localhost connections as secure (for development)
        host = request.headers.get("Host", "")
        if host.startswith("localhost") or host.startswith("127.0.0.1"):
            return True

        return False


# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

class SecurityHeadersPresets:
    """
    Preset configurations for common security header scenarios.

    This class provides ready-to-use configurations for different
    application types and security requirements.
    """

    @staticmethod
    def strict() -> SecurityHeadersConfig:
        """
        Strict security configuration for maximum protection.

        Suitable for:
        - Production applications
        - Applications handling sensitive data
        - High-security requirements

        Returns:
            SecurityHeadersConfig with strict settings
        """
        return SecurityHeadersConfig(
            enable_hsts=True,
            enable_csp=True,
            enable_x_frame_options=True,
            enable_x_content_type_options=True,
            enable_x_xss_protection=True,
            x_frame_options="DENY",
            hsts_max_age=63072000,  # 2 years
            hsts_include_subdomains=True,
            hsts_preload=True,
            referrer_policy="no-referrer",
            permissions_policy="geolocation=(), microphone=(), camera=(), payment=()",
        )

    @staticmethod
    def relaxed() -> SecurityHeadersConfig:
        """
        Relaxed security configuration for development/testing.

        Suitable for:
        - Development environments
        - Testing environments
        - Applications with less strict security requirements

        Returns:
            SecurityHeadersConfig with relaxed settings
        """
        return SecurityHeadersConfig(
            enable_hsts=False,  # Disabled for local development
            enable_csp=True,
            enable_x_frame_options=True,
            enable_x_content_type_options=True,
            enable_x_xss_protection=True,
            custom_csp="default-src 'self' 'unsafe-inline' 'unsafe-eval'",
            x_frame_options="SAMEORIGIN",
            referrer_policy="strict-origin-when-cross-origin",
        )

    @staticmethod
    def api_only() -> SecurityHeadersConfig:
        """
        Configuration optimized for API-only applications.

        Suitable for:
        - RESTful APIs
        - GraphQL APIs
        - Microservices

        Returns:
            SecurityHeadersConfig optimized for APIs
        """
        return SecurityHeadersConfig(
            enable_hsts=True,
            enable_csp=False,  # CSP less relevant for APIs
            enable_x_frame_options=True,
            enable_x_content_type_options=True,
            enable_x_xss_protection=False,  # Not needed for JSON APIs
            x_frame_options="DENY",
            hsts_max_age=31536000,  # 1 year
            hsts_include_subdomains=True,
            referrer_policy="no-referrer",
        )

    @staticmethod
    def swagger_compatible() -> SecurityHeadersConfig:
        """
        Configuration compatible with Swagger/OpenAPI documentation.

        This configuration relaxes CSP to allow Swagger UI to function
        while maintaining other security headers.

        Suitable for:
        - FastAPI applications with /docs endpoint
        - Applications using Swagger UI
        - Development with API documentation

        Returns:
            SecurityHeadersConfig compatible with Swagger UI
        """
        return SecurityHeadersConfig(
            enable_hsts=True,
            enable_csp=True,
            enable_x_frame_options=True,
            enable_x_content_type_options=True,
            enable_x_xss_protection=True,
            custom_csp=(
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self'"
            ),
            x_frame_options="SAMEORIGIN",
            hsts_max_age=31536000,
            hsts_include_subdomains=True,
            referrer_policy="strict-origin-when-cross-origin",
        )


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

"""
Example 1: Basic usage with default configuration
--------------------------------------------------
from fastapi import FastAPI
from backend.app.middleware.security_headers import SecurityHeadersMiddleware

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)


Example 2: Custom configuration
--------------------------------
from fastapi import FastAPI
from backend.app.middleware.security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig
)

app = FastAPI()
config = SecurityHeadersConfig(
    custom_csp="default-src 'self'; script-src 'self' 'unsafe-inline'",
    enable_hsts=True,
    hsts_max_age=31536000,
    x_frame_options="DENY"
)
app.add_middleware(SecurityHeadersMiddleware, config=config)


Example 3: Using presets
-------------------------
from fastapi import FastAPI
from backend.app.middleware.security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersPresets
)

app = FastAPI()

# For production
app.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.strict()
)

# For development
app.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.relaxed()
)

# For Swagger-enabled APIs
app.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.swagger_compatible()
)


Example 4: Disabling specific headers
--------------------------------------
from fastapi import FastAPI
from backend.app.middleware.security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig
)

app = FastAPI()
config = SecurityHeadersConfig(
    enable_x_xss_protection=False,  # Disable X-XSS-Protection
    enable_csp=False,  # Disable CSP
    enable_hsts=True,  # Keep HSTS enabled
)
app.add_middleware(SecurityHeadersMiddleware, config=config)


Example 5: Advanced CSP configuration
--------------------------------------
from fastapi import FastAPI
from backend.app.middleware.security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig
)

app = FastAPI()
custom_csp = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.example.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "img-src 'self' data: https:; "
    "font-src 'self' https://fonts.gstatic.com; "
    "connect-src 'self' wss: https:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

config = SecurityHeadersConfig(custom_csp=custom_csp)
app.add_middleware(SecurityHeadersMiddleware, config=config)
"""

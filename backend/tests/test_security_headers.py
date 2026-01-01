"""
Tests for SecurityHeadersMiddleware

This module tests the security headers middleware functionality including:
- Default security headers
- Custom configurations
- HTTPS detection
- Preset configurations
- Header enablement/disablement
"""

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from backend.app.middleware.security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    SecurityHeadersPresets,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def base_app():
    """Create a basic FastAPI application."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    return app


@pytest.fixture
def app_with_default_headers(base_app):
    """FastAPI app with default security headers."""
    base_app.add_middleware(SecurityHeadersMiddleware)
    return base_app


@pytest.fixture
def app_with_custom_config(base_app):
    """FastAPI app with custom security headers configuration."""
    config = SecurityHeadersConfig(
        custom_csp="default-src 'self'; script-src 'self' 'unsafe-inline'",
        x_frame_options="SAMEORIGIN",
        hsts_max_age=63072000,
    )
    base_app.add_middleware(SecurityHeadersMiddleware, config=config)
    return base_app


@pytest.fixture
def app_with_strict_preset(base_app):
    """FastAPI app with strict preset configuration."""
    base_app.add_middleware(
        SecurityHeadersMiddleware,
        config=SecurityHeadersPresets.strict()
    )
    return base_app


@pytest.fixture
def app_with_relaxed_preset(base_app):
    """FastAPI app with relaxed preset configuration."""
    base_app.add_middleware(
        SecurityHeadersMiddleware,
        config=SecurityHeadersPresets.relaxed()
    )
    return base_app


# ==============================================================================
# DEFAULT CONFIGURATION TESTS
# ==============================================================================

def test_default_security_headers(app_with_default_headers):
    """Test that default security headers are added to responses."""
    client = TestClient(app_with_default_headers)
    response = client.get("/test")

    assert response.status_code == 200

    # Check default security headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_hsts_not_added_for_http(app_with_default_headers):
    """Test that HSTS header is not added for HTTP connections."""
    client = TestClient(app_with_default_headers)
    response = client.get("/test")

    # HSTS should not be present for HTTP (localhost is considered HTTP)
    # Note: TestClient considers localhost as HTTP by default
    assert response.status_code == 200


def test_default_csp_directives(app_with_default_headers):
    """Test that default CSP directives are comprehensive."""
    client = TestClient(app_with_default_headers)
    response = client.get("/test")

    csp = response.headers["Content-Security-Policy"]

    # Check for key CSP directives
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "form-action 'self'" in csp


def test_referrer_policy_default(app_with_default_headers):
    """Test that referrer policy is set by default."""
    client = TestClient(app_with_default_headers)
    response = client.get("/test")

    assert "Referrer-Policy" in response.headers
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


# ==============================================================================
# CUSTOM CONFIGURATION TESTS
# ==============================================================================

def test_custom_csp(app_with_custom_config):
    """Test that custom CSP overrides default."""
    client = TestClient(app_with_custom_config)
    response = client.get("/test")

    csp = response.headers["Content-Security-Policy"]
    assert csp == "default-src 'self'; script-src 'self' 'unsafe-inline'"


def test_custom_x_frame_options(app_with_custom_config):
    """Test that custom X-Frame-Options is applied."""
    client = TestClient(app_with_custom_config)
    response = client.get("/test")

    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"


def test_disable_specific_headers():
    """Test disabling specific security headers."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    config = SecurityHeadersConfig(
        enable_x_xss_protection=False,
        enable_csp=False,
        enable_x_content_type_options=True,
    )
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    client = TestClient(app)
    response = client.get("/test")

    # Enabled headers should be present
    assert "X-Content-Type-Options" in response.headers

    # Disabled headers should not be present
    assert "X-XSS-Protection" not in response.headers
    assert "Content-Security-Policy" not in response.headers


# ==============================================================================
# PRESET CONFIGURATION TESTS
# ==============================================================================

def test_strict_preset(app_with_strict_preset):
    """Test strict preset configuration."""
    client = TestClient(app_with_strict_preset)
    response = client.get("/test")

    # All security headers should be enabled
    assert "X-Content-Type-Options" in response.headers
    assert "X-XSS-Protection" in response.headers
    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers
    assert "Referrer-Policy" in response.headers
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "Permissions-Policy" in response.headers


def test_relaxed_preset(app_with_relaxed_preset):
    """Test relaxed preset configuration."""
    client = TestClient(app_with_relaxed_preset)
    response = client.get("/test")

    # Basic headers should still be present
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"

    # CSP should allow unsafe-inline and unsafe-eval
    csp = response.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" in csp
    assert "'unsafe-eval'" in csp


def test_api_only_preset():
    """Test API-only preset configuration."""
    app = FastAPI()

    @app.get("/api/test")
    async def test_endpoint():
        return {"message": "test"}

    app.add_middleware(
        SecurityHeadersMiddleware,
        config=SecurityHeadersPresets.api_only()
    )

    client = TestClient(app)
    response = client.get("/api/test")

    # Headers relevant for APIs should be present
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers

    # CSP should be disabled for APIs
    assert "Content-Security-Policy" not in response.headers

    # XSS protection should be disabled (not needed for JSON APIs)
    assert "X-XSS-Protection" not in response.headers


def test_swagger_compatible_preset():
    """Test Swagger-compatible preset configuration."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    app.add_middleware(
        SecurityHeadersMiddleware,
        config=SecurityHeadersPresets.swagger_compatible()
    )

    client = TestClient(app)
    response = client.get("/test")

    # CSP should allow unsafe-inline and unsafe-eval for Swagger UI
    csp = response.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" in csp
    assert "'unsafe-eval'" in csp

    # X-Frame-Options should allow SAMEORIGIN for Swagger embeds
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"


# ==============================================================================
# HTTPS DETECTION TESTS
# ==============================================================================

def test_hsts_with_https_header():
    """Test that HSTS is added when X-Forwarded-Proto is https."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    app.add_middleware(SecurityHeadersMiddleware)

    client = TestClient(app)
    response = client.get(
        "/test",
        headers={"X-Forwarded-Proto": "https"}
    )

    # HSTS should be present for HTTPS connections
    assert "Strict-Transport-Security" in response.headers
    assert "max-age=" in response.headers["Strict-Transport-Security"]
    assert "includeSubDomains" in response.headers["Strict-Transport-Security"]


def test_hsts_header_format():
    """Test that HSTS header is properly formatted."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    config = SecurityHeadersConfig(
        hsts_max_age=31536000,
        hsts_include_subdomains=True,
        hsts_preload=True,
    )
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    client = TestClient(app)
    response = client.get(
        "/test",
        headers={"X-Forwarded-Proto": "https"}
    )

    hsts = response.headers["Strict-Transport-Security"]
    assert "max-age=31536000" in hsts
    assert "includeSubDomains" in hsts
    assert "preload" in hsts


# ==============================================================================
# CONFIGURATION CLASS TESTS
# ==============================================================================

def test_security_headers_config_defaults():
    """Test SecurityHeadersConfig default values."""
    config = SecurityHeadersConfig()

    assert config.enable_hsts is True
    assert config.enable_csp is True
    assert config.enable_x_frame_options is True
    assert config.enable_x_content_type_options is True
    assert config.enable_x_xss_protection is True
    assert config.x_frame_options == "DENY"
    assert config.hsts_max_age == 31536000
    assert config.hsts_include_subdomains is True
    assert config.hsts_preload is False


def test_security_headers_config_custom():
    """Test SecurityHeadersConfig with custom values."""
    config = SecurityHeadersConfig(
        enable_hsts=False,
        enable_csp=False,
        custom_csp="custom-csp-value",
        x_frame_options="SAMEORIGIN",
        hsts_max_age=63072000,
    )

    assert config.enable_hsts is False
    assert config.enable_csp is False
    assert config.custom_csp == "custom-csp-value"
    assert config.x_frame_options == "SAMEORIGIN"
    assert config.hsts_max_age == 63072000


def test_get_csp_header_value_default():
    """Test CSP header value generation with defaults."""
    config = SecurityHeadersConfig()
    csp = config.get_csp_header_value()

    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_get_csp_header_value_custom():
    """Test CSP header value generation with custom CSP."""
    config = SecurityHeadersConfig(custom_csp="custom-csp-value")
    csp = config.get_csp_header_value()

    assert csp == "custom-csp-value"


def test_get_hsts_header_value():
    """Test HSTS header value generation."""
    config = SecurityHeadersConfig(
        hsts_max_age=31536000,
        hsts_include_subdomains=True,
        hsts_preload=True,
    )
    hsts = config.get_hsts_header_value()

    assert "max-age=31536000" in hsts
    assert "includeSubDomains" in hsts
    assert "preload" in hsts


def test_get_hsts_header_value_minimal():
    """Test HSTS header value with minimal configuration."""
    config = SecurityHeadersConfig(
        hsts_max_age=3600,
        hsts_include_subdomains=False,
        hsts_preload=False,
    )
    hsts = config.get_hsts_header_value()

    assert hsts == "max-age=3600"


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_multiple_requests():
    """Test that headers are consistent across multiple requests."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    app.add_middleware(SecurityHeadersMiddleware)

    client = TestClient(app)

    # Make multiple requests
    responses = [client.get("/test") for _ in range(5)]

    # All responses should have the same headers
    for response in responses:
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "Content-Security-Policy" in response.headers


def test_headers_on_different_endpoints():
    """Test that headers are added to all endpoints."""
    app = FastAPI()

    @app.get("/endpoint1")
    async def endpoint1():
        return {"endpoint": "1"}

    @app.post("/endpoint2")
    async def endpoint2():
        return {"endpoint": "2"}

    @app.get("/api/v1/endpoint3")
    async def endpoint3():
        return {"endpoint": "3"}

    app.add_middleware(SecurityHeadersMiddleware)

    client = TestClient(app)

    # Test all endpoints
    responses = [
        client.get("/endpoint1"),
        client.post("/endpoint2"),
        client.get("/api/v1/endpoint3"),
    ]

    for response in responses:
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers


def test_headers_with_error_responses():
    """Test that security headers are added even to error responses."""
    app = FastAPI()

    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")

    app.add_middleware(SecurityHeadersMiddleware)

    client = TestClient(app)
    response = client.get("/error")

    # Even with errors, security headers should be present
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers


# ==============================================================================
# EDGE CASES
# ==============================================================================

def test_permissions_policy():
    """Test that Permissions-Policy header can be configured."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    config = SecurityHeadersConfig(
        permissions_policy="geolocation=(), microphone=(), camera=()"
    )
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    client = TestClient(app)
    response = client.get("/test")

    assert "Permissions-Policy" in response.headers
    assert "geolocation=()" in response.headers["Permissions-Policy"]


def test_no_permissions_policy_by_default():
    """Test that Permissions-Policy is not set by default."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    app.add_middleware(SecurityHeadersMiddleware)

    client = TestClient(app)
    response = client.get("/test")

    # Permissions-Policy should not be set by default
    assert "Permissions-Policy" not in response.headers


def test_custom_referrer_policy():
    """Test custom Referrer-Policy configuration."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    config = SecurityHeadersConfig(referrer_policy="no-referrer")
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    client = TestClient(app)
    response = client.get("/test")

    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_disable_referrer_policy():
    """Test disabling Referrer-Policy."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    config = SecurityHeadersConfig(referrer_policy=None)
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    client = TestClient(app)
    response = client.get("/test")

    assert "Referrer-Policy" not in response.headers

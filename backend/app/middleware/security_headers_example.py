"""
Security Headers Middleware - Example Usage

This script demonstrates how to use the SecurityHeadersMiddleware
in a FastAPI application with various configurations.

Run this script to see the middleware in action:
    python -m uvicorn backend.app.middleware.security_headers_example:app --reload
"""

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from .security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    SecurityHeadersPresets,
)


# ==============================================================================
# EXAMPLE 1: Basic Usage (Default Configuration)
# ==============================================================================

app_basic = FastAPI(title="Example 1: Basic Security Headers")
app_basic.add_middleware(SecurityHeadersMiddleware)


@app_basic.get("/")
async def root_basic():
    return {
        "example": "Basic Security Headers",
        "description": "All default security headers are applied",
    }


# ==============================================================================
# EXAMPLE 2: Custom Configuration
# ==============================================================================

app_custom = FastAPI(title="Example 2: Custom Security Headers")

custom_config = SecurityHeadersConfig(
    custom_csp="default-src 'self'; script-src 'self' 'unsafe-inline'",
    x_frame_options="SAMEORIGIN",
    hsts_max_age=63072000,  # 2 years
    referrer_policy="no-referrer",
    permissions_policy="geolocation=(), microphone=(), camera=()",
)

app_custom.add_middleware(SecurityHeadersMiddleware, config=custom_config)


@app_custom.get("/")
async def root_custom():
    return {
        "example": "Custom Security Headers",
        "description": "Custom CSP, HSTS, and other headers",
    }


# ==============================================================================
# EXAMPLE 3: Strict Preset (Production)
# ==============================================================================

app_strict = FastAPI(title="Example 3: Strict Security Headers (Production)")
app_strict.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.strict()
)


@app_strict.get("/")
async def root_strict():
    return {
        "example": "Strict Security Headers",
        "description": "Maximum security for production",
    }


# ==============================================================================
# EXAMPLE 4: Relaxed Preset (Development)
# ==============================================================================

app_relaxed = FastAPI(title="Example 4: Relaxed Security Headers (Development)")
app_relaxed.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.relaxed()
)


@app_relaxed.get("/")
async def root_relaxed():
    return {
        "example": "Relaxed Security Headers",
        "description": "Relaxed settings for development",
    }


# ==============================================================================
# EXAMPLE 5: API-Only Preset
# ==============================================================================

app_api = FastAPI(title="Example 5: API-Only Security Headers")
app_api.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.api_only()
)


@app_api.get("/api/data")
async def api_data():
    return {
        "example": "API-Only Security Headers",
        "description": "Optimized for JSON APIs",
        "data": [1, 2, 3, 4, 5],
    }


# ==============================================================================
# EXAMPLE 6: Swagger-Compatible Preset
# ==============================================================================

app_swagger = FastAPI(
    title="Example 6: Swagger-Compatible Security Headers",
    description="Security headers that work with Swagger UI",
)

app_swagger.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.swagger_compatible()
)


@app_swagger.get("/")
async def root_swagger():
    return {
        "example": "Swagger-Compatible Security Headers",
        "description": "Works with /docs endpoint",
        "docs": "/docs",
    }


@app_swagger.get("/data")
async def get_data():
    """Example endpoint for Swagger UI"""
    return {"message": "Data endpoint", "value": 42}


# ==============================================================================
# EXAMPLE 7: Selective Headers
# ==============================================================================

app_selective = FastAPI(title="Example 7: Selective Security Headers")

selective_config = SecurityHeadersConfig(
    enable_x_xss_protection=False,  # Disable XSS protection
    enable_csp=False,  # Disable CSP
    enable_x_content_type_options=True,  # Keep this enabled
    enable_x_frame_options=True,  # Keep this enabled
    enable_hsts=True,  # Keep this enabled
)

app_selective.add_middleware(SecurityHeadersMiddleware, config=selective_config)


@app_selective.get("/")
async def root_selective():
    return {
        "example": "Selective Security Headers",
        "description": "Only specific headers enabled",
        "enabled": ["X-Content-Type-Options", "X-Frame-Options", "HSTS"],
        "disabled": ["X-XSS-Protection", "CSP"],
    }


# ==============================================================================
# EXAMPLE 8: Headers Inspector
# ==============================================================================

app = FastAPI(
    title="Security Headers Inspector",
    description="Test and inspect security headers in responses",
    version="1.0.0",
)

# Use Swagger-compatible preset for this demo
app.add_middleware(
    SecurityHeadersMiddleware,
    config=SecurityHeadersPresets.swagger_compatible()
)


@app.get("/")
async def root():
    """
    Root endpoint - provides information about available examples
    """
    return {
        "service": "Security Headers Inspector",
        "description": "Test security headers middleware",
        "endpoints": {
            "/inspect": "Inspect current security headers",
            "/test": "Test endpoint with sample data",
            "/docs": "Swagger UI documentation",
        },
    }


@app.get("/inspect")
async def inspect_headers():
    """
    Inspect endpoint - returns information about security headers

    NOTE: The actual headers are added to the HTTP response,
    not in the JSON body. Use browser DevTools or curl to see them.
    """
    return {
        "message": "Security headers are in the HTTP response headers",
        "tip": "Use browser DevTools (Network tab) or curl -v to inspect headers",
        "expected_headers": [
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "X-Frame-Options",
            "Content-Security-Policy",
            "Strict-Transport-Security (HTTPS only)",
            "Referrer-Policy",
        ],
    }


@app.get("/test")
async def test_endpoint():
    """
    Test endpoint with sample data
    """
    return {
        "status": "success",
        "data": {
            "id": 123,
            "name": "Test Item",
            "values": [10, 20, 30, 40, 50],
        },
        "timestamp": "2025-12-26T00:00:00Z",
    }


@app.post("/test")
async def test_post():
    """
    Test POST endpoint
    """
    return {
        "status": "success",
        "message": "POST request processed",
        "note": "Security headers are also added to POST responses",
    }


@app.get("/headers/config")
async def show_config():
    """
    Show the current security headers configuration
    """
    return {
        "preset": "swagger_compatible",
        "description": "Allows Swagger UI to function while maintaining security",
        "features": {
            "hsts": "Enabled for HTTPS",
            "csp": "Relaxed for Swagger UI",
            "x_frame_options": "SAMEORIGIN",
            "x_content_type_options": "nosniff",
            "x_xss_protection": "1; mode=block",
            "referrer_policy": "strict-origin-when-cross-origin",
        },
    }


# ==============================================================================
# TESTING INSTRUCTIONS
# ==============================================================================

"""
To test this example:

1. Run the application:
   uvicorn backend.app.middleware.security_headers_example:app --reload

2. Open your browser to http://localhost:8000

3. Inspect the headers using one of these methods:

   a) Browser DevTools:
      - Open DevTools (F12)
      - Go to Network tab
      - Reload the page
      - Click on the request
      - View Response Headers

   b) Using curl:
      curl -v http://localhost:8000/inspect

   c) Using httpie:
      http -v http://localhost:8000/inspect

4. Visit the Swagger UI at http://localhost:8000/docs
   - The Swagger UI should work correctly with the security headers

5. Expected headers in the response:
   X-Content-Type-Options: nosniff
   X-XSS-Protection: 1; mode=block
   X-Frame-Options: SAMEORIGIN
   Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' ...
   Referrer-Policy: strict-origin-when-cross-origin

6. For HTTPS testing:
   - Use a reverse proxy (nginx) or
   - Deploy to a platform with HTTPS
   - Or use the X-Forwarded-Proto header:
     curl -v -H "X-Forwarded-Proto: https" http://localhost:8000/inspect
"""

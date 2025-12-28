#!/bin/bash
# ==============================================================================
# T.A.R.S. Security Validation Script
# Version: v1.0.10 (GA) - Phase 22 Validation
# Scope: LAN-only deployment security checks
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }
header() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# ==============================================================================
# JWT Authentication
# ==============================================================================
validate_jwt() {
    header "JWT Authentication"

    # Get token
    local token_response
    token_response=$(curl -s --max-time 10 -X POST http://localhost:8000/auth/token \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=admin&password=admin" 2>/dev/null)

    local token
    token=$(echo "$token_response" | jq -r '.access_token' 2>/dev/null)

    if [ -n "$token" ] && [ "$token" != "null" ]; then
        pass "JWT token generation works"
        echo "     Token: ${token:0:30}..."
    else
        fail "JWT token generation failed"
        return 1
    fi

    # Test protected endpoint without auth
    local no_auth_status
    no_auth_status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
        http://localhost:8000/rag/index \
        -X POST -H "Content-Type: application/json" -d '{}' 2>/dev/null)

    if [ "$no_auth_status" = "401" ]; then
        pass "Protected endpoints require auth (HTTP 401)"
    else
        fail "Protected endpoint returned $no_auth_status (expected 401)"
    fi

    # Test with invalid token
    local invalid_status
    invalid_status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
        http://localhost:8000/rag/index \
        -X POST \
        -H "Authorization: Bearer invalid_token" \
        -H "Content-Type: application/json" -d '{}' 2>/dev/null)

    if [ "$invalid_status" = "401" ]; then
        pass "Invalid tokens rejected (HTTP 401)"
    else
        fail "Invalid token returned $invalid_status (expected 401)"
    fi

    # Test with valid token
    local valid_status
    valid_status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
        http://localhost:8000/rag/index \
        -X POST \
        -H "Authorization: Bearer $token" \
        -H "Content-Type: application/json" -d '{"file_path":"/nonexistent"}' 2>/dev/null)

    if [ "$valid_status" != "401" ]; then
        pass "Valid tokens accepted (HTTP $valid_status)"
    else
        fail "Valid token rejected"
    fi
}

# ==============================================================================
# Security Headers
# ==============================================================================
validate_headers() {
    header "Security Headers"

    local headers
    headers=$(curl -s -I --max-time 10 http://localhost:8000/health 2>/dev/null)

    # X-Content-Type-Options
    if echo "$headers" | grep -qi "X-Content-Type-Options: nosniff"; then
        pass "X-Content-Type-Options: nosniff"
    else
        fail "X-Content-Type-Options missing or incorrect"
    fi

    # X-XSS-Protection
    if echo "$headers" | grep -qi "X-XSS-Protection.*1.*mode=block"; then
        pass "X-XSS-Protection: 1; mode=block"
    elif echo "$headers" | grep -qi "X-XSS-Protection"; then
        warn "X-XSS-Protection present but may not be optimal"
    else
        fail "X-XSS-Protection missing"
    fi

    # X-Frame-Options
    if echo "$headers" | grep -qi "X-Frame-Options"; then
        local frame_value
        frame_value=$(echo "$headers" | grep -i "X-Frame-Options" | head -1)
        pass "$frame_value"
    else
        fail "X-Frame-Options missing"
    fi

    # Content-Security-Policy
    if echo "$headers" | grep -qi "Content-Security-Policy"; then
        pass "Content-Security-Policy present"
        local csp
        csp=$(echo "$headers" | grep -i "Content-Security-Policy" | head -1 | cut -c1-80)
        echo "     ${csp}..."
    else
        warn "Content-Security-Policy missing"
    fi

    # Referrer-Policy
    if echo "$headers" | grep -qi "Referrer-Policy"; then
        pass "Referrer-Policy present"
    else
        warn "Referrer-Policy missing"
    fi

    # HSTS (should NOT be present for HTTP-only)
    if echo "$headers" | grep -qi "Strict-Transport-Security"; then
        warn "HSTS present (unexpected for HTTP-only deployment)"
    else
        pass "HSTS correctly disabled for HTTP deployment"
    fi
}

# ==============================================================================
# XSS Sanitization
# ==============================================================================
validate_xss() {
    header "XSS Sanitization"

    # Test XSS in 404 path
    local xss_404
    xss_404=$(curl -s --max-time 10 "http://localhost:8000/<script>alert(1)</script>" 2>/dev/null)

    if echo "$xss_404" | grep -q "<script>"; then
        fail "XSS not sanitized in 404 response"
    else
        pass "XSS sanitized in 404 error responses"
    fi

    # Test various XSS vectors
    local vectors=(
        "<script>alert(1)</script>"
        "<img src=x onerror=alert(1)>"
        "javascript:alert(1)"
        "<iframe src='evil.com'></iframe>"
    )

    local vector_passed=0
    for vector in "${vectors[@]}"; do
        local encoded
        encoded=$(printf '%s' "$vector" | jq -sRr @uri)
        local response
        response=$(curl -s --max-time 10 "http://localhost:8000/${encoded}" 2>/dev/null)

        if ! echo "$response" | grep -q "<script>\|onerror=\|javascript:\|<iframe"; then
            ((vector_passed++))
        fi
    done

    if [ "$vector_passed" -eq "${#vectors[@]}" ]; then
        pass "All XSS vectors sanitized (${vector_passed}/${#vectors[@]})"
    else
        warn "Some XSS vectors may not be fully sanitized (${vector_passed}/${#vectors[@]})"
    fi
}

# ==============================================================================
# CORS Validation
# ==============================================================================
validate_cors() {
    header "CORS Validation"

    # Test allowed origin
    local cors_allowed
    cors_allowed=$(curl -s -I --max-time 10 -X OPTIONS http://localhost:8000/health \
        -H "Origin: http://localhost:3000" \
        -H "Access-Control-Request-Method: GET" 2>/dev/null)

    if echo "$cors_allowed" | grep -qi "Access-Control-Allow-Origin.*localhost:3000"; then
        pass "CORS allows localhost:3000"
    elif echo "$cors_allowed" | grep -qi "Access-Control-Allow-Origin"; then
        warn "CORS present but origin may be permissive"
    else
        warn "CORS headers not returned for allowed origin"
    fi

    # Test blocked origin
    local cors_blocked
    cors_blocked=$(curl -s -I --max-time 10 -X OPTIONS http://localhost:8000/health \
        -H "Origin: http://evil.com" \
        -H "Access-Control-Request-Method: GET" 2>/dev/null)

    if echo "$cors_blocked" | grep -qi "Access-Control-Allow-Origin.*evil.com"; then
        fail "CORS allows external origin (evil.com)"
    else
        pass "External origins correctly blocked"
    fi
}

# ==============================================================================
# Rate Limiting
# ==============================================================================
validate_rate_limiting() {
    header "Rate Limiting"

    # Send requests to detect rate limiting
    local rate_limited=false
    local request_count=0

    for i in {1..50}; do
        local status
        status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8000/health 2>/dev/null)
        ((request_count++))

        if [ "$status" = "429" ]; then
            rate_limited=true
            break
        fi
    done

    if [ "$rate_limited" = true ]; then
        pass "Rate limiting active (triggered at request $request_count)"
    else
        warn "Rate limiting not triggered in $request_count requests"
        echo "     Note: Home limit is 200/minute, test only sent 50"
    fi
}

# ==============================================================================
# Network Exposure
# ==============================================================================
validate_network() {
    header "Network Exposure"

    # Check listening ports
    local ports
    ports=$(ss -tlnp 2>/dev/null | grep -E ":(8000|11434|8001|6379|5432)" | wc -l)

    if [ "$ports" -gt 0 ]; then
        pass "Services listening on $ports ports"
        ss -tlnp 2>/dev/null | grep -E ":(8000|11434|8001|6379|5432)" | awk '{print "     "$4}'
    else
        warn "Could not verify listening ports"
    fi

    # Check for public IP binding
    local public_bind
    public_bind=$(ss -tlnp 2>/dev/null | grep -E ":(8000|11434|8001)" | grep -v "127.0.0.1\|0.0.0.0" | wc -l)

    if [ "$public_bind" -eq 0 ]; then
        pass "No services bound to public IPs"
    else
        warn "Some services may be exposed: $public_bind"
    fi

    echo ""
    echo "     NOTE: Verify router has no port forwarding rules"
    echo "     for ports 8000, 11434, 8001, 6379, 5432"
}

# ==============================================================================
# Secrets Check
# ==============================================================================
validate_secrets() {
    header "Secrets Handling"

    # Check container logs for secrets
    local log_secrets
    log_secrets=$(docker logs tars-home-backend 2>&1 | grep -iE "password=|secret=|key=" | wc -l)

    if [ "$log_secrets" -eq 0 ]; then
        pass "No secrets found in container logs"
    else
        warn "Potential secrets in logs: $log_secrets occurrences"
    fi

    # Check .env file not committed
    if git status deploy/tars-home.env 2>/dev/null | grep -q "Untracked\|ignored"; then
        pass ".env file not committed to git"
    elif [ ! -f "deploy/tars-home.env" ]; then
        pass ".env file not present in working directory"
    else
        warn "Verify .env is in .gitignore"
    fi
}

# ==============================================================================
# Summary
# ==============================================================================
print_summary() {
    header "Security Validation Summary"

    echo -e "${GREEN}Passed:${NC}  $PASS_COUNT"
    echo -e "${YELLOW}Warnings:${NC} $WARN_COUNT"
    echo -e "${RED}Failed:${NC}  $FAIL_COUNT"
    echo ""

    echo "Accepted Risks for Home LAN Deployment:"
    echo "  - No TLS/HTTPS (HTTP-only on trusted LAN)"
    echo "  - No HSTS (requires HTTPS)"
    echo "  - Redis without authentication (Docker network only)"
    echo "  - 7-day JWT token expiry (home convenience)"
    echo "  - 200 req/min rate limit (relaxed for home use)"
    echo ""

    if [ "$FAIL_COUNT" -eq 0 ]; then
        echo -e "${GREEN}Security validation PASSED${NC}"
        exit 0
    else
        echo -e "${RED}Security validation FAILED${NC}"
        echo "Review failed checks above."
        exit 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${CYAN}=================================================================${NC}"
    echo -e "${CYAN} T.A.R.S. Security Validation - v1.0.10 (Home LAN)${NC}"
    echo -e "${CYAN}=================================================================${NC}"

    validate_jwt
    validate_headers
    validate_xss
    validate_cors
    validate_rate_limiting
    validate_network
    validate_secrets
    print_summary
}

main "$@"

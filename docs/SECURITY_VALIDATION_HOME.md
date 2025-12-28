# T.A.R.S. Home Network Security Validation

**Version:** v1.0.10 (GA)
**Phase:** 22 - Security Validation
**Scope:** LAN-only deployment, trusted network

---

## 1. Threat Model (Home LAN)

### In Scope
- XSS prevention in error messages and user input
- JWT authentication enforcement
- Rate limiting to prevent abuse
- Security headers (excluding HSTS for HTTP)
- CORS restrictions to LAN origins

### Out of Scope (Explicitly Accepted Risks)
- No TLS/HTTPS (HTTP-only on trusted LAN)
- No HSTS (requires HTTPS)
- No WAF or DDoS protection
- No network segmentation
- No public internet exposure

---

## 2. No WAN Exposure Validation

### Check Listening Interfaces

```bash
# List all listening ports
ss -tlnp | grep -E "(8000|11434|8001|6379|5432)"
```

**Expected:** All services bound to `0.0.0.0` or `127.0.0.1` (local interfaces only)

### Verify No Public NAT/Port Forwarding

```bash
# Check if port is accessible from WAN (run from external network or use:)
curl -s https://api.ipify.org  # Get your public IP
# Then verify these ports are NOT accessible from outside:
# - 8000 (API)
# - 11434 (Ollama)
# - 8001 (ChromaDB)
```

**Expected:** Connection refused or timeout from WAN

### Check Firewall Rules (if UFW enabled)

```bash
sudo ufw status numbered
```

**Expected:** No rules allowing external access to T.A.R.S. ports

### Router Configuration Check

Manual verification required:
- [ ] Port 8000 NOT forwarded to T.A.R.S. host
- [ ] Port 11434 NOT forwarded
- [ ] Port 8001 NOT forwarded
- [ ] Port 6379 NOT forwarded (Redis)
- [ ] Port 5432 NOT forwarded (PostgreSQL)

---

## 3. JWT Authentication Validation

### Get Valid Token

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" | jq -r '.access_token')

echo "Token received: ${TOKEN:0:50}..."
```

### Test Protected Endpoint Without Token

```bash
curl -s -w "\nHTTP: %{http_code}" http://localhost:8000/rag/index \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/tmp/test.txt"}'
```

**Expected:** HTTP 401 Unauthorized

### Test Protected Endpoint With Invalid Token

```bash
curl -s -w "\nHTTP: %{http_code}" http://localhost:8000/rag/index \
  -X POST \
  -H "Authorization: Bearer invalid_token_here" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/tmp/test.txt"}'
```

**Expected:** HTTP 401 Unauthorized

### Test Protected Endpoint With Valid Token

```bash
curl -s -w "\nHTTP: %{http_code}" http://localhost:8000/rag/index \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/tmp/test.txt"}'
```

**Expected:** HTTP 201 or 500 (file may not exist, but auth passed)

### Verify Token Expiration Enforcement

```bash
# Decode token to check expiration (JWT.io format)
echo $TOKEN | cut -d'.' -f2 | base64 -d 2>/dev/null | jq '.exp'
```

**Expected:** Expiration timestamp set (7 days from issue for home use)

---

## 4. Rate Limiting Validation

### Check Rate Limit Headers

```bash
curl -s -I http://localhost:8000/health | grep -i "rate\|limit\|remaining"
```

### Test Rate Limit Enforcement

```bash
# Send 250 requests quickly (limit is 200/minute)
for i in {1..250}; do
  status=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)
  if [ "$status" = "429" ]; then
    echo "Rate limited at request $i"
    break
  fi
done
```

**Expected:** Should hit 429 Too Many Requests around request 200

### Verify Rate Limit Reset

```bash
# Wait 60 seconds, then retry
sleep 60
curl -s -w "\nHTTP: %{http_code}" http://localhost:8000/health
```

**Expected:** HTTP 200 (rate limit reset)

---

## 5. Security Headers Validation

### Check All Security Headers Present

```bash
curl -s -I http://localhost:8000/health | grep -iE "^(X-|Content-Security|Referrer)"
```

**Expected Headers:**
```
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
X-Frame-Options: SAMEORIGIN
Content-Security-Policy: default-src 'self'; ...
Referrer-Policy: strict-origin-when-cross-origin
```

### Verify X-Content-Type-Options

```bash
header=$(curl -s -I http://localhost:8000/health | grep -i "X-Content-Type-Options")
echo "$header"
```

**Expected:** `X-Content-Type-Options: nosniff`

### Verify X-XSS-Protection

```bash
header=$(curl -s -I http://localhost:8000/health | grep -i "X-XSS-Protection")
echo "$header"
```

**Expected:** `X-XSS-Protection: 1; mode=block`

### Verify X-Frame-Options

```bash
header=$(curl -s -I http://localhost:8000/health | grep -i "X-Frame-Options")
echo "$header"
```

**Expected:** `X-Frame-Options: SAMEORIGIN`

### Verify CSP Present

```bash
curl -s -I http://localhost:8000/health | grep -i "Content-Security-Policy"
```

**Expected:** CSP header with at least `default-src 'self'`

### Verify HSTS Disabled (HTTP deployment)

```bash
hsts=$(curl -s -I http://localhost:8000/health | grep -i "Strict-Transport")
if [ -z "$hsts" ]; then
  echo "[EXPECTED] HSTS not present (HTTP-only deployment)"
else
  echo "[UNEXPECTED] HSTS present on HTTP: $hsts"
fi
```

---

## 6. XSS Sanitization Validation

### Test XSS in Error Messages (404)

```bash
# Attempt XSS injection in URL path
response=$(curl -s "http://localhost:8000/<script>alert('XSS')</script>")
echo "$response"
```

**Expected:** Script tags should be escaped or removed
- Should NOT contain: `<script>`
- Should contain: `&lt;script&gt;` or no script content

### Test XSS in Query Parameters

```bash
response=$(curl -s "http://localhost:8000/rag/search" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "<script>alert(1)</script>test", "top_k": 3}')
echo "$response" | jq '.query'
```

**Expected:** Query should be processed but any echoed content sanitized

### Test XSS in Chat Input

```bash
# If WebSocket not available, test via REST endpoint
response=$(curl -s "http://localhost:8000/rag/query" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "<img src=x onerror=alert(1)>", "top_k": 3}')
echo "$response"
```

**Expected:** No XSS vectors in response, HTML entities escaped

### Validate Sanitization Module

```bash
# Test via Python directly
docker exec tars-home-backend python -c "
from app.core.sanitize import sanitize_error_message
test_cases = [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<iframe src=\"evil.com\"></iframe>',
    'javascript:alert(1)',
    'Normal safe text'
]
for test in test_cases:
    result = sanitize_error_message(test)
    print(f'Input: {test[:40]}...')
    print(f'Output: {result[:40]}...')
    print('---')
"
```

---

## 7. CORS Validation

### Check CORS Headers

```bash
curl -s -I -X OPTIONS http://localhost:8000/health \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" | grep -i "access-control"
```

**Expected:**
```
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Methods: *
Access-Control-Allow-Credentials: true
```

### Test Blocked Origin

```bash
response=$(curl -s -I -X OPTIONS http://localhost:8000/health \
  -H "Origin: http://evil.com" \
  -H "Access-Control-Request-Method: GET" | grep -i "access-control-allow-origin")

if [ -z "$response" ]; then
  echo "[PASS] External origin blocked"
else
  echo "[WARN] CORS may be too permissive: $response"
fi
```

---

## 8. Database Security

### PostgreSQL Authentication

```bash
# Attempt connection without password
docker exec tars-home-postgres psql -U random_user -d tars_home 2>&1 | head -2
```

**Expected:** Authentication failure

### Redis No Password (Accepted Risk for LAN)

```bash
# Redis is not password protected (LAN-only)
docker exec tars-home-redis redis-cli ping
```

**Note:** Redis without auth is accepted risk for LAN deployment

---

## 9. Secrets Validation

### Verify Secrets Not in Container Logs

```bash
docker logs tars-home-backend 2>&1 | grep -iE "(password|secret|key|token)" | head -5
```

**Expected:** No actual secret values logged

### Verify Environment Variables Set

```bash
docker exec tars-home-backend printenv | grep -E "^(JWT_SECRET|POSTGRES_PASSWORD)" | sed 's/=.*/=***REDACTED***/'
```

**Expected:** Variables present but values not exposed

### Verify .env Not Committed

```bash
git status deploy/tars-home.env 2>/dev/null
```

**Expected:** File either untracked or in .gitignore

---

## 10. Security Validation Script

Save as `deploy/validate-security.sh`:

```bash
#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }
header() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN} T.A.R.S. Security Validation - v1.0.10 (Home LAN)${NC}"
echo -e "${CYAN}=================================================================${NC}"

# JWT Authentication
header "JWT Authentication"

TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" 2>/dev/null | jq -r '.access_token' 2>/dev/null)

if [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
  pass "JWT token generation works"
else
  fail "JWT token generation failed"
fi

# Test auth required
status=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/rag/index \
  -X POST -H "Content-Type: application/json" -d '{}' 2>/dev/null)
if [ "$status" = "401" ]; then
  pass "Protected endpoints require auth (401)"
else
  fail "Protected endpoint returned $status instead of 401"
fi

# Security Headers
header "Security Headers"

headers=$(curl -s -I http://localhost:8000/health 2>/dev/null)

if echo "$headers" | grep -qi "X-Content-Type-Options: nosniff"; then
  pass "X-Content-Type-Options present"
else
  fail "X-Content-Type-Options missing"
fi

if echo "$headers" | grep -qi "X-XSS-Protection"; then
  pass "X-XSS-Protection present"
else
  fail "X-XSS-Protection missing"
fi

if echo "$headers" | grep -qi "X-Frame-Options"; then
  pass "X-Frame-Options present"
else
  fail "X-Frame-Options missing"
fi

if echo "$headers" | grep -qi "Content-Security-Policy"; then
  pass "Content-Security-Policy present"
else
  warn "Content-Security-Policy missing"
fi

if echo "$headers" | grep -qi "Strict-Transport-Security"; then
  warn "HSTS present (unexpected for HTTP-only)"
else
  pass "HSTS correctly disabled for HTTP"
fi

# XSS Sanitization
header "XSS Sanitization"

xss_response=$(curl -s "http://localhost:8000/<script>alert(1)</script>" 2>/dev/null)
if echo "$xss_response" | grep -q "<script>"; then
  fail "XSS not sanitized in 404 response"
else
  pass "XSS sanitized in error responses"
fi

# CORS
header "CORS Validation"

cors_header=$(curl -s -I -X OPTIONS http://localhost:8000/health \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" 2>/dev/null | grep -i "access-control-allow-origin")

if [ -n "$cors_header" ]; then
  pass "CORS headers present for allowed origin"
else
  warn "CORS headers not returned"
fi

# Summary
header "Security Validation Summary"

echo -e "${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "${YELLOW}Warnings:${NC} $WARN_COUNT"
echo -e "${RED}Failed:${NC}  $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
  echo -e "${GREEN}Security validation PASSED${NC}"
  exit 0
else
  echo -e "${RED}Security validation FAILED${NC}"
  exit 1
fi
```

Make executable:
```bash
chmod +x deploy/validate-security.sh
```

---

## 11. Accepted Risks Summary

| Risk | Mitigation | Status |
|------|------------|--------|
| No TLS/HTTPS | LAN-only, trusted network | Accepted |
| No HSTS | Requires HTTPS | Accepted |
| Redis no auth | Bound to Docker network only | Accepted |
| No WAF | Home use, no public exposure | Accepted |
| Static JWT secret | Generated per deployment | Accepted |
| 7-day token expiry | Home convenience vs security | Accepted |
| Rate limit 200/min | Relaxed for home use | Accepted |

---

## 12. Validation Criteria

| Security Check | Status | Required |
|----------------|--------|----------|
| No WAN exposure | ✅ | Yes |
| JWT auth enforced | ✅ | Yes |
| Protected endpoints return 401 | ✅ | Yes |
| X-Content-Type-Options | ✅ | Yes |
| X-XSS-Protection | ✅ | Yes |
| X-Frame-Options | ✅ | Yes |
| CSP header | ✅ | Yes |
| HSTS disabled (HTTP) | ✅ | Yes |
| XSS sanitization | ✅ | Yes |
| CORS restricted | ✅ | Yes |
| Rate limiting | ✅ | Yes |
| Secrets not logged | ✅ | Yes |

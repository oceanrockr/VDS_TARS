# T.A.R.S. Enterprise API Guide

**Version:** v1.0.2-dev
**Last Updated:** 2025-11-26
**Base URL:** `http://localhost:8100` (default)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Authentication](#2-authentication)
3. [Endpoints](#3-endpoints)
4. [Response Models](#4-response-models)
5. [Error Handling](#5-error-handling)
6. [Rate Limiting](#6-rate-limiting)
7. [Code Examples](#7-code-examples)
8. [Integration Examples](#8-integration-examples)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start

### 1.1 Starting the API Server

**Local Development:**

```bash
# Start with default settings
python scripts/run_api_server.py

# Start with specific profile
python scripts/run_api_server.py --profile prod

# Start with custom port
python scripts/run_api_server.py --port 8200
```

**Using Docker:**

```bash
# Build and run
docker build -t tars-api .
docker run -p 8100:8100 -e TARS_PROFILE=prod tars-api
```

**Using Docker Compose:**

```bash
docker-compose up tars-api
```

### 1.2 First API Call

**Test Health Endpoint:**

```bash
curl http://localhost:8100/health

# Response:
# {
#   "status": "healthy",
#   "version": "v1.0.2-dev",
#   "timestamp": "2025-11-26T10:30:00Z"
# }
```

**Authenticate and Get GA KPI:**

```bash
# Login to get JWT token
TOKEN=$(curl -X POST http://localhost:8100/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "demo123"}' \
  | jq -r '.access_token')

# Get GA KPI with token
curl http://localhost:8100/ga \
  -H "Authorization: Bearer $TOKEN"
```

**Using API Key:**

```bash
# Get GA KPI with API key
curl http://localhost:8100/ga \
  -H "X-API-Key: dev-key-admin"
```

### 1.3 API Overview

**Available Endpoints:**

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | None | Health check |
| `/metrics` | GET | None | Prometheus metrics |
| `/auth/login` | POST | Credentials | Get JWT token |
| `/auth/refresh` | POST | Refresh token | Refresh JWT token |
| `/ga` | GET | Required | GA readiness KPI |
| `/summaries/daily` | GET | Required | Daily summaries (last 7 days) |
| `/summaries/daily/{date}` | GET | Required | Specific day summary |
| `/anomalies` | GET | Required | Recent anomalies (last 7 days) |
| `/anomalies/{date}` | GET | Required | Anomalies for specific date |
| `/regressions` | GET | Required | Recent regressions (last 7 days) |
| `/regressions/{date}` | GET | Required | Regressions for specific date |
| `/retrospective` | GET | Required | Generate retrospective report |
| `/retrospective/download` | GET | Required | Download retrospective as file |

---

## 2. Authentication

The T.A.R.S. API supports two authentication methods:

### 2.1 API Keys

**Static API keys** for service-to-service communication.

#### 2.1.1 Default API Keys

Development keys (change in production!):

| API Key | Role | Access Level |
|---------|------|--------------|
| `dev-key-admin` | admin | Full access (read + write + admin) |
| `dev-key-sre` | sre | Read + write access |
| `dev-key-readonly` | readonly | Read-only access |

#### 2.1.2 Using API Keys

**Header-based Authentication:**

```bash
curl http://localhost:8100/ga \
  -H "X-API-Key: dev-key-admin"
```

**Python Example:**

```python
import requests

response = requests.get(
    "http://localhost:8100/ga",
    headers={"X-API-Key": "dev-key-admin"}
)
print(response.json())
```

#### 2.1.3 Generating Custom API Keys

```python
from enterprise_api.auth import generate_api_key

# Generate new API key
api_key = generate_api_key(
    user="monitoring-service",
    role="readonly"
)
print(f"API Key: {api_key}")

# Add to config/api_keys.yaml
# api_keys:
#   - key: <generated_key>
#     user: monitoring-service
#     role: readonly
```

### 2.2 JWT Authentication

**JSON Web Tokens** for user authentication with expiration.

#### 2.2.1 Login Flow

**Step 1: Login to get tokens**

```bash
curl -X POST http://localhost:8100/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "demo123"
  }'

# Response:
# {
#   "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "token_type": "bearer",
#   "expires_in": 3600
# }
```

**Step 2: Use access token**

```bash
curl http://localhost:8100/ga \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Step 3: Refresh when expired**

```bash
curl -X POST http://localhost:8100/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'

# Response:
# {
#   "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "token_type": "bearer",
#   "expires_in": 3600
# }
```

#### 2.2.2 Token Expiration

| Token Type | Lifetime | Renewable |
|------------|----------|-----------|
| Access Token | 60 minutes | No (use refresh) |
| Refresh Token | 7 days | Yes (generates new access token) |

#### 2.2.3 Default Users

Development users (change in production!):

| Username | Password | Role | Access Level |
|----------|----------|------|--------------|
| `admin` | `demo123` | admin | Full access |
| `sre` | `demo123` | sre | Read + write |
| `viewer` | `demo123` | readonly | Read-only |

**Change default passwords:**

```bash
# Generate password hash
python -c "
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
print(pwd_context.hash('your_secure_password'))
"

# Update config/users.yaml
# users:
#   - username: admin
#     password_hash: <generated_hash>
#     role: admin
```

### 2.3 Role-Based Access Control (RBAC)

Three roles with different permission levels:

#### 2.3.1 Role Permissions Matrix

| Endpoint | readonly | sre | admin |
|----------|----------|-----|-------|
| `GET /health` | ✅ | ✅ | ✅ |
| `GET /metrics` | ✅ | ✅ | ✅ |
| `POST /auth/login` | ✅ | ✅ | ✅ |
| `GET /ga` | ✅ | ✅ | ✅ |
| `GET /summaries/*` | ✅ | ✅ | ✅ |
| `GET /anomalies/*` | ✅ | ✅ | ✅ |
| `GET /regressions/*` | ✅ | ✅ | ✅ |
| `GET /retrospective` | ✅ | ✅ | ✅ |
| `POST /admin/refresh-data` | ❌ | ✅ | ✅ |
| `POST /admin/clear-cache` | ❌ | ✅ | ✅ |
| `GET /admin/compliance` | ❌ | ❌ | ✅ |
| `POST /admin/users` | ❌ | ❌ | ✅ |

#### 2.3.2 Access Denied Example

```bash
# Viewer tries to access admin endpoint
curl http://localhost:8100/admin/compliance \
  -H "X-API-Key: dev-key-readonly"

# Response (403 Forbidden):
# {
#   "detail": "Insufficient permissions. Required role: admin, current role: readonly"
# }
```

---

## 3. Endpoints

### 3.1 Health & Metrics

#### 3.1.1 GET /health

**Health check endpoint (no authentication required).**

```bash
curl http://localhost:8100/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "v1.0.2-dev",
  "timestamp": "2025-11-26T10:30:00Z",
  "services": {
    "database": "healthy",
    "prometheus": "healthy",
    "compliance": "healthy"
  }
}
```

**Status Codes:**
- `200 OK` - Service healthy
- `503 Service Unavailable` - Service unhealthy

#### 3.1.2 GET /metrics

**Prometheus metrics endpoint (no authentication required).**

```bash
curl http://localhost:8100/metrics
```

**Response (Prometheus format):**

```
# HELP tars_api_requests_total Total API requests
# TYPE tars_api_requests_total counter
tars_api_requests_total{endpoint="/ga",method="GET",status="200"} 1234

# HELP tars_api_latency_seconds API request latency
# TYPE tars_api_latency_seconds histogram
tars_api_latency_seconds_bucket{endpoint="/ga",method="GET",le="0.1"} 1000
tars_api_latency_seconds_bucket{endpoint="/ga",method="GET",le="0.5"} 1200
tars_api_latency_seconds_sum{endpoint="/ga",method="GET"} 120.5
tars_api_latency_seconds_count{endpoint="/ga",method="GET"} 1234
```

### 3.2 Authentication Endpoints

#### 3.2.1 POST /auth/login

**Authenticate and receive JWT tokens.**

**Request:**

```bash
curl -X POST http://localhost:8100/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "demo123"
  }'
```

**Request Body:**

```json
{
  "username": "string",
  "password": "string"
}
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTcwMDAwMDAwMH0.signature",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInR5cGUiOiJyZWZyZXNoIiwiZXhwIjoxNzAwNjA0ODAwfQ.signature",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Error Response (401 Unauthorized):**

```json
{
  "detail": "Invalid credentials"
}
```

#### 3.2.2 POST /auth/refresh

**Refresh access token using refresh token.**

**Request:**

```bash
curl -X POST http://localhost:8100/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

**Request Body:**

```json
{
  "refresh_token": "string"
}
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 3.3 GA KPI Endpoint

#### 3.3.1 GET /ga

**Get General Availability (GA) readiness KPI.**

**Request:**

```bash
curl http://localhost:8100/ga \
  -H "X-API-Key: dev-key-readonly"
```

**Response (200 OK):**

```json
{
  "overall_availability": 99.95,
  "ga_ready": true,
  "threshold": 99.9,
  "period_start": "2025-11-19T00:00:00Z",
  "period_end": "2025-11-26T00:00:00Z",
  "metrics": {
    "uptime_hours": 167.92,
    "total_hours": 168.0,
    "downtime_minutes": 5.0,
    "incidents": 1
  },
  "services": [
    {
      "name": "api-gateway",
      "availability": 99.98,
      "uptime_hours": 167.96
    },
    {
      "name": "orchestration",
      "availability": 99.95,
      "uptime_hours": 167.92
    },
    {
      "name": "automl",
      "availability": 99.99,
      "uptime_hours": 167.98
    }
  ],
  "generated_at": "2025-11-26T10:30:00Z"
}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | integer | 7 | Number of days to analyze (1-90) |

**Example with parameters:**

```bash
curl "http://localhost:8100/ga?days=30" \
  -H "X-API-Key: dev-key-readonly"
```

### 3.4 Daily Summaries

#### 3.4.1 GET /summaries/daily

**Get daily summaries for the last 7 days.**

**Request:**

```bash
curl http://localhost:8100/summaries/daily \
  -H "X-API-Key: dev-key-readonly"
```

**Response (200 OK):**

```json
{
  "summaries": [
    {
      "date": "2025-11-26",
      "availability": 99.98,
      "total_requests": 125000,
      "error_rate": 0.02,
      "avg_latency_ms": 45.3,
      "incidents": 0,
      "alerts": 2
    },
    {
      "date": "2025-11-25",
      "availability": 99.95,
      "total_requests": 128000,
      "error_rate": 0.05,
      "avg_latency_ms": 48.1,
      "incidents": 1,
      "alerts": 3
    }
  ],
  "period_start": "2025-11-19",
  "period_end": "2025-11-26",
  "total_days": 7
}
```

#### 3.4.2 GET /summaries/daily/{date}

**Get summary for a specific date.**

**Request:**

```bash
curl http://localhost:8100/summaries/daily/2025-11-26 \
  -H "X-API-Key: dev-key-readonly"
```

**Response (200 OK):**

```json
{
  "date": "2025-11-26",
  "availability": 99.98,
  "total_requests": 125000,
  "successful_requests": 124975,
  "failed_requests": 25,
  "error_rate": 0.02,
  "avg_latency_ms": 45.3,
  "p50_latency_ms": 38.0,
  "p95_latency_ms": 85.0,
  "p99_latency_ms": 120.0,
  "incidents": 0,
  "alerts": 2,
  "services": [
    {
      "name": "api-gateway",
      "requests": 50000,
      "error_rate": 0.01,
      "avg_latency_ms": 35.0
    }
  ]
}
```

**Error Response (404 Not Found):**

```json
{
  "detail": "No summary found for date: 2025-11-26"
}
```

### 3.5 Anomalies

#### 3.5.1 GET /anomalies

**Get detected anomalies for the last 7 days.**

**Request:**

```bash
curl http://localhost:8100/anomalies \
  -H "X-API-Key: dev-key-readonly"
```

**Response (200 OK):**

```json
{
  "anomalies": [
    {
      "timestamp": "2025-11-26T08:15:00Z",
      "metric": "latency",
      "service": "api-gateway",
      "value": 250.5,
      "expected_range": [30, 100],
      "severity": "high",
      "description": "Latency spike detected (2.5x normal)",
      "resolved": false
    },
    {
      "timestamp": "2025-11-25T14:30:00Z",
      "metric": "error_rate",
      "service": "database",
      "value": 5.2,
      "expected_range": [0, 1],
      "severity": "medium",
      "description": "Error rate above threshold",
      "resolved": true,
      "resolved_at": "2025-11-25T15:00:00Z"
    }
  ],
  "total_anomalies": 2,
  "active_anomalies": 1,
  "resolved_anomalies": 1
}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `severity` | string | all | Filter by severity (low, medium, high) |
| `service` | string | all | Filter by service name |
| `resolved` | boolean | all | Filter by resolution status |

**Example with filters:**

```bash
curl "http://localhost:8100/anomalies?severity=high&resolved=false" \
  -H "X-API-Key: dev-key-readonly"
```

#### 3.5.2 GET /anomalies/{date}

**Get anomalies for a specific date.**

**Request:**

```bash
curl http://localhost:8100/anomalies/2025-11-26 \
  -H "X-API-Key: dev-key-readonly"
```

**Response:** Same structure as `/anomalies` but filtered by date.

### 3.6 Regressions

#### 3.6.1 GET /regressions

**Get detected regressions for the last 7 days.**

**Request:**

```bash
curl http://localhost:8100/regressions \
  -H "X-API-Key: dev-key-readonly"
```

**Response (200 OK):**

```json
{
  "regressions": [
    {
      "detected_at": "2025-11-26T09:00:00Z",
      "metric": "availability",
      "service": "orchestration",
      "baseline_value": 99.95,
      "current_value": 99.85,
      "degradation_percent": -0.10,
      "severity": "medium",
      "description": "Availability decreased by 0.10%",
      "root_cause": "Database connection pool exhaustion",
      "mitigated": false
    }
  ],
  "total_regressions": 1,
  "active_regressions": 1,
  "mitigated_regressions": 0
}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | string | all | Filter by metric (availability, latency, error_rate) |
| `severity` | string | all | Filter by severity |
| `service` | string | all | Filter by service name |

#### 3.6.2 GET /regressions/{date}

**Get regressions for a specific date.**

**Request:**

```bash
curl http://localhost:8100/regressions/2025-11-26 \
  -H "X-API-Key: dev-key-readonly"
```

**Response:** Same structure as `/regressions` but filtered by date.

### 3.7 Retrospective

#### 3.7.1 GET /retrospective

**Generate retrospective report (JSON response).**

**Request:**

```bash
curl http://localhost:8100/retrospective \
  -H "X-API-Key: dev-key-admin"
```

**Response (200 OK):**

```json
{
  "period": {
    "start": "2025-11-19T00:00:00Z",
    "end": "2025-11-26T00:00:00Z",
    "days": 7
  },
  "overall_metrics": {
    "availability": 99.95,
    "total_requests": 875000,
    "error_rate": 0.03,
    "avg_latency_ms": 46.2
  },
  "highlights": [
    "Achieved 99.95% availability (target: 99.9%)",
    "Handled 875K requests with 0.03% error rate",
    "Zero critical incidents"
  ],
  "concerns": [
    "1 medium-severity regression detected in orchestration service",
    "Latency p99 increased by 15ms"
  ],
  "anomalies_summary": {
    "total": 5,
    "high_severity": 1,
    "resolved": 4
  },
  "regressions_summary": {
    "total": 1,
    "mitigated": 0
  },
  "recommendations": [
    "Investigate orchestration service database connection pooling",
    "Consider scaling API gateway for peak traffic",
    "Review alerting thresholds for latency"
  ],
  "generated_at": "2025-11-26T10:30:00Z",
  "generated_by": "admin"
}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | integer | 7 | Number of days to analyze (1-90) |
| `format` | string | json | Response format (json, markdown) |
| `sign` | boolean | false | Cryptographically sign the report |

**Example with signing:**

```bash
curl "http://localhost:8100/retrospective?sign=true" \
  -H "X-API-Key: dev-key-admin"

# Response includes signature field:
# {
#   "data": {...},
#   "signature": "RSA-PSS:base64...",
#   "signed_at": "2025-11-26T10:30:00Z",
#   "signer": "admin"
# }
```

#### 3.7.2 GET /retrospective/download

**Download retrospective report as a file.**

**Request:**

```bash
curl http://localhost:8100/retrospective/download \
  -H "X-API-Key: dev-key-admin" \
  -o retrospective.json
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | integer | 7 | Number of days to analyze |
| `format` | string | json | File format (json, markdown, pdf) |
| `sign` | boolean | false | Sign the report |
| `encrypt` | boolean | false | Encrypt the report (AES-256) |

**Download encrypted and signed report:**

```bash
curl "http://localhost:8100/retrospective/download?sign=true&encrypt=true" \
  -H "X-API-Key: dev-key-admin" \
  -o retrospective.json.enc
```

**Response Headers:**

```
Content-Type: application/json
Content-Disposition: attachment; filename="retrospective_2025-11-26.json"
```

---

## 4. Response Models

### 4.1 Common Response Fields

All API responses include these standard fields:

```json
{
  "generated_at": "2025-11-26T10:30:00Z",
  "api_version": "v1.0.2",
  "request_id": "req_abc123"
}
```

### 4.2 Error Response Model

All errors follow a consistent structure:

```json
{
  "detail": "Error message here",
  "error_code": "ERROR_CODE",
  "request_id": "req_abc123",
  "timestamp": "2025-11-26T10:30:00Z"
}
```

**Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Invalid request parameters |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |

### 4.3 Pagination Model

For endpoints returning lists (future enhancement):

```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "pages": 5
}
```

---

## 5. Error Handling

### 5.1 Authentication Errors

**401 Unauthorized:**

```json
{
  "detail": "Invalid API key",
  "error_code": "UNAUTHORIZED"
}
```

**Solutions:**
- Verify API key is correct
- Check Authorization header format: `Bearer <token>`
- Ensure API key header: `X-API-Key: <key>`

**403 Forbidden:**

```json
{
  "detail": "Insufficient permissions. Required role: admin, current role: readonly",
  "error_code": "FORBIDDEN"
}
```

**Solutions:**
- Use account with appropriate role
- Request access from administrator
- Use admin API key for admin endpoints

### 5.2 Validation Errors

**422 Unprocessable Entity:**

```json
{
  "detail": [
    {
      "loc": ["query", "days"],
      "msg": "ensure this value is less than or equal to 90",
      "type": "value_error.number.not_le"
    }
  ],
  "error_code": "VALIDATION_ERROR"
}
```

**Solutions:**
- Check parameter types and ranges
- Review API documentation for valid values
- Ensure required fields are provided

### 5.3 Rate Limiting Errors

**429 Too Many Requests:**

```json
{
  "detail": "Rate limit exceeded. Try again in 45 seconds.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 45
}
```

**Solutions:**
- Wait for `retry_after` seconds before retrying
- Reduce request frequency
- Use authenticated endpoints (higher limits)
- Contact administrator to increase limits

### 5.4 Server Errors

**500 Internal Server Error:**

```json
{
  "detail": "Internal server error. Request ID: req_abc123",
  "error_code": "INTERNAL_ERROR",
  "request_id": "req_abc123"
}
```

**Solutions:**
- Retry request after a few seconds
- Check server logs with request_id
- Report to administrator if persists

---

## 6. Rate Limiting

### 6.1 Rate Limit Tiers

| Endpoint Type | Limit (req/min) | Tier |
|---------------|-----------------|------|
| Public (no auth) | 30 | Low |
| Authenticated | 10 | High |
| Admin | 100 | Premium |

**Public Endpoints:**
- `/health`
- `/metrics`
- `/auth/login`

**Authenticated Endpoints:**
- All other endpoints require authentication

### 6.2 Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1700000000
```

**Example:**

```bash
curl -I http://localhost:8100/health

# Response headers:
# X-RateLimit-Limit: 30
# X-RateLimit-Remaining: 29
# X-RateLimit-Reset: 1700000060
```

### 6.3 Handling Rate Limits

**Best Practices:**

1. **Check remaining requests:**

```python
response = requests.get("http://localhost:8100/ga", headers=headers)
remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
if remaining < 5:
    print("Warning: Approaching rate limit")
```

2. **Implement exponential backoff:**

```python
import time

def api_call_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            continue
        return response
    raise Exception("Max retries exceeded")
```

3. **Use authenticated endpoints for higher limits:**

```python
# Lower limit (30 req/min)
requests.get("http://localhost:8100/health")

# Higher limit (100 req/min)
requests.get(
    "http://localhost:8100/ga",
    headers={"X-API-Key": "dev-key-admin"}
)
```

---

## 7. Code Examples

### 7.1 Python

#### 7.1.1 Simple Client

```python
import requests
from typing import Optional

class TARSClient:
    def __init__(self, base_url: str = "http://localhost:8100", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def get_ga_kpi(self, days: int = 7) -> dict:
        """Get GA readiness KPI."""
        response = self.session.get(
            f"{self.base_url}/ga",
            params={"days": days}
        )
        response.raise_for_status()
        return response.json()

    def get_daily_summary(self, date: str) -> dict:
        """Get daily summary for specific date."""
        response = self.session.get(f"{self.base_url}/summaries/daily/{date}")
        response.raise_for_status()
        return response.json()

    def get_anomalies(self, severity: Optional[str] = None) -> dict:
        """Get detected anomalies."""
        params = {"severity": severity} if severity else {}
        response = self.session.get(
            f"{self.base_url}/anomalies",
            params=params
        )
        response.raise_for_status()
        return response.json()

# Usage
client = TARSClient(api_key="dev-key-readonly")
ga_kpi = client.get_ga_kpi(days=30)
print(f"Availability: {ga_kpi['overall_availability']}%")
```

#### 7.1.2 JWT Authentication Client

```python
import requests
from datetime import datetime, timedelta

class TARSAuthClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

    def login(self):
        """Login and get tokens."""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": self.username, "password": self.password}
        )
        response.raise_for_status()
        data = response.json()

        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])

    def refresh_access_token(self):
        """Refresh access token using refresh token."""
        response = requests.post(
            f"{self.base_url}/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )
        response.raise_for_status()
        data = response.json()

        self.access_token = data["access_token"]
        self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])

    def _ensure_valid_token(self):
        """Ensure we have a valid access token."""
        if not self.access_token:
            self.login()
        elif datetime.now() >= self.token_expires_at:
            self.refresh_access_token()

    def get(self, endpoint: str, **kwargs):
        """Make authenticated GET request."""
        self._ensure_valid_token()
        response = requests.get(
            f"{self.base_url}{endpoint}",
            headers={"Authorization": f"Bearer {self.access_token}"},
            **kwargs
        )
        response.raise_for_status()
        return response.json()

# Usage
client = TARSAuthClient(
    base_url="http://localhost:8100",
    username="admin",
    password="demo123"
)

# Automatic login and token refresh
ga_kpi = client.get("/ga")
anomalies = client.get("/anomalies", params={"severity": "high"})
```

### 7.2 curl

#### 7.2.1 Complete Workflow

```bash
#!/bin/bash
set -e

BASE_URL="http://localhost:8100"
API_KEY="dev-key-admin"

# Health check
echo "Checking health..."
curl -s "${BASE_URL}/health" | jq .

# Get GA KPI
echo "Getting GA KPI..."
curl -s "${BASE_URL}/ga" \
  -H "X-API-Key: ${API_KEY}" | jq .

# Get today's summary
TODAY=$(date +%Y-%m-%d)
echo "Getting summary for ${TODAY}..."
curl -s "${BASE_URL}/summaries/daily/${TODAY}" \
  -H "X-API-Key: ${API_KEY}" | jq .

# Get active anomalies
echo "Getting active anomalies..."
curl -s "${BASE_URL}/anomalies?resolved=false" \
  -H "X-API-Key: ${API_KEY}" | jq .

# Generate retrospective
echo "Generating retrospective..."
curl -s "${BASE_URL}/retrospective?days=7" \
  -H "X-API-Key: ${API_KEY}" | jq . > retrospective.json

echo "Done! Retrospective saved to retrospective.json"
```

### 7.3 JavaScript

#### 7.3.1 Browser Client

```javascript
class TARSClient {
  constructor(baseURL = 'http://localhost:8100', apiKey = null) {
    this.baseURL = baseURL;
    this.apiKey = apiKey;
  }

  async request(endpoint, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'API request failed');
    }

    return response.json();
  }

  async getGAKPI(days = 7) {
    return this.request(`/ga?days=${days}`);
  }

  async getAnomalies(filters = {}) {
    const params = new URLSearchParams(filters);
    return this.request(`/anomalies?${params}`);
  }

  async getRetrospective(days = 7, sign = false) {
    return this.request(`/retrospective?days=${days}&sign=${sign}`);
  }
}

// Usage
const client = new TARSClient('http://localhost:8100', 'dev-key-readonly');

async function displayGAKPI() {
  try {
    const kpi = await client.getGAKPI(30);
    console.log(`Availability: ${kpi.overall_availability}%`);
    console.log(`GA Ready: ${kpi.ga_ready ? 'Yes' : 'No'}`);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

displayGAKPI();
```

---

## 8. Integration Examples

### 8.1 Slack Bot

Post daily summaries to Slack:

```python
import requests
from datetime import date

def post_to_slack(webhook_url: str, message: str):
    """Post message to Slack webhook."""
    requests.post(webhook_url, json={"text": message})

def daily_slack_summary():
    """Post daily T.A.R.S. summary to Slack."""
    # Get today's data
    tars = TARSClient(api_key="dev-key-readonly")
    today = date.today().isoformat()
    summary = tars.get_daily_summary(today)

    # Format message
    message = f"""
📊 *T.A.R.S. Daily Summary - {today}*

✅ Availability: {summary['availability']}%
📈 Total Requests: {summary['total_requests']:,}
⚠️ Error Rate: {summary['error_rate']}%
⏱️ Avg Latency: {summary['avg_latency_ms']}ms
🚨 Incidents: {summary['incidents']}
    """.strip()

    # Post to Slack
    post_to_slack("https://hooks.slack.com/services/YOUR/WEBHOOK/URL", message)

# Run daily via cron
# 0 9 * * * python slack_summary.py
if __name__ == "__main__":
    daily_slack_summary()
```

### 8.2 GitHub Actions

Automated compliance reporting on pull requests:

```yaml
# .github/workflows/compliance-check.yml
name: T.A.R.S. Compliance Check

on:
  pull_request:
    branches: [main]

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - name: Get T.A.R.S. GA KPI
        id: kpi
        run: |
          AVAILABILITY=$(curl -s http://tars.example.com/ga \
            -H "X-API-Key: ${{ secrets.TARS_API_KEY }}" \
            | jq -r '.overall_availability')
          echo "availability=$AVAILABILITY" >> $GITHUB_OUTPUT

      - name: Check GA Readiness
        run: |
          if (( $(echo "${{ steps.kpi.outputs.availability }} < 99.9" | bc -l) )); then
            echo "❌ GA readiness threshold not met: ${{ steps.kpi.outputs.availability }}%"
            exit 1
          else
            echo "✅ GA ready: ${{ steps.kpi.outputs.availability }}%"
          fi

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## T.A.R.S. Compliance Report\n\n✅ Availability: ${{ steps.kpi.outputs.availability }}%\n\nGA Readiness: **PASS**`
            })
```

### 8.3 Grafana Data Source

Create custom Grafana data source for T.A.R.S. API:

```python
# grafana_datasource.py
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

TARS_BASE_URL = "http://localhost:8100"
TARS_API_KEY = "dev-key-readonly"

@app.route('/search', methods=['POST'])
def search():
    """Return available metrics."""
    return jsonify([
        "ga_availability",
        "daily_requests",
        "error_rate",
        "latency"
    ])

@app.route('/query', methods=['POST'])
def query():
    """Query T.A.R.S. metrics."""
    data = request.json
    target = data['targets'][0]['target']

    # Get data from T.A.R.S.
    response = requests.get(
        f"{TARS_BASE_URL}/summaries/daily",
        headers={"X-API-Key": TARS_API_KEY}
    )
    summaries = response.json()['summaries']

    # Transform to Grafana format
    if target == "ga_availability":
        datapoints = [
            [s['availability'], parse_date(s['date'])]
            for s in summaries
        ]
    elif target == "daily_requests":
        datapoints = [
            [s['total_requests'], parse_date(s['date'])]
            for s in summaries
        ]

    return jsonify([{
        "target": target,
        "datapoints": datapoints
    }])

if __name__ == "__main__":
    app.run(port=5000)
```

---

## 9. Troubleshooting

### 9.1 Connection Issues

**Problem:** Cannot connect to API

```bash
# Test connectivity
curl http://localhost:8100/health

# If connection refused:
# 1. Check if API server is running
ps aux | grep run_api_server

# 2. Check if port is correct
netstat -tlnp | grep 8100

# 3. Check firewall rules
sudo ufw status
```

### 9.2 Authentication Issues

**Problem:** 401 Unauthorized errors

```bash
# Verify API key is correct
curl http://localhost:8100/ga \
  -H "X-API-Key: dev-key-admin" \
  -v  # Verbose output

# Check API key format (no extra spaces)
echo -n "dev-key-admin" | xxd

# Try JWT authentication instead
TOKEN=$(curl -X POST http://localhost:8100/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"demo123"}' \
  | jq -r '.access_token')

curl http://localhost:8100/ga \
  -H "Authorization: Bearer $TOKEN"
```

### 9.3 Rate Limiting Issues

**Problem:** 429 Too Many Requests

```bash
# Check rate limit headers
curl -I http://localhost:8100/health

# Wait for reset time
RESET=$(curl -I http://localhost:8100/health | grep X-RateLimit-Reset | cut -d' ' -f2)
WAIT=$((RESET - $(date +%s)))
echo "Wait $WAIT seconds"

# Use authenticated endpoint for higher limits
curl http://localhost:8100/ga \
  -H "X-API-Key: dev-key-admin"  # 100 req/min instead of 30
```

### 9.4 Data Issues

**Problem:** No data returned for date

```bash
# Check if data exists
curl http://localhost:8100/summaries/daily/2025-11-26 \
  -H "X-API-Key: dev-key-readonly"

# If 404, check date format (YYYY-MM-DD)
curl http://localhost:8100/summaries/daily/$(date +%Y-%m-%d) \
  -H "X-API-Key: dev-key-readonly"

# Check available dates
curl http://localhost:8100/summaries/daily \
  -H "X-API-Key: dev-key-readonly" \
  | jq '.summaries[].date'
```

### 9.5 Performance Issues

**Problem:** Slow API responses

```bash
# Measure response time
time curl http://localhost:8100/ga \
  -H "X-API-Key: dev-key-readonly"

# Check Prometheus metrics for latency
curl http://localhost:8100/metrics | grep tars_api_latency

# Enable verbose logging
export TARS_LOG_LEVEL=DEBUG
python scripts/run_api_server.py

# Check for database issues
curl http://localhost:8100/health | jq '.services.database'
```

---

## Appendix

### A. Complete API Reference

See [OpenAPI/Swagger documentation](http://localhost:8100/docs) when server is running.

### B. Security Best Practices

1. **Change default credentials** in production
2. **Use TLS/HTTPS** for all API traffic
3. **Rotate API keys** every 90 days
4. **Use JWT tokens** for user authentication
5. **Monitor rate limits** to detect abuse
6. **Enable audit logging** for compliance
7. **Restrict CORS** to known domains

### C. Support

For issues or questions:
- GitHub Issues: https://github.com/Veleron-Dev-Studios/tars/issues
- Documentation: [docs/](../docs/)
- API Docs: http://localhost:8100/docs

---

**End of API Guide**

Version: v1.0.2-dev | Last Updated: 2025-11-26

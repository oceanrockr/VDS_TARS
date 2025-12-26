# T.A.R.S. Phase 11.5 Quickstart Guide
## Secure Multi-Agent Platform with Authentication & TLS

**Version**: v0.9.5-alpha
**Date**: November 14, 2025

---

## Table of Contents

1. [What's New in Phase 11.5](#whats-new-in-phase-115)
2. [Prerequisites](#prerequisites)
3. [Quick Start (Local Development)](#quick-start-local-development)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Authentication Guide](#authentication-guide)
6. [Common Workflows](#common-workflows)
7. [Troubleshooting](#troubleshooting)

---

## What's New in Phase 11.5

### 🔐 Security Features

1. **JWT Authentication**
   - HS256 token signing
   - 60-minute access tokens
   - 7-day refresh tokens
   - <5ms validation

2. **RBAC (3 Roles)**
   - **Viewer**: Read-only access
   - **Developer**: Modify agents, run optimizations
   - **Admin**: Full system control

3. **API Keys**
   - SHA-256 hashed storage
   - Rotatable keys
   - Service-to-service auth

4. **Rate Limiting**
   - Redis-backed sliding window
   - 30 req/min public endpoints
   - 10 req/min auth endpoints

5. **TLS/mTLS**
   - Self-signed certificate generation
   - cert-manager integration
   - Optional mTLS for internal services

### 📦 Deployment Features

1. **Helm Charts**
   - Complete Kubernetes deployment
   - HPA + PDB for HA
   - Resource limits and requests
   - Network policies

2. **Production Hardening**
   - Security contexts
   - Pod security standards
   - Secret management
   - Health checks and probes

---

## Prerequisites

### System Requirements

- **Python**: 3.9+
- **Node.js**: 18+ (for dashboard)
- **Redis**: 6.0+
- **Docker**: 20.10+ (optional)
- **Kubernetes**: 1.24+ (for production)
- **Helm**: 3.0+ (for Kubernetes deployment)

### Python Dependencies

```bash
# Install security dependencies
pip install cryptography pyjwt redis fastapi

# Or use requirements
cd cognition/shared
pip install -r requirements.txt
```

---

## Quick Start (Local Development)

### Step 1: Generate Secrets and Certificates

```bash
# Generate JWT secret
export JWT_SECRET=$(openssl rand -base64 32)
echo "JWT_SECRET=$JWT_SECRET" >> .env

# Generate API keys
export AUTOML_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export HYPERSYNC_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export ORCHESTRATION_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

echo "AUTOML_API_KEY=$AUTOML_API_KEY" >> .env
echo "HYPERSYNC_API_KEY=$HYPERSYNC_API_KEY" >> .env
echo "ORCHESTRATION_API_KEY=$ORCHESTRATION_API_KEY" >> .env

# Generate TLS certificates
python scripts/generate_certs.py --output-dir ./certs
```

### Step 2: Configure Environment

```bash
# Copy security configuration
cp .env.security.example .env.security

# Edit with your values
vim .env.security

# Key settings:
# - JWT_SECRET=<your-generated-secret>
# - USE_REDIS=true
# - TLS_ENABLED=false  # Set true if using HTTPS locally
# - RBAC_ENABLED=true
```

### Step 3: Start Redis

```bash
# Option A: Docker
docker run -d -p 6379:6379 redis:7-alpine

# Option B: Local Redis
redis-server

# Verify
redis-cli ping
# Should return: PONG
```

### Step 4: Start All Services

**Terminal 1: Orchestration** (Port 8094)
```bash
cd cognition/orchestration-agent
source ../../.env.security
python main.py
# ✓ Listening on http://localhost:8094
```

**Terminal 2: AutoML** (Port 8097)
```bash
cd cognition/automl-pipeline
source ../../.env.security
python service.py
# ✓ Listening on http://localhost:8097
```

**Terminal 3: HyperSync** (Port 8098)
```bash
cd cognition/hyperparameter-sync
source ../../.env.security
python service.py
# ✓ Listening on http://localhost:8098
```

**Terminal 4: Dashboard API** (Port 3001)
```bash
cd dashboard/api
source ../../.env.security
python main.py
# ✓ Listening on http://localhost:3001
```

**Terminal 5: Dashboard Frontend** (Port 3000)
```bash
cd dashboard/frontend
npm start
# ✓ Listening on http://localhost:3000
```

### Step 5: Test Authentication

```bash
# Login as admin
curl -X POST http://localhost:8094/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'

# Save token
export TOKEN="<access_token_from_response>"

# Test protected endpoint
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8094/api/v1/orchestration/agents/state
```

---

## Kubernetes Deployment

### Step 1: Prerequisites

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Add Bitnami repo (for Redis)
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install cert-manager (optional, for automatic TLS)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Step 2: Create Namespace and Secrets

```bash
# Create namespace
kubectl create namespace tars

# Create JWT secret
kubectl create secret generic tars-jwt-secret \
  --from-literal=secret-key="$(openssl rand -base64 32)" \
  --namespace tars

# Create API key secrets
kubectl create secret generic tars-api-keys \
  --from-literal=automl-key="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --from-literal=hypersync-key="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --from-literal=orchestration-key="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --namespace tars

# Create Redis password
kubectl create secret generic redis-password \
  --from-literal=password="$(openssl rand -base64 16)" \
  --namespace tars
```

### Step 3: Install cert-manager Issuers

```bash
# Install CA issuer
kubectl apply -f k8s/cert-manager/issuer.yaml

# Wait for CA certificate
kubectl wait --for=condition=Ready certificate/tars-ca -n cert-manager --timeout=60s

# Create service certificates
kubectl apply -f k8s/cert-manager/certificates.yaml
```

### Step 4: Deploy T.A.R.S.

```bash
# Install with Helm
helm install tars ./charts/tars \
  --namespace tars \
  -f charts/tars/values.yaml \
  -f charts/tars/values-security.yaml

# Check deployment status
kubectl get pods -n tars
kubectl get svc -n tars
kubectl get ingress -n tars

# Wait for all pods to be ready
kubectl wait --for=condition=Ready pods --all -n tars --timeout=300s
```

### Step 5: Access T.A.R.S.

```bash
# Option A: Port forward (development)
kubectl port-forward -n tars svc/dashboard-frontend 3000:3000
kubectl port-forward -n tars svc/dashboard-api 3001:3001

# Access at: http://localhost:3000

# Option B: Ingress (production)
# Add to /etc/hosts:
# <ingress-ip> tars.local

# Access at: https://tars.local
```

---

## Authentication Guide

### Default Users (Demo Mode Only)

| Username | Password | Roles | Use Case |
|----------|----------|-------|----------|
| admin | admin123 | admin | Full access |
| developer | dev123 | developer | Development |
| viewer | view123 | viewer | Read-only |

⚠️ **Disable demo users in production**: Set `DEMO_USERS_ENABLED=false`

### Login Flow

```bash
# 1. Login
curl -X POST http://localhost:8094/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "developer",
    "password": "dev123"
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "user_id": "dev-001",
    "username": "developer",
    "roles": ["developer"],
    "email": "dev@tars.ai"
  }
}

# 2. Save token
export ACCESS_TOKEN="<access_token>"

# 3. Use token in requests
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  http://localhost:8094/api/v1/orchestration/agents/state

# 4. Refresh token (when expired)
curl -X POST http://localhost:8094/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "<refresh_token>"
  }'
```

### Role-Based Access Examples

#### Viewer (Read-Only)

```bash
# ✅ Allowed: View agent state
curl -H "Authorization: Bearer $VIEWER_TOKEN" \
  http://localhost:8094/api/v1/orchestration/agents/state

# ✅ Allowed: View optimization status
curl -H "Authorization: Bearer $VIEWER_TOKEN" \
  http://localhost:8097/api/v1/optimize/dqn_1731600000

# ❌ Forbidden: Run optimization
curl -X POST -H "Authorization: Bearer $VIEWER_TOKEN" \
  http://localhost:8097/api/v1/optimize \
  -d '{"agent_type": "dqn", "n_trials": 10}'
# Response: 403 Forbidden
```

#### Developer

```bash
# ✅ Allowed: Run optimization
curl -X POST -H "Authorization: Bearer $DEVELOPER_TOKEN" \
  http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 20,
    "use_quick_mode": true
  }'

# ✅ Allowed: Reload agent hyperparameters
curl -X POST -H "Authorization: Bearer $DEVELOPER_TOKEN" \
  http://localhost:8094/api/v1/orchestration/agents/policy/reload \
  -H "Content-Type: application/json" \
  -d '{
    "hyperparameters": {
      "learning_rate": 0.002,
      "gamma": 0.97
    }
  }'

# ❌ Forbidden: Approve hyperparameter update
curl -X POST -H "Authorization: Bearer $DEVELOPER_TOKEN" \
  http://localhost:8098/api/v1/sync/approve \
  -d '{"update_id": "dqn_1731600001"}'
# Response: 403 Forbidden (requires admin)
```

#### Admin

```bash
# ✅ Allowed: Approve hyperparameter update
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8098/api/v1/sync/approve \
  -H "Content-Type: application/json" \
  -d '{"update_id": "dqn_1731600001"}'

# ✅ Allowed: Create API key
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8094/auth/service-token \
  -H "Content-Type: application/json" \
  -d '{"service_name": "Test Service"}'

# ✅ Allowed: Rotate API key
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8094/auth/service-token/test_service/rotate

# ✅ Allowed: Revoke API key
curl -X DELETE -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8094/auth/service-token/test_service
```

---

## Common Workflows

### Workflow 1: Complete Agent Optimization (Authenticated)

```bash
# Step 1: Login
TOKEN=$(curl -s -X POST http://localhost:8094/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "developer", "password": "dev123"}' | jq -r '.access_token')

# Step 2: Start optimization
OPTIM_ID=$(curl -s -X POST http://localhost:8097/api/v1/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 30,
    "use_quick_mode": true,
    "register_model": true
  }' | jq -r '.optimization_id')

echo "Optimization ID: $OPTIM_ID"

# Step 3: Monitor progress
while true; do
  STATUS=$(curl -s -H "Authorization: Bearer $TOKEN" \
    http://localhost:8097/api/v1/optimize/$OPTIM_ID | jq -r '.status')
  echo "Status: $STATUS"
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  sleep 10
done

# Step 4: Get results
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8097/api/v1/optimize/$OPTIM_ID | jq .

# Step 5: Propose update (admin only for approval)
curl -X POST http://localhost:8098/api/v1/sync/propose \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "current_params": {"learning_rate": 0.001},
    "current_score": 0.75
  }'
```

### Workflow 2: API Key Management (Admin)

```bash
# Login as admin
ADMIN_TOKEN=$(curl -s -X POST http://localhost:8094/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | jq -r '.access_token')

# List all API keys
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8094/auth/service-tokens | jq .

# Create new API key
NEW_KEY=$(curl -s -X POST http://localhost:8094/auth/service-token \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"service_name": "Monitoring Service"}' | jq -r '.api_key')

echo "New API Key: $NEW_KEY"
echo "⚠️  Store this key securely - it cannot be retrieved again!"

# Test API key
curl -H "Authorization: Bearer $NEW_KEY" \
  http://localhost:8094/api/v1/orchestration/agents/state

# Rotate API key (when compromised)
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8094/auth/service-token/monitoring_service/rotate | jq .

# Revoke API key
curl -X DELETE -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8094/auth/service-token/monitoring_service
```

### Workflow 3: Rate Limit Testing

```bash
# Test rate limiting (will hit 429 after 30 requests)
for i in {1..35}; do
  echo "Request $i:"
  curl -s -o /dev/null -w "Status: %{http_code}\n" \
    http://localhost:8094/health
done

# Expected output:
# Request 1: Status: 200
# Request 2: Status: 200
# ...
# Request 30: Status: 200
# Request 31: Status: 429
# Request 32: Status: 429
# ...

# Check rate limit headers
curl -v http://localhost:8094/health 2>&1 | grep -i ratelimit
# X-RateLimit-Limit: 30
# X-RateLimit-Remaining: 25
# X-RateLimit-Reset: 1731686400
```

---

## Troubleshooting

### Issue 1: Authentication Failed (401 Unauthorized)

**Error**: `{"detail": "Invalid username or password"}`

**Solution**:
```bash
# Check demo users enabled
cat .env.security | grep DEMO_USERS_ENABLED
# Should be: DEMO_USERS_ENABLED=true (for local testing)

# Verify JWT secret is set
echo $JWT_SECRET
# Should output a long base64 string

# Try login with correct credentials
curl -X POST http://localhost:8094/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

---

### Issue 2: Token Expired (401 Unauthorized)

**Error**: `{"detail": "Token has expired"}`

**Solution**:
```bash
# Use refresh token to get new access token
curl -X POST http://localhost:8094/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<your_refresh_token>"}'

# Or login again
TOKEN=$(curl -s -X POST http://localhost:8094/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "developer", "password": "dev123"}' | jq -r '.access_token')
```

---

### Issue 3: Forbidden (403)

**Error**: `{"detail": "Requires developer role or higher"}`

**Solution**:
```bash
# Check your user's roles
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8094/auth/me | jq .roles

# Use a user with appropriate role:
# - viewer: Read-only
# - developer: Modify agents, run optimizations
# - admin: Full access

# Login as admin if needed
ADMIN_TOKEN=$(curl -s -X POST http://localhost:8094/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | jq -r '.access_token')
```

---

### Issue 4: Rate Limited (429)

**Error**: `{"detail": "Rate limit exceeded. Retry after 42 seconds"}`

**Solution**:
```bash
# Wait for the retry-after duration
sleep 42

# Or increase rate limits in .env.security
PUBLIC_RATE_LIMIT=100
AUTH_RATE_LIMIT=50

# Restart services
```

---

### Issue 5: Redis Connection Failed

**Error**: `Redis connection failed, using in-memory`

**Solution**:
```bash
# Check Redis running
redis-cli ping
# Should return: PONG

# If not running, start Redis
docker run -d -p 6379:6379 redis:7-alpine
# Or
redis-server

# Verify connection
redis-cli
> PING
PONG
> exit
```

---

### Issue 6: Certificate Verification Failed

**Error**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**:
```bash
# For local development with self-signed certs
export PYTHONHTTPSVERIFY=0
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Or add CA to trust store
# macOS:
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain certs/ca.crt

# Linux:
sudo cp certs/ca.crt /usr/local/share/ca-certificates/tars-ca.crt
sudo update-ca-certificates

# For production, use valid certificates (Let's Encrypt)
```

---

### Issue 7: Kubernetes Pod CrashLoopBackOff

**Error**: Pods repeatedly crash

**Solution**:
```bash
# Check pod logs
kubectl logs -n tars <pod-name> --tail=50

# Common issues:
# 1. Missing secrets
kubectl get secrets -n tars
# Ensure tars-jwt-secret and tars-api-keys exist

# 2. Resource limits too low
kubectl describe pod -n tars <pod-name>
# Check OOMKilled events

# 3. Image pull errors
kubectl get events -n tars --sort-by='.lastTimestamp'

# Fix by increasing resources or pulling images
helm upgrade tars ./charts/tars -n tars \
  -f charts/tars/values-security.yaml \
  --set orchestration.resources.limits.memory=16Gi
```

---

## Next Steps

1. **Explore Dashboard**: http://localhost:3000
2. **Run Load Tests**: `k6 run tests/load/auth_load_test.js`
3. **Set Up Monitoring**: Deploy Prometheus + Grafana
4. **Configure Backup**: Enable automated backups
5. **Production Deployment**: Use Helm with Let's Encrypt

---

## Support

**Documentation**:
- [PHASE11_5_IMPLEMENTATION_REPORT.md](PHASE11_5_IMPLEMENTATION_REPORT.md): Full technical details
- [ARCHITECTURE_SECURITY.md](ARCHITECTURE_SECURITY.md): Security architecture
- [INSTALL_HELM.md](INSTALL_HELM.md): Kubernetes deployment guide

**Logs**:
- Orchestration: `./logs/orchestration.log`
- AutoML: `./logs/automl.log`
- HyperSync: `./logs/hyperparam_sync.log`

---

**Status**: ✅ **Phase 11.5 Complete - Production Ready**

🚀 **T.A.R.S. is now secure, scalable, and ready for production deployment!**

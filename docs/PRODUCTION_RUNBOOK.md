# T.A.R.S. Production Runbook

**Version:** v1.0.2
**Last Updated:** 2025-11-28
**Audience:** DevOps, SRE, Production Engineering
**Classification:** Internal - Operational

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Deployment Models](#deployment-models)
4. [Operations Guide](#operations-guide)
5. [Maintenance Procedures](#maintenance-procedures)
6. [Incident Response Playbooks](#incident-response-playbooks)
7. [Disaster Recovery](#disaster-recovery)
8. [Diagnostics & Troubleshooting](#diagnostics--troubleshooting)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Appendix](#appendix)

---

## Overview

### Purpose

This runbook provides comprehensive operational procedures for deploying, maintaining, and troubleshooting the T.A.R.S. (Telemetry, Analytics, and Response System) v1.0.2 in production environments.

### System Summary

T.A.R.S. is a multi-agent reinforcement learning system with enterprise-grade security, cognitive analytics, and supply chain integrity features.

**Core Services:**
- Multi-Agent Orchestration (Port 8094)
- AutoML Pipeline Service (Port 8096)
- Dashboard API (Port 3001)
- Dashboard Frontend (Port 3000)
- HyperSync Service (Port 8098)
- Insight Engine (Port 8090)
- Adaptive Policy Learner (Port 8091)
- Meta-Consensus Optimizer (Port 8092)
- Causal Inference Engine (Port 8095)

**Key Features:**
- JWT Authentication + RBAC
- Rate Limiting (Redis-backed)
- TLS/mTLS support
- SBOM + SLSA provenance generation
- Multi-region federation (Raft consensus)
- Real-time metrics (Prometheus)

### Critical Contacts

| Role | Responsibility | Contact |
|------|---------------|---------|
| **On-Call SRE** | 24/7 incident response | sre-oncall@company.com |
| **Security Team** | Security incidents, key rotation | security@company.com |
| **Platform Team** | Infrastructure, K8s cluster | platform@company.com |
| **ML Team** | Model performance, RL agents | ml-team@company.com |

### Service Level Objectives (SLOs)

| Metric | Target | Measurement Window |
|--------|--------|-------------------|
| **Availability** | 99.9% | 30 days |
| **API Latency (p95)** | < 200ms | 5 minutes |
| **API Latency (p99)** | < 500ms | 5 minutes |
| **Error Rate** | < 0.1% | 5 minutes |
| **Agent Training Time** | < 30s (quick) | Per training run |
| **HyperSync Latency** | < 100ms | Per sync |

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer                            │
│                    (Ingress + TLS Termination)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │Dashboard│    │Dashboard│    │ Multi-  │
    │Frontend │    │   API   │    │ Agent   │
    │(Port    │    │(Port    │    │Orchestr.│
    │ 3000)   │    │ 3001)   │    │(Port    │
    └─────────┘    └────┬────┘    │ 8094)   │
                        │         └────┬────┘
                        │              │
         ┌──────────────┼──────────────┼──────────────┐
         │              │              │              │
    ┌────▼────┐    ┌────▼────┐    ┌───▼────┐    ┌────▼────┐
    │AutoML   │    │HyperSync│    │Insight │    │Adaptive │
    │Pipeline │    │Service  │    │Engine  │    │Policy   │
    │(8096)   │    │(8098)   │    │(8090)  │    │Learner  │
    └─────────┘    └─────────┘    └────────┘    │(8091)   │
                                                 └─────────┘
         │              │              │              │
         └──────────────┼──────────────┼──────────────┘
                        │              │
                   ┌────▼────┐    ┌────▼────┐
                   │ Redis   │    │Prometheus│
                   │ Cluster │    │  Metrics │
                   └─────────┘    └──────────┘
```

### Service Dependencies

| Service | Dependencies | Required | Optional |
|---------|-------------|----------|----------|
| **Dashboard Frontend** | Dashboard API | ✓ | - |
| **Dashboard API** | Redis, PostgreSQL | ✓ | Vault |
| **Multi-Agent Orchestrator** | Redis, Prometheus | ✓ | MLflow |
| **AutoML Pipeline** | MLflow, Redis | ✓ | S3 |
| **HyperSync Service** | Redis, Multi-Agent | ✓ | - |
| **Insight Engine** | Prometheus, Redis | ✓ | - |
| **Adaptive Policy Learner** | OPA, Insight Engine | ✓ | - |

### Data Stores

| Store | Purpose | Persistence | Backup Frequency |
|-------|---------|-------------|------------------|
| **Redis** | Rate limiting, caching, sessions | In-memory (RDB/AOF) | Hourly |
| **PostgreSQL** | Dashboard data, user accounts | Persistent | Daily + WAL |
| **ChromaDB** | Vector embeddings, RAG | Persistent (SQLite) | Daily |
| **MLflow** | Experiment tracking, models | S3-backed | Real-time (S3) |
| **Prometheus** | Time-series metrics | Persistent (TSDB) | N/A (retention 30d) |

---

## Deployment Models

### Model 1: Local Development

**Use Case:** Development, testing, rapid prototyping

**Prerequisites:**
- Python 3.9+
- Redis 7.0+
- PostgreSQL 14+ (optional)
- 8GB RAM minimum

**Setup:**

```bash
# 1. Clone repository
git clone https://github.com/company/tars.git
cd tars

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with local configuration

# 5. Generate secrets
./scripts/generate_secrets.sh

# 6. Start Redis
redis-server --port 6379

# 7. Start services
python scripts/run_api_server.py --port 3001 --profile local
```

**Health Check:**
```bash
curl http://localhost:3001/health
# Expected: {"status": "healthy", "version": "1.0.2"}
```

---

### Model 2: Docker Compose

**Use Case:** Integration testing, staging environments

**Prerequisites:**
- Docker 24.0+
- Docker Compose 2.20+
- 16GB RAM minimum

**Setup:**

```bash
# 1. Clone and configure
git clone https://github.com/company/tars.git
cd tars
cp .env.example .env

# 2. Generate secrets
docker run --rm -v $(pwd):/app python:3.9 \
  python /app/scripts/generate_secrets.sh

# 3. Build images
docker-compose build

# 4. Start stack
docker-compose up -d

# 5. Check logs
docker-compose logs -f dashboard-api

# 6. Health check
curl http://localhost:3001/health
```

**Docker Compose Services:**

```yaml
services:
  redis:
    image: redis:7.0-alpine
    ports: ["6379:6379"]
    volumes: ["redis-data:/data"]
    command: redis-server --appendonly yes

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: tars
      POSTGRES_USER: tars
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets: ["db_password"]
    volumes: ["postgres-data:/var/lib/postgresql/data"]

  dashboard-api:
    build: .
    command: python scripts/run_api_server.py --profile docker
    ports: ["3001:3001"]
    depends_on: [redis, postgres]
    environment:
      REDIS_HOST: redis
      DATABASE_URL: postgresql://tars@postgres/tars
    volumes: ["./logs:/app/logs"]

  multi-agent:
    build: .
    command: python cognition/orchestration-agent/main.py
    ports: ["8094:8094"]
    depends_on: [redis]

  hypersync:
    build: .
    command: python cognition/hypersync/service.py
    ports: ["8098:8098"]
    depends_on: [redis, multi-agent]
```

**Monitoring:**
```bash
# Container status
docker-compose ps

# Resource usage
docker stats

# Service logs
docker-compose logs -f [service-name]
```

---

### Model 3: Kubernetes (Production)

**Use Case:** Production, high availability, auto-scaling

**Prerequisites:**
- Kubernetes 1.26+
- Helm 3.12+
- cert-manager (for TLS)
- Prometheus Operator (for monitoring)
- Redis Operator or managed Redis
- Managed PostgreSQL (RDS, CloudSQL, etc.)

**Architecture:**

```
Namespace: tars-prod
├── Ingress (TLS termination, rate limiting)
├── Deployments (2-10 replicas per service, HPA)
├── Services (ClusterIP, headless for StatefulSets)
├── ConfigMaps (non-sensitive config)
├── Secrets (Sealed Secrets or Vault integration)
├── PersistentVolumeClaims (MLflow, logs)
├── HorizontalPodAutoscalers (CPU/memory-based)
├── PodDisruptionBudgets (HA guarantees)
└── NetworkPolicies (service mesh or CNI-based)
```

**Deployment Steps:**

```bash
# 1. Install cert-manager (if not present)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# 2. Create namespace
kubectl create namespace tars-prod

# 3. Install Sealed Secrets (or configure Vault)
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm install sealed-secrets sealed-secrets/sealed-secrets -n kube-system

# 4. Create sealed secrets
# (Use scripts/generate_secrets.sh to generate, then seal)
./scripts/generate_secrets.sh
kubeseal --format yaml < secrets.yaml > sealed-secrets.yaml
kubectl apply -f sealed-secrets.yaml -n tars-prod

# 5. Configure values.yaml
cp charts/tars/values.yaml values-prod.yaml
# Edit: replicas, resources, ingress, persistence

# 6. Install Helm chart
helm install tars charts/tars \
  -n tars-prod \
  -f values-prod.yaml

# 7. Verify deployment
kubectl get pods -n tars-prod
kubectl get ingress -n tars-prod

# 8. Health check
curl https://tars.company.com/health
```

**Key Helm Values (values-prod.yaml):**

```yaml
replicaCount: 3

image:
  repository: company/tars
  tag: "1.0.2"
  pullPolicy: IfNotPresent

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "30"
  hosts:
    - host: tars.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: tars-tls
      hosts:
        - tars.company.com

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

redis:
  enabled: true
  cluster:
    enabled: true
    nodes: 6
  persistence:
    enabled: true
    size: 10Gi

postgresql:
  enabled: false  # Use managed RDS/CloudSQL
  external:
    host: postgres.cxyz.us-east-1.rds.amazonaws.com
    port: 5432
    database: tars
    username: tars
    existingSecret: tars-postgres-secret
```

**Monitoring:**

```bash
# Pod status
kubectl get pods -n tars-prod -w

# Pod logs
kubectl logs -f deployment/dashboard-api -n tars-prod

# Describe pod (for troubleshooting)
kubectl describe pod <pod-name> -n tars-prod

# HPA status
kubectl get hpa -n tars-prod

# Resource usage
kubectl top pods -n tars-prod
kubectl top nodes
```

---

### Model 4: Air-Gapped Deployment

**Use Case:** Regulated environments, government, finance

**Prerequisites:**
- Pre-downloaded Docker images (tarball)
- Offline Python wheels (requirements.txt)
- Pre-generated secrets
- Local artifact repository (Nexus, Artifactory)

**Setup:**

```bash
# 1. Prepare artifacts (from internet-connected machine)
./scripts/prepare_airgap_bundle.sh v1.0.2

# Output:
# tars-v1.0.2-airgap.tar.gz
# ├── images/
# │   ├── tars-1.0.2.tar
# │   ├── redis-7.0.tar
# │   └── postgres-14.tar
# ├── wheels/
# │   └── (all Python packages)
# ├── charts/
# │   └── tars-1.0.2.tgz
# └── docs/

# 2. Transfer to air-gapped environment
scp tars-v1.0.2-airgap.tar.gz user@airgap-bastion:/opt/artifacts/

# 3. On air-gapped machine
tar -xzf tars-v1.0.2-airgap.tar.gz
cd tars-v1.0.2-airgap

# 4. Load Docker images
for img in images/*.tar; do docker load -i $img; done

# 5. Install Python packages
pip install --no-index --find-links wheels/ -r requirements.txt

# 6. Deploy
docker-compose -f docker-compose.airgap.yaml up -d
# OR
helm install tars charts/tars-1.0.2.tgz -f values-airgap.yaml
```

**Configuration Differences:**

```yaml
# values-airgap.yaml
airgap:
  enabled: true
  localRegistry: "registry.airgap.local"

image:
  repository: registry.airgap.local/tars
  tag: "1.0.2"

# Disable external integrations
mlflow:
  s3:
    enabled: false
  local:
    enabled: true

telemetry:
  external: false

updates:
  autoCheck: false
```

---

### Model 5: Enterprise with Vault

**Use Case:** Large enterprises with centralized secrets management

**Prerequisites:**
- HashiCorp Vault cluster
- Vault Agent Injector (K8s)
- Service account with Vault policy

**Setup:**

```bash
# 1. Configure Vault policies
vault policy write tars-read - <<EOF
path "secret/data/tars/*" {
  capabilities = ["read"]
}
EOF

# 2. Enable Kubernetes auth
vault auth enable kubernetes
vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc:443"

# 3. Create Vault role
vault write auth/kubernetes/role/tars \
  bound_service_account_names=tars \
  bound_service_account_namespaces=tars-prod \
  policies=tars-read \
  ttl=24h

# 4. Store secrets in Vault
vault kv put secret/tars/jwt-secret value="$(openssl rand -base64 32)"
vault kv put secret/tars/aes-key value="$(openssl rand -hex 32)"
vault kv put secret/tars/rsa-private key=@rsa_private.pem
vault kv put secret/tars/redis password="$(openssl rand -base64 32)"

# 5. Deploy with Vault annotations
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashboard-api
  namespace: tars-prod
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "tars"
        vault.hashicorp.com/agent-inject-secret-jwt: "secret/data/tars/jwt-secret"
        vault.hashicorp.com/agent-inject-template-jwt: |
          {{- with secret "secret/data/tars/jwt-secret" -}}
          export JWT_SECRET="{{ .Data.data.value }}"
          {{- end }}
    spec:
      serviceAccountName: tars
      containers:
      - name: api
        image: company/tars:1.0.2
        command: ["/bin/sh", "-c"]
        args:
          - source /vault/secrets/jwt && python scripts/run_api_server.py
EOF
```

---

## Operations Guide

### Starting Services

#### Local Development

```bash
# Start Redis
redis-server --port 6379 &

# Start Dashboard API
python scripts/run_api_server.py --port 3001 --profile local &

# Start Multi-Agent Orchestrator
python cognition/orchestration-agent/main.py &

# Start HyperSync Service
python cognition/hypersync/service.py &

# Check all processes
ps aux | grep python
```

#### Docker Compose

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d dashboard-api

# Scale service
docker-compose up -d --scale multi-agent=3

# View logs
docker-compose logs -f --tail=100 dashboard-api
```

#### Kubernetes

```bash
# Start all services (if stopped)
kubectl scale deployment --all --replicas=2 -n tars-prod

# Start specific service
kubectl scale deployment dashboard-api --replicas=3 -n tars-prod

# Restart deployment (rolling)
kubectl rollout restart deployment/dashboard-api -n tars-prod

# Check rollout status
kubectl rollout status deployment/dashboard-api -n tars-prod
```

---

### Stopping Services

#### Graceful Shutdown

**Local:**
```bash
# Send SIGTERM (graceful)
pkill -TERM -f "run_api_server.py"

# Wait for shutdown (max 30s)
sleep 5

# Force kill if still running
pkill -KILL -f "run_api_server.py"
```

**Docker Compose:**
```bash
# Graceful stop (10s timeout)
docker-compose stop

# Immediate stop
docker-compose kill

# Stop and remove
docker-compose down
```

**Kubernetes:**
```bash
# Graceful scale down
kubectl scale deployment dashboard-api --replicas=0 -n tars-prod

# Delete deployment (recreate later)
kubectl delete deployment dashboard-api -n tars-prod

# Delete all (careful!)
helm uninstall tars -n tars-prod
```

---

### Configuration Profiles

T.A.R.S. supports multiple configuration profiles for different environments.

#### Profile: `local`

**File:** `enterprise_config/profiles/local.yaml`

```yaml
environment: local
log_level: DEBUG
log_format: text

api:
  host: 127.0.0.1
  port: 3001
  workers: 1
  cors_origins: ["http://localhost:3000"]

redis:
  host: localhost
  port: 6379
  db: 0
  password: null

auth:
  jwt_expiry: 3600
  refresh_expiry: 604800
  require_mfa: false

rate_limiting:
  enabled: true
  requests_per_minute: 100

encryption:
  enabled: false
  algorithm: aes-256-gcm

signing:
  enabled: false
  algorithm: rsa-pss

compliance:
  enforcement_mode: warn
```

#### Profile: `production`

**File:** `enterprise_config/profiles/production.yaml`

```yaml
environment: production
log_level: INFO
log_format: json

api:
  host: 0.0.0.0
  port: 3001
  workers: 4
  cors_origins: ["https://tars.company.com"]

redis:
  host: redis-cluster.tars-prod.svc.cluster.local
  port: 6379
  db: 0
  password: ${REDIS_PASSWORD}
  sentinel: true
  sentinel_master: mymaster

auth:
  jwt_expiry: 3600
  refresh_expiry: 86400
  require_mfa: true

rate_limiting:
  enabled: true
  requests_per_minute: 30

encryption:
  enabled: true
  algorithm: aes-256-gcm
  key_file: /run/secrets/aes.key

signing:
  enabled: true
  algorithm: rsa-pss
  key_file: /run/secrets/rsa_private.pem

compliance:
  enforcement_mode: enforce
  controls:
    - encryption_at_rest
    - encryption_in_transit
    - audit_logging
    - data_retention
```

#### Loading Profiles

```bash
# Environment variable
export TARS_PROFILE=production
python scripts/run_api_server.py

# CLI flag
python scripts/run_api_server.py --profile production

# Config file
python scripts/run_api_server.py --config enterprise_config/profiles/production.yaml
```

---

### Managing Encryption Keys

#### AES-256-GCM Keys

**Generation:**
```bash
# Generate 256-bit key
openssl rand -hex 32 > aes.key

# Store in Vault (K8s)
kubectl create secret generic tars-aes-key \
  --from-file=aes.key \
  -n tars-prod

# Store in environment
export AES_KEY=$(cat aes.key)
```

**Rotation:**
```bash
# 1. Generate new key
openssl rand -hex 32 > aes-new.key

# 2. Re-encrypt existing data with new key
python scripts/rotate_encryption_keys.py \
  --old-key aes.key \
  --new-key aes-new.key \
  --data-dir /opt/tars/data

# 3. Update secrets
kubectl create secret generic tars-aes-key \
  --from-file=aes.key=aes-new.key \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Rolling restart
kubectl rollout restart deployment/dashboard-api -n tars-prod

# 5. Verify
kubectl logs -f deployment/dashboard-api -n tars-prod | grep "AES key loaded"

# 6. Secure old key
mv aes.key aes-old-$(date +%Y%m%d).key
shred -u aes-old-*.key  # Secure delete
```

**Key Expiry Policy:**
- Rotate every **90 days**
- Emergency rotation: **within 24 hours** of suspected compromise
- Keys stored in Vault expire automatically (TTL: 90 days)

---

#### RSA-PSS Signing Keys

**Generation:**
```bash
# Generate 4096-bit RSA key pair
openssl genpkey -algorithm RSA -out rsa_private.pem -pkeyopt rsa_keygen_bits:4096
openssl rsa -pubout -in rsa_private.pem -out rsa_public.pem

# Secure private key
chmod 600 rsa_private.pem

# Store in Vault
kubectl create secret generic tars-rsa-keys \
  --from-file=rsa_private.pem \
  --from-file=rsa_public.pem \
  -n tars-prod
```

**Rotation:**
```bash
# 1. Generate new key pair
openssl genpkey -algorithm RSA -out rsa_new_private.pem -pkeyopt rsa_keygen_bits:4096
openssl rsa -pubout -in rsa_new_private.pem -out rsa_new_public.pem

# 2. Update signing service (dual-key mode for transition)
# - Keep old key for verification
# - Use new key for signing

# 3. Update secrets (append new key)
kubectl create secret generic tars-rsa-keys-new \
  --from-file=rsa_private.pem=rsa_new_private.pem \
  --from-file=rsa_public.pem=rsa_new_public.pem \
  -n tars-prod

# 4. Update deployment to mount both secrets
# (See dual-key configuration below)

# 5. After 30 days (old signatures expired), remove old key
kubectl delete secret tars-rsa-keys -n tars-prod
```

**Dual-Key Configuration (Transition Period):**

```yaml
# deployment-dual-keys.yaml
spec:
  template:
    spec:
      volumes:
        - name: rsa-old
          secret:
            secretName: tars-rsa-keys
        - name: rsa-new
          secret:
            secretName: tars-rsa-keys-new
      containers:
        - name: api
          volumeMounts:
            - name: rsa-old
              mountPath: /run/secrets/rsa-old
              readOnly: true
            - name: rsa-new
              mountPath: /run/secrets/rsa-new
              readOnly: true
          env:
            - name: SIGNING_KEY_NEW
              value: /run/secrets/rsa-new/rsa_private.pem
            - name: SIGNING_KEY_OLD
              value: /run/secrets/rsa-old/rsa_private.pem
            - name: VERIFY_KEYS
              value: /run/secrets/rsa-new/rsa_public.pem,/run/secrets/rsa-old/rsa_public.pem
```

---

### SBOM and SLSA Workflows

#### Generating SBOM

**Manual Generation:**
```bash
# Generate CycloneDX and SPDX
python security/sbom_generator.py \
  --output-dir release/v1.0.2/sbom \
  --formats cyclonedx spdx \
  --verbose

# Output:
# release/v1.0.2/sbom/
# ├── tars-v1.0.2-cyclonedx.json
# └── tars-v1.0.2-spdx.json
```

**With Signing:**
```bash
python security/sbom_generator.py \
  --output-dir release/v1.0.2/sbom \
  --formats cyclonedx spdx \
  --sign \
  --signing-key /run/secrets/rsa_private.pem \
  --verbose

# Output includes .sig files:
# ├── tars-v1.0.2-cyclonedx.json
# ├── tars-v1.0.2-cyclonedx.json.sig
# ├── tars-v1.0.2-spdx.json
# └── tars-v1.0.2-spdx.json.sig
```

**Automated (CI/CD):**
```yaml
# .github/workflows/release.yml
- name: Generate SBOM
  run: |
    python security/sbom_generator.py \
      --output-dir ${{ github.workspace }}/release/sbom \
      --formats cyclonedx spdx \
      --sign \
      --signing-key ${{ secrets.RSA_PRIVATE_KEY }}

- name: Upload SBOM
  uses: actions/upload-artifact@v3
  with:
    name: sbom
    path: release/sbom/
```

---

#### Generating SLSA Provenance

**Manual Generation:**
```bash
# Generate provenance for release artifact
python security/slsa_generator.py \
  dist/tars-v1.0.2.tar.gz \
  --output release/v1.0.2/slsa/tars-v1.0.2.provenance.json \
  --verbose

# With signing
python security/slsa_generator.py \
  dist/tars-v1.0.2.tar.gz \
  --output release/v1.0.2/slsa/tars-v1.0.2.provenance.json \
  --sign \
  --signing-key /run/secrets/rsa_private.pem \
  --verbose
```

**Verify Provenance:**
```bash
# Verify signature
python security/slsa_generator.py \
  --verify \
  --output release/v1.0.2/slsa/tars-v1.0.2.provenance.json \
  --public-key /run/secrets/rsa_public.pem

# Expected output:
# ✓ Signature valid
# ✓ Provenance integrity verified
```

**Automated (Release Pipeline):**
```bash
# In prepare_release_artifacts.py
python scripts/prepare_release_artifacts.py \
  --include-sbom \
  --include-slsa \
  --sign \
  --output-dir release/v1.0.2
```

---

### Log Management

#### Log Formats

**Text (Development):**
```
2025-11-28 10:15:32 INFO [dashboard-api] Starting API server on 0.0.0.0:3001
2025-11-28 10:15:33 INFO [dashboard-api] JWT authentication enabled
2025-11-28 10:15:33 INFO [dashboard-api] Rate limiting enabled (30 req/min)
```

**JSON (Production):**
```json
{
  "timestamp": "2025-11-28T10:15:32Z",
  "level": "INFO",
  "service": "dashboard-api",
  "message": "Starting API server on 0.0.0.0:3001",
  "context": {
    "pid": 12345,
    "hostname": "tars-api-7d8f9c-xk2l",
    "version": "1.0.2"
  }
}
```

#### Log Locations

**Local:**
```
logs/
├── dashboard-api.log
├── multi-agent.log
├── hypersync.log
├── insight-engine.log
└── error.log
```

**Docker:**
```bash
# View logs
docker-compose logs -f dashboard-api

# Follow all services
docker-compose logs -f

# Export logs
docker-compose logs --no-color > tars-logs-$(date +%Y%m%d).log
```

**Kubernetes:**
```bash
# View logs
kubectl logs -f deployment/dashboard-api -n tars-prod

# View logs from all pods
kubectl logs -f -l app=dashboard-api -n tars-prod

# Export logs (last 1 hour)
kubectl logs --since=1h -l app=dashboard-api -n tars-prod > api-logs.txt

# Stream to log aggregator (Fluentd, Loki)
# (Configured via DaemonSet)
```

#### Log Aggregation

**ELK Stack:**
```yaml
# filebeat-config.yaml
filebeat.inputs:
  - type: container
    paths:
      - '/var/lib/docker/containers/*/*.log'
    processors:
      - add_kubernetes_metadata:
          host: ${NODE_NAME}
          matchers:
            - logs_path:
                logs_path: "/var/lib/docker/containers/"

output.elasticsearch:
  hosts: ["https://elasticsearch.company.com:9200"]
  username: "tars-logger"
  password: "${ELASTICSEARCH_PASSWORD}"
  index: "tars-logs-%{+yyyy.MM.dd}"
```

**Grafana Loki:**
```yaml
# promtail-config.yaml
server:
  http_listen_port: 9080

clients:
  - url: https://loki.company.com/loki/api/v1/push

scrape_configs:
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [tars-prod]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
```

**CloudWatch Logs (AWS):**
```bash
# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cloudwatch-namespace.yaml

# Configure log group
aws logs create-log-group --log-group-name /aws/eks/tars-prod
```

---

### Health Checks

#### Endpoint: `/health`

**Request:**
```bash
curl http://localhost:3001/health
```

**Response (Healthy):**
```json
{
  "status": "healthy",
  "version": "1.0.2",
  "timestamp": "2025-11-28T10:15:32Z",
  "checks": {
    "redis": "ok",
    "database": "ok",
    "disk_space": "ok"
  }
}
```

**Response (Unhealthy):**
```json
{
  "status": "unhealthy",
  "version": "1.0.2",
  "timestamp": "2025-11-28T10:15:32Z",
  "checks": {
    "redis": "error",
    "database": "ok",
    "disk_space": "ok"
  },
  "errors": ["Redis connection timeout"]
}
```

#### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 3001
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 3001
  initialDelaySeconds: 15
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

---

## Maintenance Procedures

### Rotating Secrets

#### JWT Secrets

**Impact:** Active sessions will be invalidated.

**Procedure:**

1. **Schedule maintenance window** (off-peak hours)
2. **Generate new secret:**
   ```bash
   NEW_JWT_SECRET=$(openssl rand -base64 32)
   ```
3. **Update secret in Vault/K8s:**
   ```bash
   kubectl create secret generic tars-jwt-secret \
     --from-literal=jwt-secret=$NEW_JWT_SECRET \
     --dry-run=client -o yaml | kubectl apply -f -
   ```
4. **Rolling restart:**
   ```bash
   kubectl rollout restart deployment/dashboard-api -n tars-prod
   ```
5. **Monitor rollout:**
   ```bash
   kubectl rollout status deployment/dashboard-api -n tars-prod
   ```
6. **Verify health:**
   ```bash
   curl -H "Authorization: Bearer $OLD_TOKEN" https://tars.company.com/api/agents
   # Expected: 401 Unauthorized

   # Login with new secret
   curl -X POST https://tars.company.com/auth/login \
     -d '{"username":"admin","password":"..."}' \
     -H "Content-Type: application/json"
   # Expected: 200 with new token
   ```
7. **Notify users:** Send email about session invalidation

**Rollback Plan:**
```bash
# If new secret fails
kubectl create secret generic tars-jwt-secret \
  --from-literal=jwt-secret=$OLD_JWT_SECRET \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/dashboard-api -n tars-prod
```

---

#### Redis Password Rotation

**Impact:** Brief connection errors during rotation (< 10s with proper procedure).

**Procedure:**

1. **Generate new password:**
   ```bash
   NEW_REDIS_PASSWORD=$(openssl rand -base64 32)
   ```

2. **Enable ACL multi-user mode (Redis 6+):**
   ```bash
   redis-cli ACL SETUSER tars on >$NEW_REDIS_PASSWORD ~* +@all
   ```

3. **Update application secrets:**
   ```bash
   kubectl create secret generic tars-redis \
     --from-literal=password=$NEW_REDIS_PASSWORD \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

4. **Rolling restart (zero-downtime):**
   ```bash
   kubectl rollout restart deployment/dashboard-api -n tars-prod
   kubectl rollout status deployment/dashboard-api -n tars-prod
   ```

5. **Remove old ACL user:**
   ```bash
   redis-cli ACL DELUSER tars-old
   ```

6. **Verify connections:**
   ```bash
   redis-cli --user tars --pass $NEW_REDIS_PASSWORD PING
   # Expected: PONG
   ```

---

### Database Maintenance

#### PostgreSQL Vacuum

**Purpose:** Reclaim storage, update statistics, prevent transaction ID wraparound.

**Frequency:** Weekly (auto-vacuum enabled by default).

**Manual Vacuum:**
```bash
# Full vacuum (requires downtime)
kubectl exec -it postgres-0 -n tars-prod -- psql -U tars -c "VACUUM FULL;"

# Analyze (no downtime, update statistics)
kubectl exec -it postgres-0 -n tars-prod -- psql -U tars -c "VACUUM ANALYZE;"
```

**Monitoring:**
```sql
-- Check last autovacuum
SELECT schemaname, relname, last_autovacuum, last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY last_autovacuum DESC;

-- Check bloat
SELECT schemaname, tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS bloat
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

#### Backup and Restore

**Backup (Daily via CronJob):**

```yaml
# k8s/cronjob-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: tars-prod
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:14
              command:
                - /bin/bash
                - -c
                - |
                  BACKUP_FILE=/backups/tars-$(date +%Y%m%d-%H%M%S).sql.gz
                  pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d tars | gzip > $BACKUP_FILE
                  echo "Backup complete: $BACKUP_FILE"

                  # Upload to S3
                  aws s3 cp $BACKUP_FILE s3://tars-backups/postgres/

                  # Cleanup old local backups (keep 7 days)
                  find /backups -name "tars-*.sql.gz" -mtime +7 -delete
              env:
                - name: POSTGRES_HOST
                  value: postgres.tars-prod.svc.cluster.local
                - name: POSTGRES_USER
                  valueFrom:
                    secretKeyRef:
                      name: postgres-credentials
                      key: username
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: postgres-credentials
                      key: password
              volumeMounts:
                - name: backups
                  mountPath: /backups
          volumes:
            - name: backups
              persistentVolumeClaim:
                claimName: postgres-backups
          restartPolicy: OnFailure
```

**Restore:**
```bash
# Download backup
aws s3 cp s3://tars-backups/postgres/tars-20251128-020000.sql.gz .

# Restore to database
gunzip -c tars-20251128-020000.sql.gz | \
  kubectl exec -i postgres-0 -n tars-prod -- psql -U tars -d tars

# Verify
kubectl exec -it postgres-0 -n tars-prod -- psql -U tars -d tars -c "\dt"
```

---

### Rolling Restarts

#### Zero-Downtime Deployment

**Prerequisites:**
- Multiple replicas (min 2)
- PodDisruptionBudget configured
- Health checks configured

**Procedure:**

1. **Check current state:**
   ```bash
   kubectl get pods -n tars-prod -l app=dashboard-api
   ```

2. **Initiate rolling restart:**
   ```bash
   kubectl rollout restart deployment/dashboard-api -n tars-prod
   ```

3. **Monitor progress:**
   ```bash
   kubectl rollout status deployment/dashboard-api -n tars-prod
   # Expected: "deployment "dashboard-api" successfully rolled out"
   ```

4. **Watch pod recreation:**
   ```bash
   kubectl get pods -n tars-prod -l app=dashboard-api -w
   ```

5. **Verify health:**
   ```bash
   for i in {1..10}; do
     curl -s https://tars.company.com/health | jq .status
     sleep 1
   done
   # All should return "healthy"
   ```

**Rollback (if needed):**
```bash
# Rollback to previous version
kubectl rollout undo deployment/dashboard-api -n tars-prod

# Rollback to specific revision
kubectl rollout undo deployment/dashboard-api --to-revision=3 -n tars-prod

# Check rollout history
kubectl rollout history deployment/dashboard-api -n tars-prod
```

---

### Certificate Renewal

#### cert-manager (Automatic)

**Configured via Ingress annotations:**
```yaml
metadata:
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
```

**Monitor certificate expiry:**
```bash
# Check certificate
kubectl get certificate -n tars-prod

# Describe certificate
kubectl describe certificate tars-tls -n tars-prod

# Force renewal (if needed)
kubectl delete certificate tars-tls -n tars-prod
# cert-manager will recreate automatically
```

**Manual verification:**
```bash
# Check expiry date
echo | openssl s_client -servername tars.company.com -connect tars.company.com:443 2>/dev/null | \
  openssl x509 -noout -dates
```

#### Manual Certificate Renewal

**If not using cert-manager:**

1. **Generate CSR:**
   ```bash
   openssl req -new -newkey rsa:2048 -nodes \
     -keyout tars.key \
     -out tars.csr \
     -subj "/CN=tars.company.com/O=Company/C=US"
   ```

2. **Submit CSR to CA** (Let's Encrypt, DigiCert, etc.)

3. **Update Kubernetes secret:**
   ```bash
   kubectl create secret tls tars-tls \
     --cert=tars.crt \
     --key=tars.key \
     --dry-run=client -o yaml | kubectl apply -f - -n tars-prod
   ```

4. **Reload Ingress (if needed):**
   ```bash
   kubectl delete pod -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx
   ```

---

## Incident Response Playbooks

### Playbook 1: High API Latency

**Symptoms:**
- p95 latency > 500ms
- p99 latency > 1000ms
- User reports of slow response

**Triage Steps:**

1. **Check current metrics:**
   ```bash
   # Prometheus query (via port-forward or Grafana)
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
   ```

2. **Identify slow endpoints:**
   ```bash
   kubectl logs -l app=dashboard-api -n tars-prod --tail=1000 | \
     grep -E "duration=[0-9]+" | \
     awk -F'duration=' '{print $2}' | \
     sort -n | tail -20
   ```

3. **Check resource utilization:**
   ```bash
   kubectl top pods -n tars-prod -l app=dashboard-api
   ```

**Root Cause Analysis:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High CPU (>80%) | Compute-bound operation | Scale horizontally or vertically |
| High Memory (>4GB) | Memory leak or large response | Check for leaks, optimize queries |
| Low CPU/Memory | External dependency slow | Check Redis, PostgreSQL, external APIs |
| Intermittent spikes | Thundering herd, rate limit | Implement circuit breaker, adjust rate limits |

**Resolution:**

```bash
# Horizontal scaling (immediate relief)
kubectl scale deployment dashboard-api --replicas=5 -n tars-prod

# Vertical scaling (requires restart)
kubectl set resources deployment dashboard-api \
  --limits=cpu=4000m,memory=8Gi \
  --requests=cpu=1000m,memory=2Gi \
  -n tars-prod

# Check external dependencies
redis-cli --latency-history -i 1
kubectl exec -it postgres-0 -n tars-prod -- psql -U tars -c "SELECT * FROM pg_stat_activity;"
```

**Post-Incident:**
- Update HPA targets if load pattern changed
- Optimize slow queries
- Add caching layer if applicable

---

### Playbook 2: Elevated Anomaly Rate

**Symptoms:**
- Anomaly detection rate > 5%
- Drift detector alerts firing
- KPI regressions detected

**Triage Steps:**

1. **Check anomaly dashboard:**
   ```bash
   # Query Prometheus
   rate(anomaly_detected_total[5m]) > 0.05
   ```

2. **Identify affected services:**
   ```bash
   kubectl logs -l app=anomaly-detector -n tars-prod --tail=100 | grep "ANOMALY"
   ```

3. **Check recent changes:**
   ```bash
   # Recent deployments
   kubectl rollout history deployment -n tars-prod

   # Git commits
   git log --since="1 hour ago" --oneline
   ```

**Root Cause Analysis:**

| Anomaly Type | Likely Cause | Solution |
|--------------|--------------|----------|
| Latency spike | New deployment, resource saturation | Rollback or scale |
| Error rate increase | Bug introduced, external API down | Rollback, fix bug |
| Traffic pattern change | Marketing campaign, bot attack | Adjust rate limits, block IPs |
| Data drift | Model staleness, input distribution shift | Retrain model, update features |

**Resolution:**

```bash
# Rollback deployment
kubectl rollout undo deployment/dashboard-api -n tars-prod

# Adjust anomaly threshold (if false positive)
# Edit: observability/anomaly_detector_lightweight.py
# ANOMALY_THRESHOLD = 3.0  # Increase from 2.5

# Retrain drift detector
python observability/drift_detector.py --retrain --baseline-window 7d
```

**Post-Incident:**
- Update baseline metrics
- Tune anomaly thresholds
- Add pre-deployment canary checks

---

### Playbook 3: GA KPI Regression

**Symptoms:**
- General Availability KPIs < target (99.9% availability, <200ms p95)
- Stability monitor alerts
- Customer complaints

**Triage Steps:**

1. **Check KPI dashboard:**
   ```bash
   python observability/ga_kpi_collector.py --report
   ```

2. **Identify regression window:**
   ```bash
   python observability/regression_analyzer.py --window 24h
   ```

3. **Check 7-day stability:**
   ```bash
   python observability/stability_monitor_7day.py
   ```

**Root Cause Analysis:**

```bash
# Generate retrospective
python scripts/generate_retrospective.py --incident INC-12345

# Output:
# - Timeline of events
# - Affected services
# - Deployment changes
# - External factors (cloud provider issues)
```

**Resolution:**

```bash
# Emergency hotfix process
git checkout -b hotfix/inc-12345
# Apply fix
git commit -m "hotfix: Fix KPI regression in agent orchestration"
git push origin hotfix/inc-12345

# Deploy hotfix (bypass CI for emergency)
kubectl set image deployment/multi-agent \
  multi-agent=company/tars:1.0.2-hotfix-inc12345 \
  -n tars-prod

# Monitor KPIs
watch -n 5 'python observability/ga_kpi_collector.py --format json | jq ".availability,.p95_latency"'
```

**Post-Incident:**
- Post-mortem meeting (within 48h)
- Update monitoring thresholds
- Add pre-deployment KPI gates

---

### Playbook 4: Security Incident

**Symptoms:**
- Unusual authentication attempts
- Unauthorized API access
- Data exfiltration detected
- CVE alert for dependency

**Triage Steps:**

1. **Isolate affected services:**
   ```bash
   # Block external access (emergency)
   kubectl scale deployment ingress-nginx --replicas=0 -n ingress-nginx
   ```

2. **Audit logs:**
   ```bash
   # Authentication failures
   kubectl logs -l app=dashboard-api -n tars-prod | grep "401 Unauthorized" | tail -100

   # Unusual access patterns
   kubectl logs -l app=dashboard-api -n tars-prod | grep -E "rate_limit_exceeded|suspicious"
   ```

3. **Check compromised credentials:**
   ```bash
   # JWT tokens issued recently
   kubectl exec -it redis-0 -n tars-prod -- redis-cli KEYS "jwt:*" | wc -l

   # Invalidate all sessions
   kubectl exec -it redis-0 -n tars-prod -- redis-cli FLUSHDB
   ```

**Root Cause Analysis:**

| Incident Type | Action |
|---------------|--------|
| **Compromised credentials** | Rotate all secrets, force re-authentication |
| **CVE in dependency** | Emergency patch, regenerate SBOM, scan with Trivy |
| **DDoS attack** | Enable aggressive rate limiting, block IPs |
| **Data breach** | Isolate service, audit logs, notify security team |

**Resolution:**

```bash
# CVE response
# 1. Scan SBOM
trivy sbom release/v1.0.2/sbom/tars-v1.0.2-cyclonedx.json

# 2. Patch dependency
pip install --upgrade <vulnerable-package>

# 3. Regenerate SBOM
python security/sbom_generator.py --output-dir release/v1.0.2-patch/sbom

# 4. Deploy patch
docker build -t company/tars:1.0.2-security-patch .
kubectl set image deployment/dashboard-api dashboard-api=company/tars:1.0.2-security-patch -n tars-prod

# 5. Verify
trivy image company/tars:1.0.2-security-patch
```

**Post-Incident:**
- Security audit (use security/security_audit.py)
- Penetration testing
- Update security policies

---

### Playbook 5: Compliance Control Violation

**Symptoms:**
- Compliance enforcer blocking operations
- Audit log entries for violations
- Failed compliance checks

**Triage Steps:**

1. **Check compliance status:**
   ```bash
   # Query compliance API
   curl -H "Authorization: Bearer $TOKEN" \
     https://tars.company.com/api/compliance/status | jq
   ```

2. **Identify violated control:**
   ```bash
   kubectl logs -l app=dashboard-api -n tars-prod | grep "ComplianceViolation"
   ```

3. **Check enforcement mode:**
   ```bash
   # If in "enforce" mode, operations are blocked
   # If in "warn" mode, operations proceed with warnings
   kubectl get configmap tars-config -n tars-prod -o yaml | grep enforcement_mode
   ```

**Root Cause Analysis:**

| Control | Violation | Resolution |
|---------|-----------|------------|
| **encryption_at_rest** | Data written without encryption | Enable AES encryption |
| **encryption_in_transit** | TLS disabled or expired cert | Renew certificate |
| **audit_logging** | Audit logs not being written | Check log aggregator |
| **data_retention** | Old data not purged | Run retention cleanup job |

**Resolution:**

```bash
# Enable encryption (if disabled)
kubectl set env deployment/dashboard-api \
  ENCRYPTION_ENABLED=true \
  AES_KEY_FILE=/run/secrets/aes.key \
  -n tars-prod

# Renew TLS certificate (see Certificate Renewal)

# Enable audit logging
kubectl set env deployment/dashboard-api \
  AUDIT_LOG_ENABLED=true \
  AUDIT_LOG_PATH=/var/log/audit/audit.json \
  -n tars-prod

# Run data retention cleanup
kubectl create job --from=cronjob/data-retention-cleanup manual-cleanup-$(date +%s) -n tars-prod
```

**Post-Incident:**
- Review compliance policies
- Update documentation
- Add compliance checks to CI/CD

---

## Disaster Recovery

### Backup Strategy

**Backup Types:**

| Type | Frequency | Retention | Storage |
|------|-----------|-----------|---------|
| **Full Backup** | Weekly | 90 days | S3 Glacier |
| **Incremental Backup** | Daily | 30 days | S3 Standard |
| **WAL Archives** | Continuous | 7 days | S3 Standard |
| **Config Snapshots** | On change | 30 days | Git |

**Backup Procedures:**

```bash
# PostgreSQL full backup
pg_basebackup -h postgres.tars-prod.svc.cluster.local -U tars \
  -D /backups/postgres-full-$(date +%Y%m%d) -Ft -z -P

# Redis backup (RDB)
kubectl exec -it redis-0 -n tars-prod -- redis-cli BGSAVE

# ChromaDB backup
tar -czf chromadb-backup-$(date +%Y%m%d).tar.gz /opt/tars/chromadb

# Configuration backup (Git)
kubectl get configmap,secret -n tars-prod -o yaml > config-backup-$(date +%Y%m%d).yaml
```

---

### Recovery Procedures

#### Recovery Time Objective (RTO): 4 hours
#### Recovery Point Objective (RPO): 24 hours

**Full System Recovery:**

1. **Provision infrastructure:**
   ```bash
   # Kubernetes cluster
   eksctl create cluster -f cluster-config.yaml

   # Or Terraform
   terraform apply -var-file=prod.tfvars
   ```

2. **Restore configuration:**
   ```bash
   # Apply secrets
   kubectl apply -f config-backup-20251128.yaml

   # Deploy Helm chart
   helm install tars charts/tars -n tars-prod -f values-prod.yaml
   ```

3. **Restore databases:**
   ```bash
   # PostgreSQL
   gunzip -c postgres-full-20251128.tar.gz | \
     kubectl exec -i postgres-0 -n tars-prod -- tar -xf - -C /var/lib/postgresql/data

   # Redis (from RDB)
   kubectl cp dump.rdb redis-0:/data/dump.rdb -n tars-prod
   kubectl exec -it redis-0 -n tars-prod -- redis-cli SHUTDOWN NOSAVE
   kubectl delete pod redis-0 -n tars-prod  # Restart to load RDB

   # ChromaDB
   kubectl cp chromadb-backup-20251128.tar.gz chromadb-0:/tmp/ -n tars-prod
   kubectl exec -it chromadb-0 -n tars-prod -- tar -xzf /tmp/chromadb-backup-20251128.tar.gz -C /
   ```

4. **Verify recovery:**
   ```bash
   # Health checks
   kubectl get pods -n tars-prod
   curl https://tars.company.com/health

   # Data integrity
   kubectl exec -it postgres-0 -n tars-prod -- psql -U tars -c "SELECT COUNT(*) FROM agents;"
   ```

5. **Resume operations:**
   ```bash
   # Enable external access
   kubectl scale deployment ingress-nginx --replicas=3 -n ingress-nginx

   # Notify users
   echo "System restored at $(date)" | mail -s "T.A.R.S. Recovery Complete" ops@company.com
   ```

---

### Cold Start Procedure

**Scenario:** Complete system restart from scratch (e.g., region failure, infrastructure replacement).

**Procedure:**

1. **Validate backups:**
   ```bash
   # Test restore in isolated environment
   aws s3 cp s3://tars-backups/postgres/latest.tar.gz .
   tar -tzf latest.tar.gz  # List contents
   ```

2. **Deploy infrastructure (IaC):**
   ```bash
   # Terraform
   cd infra/terraform
   terraform init
   terraform plan -var-file=prod.tfvars
   terraform apply -var-file=prod.tfvars -auto-approve

   # Or CloudFormation
   aws cloudformation create-stack \
     --stack-name tars-prod \
     --template-body file://infrastructure.yaml \
     --parameters file://prod-params.json
   ```

3. **Bootstrap Kubernetes:**
   ```bash
   # Install cert-manager
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

   # Install Prometheus Operator
   helm install prometheus-operator prometheus-community/kube-prometheus-stack -n monitoring

   # Install Redis Operator
   helm install redis-operator ot-helm/redis-operator -n operators
   ```

4. **Deploy T.A.R.S.:**
   ```bash
   # Create namespace
   kubectl create namespace tars-prod

   # Apply secrets (from Vault or sealed secrets)
   kubectl apply -f sealed-secrets.yaml

   # Install Helm chart
   helm install tars charts/tars -n tars-prod -f values-prod.yaml
   ```

5. **Restore data** (see Recovery Procedures above)

6. **Smoke test:**
   ```bash
   # Run end-to-end tests
   python tests/e2e/test_full_system.py --env production
   ```

7. **Go live:**
   ```bash
   # Update DNS
   aws route53 change-resource-record-sets --hosted-zone-id Z1234 --change-batch file://dns-update.json

   # Monitor
   kubectl logs -f -l app=dashboard-api -n tars-prod
   ```

**Expected Duration:** 2-4 hours (depending on data size)

---

### Air-Gapped Fallback

**Scenario:** Loss of internet connectivity, need to operate in isolated mode.

**Prerequisites:**
- Offline artifact bundle (see Deployment Model 4)
- Local Docker registry
- Pre-downloaded Docker images
- Offline Python wheels

**Procedure:**

1. **Verify offline bundle:**
   ```bash
   ls -lh tars-v1.0.2-airgap.tar.gz
   # Expected: ~5-10GB
   ```

2. **Extract and load:**
   ```bash
   tar -xzf tars-v1.0.2-airgap.tar.gz
   cd tars-v1.0.2-airgap

   # Load Docker images
   for img in images/*.tar; do docker load -i $img; done
   ```

3. **Push to local registry:**
   ```bash
   # Tag and push
   docker tag company/tars:1.0.2 registry.local/tars:1.0.2
   docker push registry.local/tars:1.0.2
   ```

4. **Deploy with air-gap config:**
   ```bash
   helm install tars charts/tars-1.0.2.tgz \
     -n tars-prod \
     -f values-airgap.yaml \
     --set image.repository=registry.local/tars \
     --set airgap.enabled=true
   ```

5. **Disable external integrations:**
   ```yaml
   # values-airgap.yaml
   mlflow:
     s3:
       enabled: false

   telemetry:
     external: false

   updates:
     autoCheck: false
   ```

6. **Operate in isolated mode:**
   - No external API calls
   - Local MLflow tracking only
   - Manual updates (USB transfer)

---

## Diagnostics & Troubleshooting

### Common Failures

#### 1. "Connection refused" (Redis)

**Symptoms:**
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
```

**Diagnosis:**
```bash
# Check Redis status
redis-cli ping
# OR (K8s)
kubectl exec -it redis-0 -n tars-prod -- redis-cli ping
```

**Solutions:**

```bash
# Local: Start Redis
redis-server --port 6379 &

# Docker: Check container
docker-compose ps redis
docker-compose up -d redis

# K8s: Check pod
kubectl get pods -l app=redis -n tars-prod
kubectl logs -f redis-0 -n tars-prod

# If CrashLoopBackOff, check config
kubectl describe pod redis-0 -n tars-prod
```

---

#### 2. "401 Unauthorized" (JWT)

**Symptoms:**
```json
{"error": "Unauthorized", "message": "Invalid or expired token"}
```

**Diagnosis:**
```bash
# Decode JWT
echo $TOKEN | cut -d. -f2 | base64 -d | jq

# Check expiry
echo $TOKEN | cut -d. -f2 | base64 -d | jq .exp
date -d @$(echo $TOKEN | cut -d. -f2 | base64 -d | jq -r .exp)
```

**Solutions:**

```bash
# Renew token
curl -X POST https://tars.company.com/auth/refresh \
  -H "Authorization: Bearer $REFRESH_TOKEN" \
  -H "Content-Type: application/json"

# Or re-login
curl -X POST https://tars.company.com/auth/login \
  -d '{"username":"admin","password":"..."}' \
  -H "Content-Type: application/json"
```

---

#### 3. "Rate limit exceeded"

**Symptoms:**
```json
{"error": "Too Many Requests", "retry_after": 60}
```

**Diagnosis:**
```bash
# Check rate limit status
redis-cli GET "rate_limit:192.168.1.100"

# Check global rate limit
redis-cli GET "rate_limit:global"
```

**Solutions:**

```bash
# Wait for window to reset (60s)
sleep 60

# Or reset rate limit (emergency only)
redis-cli DEL "rate_limit:192.168.1.100"

# Or adjust rate limit config
kubectl set env deployment/dashboard-api \
  RATE_LIMIT_REQUESTS=60 \
  -n tars-prod
```

---

#### 4. "Signature verification failed"

**Symptoms:**
```
SecurityError: RSA signature verification failed
```

**Diagnosis:**
```bash
# Verify public key matches private key
openssl rsa -in rsa_private.pem -pubout | diff - rsa_public.pem

# Check signature
python security/slsa_generator.py \
  --verify \
  --output file.json \
  --public-key rsa_public.pem
```

**Solutions:**

```bash
# Re-sign with correct key
python security/slsa_generator.py \
  artifact.tar.gz \
  --output file.json \
  --sign \
  --signing-key rsa_private.pem

# Or update public key
kubectl create secret generic tars-rsa-keys \
  --from-file=rsa_public.pem \
  --dry-run=client -o yaml | kubectl apply -f - -n tars-prod
```

---

#### 5. "Out of memory" (Pod evicted)

**Symptoms:**
```
OOMKilled: Memory limit exceeded
```

**Diagnosis:**
```bash
# Check pod events
kubectl describe pod dashboard-api-7d8f9c-xk2l -n tars-prod | grep -A5 "Events:"

# Check memory usage
kubectl top pod dashboard-api-7d8f9c-xk2l -n tars-prod
```

**Solutions:**

```bash
# Increase memory limit
kubectl set resources deployment dashboard-api \
  --limits=memory=8Gi \
  --requests=memory=2Gi \
  -n tars-prod

# Or add memory profiling
kubectl set env deployment/dashboard-api \
  PYTHONMALLOC=malloc \
  MALLOC_TRIM_THRESHOLD_=100000 \
  -n tars-prod

# Check for memory leaks
kubectl exec -it dashboard-api-7d8f9c-xk2l -n tars-prod -- \
  python -c "import gc; gc.collect(); print(len(gc.get_objects()))"
```

---

### Log Patterns

#### Success Patterns

```bash
# Successful authentication
grep "JWT token generated" logs/dashboard-api.log

# Successful agent training
grep "Training completed.*episodes=50" logs/multi-agent.log

# Successful hyperparameter sync
grep "HyperSync applied.*latency=[0-9]+ms" logs/hypersync.log
```

#### Error Patterns

```bash
# Authentication failures
grep "401 Unauthorized" logs/dashboard-api.log | tail -20

# Rate limit violations
grep "rate_limit_exceeded" logs/dashboard-api.log

# Database connection errors
grep "psycopg2.OperationalError" logs/dashboard-api.log

# Redis connection errors
grep "redis.exceptions.ConnectionError" logs/*.log

# Agent training failures
grep "Training failed" logs/multi-agent.log
```

---

### Command-Level Triage

#### Quick Diagnostics

```bash
# Check all services (K8s)
kubectl get pods -n tars-prod | grep -v Running

# Check recent errors (last 5 minutes)
kubectl logs --since=5m -l app=dashboard-api -n tars-prod | grep ERROR

# Check disk space
df -h /var/lib/docker

# Check network connectivity
kubectl exec -it dashboard-api-7d8f9c-xk2l -n tars-prod -- \
  nc -zv redis-0.redis-headless 6379

# Check DNS resolution
kubectl exec -it dashboard-api-7d8f9c-xk2l -n tars-prod -- \
  nslookup redis-0.redis-headless.tars-prod.svc.cluster.local
```

#### Performance Diagnostics

```bash
# Check API latency
time curl https://tars.company.com/api/agents

# Check database query performance
kubectl exec -it postgres-0 -n tars-prod -- psql -U tars -c "
  SELECT query, mean_exec_time, calls
  FROM pg_stat_statements
  ORDER BY mean_exec_time DESC
  LIMIT 10;
"

# Check Redis latency
redis-cli --latency-history -i 1

# Check network latency
kubectl exec -it dashboard-api-7d8f9c-xk2l -n tars-prod -- \
  ping -c 10 redis-0.redis-headless
```

---

## Performance Benchmarks

### Baseline Metrics (v1.0.2)

| Metric | Value | Measurement Method |
|--------|-------|-------------------|
| **API Latency (p50)** | 45ms | Load test, 100 concurrent users |
| **API Latency (p95)** | 120ms | Load test, 100 concurrent users |
| **API Latency (p99)** | 180ms | Load test, 100 concurrent users |
| **Throughput** | 2,500 req/s | Load test, 10-minute sustained |
| **Agent Training (quick)** | 2-5s | Single agent, 10 episodes |
| **Agent Training (full)** | 10-20s | Single agent, 50 episodes |
| **HyperSync Latency** | 50-100ms | Single sync operation |
| **SBOM Generation** | 60-90s | Full dependencies, CycloneDX + SPDX |
| **SLSA Provenance** | <1s | Single artifact |
| **Memory Usage (idle)** | 500MB | Per pod |
| **Memory Usage (load)** | 1.5-2GB | Per pod, 100 concurrent users |
| **CPU Usage (idle)** | 50m | Per pod |
| **CPU Usage (load)** | 500-1000m | Per pod, 100 concurrent users |

### Load Test Commands

```bash
# Install dependencies
pip install locust

# Run load test (API)
locust -f tests/load/locustfile.py \
  --host https://tars.company.com \
  --users 100 \
  --spawn-rate 10 \
  --run-time 10m \
  --headless

# Run performance test suite (see Part B)
python performance/run_performance_tests.py \
  --url https://tars.company.com \
  --duration 600 \
  --concurrency 100 \
  --output-json performance-results.json \
  --output-md performance-report.md
```

### Regression Detection

```bash
# Compare against baseline
python performance/run_performance_tests.py \
  --url https://tars-staging.company.com \
  --baseline performance-baseline.json \
  --duration 300 \
  --concurrency 50

# Expected output:
# ✓ p95 latency: 125ms (baseline: 120ms, +4.2%)
# ✗ p99 latency: 250ms (baseline: 180ms, +38.9%) [REGRESSION]
# ✓ Throughput: 2450 req/s (baseline: 2500 req/s, -2.0%)
```

---

## Appendix

### A. CLI Quick Reference

#### Docker Compose

```bash
# Start
docker-compose up -d

# Stop
docker-compose stop

# Logs
docker-compose logs -f [service]

# Restart
docker-compose restart [service]

# Rebuild
docker-compose up -d --build

# Remove
docker-compose down -v
```

#### Kubernetes

```bash
# Pods
kubectl get pods -n tars-prod
kubectl logs -f <pod> -n tars-prod
kubectl describe pod <pod> -n tars-prod
kubectl delete pod <pod> -n tars-prod

# Deployments
kubectl get deployments -n tars-prod
kubectl scale deployment <name> --replicas=3 -n tars-prod
kubectl rollout restart deployment/<name> -n tars-prod
kubectl rollout status deployment/<name> -n tars-prod
kubectl rollout undo deployment/<name> -n tars-prod

# Services
kubectl get svc -n tars-prod
kubectl port-forward svc/<service> 8080:80 -n tars-prod

# Secrets
kubectl get secrets -n tars-prod
kubectl create secret generic <name> --from-literal=key=value
kubectl describe secret <name> -n tars-prod

# ConfigMaps
kubectl get configmap -n tars-prod
kubectl create configmap <name> --from-file=config.yaml

# Ingress
kubectl get ingress -n tars-prod
kubectl describe ingress tars-ingress -n tars-prod

# HPA
kubectl get hpa -n tars-prod
kubectl autoscale deployment <name> --min=2 --max=10 --cpu-percent=70
```

#### Helm

```bash
# Install
helm install tars charts/tars -n tars-prod -f values.yaml

# Upgrade
helm upgrade tars charts/tars -n tars-prod -f values.yaml

# Rollback
helm rollback tars <revision> -n tars-prod

# History
helm history tars -n tars-prod

# Uninstall
helm uninstall tars -n tars-prod

# List
helm list -n tars-prod

# Get values
helm get values tars -n tars-prod
```

---

### B. Configuration Matrix

| Environment | Profile | Encryption | Signing | Rate Limit | Auth | Redis Mode |
|-------------|---------|------------|---------|------------|------|------------|
| **Local** | local | Disabled | Disabled | 100/min | JWT (optional) | Standalone |
| **Docker** | docker | Optional | Optional | 60/min | JWT | Standalone |
| **Staging** | staging | Enabled | Enabled | 60/min | JWT + RBAC | Sentinel |
| **Production** | production | Enabled | Enabled | 30/min | JWT + RBAC + MFA | Cluster |
| **Air-Gap** | airgap | Enabled | Enabled | 30/min | JWT + RBAC | Standalone |

---

### C. File/Directory Map

```
tars/
├── cognition/                      # Cognitive services
│   ├── orchestration-agent/        # Multi-agent orchestrator (Port 8094)
│   ├── automl-pipeline/            # AutoML service (Port 8096)
│   ├── hypersync/                  # HyperSync service (Port 8098)
│   ├── insight-engine/             # Insight engine (Port 8090)
│   ├── adaptive-policy-learner/    # Policy learner (Port 8091)
│   ├── meta-consensus-optimizer/   # Meta optimizer (Port 8092)
│   ├── causal-inference-engine/    # Causal engine (Port 8095)
│   └── shared/                     # Shared libraries (auth, rate limiter)
├── dashboard/
│   ├── api/                        # Dashboard API (Port 3001)
│   └── frontend/                   # React frontend (Port 3000)
├── security/
│   ├── sbom_generator.py           # SBOM generator
│   ├── slsa_generator.py           # SLSA provenance
│   ├── security_audit.py           # Security audit tool (Part C)
│   ├── encryption.py               # AES-256-GCM encryption
│   └── signing.py                  # RSA-PSS signing
├── enterprise_config/              # Enterprise configuration
│   ├── profiles/                   # Environment profiles
│   ├── config_loader.py            # Config loader
│   └── compliance_enforcer.py      # Compliance engine
├── observability/                  # Monitoring and observability
│   ├── anomaly_detector_lightweight.py
│   ├── drift_detector.py
│   ├── ga_kpi_collector.py
│   ├── stability_monitor_7day.py
│   └── regression_analyzer.py
├── performance/                    # Performance testing (Part B)
│   └── run_performance_tests.py
├── scripts/                        # Utility scripts
│   ├── run_api_server.py           # API server launcher
│   ├── generate_secrets.sh         # Secret generator
│   ├── prepare_release_artifacts.py # Release preparation
│   └── rotate_encryption_keys.py   # Key rotation
├── charts/tars/                    # Helm chart
├── k8s/                            # Kubernetes manifests
├── docs/                           # Documentation
│   └── PRODUCTION_RUNBOOK.md       # This file
├── tests/                          # Test suite
│   ├── e2e/                        # End-to-end tests
│   ├── integration/                # Integration tests
│   └── unit/                       # Unit tests
├── logs/                           # Log files (local)
├── .env                            # Environment variables
├── docker-compose.yaml             # Docker Compose config
├── requirements.txt                # Python dependencies
└── README.md                       # Project README
```

---

### D. Emergency Contacts

| Escalation Level | Contact | Response Time |
|------------------|---------|---------------|
| **L1 - On-Call SRE** | sre-oncall@company.com | 15 minutes |
| **L2 - Platform Lead** | platform-lead@company.com | 30 minutes |
| **L3 - Engineering Manager** | eng-manager@company.com | 1 hour |
| **L4 - CTO** | cto@company.com | 2 hours |

**Pager Duty:** https://company.pagerduty.com
**Incident Channel:** #incident-tars (Slack)
**Status Page:** https://status.company.com

---

### E. Glossary

| Term | Definition |
|------|------------|
| **SBOM** | Software Bill of Materials - Inventory of software components |
| **SLSA** | Supply-chain Levels for Software Artifacts - Security framework |
| **PURL** | Package URL - Universal package identifier |
| **CPE** | Common Platform Enumeration - Vulnerability scanner identifier |
| **JWT** | JSON Web Token - Authentication token format |
| **RBAC** | Role-Based Access Control - Authorization model |
| **HPA** | Horizontal Pod Autoscaler - Kubernetes auto-scaling |
| **PDB** | Pod Disruption Budget - High availability guarantee |
| **RTO** | Recovery Time Objective - Max acceptable downtime |
| **RPO** | Recovery Point Objective - Max acceptable data loss |
| **WAL** | Write-Ahead Log - PostgreSQL transaction log |
| **RDB** | Redis Database - Redis snapshot format |
| **AOF** | Append-Only File - Redis persistence format |

---

### F. Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.0.0** | 2025-11-15 | Initial production runbook |
| **1.0.1** | 2025-11-20 | Added air-gap deployment, Vault integration |
| **1.0.2** | 2025-11-28 | Added SBOM/SLSA workflows, security incident playbook |

---

### G. Related Documentation

- [Phase 14.6 Quickstart](PHASE14_6_QUICKSTART.md)
- [Phase 14.6 API Guide](PHASE14_6_API_GUIDE.md)
- [Phase 14.6 Enterprise Hardening](PHASE14_6_ENTERPRISE_HARDENING.md)
- [Phase 14.7 Task 1 Summary](PHASE14_7_TASK1_COMPLETION_SUMMARY.md)
- [SBOM Generator README](../security/README_SBOM.md)
- [SLSA Provenance README](../security/README_SLSA.md)

---

## Feedback and Updates

This runbook is a living document. Please submit updates via:

**Git:** `git add docs/PRODUCTION_RUNBOOK.md && git commit -m "docs: Update runbook"`
**Jira:** Project TARS, Component "Documentation"
**Slack:** #tars-ops

---

**Document Owner:** SRE Team
**Last Reviewed:** 2025-11-28
**Next Review:** 2025-12-28

---

*End of Production Runbook*

# T.A.R.S. Phase 6 Quick Start Guide
## Kubernetes Deployment, Load Testing & Security

**Version:** v0.3.0-alpha
**Phase:** 6 - Production Scaling, Monitoring & Security
**Last Updated:** November 8, 2025

---

## Overview

This guide provides step-by-step instructions for deploying T.A.R.S. to Kubernetes with production-grade features:

✅ **Kubernetes Orchestration** - Multi-pod deployment with autoscaling
✅ **Redis Caching** - Sub-10ms cache with graceful fallback
✅ **PostgreSQL Analytics** - Persistent analytics storage
✅ **Security Hardening** - HTTPS, rate limiting, RBAC
✅ **Prometheus Metrics** - Production-ready observability
✅ **Load Testing** - Validated for 200+ QPS with <250ms P95 latency

**Estimated Setup Time:** 30-45 minutes

---

## Prerequisites

### Required Software

- **Kubernetes Cluster** (v1.24+)
  - Minikube, kind, k3s, or cloud provider (GKE, EKS, AKS)
  - Minimum: 8 CPU cores, 16GB RAM, 100GB storage
  - GPU node(s) recommended for Ollama

- **kubectl** (v1.24+)
  ```bash
  kubectl version --client
  ```

- **Helm** (v3.0+) - Optional but recommended
  ```bash
  helm version
  ```

- **cert-manager** - For TLS certificate management
  ```bash
  kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
  ```

- **NGINX Ingress Controller**
  ```bash
  kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.0/deploy/static/provider/cloud/deploy.yaml
  ```

### Optional Tools

- **k6** - Load testing tool
  ```bash
  # macOS
  brew install k6

  # Linux
  wget https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz
  tar -xzf k6-v0.47.0-linux-amd64.tar.gz
  sudo mv k6 /usr/local/bin/
  ```

- **Locust** - Python load testing
  ```bash
  pip install locust==2.26.0
  ```

---

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/oceanrockr/VDS_TARS.git
cd VDS_TARS
```

### 2. Configure Secrets

**IMPORTANT:** Replace placeholder values in secrets before deploying!

```bash
# Generate JWT secret
JWT_SECRET=$(openssl rand -base64 32)

# Generate PostgreSQL password
POSTGRES_PASSWORD=$(openssl rand -base64 16)

# Update secrets.yaml
sed -i "s/REPLACE_WITH_SECURE_RANDOM_KEY_MINIMUM_32_CHARACTERS/$JWT_SECRET/g" k8s/secrets.yaml
sed -i "s/REPLACE_WITH_SECURE_PASSWORD/$POSTGRES_PASSWORD/g" k8s/secrets.yaml

# Update email for Let's Encrypt
sed -i "s/admin@tars.local/your-email@example.com/g" k8s/ingress.yaml
```

### 3. Build and Push Docker Image

```bash
# Build backend image
cd backend
docker build -t your-registry/tars-backend:0.3.0-alpha .

# Push to registry (replace with your registry)
docker push your-registry/tars-backend:0.3.0-alpha

# Update deployment to use your image
cd ../k8s
sed -i "s|tars-backend:0.3.0-alpha|your-registry/tars-backend:0.3.0-alpha|g" backend-deployment.yaml
```

### 4. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy secrets and config
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml

# Deploy persistent storage
kubectl apply -f k8s/pvc.yaml

# Wait for PVCs to be bound
kubectl get pvc -n tars --watch

# Deploy databases
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/chromadb-deployment.yaml
kubectl apply -f k8s/ollama-deployment.yaml

# Deploy services
kubectl apply -f k8s/service-postgres.yaml
kubectl apply -f k8s/service-redis.yaml
kubectl apply -f k8s/service-chromadb.yaml
kubectl apply -f k8s/service-ollama.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n tars --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n tars --timeout=300s
kubectl wait --for=condition=ready pod -l app=chromadb -n tars --timeout=300s

# Deploy backend
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/service-backend.yaml

# Wait for backend to be ready
kubectl wait --for=condition=ready pod -l app=tars-backend -n tars --timeout=300s

# Deploy ingress
kubectl apply -f k8s/ingress.yaml
```

### 5. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n tars

# Expected output:
# NAME                            READY   STATUS    RESTARTS
# tars-backend-xxxxx-xxxxx        1/1     Running   0
# tars-backend-xxxxx-xxxxx        1/1     Running   0
# tars-backend-xxxxx-xxxxx        1/1     Running   0
# postgres-xxxxx-xxxxx            1/1     Running   0
# redis-xxxxx-xxxxx               1/1     Running   0
# chromadb-xxxxx-xxxxx            1/1     Running   0
# ollama-xxxxx-xxxxx              1/1     Running   0

# Check services
kubectl get svc -n tars

# Check ingress
kubectl get ingress -n tars
```

### 6. Access the Application

```bash
# Get Ingress IP
INGRESS_IP=$(kubectl get ingress tars-ingress -n tars -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Add to /etc/hosts (for local testing)
echo "$INGRESS_IP tars.local api.tars.local" | sudo tee -a /etc/hosts

# Test health endpoint
curl https://tars.local/health

# Expected response:
{
  "status": "healthy",
  "service": "T.A.R.S. Backend",
  "version": "v0.3.0-alpha",
  "timestamp": "2025-11-08T12:00:00.000000"
}
```

---

## Configuration

### Environment Variables

All configuration is managed via [k8s/configmap.yaml](k8s/configmap.yaml). Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_ENABLED` | `true` | Enable Redis caching |
| `REDIS_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `RATE_LIMIT_PER_MINUTE` | `30` | API rate limit per client |
| `ENABLE_HTTPS` | `true` | Enforce HTTPS connections |
| `USE_ADVANCED_RERANKING` | `true` | Enable cross-encoder reranking |
| `PROMETHEUS_ENABLED` | `true` | Enable Prometheus metrics |

### Scaling

```bash
# Horizontal scaling
kubectl scale deployment tars-backend -n tars --replicas=5

# Auto-scaling (HPA)
kubectl autoscale deployment tars-backend -n tars \
  --cpu-percent=70 \
  --min=3 \
  --max=10
```

### Storage Classes

Update PVC storage class if needed:

```yaml
# k8s/pvc.yaml
storageClassName: standard  # Change to: gp2, fast, ssd, etc.
```

---

## Load Testing

### Using k6 (Recommended)

```bash
# Navigate to load test directory
cd backend/tests/load

# Run basic load test (10 VUs, 60 seconds)
k6 run --vus 10 --duration 60s load_test_k6.js

# Run stress test (100 VUs, 5 minutes)
k6 run --vus 100 --duration 300s load_test_k6.js

# Custom base URL
BASE_URL=https://tars.local k6 run --vus 50 --duration 120s load_test_k6.js
```

**Expected Results:**
```
=== T.A.R.S. Load Test Summary ===
Total Requests: 12,453
Requests/sec: 207.55
Error Rate: 0.12%
P50 Latency: 142.34ms
P95 Latency: 238.67ms
P99 Latency: 456.12ms
Cache Hit Ratio: 78.45%
=================================
```

### Using Locust

```bash
# Web UI mode (recommended for first-time users)
locust -f load_test_locust.py --host=https://tars.local

# Open browser to http://localhost:8089
# Configure: 100 users, spawn rate 10/sec, duration 5 minutes

# Headless mode
locust -f load_test_locust.py --host=https://tars.local \
       --users 100 --spawn-rate 10 --run-time 5m --headless
```

### Performance Targets

| Metric | Target | Validation |
|--------|--------|------------|
| P95 Latency | ≤ 250ms | k6 threshold check |
| Throughput | ≥ 200 QPS | Sustained load test |
| Cache Hit Rate | ≥ 75% | Custom metric tracking |
| Error Rate | ≤ 0.5% | HTTP error monitoring |
| Pod Ready Time | < 30s | Kubernetes readiness probes |

---

## Security

### HTTPS/TLS Configuration

**Let's Encrypt (Production):**

```bash
# Cert-manager will automatically issue certificates
# Check certificate status
kubectl get certificate -n tars

# Expected output:
# NAME            READY   SECRET          AGE
# tars-tls-cert   True    tars-tls-cert   2m
```

**Self-Signed Certificates (Development):**

```bash
# Generate self-signed cert
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tars-tls.key -out tars-tls.crt \
  -subj "/CN=tars.local/O=TARS"

# Create secret
kubectl create secret tls tars-tls-cert \
  --cert=tars-tls.crt \
  --key=tars-tls.key \
  -n tars

# Update ingress annotation
# Comment out: cert-manager.io/cluster-issuer: "letsencrypt-prod"
```

### Rate Limiting

**Application-Level (30 requests/min):**

Configured in [backend/app/middleware/security.py](backend/app/middleware/security.py). Adjust in ConfigMap:

```yaml
RATE_LIMIT_PER_MINUTE: "60"  # Increase to 60 requests/min
```

**Ingress-Level (100 requests/sec):**

Configured in [k8s/ingress.yaml](k8s/ingress.yaml):

```yaml
annotations:
  nginx.ingress.kubernetes.io/limit-rps: "200"  # Increase to 200 RPS
```

### Admin Access (RBAC)

Admin client IDs are configured in [k8s/secrets.yaml](k8s/secrets.yaml):

```yaml
ADMIN_CLIENT_IDS: "admin_001,admin_002,admin_003"
```

**Using RBAC in Code:**

```python
from app.middleware.security import RBACMiddleware

@app.delete("/admin/delete-all")
async def admin_only_endpoint(token_data = Depends(get_current_user)):
    RBACMiddleware.require_admin(token_data.client_id)
    # Admin-only logic
    return {"status": "success"}
```

---

## Monitoring

### Prometheus Metrics

**Scrape Configuration:**

```yaml
# Add to Prometheus config
scrape_configs:
  - job_name: 'tars-backend'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - tars
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

**Available Metrics:**

- `tars_http_requests_total` - Total HTTP requests
- `tars_http_request_duration_seconds` - Request latency histogram
- `tars_rag_queries_total` - RAG query counter
- `tars_cache_hits_total` / `tars_cache_misses_total` - Cache metrics
- `tars_active_connections` - WebSocket connections

**Query Examples:**

```promql
# P95 latency
histogram_quantile(0.95, rate(tars_http_request_duration_seconds_bucket[5m]))

# Cache hit rate
rate(tars_cache_hits_total[5m]) /
(rate(tars_cache_hits_total[5m]) + rate(tars_cache_misses_total[5m]))

# Error rate
rate(tars_http_requests_total{status=~"5.."}[5m])
```

### Grafana Dashboards

Import the T.A.R.S. dashboard (create if needed):

1. Open Grafana
2. Import Dashboard → Upload JSON
3. Configure Prometheus data source
4. Dashboard displays:
   - Request rate, latency, errors
   - Cache hit ratio
   - RAG query performance
   - Pod resource usage

---

## Troubleshooting

### Issue: Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n tars

# Common issues:
# 1. Image pull error → Check image registry and credentials
# 2. PVC not bound → Check storage class availability
# 3. Resource limits → Reduce requests/limits in deployment
```

**Solution for GPU Nodes:**

```bash
# If Ollama pod pending due to GPU node selector
# Option 1: Add GPU node
# Option 2: Remove GPU requirement (CPU-only mode)

# Edit ollama-deployment.yaml:
# Remove nodeSelector and GPU resource requests
```

### Issue: Backend Crashes on Startup

```bash
# Check logs
kubectl logs -n tars deployment/tars-backend --tail=100

# Common errors:
# 1. "Redis connection failed" → Check Redis pod status
# 2. "PostgreSQL not available" → Check PostgreSQL pod status
# 3. "Ollama service not available" → Expected, will retry
```

**Solution:**

```bash
# Restart backend after databases are ready
kubectl rollout restart deployment/tars-backend -n tars
```

### Issue: Ingress Certificate Not Issued

```bash
# Check certificate request
kubectl describe certificaterequest -n tars

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Common issues:
# 1. DNS not pointing to Ingress IP
# 2. HTTP-01 challenge failing (firewall?)
# 3. Let's Encrypt rate limit
```

**Solution (Staging First):**

```yaml
# Use staging issuer first
cert-manager.io/cluster-issuer: "letsencrypt-staging"

# After success, switch to production
cert-manager.io/cluster-issuer: "letsencrypt-prod"
```

### Issue: High Latency (>500ms)

```bash
# Check resource usage
kubectl top pods -n tars

# Check for throttling
kubectl describe pod <backend-pod> -n tars | grep -i throttl
```

**Solution:**

```bash
# Increase resource limits
kubectl edit deployment tars-backend -n tars

# Update:
resources:
  limits:
    memory: "8Gi"  # Increase from 4Gi
    cpu: "4000m"   # Increase from 2000m
```

### Issue: Rate Limiting Too Aggressive

```bash
# Check rate limit configuration
kubectl get configmap tars-config -n tars -o yaml | grep RATE_LIMIT

# Update rate limit
kubectl edit configmap tars-config -n tars
# Change RATE_LIMIT_PER_MINUTE: "60"

# Restart backend to apply
kubectl rollout restart deployment/tars-backend -n tars
```

---

## Maintenance

### Update Configuration

```bash
# Edit ConfigMap
kubectl edit configmap tars-config -n tars

# Restart pods to apply changes
kubectl rollout restart deployment/tars-backend -n tars
```

### Backup Databases

```bash
# PostgreSQL backup
kubectl exec -n tars deployment/postgres -- pg_dump -U tars_user tars_analytics > backup.sql

# ChromaDB backup (copy PVC data)
kubectl cp tars/<chromadb-pod>:/chroma/chroma ./chromadb-backup
```

### Update Backend Version

```bash
# Build new image
docker build -t your-registry/tars-backend:0.4.0 backend/
docker push your-registry/tars-backend:0.4.0

# Update deployment
kubectl set image deployment/tars-backend \
  backend=your-registry/tars-backend:0.4.0 \
  -n tars

# Monitor rollout
kubectl rollout status deployment/tars-backend -n tars
```

---

## Next Steps

1. **Integrate with CI/CD**
   - GitHub Actions for automated builds
   - ArgoCD for GitOps deployment
   - Automated testing on PR

2. **Advanced Monitoring**
   - Grafana Loki for log aggregation
   - Jaeger for distributed tracing
   - AlertManager for incident alerts

3. **High Availability**
   - Multi-region deployment
   - PostgreSQL replication
   - Redis Sentinel/Cluster

4. **Cost Optimization**
   - Spot instances for non-critical pods
   - Cluster autoscaling
   - Resource right-sizing

---

## Resources

- **Kubernetes Manifests:** [k8s/](k8s/)
- **Load Testing Scripts:** [backend/tests/load/](backend/tests/load/)
- **Security Middleware:** [backend/app/middleware/security.py](backend/app/middleware/security.py)
- **Implementation Report:** [PHASE6_IMPLEMENTATION_REPORT.md](PHASE6_IMPLEMENTATION_REPORT.md)
- **API Documentation:** `https://tars.local/docs` (after deployment)

---

## Support

For issues or questions:
1. Check pod logs: `kubectl logs -n tars <pod-name>`
2. Review events: `kubectl get events -n tars --sort-by='.lastTimestamp'`
3. Check health: `curl https://tars.local/ready`
4. Consult implementation report for architecture details

---

**Quick Start Guide** | T.A.R.S. Phase 6 | v0.3.0-alpha

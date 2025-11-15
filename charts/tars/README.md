# T.A.R.S. Helm Chart

![Version: 0.3.0](https://img.shields.io/badge/Version-0.3.0-informational?style=flat-square)
![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square)
![AppVersion: 0.3.0-alpha](https://img.shields.io/badge/AppVersion-0.3.0--alpha-informational?style=flat-square)

T.A.R.S. (Temporal Augmented Retrieval System) - Production-grade RAG platform with local LLMs

## Description

This Helm chart deploys the complete T.A.R.S. stack on Kubernetes, including:

- **Backend API** (FastAPI) with advanced RAG capabilities
- **PostgreSQL** for analytics and persistence
- **Redis** for caching (1-hour TTL by default)
- **ChromaDB** for vector embeddings storage
- **Ollama** for local LLM inference (Mistral by default)
- **Ingress** with automatic TLS via cert-manager
- **Prometheus** metrics integration

## Prerequisites

- Kubernetes 1.24+
- Helm 3.8+
- cert-manager (for automatic TLS certificates)
- NGINX Ingress Controller
- (Optional) GPU nodes for Ollama acceleration

## Installation

### Quick Start

```bash
# Add your container registry credentials if using private images
kubectl create secret docker-registry tars-registry \
  --docker-server=your-registry.io \
  --docker-username=your-username \
  --docker-password=your-password \
  --namespace=tars

# Install T.A.R.S.
helm install tars ./charts/tars \
  --namespace tars \
  --create-namespace \
  --set secrets.jwtSecretKey=$(openssl rand -base64 32) \
  --set secrets.postgresPassword=$(openssl rand -base64 16) \
  --set postgresql.auth.password=$(openssl rand -base64 16) \
  --set certManager.email=your-email@domain.com \
  --set ingress.hosts[0].host=tars.yourdomain.com
```

### Custom Values File

Create a `custom-values.yaml`:

```yaml
# Custom domain
ingress:
  hosts:
    - host: tars.yourdomain.com
      paths:
        - path: /
          pathType: Prefix

# Production secrets
secrets:
  jwtSecretKey: "your-secure-jwt-secret-here"
  postgresPassword: "your-secure-postgres-password"
  adminClientIds: "admin_prod_001,admin_prod_002"

# Enable autoscaling
backend:
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

# GPU configuration for Ollama
ollama:
  gpu:
    enabled: true
    count: 1
```

Install with custom values:

```bash
helm install tars ./charts/tars \
  --namespace tars \
  --create-namespace \
  --values custom-values.yaml
```

## Configuration

The following table lists the configurable parameters of the T.A.R.S. chart and their default values.

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.namespace` | Kubernetes namespace | `tars` |
| `global.nameOverride` | Override the chart name | `""` |
| `global.fullnameOverride` | Override the full resource names | `""` |

### Application Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `app.name` | Application name | `T.A.R.S.` |
| `app.version` | Application version | `0.3.0-alpha` |
| `app.environment` | Environment (development/staging/production) | `production` |
| `app.logLevel` | Log level | `INFO` |

### Backend Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend.replicaCount` | Number of backend replicas | `3` |
| `backend.image.repository` | Backend image repository | `tars-backend` |
| `backend.image.tag` | Backend image tag | `0.3.0-alpha` |
| `backend.image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `backend.service.type` | Service type | `ClusterIP` |
| `backend.service.port` | Service port | `8000` |
| `backend.resources.requests.memory` | Memory request | `1Gi` |
| `backend.resources.requests.cpu` | CPU request | `500m` |
| `backend.resources.limits.memory` | Memory limit | `4Gi` |
| `backend.resources.limits.cpu` | CPU limit | `2000m` |

### Autoscaling Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend.autoscaling.enabled` | Enable HPA | `false` |
| `backend.autoscaling.minReplicas` | Minimum replicas | `3` |
| `backend.autoscaling.maxReplicas` | Maximum replicas | `10` |
| `backend.autoscaling.targetCPUUtilizationPercentage` | Target CPU % | `70` |
| `backend.autoscaling.targetMemoryUtilizationPercentage` | Target memory % | `75` |

### PostgreSQL Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `postgresql.enabled` | Enable PostgreSQL | `true` |
| `postgresql.image.repository` | PostgreSQL image | `postgres` |
| `postgresql.image.tag` | PostgreSQL version | `15-alpine` |
| `postgresql.persistence.enabled` | Enable persistence | `true` |
| `postgresql.persistence.size` | PVC size | `10Gi` |
| `postgresql.auth.database` | Database name | `tars_analytics` |
| `postgresql.auth.username` | Database user | `tars_user` |
| `postgresql.auth.password` | Database password | `REPLACE_WITH_SECURE_PASSWORD` |

### Redis Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `redis.enabled` | Enable Redis | `true` |
| `redis.config.maxMemory` | Max memory | `512mb` |
| `redis.config.cacheTTL` | Cache TTL (seconds) | `3600` |
| `redis.persistence.enabled` | Enable persistence | `false` |

### ChromaDB Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chromadb.enabled` | Enable ChromaDB | `true` |
| `chromadb.persistence.enabled` | Enable persistence | `true` |
| `chromadb.persistence.size` | PVC size | `20Gi` |

### Ollama Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ollama.enabled` | Enable Ollama | `true` |
| `ollama.model.name` | LLM model | `mistral:latest` |
| `ollama.persistence.size` | PVC size | `30Gi` |
| `ollama.gpu.enabled` | Enable GPU | `false` |
| `ollama.gpu.count` | Number of GPUs | `1` |

### Ingress Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable Ingress | `true` |
| `ingress.className` | Ingress class | `nginx` |
| `ingress.hosts[0].host` | Hostname | `tars.local` |
| `ingress.tls[0].secretName` | TLS secret name | `tars-tls-cert` |

### cert-manager Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `certManager.enabled` | Enable cert-manager integration | `true` |
| `certManager.email` | Email for Let's Encrypt | `admin@tars.local` |

### RAG Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `config.rag.useSemanticChunking` | Enable semantic chunking | `true` |
| `config.rag.useAdvancedReranking` | Enable reranking | `true` |
| `config.rag.useHybridSearch` | Enable hybrid search | `true` |

### Security Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.jwtSecretKey` | JWT secret key | `REPLACE_WITH_SECURE_RANDOM_KEY_MINIMUM_32_CHARACTERS` |
| `config.security.enableHTTPS` | Enforce HTTPS | `true` |
| `config.security.rateLimitPerMinute` | Rate limit per minute | `30` |

## Upgrading

### Standard Upgrade

```bash
helm upgrade tars ./charts/tars \
  --namespace tars \
  --values custom-values.yaml
```

### Rolling Back

```bash
# View release history
helm history tars -n tars

# Roll back to previous version
helm rollback tars -n tars

# Roll back to specific revision
helm rollback tars 2 -n tars
```

## Uninstalling

```bash
# Uninstall the release
helm uninstall tars -n tars

# Delete the namespace (optional)
kubectl delete namespace tars
```

## Persistence

The chart creates PersistentVolumeClaims for:

- PostgreSQL: 10GB (analytics data)
- ChromaDB: 20GB (vector embeddings)
- Ollama: 30GB (LLM models)
- Logs: 5GB (application logs)

## Production Considerations

### Security

1. **Always** replace default secrets before production deployment
2. Enable HTTPS with valid certificates (via cert-manager)
3. Configure firewall rules to restrict access
4. Use strong admin client IDs
5. Enable rate limiting at both Ingress and application levels

### Performance

1. Enable autoscaling for dynamic workloads
2. Use SSD storage class for ChromaDB (vector search performance)
3. Enable GPU for Ollama if available
4. Adjust resource limits based on actual usage
5. Monitor Prometheus metrics

### High Availability

1. Deploy at least 3 backend replicas
2. Consider PostgreSQL replication (StatefulSet)
3. Use Redis Cluster for multi-master setup
4. Implement backup strategies for all PVCs
5. Set up multi-region deployment (Phase 7 Part 2)

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics/prometheus` on port 9090:

```bash
# Port-forward to view metrics
kubectl port-forward -n tars svc/tars-backend 9090:9090

# View metrics
curl http://localhost:9090/metrics/prometheus
```

### Health Checks

- Liveness: `/health` (pod restart if failing)
- Readiness: `/ready` (remove from load balancer if failing)

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl get pods -n tars

# View pod logs
kubectl logs -n tars <pod-name>

# Describe pod for events
kubectl describe pod -n tars <pod-name>
```

### Database connection issues

```bash
# Check PostgreSQL logs
kubectl logs -n tars deployment/tars-postgres

# Test connection from backend pod
kubectl exec -it -n tars <backend-pod> -- \
  psql -h tars-postgres -U tars_user -d tars_analytics
```

### Certificate issues

```bash
# Check certificate status
kubectl get certificate -n tars

# View cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Use staging issuer for testing
helm upgrade tars ./charts/tars \
  --set ingress.annotations."cert-manager\.io/cluster-issuer"=letsencrypt-staging
```

## Examples

### Development Environment

```yaml
# dev-values.yaml
app:
  environment: development
  logLevel: DEBUG

backend:
  replicaCount: 1
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"

postgresql:
  persistence:
    size: 5Gi

chromadb:
  persistence:
    size: 10Gi

ollama:
  persistence:
    size: 15Gi

ingress:
  hosts:
    - host: tars-dev.local
      paths:
        - path: /
          pathType: Prefix
```

### Production Environment

```yaml
# prod-values.yaml
app:
  environment: production
  logLevel: INFO

backend:
  replicaCount: 5
  autoscaling:
    enabled: true
    minReplicas: 5
    maxReplicas: 20
    targetCPUUtilizationPercentage: 60

postgresql:
  persistence:
    size: 50Gi
    storageClass: fast-ssd

chromadb:
  persistence:
    size: 100Gi
    storageClass: fast-ssd

ollama:
  gpu:
    enabled: true
    count: 2

ingress:
  hosts:
    - host: tars.production.com
      paths:
        - path: /
          pathType: Prefix
```

## Support

- **GitHub Issues**: https://github.com/oceanrockr/VDS_TARS/issues
- **Documentation**: See [PHASE7_QUICKSTART.md](../../PHASE7_QUICKSTART.md)
- **Full Report**: See [PHASE7_IMPLEMENTATION_REPORT.md](../../PHASE7_IMPLEMENTATION_REPORT.md)

## License

MIT License - See LICENSE file for details

## Maintainers

| Name | Email |
|------|-------|
| VDS Development Team | dev@velerondevstudios.com |

---

**Chart Version:** 0.3.0
**App Version:** 0.3.0-alpha
**Maintained By:** Veleron Dev Studios

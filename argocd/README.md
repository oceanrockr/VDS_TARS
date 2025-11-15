# T.A.R.S. ArgoCD Configuration

This directory contains ArgoCD Application manifests for deploying T.A.R.S. to different environments.

## Prerequisites

1. **ArgoCD Installation**

```bash
# Install ArgoCD in your cluster
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
kubectl wait --for=condition=available --timeout=600s \
  deployment/argocd-server -n argocd
```

2. **ArgoCD CLI**

```bash
# Install ArgoCD CLI
wget https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo mv argocd-linux-amd64 /usr/local/bin/argocd
sudo chmod +x /usr/local/bin/argocd
```

3. **Access ArgoCD UI**

```bash
# Get initial admin password
argocd admin initial-password -n argocd

# Port-forward to access UI
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Open browser: https://localhost:8080
# Username: admin
# Password: <from command above>
```

## Deployment

### Staging Environment

```bash
# Apply staging application
kubectl apply -f argocd/tars-staging.yaml

# Sync application
argocd app sync tars-staging

# Watch deployment
argocd app get tars-staging --refresh
```

### Production Environment

```bash
# Apply production application
kubectl apply -f argocd/tars-production.yaml

# Sync application (manual approval)
argocd app sync tars-production

# Watch deployment
argocd app get tars-production --refresh
```

## Configuration

### Staging (`tars-staging.yaml`)

- **Namespace**: `tars-staging`
- **Branch**: `develop`
- **Replicas**: 2
- **Auto-sync**: Enabled
- **Auto-prune**: Enabled
- **Resources**: Reduced for cost savings

### Production (`tars-production.yaml`)

- **Namespace**: `tars`
- **Branch**: `main`
- **Replicas**: 5 (with HPA up to 20)
- **Auto-sync**: Enabled (with manual prune)
- **Auto-prune**: Disabled (manual approval)
- **Resources**: Production-grade with GPU support

## Secret Management

Production secrets should be managed via external secret management systems:

### Option 1: Sealed Secrets

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Create sealed secret
kubeseal --format=yaml < secret.yaml > sealed-secret.yaml

# Apply sealed secret
kubectl apply -f sealed-secret.yaml
```

### Option 2: External Secrets Operator

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets-system \
  --create-namespace

# Configure secret store (e.g., AWS Secrets Manager, Vault)
kubectl apply -f external-secret-store.yaml
```

### Option 3: HashiCorp Vault

```bash
# Install Vault
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault

# Configure Vault auth
vault auth enable kubernetes
vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc"

# Create secrets
vault kv put secret/tars/production \
  jwt_secret_key="..." \
  postgres_password="..."
```

## Monitoring Deployments

### Check Application Status

```bash
# List all applications
argocd app list

# Get application details
argocd app get tars-production

# View sync status
argocd app sync-status tars-production
```

### View Application Logs

```bash
# Get sync operation logs
argocd app logs tars-production

# Get real-time logs
argocd app logs tars-production --follow
```

### Rollback

```bash
# View history
argocd app history tars-production

# Rollback to previous version
argocd app rollback tars-production

# Rollback to specific revision
argocd app rollback tars-production 5
```

## Troubleshooting

### Application is OutOfSync

```bash
# Check differences
argocd app diff tars-production

# Force sync
argocd app sync tars-production --force

# Refresh application
argocd app get tars-production --refresh --hard-refresh
```

### Sync Failed

```bash
# View sync errors
argocd app get tars-production

# View detailed logs
argocd app logs tars-production

# Delete and recreate application
kubectl delete -f argocd/tars-production.yaml
kubectl apply -f argocd/tars-production.yaml
```

### Health Check Failing

```bash
# Check pod status
kubectl get pods -n tars

# Check pod logs
kubectl logs -n tars deployment/tars-backend

# Check events
kubectl get events -n tars --sort-by='.lastTimestamp'
```

## Best Practices

1. **Use Git as Source of Truth**
   - All configuration changes should be committed to Git
   - ArgoCD will automatically sync from Git repository

2. **Environment Separation**
   - Staging: Automatic sync with auto-prune
   - Production: Automatic sync with manual prune approval

3. **Secret Management**
   - Never commit secrets to Git
   - Use external secret management (Sealed Secrets, External Secrets, Vault)

4. **Monitoring**
   - Enable ArgoCD notifications for deployment events
   - Monitor application health via Prometheus/Grafana

5. **Rollback Strategy**
   - Keep revision history (10 for staging, 20 for production)
   - Test rollbacks in staging before production

## Notifications

Configure ArgoCD notifications for deployment events:

```yaml
# argocd-notifications-cm.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-notifications-cm
  namespace: argocd
data:
  service.slack: |
    token: $slack-token
  template.app-deployed: |
    message: |
      Application {{.app.metadata.name}} is now running version {{.app.status.sync.revision}}.
  trigger.on-deployed: |
    - when: app.status.operationState.phase in ['Succeeded']
      send: [app-deployed]
```

## CI/CD Integration

The GitHub Actions workflow automatically triggers ArgoCD sync:

```yaml
# In .github/workflows/deploy.yml
- name: Deploy via ArgoCD
  run: |
    argocd app sync tars-production \
      --revision ${{ github.sha }} \
      --prune \
      --timeout 600
```

## References

- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [GitOps Principles](https://opengitops.dev/)
- [T.A.R.S. Phase 7 Report](../PHASE7_IMPLEMENTATION_REPORT.md)

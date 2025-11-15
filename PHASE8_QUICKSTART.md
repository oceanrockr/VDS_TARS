# T.A.R.S. Phase 8 Quick Start — AI-Powered Ops + Security
**Get AI anomaly detection, auto-remediation, and production security running in 20 minutes**

---

## What You're Getting

**AI-Powered Operations:**
- ML-based anomaly detection (STL + IsolationForest + EWMA ensemble)
- Automated incident correlation and deduplication
- Safe auto-remediation with 5 production playbooks
- Real-time Grafana dashboards for anomalies

**Security Hardening:**
- mTLS ingress + service mesh
- NetworkPolicies (zero-trust default deny)
- Signed images with Cosign + SBOM
- Kyverno admission policies (PSS restricted enforcement)
- External Secrets Operator + SOPS encryption

---

## Prerequisites (2 minutes)

- Kubernetes cluster (v1.25+) with Phase 7.2 observability running
- Prometheus + Alertmanager + Grafana + Loki + Jaeger operational
- `kubectl` and `helm` configured
- Docker registry access for aiops images

---

## Part 1: Deploy AIops Stack (10 minutes)

### Step 1: Install CRDs

```bash
kubectl apply -f aiops/auto-remediator/crds/remediationpolicy.yaml
kubectl apply -f aiops/auto-remediator/crds/remediationaction.yaml
```

### Step 2: Build and Push AIops Images

```bash
# Build anomaly detector
cd aiops/anomaly-detector
docker build -t your-registry/tars-anomaly-detector:0.6.0-alpha .
docker push your-registry/tars-anomaly-detector:0.6.0-alpha

# Build auto-remediator
cd ../auto-remediator
docker build -t your-registry/tars-auto-remediator:0.6.0-alpha .
docker push your-registry/tars-auto-remediator:0.6.0-alpha
```

### Step 3: Deploy via Helm

```bash
helm upgrade tars ./charts/tars \
  --namespace tars \
  --set aiops.anomalyDetector.enabled=true \
  --set aiops.anomalyDetector.image.repository=your-registry/tars-anomaly-detector \
  --set aiops.autoRemediator.enabled=true \
  --set aiops.autoRemediator.image.repository=your-registry/tars-auto-remediator \
  --set aiops.autoRemediator.config.dryRun=true \
  --wait
```

### Step 4: Configure Alertmanager

```bash
kubectl apply -f observability/alertmanager-config-aiops.yaml
```

### Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods -n tars | grep -E 'anomaly|remediator'

# Expected output:
# anomaly-detector-xxxx      1/1     Running
# auto-remediator-yyyy       1/1     Running
```

### Step 6: Import Grafana Dashboard

```bash
kubectl port-forward -n tars svc/grafana 3000:80 &

# Get Grafana password
GRAFANA_PASSWORD=$(kubectl get secret --namespace tars grafana -o jsonpath="{.data.admin-password}" | base64 --decode)

# Import dashboard
curl -X POST http://admin:${GRAFANA_PASSWORD}@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @observability/grafana-dashboard-anomaly.json
```

---

## Part 2: Deploy a Remediation Policy (3 minutes)

### Example: Auto-scale on high latency

```bash
kubectl apply -f - <<EOF
apiVersion: aiops.tars/v1
kind: RemediationPolicy
metadata:
  name: backend-latency-spike
  namespace: tars
spec:
  triggers:
    - type: anomaly
      metric: tars_backend_latency_p95
      minScore: 0.8
  actions:
    - type: ScaleOut
      params:
        deployment: tars-backend
        step: "2"
        maxReplicas: "10"
  cooldownSeconds: 900
  maxExecutionsPerHour: 2
  dryRun: false
  allowedResources:
    - kind: Deployment
      name: tars-backend
EOF
```

### Verify Policy

```bash
kubectl get remediationpolicies -n tars
kubectl describe remediationpolicy backend-latency-spike -n tars
```

---

## Part 3: Security Hardening (10 minutes)

### Step 1: Install cert-manager (if not already installed)

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Step 2: Install Kyverno

```bash
helm repo add kyverno https://kyverno.github.io/kyverno/
helm install kyverno kyverno/kyverno --namespace kyverno --create-namespace
```

### Step 3: Install External Secrets Operator

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets-system --create-namespace
```

### Step 4: Enable Security Features

```bash
helm upgrade tars ./charts/tars \
  --namespace tars \
  --set security.networkPolicy.enabled=true \
  --set security.kyverno.enabled=true \
  --set security.podSecurity.enforce=restricted \
  --set security.secrets.externalSecretsOperator.enabled=true \
  --reuse-values \
  --wait
```

### Step 5: Sign Images with Cosign

```bash
# Generate cosign key pair
cosign generate-key-pair

# Sign anomaly-detector image
cosign sign --key cosign.key your-registry/tars-anomaly-detector:0.6.0-alpha

# Update Kyverno policy with public key
kubectl edit clusterpolicy verify-image-signatures
# Paste contents of cosign.pub into policy
```

### Step 6: Verify Security Policies

```bash
# Test: Try to deploy unsigned image (should be rejected)
kubectl run test-unsigned --image=nginx:latest -n tars
# Expected: Error from admission webhook (Kyverno)

# Test: Network policy (should block unauthorized traffic)
kubectl run test-network --image=busybox -n tars -- wget http://anomaly-detector:8080/health
# Expected: Connection timeout
```

---

## Validation Tests

### Test 1: Anomaly Detection

```bash
# Generate anomalous load
kubectl port-forward -n tars svc/tars-backend 8000:8000 &

# Install k6
snap install k6

# Spike load
cat <<EOF > spike.js
import http from 'k6/http';
export default function() {
  http.get('http://localhost:8000/health');
}
EOF

k6 run --vus 100 --duration 2m spike.js

# Check anomaly detector
kubectl port-forward -n tars svc/anomaly-detector 8080:8080 &
curl http://localhost:8080/anomalies/recent?hours=1 | jq
```

### Test 2: Auto-Remediation (Dry-Run)

```bash
# Check auto-remediator logs
kubectl logs -n tars -l app=auto-remediator --tail=50 -f

# During anomaly, you should see logs like:
# [DRY-RUN] Scaling tars-backend from 3 to 5 replicas
```

### Test 3: Security - Image Signature Verification

```bash
# Deploy a policy that requires only your signed images
kubectl get clusterpolicy verify-image-signatures -o yaml

# Try deploying unsigned image
kubectl run test --image=busybox:latest -n tars
# Should fail with: image signature verification failed
```

---

## Common Issues & Fixes

### Issue: Anomaly detector can't connect to Prometheus

**Solution:**
```bash
# Verify Prometheus URL
kubectl get svc -n tars | grep prometheus

# Update anomaly detector config
kubectl set env deployment/anomaly-detector -n tars \
  PROMETHEUS_URL=http://prometheus-kube-prometheus-prometheus:9090
```

### Issue: Auto-remediator can't modify deployments (permission denied)

**Solution:**
```bash
# Verify RBAC
kubectl get role -n tars auto-remediator -o yaml

# Check service account
kubectl describe sa auto-remediator -n tars
```

### Issue: Kyverno policy blocks all pods

**Solution:**
```bash
# Temporarily set to audit mode
kubectl patch clusterpolicy require-signed-images \
  --type=merge -p '{"spec":{"validationFailureAction":"audit"}}'

# Check policy reports
kubectl get policyreport -A
```

---

## Next Steps

### 1. Configure Slack Notifications

Edit `observability/alertmanager-config-aiops.yaml`:

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#tars-alerts'
```

Apply:
```bash
kubectl apply -f observability/alertmanager-config-aiops.yaml
kubectl rollout restart statefulset/alertmanager-prometheus-kube-prometheus-alertmanager -n tars
```

### 2. Enable Real Remediation

**⚠️ CAUTION:** Test thoroughly in non-production first!

```bash
helm upgrade tars ./charts/tars \
  --namespace tars \
  --set aiops.autoRemediator.config.dryRun=false \
  --reuse-values
```

### 3. Set Up AWS Secrets Manager

```bash
# Create AWS secret
aws secretsmanager create-secret \
  --name tars/jwt-secret \
  --secret-string '{"JWT_SECRET_KEY":"your-secret-key"}'

# Update values
helm upgrade tars ./charts/tars \
  --set security.secrets.externalSecretsOperator.enabled=true \
  --set security.secrets.externalSecretsOperator.backend=aws \
  --set security.secrets.externalSecretsOperator.aws.region=us-east-1 \
  --reuse-values
```

### 4. Enable SOPS for Git-Encrypted Values

```bash
# Encrypt production values
sops --encrypt \
  --kms 'arn:aws:kms:us-east-1:ACCOUNT:key/KEY_ID' \
  --encrypted-regex '^(secrets|jwt)$' \
  charts/tars/values-production.yaml > charts/tars/values-production.enc.yaml

# Deploy with encrypted values
sops --decrypt charts/tars/values-production.enc.yaml | \
  helm upgrade tars ./charts/tars -f -
```

---

##Performance Expectations

| Metric | Target |
|--------|--------|
| Anomaly detection latency | <2s |
| Anomaly precision | ≥85% |
| Alert noise reduction | ≥40% |
| MTTR improvement | ≥30% |
| Auto-remediation success rate | ≥90% (dry-run validated) |
| Policy evaluation time (Kyverno) | <100ms |
| Image signature verification | <500ms |

---

## Runbooks

Quick links to operational runbooks:

- [High Latency → ScaleOut](runbooks/RUNBOOK_HighLatency_ScaleOut.md)
- [High 5xx → Rollback](runbooks/RUNBOOK_High5xx_Rollback.md)
- [Anomaly Triage](runbooks/RUNBOOK_Anomaly_Triage.md)

---

## Support & Documentation

**Full Documentation**: [PHASE8_IMPLEMENTATION_SUMMARY.md](PHASE8_IMPLEMENTATION_SUMMARY.md)

**Quick Commands Reference**:

```bash
# Check anomaly scores
kubectl port-forward -n tars svc/anomaly-detector 8080:8080
curl http://localhost:8080/models/status | jq

# List remediation policies
kubectl get remediationpolicies -n tars

# View recent remediation actions
kubectl get remediationactions -n tars --sort-by=.metadata.creationTimestamp

# Check security policies
kubectl get clusterpolicies

# View network policies
kubectl get networkpolicies -n tars
```

---

**Version**: 1.0
**T.A.R.S. Version**: v0.6.0-alpha
**Estimated Setup Time**: 20 minutes
**Last Updated**: November 9, 2025

🎉 **Congratulations!** T.A.R.S. now has production-grade AI-powered operations and security hardening!

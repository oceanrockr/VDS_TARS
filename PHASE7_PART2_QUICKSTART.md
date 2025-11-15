# T.A.R.S. Phase 7 Part 2 — Quick Start Guide
**Get Observability + Multi-Region + Cost Optimization Running in 30 Minutes**

## What You're Getting

- **Grafana**: 13 production dashboards with real-time RAG metrics
- **Loki**: Centralized logging with 30-day retention
- **Jaeger**: Distributed tracing across all services
- **Prometheus**: 42 production alert rules
- **Multi-Region**: PostgreSQL replication + GeoDNS failover
- **Cost Tools**: Kubecost, VPA, and Cluster Autoscaler

---

## Prerequisites (5 minutes)

### Required
- Kubernetes cluster (v1.25+) with 3+ nodes
- `kubectl` configured and working
- `helm` v3.x installed
- 12 CPU cores total (4 per node)
- 24Gi RAM total (8Gi per node)
- 500Gi storage available

### Verify Your Setup
```bash
# Check kubectl
kubectl version --client

# Check Helm
helm version

# Check cluster access
kubectl cluster-info

# Check resources
kubectl top nodes
```

---

## Part 1: Deploy Observability Stack (15 minutes)

### Step 1: Run Observability Setup

```bash
cd observability
chmod +x setup.sh
./setup.sh
```

**What it does**:
- Installs Prometheus with kube-prometheus-stack
- Deploys Grafana with pre-configured dashboards
- Sets up Loki for log aggregation
- Configures Jaeger for distributed tracing
- Deploys OpenTelemetry Collector
- Applies 42 Prometheus alert rules

**Expected output**:
```
================================================
T.A.R.S. Observability Stack Setup
================================================
✓ Prerequisites satisfied
✓ Namespace ready
✓ Helm repositories added
✓ Prometheus installed
✓ Alert rules configured
✓ Loki installed
✓ Jaeger installed
✓ OpenTelemetry Collector installed
✓ Grafana installed
✓ Dashboard imported
✓ ServiceMonitors created
================================================
Observability Stack Installation Complete!
================================================
```

### Step 2: Verify Deployment

```bash
# Check all observability pods are running
kubectl get pods -n tars | grep -E 'grafana|loki|jaeger|prometheus'

# Expected: All pods should show "Running" status
```

### Step 3: Access the Dashboards

**Grafana**:
```bash
# Get Grafana password
kubectl get secret --namespace tars grafana -o jsonpath="{.data.admin-password}" | base64 --decode
echo

# Port forward
kubectl port-forward -n tars svc/grafana 3000:80

# Open browser: http://localhost:3000
# Login: admin / <password from above>
```

**Prometheus**:
```bash
kubectl port-forward -n tars svc/prometheus-kube-prometheus-prometheus 9090:9090

# Open browser: http://localhost:9090
```

**Jaeger**:
```bash
kubectl port-forward -n tars svc/jaeger-query 16686:16686

# Open browser: http://localhost:16686
```

### Step 4: Verify Metrics Are Flowing

**In Grafana**:
1. Go to http://localhost:3000
2. Click "Dashboards" → "T.A.R.S. - Advanced RAG Observability"
3. You should see live metrics populating

**In Prometheus**:
1. Go to http://localhost:9090/targets
2. Verify all T.A.R.S. services show as "UP"

**In Jaeger**:
1. Go to http://localhost:16686
2. Select "tars-backend" from services
3. Click "Find Traces" (traces appear after backend activity)

---

## Part 2: Cost Optimization Tools (10 minutes)

### Step 1: Install Kubecost

```bash
cd scripts
./setup_kubecost.sh
```

**Wait for pods**:
```bash
kubectl wait --for=condition=ready pod -l app=cost-analyzer -n kubecost --timeout=300s
```

**Access Kubecost**:
```bash
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090

# Open: http://localhost:9090
```

### Step 2: Install Vertical Pod Autoscaler (VPA)

```bash
./setup_vpa.sh
```

**View VPA recommendations**:
```bash
vpa-monitor
```

**Example output**:
```
=== VPA Recommendations for T.A.R.S. ===

=== tars-backend-vpa ===
Target: Deployment/tars-backend
Update Mode: Auto

Current Resource Recommendations:
  Container: backend
    CPU:
      Target: 750m
      Lower Bound: 500m
      Upper Bound: 1500m
    Memory:
      Target: 1536Mi
      Lower Bound: 1Gi
      Upper Bound: 2Gi
```

### Step 3: Install Cluster Autoscaler

```bash
# Set your cluster name
CLUSTER_NAME=tars-cluster ./setup_autoscaler.sh
```

**Monitor autoscaler**:
```bash
autoscaler-monitor
```

**Verify it's working**:
```bash
kubectl logs -n kube-system deployment/cluster-autoscaler --tail=50
```

---

## Part 3: Multi-Region Deployment (Optional, 30 minutes)

**Skip this section if you're deploying single-region.**

### Step 1: Deploy Primary Region (US-East)

```bash
# Install Helm chart with primary region values
helm install tars ./charts/tars \
  --namespace tars \
  --create-namespace \
  --values charts/tars/values.yaml \
  --values region-overlays/us-east/values-east.yaml \
  --wait

# Deploy PostgreSQL primary
kubectl apply -f region-overlays/us-east/postgres-primary.yaml

# Deploy Redis cluster
kubectl apply -f region-overlays/redis-cluster.yaml
```

### Step 2: Deploy Secondary Region (US-West)

**On your secondary cluster**:
```bash
# Install Helm chart with secondary region values
helm install tars ./charts/tars \
  --namespace tars \
  --create-namespace \
  --values charts/tars/values.yaml \
  --values region-overlays/us-west/values-west.yaml \
  --wait

# Deploy PostgreSQL replica
kubectl apply -f region-overlays/us-west/postgres-replica.yaml
```

### Step 3: Verify Replication

**Check PostgreSQL replication status**:
```bash
# On primary cluster
kubectl exec -n tars postgres-primary-0 -- \
  psql -U postgres -c "SELECT * FROM pg_stat_replication;"

# On secondary cluster
kubectl exec -n tars postgres-replica-0 -- \
  psql -U postgres -c "SELECT now() - pg_last_xact_replay_timestamp() AS lag;"
```

**Expected lag**: <5 seconds

**Test replication**:
```bash
# Insert data on primary
kubectl exec -n tars postgres-primary-0 -- \
  psql -U postgres -c "CREATE TABLE test (id INT, data TEXT); INSERT INTO test VALUES (1, 'hello');"

# Verify on replica (wait 5 seconds)
sleep 5
kubectl exec -n tars postgres-replica-0 -- \
  psql -U postgres -c "SELECT * FROM test;"
```

### Step 4: Configure GeoDNS (Optional)

**For AWS Route 53**:
```bash
cd region-overlays
terraform init
terraform plan -var-file=geodns-config.tfvars
terraform apply
```

**For Cloudflare**:
- Use the Cloudflare Terraform configuration in `geodns-config.yaml`
- Update with your account ID and zone ID
- Apply with `terraform apply`

---

## Quick Validation Tests

### Test 1: Verify Metrics Collection

```bash
# Check if backend is exporting metrics
kubectl exec -n tars deployment/tars-backend -- \
  curl -s http://localhost:8000/metrics/prometheus | grep tars_rag_query

# Should show metrics like:
# tars_rag_query_duration_seconds_bucket{le="0.1"} 45
# tars_rag_query_duration_seconds_count 127
```

### Test 2: Verify Logs Are Being Collected

```bash
# Query Loki for recent logs
kubectl port-forward -n tars svc/loki 3100:3100 &

curl -G "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={namespace="tars"}' \
  | jq '.data.result[].stream'
```

### Test 3: Verify Alerts Are Active

```bash
kubectl port-forward -n tars svc/prometheus-kube-prometheus-prometheus 9090:9090 &

# Check loaded alert rules
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].name'
```

### Test 4: Verify Cost Tracking

```bash
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090 &

# Get total costs
curl http://localhost:9090/model/aggregatedCostModel?window=1h | jq
```

---

## Common Issues & Fixes

### Issue: Grafana dashboard is empty

**Solution**:
```bash
# Wait 2-3 minutes for metrics to populate
# Force refresh datasources
kubectl exec -n tars deployment/grafana -- \
  curl -X POST http://admin:admin@localhost:3000/api/admin/provisioning/datasources/reload
```

### Issue: Loki can't find logs

**Solution**:
```bash
# Check Promtail is running
kubectl get pods -n tars -l app.kubernetes.io/name=promtail

# Check Promtail logs
kubectl logs -n tars -l app.kubernetes.io/name=promtail --tail=50
```

### Issue: Jaeger has no traces

**Solution**:
```bash
# Traces require application instrumentation
# For now, verify Jaeger is receiving health checks
kubectl exec -n tars deployment/jaeger-collector -- \
  curl -s http://localhost:14269/ | grep -i health
```

### Issue: VPA not updating pods

**Solution**:
```bash
# VPA may take 4-8 minutes to generate recommendations
# Check VPA recommender logs
kubectl logs -n kube-system deployment/vpa-recommender --tail=50

# Verify update mode
kubectl get vpa -n tars -o jsonpath='{.items[*].spec.updatePolicy.updateMode}'
```

### Issue: Cluster Autoscaler not scaling

**Solution**:
```bash
# Check autoscaler logs for errors
kubectl logs -n kube-system deployment/cluster-autoscaler --tail=100

# Verify node group auto-discovery tags are set
# (AWS ASG should have tags: k8s.io/cluster-autoscaler/enabled=true)

# Check for pending pods
kubectl get pods --all-namespaces --field-selector=status.phase==Pending
```

### Issue: PostgreSQL replication lag is high

**Solution**:
```bash
# Check network connectivity
kubectl exec -n tars postgres-replica-0 -- ping -c 3 postgres-primary

# Check WAL sender queue
kubectl exec -n tars postgres-primary-0 -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_replication;"

# Consider increasing max_wal_senders if needed
```

---

## Next Steps

### 1. Configure Alerts

Edit alert thresholds in `observability/prom-alerts.yaml`:

```yaml
# Example: Adjust latency threshold
- alert: HighRAGQueryLatency
  expr: |
    histogram_quantile(0.95, ...) > 0.5  # Change to 500ms
```

Apply changes:
```bash
kubectl apply -f observability/prom-alerts.yaml
```

### 2. Set Up Notification Channels

**Slack integration**:
```yaml
# Add to AlertManager config
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#tars-alerts'
```

**Email integration**:
```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'team@example.com'
        from: 'alertmanager@tars.local'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alerts@example.com'
        auth_password: 'YOUR_PASSWORD'
```

### 3. Enable Application Instrumentation

**Add to backend code** (`backend/app/main.py`):

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)
```

**Update environment**:
```bash
OTEL_SERVICE_NAME=tars-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

### 4. Customize Grafana Dashboards

1. Go to http://localhost:3000
2. Navigate to existing dashboard
3. Click "Settings" (gear icon) → "Save As"
4. Customize panels and queries
5. Save your custom dashboard

### 5. Set Up Regular Backups

**PostgreSQL backups** (already configured):
- Schedule: Daily at 2 AM
- Retention: 30 days
- Location: PVC or S3

**Verify backup CronJob**:
```bash
kubectl get cronjob -n tars postgres-backup
```

**Manual backup**:
```bash
kubectl create job --from=cronjob/postgres-backup manual-backup-1 -n tars
```

### 6. Review Cost Reports Weekly

```bash
# Generate cost report
kubecost-report

# Export VPA recommendations
vpa-export
```

Review output and adjust resource requests/limits based on recommendations.

---

## Useful Commands Reference

### Observability

```bash
# Grafana
kubectl port-forward -n tars svc/grafana 3000:80

# Prometheus
kubectl port-forward -n tars svc/prometheus-kube-prometheus-prometheus 9090:9090

# Jaeger
kubectl port-forward -n tars svc/jaeger-query 16686:16686

# Loki
kubectl port-forward -n tars svc/loki 3100:3100

# View logs
kubectl logs -n tars deployment/tars-backend --tail=100 -f

# Query Loki
curl -G "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={namespace="tars",level="ERROR"}'
```

### Cost Optimization

```bash
# Kubecost UI
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090

# Cost report
kubecost-report

# VPA recommendations
vpa-monitor

# Export VPA configs
vpa-export

# Autoscaler status
autoscaler-monitor

# Autoscaler logs
kubectl logs -n kube-system deployment/cluster-autoscaler -f
```

### Multi-Region

```bash
# Check replication status
kubectl exec -n tars postgres-primary-0 -- \
  psql -U postgres -c "SELECT * FROM pg_stat_replication;"

# Check replica lag
kubectl exec -n tars postgres-replica-0 -- \
  psql -U postgres -c "SELECT now() - pg_last_xact_replay_timestamp();"

# Redis cluster status
kubectl exec -n tars redis-cluster-0 -- redis-cli cluster info

# Test GeoDNS
dig tars.example.com
```

### Troubleshooting

```bash
# Check all pods
kubectl get pods -n tars

# Describe pod
kubectl describe pod <pod-name> -n tars

# Pod logs
kubectl logs <pod-name> -n tars --tail=100

# Exec into pod
kubectl exec -it <pod-name> -n tars -- /bin/sh

# Check events
kubectl get events -n tars --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n tars
kubectl top nodes
```

---

## Performance Expectations

### Observability Stack

| Metric | Expected Value |
|--------|---------------|
| Metrics ingestion latency | <100ms |
| Log ingestion latency | <200ms |
| Trace ingestion latency | <50ms |
| Dashboard load time | <2s |
| Query response time | <500ms |
| Alert detection time | <30s |

### Multi-Region

| Metric | Expected Value |
|--------|---------------|
| Replication lag | <5s |
| Failover time | <90s |
| Cross-region latency | 45-60ms (us-east to us-west) |
| Data consistency | 100% (synchronous) |

### Cost Optimization

| Metric | Expected Impact |
|--------|----------------|
| CPU efficiency improvement | +20-30% |
| Memory efficiency improvement | +15-25% |
| Cost reduction | 15-25% |
| Over-provisioning reduction | 60-70% |

---

## Security Checklist

Before going to production:

- [ ] Change Grafana admin password
- [ ] Change PostgreSQL passwords
- [ ] Generate new JWT secret (32+ characters)
- [ ] Enable TLS for all services
- [ ] Configure network policies
- [ ] Set up RBAC with least privilege
- [ ] Enable secret encryption at rest
- [ ] Configure backup encryption
- [ ] Set up audit logging
- [ ] Review and restrict ingress rules
- [ ] Enable pod security policies
- [ ] Configure image scanning
- [ ] Set up vulnerability scanning

---

## Support & Documentation

**Full Documentation**: `PHASE7_PART2_IMPLEMENTATION_REPORT.md`
**Architecture Diagrams**: See implementation report
**Operational Runbooks**: See implementation report Section 9

**Quick Links**:
- Grafana Docs: https://grafana.com/docs/
- Prometheus Docs: https://prometheus.io/docs/
- Loki Docs: https://grafana.com/docs/loki/
- Jaeger Docs: https://www.jaegertracing.io/docs/
- Kubecost Docs: https://docs.kubecost.com/
- VPA Docs: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler
- Cluster Autoscaler Docs: https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler

---

**Version**: 1.0
**Last Updated**: November 9, 2025
**Estimated Setup Time**: 30-45 minutes (observability + cost tools)
**Multi-Region Add-on Time**: +30 minutes

🎉 **Congratulations!** You now have production-grade observability, cost optimization, and multi-region capabilities for T.A.R.S.!

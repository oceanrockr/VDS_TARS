# T.A.R.S. Phase 10 Quick Start Guide
## Cognitive Federation & Adaptive Policy Learning

**Version**: v0.8.0-alpha
**Estimated Time**: 30 minutes

---

## Prerequisites

- ✅ Phase 9 (Federation & Governance) deployed and operational
- ✅ Kubernetes cluster running (or Docker Compose)
- ✅ PostgreSQL database accessible
- ✅ Prometheus and Grafana installed
- ✅ OPA (Open Policy Agent) running

---

## Quick Start

### Step 1: Update Environment Configuration

```bash
cd VDS_TARS

# Copy Phase 10 configuration
cat >> .env <<EOF
# Phase 10: Cognitive Federation
COGNITION_ENABLED=true
INSIGHT_REFRESH_INTERVAL=60
ANALYSIS_WINDOW_MINUTES=60
ADAPTIVE_POLICY_LEARNING=true
META_CONSENSUS_OPTIMIZER=true
ETHICAL_TRAINER_ENABLED=true
EOF
```

### Step 2: Build Docker Images

```bash
# Build all Phase 10 images
docker build -t tars/cognition-insight-engine:0.8.0-alpha cognition/insight-engine/
docker build -t tars/policy-learner:0.8.0-alpha governance/policy-learner/
docker build -t tars/meta-consensus-optimizer:0.8.0-alpha federation/meta-controller/
docker build -t tars/ethical-trainer:0.8.0-alpha governance/ethical-trainer/

# Verify images
docker images | grep -E "(cognition|policy-learner|meta-optimizer|ethical-trainer)"
```

### Step 3: Deploy Services (Kubernetes)

```bash
# Deploy Phase 10 services via Helm
helm upgrade --install tars ./charts/tars \
    --namespace tars \
    --create-namespace \
    --set cognition.enabled=true \
    --set governance.policyLearner.enabled=true \
    --set federation.metaOptimizer.enabled=true \
    --set governance.ethicalTrainer.enabled=true \
    --wait

# Verify deployment
kubectl get pods -n tars | grep -E "(cognition|policy-learner|meta-optimizer|ethical-trainer)"

# Expected output:
# cognition-insight-engine-xxxxx    1/1  Running  0  2m
# cognition-insight-engine-yyyyy    1/1  Running  0  2m
# policy-learner-xxxxx              1/1  Running  0  2m
# policy-learner-yyyyy              1/1  Running  0  2m
# meta-consensus-optimizer-xxxxx    1/1  Running  0  2m
# ethical-trainer-xxxxx             1/1  Running  0  2m
# ethical-trainer-yyyyy             1/1  Running  0  2m
```

### Step 4: Verify Service Health

```bash
# Check all services are healthy
kubectl get svc -n tars | grep -E "(cognition|policy-learner|meta-optimizer|ethical-trainer)"

# Test health endpoints
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- sh

# Inside debug pod:
curl http://cognition-insight-engine:8090/health
curl http://policy-learner:8091/health
curl http://meta-consensus-optimizer:8092/health
curl http://ethical-trainer:8093/health

# All should return: {"status": "healthy", ...}
```

### Step 5: Import Grafana Dashboard

```bash
# Copy dashboard JSON to Grafana pod
kubectl cp observability/grafana-dashboard-adaptive-governance.json \
    tars/grafana-xxxxx:/tmp/dashboard.json

# Import via Grafana UI or API
curl -X POST http://grafana:3000/api/dashboards/db \
    -H "Content-Type: application/json" \
    -d @observability/grafana-dashboard-adaptive-governance.json

# Open dashboard: http://grafana:3000/d/adaptive-governance
```

### Step 6: Run Cognitive Simulation

```bash
# Run simulation to validate Phase 10
python scripts/cognitive-sim.py \
    --insight-engine http://localhost:8090 \
    --policy-learner http://localhost:8091 \
    --meta-optimizer http://localhost:8092 \
    --ethical-trainer http://localhost:8093 \
    --duration 180 \
    --interval 15

# Expected output:
# ============================================================
# T.A.R.S. Cognitive Federation Simulator
# ============================================================
#
# ✓ Insight Engine is healthy
# ✓ Policy Learner is healthy
# ✓ Meta Optimizer is healthy
# ✓ Ethical Trainer is healthy
#
# Starting simulation (duration: 180s, interval: 15s)
# ...
# 🎉 ALL SUCCESS CRITERIA MET!
```

---

## Testing Individual Components

### Cognitive Analytics Core

```bash
# Manually trigger insight generation
curl -X POST http://localhost:8090/api/v1/insights/trigger

# Get recommendations
curl -X POST http://localhost:8090/api/v1/insights/recommendations \
    -H "Content-Type: application/json" \
    -d '{
        "min_confidence": 0.7,
        "limit": 10
    }'

# Get cognitive state
curl http://localhost:8090/api/v1/state

# Check metrics
curl http://localhost:8090/metrics | grep tars_cognitive
```

### Adaptive Policy Learner

```bash
# Trigger policy adaptation
curl -X POST http://localhost:8091/api/v1/adapt/trigger

# List active proposals
curl http://localhost:8091/api/v1/proposals

# Check metrics
curl http://localhost:8091/metrics | grep tars_policy_adaptation
```

### Meta-Consensus Optimizer

```bash
# Submit consensus state for optimization
curl -X POST http://localhost:8092/api/v1/consensus/optimize \
    -H "Content-Type: application/json" \
    -d '{
        "avg_latency_ms": 420,
        "p95_latency_ms": 580,
        "success_rate": 0.94,
        "quorum_failures": 8,
        "total_votes": 150,
        "algorithm": "raft",
        "current_timeout_ms": 500
    }'

# Get optimizer statistics
curl http://localhost:8092/api/v1/consensus/statistics

# Check metrics
curl http://localhost:8092/metrics | grep tars_consensus
```

### Ethical Fairness Trainer

```bash
# Trigger model training (background task)
curl -X POST http://localhost:8093/api/v1/train \
    -H "Content-Type: application/json" \
    -d '{
        "lookback_days": 30,
        "test_size": 0.2
    }'

# Predict fairness for a decision
curl -X POST http://localhost:8093/api/v1/predict \
    -H "Content-Type: application/json" \
    -d '{
        "fairness_score": 0.72,
        "training_data_distribution": {
            "age": 18.5,
            "gender": 48.2,
            "race": 15.3,
            "disability": 3.1,
            "religion": 8.9
        },
        "outcome_by_group": {
            "age": 0.65,
            "gender": 0.68,
            "race": 0.58,
            "disability": 0.52,
            "religion": 0.63
        },
        "sample_size": 1250
    }'

# Get model statistics
curl http://localhost:8093/api/v1/statistics

# Check metrics
curl http://localhost:8093/metrics | grep tars_fairness
```

---

## Monitoring

### Key Metrics to Watch

**Cognitive Insights**:
- `tars_cognitive_insights_generated_total` - Total insights by type/priority
- `tars_cognitive_recommendation_score` - Current recommendation scores
- `tars_cognitive_insight_processing_seconds` - Processing latency

**Policy Adaptation**:
- `tars_policy_adaptation_success_total` - Successful adaptations
- `tars_policy_adaptation_failures_total` - Failed attempts
- `tars_policy_proposals_active` - Active proposals pending approval

**Consensus Optimization**:
- `tars_consensus_reward` - RL reward signal
- `tars_consensus_timeout_ms` - Current consensus timeout
- `tars_consensus_q_table_size` - Q-learning table size

**Ethical Learning**:
- `tars_fairness_model_accuracy` - Model accuracy
- `tars_fairness_predictions_total` - Predictions by fair/unfair
- `tars_fairness_training_runs_total` - Training runs by status

### Grafana Queries

```promql
# Insight generation rate
rate(tars_cognitive_insights_generated_total[5m])

# Policy adaptation success rate
sum(rate(tars_policy_adaptation_success_total[5m])) /
(sum(rate(tars_policy_adaptation_success_total[5m])) +
 sum(rate(tars_policy_adaptation_failures_total[5m])))

# Consensus reward trend
deriv(tars_consensus_reward{component="total"}[10m])

# Ethical model accuracy
tars_fairness_model_accuracy
```

---

## Troubleshooting

### Issue: No Insights Generated

**Symptoms**: `tars_cognitive_insights_generated_total` remains at 0

**Diagnosis**:
```bash
# Check if policy_audit table has recent data
kubectl exec -it postgres-0 -n tars -- psql -U tars -d tars -c \
    "SELECT COUNT(*) FROM policy_audit WHERE timestamp >= NOW() - INTERVAL '1 hour';"

# If count is 0, policies are not being evaluated
```

**Solution**:
- Ensure Phase 9 governance engine is operational
- Verify policies are being actively evaluated
- Manually create test audit entries for development

### Issue: Policy Validation Failing

**Symptoms**: `tars_policy_adaptation_failures_total{reason="opa_validation_failed"}` increasing

**Diagnosis**:
```bash
# Check OPA is accessible
curl http://opa:8181/health

# Check OPA logs
kubectl logs -n tars -l app=opa --tail=50
```

**Solution**:
- Verify OPA_URL environment variable is correct
- Check OPA has required policies loaded
- Review generated Rego patches for syntax errors

### Issue: RL Agent Not Learning

**Symptoms**: `tars_consensus_q_table_size` remains at 0 or very small

**Diagnosis**:
```bash
# Check meta-optimizer logs
kubectl logs -n tars -l app=meta-consensus-optimizer --tail=100

# Check if agent pickle exists
kubectl exec -it meta-consensus-optimizer-xxxxx -n tars -- ls -lh /data/
```

**Solution**:
- Ensure consensus metrics are being fed to optimizer
- Verify PVC is correctly mounted
- Check for agent save errors in logs
- Manually trigger optimization to bootstrap learning

### Issue: High Latency

**Symptoms**: Insight processing > 5 seconds

**Diagnosis**:
```bash
# Check database query performance
kubectl exec -it postgres-0 -n tars -- psql -U tars -d tars -c \
    "EXPLAIN ANALYZE SELECT * FROM policy_audit WHERE timestamp >= NOW() - INTERVAL '1 hour';"

# Check CPU/memory usage
kubectl top pods -n tars | grep cognition
```

**Solution**:
- Add database indexes on `timestamp` column
- Reduce `ANALYSIS_WINDOW_MINUTES` if data volume is high
- Scale up Insight Engine replicas

---

## Configuration Tuning

### Performance Optimization

```yaml
# values.yaml adjustments

# For high-throughput environments:
cognition:
  insightRefreshInterval: 30  # Faster refresh
  analysisWindowMinutes: 30   # Smaller window

# For large deployments:
cognition:
  insightEngine:
    replicas: 4               # More replicas
    resources:
      requests:
        cpu: 1000m
        memory: 2Gi
```

### Conservative Settings (Production)

```yaml
# Start with conservative thresholds
governance:
  policyLearner:
    minConfidence: 0.85       # Higher confidence required
    approvalThreshold: 0.90   # Higher approval threshold
    dryRunEnabled: true       # Always validate first

federation:
  metaOptimizer:
    learningRate: 0.05        # Slower learning
    explorationRate: 0.1      # Less exploration
```

---

## Next Steps

1. **Monitor for 48 Hours**:
   - Watch Grafana dashboard
   - Review generated insights
   - Verify policy adaptations are sensible

2. **Tune Thresholds**:
   - Adjust based on observed behavior
   - Increase confidence thresholds if too many false positives
   - Decrease if too conservative

3. **Enable Advanced Features**:
   - Federated learning across clusters
   - Cross-cluster insight sharing
   - Advanced RL algorithms (DQN)

4. **Integrate with CI/CD**:
   - Automated policy testing
   - A/B testing for policy changes
   - Canary deployments for critical policies

---

## Additional Resources

- [Full Implementation Report](PHASE10_IMPLEMENTATION_REPORT.md)
- [Cognitive Insight Engine README](cognition/insight-engine/README.md)
- [Policy Learner Documentation](governance/policy-learner/README.md)
- [Meta-Optimizer Guide](federation/meta-controller/README.md)
- [Ethical Trainer Documentation](governance/ethical-trainer/README.md)

---

**Questions?** Check the troubleshooting section or review service logs.

**Success?** Proceed to Phase 11 (coming soon) for advanced cognitive orchestration.

🎉 **Welcome to Adaptive Governance!**

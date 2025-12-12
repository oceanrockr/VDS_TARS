# Phase 14.3 Quick Start Guide - Production Deployment & GA Release

**Version:** v1.0.1
**Last Updated:** 2025-11-20
**Purpose:** Quick reference for executing production deployment and final GA release

---

## 🚀 Quick Start (Production Deployment)

### Prerequisites

- ✅ Staging deployment completed and validated
- ✅ All stakeholder sign-offs obtained
- ✅ Production deployment window scheduled
- ✅ Rollback plan reviewed
- ✅ On-call team notified

### Execute Production Deployment

**Option 1: GitHub Actions UI (Recommended)**
```
1. Go to: https://github.com/YOUR_ORG/tars/actions
2. Select workflow: "T.A.R.S. v1.0.1 Production Deployment"
3. Click "Run workflow"
4. Branch: release/v1.0.1
5. Options:
   - deployment_strategy: canary (recommended)
   - canary_stages: 1,10,25,50,100
   - stage_duration_minutes: 10
   - enable_feature_flags: true
   - notify_pagerduty: true
6. Click "Run workflow"
7. WAIT for manual approval (Release Manager + SRE)
```

**Option 2: GitHub CLI**
```bash
gh workflow run production_deploy_pipeline.yaml \
  --ref release/v1.0.1 \
  --field deployment_strategy=canary \
  --field canary_stages="1,10,25,50,100" \
  --field stage_duration_minutes=10 \
  --field enable_feature_flags=true \
  --field notify_pagerduty=true
```

---

## 📊 Monitor Production Deployment

### Watch Pipeline Status

```bash
# List recent runs
gh run list --workflow=production_deploy_pipeline.yaml --limit 5

# Watch specific run
gh run watch <run-id>

# View logs
gh run view <run-id> --log
```

### Check Deployment Status

```bash
# Kubernetes deployment status
kubectl get deployments -n tars-production

# Pod status
kubectl get pods -n tars-production -o wide

# Canary status (if enabled)
kubectl get pods -n tars-production -l deployment=canary

# Helm release status
helm list -n tars-production

# View recent events
kubectl get events -n tars-production --sort-by='.lastTimestamp'
```

---

## ✅ Production Validation Checklist

### 1. Pipeline Status
- [ ] All 9 stages completed successfully
- [ ] Manual approvals obtained (Release Manager + SRE)
- [ ] No rollback triggered
- [ ] PagerDuty notifications sent

### 2. Deployment Health
- [ ] All pods in Running state (no CrashLoopBackOff)
- [ ] All deployments have desired replicas
- [ ] Canary deployment successful (if enabled)
- [ ] Database migrations applied

### 3. Service Health
```bash
# Check all service endpoints
kubectl exec -n tars-production deploy/tars-orchestration-agent -- \
  curl -f http://tars-insight-engine:8090/health

# Check dashboard
curl -f https://tars.ai

# Check API
curl -f https://api.tars.ai/health
```

### 4. SLO Verification
```bash
# Run production validation suite
pytest release/v1_0_1/production_validation_suite.py \
  --environment=production \
  --namespace=tars-production \
  -v

# Check SLO compliance
# - Availability: ≥ 99.9%
# - API p95 latency: < 100ms
# - Error rate: < 1%
```

### 5. Monitoring & Alerting
- [ ] Prometheus recording rules active
- [ ] Grafana dashboards loading < 5s
- [ ] Jaeger traces visible (100% coverage)
- [ ] PagerDuty integration active
- [ ] Alert rules loaded (50+ rules)

---

## 📈 Performance Validation

### Quick Performance Checks

```bash
# 1. Dashboard load time
time curl -s https://tars.ai > /dev/null
# Expected: <2s

# 2. API latency
kubectl exec -n tars-production deploy/tars-orchestration-agent -- \
  curl -w "@curl-format.txt" -o /dev/null -s http://tars-dashboard-api:3001/api/v1/agents
# Expected p95: <100ms

# 3. Grafana dashboard performance
time curl -s https://tars.ai/grafana/d/tars-evaluation > /dev/null
# Expected: <5s

# 4. Database query performance
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -c "\timing on" -c "SELECT * FROM missions WHERE status='active' ORDER BY created_at DESC LIMIT 100;"
# Expected: <50ms

# 5. PPO agent memory
kubectl top pod -n tars-production -l agent_type=ppo
# Expected: <2GB per pod
```

---

## 🔄 Rollback Procedure (If Needed)

### Automatic Rollback

The pipeline automatically rolls back if validation fails. No action required.

### Manual Rollback

```bash
# 1. Execute rollback
helm rollback tars -n tars-production

# 2. Wait for rollback completion
kubectl rollout status deployment -n tars-production --timeout=3m

# 3. Verify rollback
kubectl get pods -n tars-production
helm history tars -n tars-production

# 4. Check service health
kubectl get pods -n tars-production
curl -f https://tars.ai/health

# 5. Notify stakeholders
echo "Production rollback completed at $(date)" | \
  mail -s "T.A.R.S. v1.0.1 Rollback Notification" team@tars.ai
```

---

## 📥 Download Production Artifacts

### Pipeline Artifacts

```bash
# Download all artifacts from latest run
gh run download <run-id>

# Specific artifacts
gh run download <run-id> -n tars-v1.0.1-production-artifacts
gh run download <run-id> -n production-release-report
gh run download <run-id> -n production-validation-results
```

### Artifact Contents

```
artifacts/
├── tars-v1.0.1-production-artifacts/
│   ├── tars-1.0.1.tgz (Helm chart)
│   ├── manifest.json (build metadata)
│   ├── manifest.production.json (production-specific)
│   └── SHA256SUMS (checksums)
├── production-release-report/
│   └── PRODUCTION_RELEASE_REPORT_<timestamp>.md
└── production-validation-results/
    ├── production_validation_report.html
    └── production_validation_junit.xml
```

---

## 📝 Post-Deployment Actions

### Immediate (0-2 hours)

1. **Monitor Dashboards**
   - Grafana: https://tars.ai/grafana
   - Prometheus: https://tars.ai/prometheus
   - Jaeger: https://tars.ai/jaeger

2. **Check Alert Status**
   - PagerDuty: No active incidents
   - Slack: #tars-deployments channel

3. **Verify Customer Experience**
   - Test key user workflows
   - Check support tickets for issues
   - Monitor social media/forums

### Short-Term (2-24 hours)

1. **Performance Analysis**
   - API latency trends
   - Error rate analysis
   - Resource utilization

2. **Cost Analysis**
   - Cloud costs vs. baseline
   - Resource optimization opportunities

3. **Customer Feedback**
   - Support ticket analysis
   - User sentiment tracking

### Long-Term (1-7 days)

1. **Stability Monitoring**
   - 7-day performance trends
   - SLO compliance report
   - Capacity planning

2. **Post-Deployment Review**
   - What went well
   - What could be improved
   - Action items for next release

3. **Documentation Updates**
   - Update runbooks
   - Capture lessons learned
   - Update monitoring dashboards

---

## 📧 Communication

### Deployment Notification Template

```
Subject: T.A.R.S. v1.0.1 Production Deployment Complete

Team,

T.A.R.S. v1.0.1 has been successfully deployed to production.

**Deployment Summary:**
- Version: v1.0.1
- Environment: Production
- Strategy: Canary (1→10→25→50→100)
- Duration: [X] minutes
- Status: ✅ SUCCESS

**Key Improvements:**
- 🔌 WebSocket reconnection: 99.9% uptime
- 📊 Grafana dashboards: 5x faster (< 3s)
- 🔍 Distributed tracing: 100% coverage
- ⚡ Database queries: 10x faster
- 🧠 PPO memory: 60% reduction

**Validation Results:**
- Production validation: 200+ tests passed (100%)
- SLO compliance: All targets met
- Zero downtime deployment: ✅

**Monitoring:**
- Grafana: https://tars.ai/grafana
- Prometheus: https://tars.ai/prometheus
- Status Page: https://status.tars.ai

Please report any issues to #tars-support or PagerDuty.

**Next Steps:**
- 24-hour monitoring active
- Post-deployment review scheduled for [DATE]

Thanks,
T.A.R.S. Release Team
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Manual approval timeout**
```bash
# Check approval status
gh run view <run-id>

# Approve manually (if needed)
# Via GitHub UI: Actions → Run → Review required → Approve
```

**Issue 2: Canary stage failing**
```bash
# Check canary pod logs
kubectl logs -n tars-production -l deployment=canary --tail=100

# Check canary metrics
kubectl exec -n tars-production prometheus-0 -- \
  wget -qO- "http://localhost:9090/api/v1/query?query=rate(http_requests_total{deployment=\"canary\",status=~\"5..\"}[5m])"

# Rollback if needed
helm rollback tars -n tars-production
```

**Issue 3: Database migration fails**
```bash
# Check migration job logs
kubectl logs -n tars-production -l job-name=tars-db-migrate-v1-0-1-pre

# Manually apply migration (if safe)
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -f /migrations/v1_0_1_add_indexes.sql

# Verify indexes
kubectl exec -n tars-production deploy/tars-postgres -- \
  psql -U tars -d tars -c "SELECT indexname FROM pg_indexes WHERE schemaname='public' AND indexname LIKE 'idx_%';"
```

**Issue 4: SLO violations**
```bash
# Check which SLOs are violated
kubectl exec -n tars-production prometheus-0 -- \
  wget -qO- "http://localhost:9090/api/v1/query?query=ALERTS{alertstate=\"firing\"}"

# Common fixes:
# - High error rate: Check service logs
# - High latency: Check resource utilization
# - Low availability: Check pod status
```

**Issue 5: PagerDuty not notifying**
```bash
# Check PagerDuty integration
kubectl get configmap alertmanager-config -n tars-production -o yaml

# Test PagerDuty manually
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "YOUR_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Test alert",
      "severity": "info",
      "source": "manual-test"
    }
  }'
```

---

## 📊 Success Criteria

Production deployment is successful when:

- ✅ All 9 pipeline stages complete without errors
- ✅ All pods are Running with 0 restarts
- ✅ Production validation passes 100% (200+ tests)
- ✅ SLOs met (availability ≥99.9%, latency <100ms, error rate <1%)
- ✅ All 5 hotfixes validated in production
- ✅ Zero-downtime deployment achieved
- ✅ Monitoring and alerting functional
- ✅ No customer-reported issues in first 24 hours

---

## 📞 Get Help

### Resources

- **Documentation:** [release/v1_0_1/](release/v1_0_1/)
- **Production Pipeline:** [production_deploy_pipeline.yaml](release/v1_0_1/production_deploy_pipeline.yaml)
- **Validation Suite:** [production_validation_suite.py](release/v1_0_1/production_validation_suite.py)
- **Release Notes:** [PRODUCTION_RELEASE_NOTES.md](release/v1_0_1/PRODUCTION_RELEASE_NOTES.md)
- **Monitoring Plan:** [production_monitoring_plan.md](release/v1_0_1/production_monitoring_plan.md)

### Support Channels

- **PagerDuty:** Immediate (P0/P1 incidents)
- **Slack:** #tars-deployments (general), #tars-support (issues)
- **Email:** release-manager@tars.ai
- **Status Page:** https://status.tars.ai

---

## 🎯 Post-GA Actions

After successful production deployment:

### Week 1

1. **Daily Monitoring**
   - Review 24-hour metrics
   - Check for anomalies
   - Address any issues promptly

2. **Customer Communication**
   - Publish release notes
   - Update documentation
   - Respond to feedback

3. **Performance Analysis**
   - Compare staging vs. production metrics
   - Identify optimization opportunities
   - Update capacity plans

### Week 2-4

1. **Stability Assessment**
   - 30-day SLO compliance report
   - Incident analysis
   - Cost optimization review

2. **Post-Deployment Review**
   - Team retrospective
   - Document lessons learned
   - Update processes

3. **Future Planning**
   - Plan v1.1.0 features
   - Address technical debt
   - Capacity planning

---

## 📋 Commands Reference

### Deployment Commands

```bash
# Trigger production deployment
gh workflow run production_deploy_pipeline.yaml --ref release/v1.0.1

# Monitor deployment
gh run watch <run-id>

# Check status
kubectl get all -n tars-production
```

### Validation Commands

```bash
# Run production validation suite
pytest release/v1_0_1/production_validation_suite.py \
  --environment=production \
  --namespace=tars-production \
  -v

# Run specific test category
pytest release/v1_0_1/production_validation_suite.py::TestAPIEndpoints -v

# Generate HTML report
pytest release/v1_0_1/production_validation_suite.py \
  --html=production_report.html \
  --self-contained-html
```

### Monitoring Commands

```bash
# Check SLOs
kubectl exec -n tars-production prometheus-0 -- \
  wget -qO- "http://localhost:9090/api/v1/query?query=avg_over_time(up{job=\"tars\"}[10m])*100"

# View active alerts
kubectl exec -n tars-production prometheus-0 -- \
  wget -qO- "http://localhost:9090/api/v1/query?query=ALERTS{alertstate=\"firing\"}"

# Check Grafana dashboards
curl -f https://tars.ai/grafana/d/tars-overview
```

### Rollback Commands

```bash
# Helm rollback
helm rollback tars -n tars-production

# Verify rollback
kubectl rollout status deployment -n tars-production

# Check history
helm history tars -n tars-production
```

---

**Quick Start Guide Version:** 1.0
**Last Updated:** 2025-11-20
**Status:** Production Ready

🚀 Generated with [Claude Code](https://claude.com/claude-code)

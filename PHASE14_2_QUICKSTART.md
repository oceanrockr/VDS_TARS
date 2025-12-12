# Phase 14.2 Quick Start Guide - Staging Deployment Validation

**Version:** v1.0.1
**Last Updated:** 2025-11-20
**Purpose:** Quick reference for executing staging deployment and validation

---

## 🚀 Quick Start (5 Minutes)

### Execute Staging Deployment

**Option 1: GitHub Actions UI (Recommended)**
```
1. Go to: https://github.com/YOUR_ORG/tars/actions
2. Select workflow: "T.A.R.S. v1.0.1 Staging Deployment"
3. Click "Run workflow"
4. Branch: release/v1.0.1
5. Options:
   - skip_tests: false
   - force_deploy: false
   - enable_canary: false (recommended for first deploy)
6. Click "Run workflow"
```

**Option 2: GitHub CLI**
```bash
gh workflow run staging_deploy_pipeline.yaml \
  --ref release/v1.0.1 \
  --field skip_tests=false \
  --field force_deploy=false \
  --field enable_canary=false
```

**Option 3: Push to Release Branch**
```bash
git push origin release/v1.0.1
```

---

## 📊 Monitor Deployment Progress

### Watch Pipeline Status

```bash
# List recent runs
gh run list --workflow=staging_deploy_pipeline.yaml --limit 5

# Watch specific run
gh run watch <run-id>

# View logs
gh run view <run-id> --log
```

### Check Deployment Status

```bash
# Kubernetes deployment status
kubectl get deployments -n tars-staging

# Pod status
kubectl get pods -n tars-staging -o wide

# Helm release status
helm list -n tars-staging

# View recent events
kubectl get events -n tars-staging --sort-by='.lastTimestamp'
```

---

## ✅ Validation Checklist

After pipeline completes, verify:

### 1. Pipeline Status
- [ ] All 8 stages completed successfully
- [ ] No rollback triggered
- [ ] Slack notification received (if configured)

### 2. Deployment Health
- [ ] All pods in Running state
- [ ] All deployments have desired replicas
- [ ] No CrashLoopBackOff or Error pods

### 3. Service Health
```bash
# Check all service endpoints
kubectl exec -n tars-staging deploy/tars-orchestration-agent -- \
  curl -f http://tars-insight-engine:8090/health

# Check dashboard
curl -f https://staging.tars.ai
```

### 4. Test Results
- [ ] Regression suite: 100% pass rate
- [ ] Staging validation: 100% pass rate
- [ ] Performance benchmarks meet targets

### 5. Monitoring
- [ ] Prometheus recording rules active
- [ ] Grafana dashboards loading <5s
- [ ] Jaeger traces visible

---

## 📈 Performance Validation

### Quick Performance Checks

```bash
# 1. Dashboard load time
time curl -s https://staging.tars.ai > /dev/null
# Expected: <2s

# 2. API latency
kubectl exec -n tars-staging deploy/tars-orchestration-agent -- \
  curl -w "@curl-format.txt" -o /dev/null -s http://tars-dashboard-api:3001/api/v1/agents
# Expected p95: <150ms

# 3. Query execution time
# Open Grafana -> T.A.R.S. Evaluation Dashboard
# Expected load time: <5s
```

---

## 🔄 Rollback Procedure (If Needed)

### Automatic Rollback

The pipeline automatically rolls back if tests fail. No action required.

### Manual Rollback

```bash
# Rollback Helm release
helm rollback tars -n tars-staging

# Verify rollback
kubectl rollout status deployment -n tars-staging --timeout=5m

# Check service health
kubectl get pods -n tars-staging
```

---

## 📥 Download Artifacts

### Pipeline Artifacts

```bash
# Download all artifacts from latest run
gh run download <run-id>

# Specific artifacts
gh run download <run-id> -n tars-v1.0.1-artifacts
gh run download <run-id> -n staging-release-report
gh run download <run-id> -n regression-test-results
gh run download <run-id> -n performance-benchmarks
```

### Artifact Contents

```
artifacts/
├── tars-v1.0.1-artifacts/
│   ├── tars-1.0.1.tgz (Helm chart)
│   ├── manifest.json (artifact metadata)
│   └── SHA256SUMS (checksums)
├── staging-release-report/
│   └── STAGING_RELEASE_REPORT_<timestamp>.md
├── regression-test-results/
│   ├── junit.xml
│   └── report.html
└── performance-benchmarks/
    ├── api_load_test.csv
    ├── websocket_stress_test.json
    ├── grafana_load_test.json
    └── db_benchmark.json
```

---

## 📝 Review Release Report

### Generate/View Report

```bash
# View staging release report
cat artifacts/staging-release-report/STAGING_RELEASE_REPORT_*.md

# Or download from GitHub Actions UI:
# Actions -> Run -> Artifacts -> staging-release-report
```

### Key Sections to Review

1. **Executive Summary** - Overall status
2. **Hotfix Validation Results** - All 5 fixes validated
3. **Regression Test Results** - Pass rates
4. **Performance Benchmarks** - Comparison to baseline
5. **Sign-Off Checklist** - Stakeholder approval

---

## ✍️ Sign-Off Process

### Stakeholder Sign-Off

After reviewing the staging release report:

1. **Engineering Lead** - Code quality and implementation
2. **QA Lead** - Test coverage and results
3. **SRE Lead** - Deployment and monitoring
4. **Security Lead** - Security validation
5. **Release Manager** - Overall release readiness

### Sign-Off Procedure

```bash
# Edit staging release report
vim release/v1_0_1/STAGING_RELEASE_REPORT_<timestamp>.md

# Update sign-off checklist
- [x] **Engineering Lead** - Approved
- [x] **QA Lead** - Approved
- [x] **SRE Lead** - Approved
- [x] **Security Lead** - Approved
- [x] **Release Manager** - Approved

# Commit sign-offs
git add release/v1_0_1/STAGING_RELEASE_REPORT_*.md
git commit -m "chore: Phase 14.2 sign-offs for v1.0.1 staging deployment"
git push origin release/v1.0.1
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Pipeline fails at "Build & Package" stage**
```bash
# Check Docker build logs
gh run view <run-id> --log | grep "docker build"

# Fix: Ensure all Dockerfiles exist and are valid
ls -la cognition/*/Dockerfile dashboard/*/Dockerfile
```

**Issue 2: Tests fail in staging**
```bash
# Download test results
gh run download <run-id> -n regression-test-results

# View failed tests
cat regression-test-results/junit.xml | grep "failure"

# Fix: Investigate specific test failures
```

**Issue 3: Pods not starting**
```bash
# Check pod events
kubectl describe pod <pod-name> -n tars-staging

# Check logs
kubectl logs <pod-name> -n tars-staging --previous

# Common fixes:
# - Check image pull secrets
# - Verify resource limits
# - Check ConfigMap/Secret mounts
```

**Issue 4: Database migration fails**
```bash
# Check migration logs
kubectl logs -n tars-staging -l job-name=tars-db-migrate-v1-0-1-pre

# Manually apply migration
kubectl exec -n tars-staging deploy/tars-postgres -- \
  psql -U tars -d tars -f /migrations/v1_0_1_add_indexes.sql
```

**Issue 5: Prometheus recording rules not loading**
```bash
# Check Prometheus config
kubectl exec -n tars-staging prometheus-0 -- \
  promtool check config /etc/prometheus/prometheus.yml

# Reload Prometheus
kubectl exec -n tars-staging prometheus-0 -- kill -HUP 1

# Verify rules loaded
kubectl exec -n tars-staging prometheus-0 -- \
  promtool check rules /etc/prometheus/recording_rules.yaml
```

---

## 🔍 Manual Validation

If you want to manually validate without running the full suite:

### Quick Manual Validation

```bash
# 1. Check WebSocket reconnection (TARS-1001)
python fixes/fix_websocket_reconnect/websocket_reconnect_test.py

# 2. Check Grafana dashboard load time (TARS-1002)
time curl -s https://staging.tars.ai/grafana/d/tars-evaluation

# 3. Check Jaeger traces (TARS-1003)
curl http://jaeger-query.tars-staging.svc.cluster.local:16686/api/traces?service=tars-orchestration-agent

# 4. Check database query performance (TARS-1004)
python fixes/fix_database_indexes/db_index_tests.py

# 5. Check PPO memory usage (TARS-1005)
kubectl top pod -n tars-staging -l agent_type=ppo
```

---

## 📞 Get Help

### Resources

- **Documentation:** [PHASE14_2_SESSION1_SUMMARY.md](PHASE14_2_SESSION1_SUMMARY.md)
- **Staging Pipeline:** [staging_deploy_pipeline.yaml](release/v1_0_1/staging_deploy_pipeline.yaml)
- **Validation Suite:** [staging_validation_suite.py](release/v1_0_1/staging_validation_suite.py)
- **Upgrade Playbook:** [upgrade_playbook.md](release/v1_0_1/upgrade_playbook.md)

### Support Channels

- **Slack:** #tars-deployments
- **Issues:** https://github.com/YOUR_ORG/tars/issues
- **Runbooks:** docs/runbooks/

---

## 🎯 Next Steps (Phase 14.3)

After successful staging deployment and sign-off:

1. **Production Deployment Planning**
   - Review production deployment strategy (canary vs rolling)
   - Schedule deployment window
   - Notify stakeholders

2. **Production Pipeline Creation**
   - Adapt staging pipeline for production
   - Add additional safeguards
   - Configure production secrets

3. **Production Validation**
   - Adapt staging validation for production
   - Add production-specific tests
   - Configure monitoring alerts

4. **Release Communication**
   - Finalize customer-facing release notes
   - Prepare maintenance window notification
   - Update documentation

---

## 📋 Commands Reference

### Pipeline Commands

```bash
# Trigger staging deployment
gh workflow run staging_deploy_pipeline.yaml --ref release/v1.0.1

# Monitor pipeline
gh run list --workflow=staging_deploy_pipeline.yaml
gh run watch <run-id>
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

### Kubernetes Commands

```bash
# Deployment status
kubectl get deployments -n tars-staging
kubectl get pods -n tars-staging
kubectl get services -n tars-staging
kubectl get ingress -n tars-staging

# Logs
kubectl logs -f -n tars-staging deploy/tars-orchestration-agent
kubectl logs -f -n tars-staging deploy/tars-dashboard-api

# Events
kubectl get events -n tars-staging --sort-by='.lastTimestamp'

# Resources
kubectl top pods -n tars-staging
kubectl top nodes
```

### Helm Commands

```bash
# Release status
helm list -n tars-staging
helm status tars -n tars-staging
helm history tars -n tars-staging

# Rollback
helm rollback tars -n tars-staging
helm rollback tars <revision> -n tars-staging
```

### Testing Commands

```bash
# Run regression suite
pytest release/v1_0_1/regression_suite_v1_0_1.py \
  --environment=staging \
  --namespace=tars-staging \
  -v

# Run staging validation
pytest release/v1_0_1/staging_validation_suite.py \
  --namespace=tars-staging \
  -v

# Run specific test category
pytest release/v1_0_1/staging_validation_suite.py::TestKubernetesDeployment -v
```

---

## 📊 Success Criteria

Staging deployment is successful when:

- ✅ All 8 pipeline stages complete without errors
- ✅ All pods are Running with 0 restarts
- ✅ Regression suite passes 100% (260+ tests)
- ✅ Staging validation passes 100% (150+ tests)
- ✅ Performance benchmarks meet targets
- ✅ All 5 hotfixes validated
- ✅ Zero-downtime deployment achieved
- ✅ Monitoring and alerting functional
- ✅ Stakeholder sign-offs obtained
- ✅ Staging release report generated

---

**Quick Start Guide Version:** 1.0
**Last Updated:** 2025-11-20
**Status:** Production Ready

🚀 Generated with [Claude Code](https://claude.com/claude-code)

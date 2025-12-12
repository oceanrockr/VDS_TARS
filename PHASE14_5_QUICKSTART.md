# Phase 14.5 Quick Start Guide: GA Day Automation

**Version:** T.A.R.S. v1.0.1
**Phase:** 14.5 - GA Day Automation Completion
**Est. Execution Time:** 25 hours (24h monitoring + 1h certification)

---

## Prerequisites

✅ Phase 14.4 components installed and verified
✅ Production environment deployed with v1.0.1
✅ Prometheus accessible at `http://prometheus.tars-production.svc.cluster.local:9090`
✅ Python 3.11+ with dependencies from `requirements-dev.txt`
✅ (Optional) Pandoc + XeLaTeX for PDF generation

---

## Quick Start: Automated Pipeline (Recommended)

### Option 1: GitHub Actions Workflow

```bash
# Trigger production deployment with GA mode
gh workflow run production-deploy.yml \
  -f deployment_strategy=canary \
  -f ga_mode=true \
  -f notify_pagerduty=true \
  -f enable_feature_flags=true

# Monitor workflow progress
gh run watch

# Download artifacts after completion (25 hours later)
gh run download --name ga-certification-package
```

**What Happens:**
1. Deploys v1.0.1 to production using canary strategy
2. Launches 24-hour GA Day monitoring (KPI collector, drift detector, SLO monitor, WebSocket validator)
3. Runs GA-specific validation tests
4. Generates certification package
5. Publishes to GitHub Release
6. Sends Slack/StatusPage/PagerDuty notifications

**Artifacts Generated:**
- `ga_certification_package.tar.gz` (complete certification bundle)
- `ga-kpi-snapshots/` (288 KPI snapshots @ 5min intervals)
- `ga-monitoring-logs/` (KPI, SLO, drift detector logs)

---

## Manual Execution (For Testing/Development)

### Step 1: Prepare Baseline

```bash
# Create baseline snapshot from staging/pre-production
cd /path/to/tars

# Run 1-hour baseline capture
python observability/ga_kpi_collector.py \
  --duration 1 \
  --interval 60 \
  --output baseline_capture \
  --prometheus-url http://prometheus.tars-staging.svc.cluster.local:9090

# Verify baseline created
ls -lh baseline_capture/baseline_metrics.json
cat baseline_capture/baseline_metrics.json
```

**Expected Output:**
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 62.5,
  "api_latency_p95": 95.3,
  "cluster_cpu_utilization": 38.7,
  "cluster_memory_utilization": 54.2,
  "db_latency_p95": 12.5,
  "redis_hit_rate": 97.8,
  "error_rate": 0.05,
  "timestamp": "2025-01-14T18:00:00Z"
}
```

### Step 2: Start GA Day Monitoring (24 Hours)

```bash
# Terminal 1: KPI Collector
python observability/ga_kpi_collector.py \
  --duration 24 \
  --interval 5 \
  --output ga_kpis \
  --prometheus-url http://prometheus.tars-production.svc.cluster.local:9090

# Terminal 2: Drift Detector
python observability/drift_detector.py \
  --baseline-file baseline_capture/baseline_metrics.json \
  --duration 24 \
  --check-interval 300 \
  --output drift_analysis.json \
  --prometheus-url http://prometheus.tars-production.svc.cluster.local:9090

# Terminal 3: Real-time SLO Monitor
python observability/monitor_realtime_slos.py \
  --duration 86400 \
  --interval 60 \
  --output slo_monitor \
  --prometheus-url http://prometheus.tars-production.svc.cluster.local:9090

# Wait 24 hours for collection to complete...
```

**Monitoring During Collection:**
```bash
# Check progress
tail -f ga_kpis/*.log

# View latest KPI snapshot
jq . ga_kpis/snapshot_$(ls ga_kpis/snapshot_*.json | tail -n1 | sed 's/.*snapshot_//' | sed 's/.json//).json

# Check drift alerts
jq '.metrics[] | select(.status == "critical")' observability/drift_check_*.json
```

### Step 3: WebSocket Health Validation (1 Hour)

```bash
# Run during GA window (anytime within 24 hours)
python observability/monitor_websocket_health.py \
  --endpoint wss://api.tars.ai/ws \
  --duration 3600 \
  --interval 300 \
  --output ws_health_metrics.json \
  --token $TARS_API_TOKEN
```

**Expected Output:**
```json
{
  "duration_seconds": 3600,
  "total_reconnection_attempts": 12,
  "successful_reconnections": 12,
  "reconnection_success_rate": 100.0,
  "avg_reconnection_latency_ms": 1250.0,
  "p99_reconnection_latency_ms": 2800.0,
  "max_downtime_seconds": 3.5,
  "tars_1001_compliant": true,
  "tars_1001_notes": [
    "All reconnection attempts succeeded",
    "P99 latency <30s (target: <30s)",
    "Max downtime <60s (target: <60s)"
  ]
}
```

### Step 4: Run GA Validation Tests

```bash
# After 6+ hours of GA window (to accumulate metrics)
pytest release/v1_0_1/production_validation_suite.py \
  --ga-mode \
  --namespace tars-production \
  --version v1.0.1 \
  -v \
  --tb=short \
  --maxfail=5 \
  --html=ga_validation_results.html \
  --self-contained-html

# View results
open ga_validation_results.html  # macOS
xdg-open ga_validation_results.html  # Linux
start ga_validation_results.html  # Windows
```

**Test Categories:**
- Cross-region replication consistency
- KPI threshold assertions (availability, error rate, latency)
- Database performance (<50ms with indexes)
- Grafana dashboard availability (<2s load time)
- Agent memory stability (<1.5GB PPO)
- WebSocket uptime (>99.9%)
- Drift stability (≤10% variance)

### Step 5: Generate Certification Package

```bash
# After 24-hour GA window completes
python scripts/generate_ga_certification_package.py \
  --ga-start "2025-01-15T00:00:00Z" \
  --ga-end "2025-01-16T00:00:00Z" \
  --output-dir ./ga_certification \
  --include-artifacts

# Verify package created
ls -lh ga_certification/ga_certification_package.tar.gz
sha256sum ga_certification/ga_certification_package.tar.gz

# Extract and review
cd ga_certification
tar -tzf ga_certification_package.tar.gz  # List contents
tar -xzf ga_certification_package.tar.gz  # Extract

# View GA Day Report
cat package/GA_DAY_REPORT.md
open package/GA_DAY_REPORT.pdf  # If PDF generated
```

---

## Verification Checklist

After GA Day monitoring completes, verify:

### KPI Collection
- [ ] 288 snapshot files created (24h * 12 snapshots/hour)
- [ ] `baseline_metrics.json` created with drift baseline
- [ ] `ga_kpi_summary.json` shows `certification_status: "PASS"`
- [ ] `ga_kpi_summary.json.sha256` hash matches file

### Drift Detection
- [ ] 288 drift check files created (24h * 12 checks/hour @ 5min intervals)
- [ ] `drift_analysis.json` shows `critical_drifts: 0` (or acceptable count)
- [ ] Top drifts are within acceptable ranges (<10%)
- [ ] Mitigation actions are reasonable

### WebSocket Health
- [ ] `ws_health_metrics.json` shows `tars_1001_compliant: true`
- [ ] Reconnection success rate ≥95%
- [ ] Max downtime <60s

### Validation Tests
- [ ] All GA-specific tests passed (or documented failures)
- [ ] No critical test failures
- [ ] HTML report generated with detailed results

### Certification Package
- [ ] `ga_certification_package.tar.gz` created
- [ ] `GA_DAY_REPORT.md` fully populated (no `{{PLACEHOLDERS}}` remaining)
- [ ] `GA_DAY_REPORT.pdf` generated (or placeholder exists)
- [ ] `MANIFEST.sha256` contains all file hashes
- [ ] `certification_metadata.json` shows `certification_status: "✅ CERTIFIED"`

---

## Troubleshooting

### Issue: KPI Collector Returns Empty Metrics

**Symptoms:**
```json
{
  "overall_availability": 0.0,
  "total_requests": 0,
  "error_rate": 0.0
}
```

**Solution:**
1. Verify Prometheus is accessible:
   ```bash
   curl http://prometheus.tars-production.svc.cluster.local:9090/api/v1/query?query=up
   ```
2. Check Prometheus targets are being scraped:
   ```bash
   kubectl exec -n tars-production prometheus-0 -- \
     wget -qO- http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health == "down")'
   ```
3. Verify service label selectors match:
   ```bash
   kubectl get pods -n tars-production -l app.kubernetes.io/name=tars --show-labels
   ```

### Issue: Drift Detector Shows All Metrics as Critical

**Symptoms:**
```json
{
  "critical_drifts": 1152,
  "warning_drifts": 0
}
```

**Solution:**
1. Verify baseline was captured from appropriate environment (staging, not dev)
2. Check baseline timestamp is recent (within 7 days):
   ```bash
   jq '.timestamp' baseline_metrics.json
   ```
3. Lower critical threshold if baseline is from different load profile:
   ```bash
   python observability/drift_detector.py --critical-threshold 20.0  # 20% instead of 10%
   ```

### Issue: PDF Generation Fails

**Symptoms:**
```
PDF generation requires pandoc. Please install pandoc and regenerate.
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install -y pandoc texlive-xetex

# macOS
brew install pandoc
brew install --cask mactex

# Verify installation
pandoc --version
xelatex --version

# Regenerate PDF
python scripts/generate_ga_certification_package.py --certify-only
```

### Issue: Cross-Region Metrics Are All Zero

**Symptoms:**
```json
{
  "cross_region": [
    {"region_name": "us-east-1", "availability": 0.0},
    {"region_name": "us-west-2", "availability": 0.0},
    {"region_name": "eu-central-1", "availability": 0.0}
  ]
}
```

**Solution:**
1. Check Prometheus metrics have `region` label:
   ```bash
   curl "http://prometheus.tars-production.svc.cluster.local:9090/api/v1/query?query=up{region=~'.*'}"
   ```
2. If no `region` label, add Prometheus relabeling config:
   ```yaml
   # prometheus.yml
   - job_name: 'tars'
     relabel_configs:
       - source_labels: [__meta_kubernetes_pod_label_topology_kubernetes_io_region]
         target_label: region
   ```

### Issue: WebSocket Monitor Hangs

**Symptoms:**
Monitor runs indefinitely without completing.

**Solution:**
1. Verify WebSocket endpoint is accessible:
   ```bash
   curl -i -N -H "Upgrade: websocket" \
     -H "Connection: Upgrade" \
     -H "Authorization: Bearer $TARS_API_TOKEN" \
     wss://api.tars.ai/ws
   ```
2. Check WebSocket pod logs for connection issues:
   ```bash
   kubectl logs -n tars-production -l app=dashboard-api --tail=100 | grep websocket
   ```
3. Use shorter duration for testing:
   ```bash
   python observability/monitor_websocket_health.py --duration 300  # 5 minutes
   ```

---

## Post-GA Day Actions

### Immediate (0-24h)
- [ ] Review GA_DAY_REPORT.md for any warnings or failures
- [ ] Monitor production metrics dashboard
- [ ] Check for unexpected drift patterns
- [ ] Verify all alerts cleared

### Short-term (1-7 days)
- [ ] Schedule post-mortem if any incidents occurred
- [ ] Update runbooks based on GA learnings
- [ ] Optimize queries identified as slow in drift analysis
- [ ] Plan capacity adjustments based on cost estimates

### Long-term (1-4 weeks)
- [ ] Incorporate GA metrics into monthly reports
- [ ] Update baseline for next release (v1.0.2)
- [ ] Archive certification package for compliance
- [ ] Implement process improvements from retrospective

---

## Example: Full Execution Script

```bash
#!/bin/bash
# ga_day_automation.sh - Full GA Day execution script

set -e

# Configuration
GA_START=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PROMETHEUS_URL="http://prometheus.tars-production.svc.cluster.local:9090"
OUTPUT_DIR="ga_certification_$(date +%Y%m%d)"
BASELINE_FILE="baseline_metrics.json"

echo "🚀 Starting GA Day Automation"
echo "Start Time: $GA_START"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Step 1: Launch background monitors
echo "📊 Launching KPI collector..."
python observability/ga_kpi_collector.py \
  --duration 24 \
  --interval 5 \
  --output ga_kpis \
  --prometheus-url $PROMETHEUS_URL \
  > kpi_collector.log 2>&1 &
KPI_PID=$!

echo "🔍 Launching drift detector..."
python observability/drift_detector.py \
  --baseline-file $BASELINE_FILE \
  --duration 24 \
  --check-interval 300 \
  --output drift_analysis.json \
  --prometheus-url $PROMETHEUS_URL \
  > drift_detector.log 2>&1 &
DRIFT_PID=$!

echo "📈 Launching SLO monitor..."
python observability/monitor_realtime_slos.py \
  --duration 86400 \
  --interval 60 \
  --output slo_monitor \
  --prometheus-url $PROMETHEUS_URL \
  > slo_monitor.log 2>&1 &
SLO_PID=$!

# Step 2: Run WebSocket validation
echo "🌐 Running WebSocket health validation (1 hour)..."
python observability/monitor_websocket_health.py \
  --endpoint wss://api.tars.ai/ws \
  --duration 3600 \
  --interval 300 \
  --output ws_health_metrics.json

# Step 3: Wait for 24-hour window
echo "⏱️  Waiting for 24-hour GA window to complete..."
for hour in {1..24}; do
    echo "  Hour $hour/24 elapsed"
    sleep 3600

    # Check monitors still running
    kill -0 $KPI_PID 2>/dev/null || echo "⚠️  KPI collector stopped"
    kill -0 $DRIFT_PID 2>/dev/null || echo "⚠️  Drift detector stopped"
    kill -0 $SLO_PID 2>/dev/null || echo "⚠️  SLO monitor stopped"
done

# Step 4: Stop monitors
echo "🛑 Stopping monitors..."
kill $KPI_PID $DRIFT_PID $SLO_PID 2>/dev/null || true
wait $KPI_PID $DRIFT_PID $SLO_PID 2>/dev/null || true

# Step 5: Run validation tests
echo "🧪 Running GA validation tests..."
pytest release/v1_0_1/production_validation_suite.py \
  --ga-mode \
  --namespace tars-production \
  --version v1.0.1 \
  --html=ga_validation_results.html \
  --self-contained-html

# Step 6: Generate certification package
echo "📦 Generating certification package..."
GA_END=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python scripts/generate_ga_certification_package.py \
  --ga-start "$GA_START" \
  --ga-end "$GA_END" \
  --output-dir ./$OUTPUT_DIR \
  --include-artifacts

# Step 7: Verify and report
echo ""
echo "✅ GA Day Automation Complete!"
echo "   Start: $GA_START"
echo "   End: $GA_END"
echo "   Output: $OUTPUT_DIR/"
echo ""
echo "Certification Status:"
jq -r '.certification_status' $OUTPUT_DIR/package/certification_metadata.json
echo ""
echo "Next steps:"
echo "  1. Review: cat $OUTPUT_DIR/package/GA_DAY_REPORT.md"
echo "  2. Archive: tar -czf ga_cert_archive.tar.gz $OUTPUT_DIR/"
echo "  3. Upload to compliance storage"
```

---

## Additional Resources

- [Phase 14.4 Implementation Report](./PHASE14_4_IMPLEMENTATION_REPORT.md)
- [Phase 14.5 Implementation Report](./PHASE14_5_IMPLEMENTATION_REPORT.md)
- [GA Day Runbook](./docs/final/GA_DAY_RUNBOOK.md)
- [Production Validation Suite README](./release/v1_0_1/README.md)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Maintainer:** T.A.R.S. Platform Team

🚀 Generated with [Claude Code](https://claude.com/claude-code)

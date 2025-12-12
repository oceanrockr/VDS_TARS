# Phase 14.6 - Production Runbook

**T.A.R.S. v1.0.2-pre - Production Operations Guide**

This runbook provides step-by-step procedures for operating Phase 14.6 monitoring in production environments.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Day-by-Day Operations](#day-by-day-operations)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Incident Response](#incident-response)
7. [Data Management](#data-management)
8. [Automation Setup](#automation-setup)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Rollback Procedures](#rollback-procedures)

---

## Overview

### Purpose

Phase 14.6 provides post-GA 7-day stabilization monitoring for T.A.R.S. production deployments, including:

- **GA Day (Day 0):** Baseline KPI collection
- **Days 1-7:** Daily stability monitoring, anomaly detection, health reporting
- **Day 7:** Regression analysis and retrospective generation

### Success Criteria

- ✅ All monitoring tools run successfully for 7 days
- ✅ GA Day baseline captured accurately
- ✅ Daily drift detection operates within thresholds
- ✅ Comprehensive retrospective generated on Day 7
- ✅ Zero data loss or corruption
- ✅ SLO compliance maintained (99.9% availability)

### Key Stakeholders

| Role | Responsibility | Contact |
|------|----------------|---------|
| SRE Team | Run daily monitoring, respond to alerts | sre@veleron.dev |
| Engineering | Review retrospective, implement fixes | engineering@veleron.dev |
| Product | Approve v1.0.2 roadmap | product@veleron.dev |
| On-Call | 24/7 incident response | oncall@veleron.dev |

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 14.6 Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │  Prometheus  │────────▶│   GA KPI Collector (Day 0)   │  │
│  │   Metrics    │         └──────────────────────────────┘  │
│  └──────────────┘                        │                   │
│         │                                ▼                   │
│         │                    ┌──────────────────────────┐    │
│         │                    │  ga_kpi_summary.json      │    │
│         │                    │  (Baseline)               │    │
│         │                    └──────────────────────────┘    │
│         │                                │                   │
│         ▼                                ▼                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      7-Day Stability Monitor (Days 1-7)              │   │
│  │      - Collects daily metrics                        │   │
│  │      - Compares vs GA baseline                       │   │
│  │      - Detects drift > 10%                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                      │                                       │
│                      ▼                                       │
│          ┌──────────────────────────────┐                   │
│          │  day_01-07_summary.json      │                   │
│          └──────────────────────────────┘                   │
│                      │                                       │
│         ┌────────────┼────────────┐                         │
│         ▼            ▼            ▼                         │
│  ┌────────────┐ ┌────────────┐ ┌─────────────────────┐    │
│  │  Anomaly   │ │   Health   │ │    Regression       │    │
│  │  Detector  │ │  Reporter  │ │    Analyzer (Day 7) │    │
│  └────────────┘ └────────────┘ └─────────────────────┘    │
│         │            │                    │                 │
│         ▼            ▼                    ▼                 │
│  ┌───────────┐ ┌───────────┐  ┌─────────────────────┐     │
│  │ anomaly_  │ │  day_XX_  │  │ regression_summary  │     │
│  │ events    │ │  HEALTH   │  │ .json               │     │
│  └───────────┘ └───────────┘  └─────────────────────┘     │
│                                           │                 │
│                                           ▼                 │
│                        ┌──────────────────────────────────┐ │
│                        │  Retrospective Generator (Day 7) │ │
│                        │  - Aggregates all data           │ │
│                        │  - Analyzes trends                │ │
│                        │  - Generates v1.0.2 roadmap      │ │
│                        └──────────────────────────────────┘ │
│                                           │                 │
│                                           ▼                 │
│                        ┌──────────────────────────────────┐ │
│                        │  GA_7DAY_RETROSPECTIVE.md/.json  │ │
│                        └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
/data/
├── ga_kpis/
│   └── ga_kpi_summary.json                 # GA Day baseline
├── stability/
│   ├── day_01_summary.json                 # Day 1-7 summaries
│   ├── day_02_summary.json
│   ├── ...
│   └── day_07_summary.json
├── anomalies/
│   └── anomaly_events.json                 # All anomaly events
├── health/
│   ├── day_01_HEALTH.json                  # Daily health scores
│   ├── ...
│   └── day_07_HEALTH.json
├── regression/
│   └── regression_summary.json             # Day 7 regression analysis
└── output/
    ├── GA_7DAY_RETROSPECTIVE.md            # Final retrospective
    └── GA_7DAY_RETROSPECTIVE.json
```

---

## Pre-Deployment Checklist

### T-7 Days (Before GA)

- [ ] Install Phase 14.6 package: `pip install tars-observability`
- [ ] Verify Prometheus is accessible: `curl http://prometheus:9090/-/healthy`
- [ ] Create data directories: `mkdir -p /data/{ga_kpis,stability,anomalies,health,regression,output}`
- [ ] Set file permissions: `chown -R tars:tars /data`
- [ ] Configure cron jobs or Kubernetes CronJobs (see [Automation Setup](#automation-setup))
- [ ] Set up alerting (Slack, PagerDuty, email)
- [ ] Run smoke test: `bash scripts/test_phase14_6_pipeline.sh`
- [ ] Document GA timestamp (UTC): `export GA_TIMESTAMP="2025-11-18T00:00:00Z"`
- [ ] Schedule retrospective review meeting for Day 7 + 1

### GA Day (Day 0) - Before Launch

- [ ] Verify all T.A.R.S. services are running
- [ ] Verify Prometheus is scraping metrics
- [ ] Verify disk space available (minimum 500 MB)
- [ ] Test GA KPI collector dry-run: `tars-ga-kpi --help`
- [ ] Set up monitoring dashboard

### GA Day (Day 0) - At Launch

- [ ] Capture exact GA timestamp (UTC)
- [ ] Update `GA_TIMESTAMP` environment variable
- [ ] Trigger GA KPI collection (see [Day 0 Operations](#day-0-ga-day))

---

## Day-by-Day Operations

### Day 0: GA Day

**Objective:** Collect baseline KPIs on launch day.

**When to Run:** Within first hour of GA launch, then again at EOD (End of Day).

**Command:**

```bash
# Run GA KPI collection
tars-ga-kpi \
  --prometheus-url http://prometheus:9090 \
  --output-dir /data/ga_kpis \
  --ga-timestamp "2025-11-18T00:00:00Z"
```

**Expected Output:**

```
/data/ga_kpis/
└── ga_kpi_summary.json
```

**Verification:**

```bash
# Verify file exists
ls -lh /data/ga_kpis/ga_kpi_summary.json

# Check JSON structure
jq '.overall_availability, .error_rate, .p99_latency_ms' /data/ga_kpis/ga_kpi_summary.json

# Expected:
# "99.95"      (availability)
# "0.05"       (error rate)
# "120.5"      (P99 latency)
```

**Alerts:**

- ❌ **Critical:** GA KPI collection fails → Page on-call
- ⚠️ **Warning:** Availability < 99.9% on GA Day → Notify SRE

---

### Days 1-6: Daily Stability Monitoring

**Objective:** Monitor daily metrics, detect drift vs GA baseline.

**When to Run:** 11:59 PM UTC daily (automated via cron/K8s).

**Command:**

```bash
# Run stability monitor for Day 1
tars-stability-monitor \
  --prometheus-url http://prometheus:9090 \
  --output-dir /data/stability \
  --day-number 1 \
  --ga-baseline /data/ga_kpis/ga_kpi_summary.json

# Run health reporter after stability monitor completes
tars-health-report \
  --stability-data /data/stability/day_01_summary.json \
  --anomaly-data /data/anomalies/anomaly_events.json \
  --output-file /data/health/day_01_HEALTH.json \
  --day-number 1
```

**Expected Output:**

```
/data/stability/
├── day_01_summary.json
├── day_02_summary.json
├── ...

/data/health/
├── day_01_HEALTH.json
├── day_02_HEALTH.json
├── ...
```

**Daily Checklist:**

- [ ] Verify daily summary JSON created
- [ ] Check health score (0-100): `jq '.health_score' /data/health/day_01_HEALTH.json`
- [ ] Review drift metrics: `jq '.metrics_comparison' /data/stability/day_01_summary.json`
- [ ] Check for rollback recommendation: `jq '.rollback_recommendation' /data/stability/day_01_summary.json`

**Health Score Interpretation:**

| Score | Status | Action |
|-------|--------|--------|
| 90-100 | 🟢 Excellent | No action required |
| 70-89 | 🟡 Good | Monitor closely |
| 50-69 | 🟠 Fair | Investigate degradations |
| 0-49 | 🔴 Poor | **IMMEDIATE ACTION REQUIRED** |

**Alerts:**

- ❌ **Critical:** Health score < 50 → Page on-call, consider rollback
- ⚠️ **High:** Drift > 30% on critical metric → Notify SRE
- ⚡ **Medium:** Drift > 15% on any metric → Log for review

---

### Day 7: Regression Analysis & Retrospective

**Objective:** Analyze 7-day trends, generate comprehensive retrospective.

**When to Run:** End of Day 7 (after daily stability monitor).

**Commands:**

```bash
# Step 1: Run Day 7 stability monitoring (as usual)
tars-stability-monitor \
  --prometheus-url http://prometheus:9090 \
  --output-dir /data/stability \
  --day-number 7 \
  --ga-baseline /data/ga_kpis/ga_kpi_summary.json

# Step 2: Run regression analysis
tars-regression-analyzer \
  --ga-baseline /data/ga_kpis/ga_kpi_summary.json \
  --7day-summaries /data/stability \
  --output /data/regression/regression_summary.json

# Step 3: Generate retrospective
tars-retro --auto --output-dir /data/output
```

**Expected Output:**

```
/data/regression/
└── regression_summary.json

/data/output/
├── GA_7DAY_RETROSPECTIVE.md      # Human-readable report
└── GA_7DAY_RETROSPECTIVE.json    # Machine-readable data
```

**Post-Generation Tasks:**

1. **Review Retrospective:**
   ```bash
   cat /data/output/GA_7DAY_RETROSPECTIVE.md
   ```

2. **Share with Team:**
   ```bash
   # Slack notification
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"🎉 GA Day +7 Retrospective Ready!\nView at: /data/output/GA_7DAY_RETROSPECTIVE.md"}' \
     https://hooks.slack.com/services/YOUR/WEBHOOK/URL

   # Email report
   mail -s "T.A.R.S. GA +7 Retrospective" team@veleron.dev < /data/output/GA_7DAY_RETROSPECTIVE.md
   ```

3. **Create v1.0.2 Issues:**
   ```bash
   # Parse action items from JSON
   jq -r '.action_items[] | "\(.priority): \(.description)"' /data/output/GA_7DAY_RETROSPECTIVE.json

   # Or import to Jira/GitHub Issues (automation recommended)
   ```

4. **Archive Data:**
   ```bash
   # Create archive
   tar -czf tars_ga_7day_$(date +%Y%m%d).tar.gz /data

   # Upload to S3/GCS
   aws s3 cp tars_ga_7day_$(date +%Y%m%d).tar.gz s3://tars-archives/
   ```

---

## Monitoring & Alerting

### Continuous Anomaly Detection (Optional)

Run anomaly detector continuously to catch issues in real-time:

```bash
# Run in background (or as systemd service/K8s Deployment)
nohup tars-anomaly-detector \
  --prometheus-url http://prometheus:9090 \
  --output-file /data/anomalies/anomaly_events.json \
  --baseline /data/ga_kpis/ga_kpi_summary.json \
  --z-threshold 3.0 \
  > /var/log/tars/anomaly_detector.log 2>&1 &
```

**Alert on Anomalies:**

```bash
# Check for new anomalies every 5 minutes
*/5 * * * * /usr/local/bin/check_anomalies.sh

# check_anomalies.sh:
#!/bin/bash
ANOMALY_COUNT=$(jq 'length' /data/anomalies/anomaly_events.json)
if [ "$ANOMALY_COUNT" -gt 0 ]; then
  echo "⚠️ Detected $ANOMALY_COUNT anomalies!"
  # Send alert to Slack/PagerDuty
fi
```

### Key Metrics to Monitor

| Metric | Threshold | Alert Level |
|--------|-----------|-------------|
| Health Score | < 50 | 🔴 Critical |
| Availability | < 99.9% | 🔴 Critical |
| P99 Latency | > 200ms | 🟠 High |
| Error Rate | > 1% | 🟠 High |
| CPU Drift | > 30% | 🟡 Medium |
| Memory Drift | > 30% | 🟡 Medium |
| Anomaly Count | > 10/day | 🟡 Medium |

### Dashboard Setup

**Grafana Dashboard (Example):**

```json
{
  "dashboard": {
    "title": "T.A.R.S. Phase 14.6 - GA +7 Monitoring",
    "panels": [
      {
        "title": "Health Score (7-Day Trend)",
        "targets": [{
          "expr": "tars_health_score"
        }]
      },
      {
        "title": "Availability vs SLO (99.9%)",
        "targets": [{
          "expr": "tars_availability"
        }]
      },
      {
        "title": "P99 Latency Drift",
        "targets": [{
          "expr": "(tars_p99_latency - tars_ga_p99_latency) / tars_ga_p99_latency * 100"
        }]
      }
    ]
  }
}
```

---

## Incident Response

### Scenario 1: Health Score < 50

**Severity:** 🔴 Critical

**Immediate Actions:**

1. Check current health report:
   ```bash
   jq '.' /data/health/day_0X_HEALTH.json
   ```

2. Identify degradations:
   ```bash
   jq '.degradations[] | select(.severity == "critical" or .severity == "high")' /data/health/day_0X_HEALTH.json
   ```

3. Check mitigation recommendations:
   ```bash
   jq '.mitigation_plan' /data/health/day_0X_HEALTH.json
   ```

4. Execute mitigation (example):
   - Restart affected services
   - Scale up resources
   - Apply hotfix

5. Re-run health reporter to verify:
   ```bash
   tars-health-report --day-number X ...
   ```

6. Document in incident log

---

### Scenario 2: Drift > 30% on Critical Metric

**Severity:** 🟠 High

**Immediate Actions:**

1. Verify drift is real (not data collection error):
   ```bash
   # Check Prometheus directly
   curl 'http://prometheus:9090/api/v1/query?query=tars_availability'
   ```

2. Check daily summary for details:
   ```bash
   jq '.metrics_comparison | .[] | select(.drift_percent > 30)' /data/stability/day_0X_summary.json
   ```

3. Review potential causes (from daily summary):
   ```bash
   jq '.drift_analysis' /data/stability/day_0X_summary.json
   ```

4. Check rollback recommendation:
   ```bash
   jq '.rollback_recommendation' /data/stability/day_0X_summary.json
   ```

5. If rollback recommended → Execute rollback procedure

---

### Scenario 3: Data Collection Failure

**Severity:** 🟡 Medium

**Immediate Actions:**

1. Check logs:
   ```bash
   tail -f /var/log/tars/stability_monitor.log
   ```

2. Verify Prometheus connectivity:
   ```bash
   curl http://prometheus:9090/-/healthy
   ```

3. Check disk space:
   ```bash
   df -h /data
   ```

4. Re-run failed collection:
   ```bash
   tars-stability-monitor --day-number X ...
   ```

5. If persistent → Page on-call

---

## Data Management

### Data Retention

| Data Type | Retention Period | Storage Location |
|-----------|------------------|------------------|
| GA KPI | Permanent | `/data/ga_kpis` |
| Daily Summaries | 90 days | `/data/stability` |
| Anomaly Events | 30 days | `/data/anomalies` |
| Health Reports | 90 days | `/data/health` |
| Regression Analysis | Permanent | `/data/regression` |
| Retrospectives | Permanent | `/data/output` |

### Backup Procedure

**Daily Backup (Automated):**

```bash
#!/bin/bash
# /usr/local/bin/backup_tars_data.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/tars"
DATA_DIR="/data"

# Create backup
tar -czf "${BACKUP_DIR}/tars_data_${DATE}.tar.gz" "${DATA_DIR}"

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}/tars_data_${DATE}.tar.gz" s3://tars-backups/daily/

# Retain last 30 days locally
find "${BACKUP_DIR}" -name "tars_data_*.tar.gz" -mtime +30 -delete

echo "Backup completed: tars_data_${DATE}.tar.gz"
```

**Cron schedule:**

```bash
# Daily at 2 AM
0 2 * * * /usr/local/bin/backup_tars_data.sh
```

### Restore Procedure

```bash
# Download backup
aws s3 cp s3://tars-backups/daily/tars_data_20251118.tar.gz .

# Extract
tar -xzf tars_data_20251118.tar.gz -C /

# Verify
ls -la /data/ga_kpis/ga_kpi_summary.json
```

### Log Rotation

**logrotate configuration:**

```bash
# /etc/logrotate.d/tars
/var/log/tars/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 tars tars
}
```

---

## Automation Setup

### Option 1: Cron Jobs (Linux/macOS)

**Edit crontab:**

```bash
crontab -e
```

**Add entries:**

```bash
# Phase 14.6 Monitoring Automation

# Daily Stability Monitor (11:59 PM UTC)
59 23 * * * /usr/local/bin/run_stability_monitor.sh >> /var/log/tars/stability.log 2>&1

# Daily Health Reporter (12:05 AM UTC)
5 0 * * * /usr/local/bin/run_health_reporter.sh >> /var/log/tars/health.log 2>&1

# Continuous Anomaly Detection (every 5 minutes)
*/5 * * * * /usr/local/bin/check_anomalies.sh >> /var/log/tars/anomaly.log 2>&1

# Day 7 Retrospective (manual trigger)
# Run manually: /usr/local/bin/run_day7_analysis.sh
```

**Helper scripts:**

`/usr/local/bin/run_stability_monitor.sh`:

```bash
#!/bin/bash
DAY_NUMBER=$(( ($(date +%s) - $(date -d "2025-11-18" +%s)) / 86400 ))

if [ "$DAY_NUMBER" -ge 1 ] && [ "$DAY_NUMBER" -le 7 ]; then
  tars-stability-monitor \
    --prometheus-url http://localhost:9090 \
    --output-dir /data/stability \
    --day-number "$DAY_NUMBER" \
    --ga-baseline /data/ga_kpis/ga_kpi_summary.json
else
  echo "Outside 7-day monitoring window (Day $DAY_NUMBER)"
fi
```

### Option 2: Kubernetes CronJobs

See [Docker Deployment Guide - Kubernetes](./PHASE14_6_DOCKER.md#kubernetes-deployment).

### Option 3: Systemd Services (Linux)

**systemd service for continuous anomaly detection:**

`/etc/systemd/system/tars-anomaly-detector.service`:

```ini
[Unit]
Description=T.A.R.S. Phase 14.6 - Anomaly Detector
After=network.target

[Service]
Type=simple
User=tars
Group=tars
WorkingDirectory=/app
ExecStart=/usr/local/bin/tars-anomaly-detector \
  --prometheus-url http://localhost:9090 \
  --output-file /data/anomalies/anomaly_events.json \
  --baseline /data/ga_kpis/ga_kpi_summary.json \
  --z-threshold 3.0
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl enable tars-anomaly-detector
sudo systemctl start tars-anomaly-detector
sudo systemctl status tars-anomaly-detector
```

---

## Troubleshooting Guide

### Issue: "Prometheus connection refused"

**Symptoms:**
```
Error: Could not connect to Prometheus at http://prometheus:9090
```

**Diagnosis:**
```bash
# Check Prometheus is running
curl http://prometheus:9090/-/healthy

# Check network connectivity
ping prometheus
```

**Resolution:**
1. Verify Prometheus is running: `systemctl status prometheus`
2. Check firewall rules: `sudo iptables -L`
3. Update `PROMETHEUS_URL` environment variable
4. Retry collection

---

### Issue: "FileNotFoundError: ga_kpi_summary.json not found"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/data/ga_kpis/ga_kpi_summary.json'
```

**Diagnosis:**
```bash
ls -la /data/ga_kpis/
```

**Resolution:**
1. Ensure GA Day KPI collection completed successfully
2. Check file permissions: `chown -R tars:tars /data`
3. Re-run GA KPI collection if needed

---

### Issue: "Disk space full"

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Diagnosis:**
```bash
df -h /data
du -sh /data/*
```

**Resolution:**
1. Clean old logs: `find /var/log/tars -mtime +7 -delete`
2. Rotate anomaly events: `> /data/anomalies/anomaly_events.json`
3. Archive and compress old data
4. Increase disk size if needed

---

## Rollback Procedures

### When to Rollback

Initiate rollback if:

- ❌ Health score < 40 for 2+ consecutive days
- ❌ Availability < 99% for 24 hours
- ❌ Critical regression detected (drift > 50% on SLO metric)
- ❌ Daily summary recommends rollback

### Rollback Decision Tree

```
Health Score < 50?
  ├─ YES → Check Mitigation Plan
  │        ├─ Mitigations Available? → Execute Mitigations → Re-assess
  │        └─ No Mitigations? → ROLLBACK
  └─ NO → Continue Monitoring

Critical Degradation?
  ├─ YES → Check Severity
  │        ├─ Availability < 99%? → IMMEDIATE ROLLBACK
  │        ├─ Error Rate > 5%? → IMMEDIATE ROLLBACK
  │        └─ Other? → Investigate → Consider Rollback
  └─ NO → Continue Monitoring

Day 3+ with degradations?
  ├─ YES → Review Trend
  │        ├─ Improving? → Continue Monitoring
  │        └─ Degrading? → ROLLBACK
  └─ NO → Continue Monitoring
```

### Rollback Steps

1. **Notify Stakeholders:**
   ```bash
   # Slack alert
   curl -X POST https://hooks.slack.com/... \
     -d '{"text":"🚨 Initiating rollback to pre-GA version due to health score < 40"}'
   ```

2. **Execute Rollback:**
   ```bash
   # Kubernetes (recommended)
   kubectl rollout undo deployment/tars-agent

   # Or manual rollback
   git checkout v1.0.0
   ./deploy.sh
   ```

3. **Verify Rollback:**
   ```bash
   # Check version
   curl http://tars-api:8080/version

   # Monitor health
   watch -n 5 'curl -s http://tars-api:8080/health | jq .'
   ```

4. **Document Rollback:**
   - Update incident log
   - Record in retrospective
   - Schedule post-mortem

---

## Best Practices

1. **Always run GA KPI collection on Day 0** - This is the baseline for all subsequent comparisons
2. **Monitor health scores daily** - Don't wait for Day 7 to discover issues
3. **Automate everything** - Use cron/K8s CronJobs for reliability
4. **Back up data daily** - Protect against data loss
5. **Set up alerting** - Catch critical issues immediately
6. **Review retrospective with full team** - Cross-functional insights matter
7. **Archive retrospectives permanently** - Historical data is valuable for future GAs
8. **Test runbook procedures quarterly** - Ensure team is familiar with processes

---

## Additional Resources

- [Phase 14.6 Quickstart](./PHASE14_6_QUICKSTART.md)
- [Docker Deployment Guide](./PHASE14_6_DOCKER.md)
- [GitHub Repository](https://github.com/veleron-dev/tars)

---

**Generated:** 2025-11-26
**Version:** v1.0.2-pre
**Phase:** 14.6 - Production Runbook

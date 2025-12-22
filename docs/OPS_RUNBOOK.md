# T.A.R.S. Operator Runbook

**Version:** 1.0.5
**Phase:** 15 - Post-GA Operations Enablement
**Status:** Production
**Last Updated:** December 22, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Daily Operations](#daily-operations)
4. [Weekly Operations](#weekly-operations)
5. [Exit Code Reference](#exit-code-reference)
6. [Artifact Storage](#artifact-storage)
7. [Golden Path Commands](#golden-path-commands)
8. [Troubleshooting Quick Reference](#troubleshooting-quick-reference)

---

## Overview

This runbook provides step-by-step guidance for operators running T.A.R.S. Organization Health Governance tools in production environments. It covers daily monitoring, weekly trend reviews, and recommended operational patterns.

### Scope

- **Daily Run:** Quick health checks, alert review, SLA compliance
- **Weekly Run:** Trend analysis, correlation review, baseline comparisons
- **Ad-Hoc:** Incident investigation, executive reporting

---

## Prerequisites

### Environment Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.9+ | 3.11+ |
| **Memory** | 4 GB | 8 GB |
| **Disk** | 2 GB free | 10 GB free |
| **OS** | Linux/macOS/Windows | Linux (Ubuntu 22.04+) |

### Optional Tools

| Tool | Purpose | Required For |
|------|---------|--------------|
| **Redis** | Rate limiting, caching | API server, high-volume operations |
| **GPG** | Artifact signing | Secure release packaging |
| **Docker** | Containerized execution | K8s deployments |

### Installation Verification

```bash
# Verify Python version
python --version  # Should be 3.9+

# Verify T.A.R.S. installation
python -c "import analytics; print('T.A.R.S. OK')"

# Verify GA release validator
python scripts/ga_release_validator.py --help
```

---

## Daily Operations

### Morning Health Check (Recommended: 08:00 UTC)

Run the daily health check to assess organization-wide health status.

#### Step 1: Generate Org Health Report

```bash
# Generate org health report from repository health dashboards
python -m analytics.run_org_health \
    --root-dir ./org-health \
    --output ./reports/daily/org-health-$(date +%Y%m%d).json
```

**Expected Exit Codes:**
- `90` - All healthy, no SLO violations
- `91` - SLO violations detected (review required)
- `92` - High org risk (escalate immediately)

#### Step 2: Generate Org Alerts

```bash
# Generate alerts from org health report
python -m analytics.run_org_alerts \
    --org-report ./reports/daily/org-health-$(date +%Y%m%d).json \
    --output ./reports/daily/org-alerts-$(date +%Y%m%d).json
```

**Expected Exit Codes:**
- `100` - No alerts
- `101` - Alerts present (non-critical)
- `102` - Critical alerts (immediate action required)

#### Step 3: Run SLA Compliance Check

```bash
# Evaluate SLA compliance
python -m analytics.run_org_sla_intelligence \
    --org-report ./reports/daily/org-health-$(date +%Y%m%d).json \
    --alerts-report ./reports/daily/org-alerts-$(date +%Y%m%d).json \
    --output ./reports/daily/sla-report-$(date +%Y%m%d).json \
    --summary-only
```

**Expected Exit Codes:**
- `140` - All SLAs compliant
- `141` - At-risk SLAs (monitor closely)
- `142` - SLA breach (escalate to leadership)

#### Daily Checklist

- [ ] Org health report generated successfully
- [ ] Alert count reviewed (target: 0 critical)
- [ ] SLA compliance status verified (target: GREEN tier)
- [ ] Any breaches escalated per incident playbook
- [ ] Reports archived to designated storage

---

## Weekly Operations

### Weekly Trend Review (Recommended: Monday 10:00 UTC)

Perform deeper analysis including cross-repository correlations and temporal patterns.

#### Step 1: Generate Trend Correlation Report

```bash
# Analyze cross-repository trends
python -m analytics.run_org_trend_correlation \
    --org-report ./reports/daily/org-health-$(date +%Y%m%d).json \
    --output ./reports/weekly/trend-correlation-$(date +%Y%m%d).json
```

**Expected Exit Codes:**
- `120` - No concerning correlations
- `121` - Correlations found (investigate patterns)
- `122` - Critical cross-repo anomaly (immediate review)

#### Step 2: Generate Temporal Intelligence Report

```bash
# Analyze temporal patterns and propagation paths
python -m analytics.run_org_temporal_intelligence \
    --org-report ./reports/daily/org-health-$(date +%Y%m%d).json \
    --trend-correlation-report ./reports/weekly/trend-correlation-$(date +%Y%m%d).json \
    --output ./reports/weekly/temporal-intelligence-$(date +%Y%m%d).json
```

**Expected Exit Codes:**
- `130` - No temporal risks
- `131` - Temporal correlations found (monitor propagation)
- `132` - Critical propagation risk (immediate containment)

#### Step 3: Generate Full SLA Intelligence Report

```bash
# Full SLA intelligence with all inputs
python -m analytics.run_org_sla_intelligence \
    --org-report ./reports/daily/org-health-$(date +%Y%m%d).json \
    --alerts-report ./reports/daily/org-alerts-$(date +%Y%m%d).json \
    --trend-correlation-report ./reports/weekly/trend-correlation-$(date +%Y%m%d).json \
    --temporal-intelligence-report ./reports/weekly/temporal-intelligence-$(date +%Y%m%d).json \
    --output ./reports/weekly/sla-intelligence-$(date +%Y%m%d).json
```

#### Step 4: Baseline Comparison

Compare current metrics against established baselines:

```bash
# Compare with previous week
diff -u \
    ./reports/weekly/sla-intelligence-$(date -d "last week" +%Y%m%d).json \
    ./reports/weekly/sla-intelligence-$(date +%Y%m%d).json \
    > ./reports/weekly/baseline-diff-$(date +%Y%m%d).txt
```

#### Weekly Checklist

- [ ] Trend correlation report analyzed
- [ ] Temporal intelligence patterns reviewed
- [ ] Leader/follower relationships documented
- [ ] Propagation paths identified and monitored
- [ ] Baseline comparison shows no regression
- [ ] Weekly summary prepared for stakeholders
- [ ] Reports archived to long-term storage

---

## Exit Code Reference

### Quick Reference Table

| Code Range | Module | Description |
|------------|--------|-------------|
| 90-99 | Org Health | Organization health aggregation |
| 100-109 | Org Alerts | Organization alerting engine |
| 120-129 | Trend Correlation | Cross-repo trend analysis |
| 130-139 | Temporal Intelligence | Time-lagged correlation |
| 140-149 | SLA Intelligence | SLA compliance & readiness |
| 150-159 | GA Validator | Release validation |
| 199 | All | General error (any module) |

### Org Health Exit Codes (90-99)

| Code | Constant | Operator Action |
|------|----------|-----------------|
| 90 | `EXIT_ORG_SUCCESS` | No action - all healthy |
| 91 | `EXIT_SLO_VIOLATIONS` | Review SLO violations, create tickets |
| 92 | `EXIT_HIGH_ORG_RISK` | Escalate to leadership immediately |
| 93 | `EXIT_NO_REPOS_DISCOVERED` | Check --root-dir path, verify repo structure |
| 94 | `EXIT_CONFIG_ERROR` | Review configuration file syntax |
| 95 | `EXIT_AGGREGATION_ERROR` | Check input file permissions and format |
| 99 | `EXIT_GENERAL_ORG_ERROR` | Check logs, contact engineering |

### Org Alerts Exit Codes (100-109)

| Code | Constant | Operator Action |
|------|----------|-----------------|
| 100 | `EXIT_ORG_ALERT_SUCCESS` | No action - no alerts |
| 101 | `EXIT_ALERTS_PRESENT` | Review non-critical alerts, triage |
| 102 | `EXIT_CRITICAL_ALERTS` | Follow incident playbook immediately |
| 103 | `EXIT_ALERTING_CONFIG_ERROR` | Review escalation config syntax |
| 104 | `EXIT_ORG_REPORT_PARSE_ERROR` | Regenerate org-health report |
| 105 | `EXIT_ROUTING_FAILURE` | Check notification channel configs |
| 199 | `EXIT_GENERAL_ALERTING_ERROR` | Check logs, verify dependencies |

### Trend Correlation Exit Codes (120-129)

| Code | Constant | Operator Action |
|------|----------|-----------------|
| 120 | `EXIT_CORRELATION_SUCCESS` | No action - normal patterns |
| 121 | `EXIT_CORRELATIONS_FOUND` | Review correlation clusters |
| 122 | `EXIT_CRITICAL_ANOMALY` | Investigate synchronized decline |
| 123 | `EXIT_CORRELATION_CONFIG_ERROR` | Check threshold configuration |
| 124 | `EXIT_CORRELATION_PARSE_ERROR` | Regenerate org-health report |
| 199 | `EXIT_GENERAL_CORRELATION_ERROR` | Check logs, verify data format |

### Temporal Intelligence Exit Codes (130-139)

| Code | Constant | Operator Action |
|------|----------|-----------------|
| 130 | `EXIT_TEMPORAL_SUCCESS` | No action - no temporal risks |
| 131 | `EXIT_TEMPORAL_CORRELATIONS_FOUND` | Monitor identified leader repos |
| 132 | `EXIT_CRITICAL_PROPAGATION_RISK` | Contain propagation, isolate leader |
| 133 | `EXIT_TEMPORAL_CONFIG_ERROR` | Check lag window configuration |
| 134 | `EXIT_TEMPORAL_PARSE_ERROR` | Regenerate required input reports |
| 199 | `EXIT_GENERAL_TEMPORAL_ERROR` | Check logs, verify dependencies |

### SLA Intelligence Exit Codes (140-149)

| Code | Constant | Operator Action |
|------|----------|-----------------|
| 140 | `EXIT_SLA_SUCCESS` | No action - all SLAs met |
| 141 | `EXIT_SLA_AT_RISK` | Increase monitoring frequency |
| 142 | `EXIT_SLA_BREACH` | Escalate to leadership, initiate RCA |
| 143 | `EXIT_SLA_CONFIG_ERROR` | Review SLA policy file syntax |
| 144 | `EXIT_SLA_PARSE_ERROR` | Check input report formats |
| 199 | `EXIT_GENERAL_SLA_ERROR` | Check logs, contact engineering |

### GA Validator Exit Codes (150-159)

| Code | Constant | Operator Action |
|------|----------|-----------------|
| 150 | `EXIT_GA_READY` | Safe to release |
| 151 | `EXIT_GA_BLOCKED` | Review warnings before release |
| 152 | `EXIT_GA_FAILED` | Fix errors before release |
| 199 | `EXIT_GENERAL_ERROR` | Check validator logs |

---

## Artifact Storage

### Recommended Directory Structure

```
reports/
├── daily/                      # Daily health reports
│   ├── org-health-YYYYMMDD.json
│   ├── org-alerts-YYYYMMDD.json
│   └── sla-report-YYYYMMDD.json
├── weekly/                     # Weekly trend reports
│   ├── trend-correlation-YYYYMMDD.json
│   ├── temporal-intelligence-YYYYMMDD.json
│   ├── sla-intelligence-YYYYMMDD.json
│   └── baseline-diff-YYYYMMDD.txt
├── executive/                  # Board-ready reports
│   ├── executive-summary-YYYYMM.md
│   └── sla-scorecard-YYYYMM.json
└── archive/                    # Long-term retention
    └── YYYY/
        └── MM/
            └── ...
```

### Retention Policy

| Report Type | Retention Period | Storage Tier |
|-------------|------------------|--------------|
| Daily reports | 30 days | Hot storage |
| Weekly reports | 90 days | Warm storage |
| Executive reports | 1 year | Cold storage |
| Incident reports | 7 years | Archive |

### Archive Script Example

```bash
#!/bin/bash
# archive_reports.sh - Run monthly

ARCHIVE_DIR="./reports/archive/$(date +%Y/%m)"
mkdir -p "$ARCHIVE_DIR"

# Archive daily reports older than 30 days
find ./reports/daily -name "*.json" -mtime +30 -exec mv {} "$ARCHIVE_DIR/" \;

# Compress archived reports
gzip "$ARCHIVE_DIR"/*.json
```

---

## Golden Path Commands

### Minimal Daily Run (Copy-Paste Ready)

```bash
# One-liner for daily health check
python -m analytics.run_org_health --root-dir ./org-health --output ./reports/daily/org-health-$(date +%Y%m%d).json && \
python -m analytics.run_org_alerts --org-report ./reports/daily/org-health-$(date +%Y%m%d).json --output ./reports/daily/org-alerts-$(date +%Y%m%d).json && \
python -m analytics.run_org_sla_intelligence --org-report ./reports/daily/org-health-$(date +%Y%m%d).json --output ./reports/daily/sla-report-$(date +%Y%m%d).json --summary-only
```

### Minimal Weekly Run (Copy-Paste Ready)

```bash
# One-liner for weekly trend analysis
python -m analytics.run_org_trend_correlation --org-report ./reports/daily/org-health-$(date +%Y%m%d).json --output ./reports/weekly/trend-correlation-$(date +%Y%m%d).json && \
python -m analytics.run_org_temporal_intelligence --org-report ./reports/daily/org-health-$(date +%Y%m%d).json --trend-correlation-report ./reports/weekly/trend-correlation-$(date +%Y%m%d).json --output ./reports/weekly/temporal-intelligence-$(date +%Y%m%d).json && \
python -m analytics.run_org_sla_intelligence --org-report ./reports/daily/org-health-$(date +%Y%m%d).json --alerts-report ./reports/daily/org-alerts-$(date +%Y%m%d).json --trend-correlation-report ./reports/weekly/trend-correlation-$(date +%Y%m%d).json --temporal-intelligence-report ./reports/weekly/temporal-intelligence-$(date +%Y%m%d).json --output ./reports/weekly/sla-intelligence-$(date +%Y%m%d).json
```

### CI/CD Integration Example

```yaml
# GitHub Actions daily check
name: Daily Health Check
on:
  schedule:
    - cron: '0 8 * * *'  # 08:00 UTC daily

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements-dev.txt
      - run: |
          python -m analytics.run_org_health \
            --root-dir ./org-health \
            --output ./org-health-report.json \
            --fail-on-slo-violation
      - uses: actions/upload-artifact@v4
        with:
          name: health-report
          path: ./org-health-report.json
```

---

## Troubleshooting Quick Reference

### Common Issues

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Exit code 93 | Missing repos | Check `--root-dir` path exists |
| Exit code 94/103 | Bad YAML | Validate config with `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` |
| Exit code 199 | Python error | Check full stack trace in logs |
| Empty reports | No data | Verify input files have content |
| Slow execution | Large datasets | Use `--summary-only` for faster runs |

### Log Locations

```bash
# Default log output
~/.tars/logs/

# View recent errors
grep -i error ~/.tars/logs/*.log | tail -20

# Enable verbose logging
export TARS_LOG_LEVEL=DEBUG
```

### Getting Help

```bash
# Module-specific help
python -m analytics.run_org_health --help
python -m analytics.run_org_alerts --help
python -m analytics.run_org_trend_correlation --help
python -m analytics.run_org_temporal_intelligence --help
python -m analytics.run_org_sla_intelligence --help

# GA validator help
python scripts/ga_release_validator.py --help
```

---

## Related Documentation

- [Incident Playbook](INCIDENT_PLAYBOOK.md) - Incident response procedures
- [Post-GA Governance](POST_GA_GOVERNANCE.md) - Change management policy
- [SLA Intelligence Engine Guide](ORG_SLA_INTELLIGENCE_ENGINE.md) - Detailed SLA engine docs
- [Temporal Intelligence Engine Guide](ORG_TEMPORAL_INTELLIGENCE_ENGINE.md) - Temporal analysis docs

---

**Document Version:** 1.0.0
**Maintained By:** T.A.R.S. Operations Team

# T.A.R.S. Operator Runbook

**Version:** 1.0.9
**Phase:** 19 - Production Ops Maturity
**Status:** Production
**Last Updated:** December 26, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Platform Quick Start](#platform-quick-start)
4. [If You See This Exit Code, Do This](#if-you-see-this-exit-code-do-this)
5. [Daily Operations](#daily-operations)
6. [Weekly Operations](#weekly-operations)
7. [30-Minute Operator Checklist](#30-minute-operator-checklist)
8. [Exit Code Reference](#exit-code-reference)
9. [Artifact Storage](#artifact-storage)
10. [Golden Path Commands](#golden-path-commands)
11. [CI/CD Integration](#cicd-integration)
12. [Troubleshooting Quick Reference](#troubleshooting-quick-reference)

---

## Overview

This runbook provides step-by-step guidance for operators running T.A.R.S. Organization Health Governance tools in production environments. It covers daily monitoring, weekly trend reviews, and recommended operational patterns.

### Scope

- **Daily Run:** Quick health checks, alert review, SLA compliance
- **Weekly Run:** Trend analysis, correlation review, baseline comparisons
- **Ad-Hoc:** Incident investigation, executive reporting

### Cross-Platform Support

T.A.R.S. v1.0.6 (Phase 16) provides full cross-platform support:
- **Linux/macOS:** Bash shell
- **Windows:** PowerShell 5.1+
- **CI/CD:** GitHub Actions, GitLab CI, Jenkins

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

**Linux/macOS (Bash):**
```bash
# Verify Python version
python --version  # Should be 3.9+

# Verify T.A.R.S. installation
python -c "import analytics; print('T.A.R.S. OK')"

# Verify orchestrator
python scripts/run_full_org_governance_pipeline.py --help
```

**Windows (PowerShell):**
```powershell
# Verify Python version
python --version  # Should be 3.9+

# Verify T.A.R.S. installation
python -c "import analytics; print('T.A.R.S. OK')"

# Verify orchestrator
python scripts/run_full_org_governance_pipeline.py --help
```

---

## Platform Quick Start

### Linux/macOS Quick Start (Bash)

```bash
# Set timestamp (optional - orchestrator auto-generates if not provided)
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)

# Run full pipeline with timestamp
python scripts/run_full_org_governance_pipeline.py \
    --root ./org-health \
    --timestamp $TIMESTAMP \
    --print-paths

# Package executive bundle
python scripts/package_executive_bundle.py \
    --run-dir ./reports/runs/tars-run-$TIMESTAMP
```

### Windows Quick Start (PowerShell)

```powershell
# Set timestamp (optional - orchestrator auto-generates if not provided)
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")

# Run full pipeline with timestamp
python scripts/run_full_org_governance_pipeline.py `
    --root ./org-health `
    --timestamp $Timestamp `
    --print-paths

# Package executive bundle
python scripts/package_executive_bundle.py `
    --run-dir ./reports/runs/tars-run-$Timestamp
```

### CI Agent Quick Start

The orchestrator auto-generates timestamps when not provided:

```bash
# Minimal CI command (timestamp auto-generated)
python scripts/run_full_org_governance_pipeline.py --root ./org-health --print-paths

# The output will show:
# Run Directory: ./reports/runs/tars-run-20251222-140000
# (timestamp is generated automatically)
```

---

## If You See This Exit Code, Do This

**Quick Action Table for Common Exit Codes**

| Exit Code | Meaning | Immediate Action | Escalate? |
|-----------|---------|------------------|-----------|
| **0** | Success | No action needed | No |
| **92** | High Org Risk | Review org health report, identify failing repos | Yes - Team Lead |
| **102** | Critical Alerts | Follow [Incident Playbook](INCIDENT_PLAYBOOK.md) SEV-1 | Yes - Immediately |
| **122** | Critical Anomaly | Investigate correlation clusters | Yes - Team Lead |
| **132** | Propagation Risk | Isolate leader repos, freeze deployments | Yes - Immediately |
| **141** | At-Risk SLAs | Increase monitoring frequency | No - Monitor |
| **142** | SLA Breach | Initiate incident response, notify stakeholders | Yes - Immediately |
| **199** | General Error | Check logs, verify configuration | No - Debug |

**Decision Flow:**
```
Exit Code >= 142?  --> YES --> Escalate to leadership, initiate incident response
                   --> NO  --> Exit Code >= 102?
                                --> YES --> Follow incident playbook, alert on-call
                                --> NO  --> Exit Code >= 92?
                                             --> YES --> Review and create tickets
                                             --> NO  --> Document and continue
```

---

## Daily Operations

### Morning Health Check (Recommended: 08:00 UTC)

Run the daily health check to assess organization-wide health status.

#### Using the Pipeline Orchestrator (Recommended)

The orchestrator handles all timestamp generation and file naming automatically:

**Linux/macOS:**
```bash
python scripts/run_full_org_governance_pipeline.py \
    --root ./org-health \
    --format structured \
    --print-paths
```

**Windows (PowerShell):**
```powershell
python scripts/run_full_org_governance_pipeline.py `
    --root ./org-health `
    --format structured `
    --print-paths
```

**Expected Exit Codes:**
- `0` - Pipeline completed successfully
- `1` - Pipeline error (one or more steps failed)
- `142` - SLA breach detected (with `--fail-on-breach`)

#### Manual Step-by-Step (Alternative)

For more control, run individual engines. Note: The orchestrator is preferred as it handles timestamps automatically.

**Step 1: Generate Org Health Report**

```bash
# Using the orchestrator is recommended, but for manual runs:
python -m analytics.run_org_health \
    --root-dir ./org-health \
    --output ./reports/daily/org-health-report.json
```

**Step 2: Generate Org Alerts**

```bash
python -m analytics.run_org_alerts \
    --org-report ./reports/daily/org-health-report.json \
    --output ./reports/daily/org-alerts.json
```

**Step 3: Run SLA Compliance Check**

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./reports/daily/org-health-report.json \
    --output ./reports/daily/sla-report.json \
    --summary-only
```

**Expected Exit Codes (Manual Run):**
- `90` - All healthy, no SLO violations
- `91` - SLO violations detected (review required)
- `92` - High org risk (escalate immediately)
- `100` - No alerts
- `101` - Alerts present (non-critical)
- `102` - Critical alerts (immediate action required)
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

#### Using the Pipeline Orchestrator (Recommended)

**Linux/macOS:**
```bash
python scripts/run_full_org_governance_pipeline.py \
    --root ./org-health \
    --format structured \
    --print-paths
```

**Windows (PowerShell):**
```powershell
python scripts/run_full_org_governance_pipeline.py `
    --root ./org-health `
    --format structured `
    --print-paths
```

#### Packaging Executive Bundle

After running the pipeline, package reports for stakeholders:

**Linux/macOS:**
```bash
# Get the latest run directory
LATEST_RUN=$(ls -td ./reports/runs/tars-run-* | head -1)

# Package for distribution
python scripts/package_executive_bundle.py \
    --run-dir $LATEST_RUN \
    --tar
```

**Windows (PowerShell):**
```powershell
# Get the latest run directory
$LatestRun = Get-ChildItem -Path "./reports/runs" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

# Package for distribution
python scripts/package_executive_bundle.py `
    --run-dir $LatestRun.FullName `
    --tar
```

#### Weekly Checklist

- [ ] Full pipeline run completed
- [ ] Trend correlation report analyzed
- [ ] Temporal intelligence patterns reviewed
- [ ] Leader/follower relationships documented
- [ ] Propagation paths identified and monitored
- [ ] Executive bundle packaged
- [ ] Reports archived to long-term storage

---

## 30-Minute Operator Checklist

**What Must Be Done Every Morning (08:00 UTC)**

Use this checklist to ensure all critical tasks are completed within 30 minutes.

### Phase 1: Execute Pipeline (5 minutes)

```bash
# Run the full pipeline
python scripts/run_full_org_governance_pipeline.py --root ./org-health --print-paths
```

- [ ] Pipeline completed without errors
- [ ] Note the exit code: ______

### Phase 2: Review Exit Code (2 minutes)

- [ ] If exit code is **0**: Proceed to Phase 3
- [ ] If exit code is **92, 102, 122, 132, or 142**: Stop and follow [If You See This Exit Code, Do This](#if-you-see-this-exit-code-do-this)
- [ ] If exit code is **141**: Note for monitoring, proceed to Phase 3
- [ ] If exit code is **199**: Check logs, fix configuration, re-run

### Phase 3: Quick Report Review (10 minutes)

```bash
# View executive summary
cat ./reports/runs/tars-run-*/executive-summary.md | head -50
```

- [ ] Executive readiness tier is GREEN or YELLOW
- [ ] No SLA breaches detected
- [ ] Critical alert count is 0

### Phase 4: Archive and Notify (5 minutes)

```bash
# Package executive bundle
python scripts/package_executive_bundle.py --run-dir $(ls -td ./reports/runs/tars-run-* | head -1)
```

- [ ] Bundle created successfully
- [ ] Compliance index generated
- [ ] Archive stored in designated location

### Phase 5: Document (8 minutes)

- [ ] Record exit code in operations log
- [ ] Note any at-risk SLAs for monitoring
- [ ] Create tickets for any non-critical issues found
- [ ] Update status dashboard if applicable

**Total Time Target: < 30 minutes**

---

## Exit Code Reference

### Quick Reference Table

| Code Range | Module | Description |
|------------|--------|-------------|
| 0-1 | Orchestrator | Pipeline success/failure |
| 90-99 | Org Health | Organization health aggregation |
| 100-109 | Org Alerts | Organization alerting engine |
| 120-129 | Trend Correlation | Cross-repo trend analysis |
| 130-139 | Temporal Intelligence | Time-lagged correlation |
| 140-149 | SLA Intelligence | SLA compliance & readiness |
| 150-159 | GA Validator | Release validation |
| 199 | All | General error (any module) |

### Orchestrator Exit Codes (0-1)

| Code | Constant | Operator Action |
|------|----------|-----------------|
| 0 | `EXIT_SUCCESS` | No action - pipeline completed |
| 1 | `EXIT_PIPELINE_ERROR` | Review step logs, check module availability |

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
├── runs/                       # Pipeline orchestrator outputs
│   ├── tars-run-20251222-080000/
│   │   ├── org-health-report.json
│   │   ├── org-alerts.json
│   │   ├── trend-correlation-report.json
│   │   ├── temporal-intelligence-report.json
│   │   ├── sla-intelligence-report.json
│   │   ├── executive-summary.md
│   │   └── bundle-manifest.json
│   └── tars-run-20251222-100000/
│       └── ...
├── executive/                  # Board-ready bundles
│   ├── tars-exec-bundle-1.0.6-20251222-120000.zip
│   ├── tars-exec-bundle-1.0.6-20251222-120000-manifest.json
│   └── tars-exec-bundle-1.0.6-20251222-120000-checksums.sha256
└── archive/                    # Long-term retention
    └── 2025/
        └── 12/
            └── ...
```

### Retention Policy

| Report Type | Retention Period | Storage Tier |
|-------------|------------------|--------------|
| Pipeline runs | 30 days | Hot storage |
| Executive bundles | 90 days | Warm storage |
| Monthly summaries | 1 year | Cold storage |
| Incident reports | 7 years | Archive |

### Archive Script Example

**Linux/macOS:**
```bash
#!/bin/bash
# archive_reports.sh - Run monthly

ARCHIVE_DIR="./reports/archive/$(date +%Y/%m)"
mkdir -p "$ARCHIVE_DIR"

# Archive pipeline runs older than 30 days
find ./reports/runs -name "tars-run-*" -type d -mtime +30 -exec mv {} "$ARCHIVE_DIR/" \;

# Compress archived runs
for dir in "$ARCHIVE_DIR"/tars-run-*; do
    tar -czf "${dir}.tar.gz" -C "$ARCHIVE_DIR" "$(basename $dir)"
    rm -rf "$dir"
done
```

**Windows (PowerShell):**
```powershell
# archive_reports.ps1 - Run monthly

$ArchiveDir = "./reports/archive/$(Get-Date -Format 'yyyy/MM')"
New-Item -ItemType Directory -Force -Path $ArchiveDir

# Archive pipeline runs older than 30 days
Get-ChildItem -Path "./reports/runs" -Directory |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    ForEach-Object { Move-Item $_.FullName -Destination $ArchiveDir }
```

---

## Golden Path Commands

### Golden Path CLI (Phase 19)

T.A.R.S. v1.0.9 introduces `tars_ops.py`, a single-command wrapper for common operations:

```bash
# Daily health check (quick, flat output)
python scripts/tars_ops.py daily

# Weekly trend analysis (full output, executive bundle)
python scripts/tars_ops.py weekly

# Incident response mode (full output, signed bundle)
python scripts/tars_ops.py incident --incident-id INC-12345

# All commands support config override
python scripts/tars_ops.py daily --config .github/config/tars.ci.yml
```

**Exit Code Guidance:** The wrapper prints next-action guidance after each run based on the exit code.

### Minimal Daily Run (Cross-Platform)

The orchestrator handles all platform differences:

```bash
# Works on all platforms - timestamp auto-generated
python scripts/run_full_org_governance_pipeline.py \
    --root ./org-health \
    --print-paths
```

### Full Weekly Run with Executive Bundle

**Linux/macOS:**
```bash
# Run pipeline and package results
python scripts/run_full_org_governance_pipeline.py --root ./org-health --print-paths && \
python scripts/package_executive_bundle.py --run-dir $(ls -td ./reports/runs/tars-run-* | head -1) --tar
```

**Windows (PowerShell):**
```powershell
# Run pipeline
python scripts/run_full_org_governance_pipeline.py --root ./org-health --print-paths

# Get latest run and package
$LatestRun = Get-ChildItem -Path "./reports/runs" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
python scripts/package_executive_bundle.py --run-dir $LatestRun.FullName --tar
```

### Dry Run (Preview Commands)

```bash
# Preview what commands would be executed
python scripts/run_full_org_governance_pipeline.py --root ./org-health --dry-run
```

---

## CI/CD Integration

### GitHub Actions Daily Check

See `.github/workflows/tars_daily_ops.yml` for the full workflow.

```yaml
name: Daily Health Check
on:
  schedule:
    - cron: '0 8 * * *'  # 08:00 UTC daily
  workflow_dispatch:
    inputs:
      fail_on_breach:
        description: 'Fail if SLA breach detected'
        type: boolean
        default: false

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
          python scripts/run_full_org_governance_pipeline.py \
            --root ./org-health \
            --print-paths \
            ${{ github.event.inputs.fail_on_breach == 'true' && '--fail-on-breach' || '' }}
      - uses: actions/upload-artifact@v4
        with:
          name: health-reports
          path: ./reports/runs/
```

### GitHub Actions Weekly Report

See `.github/workflows/tars_weekly_ops.yml` for the full workflow.

```yaml
name: Weekly Trend Report
on:
  schedule:
    - cron: '0 10 * * 1'  # 10:00 UTC Monday
  workflow_dispatch:

jobs:
  weekly-report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements-dev.txt
      - run: |
          python scripts/run_full_org_governance_pipeline.py \
            --root ./org-health \
            --format structured \
            --print-paths
      - run: |
          LATEST_RUN=$(ls -td ./reports/runs/tars-run-* | head -1)
          python scripts/package_executive_bundle.py \
            --run-dir $LATEST_RUN \
            --tar
      - uses: actions/upload-artifact@v4
        with:
          name: weekly-bundle
          path: ./release/executive/
```

---

## Troubleshooting Quick Reference

### Common Issues

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Exit code 93 | Missing repos | Check `--root` path exists |
| Exit code 94/103 | Bad YAML | Validate config syntax |
| Exit code 199 | Python error | Check full stack trace in logs |
| Empty reports | No data | Verify input files have content |
| Slow execution | Large datasets | Use `--summary-only` for faster runs |
| Module not found | Missing deps | Run `pip install -r requirements-dev.txt` |

### Log Locations

**Linux/macOS:**
```bash
# Default log output
~/.tars/logs/

# View recent errors
grep -i error ~/.tars/logs/*.log | tail -20

# Enable verbose logging
export TARS_LOG_LEVEL=DEBUG
```

**Windows (PowerShell):**
```powershell
# Default log output
$env:USERPROFILE\.tars\logs\

# View recent errors
Get-Content "$env:USERPROFILE\.tars\logs\*.log" |
    Select-String -Pattern "error" |
    Select-Object -Last 20

# Enable verbose logging
$env:TARS_LOG_LEVEL = "DEBUG"
```

### Getting Help

```bash
# Orchestrator and packager help
python scripts/run_full_org_governance_pipeline.py --help
python scripts/package_executive_bundle.py --help

# Module-specific help
python -m analytics.run_org_health --help
python -m analytics.run_org_alerts --help
python -m analytics.run_org_trend_correlation --help
python -m analytics.run_org_temporal_intelligence --help
python -m analytics.run_org_sla_intelligence --help
```

---

## Related Documentation

- [Incident Playbook](INCIDENT_PLAYBOOK.md) - Incident response procedures
- [Post-GA Governance](POST_GA_GOVERNANCE.md) - Change management policy
- [SLA Intelligence Engine Guide](ORG_SLA_INTELLIGENCE_ENGINE.md) - Detailed SLA engine docs
- [Temporal Intelligence Engine Guide](ORG_TEMPORAL_INTELLIGENCE_ENGINE.md) - Temporal analysis docs

---

**Document Version:** 1.0.1
**Maintained By:** T.A.R.S. Operations Team

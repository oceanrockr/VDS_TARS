# T.A.R.S. Organization Alerting & Escalation Engine Guide

**Phase:** 14.8 Task 2
**Version:** 1.0.0
**Author:** T.A.R.S. Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Alert Sources](#alert-sources)
4. [Alert Types & Severity](#alert-types--severity)
5. [Escalation Rules](#escalation-rules)
6. [Routing Channels](#routing-channels)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [CLI Reference](#cli-reference)
10. [Programmatic API](#programmatic-api)
11. [Exit Codes](#exit-codes)
12. [OrgAlertReport JSON Schema](#orgalertreport-json-schema)
13. [Configuration Reference](#configuration-reference)
14. [CI/CD Integration](#cicd-integration)
15. [Interpreting Results](#interpreting-results)
16. [Troubleshooting](#troubleshooting)
17. [Best Practices](#best-practices)

---

## Overview

The Organization Alerting & Escalation Engine provides unified, organization-wide alerting capabilities for the T.A.R.S. observability ecosystem. It serves as the **governance enforcement layer**, translating org-level health signals into actionable alerts with configurable escalation policies.

### What It Does

- **Generates alerts** from SLO violations, high-risk repos, trend degradation, and integrity issues
- **Evaluates escalation rules** to determine response actions
- **Routes alerts** to multiple channels (console, JSON, email, Slack stubs)
- **Integrates with CI/CD** pipelines via exit codes (100-109)
- **Provides actionable recommendations** for each alert

### Relationship to Task 1

This engine operates **downstream** of the Org Health Governance engine:

| Phase | Task | Tool | Produces | Consumes |
|-------|------|------|----------|----------|
| 14.8 | 1 | Org Health Aggregator | `org-health-report.json` | Per-repo artifacts |
| **14.8** | **2** | **Org Alerting Engine** | **`org-alerts.json`** | **`org-health-report.json`** |

### Use Cases

- **SLO Breach Notification**: Alert when SLO policies are violated
- **Risk Escalation**: Escalate when repos reach HIGH/CRITICAL risk
- **Trend Monitoring**: Alert on org-wide health degradation
- **Integrity Assurance**: Alert when data loading fails
- **CI/CD Gating**: Block deployments based on alert severity

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │           Org Health Governance Engine              │
                    │               (Task 1)                              │
                    │                                                     │
                    │  org-health-report.json                             │
                    │   - SLO evaluations (satisfied/violated)            │
                    │   - Top risk repositories                           │
                    │   - Org metrics (percent_green, declining, etc.)    │
                    │   - Load errors                                     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │        Org Alerting & Escalation Engine             │
                    │                   (Task 2)                          │
                    │                                                     │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │              OrgAlertGenerator                │  │
                    │  │                                               │  │
                    │  │  ┌─────────────┐  ┌─────────────────────────┐ │  │
                    │  │  │ SLO Alerts  │  │ Risk Alerts             │ │  │
                    │  │  │ (violated   │  │ (HIGH/CRITICAL repos)   │ │  │
                    │  │  │ policies)   │  │                         │ │  │
                    │  │  └─────────────┘  └─────────────────────────┘ │  │
                    │  │                                               │  │
                    │  │  ┌─────────────┐  ┌─────────────────────────┐ │  │
                    │  │  │ Trend       │  │ Integrity Alerts        │ │  │
                    │  │  │ Alerts      │  │ (load errors)           │ │  │
                    │  │  │ (declining) │  │                         │ │  │
                    │  │  └─────────────┘  └─────────────────────────┘ │  │
                    │  └───────────────────────────────────────────────┘  │
                    │                         │                           │
                    │                         ▼                           │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │            EscalationEngine                   │  │
                    │  │                                               │  │
                    │  │  - Match alerts against rules                 │  │
                    │  │  - Execute actions (stubs)                    │  │
                    │  │  - Track escalation history                   │  │
                    │  └───────────────────────────────────────────────┘  │
                    │                         │                           │
                    │                         ▼                           │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │            OrgAlertDispatcher                 │  │
                    │  │                                               │  │
                    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐          │  │
                    │  │  │ Console │ │ JSON    │ │ Email   │  ...     │  │
                    │  │  │         │ │ File    │ │ (stub)  │          │  │
                    │  │  └─────────┘ └─────────┘ └─────────┘          │  │
                    │  └───────────────────────────────────────────────┘  │
                    │                                                     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                  OrgAlertReport                     │
                    │                                                     │
                    │  - Alert list with severity & category              │
                    │  - Escalation actions taken                         │
                    │  - Routing status                                   │
                    │  - Exit codes (100-109) for CI/CD                   │
                    │                                                     │
                    └─────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **OrgAlertGenerator** | Generates alerts from org health report |
| **EscalationEngine** | Evaluates rules and executes escalation actions |
| **OrgAlertDispatcher** | Routes alerts to configured channels |
| **OrgAlertingEngine** | Main orchestrator for the pipeline |

---

## Alert Sources

The engine generates alerts from four sources:

### 1. SLO Violations

When an SLO policy in the org health report is marked `satisfied: false`:

```json
{
  "category": "slo",
  "severity": "medium",  // From violation_severity
  "title": "SLO Violated: org-percent-green",
  "message": "At least 80% green. Current: 0.60, Target: 0.80 (>=)",
  "slo_id": "org-percent-green",
  "current_value": 0.60,
  "target_value": 0.80,
  "violating_repos": ["repo-c", "repo-d", "repo-e"]
}
```

### 2. High-Risk Repositories

When repositories in `top_risk_repos` have `risk_tier` of HIGH or CRITICAL:

```json
{
  "category": "risk",
  "severity": "critical",
  "title": "CRITICAL Risk: repo-x",
  "message": "Repository 'Legacy Service' is at CRITICAL risk (score: 85.0)",
  "risk_tier": "critical",
  "risk_score": 85.0,
  "affected_repos": ["repo-x"],
  "reason_codes": ["health_red", "critical_issues:8"]
}
```

### 3. Org-Wide Trend Signals

When org metrics exceed configured thresholds:

| Metric | Warning | Critical | Description |
|--------|---------|----------|-------------|
| `percent_declining` | >= 20% | >= 40% | Repos with declining trends |
| `percent_green` | < 60% | < 40% | Repos with GREEN status |
| `avg_score` | < 70 | < 50 | Organization average score |

```json
{
  "category": "trend",
  "severity": "critical",
  "title": "Critical: High Percentage of Declining Repos",
  "message": "80.0% of repositories have declining health trends (threshold: 40%)",
  "metric_name": "percent_declining",
  "metric_value": 0.80,
  "threshold": 0.40
}
```

### 4. Integrity Issues

When `load_errors` exist in the org health report:

```json
{
  "category": "integrity",
  "severity": "high",
  "title": "Data Integrity: 2 Repository Load Error(s)",
  "message": "Failed to load health data for 2 repository(ies)",
  "affected_repos": ["repo-fail-1", "repo-fail-2"],
  "details": {"load_errors": [...]}
}
```

---

## Alert Types & Severity

### Categories

| Category | Code | Description |
|----------|------|-------------|
| **SLO** | `slo` | SLO/SLA policy violations |
| **RISK** | `risk` | High-risk repository alerts |
| **TREND** | `trend` | Org-wide trend signals |
| **INTEGRITY** | `integrity` | Data integrity issues |
| **CONFIG** | `config` | Configuration warnings |

### Severity Levels

| Severity | Code | Description | Response Time |
|----------|------|-------------|---------------|
| **CRITICAL** | `critical` | Emergency - immediate action | Minutes |
| **HIGH** | `high` | Urgent - same day | Hours |
| **MEDIUM** | `medium` | Notable - this week | Days |
| **LOW** | `low` | Informational - backlog | As available |

### Severity Determination

| Source | Condition | Severity |
|--------|-----------|----------|
| SLO | `violation_severity` field | As configured |
| Risk | `risk_tier == "critical"` | CRITICAL |
| Risk | `risk_tier == "high"` | HIGH |
| Trend | Exceeds critical threshold | CRITICAL |
| Trend | Exceeds warning threshold | MEDIUM |
| Integrity | > 2 load errors | HIGH |
| Integrity | <= 2 load errors | MEDIUM |

---

## Escalation Rules

### Rule Structure

```yaml
escalation_rules:
  - id: "slo-critical"
    description: "Escalate critical SLO violations"
    when:
      alert_category: "slo"
      severity: "critical"
    actions:
      - "escalate_to:oncall"
      - "notify:slack:org-slo-critical"
      - "notify:email:leadership"
    priority: 100
    enabled: true
```

### Condition Matching

| Field | Type | Description |
|-------|------|-------------|
| `alert_category` | enum | Match by category (slo, risk, trend, integrity) |
| `severity` | enum | Match by severity level |
| `metric` | string | Match by metric name (for trend alerts) |
| `operator` | string | Comparison operator (==, !=, <, <=, >, >=) |
| `value` | number | Threshold value for metric comparison |

### Action Types

| Action | Format | Description |
|--------|--------|-------------|
| **escalate_to** | `escalate_to:<target>` | Escalate to team (oncall, leadership) |
| **notify** | `notify:<channel>:<recipient>` | Send notification |
| **log** | `log` | Log the escalation |
| **suppress** | `suppress` | Suppress the alert |

### Action Examples

```yaml
# Escalate to on-call team
- "escalate_to:oncall"

# Notify Slack channel
- "notify:slack:org-alerts"

# Notify email recipient
- "notify:email:team-lead@example.com"

# Log the action
- "log"
```

### Default Escalation Rules

The engine includes default rules (use `--use-default-escalations`):

1. **slo-critical**: Escalate critical SLO violations to oncall + leadership
2. **high-risk-repo**: Notify Slack on high-risk repos
3. **critical-risk-repo**: Escalate critical-risk repos
4. **org-declining**: Alert when >= 30% declining
5. **integrity-issues**: Notify infrastructure team

---

## Routing Channels

### Available Channels

| Channel | Type | Status | Description |
|---------|------|--------|-------------|
| **Console** | `console` | Implemented | Terminal output |
| **JSON File** | `json_file` | Implemented | Write to file |
| **Stdout** | `stdout` | Implemented | JSON to stdout |
| **Email** | `email` | Stub | Email notifications |
| **Slack** | `slack` | Stub | Slack webhooks |
| **Webhook** | `webhook` | Stub | Generic webhooks |

### Channel Configuration

```yaml
channels:
  - channel_type: "console"
    enabled: true
    min_severity: "medium"

  - channel_type: "json_file"
    enabled: true
    output_path: "./org-alerts.json"

  - channel_type: "email"
    enabled: true
    email_to: "alerts@example.com"
    email_from: "tars@example.com"
    min_severity: "high"

  - channel_type: "slack"
    enabled: true
    slack_webhook_url: "https://hooks.slack.com/..."
    slack_channel: "#alerts"

  - channel_type: "webhook"
    enabled: true
    webhook_url: "https://api.example.com/alerts"
    webhook_headers:
      Authorization: "Bearer token"
```

### Channel Filtering

Each channel can filter alerts by:
- **min_severity**: Only dispatch alerts >= this severity
- **categories**: Only dispatch specific alert categories

---

## Installation

### Prerequisites

- Python 3.8+
- `org-health-report.json` (from Task 1)
- Optional: PyYAML for YAML config support

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from analytics.org_alerting_engine import OrgAlertingEngine; print('OK')"

# Check CLI
python -m analytics.run_org_alerts --help
```

---

## Quick Start

### Example 1: Basic Alerting

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json
```

### Example 2: With Default Escalation Rules

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json \
  --use-default-escalations
```

### Example 3: Output to JSON File

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json \
  --output ./org-alerts.json
```

### Example 4: CI/CD Mode (Fail on Critical)

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json \
  --fail-on-critical \
  --output ./org-alerts.json

# Exit code will be 102 if critical alerts present
echo "Exit code: $?"
```

### Example 5: Custom Configuration

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json \
  --config ./org-alerting-config.yaml \
  --fail-on-any-alerts
```

### Example 6: JSON Output to Stdout

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json \
  --json | jq '.total_alerts'
```

### Example 7: Custom Thresholds

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json \
  --declining-warning 0.15 \
  --declining-critical 0.30 \
  --green-warning 0.70 \
  --green-critical 0.50
```

---

## CLI Reference

### Basic Usage

```bash
python -m analytics.run_org_alerts [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--org-report PATH` | Path to org-health-report.json |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config PATH` | None | YAML/JSON config with escalation rules |
| `--output PATH` | None | Output path for alerts JSON |
| `--fail-on-critical` | False | Exit 102 on critical alerts |
| `--fail-on-any-alerts` | False | Exit 101 on any alerts |
| `--use-default-escalations` | False | Use default escalation rules |
| `--no-slo-alerts` | False | Disable SLO violation alerts |
| `--no-risk-alerts` | False | Disable risk alerts |
| `--no-trend-alerts` | False | Disable trend alerts |
| `--no-integrity-alerts` | False | Disable integrity alerts |
| `--summary-only` | False | Print quick summary only |
| `--json` | False | Output JSON to stdout |
| `--quiet` | False | Suppress output |
| `--verbose` | False | Enable debug output |

### Threshold Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--declining-warning` | 0.20 | Warning threshold for declining repos |
| `--declining-critical` | 0.40 | Critical threshold for declining repos |
| `--green-warning` | 0.60 | Warning threshold for green repos |
| `--green-critical` | 0.40 | Critical threshold for green repos |
| `--score-warning` | 70.0 | Warning threshold for avg score |
| `--score-critical` | 50.0 | Critical threshold for avg score |

---

## Programmatic API

### Basic Usage

```python
from pathlib import Path
from analytics.org_alerting_engine import (
    OrgAlertConfig,
    OrgAlertingEngine,
    create_default_escalation_rules,
    create_default_channels,
)

# Step 1: Configure
config = OrgAlertConfig(
    org_report_path=Path("./org-health-report.json"),
    output_path=Path("./org-alerts.json"),
    channels=create_default_channels(),
    escalation_rules=create_default_escalation_rules(),
    fail_on_critical=True,
)

# Step 2: Run analysis
engine = OrgAlertingEngine(config)
report, exit_code = engine.run()

# Step 3: Use results
print(f"Total Alerts: {report.total_alerts}")
print(f"Critical: {report.critical_alerts}")
print(f"Escalations: {report.escalations_triggered}")
print(f"Exit Code: {exit_code}")
```

### Custom Escalation Rules

```python
from analytics.org_alerting_engine import (
    EscalationRule,
    EscalationCondition,
    EscalationAction,
    OrgAlertCategory,
    AlertSeverity,
)

rules = [
    EscalationRule(
        id="custom-slo-critical",
        description="Escalate critical SLOs immediately",
        condition=EscalationCondition(
            alert_category=OrgAlertCategory.SLO,
            severity=AlertSeverity.CRITICAL
        ),
        actions=[
            EscalationAction.from_string("escalate_to:oncall"),
            EscalationAction.from_string("notify:slack:org-critical"),
            EscalationAction.from_string("log")
        ],
        priority=100
    )
]

config = OrgAlertConfig(
    org_report_path=Path("./org-health-report.json"),
    escalation_rules=rules
)
```

### Direct Generator Usage

```python
from analytics.org_alerting_engine import OrgAlertGenerator
import json

# Load org report
with open("./org-health-report.json") as f:
    org_report = json.load(f)

# Generate alerts
generator = OrgAlertGenerator(config)
alerts = generator.generate_all_alerts(org_report)

# Process alerts
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.category.value}: {alert.title}")
```

---

## Exit Codes

| Code | Constant | Description |
|------|----------|-------------|
| 100 | `EXIT_ORG_ALERT_SUCCESS` | Success, no alerts |
| 101 | `EXIT_ALERTS_PRESENT` | Alerts generated (non-critical) |
| 102 | `EXIT_CRITICAL_ALERTS` | Critical alerts present |
| 103 | `EXIT_ALERTING_CONFIG_ERROR` | Configuration error |
| 104 | `EXIT_ORG_REPORT_PARSE_ERROR` | Failed to parse org report |
| 105 | `EXIT_ROUTING_FAILURE` | All channels failed |
| 199 | `EXIT_GENERAL_ALERTING_ERROR` | General error |

### Using Exit Codes in CI/CD

```bash
python -m analytics.run_org_alerts \
  --org-report ./org-health-report.json \
  --fail-on-critical

EXIT_CODE=$?

case $EXIT_CODE in
  100) echo "No alerts - all clear!" ;;
  101) echo "Alerts present - review recommended" ;;
  102) echo "CRITICAL alerts - blocking deployment" && exit 1 ;;
  104) echo "Failed to parse report" && exit 1 ;;
  *) echo "Error - investigate" && exit 1 ;;
esac
```

---

## OrgAlertReport JSON Schema

```json
{
  "report_id": "org_alert_report_20250107_120000",
  "generated_at": "2025-01-07T12:00:00.000000",
  "org_report_path": "/path/to/org-health-report.json",

  "total_alerts": 5,
  "critical_alerts": 1,
  "high_alerts": 2,
  "medium_alerts": 1,
  "low_alerts": 1,

  "slo_alerts": 2,
  "risk_alerts": 2,
  "trend_alerts": 1,
  "integrity_alerts": 0,

  "alerts": [
    {
      "alert_id": "org_alert_20250107_120000_0001",
      "category": "slo",
      "severity": "medium",
      "title": "SLO Violated: org-percent-green",
      "message": "At least 80% green. Current: 0.60, Target: 0.80",
      "timestamp": "2025-01-07T12:00:00.000000",
      "source_type": "slo",
      "slo_id": "org-percent-green",
      "current_value": 0.60,
      "target_value": 0.80,
      "violating_repos": ["repo-c", "repo-d"],
      "recommendations": [
        "Review and address issues in violating repositories"
      ],
      "escalated": true,
      "escalation_actions": ["notify:slack:org-alerts"]
    }
  ],

  "escalations_triggered": 3,
  "escalation_actions": [
    {
      "action_type": "notify",
      "rule_id": "slo-violations",
      "alert_id": "org_alert_...",
      "channel": "slack",
      "recipient": "org-alerts",
      "message": "Would notify org-alerts via slack"
    }
  ],

  "channels_dispatched": ["console", "json_file"],
  "dispatch_errors": [],

  "org_health_status": "yellow",
  "org_health_score": 75.0,
  "org_risk_tier": "medium",
  "slos_violated": 1,
  "total_repos": 10,

  "evaluation_duration_ms": 125.5
}
```

---

## Configuration Reference

### Full Configuration File

```yaml
# org-alerting-config.yaml

# Thresholds for trend-based alert generation
thresholds:
  percent_declining_warning: 0.20
  percent_declining_critical: 0.40
  percent_green_warning: 0.60
  percent_green_critical: 0.40
  avg_score_warning: 70.0
  avg_score_critical: 50.0
  high_volatility_threshold: 15.0

# Escalation rules
escalation_rules:
  - id: "slo-critical"
    description: "Escalate critical SLO violations"
    when:
      alert_category: "slo"
      severity: "critical"
    actions:
      - "escalate_to:oncall"
      - "notify:slack:org-slo-critical"
      - "notify:email:leadership@example.com"
    priority: 100
    enabled: true

  - id: "high-risk-repo"
    description: "Notify on high-risk repositories"
    when:
      alert_category: "risk"
      severity: "high"
    actions:
      - "notify:slack:repo-high-risk"
      - "log"
    priority: 80

  - id: "critical-risk-repo"
    description: "Escalate critical-risk repositories"
    when:
      alert_category: "risk"
      severity: "critical"
    actions:
      - "escalate_to:oncall"
      - "notify:slack:org-critical"
    priority: 90

  - id: "org-declining"
    description: "Alert on org-wide declining trend"
    when:
      metric: "percent_declining"
      operator: ">="
      value: 0.30
    actions:
      - "notify:email:org-devops@example.com"
      - "log"
    priority: 50

  - id: "integrity-issues"
    description: "Notify on integrity issues"
    when:
      alert_category: "integrity"
    actions:
      - "notify:slack:infrastructure"
      - "log"
    priority: 60

# Routing channels
channels:
  - channel_type: "console"
    enabled: true
    min_severity: "low"

  - channel_type: "json_file"
    enabled: true
    output_path: "./org-alerts.json"

  - channel_type: "email"
    enabled: false
    email_to: "alerts@example.com"
    email_from: "tars@example.com"
    min_severity: "high"

  - channel_type: "slack"
    enabled: false
    slack_webhook_url: "https://hooks.slack.com/services/..."
    slack_channel: "#tars-alerts"
    min_severity: "medium"

# Behavior flags
fail_on_critical: false
fail_on_any_alerts: false
verbose: false

# Alert generation flags
generate_slo_alerts: true
generate_risk_alerts: true
generate_trend_alerts: true
generate_integrity_alerts: true
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Org Alerting

on:
  workflow_run:
    workflows: ["Org Health Governance"]
    types: [completed]

jobs:
  org-alerts:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Download Org Health Report
        uses: actions/download-artifact@v4
        with:
          name: org-health-report
          path: ./

      - name: Run Org Alerting
        id: alerting
        run: |
          python -m analytics.run_org_alerts \
            --org-report ./org-health-report.json \
            --config ./org-alerting-config.yaml \
            --output ./org-alerts.json \
            --use-default-escalations
        continue-on-error: true

      - name: Upload Alert Report
        uses: actions/upload-artifact@v4
        with:
          name: org-alerts
          path: org-alerts.json

      - name: Check for Critical Alerts
        if: ${{ steps.alerting.outcome == 'failure' }}
        run: |
          CRITICAL=$(jq '.critical_alerts' org-alerts.json)
          if [ "$CRITICAL" -gt 0 ]; then
            echo "::error::$CRITICAL critical alerts detected!"
            exit 1
          fi
```

### GitLab CI

```yaml
org-alerting:
  stage: alerts
  needs:
    - org-health-governance
  script:
    - python -m analytics.run_org_alerts
        --org-report ./org-health-report.json
        --config ./org-alerting-config.yaml
        --output ./org-alerts.json
        --fail-on-critical
  artifacts:
    paths:
      - org-alerts.json
    expire_in: 30 days
  allow_failure: false
```

---

## Interpreting Results

### Alert Priority Matrix

| Category + Severity | Priority | Response |
|---------------------|----------|----------|
| SLO + CRITICAL | P1 | Immediate escalation |
| RISK + CRITICAL | P1 | Immediate escalation |
| SLO + HIGH | P2 | Same-day response |
| RISK + HIGH | P2 | Same-day response |
| TREND + CRITICAL | P2 | Same-day response |
| INTEGRITY + HIGH | P2 | Same-day response |
| Any + MEDIUM | P3 | This week |
| Any + LOW | P4 | Backlog |

### Escalation Flow

```
Alert Generated
      │
      ▼
Match Escalation Rules (by priority)
      │
      ├─► Rule Matched
      │        │
      │        ▼
      │   Execute Actions (stubs)
      │        │
      │        ├─► escalate_to: Log escalation target
      │        ├─► notify: Log notification details
      │        └─► log: Record in escalation log
      │
      └─► No Rule Matched
               │
               ▼
         Alert dispatched to channels only
```

---

## Troubleshooting

### No Alerts Generated

**Symptoms:** Exit code 100, no alerts in report

**Causes:**
- Org health report has no issues
- Alert generation disabled
- Thresholds too permissive

**Solutions:**
```bash
# Check org report status
cat org-health-report.json | jq '.org_health_status, .slos_violated'

# Run with verbose mode
python -m analytics.run_org_alerts --org-report ./report.json --verbose
```

### Config Error

**Symptoms:** Exit code 103

**Causes:**
- Invalid YAML/JSON syntax
- Invalid escalation rule format
- Unknown action type

**Solutions:**
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check for typos in category/severity
cat config.yaml | grep -E "alert_category|severity"
```

### Unexpected Alert Severity

**Symptoms:** Alerts with wrong severity

**Causes:**
- SLO `violation_severity` not set
- Threshold misconfiguration

**Solutions:**
```bash
# Check SLO violation severity in org report
cat org-health-report.json | jq '.slo_results[] | {slo_id, violation_severity}'

# Override thresholds
python -m analytics.run_org_alerts --org-report ./report.json \
  --declining-critical 0.50 \
  --green-critical 0.30
```

---

## Best Practices

### 1. Layer Your Escalations

Configure escalation rules from most to least severe:

```yaml
# Priority 100: Immediate
- id: critical-slo
  priority: 100
  when: {severity: critical}
  actions: [escalate_to:oncall]

# Priority 80: Same-day
- id: high-severity
  priority: 80
  when: {severity: high}
  actions: [notify:slack:alerts]

# Priority 50: Tracking
- id: all-alerts
  priority: 50
  when: {}
  actions: [log]
```

### 2. Use CI/CD Gates Appropriately

```bash
# Development: Warn but don't fail
python -m analytics.run_org_alerts --org-report ./report.json

# Staging: Fail on critical
python -m analytics.run_org_alerts --org-report ./report.json --fail-on-critical

# Production: Fail on any
python -m analytics.run_org_alerts --org-report ./report.json --fail-on-any-alerts
```

### 3. Configure Channel Filtering

Don't spam all channels with all alerts:

```yaml
channels:
  - channel_type: console
    min_severity: low       # See everything locally

  - channel_type: slack
    min_severity: medium    # Only notable issues to Slack

  - channel_type: email
    min_severity: high      # Email only urgent items
```

### 4. Review and Act

- Check alert reports daily
- Track escalation action completion
- Tune thresholds based on noise levels
- Update escalation rules as team structure changes

---

## Summary

The Organization Alerting & Escalation Engine provides:

- **Multi-Source Alerts**: SLO violations, risk tiers, trends, integrity
- **Escalation Rules**: Configurable conditions and actions
- **Routing Channels**: Console, JSON, email, Slack stubs
- **CI/CD Integration**: Exit codes (100-109) for pipeline gates
- **Actionable Output**: Recommendations and escalation tracking

For related documentation:
- [Organization Health Governance Guide](./ORG_HEALTH_GOVERNANCE_GUIDE.md) - Task 1
- [Repository Health Dashboard Guide](./REPOSITORY_HEALTH_DASHBOARD_GUIDE.md) - Per-repo dashboards
- [Alerting Engine Guide](./ALERTING_ENGINE_GUIDE.md) - Per-repo alerting

---

**Version:** 1.0.0
**Phase:** 14.8 Task 2
**Last Updated:** 2025-01-07

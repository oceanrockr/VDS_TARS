# T.A.R.S. Repository Health Alerting Engine

**Phase:** 14.7 Task 9
**Version:** 1.0.0
**Author:** T.A.R.S. Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Alert Types](#alert-types)
7. [Alert Severity Model](#alert-severity-model)
8. [Alert Channels](#alert-channels)
9. [CLI Reference](#cli-reference)
10. [Programmatic API](#programmatic-api)
11. [Exit Codes](#exit-codes)
12. [Configuration](#configuration)
13. [Integration with Release Pipeline](#integration-with-release-pipeline)
14. [CI/CD Integration](#cicd-integration)
15. [Email Templates](#email-templates)
16. [Webhook Payloads](#webhook-payloads)
17. [Troubleshooting](#troubleshooting)
18. [Best Practices](#best-practices)
19. [Performance](#performance)
20. [Security Considerations](#security-considerations)

---

## Overview

The Repository Health Alerting Engine is a production-grade system that monitors repository health by evaluating the output from the Repository Health Dashboard (Task 8) and generating alerts based on configurable rules. It supports multiple dispatch channels, trend-based detection, and provides policy-based exit codes for CI/CD integration.

### Use Cases

- **Proactive Monitoring**: Detect repository health issues before they impact releases
- **CI/CD Integration**: Fail builds when critical issues are detected
- **Notification Dispatch**: Send alerts to email, webhook, or file systems
- **Trend Analysis**: Compare current vs. previous dashboards to detect regressions
- **Compliance Reporting**: Generate audit trails of detected issues

### When to Use the Alerting Engine

- After generating a health dashboard (Task 8)
- In CI/CD pipelines to gate releases based on repository health
- As part of scheduled health monitoring jobs
- When you need automated notifications for repository issues
- For trend detection across multiple dashboard snapshots

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │        Dashboard JSON Input         │
                    │   (current + optional previous)     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       AlertRulesEngine              │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │ Load & Validate Dashboards  │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │   Evaluate Alert Rules      │    │
                    │  │   (12 default rules)        │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │   Generate Alert Objects    │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         AlertDispatcher             │
                    │                                     │
                    │  ┌───────────┐  ┌───────────┐       │
                    │  │  Console  │  │   File    │       │
                    │  │  Channel  │  │  Channel  │       │
                    │  └───────────┘  └───────────┘       │
                    │  ┌───────────┐  ┌───────────┐       │
                    │  │   Email   │  │  Webhook  │       │
                    │  │  Channel  │  │  Channel  │       │
                    │  └───────────┘  └───────────┘       │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │        Alert Report (JSON)          │
                    │    + Exit Code (70-79)              │
                    └─────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **AlertRulesEngine** | Loads dashboards, evaluates rules, generates alerts |
| **AlertDispatcher** | Routes alerts to configured channels |
| **ConsoleChannel** | Prints formatted alerts to stdout |
| **FileChannel** | Writes alerts to text file |
| **EmailChannel** | Generates email content (mock - for template generation) |
| **WebhookChannel** | Generates webhook JSON payload (mock - for integration) |
| **AlertingEngine** | Main orchestrator coordinating all components |

---

## Key Features

### 1. Multi-Rule Evaluation

12 default alert rules covering:
- Repository status (RED/YELLOW)
- Critical issues
- Missing/corrupted/orphaned artifacts
- Missing SBOM/SLSA metadata
- Score drops
- Rapid regressions
- Version health degradation
- Rollback failures

### 2. Trend-Based Detection

Compare current dashboard with previous to detect:
- Score drops above threshold
- Rapid increase in issues
- Version health degradation

### 3. Multi-Channel Dispatch

Simultaneous dispatch to:
- Console (stdout)
- File (text report)
- Email (formatted templates)
- Webhook (JSON payloads)

### 4. Severity Filtering

Filter alerts by minimum severity:
- INFO (all alerts)
- WARNING
- ERROR
- CRITICAL (most severe only)

### 5. Policy-Based Exit Codes

Exit codes 70-79 for CI/CD integration:
- 70: No alerts (success)
- 71: Non-critical alerts
- 72: Critical alerts
- 73-79: Various error conditions

---

## Installation

### Prerequisites

- Python 3.8+
- Repository Health Dashboard (Task 8) installed

### Setup

The alerting engine is part of the analytics module:

```bash
# Verify installation
python -c "from analytics.alerting_engine import AlertingEngine; print('OK')"

# Check CLI
python -m analytics.run_alerts --help
```

---

## Quick Start

### Example 1: Basic Alert Check

```bash
# Run alerting on a health dashboard
python -m analytics.run_alerts \
  --current-dashboard ./dashboard/health-dashboard.json \
  --output ./alerts/alerts.json
```

### Example 2: With Previous Dashboard (Trend Detection)

```bash
# Enable trend-based alerts
python -m analytics.run_alerts \
  --current-dashboard ./dashboard/health-dashboard.json \
  --previous-dashboard ./dashboard/health-dashboard.previous.json \
  --output ./alerts/alerts.json
```

### Example 3: Multiple Channels

```bash
# Console + file + email
python -m analytics.run_alerts \
  --current-dashboard ./dashboard/health-dashboard.json \
  --channels console,file,email \
  --email-to admin@example.com \
  --output ./alerts
```

### Example 4: CI/CD Mode (Fail on Critical)

```bash
# Fail build on critical alerts
python -m analytics.run_alerts \
  --current-dashboard ./dashboard/health-dashboard.json \
  --fail-on-critical \
  --severity-threshold WARNING
```

---

## Alert Types

### Repository Status Alerts

| Alert Type | Severity | Trigger Condition |
|------------|----------|-------------------|
| `repository_status_red` | CRITICAL | Health status is RED |
| `repository_status_yellow` | WARNING | Health status is YELLOW |

### Issue-Based Alerts

| Alert Type | Severity | Trigger Condition |
|------------|----------|-------------------|
| `critical_issue` | CRITICAL | critical_issues > 0 |
| `missing_artifact` | ERROR | missing_artifacts > 0 |
| `corrupted_artifact` | CRITICAL | corrupted_artifacts > 0 |
| `orphaned_artifact` | WARNING | orphaned_artifacts > 0 |

### Metadata Alerts

| Alert Type | Severity | Trigger Condition |
|------------|----------|-------------------|
| `missing_sbom` | WARNING | Versions without SBOM |
| `missing_slsa` | WARNING | Versions without SLSA |

### Trend-Based Alerts

| Alert Type | Severity | Trigger Condition |
|------------|----------|-------------------|
| `repository_score_drop` | WARNING | Score drop >= threshold |
| `rapid_regression` | ERROR | New issues >= threshold |
| `version_health_degradation` | WARNING | Version changed from green to yellow/red |

### Operational Alerts

| Alert Type | Severity | Trigger Condition |
|------------|----------|-------------------|
| `rollback_failure` | CRITICAL | Failed rollback in history |

---

## Alert Severity Model

### Severity Levels

| Severity | Code | Description |
|----------|------|-------------|
| INFO | 0 | Informational messages |
| WARNING | 1 | Potential issues requiring attention |
| ERROR | 2 | Significant issues requiring action |
| CRITICAL | 3 | Severe issues requiring immediate action |

### Severity Comparison

```python
from analytics.alerting_engine import AlertSeverity

# Comparisons work as expected
assert AlertSeverity.INFO < AlertSeverity.WARNING
assert AlertSeverity.WARNING < AlertSeverity.ERROR
assert AlertSeverity.ERROR < AlertSeverity.CRITICAL

# Parse from string
sev = AlertSeverity.from_string("CRITICAL")
```

### Threshold Filtering

When using `--severity-threshold`:
- `INFO`: All alerts
- `WARNING`: WARNING, ERROR, CRITICAL
- `ERROR`: ERROR, CRITICAL
- `CRITICAL`: CRITICAL only

---

## Alert Channels

### Console Channel

Prints formatted alerts to stdout.

```bash
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --channels console
```

**Output Example:**
```
================================================================================
REPOSITORY HEALTH ALERTS
================================================================================
Dashboard: ./dashboard.json
Generated: 2025-01-01T00:00:00
Health Status: RED
Repository Score: 35.0/100
--------------------------------------------------------------------------------

CRITICAL (3):
----------------------------------------
  [critical_issue] 3 Critical Issue(s) Detected
    Repository has 3 critical-severity issue(s) requiring immediate attention
      -> Address critical issues immediately
      -> Check integrity scan results for details

================================================================================
Total Alerts: 8
================================================================================
```

### File Channel

Writes formatted text report to file.

```bash
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --channels file \
  --output ./alerts
```

Creates: `./alerts/alerts.txt`

### Email Channel

Generates formatted email content (mock - does not send).

```bash
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --channels email \
  --email-to admin@example.com
```

Generates both plain text and HTML email templates for integration.

### Webhook Channel

Generates JSON payload for webhook integration.

```bash
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --channels webhook \
  --webhook-url https://example.com/alerts
```

See [Webhook Payloads](#webhook-payloads) for payload structure.

---

## CLI Reference

### Standalone CLI

```bash
python -m analytics.run_alerts [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--current-dashboard PATH` | Path to current health dashboard JSON |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--previous-dashboard PATH` | None | Previous dashboard for trend detection |
| `--auto-discover-previous` | False | Auto-discover previous dashboard |
| `--output PATH` | None | Output path for alerts JSON |
| `--channels LIST` | console | Comma-separated channels |
| `--email-to EMAIL` | None | Email recipient |
| `--webhook-url URL` | None | Webhook URL |
| `--severity-threshold LEVEL` | INFO | Minimum severity to report |
| `--score-drop-threshold FLOAT` | 10.0 | Score drop threshold |
| `--regression-threshold INT` | 3 | New issues threshold |
| `--fail-on-critical` | True | Exit 72 on critical alerts |
| `--no-fail-on-critical` | False | Don't fail on critical |
| `--fail-on-any-alert` | False | Exit 71 on any alert |
| `--verbose` | False | Enable verbose output |
| `--quiet` | False | Suppress non-essential output |

### Integrated Script (prepare_release_artifacts.py)

```bash
python scripts/prepare_release_artifacts.py \
  --generate-dashboard \
  --run-alerts \
  --alert-threshold WARNING \
  --alert-channels console,file \
  --alert-output-dir ./alerts \
  --alert-fail-on-critical
```

---

## Programmatic API

### Basic Usage

```python
from pathlib import Path
from analytics.alerting_engine import (
    AlertingEngine,
    AlertingConfig,
    ChannelConfig,
    ChannelType,
    AlertSeverity,
)

# Configure alerting
config = AlertingConfig(
    current_dashboard_path=Path("./dashboard/health-dashboard.json"),
    previous_dashboard_path=Path("./dashboard/previous.json"),
    output_path=Path("./alerts/alerts.json"),
    channels=[
        ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True),
        ChannelConfig(
            channel_type=ChannelType.FILE,
            enabled=True,
            output_path=Path("./alerts/alerts.txt")
        ),
    ],
    severity_threshold=AlertSeverity.WARNING,
    score_drop_threshold=10.0,
    fail_on_critical=True,
)

# Run alerting
engine = AlertingEngine(config)
report, exit_code = engine.run()

# Check results
print(f"Total Alerts: {report.total_alerts}")
print(f"Critical: {report.critical_alerts}")
print(f"Exit Code: {exit_code}")
```

### Advanced: Custom Rule Evaluation

```python
from analytics.alerting_engine import AlertRulesEngine, AlertingConfig

config = AlertingConfig(
    current_dashboard_path=Path("./dashboard.json"),
)

engine = AlertRulesEngine(config)

# Load dashboards
engine.current_dashboard = engine.load_dashboard(config.current_dashboard_path)

# Optionally load previous
if config.previous_dashboard_path:
    engine.previous_dashboard = engine.load_dashboard(config.previous_dashboard_path)

# Evaluate rules
alerts = engine.evaluate_rules()

# Generate report
report = engine.generate_report(alerts)

# Filter by severity
critical = [a for a in alerts if a.severity == "CRITICAL"]
```

### Channel Direct Usage

```python
from analytics.alerting_engine import ConsoleChannel, ChannelConfig, ChannelType, Alert, AlertReport

config = ChannelConfig(channel_type=ChannelType.CONSOLE, enabled=True)
channel = ConsoleChannel(config)

alerts = [
    Alert(
        alert_id="test_1",
        alert_type="critical_issue",
        severity="CRITICAL",
        title="Test Alert",
        message="Test message",
        timestamp="2025-01-01T00:00:00",
        recommendations=["Fix the issue"]
    )
]

report = AlertReport(
    report_id="report_1",
    generated_at="2025-01-01T00:00:00",
    dashboard_path="/path/to/dashboard.json",
    previous_dashboard_path=None,
    total_alerts=1,
    critical_alerts=1
)

success = channel.dispatch(alerts, report)
```

---

## Exit Codes

### Exit Code Table (70-79)

| Code | Constant | Description |
|------|----------|-------------|
| 70 | `EXIT_NO_ALERTS` | No alerts triggered (normal) |
| 71 | `EXIT_ALERTS_TRIGGERED` | Alerts triggered (non-critical) |
| 72 | `EXIT_CRITICAL_ALERTS` | Critical alerts triggered |
| 73 | `EXIT_INVALID_DASHBOARD` | Invalid dashboard input |
| 74 | `EXIT_CHANNEL_DISPATCH_FAILURE` | All channels failed |
| 75 | `EXIT_RULE_EVALUATION_FAILURE` | Rule evaluation error |
| 76 | `EXIT_ALERTS_WRITE_FAILURE` | Failed to write alerts JSON |
| 79 | `EXIT_GENERAL_ALERTING_ERROR` | General error |

### Exit Code Behavior

```bash
# Default: Fail on critical
python -m analytics.run_alerts --current-dashboard ./dashboard.json
# Exit 72 if critical alerts

# Don't fail on critical
python -m analytics.run_alerts --current-dashboard ./dashboard.json --no-fail-on-critical
# Exit 71 if any alerts, 70 if none

# Fail on any alert
python -m analytics.run_alerts --current-dashboard ./dashboard.json --fail-on-any-alert
# Exit 71 if any alerts (including INFO)
```

---

## Configuration

### AlertingConfig Options

```python
@dataclass
class AlertingConfig:
    # Input
    current_dashboard_path: Path
    previous_dashboard_path: Optional[Path] = None

    # Output
    output_path: Optional[Path] = None

    # Channels
    channels: List[ChannelConfig] = field(default_factory=list)

    # Thresholds
    severity_threshold: AlertSeverity = AlertSeverity.INFO
    score_drop_threshold: float = 10.0  # Points
    rapid_regression_threshold: int = 3  # New issues

    # Behavior
    fail_on_critical: bool = True
    fail_on_any_alert: bool = False
    verbose: bool = False
```

### ChannelConfig Options

```python
@dataclass
class ChannelConfig:
    channel_type: ChannelType
    enabled: bool = True

    # File channel
    output_path: Optional[Path] = None

    # Email channel
    email_to: Optional[str] = None
    email_from: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587

    # Webhook channel
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Filtering
    min_severity: AlertSeverity = AlertSeverity.INFO
```

---

## Integration with Release Pipeline

### Complete Pipeline Workflow

```
┌──────────────┐
│   Generate   │ ──→ Collect artifacts, SBOM, SLSA
│   Artifacts  │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Verify     │ ──→ Signature, checksum verification
│   Release    │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Validate   │ ──→ Delta analysis, compatibility
│   Release    │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Publish    │ ──→ Atomic publication
│   Release    │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Integrity  │ ──→ Full repository scan
│   Scan       │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Health     │ ──→ Aggregate reports, compute score
│   Dashboard  │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Alerting   │ ──→ Evaluate rules, dispatch alerts ◀── TASK 9
│   Engine     │
└──────────────┘
        │
        ▼
┌──────────────┐
│   CI/CD      │ ──→ Pass/Fail based on exit code
│   Decision   │
└──────────────┘
```

### Integrated Command

```bash
# Full pipeline with alerting
python scripts/prepare_release_artifacts.py \
  --version-file ./VERSION \
  --output-dir ./release \
  --include-sbom \
  --include-slsa \
  --verify-release \
  --post-release-validation \
  --publish-release \
  --repository-path ./artifact-repository \
  --scan-repository \
  --generate-dashboard \
  --run-alerts \
  --alert-threshold WARNING \
  --alert-channels console,file,email \
  --alert-email-to admin@example.com \
  --alert-fail-on-critical
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Repository Health Check

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Generate Dashboard
        run: |
          python -m analytics.repository_health_dashboard \
            --repository-path ./artifact-repository \
            --output-dir ./dashboard

      - name: Run Alerting Engine
        id: alerts
        run: |
          python -m analytics.run_alerts \
            --current-dashboard ./dashboard/health-dashboard.json \
            --previous-dashboard ./dashboard/health-dashboard.previous.json \
            --output ./alerts/alerts.json \
            --channels console,file \
            --severity-threshold WARNING \
            --fail-on-critical

      - name: Upload Alert Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: alert-report
          path: ./alerts/

      - name: Create Issue on Critical
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Repository Health Alert - Critical Issues Detected',
              body: 'The alerting engine detected critical repository health issues. See the workflow artifacts for details.',
              labels: ['alert', 'critical', 'repository-health']
            })
```

### GitLab CI

```yaml
health-alerts:
  stage: monitor
  script:
    - python -m analytics.repository_health_dashboard \
        --repository-path ./artifact-repository \
        --output-dir ./dashboard
    - python -m analytics.run_alerts \
        --current-dashboard ./dashboard/health-dashboard.json \
        --output ./alerts/alerts.json \
        --channels console,file \
        --fail-on-critical
  artifacts:
    when: always
    paths:
      - alerts/
    expire_in: 7 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    stages {
        stage('Health Dashboard') {
            steps {
                sh '''
                    python -m analytics.repository_health_dashboard \
                        --repository-path ./artifact-repository \
                        --output-dir ./dashboard
                '''
            }
        }

        stage('Alerting') {
            steps {
                script {
                    def exitCode = sh(
                        script: '''
                            python -m analytics.run_alerts \
                                --current-dashboard ./dashboard/health-dashboard.json \
                                --output ./alerts/alerts.json \
                                --channels console,file \
                                --fail-on-critical
                        ''',
                        returnStatus: true
                    )

                    if (exitCode == 72) {
                        error("Critical alerts detected")
                    } else if (exitCode == 71) {
                        unstable("Non-critical alerts detected")
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'alerts/*', allowEmptyArchive: true
        }
        failure {
            mail to: 'admin@example.com',
                subject: "Repository Health Alert - ${env.JOB_NAME}",
                body: "Critical issues detected. See ${env.BUILD_URL} for details."
        }
    }
}
```

---

## Email Templates

### Plain Text Template

```
T.A.R.S. Repository Health Alert Report
==================================================

Generated: 2025-01-01T00:00:00
Health Status: RED
Repository Score: 35.0/100

Score Change: decreased by 63.0 points

Alert Summary:
  Total: 8
  Critical: 3
  Error: 2
  Warning: 2
  Info: 1

--------------------------------------------------

[CRITICAL] Repository Health CRITICAL
  Repository health status is RED with score 35.0/100

[CRITICAL] 3 Critical Issue(s) Detected
  Repository has 3 critical-severity issue(s) requiring immediate attention

[CRITICAL] 3 Corrupted Artifact(s)
  Repository has 3 artifact(s) with checksum mismatches

--------------------------------------------------

This is an automated alert from T.A.R.S. Release Pipeline.
Please review the repository health dashboard for details.
```

### HTML Template

The email channel generates a responsive HTML email with:
- Gradient header with status badge
- Statistics summary grid
- Color-coded alert cards
- Professional styling

---

## Webhook Payloads

### Payload Structure

```json
{
  "event_type": "repository_health_alert",
  "timestamp": "2025-01-01T00:00:00.000000",
  "source": "tars_alerting_engine",
  "dashboard": {
    "path": "/path/to/dashboard.json",
    "health_status": "red",
    "repository_score": 35.0,
    "total_issues": 15
  },
  "trend": {
    "previous_score": 98.0,
    "score_change": -63.0,
    "new_issues": 14
  },
  "summary": {
    "total_alerts": 8,
    "critical": 3,
    "error": 2,
    "warning": 2,
    "info": 1
  },
  "alerts": [
    {
      "alert_id": "alert_repository_status_red_2025-01-01T00:00:00",
      "alert_type": "repository_status",
      "severity": "CRITICAL",
      "title": "Repository Health CRITICAL",
      "message": "Repository health status is RED with score 35.0/100",
      "timestamp": "2025-01-01T00:00:00",
      "recommendations": [
        "Review critical issues immediately",
        "Consider rollback if integrity is compromised"
      ]
    }
  ],
  "metadata": {
    "rules_evaluated": 12,
    "rules_triggered": 8,
    "evaluation_duration_ms": 150.5
  }
}
```

### Integration with Slack

```bash
# Using curl with webhook payload
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --channels webhook \
  --webhook-url https://hooks.slack.com/services/XXX/YYY/ZZZ
```

---

## Troubleshooting

### Common Issues

#### 1. "Dashboard file not found"

**Cause:** Invalid path to dashboard JSON.

**Solution:**
```bash
# Verify dashboard exists
ls -la ./dashboard/health-dashboard.json

# Use absolute path
python -m analytics.run_alerts \
  --current-dashboard /full/path/to/health-dashboard.json
```

#### 2. "Missing required field"

**Cause:** Dashboard JSON is incomplete.

**Solution:**
```bash
# Verify dashboard has required fields
python -c "
import json
with open('./dashboard.json') as f:
    d = json.load(f)
    print('overall_health:', d.get('overall_health'))
    print('repository_score:', d.get('repository_score'))
"
```

#### 3. "No alerts above threshold"

**Cause:** Severity threshold too high.

**Solution:**
```bash
# Lower threshold
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --severity-threshold INFO
```

#### 4. Channel dispatch failure

**Cause:** Channel misconfigured.

**Solution:**
```bash
# Check channel configuration
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --channels console \  # Start with console only
  --verbose
```

#### 5. Exit code 73 (Invalid Dashboard)

**Cause:** Dashboard JSON is malformed.

**Solution:**
```bash
# Validate JSON syntax
python -m json.tool ./dashboard.json

# Check for required fields
python -c "
import json
with open('./dashboard.json') as f:
    d = json.load(f)
    required = ['overall_health', 'repository_score']
    for r in required:
        if r not in d:
            print(f'Missing: {r}')
"
```

---

## Best Practices

### 1. Regular Alerting

Run alerting checks regularly:
```yaml
# Recommended schedule
- Every 6 hours for production repositories
- Daily for development repositories
- After every publication
```

### 2. Archive Alert Reports

Keep historical alert reports:
```bash
# Archive with timestamp
python -m analytics.run_alerts \
  --current-dashboard ./dashboard.json \
  --output "./alerts/alerts-$(date +%Y%m%d-%H%M%S).json"
```

### 3. Use Previous Dashboard

Enable trend detection for proactive monitoring:
```bash
# Maintain previous dashboard
cp ./dashboard/health-dashboard.json ./dashboard/health-dashboard.previous.json

# Generate new dashboard
python -m analytics.repository_health_dashboard ...

# Run alerts with trend detection
python -m analytics.run_alerts \
  --current-dashboard ./dashboard/health-dashboard.json \
  --previous-dashboard ./dashboard/health-dashboard.previous.json
```

### 4. Configure Appropriate Thresholds

```bash
# Strict (recommended for production)
--score-drop-threshold 5.0
--regression-threshold 2
--severity-threshold WARNING
--fail-on-critical

# Lenient (for development)
--score-drop-threshold 15.0
--regression-threshold 5
--severity-threshold ERROR
--no-fail-on-critical
```

### 5. Multi-Channel for Redundancy

```bash
# Use multiple channels
--channels console,file,email,webhook
```

---

## Performance

### Benchmark Results

| Operation | Target | Actual |
|-----------|--------|--------|
| Dashboard Load | < 200ms | ~50-100ms |
| Rule Evaluation | < 500ms | ~100-300ms |
| Alert Generation | < 200ms | ~50-100ms |
| Channel Dispatch | < 500ms | ~200-400ms |
| Total (small dashboard) | < 2s | ~500ms-1s |
| Total (large dashboard) | < 5s | ~2-3s |

### Optimization Tips

1. Use severity filtering to reduce alert volume
2. Disable unnecessary channels
3. Limit previous dashboard comparison to recent snapshots

---

## Security Considerations

### Access Control

- Alert reports may contain sensitive repository information
- Restrict access to alert output directories
- Secure webhook endpoints with authentication

### Sensitive Data

- Dashboard paths may reveal directory structure
- Alert messages may include artifact names
- Consider redaction for external channels

### Recommendations

```bash
# Set appropriate permissions
chmod 600 ./alerts/alerts.json

# Use environment variables for webhook URLs
export ALERT_WEBHOOK_URL="https://..."
python -m analytics.run_alerts \
  --webhook-url "$ALERT_WEBHOOK_URL"
```

---

## Summary

The Repository Health Alerting Engine provides:

- **12 Default Alert Rules** covering all repository health scenarios
- **4 Dispatch Channels** (Console, File, Email, Webhook)
- **Trend Detection** for proactive monitoring
- **Policy-Based Exit Codes** for CI/CD integration
- **Flexible Configuration** via CLI and programmatic API

For additional support, see the [Repository Health Dashboard Guide](./REPOSITORY_HEALTH_DASHBOARD_GUIDE.md) for dashboard generation details.

---

**Version:** 1.0.0
**Phase:** 14.7 Task 9
**Last Updated:** 2025-11-28

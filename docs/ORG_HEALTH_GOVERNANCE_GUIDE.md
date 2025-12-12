# T.A.R.S. Organization Health Governance Guide

**Phase:** 14.8 Task 1
**Version:** 1.0.0
**Author:** T.A.R.S. Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Directory Structure](#directory-structure)
7. [SLO/SLA Configuration](#slosla-configuration)
8. [CLI Reference](#cli-reference)
9. [Programmatic API](#programmatic-api)
10. [Exit Codes](#exit-codes)
11. [OrgHealthReport JSON Schema](#orghealthreport-json-schema)
12. [Integration with Pipeline](#integration-with-pipeline)
13. [CI/CD Integration](#cicd-integration)
14. [Interpreting Results](#interpreting-results)
15. [Troubleshooting](#troubleshooting)
16. [Best Practices](#best-practices)

---

## Overview

The Organization Health Governance Engine provides multi-repository health aggregation and SLO/SLA policy evaluation for T.A.R.S. release pipelines.

### What It Does

- **Aggregates** health data from multiple repositories
- **Evaluates** SLO/SLA policies across the organization
- **Computes** org-level risk metrics and health scores
- **Produces** comprehensive reports with actionable recommendations
- **Integrates** with CI/CD pipelines via exit codes

### Use Cases

- **Org-Wide Monitoring**: Single-pane view of all repository health
- **SLO Compliance**: Track and enforce service level objectives
- **Risk Assessment**: Identify and prioritize high-risk repositories
- **Capacity Planning**: Forecast org-wide health trends
- **Governance Reporting**: Compliance reports for audits

### Relationship to Per-Repo Tools

This tool operates **on top of** the per-repository observability stack:

| Phase | Task | Tool | Produces |
|-------|------|------|----------|
| 14.7 | 8 | Repository Health Dashboard | `health-dashboard.json` |
| 14.7 | 9 | Alerting Engine | `alerts.json` |
| 14.7 | 10 | Trend Analyzer | `trend-report.json` |
| **14.8** | **1** | **Org Health Governance** | **`org-health-report.json`** |

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              Per-Repository Pipelines               │
                    │                                                     │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
                    │  │  Repo A  │  │  Repo B  │  │  Repo C  │   ...    │
                    │  │          │  │          │  │          │          │
                    │  │ Task 8   │  │ Task 8   │  │ Task 8   │          │
                    │  │ Task 9   │  │ Task 9   │  │ Task 9   │          │
                    │  │ Task 10  │  │ Task 10  │  │ Task 10  │          │
                    │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
                    │       │             │             │                 │
                    └───────┼─────────────┼─────────────┼─────────────────┘
                            │             │             │
                            ▼             ▼             ▼
                    ┌─────────────────────────────────────────────────────┐
                    │              Org Health Artifacts                   │
                    │                                                     │
                    │  org-health/                                        │
                    │    ├── repo-a/                                      │
                    │    │   ├── dashboard/health-dashboard.json          │
                    │    │   ├── alerts/alerts.json                       │
                    │    │   └── trends/trend-report.json                 │
                    │    ├── repo-b/                                      │
                    │    │   └── ...                                      │
                    │    └── repo-c/                                      │
                    │        └── ...                                      │
                    │                                                     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │           Org Health Governance Engine              │
                    │                                                     │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │             OrgHealthAggregator               │  │
                    │  │                                               │  │
                    │  │  ┌─────────────┐  ┌─────────────────────────┐ │  │
                    │  │  │ Discovery   │  │ Per-Repo Loading        │ │  │
                    │  │  │ & Loading   │  │ (Dashboard+Alerts+Trend)│ │  │
                    │  │  └─────────────┘  └─────────────────────────┘ │  │
                    │  │                                               │  │
                    │  │  ┌─────────────┐  ┌─────────────────────────┐ │  │
                    │  │  │ Risk Score  │  │ SLO Policy Evaluation   │ │  │
                    │  │  │ Computation │  │                         │ │  │
                    │  │  └─────────────┘  └─────────────────────────┘ │  │
                    │  │                                               │  │
                    │  │  ┌─────────────┐  ┌─────────────────────────┐ │  │
                    │  │  │ Org Metrics │  │ Recommendations         │ │  │
                    │  │  │ Aggregation │  │ Generator               │ │  │
                    │  │  └─────────────┘  └─────────────────────────┘ │  │
                    │  └───────────────────────────────────────────────┘  │
                    │                                                     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                  OrgHealthReport                    │
                    │                                                     │
                    │  - Org-level health status & score                  │
                    │  - SLO compliance summary                           │
                    │  - Risk tier distribution                           │
                    │  - Top risk repositories                            │
                    │  - Actionable recommendations                       │
                    │  - Exit codes (90-99) for CI/CD                     │
                    │                                                     │
                    └─────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **OrgHealthConfig** | Configuration for paths, SLOs, thresholds |
| **OrgHealthAggregator** | Core aggregation and evaluation engine |
| **OrgHealthEngine** | Main orchestrator for the analysis pipeline |
| **SloPolicy** | SLO policy definition with selectors and metrics |
| **OrgHealthReport** | Comprehensive output with all aggregated data |

---

## Key Features

### 1. Multi-Repository Aggregation

- Automatic discovery of repositories
- Loads dashboard, alerts, and trends for each repo
- Handles missing or partial artifacts gracefully
- Supports repository filtering by ID or tags

### 2. SLO/SLA Policy Evaluation

- Configurable SLO policies via YAML/JSON
- Tag-based and pattern-based repo selection
- Multiple metric types supported
- Per-repo and aggregate constraints

### 3. Risk Scoring & Tiering

Per-repository risk computed from:
- Health score (inverse relationship)
- Critical issue count
- Critical alert count
- Trend direction

Risk tiers: `LOW` | `MEDIUM` | `HIGH` | `CRITICAL`

### 4. Org-Level Metrics

- Health status distribution (GREEN/YELLOW/RED)
- Risk tier distribution
- Score statistics (avg, min, max, median)
- Issue and alert totals
- Trend direction distribution
- Derived KPIs (percent_green, percent_improving, etc.)

### 5. Recommendations Engine

Generates actionable recommendations based on:
- Critical risk repositories
- Declining health trends
- SLO violations
- High volatility patterns
- Missing monitoring data

### 6. CI/CD Integration

Exit codes (90-99) enable pipeline gates:
- `90`: Success, all SLOs satisfied
- `91`: SLO violations detected
- `92`: Org risk >= HIGH
- `93-99`: Various error conditions

---

## Installation

### Prerequisites

- Python 3.8+
- Per-repo artifacts (from Tasks 8-10)
- Optional: PyYAML for YAML config support

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from analytics.org_health_aggregator import OrgHealthEngine; print('OK')"

# Check CLI
python -m analytics.run_org_health --help
```

---

## Quick Start

### Example 1: Basic Org Health Analysis

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --output ./org-health-report.json
```

### Example 2: With Default SLO Policies

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --use-default-slos \
  --output ./org-health-report.json
```

### Example 3: Custom SLO Configuration

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --config ./org-health-config.yaml \
  --output ./org-health-report.json
```

### Example 4: CI/CD Mode (Fail on Violations)

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --config ./org-health-config.yaml \
  --fail-on-slo-violation \
  --fail-on-critical-risk \
  --output ./org-health-report.json

# Exit code will be 91 if SLOs violated, 92 if critical risk
echo "Exit code: $?"
```

### Example 5: Quick Summary

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --summary-only
```

### Example 6: JSON Output to Stdout

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --json | jq '.org_health_status'
```

---

## Directory Structure

### Expected Input Layout

```
org-health/
  ├── repo-a/
  │   ├── dashboard/
  │   │   └── health-dashboard.json
  │   ├── alerts/
  │   │   └── alerts.json
  │   └── trends/
  │       └── trend-report.json
  ├── repo-b/
  │   ├── dashboard/
  │   │   └── health-dashboard.json
  │   └── alerts/
  │       └── alerts.json
  ├── repo-c/
  │   └── dashboard/
  │       └── health-dashboard.json
  └── .hidden-repo/          # Ignored (hidden)
      └── ...
```

### Customizing Paths

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --dashboard-subdir dashboard \
  --alerts-subdir alerts \
  --trends-subdir trends \
  --dashboard-filename health-dashboard.json \
  --alerts-filename alerts.json \
  --trends-filename trend-report.json
```

---

## SLO/SLA Configuration

### Configuration File Format

**YAML Example (`org-health-config.yaml`):**

```yaml
# SLO Policies
slo_policies:
  # Percentage of GREEN repos
  - id: "org-percent-green"
    description: "At least 90% of repositories should be GREEN"
    repo_selector:
      all: true
    metric: "percent_green"
    target: 0.90
    operator: ">="
    violation_severity: "medium"

  # Core repos must be healthy
  - id: "core-repos-healthy"
    description: "All core repositories must be GREEN or YELLOW"
    repo_selector:
      tags: ["core", "critical"]
    metric: "percent_yellow_or_better"
    target: 1.0
    operator: "=="
    violation_severity: "high"

  # Critical issue cap
  - id: "critical-issue-cap"
    description: "No repository may have more than 5 critical issues"
    repo_selector:
      all: true
    metric: "critical_issues"
    target: 5
    operator: "<="
    violation_severity: "critical"

  # Minimum score threshold
  - id: "min-score-threshold"
    description: "All repositories should have score >= 60"
    repo_selector:
      all: true
    metric: "repository_score"
    target: 60.0
    operator: ">="
    aggregation: "min"
    violation_severity: "high"

  # Improving trend target
  - id: "improving-repos"
    description: "At least 50% of repos should be improving"
    repo_selector:
      all: true
    metric: "percent_improving"
    target: 0.50
    operator: ">="
    violation_severity: "low"

# Repository tags (optional)
repo_tags:
  repo-api-gateway: ["core", "critical"]
  repo-auth-service: ["core", "security"]
  repo-web-frontend: ["frontend"]
  repo-analytics: ["data"]

# Risk tier thresholds (optional)
risk_thresholds:
  low_score_threshold: 80.0
  medium_score_threshold: 60.0
  high_score_threshold: 40.0
  critical_issues_medium: 1
  critical_issues_high: 3
  critical_issues_critical: 5
```

### Repo Selector Options

| Field | Type | Description |
|-------|------|-------------|
| `all` | bool | Match all repositories |
| `tags` | list | Match repos with any of these tags |
| `id_pattern` | string | Regex pattern for repo ID |

### Supported Metrics

| Metric | Description | Aggregation |
|--------|-------------|-------------|
| `percent_green` | % of repos with GREEN status | N/A |
| `percent_yellow_or_better` | % of repos GREEN or YELLOW | N/A |
| `critical_issues` | Critical issue count | max |
| `total_issues` | Total issue count | max |
| `repository_score` | Repository health score | min/avg/max |
| `percent_improving` | % of repos with IMPROVING trend | N/A |

### Supported Operators

| Operator | Symbol | Description |
|----------|--------|-------------|
| EQUALS | `==` | Exact match |
| NOT_EQUALS | `!=` | Not equal |
| LESS_THAN | `<` | Less than |
| LESS_THAN_OR_EQUALS | `<=` | Less than or equal |
| GREATER_THAN | `>` | Greater than |
| GREATER_THAN_OR_EQUALS | `>=` | Greater than or equal |

---

## CLI Reference

### Basic Usage

```bash
python -m analytics.run_org_health [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--root-dir PATH` | Root directory containing per-repo health artifacts |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config PATH` | None | YAML/JSON config with SLO policies |
| `--output PATH` | None | Output path for report JSON |
| `--repos LIST` | All | Comma-separated repo IDs to analyze |
| `--use-default-slos` | False | Use default SLO policies |
| `--fail-on-slo-violation` | False | Exit 91 on SLO violations |
| `--fail-on-critical-risk` | False | Exit 92 on HIGH/CRITICAL org risk |
| `--summary-only` | False | Print quick summary only |
| `--json` | False | Output JSON to stdout |
| `--quiet` | False | Suppress output |
| `--verbose` | False | Enable debug output |

---

## Programmatic API

### Basic Usage

```python
from pathlib import Path
from analytics.org_health_aggregator import (
    OrgHealthConfig,
    OrgHealthEngine,
    SloPolicy,
    RepoSelector,
    SloOperator,
    create_default_slo_policies,
)

# Step 1: Configure
config = OrgHealthConfig(
    root_dir=Path("./org-health"),
    output_path=Path("./org-health-report.json"),
    slo_policies=create_default_slo_policies(),
    fail_on_slo_violation=True,
)

# Step 2: Run analysis
engine = OrgHealthEngine(config)
report, exit_code = engine.run()

# Step 3: Use results
print(f"Org Health: {report.org_health_status}")
print(f"Org Score: {report.org_health_score:.1f}")
print(f"Risk Tier: {report.org_risk_tier}")
print(f"SLOs: {report.slos_satisfied}/{report.total_slos} satisfied")
print(f"Exit Code: {exit_code}")
```

### Custom SLO Policies

```python
from analytics.org_health_aggregator import (
    SloPolicy,
    RepoSelector,
    SloOperator,
)

# Define custom SLOs
custom_slos = [
    SloPolicy(
        id="all-repos-green",
        description="All repositories must be GREEN",
        repo_selector=RepoSelector(all=True),
        metric="percent_green",
        target=1.0,
        operator=SloOperator.EQUALS,
        violation_severity="critical"
    ),
    SloPolicy(
        id="core-repos-high-score",
        description="Core repos must have score >= 85",
        repo_selector=RepoSelector(tags=["core"]),
        metric="repository_score",
        target=85.0,
        operator=SloOperator.GREATER_THAN_OR_EQUALS,
        aggregation="min",
        violation_severity="high"
    )
]

config = OrgHealthConfig(
    root_dir=Path("./org-health"),
    slo_policies=custom_slos,
    repo_tags={
        "api-gateway": ["core"],
        "auth-service": ["core"],
        "web-app": ["frontend"]
    }
)
```

### Direct Aggregator Usage

```python
from analytics.org_health_aggregator import OrgHealthAggregator

aggregator = OrgHealthAggregator(config)

# Step-by-step analysis
repos = aggregator.discover_repositories()
loaded = aggregator.load_all_repositories()
metrics = aggregator.compute_org_metrics()
slo_results = aggregator.evaluate_slos()
recommendations = aggregator.generate_recommendations()
report = aggregator.generate_org_health_report()

# Access specific data
for repo_id, snapshot in aggregator._repositories.items():
    print(f"{repo_id}: {snapshot.health_status.value}, risk={snapshot.risk_tier.value}")
```

---

## Exit Codes

| Code | Constant | Description |
|------|----------|-------------|
| 90 | `EXIT_ORG_SUCCESS` | Success, all SLOs satisfied |
| 91 | `EXIT_SLO_VIOLATIONS` | SLO violations detected |
| 92 | `EXIT_HIGH_ORG_RISK` | Org risk >= HIGH tier |
| 93 | `EXIT_NO_REPOS_DISCOVERED` | No repositories found |
| 94 | `EXIT_CONFIG_ERROR` | Configuration error |
| 95 | `EXIT_AGGREGATION_ERROR` | Data aggregation failed |
| 99 | `EXIT_GENERAL_ORG_ERROR` | General error |

### Using Exit Codes in CI/CD

```bash
python -m analytics.run_org_health \
  --root-dir ./org-health \
  --config ./config.yaml \
  --fail-on-slo-violation \
  --fail-on-critical-risk

EXIT_CODE=$?

case $EXIT_CODE in
  90) echo "All healthy!" ;;
  91) echo "SLO violations - review required" ;;
  92) echo "Critical org risk - block deployment" ;;
  *) echo "Error - investigate" ;;
esac
```

---

## OrgHealthReport JSON Schema

```json
{
  "report_id": "org_health_20250107_120000",
  "generated_at": "2025-01-07T12:00:00.000000",
  "root_dir": "/path/to/org-health",

  "org_health_status": "green|yellow|red",
  "org_health_score": 85.5,
  "org_risk_tier": "low|medium|high|critical",

  "metrics": {
    "total_repos": 10,
    "repos_green": 7,
    "repos_yellow": 2,
    "repos_red": 1,
    "repos_low_risk": 6,
    "repos_medium_risk": 3,
    "repos_high_risk": 1,
    "repos_critical_risk": 0,
    "repos_improving": 4,
    "repos_stable": 4,
    "repos_declining": 2,
    "avg_score": 82.3,
    "min_score": 55.0,
    "max_score": 98.0,
    "total_issues": 45,
    "total_critical_issues": 3,
    "percent_green": 70.0,
    "percent_improving": 40.0
  },

  "total_slos": 5,
  "slos_satisfied": 4,
  "slos_violated": 1,
  "slo_results": [
    {
      "slo_id": "org-percent-green",
      "slo_description": "At least 90% green",
      "satisfied": false,
      "current_value": 0.70,
      "target_value": 0.90,
      "operator": ">=",
      "repos_evaluated": 10,
      "violating_repos": ["repo-d", "repo-e", "repo-f"]
    }
  ],

  "repositories": [
    {
      "repo_id": "repo-a",
      "repo_name": "API Gateway",
      "health_status": "green",
      "repository_score": 92.0,
      "risk_tier": "low",
      "critical_issues": 0,
      "trends": {"overall_trend": "improving"}
    }
  ],

  "top_risk_repos": [
    {
      "repo_id": "repo-d",
      "repo_name": "Legacy Service",
      "risk_tier": "high",
      "risk_score": 65.0,
      "health_status": "red",
      "repository_score": 45.0,
      "trend_direction": "declining",
      "critical_issues": 5,
      "reason_codes": ["health_red", "critical_issues:5", "declining_trend"]
    }
  ],

  "policy_violations": [
    {
      "slo_id": "org-percent-green",
      "description": "At least 90% green",
      "current_value": 0.70,
      "target_value": 0.90,
      "violating_repos": ["repo-d", "repo-e", "repo-f"]
    }
  ],

  "recommendations": [
    {
      "recommendation_id": "rec_001",
      "priority": "high",
      "title": "Address Critical Risk Repositories",
      "message": "2 repositories are at HIGH risk level",
      "affected_repos": ["repo-d", "repo-e"],
      "suggested_actions": [
        "Review critical issues",
        "Consider rollback",
        "Escalate to team leads"
      ]
    }
  ],

  "repos_discovered": 10,
  "repos_loaded": 10,
  "repos_failed": 0,
  "analysis_duration_ms": 1250.5,
  "load_errors": []
}
```

---

## Integration with Pipeline

### Per-Repo to Org Flow

```
                Per-Repo Pipeline (runs for each repo)
                ┌─────────────────────────────────────────┐
                │                                         │
                │  1. Generate Dashboard (Task 8)         │
                │  2. Run Alerts (Task 9)                 │
                │  3. Update History + Run Trends (Task 10)
                │  4. Copy artifacts to org-health/       │
                │                                         │
                └──────────────┬──────────────────────────┘
                               │
                               ▼ (after all repos complete)
                ┌─────────────────────────────────────────┐
                │         Org Health Pipeline             │
                │                                         │
                │  1. Aggregate all repo data             │
                │  2. Evaluate SLOs                       │
                │  3. Generate org report                 │
                │  4. Gate on exit code                   │
                │                                         │
                └─────────────────────────────────────────┘
```

### Standalone Script

```bash
# Run org health governance
python scripts/run_org_health_governance.py \
  --root-dir ./org-health \
  --config ./org-health-config.yaml \
  --output ./release/org-health-report.json \
  --fail-on-slo-violation \
  --fail-on-critical-risk
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Org Health Governance

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:
  workflow_run:
    workflows: ["Per-Repo Health Pipelines"]
    types: [completed]

jobs:
  org-health:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Download Per-Repo Artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: health-*
          path: ./org-health
          merge-multiple: true

      - name: Run Org Health Analysis
        id: org-health
        run: |
          python -m analytics.run_org_health \
            --root-dir ./org-health \
            --config ./org-health-config.yaml \
            --output ./org-health-report.json \
            --use-default-slos \
            --fail-on-slo-violation
        continue-on-error: true

      - name: Upload Org Health Report
        uses: actions/upload-artifact@v4
        with:
          name: org-health-report
          path: org-health-report.json

      - name: Check Results
        run: |
          if [ "${{ steps.org-health.outcome }}" == "failure" ]; then
            echo "::warning::Org health check found issues"
            cat org-health-report.json | jq '.recommendations'
          fi
```

### GitLab CI

```yaml
org-health-governance:
  stage: governance
  needs:
    - repo-a-health
    - repo-b-health
    - repo-c-health
  script:
    - python -m analytics.run_org_health
        --root-dir ./org-health
        --config ./org-health-config.yaml
        --output ./org-health-report.json
        --fail-on-slo-violation
  artifacts:
    paths:
      - org-health-report.json
    expire_in: 30 days
  allow_failure: false
```

---

## Interpreting Results

### Health Status Levels

| Status | Score Range | Description |
|--------|-------------|-------------|
| GREEN | >= 80 | Healthy |
| YELLOW | 60-79 | Warning |
| RED | < 60 | Critical |

### Risk Tiers

| Tier | Meaning | Response |
|------|---------|----------|
| LOW | Healthy | Routine monitoring |
| MEDIUM | Attention needed | Review this week |
| HIGH | Urgent | Review today |
| CRITICAL | Emergency | Immediate action |

### SLO Violation Severity

| Severity | Description | Action |
|----------|-------------|--------|
| low | Informational | Track for improvement |
| medium | Notable issue | Plan remediation |
| high | Significant gap | Prioritize this sprint |
| critical | Severe breach | Immediate escalation |

---

## Troubleshooting

### No Repositories Discovered

**Symptoms:** Exit code 93, "No repositories discovered"

**Causes:**
- Root directory doesn't exist
- No subdirectories with health artifacts
- All repos filtered out

**Solutions:**
```bash
# Check directory structure
ls -la ./org-health

# Verify artifacts exist
find ./org-health -name "health-dashboard.json"

# Try without filter
python -m analytics.run_org_health --root-dir ./org-health --verbose
```

### Config Error

**Symptoms:** Exit code 94, "Failed to load config"

**Causes:**
- Invalid YAML/JSON syntax
- Missing required fields
- Invalid operator or metric

**Solutions:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check JSON config
python -c "import json; json.load(open('config.json'))"

# Use verbose mode for details
python -m analytics.run_org_health --config ./config.yaml --verbose
```

### SLO Violations Unexpected

**Symptoms:** SLOs failing when repos seem healthy

**Causes:**
- Tag mismatch (wrong repos selected)
- Metric aggregation confusion
- Target too strict

**Solutions:**
```bash
# Use dry-run to see what's discovered
python scripts/run_org_health_governance.py --root-dir ./org-health --dry-run

# Check detailed output
python -m analytics.run_org_health --root-dir ./org-health --verbose --json | jq '.slo_results'
```

---

## Best Practices

### 1. Consistent Artifact Structure

Ensure all repos produce artifacts in the same structure:
```
repo-X/
  dashboard/health-dashboard.json
  alerts/alerts.json
  trends/trend-report.json
```

### 2. Meaningful Tags

Use tags to group repos by purpose:
```yaml
repo_tags:
  api-gateway: ["core", "public-facing"]
  auth-service: ["core", "security"]
  analytics: ["data", "internal"]
```

### 3. Layered SLOs

Start with relaxed SLOs and tighten over time:
```yaml
# Phase 1: Basic hygiene
- metric: percent_yellow_or_better
  target: 0.95

# Phase 2: Stricter compliance
- metric: percent_green
  target: 0.80

# Phase 3: Excellence
- metric: percent_green
  target: 0.95
```

### 4. Regular Monitoring

Run org health governance:
- Daily for production repos
- On-demand before major releases
- After significant changes

### 5. Act on Recommendations

Review and act on generated recommendations:
- Critical: Same day
- High: This week
- Medium: This sprint
- Low: Backlog

---

## Summary

The Organization Health Governance Engine provides:

- **Multi-Repo Aggregation**: Single view of org-wide health
- **SLO/SLA Policies**: Configurable compliance tracking
- **Risk Assessment**: Per-repo and org-level risk tiers
- **Recommendations**: Actionable improvement guidance
- **CI/CD Integration**: Exit codes (90-99) for pipeline gates

For related documentation:
- [Repository Health Dashboard Guide](./REPOSITORY_HEALTH_DASHBOARD_GUIDE.md) - Per-repo dashboards
- [Alerting Engine Guide](./ALERTING_ENGINE_GUIDE.md) - Alert rules and dispatch
- [Trend Analyzer Guide](./TREND_ANALYZER_GUIDE.md) - Time-series analysis

---

**Version:** 1.0.0
**Phase:** 14.8 Task 1
**Last Updated:** 2025-01-07

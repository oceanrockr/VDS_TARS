# SLA Reporting & Executive Readiness Dashboard Engine

## Overview

The **SLA Reporting & Executive Readiness Dashboard Engine** is an organization-level SLA intelligence layer that:

- Translates technical signals into executive-grade SLA insights
- Aggregates risk, breaches, and trends across repositories
- Produces board-ready summaries, not raw metrics
- Is fully CI/CD-compatible and audit-friendly

This module is part of **Phase 14.8 - Task 5: SLA Reporting & Executive Readiness** and builds upon:
- Task 1: Org Health Governance Engine (`org-health-report.json`)
- Task 2: Org Alerting & Escalation Engine (`org-alerts.json`)
- Task 3: Multi-Repository Trend Correlation Engine (`trend-correlation-report.json`)
- Task 4: Advanced Temporal Intelligence Engine (`temporal-intelligence-report.json`)

This task answers:

> "Are we meeting our commitments — and if not, where, why, and what is the business impact?"

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Programmatic API](#programmatic-api)
6. [SLA Policy Configuration](#sla-policy-configuration)
7. [Compliance Evaluation](#compliance-evaluation)
8. [Breach Attribution](#breach-attribution)
9. [Executive Readiness Scoring](#executive-readiness-scoring)
10. [Exit Codes](#exit-codes)
11. [Configuration Reference](#configuration-reference)
12. [JSON Schema](#json-schema)
13. [CI/CD Integration](#cicd-integration)
14. [Interpreting Results](#interpreting-results)
15. [Best Practices](#best-practices)

---

## Architecture

```
org-health-report.json (Task 1)    org-alerts.json (Task 2)
         │                                  │
         │    trend-correlation-report.json (Task 3)
         │              │
         │    temporal-intelligence-report.json (Task 4)
         │              │                   │
         └──────────────┼───────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    SLAIntelligenceEngine                                 │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    SLAPolicyLoader                                 │  │
│  │                                                                    │  │
│  │  Load policies from YAML/JSON → Validate → Create SLAPolicy       │  │
│  │  Support: Availability, Reliability, Incident Response, CFR       │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                  SLAComplianceEngine                               │  │
│  │                                                                    │  │
│  │  Evaluate per-policy compliance across windows (7, 30, 90)        │  │
│  │  Status: COMPLIANT | AT_RISK | BREACHED                           │  │
│  │  Calculate trend direction and breach probability                 │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │               SLABreachAttributionEngine                           │  │
│  │                                                                    │  │
│  │  Attribute breaches to: Repos | Clusters | Paths | Alerts         │  │
│  │  Generate root causes with confidence scores                      │  │
│  │  Map to correlation/temporal findings                             │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │               ExecutiveReadinessEngine                             │  │
│  │                                                                    │  │
│  │  Calculate readiness score (0-100)                                │  │
│  │  Determine tier: GREEN | YELLOW | RED                             │  │
│  │  Generate scorecards and risk narrative                           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
   sla-intelligence-report.json + Exit Code (140-149)
```

---

## Installation

### Requirements

- Python 3.10+
- No external dependencies (standard library only)
- Optional: PyYAML for YAML policy files

### File Structure

```
analytics/
├── org_sla_intelligence.py        # Core module (~1,700 LOC)
├── run_org_sla_intelligence.py    # CLI tool (~350 LOC)
├── __init__.py

tests/integration/
├── test_org_sla_intelligence.py   # Test suite (~1,000 LOC)

docs/
├── ORG_SLA_INTELLIGENCE_ENGINE.md     # This documentation
├── PHASE14_8_TASK5_COMPLETION_SUMMARY.md
```

---

## Quick Start

### 1. Basic Analysis

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json
```

### 2. Full Analysis with All Inputs

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --alerts-report ./org-alerts.json \
    --trend-correlation-report ./trend-correlation-report.json \
    --temporal-intelligence-report ./temporal-intelligence-report.json \
    --output ./sla-intelligence-report.json
```

### 3. Custom SLA Policy

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --sla-policy ./custom-sla-policies.yaml \
    --output ./sla-report.json
```

### 4. JSON Output to Stdout

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./report.json \
    --json
```

### 5. CI/CD Mode - Fail on Breach

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./report.json \
    --fail-on-breach
```

### 6. Summary Only (Minimal Output)

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./report.json \
    --summary-only --json
```

---

## CLI Reference

```
usage: run_org_sla_intelligence [-h] --org-report ORG_REPORT
                                 [--alerts-report ALERTS_REPORT]
                                 [--trend-correlation-report REPORT]
                                 [--temporal-intelligence-report REPORT]
                                 [--sla-policy SLA_POLICY]
                                 [--output OUTPUT] [--json] [--summary-only]
                                 [--verbose] [--window WINDOW]
                                 [--fail-on-breach] [--fail-on-at-risk]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--org-report` | Path to org-health-report.json from Task 1 |

### Optional Input Reports

| Option | Description |
|--------|-------------|
| `--alerts-report` | Path to org-alerts.json from Task 2 |
| `--trend-correlation-report` | Path to trend-correlation-report.json from Task 3 |
| `--temporal-intelligence-report` | Path to temporal-intelligence-report.json from Task 4 |
| `--sla-policy` | Path to SLA policy file (YAML or JSON) |

### Output Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Path to write sla-intelligence-report.json |
| `--json` | Output full report as JSON to stdout |
| `--summary-only` | Only output summary statistics |
| `--verbose, -v` | Enable verbose logging |

### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--window` | 7, 30, 90 | Evaluation window size (can specify multiple) |

### CI/CD Options

| Option | Description |
|--------|-------------|
| `--fail-on-breach` | Exit 142 if any SLA is breached |
| `--fail-on-at-risk` | Exit 141 if any SLA is at risk |

---

## Programmatic API

### Basic Usage

```python
from pathlib import Path
from analytics.org_sla_intelligence import (
    SLAIntelligenceEngine,
    SLAIntelligenceConfig,
    SLAThresholds
)

# Configure engine
config = SLAIntelligenceConfig(
    org_report_path=Path("./org-health-report.json"),
    alerts_report_path=Path("./org-alerts.json"),
    temporal_report_path=Path("./temporal-intelligence-report.json"),
    sla_policy_path=Path("./sla-policies.yaml"),
    output_path=Path("./sla-intelligence-report.json"),
    thresholds=SLAThresholds(
        green_threshold=85.0,
        yellow_threshold=65.0
    ),
    evaluation_windows=[7, 30, 90]
)

# Run analysis
engine = SLAIntelligenceEngine(config)
report, exit_code = engine.run()

# Access results
print(f"Executive Readiness: {report.executive_readiness.readiness_score:.1f}")
print(f"Tier: {report.executive_readiness.readiness_tier.value}")
print(f"Compliant SLAs: {report.summary.compliant_slas}")
print(f"At Risk SLAs: {report.summary.at_risk_slas}")
print(f"Breached SLAs: {report.summary.breached_slas}")
```

### Using Individual Components

```python
from analytics.org_sla_intelligence import (
    SLAPolicyLoader,
    SLAComplianceEngine,
    SLABreachAttributionEngine,
    ExecutiveReadinessEngine,
    SLAIntelligenceConfig
)
import json

config = SLAIntelligenceConfig(org_report_path=Path("./report.json"))

# Load org report
with open("./org-health-report.json") as f:
    org_report = json.load(f)

# Load policies
loader = SLAPolicyLoader(config)
policies = loader.load_policies()

# Evaluate compliance
compliance_engine = SLAComplianceEngine(config)
compliance_results = compliance_engine.evaluate_compliance(policies, org_report)

# Attribute breaches
attribution_engine = SLABreachAttributionEngine(config)
breaches = attribution_engine.attribute_breaches(
    compliance_results, policies, org_report
)

# Calculate readiness
readiness_engine = ExecutiveReadinessEngine(config)
readiness = readiness_engine.calculate_readiness(
    compliance_results, breaches, org_report
)
```

---

## SLA Policy Configuration

### Supported SLA Types

| Type | Description | Default Target |
|------|-------------|----------------|
| `availability` | Service availability percentage | 99.0% |
| `reliability` | System reliability score | 80.0 |
| `incident_response` | Time to respond to incidents | 4 hours |
| `change_failure_rate` | Percentage of failed changes | 10% |
| `mttr` | Mean time to recovery | - |
| `deployment_frequency` | How often deployments occur | - |
| `lead_time` | Time from commit to deploy | - |
| `custom` | User-defined metrics | - |

### Policy File Format (YAML)

```yaml
policies:
  - policy_id: sla_availability
    policy_name: "Service Availability SLA"
    sla_type: availability
    description: "Measures overall service availability"
    targets:
      - metric_name: availability_score
        target_value: 99.0
        warning_threshold: 95.0
        breach_threshold: 90.0
        unit: "%"
        higher_is_better: true
    priority: 1
    severity_on_breach: critical
    business_impact: "Direct impact on customer experience and revenue"
    stakeholders:
      - engineering-lead
      - product-manager
    evaluation_windows: [7, 30, 90]
    applies_to_repos: []  # Empty = all repos
    applies_to_org: true

  - policy_id: sla_reliability
    policy_name: "System Reliability SLA"
    sla_type: reliability
    targets:
      - metric_name: health_score
        target_value: 80.0
        warning_threshold: 70.0
        breach_threshold: 60.0
        unit: "score"
        higher_is_better: true
    priority: 2
    severity_on_breach: high
```

### Policy File Format (JSON)

```json
{
  "policies": [
    {
      "policy_id": "sla_availability",
      "policy_name": "Service Availability SLA",
      "sla_type": "availability",
      "targets": [
        {
          "metric_name": "availability_score",
          "target_value": 99.0,
          "warning_threshold": 95.0,
          "breach_threshold": 90.0,
          "unit": "%",
          "higher_is_better": true
        }
      ],
      "priority": 1,
      "severity_on_breach": "critical"
    }
  ]
}
```

### Default Policies

If no policy file is provided, the engine uses these default policies:

1. **Service Availability SLA** (Critical)
   - Target: 99% availability
   - Warning: 95%
   - Breach: 90%

2. **System Reliability SLA** (High)
   - Target: 80 health score
   - Warning: 70
   - Breach: 60

3. **Incident Response SLA** (High)
   - Target: 4 hours response time
   - Warning: 8 hours
   - Breach: 24 hours

4. **Change Failure Rate SLA** (Medium)
   - Target: 10% failure rate
   - Warning: 15%
   - Breach: 25%

---

## Compliance Evaluation

### Status Classification

| Status | Criteria (higher_is_better=true) | Description |
|--------|----------------------------------|-------------|
| COMPLIANT | actual >= target | Meeting SLA |
| AT_RISK | actual >= warning_threshold | Below target but not breached |
| BREACHED | actual < breach_threshold | SLA violated |

For metrics where lower is better (e.g., response time):
- COMPLIANT: actual <= target
- AT_RISK: actual <= warning_threshold
- BREACHED: actual > breach_threshold

### Multi-Window Evaluation

Compliance is evaluated across multiple time windows:

```
Window 7:   [-----7 intervals-----]→ Now
Window 30:  [----------30 intervals----------]→ Now
Window 90:  [-------------------90 intervals-------------------]→ Now
```

The overall status is the **worst** status across all windows.

### Trend Analysis

For each SLA, the engine calculates:
- **Trend Direction**: improving, stable, degrading
- **Trend Confidence**: 0-1 confidence score
- **Breach Probability**: 0-1 estimated probability of breach
- **Days Until Breach**: Projected days until breach (if at_risk)

---

## Breach Attribution

### Attribution Sources

The engine attributes breaches to root causes from multiple sources:

| Source | Type | Description |
|--------|------|-------------|
| Org Report | `repo_degradation` | Repos with low health scores |
| Task 3 | `correlation_cluster` | Correlated repo clusters |
| Task 4 | `propagation_path` | Temporal propagation paths |
| Task 4 | `temporal_anomaly` | Critical temporal anomalies |
| Task 2 | `alert_pattern` | Recurring alert patterns |

### Root Cause Structure

```json
{
  "cause_id": "repo_cause_failing-repo",
  "cause_type": "repo_degradation",
  "title": "Repository Degradation: failing-repo",
  "description": "failing-repo has a health score of 45.0 (risk tier: critical)",
  "confidence_score": 0.9,
  "contribution_percentage": 55.0,
  "evidence": [
    "Health score: 45.0",
    "Risk tier: critical"
  ],
  "related_repos": ["failing-repo"],
  "related_alerts": [],
  "related_correlations": [],
  "related_paths": []
}
```

### Confidence Scoring

Root cause confidence is calculated based on:
- Health score deviation from target
- Correlation strength (for clusters)
- Path confidence (for temporal paths)
- Alert frequency (for alert patterns)

---

## Executive Readiness Scoring

### Readiness Score Formula

```
Readiness Score =
    SLA Compliance Score × 0.40 +
    Trend Health Score × 0.25 +
    Temporal Risk Score × 0.20 +
    Propagation Exposure Score × 0.15
```

### Component Scores

| Component | Weight | Description |
|-----------|--------|-------------|
| SLA Compliance | 40% | Based on compliant/at-risk/breached ratios |
| Trend Health | 25% | Based on improving/stable/declining repos |
| Temporal Risk | 20% | Based on temporal anomalies (inverted) |
| Propagation Exposure | 15% | Based on propagation paths (inverted) |

### Readiness Tiers

| Tier | Score Range | Meaning |
|------|-------------|---------|
| GREEN | >= 80 | All systems go - meeting/exceeding SLAs |
| YELLOW | >= 60 | Caution - some SLAs at risk |
| RED | < 60 | Critical - SLAs breached, action required |

### Risk Outlook

| Outlook | Criteria |
|---------|----------|
| IMPROVING | More SLAs improving than degrading |
| STABLE | Balanced or no significant change |
| DEGRADING | More SLAs degrading than improving |
| CRITICAL | Critical org health status |

---

## Exit Codes

| Code | Constant | Description |
|------|----------|-------------|
| 140 | `EXIT_SLA_SUCCESS` | All SLAs compliant |
| 141 | `EXIT_SLA_AT_RISK` | At-risk SLAs detected |
| 142 | `EXIT_SLA_BREACH` | SLA breach detected |
| 143 | `EXIT_SLA_CONFIG_ERROR` | Configuration error |
| 144 | `EXIT_SLA_PARSE_ERROR` | Failed to parse input report |
| 199 | `EXIT_GENERAL_SLA_ERROR` | General error |

---

## Configuration Reference

### Full Configuration Options

```python
config = SLAIntelligenceConfig(
    # Required
    org_report_path=Path("./org-health-report.json"),

    # Optional inputs
    alerts_report_path=Path("./org-alerts.json"),
    correlation_report_path=Path("./trend-correlation-report.json"),
    temporal_report_path=Path("./temporal-intelligence-report.json"),
    sla_policy_path=Path("./sla-policies.yaml"),

    # Output
    output_path=Path("./sla-intelligence-report.json"),

    # Thresholds
    thresholds=SLAThresholds(
        at_risk_percentage=90.0,
        breach_percentage=80.0,
        min_confidence_score=0.3,
        min_contribution_percentage=10.0,
        sla_compliance_weight=0.40,
        trend_health_weight=0.25,
        temporal_risk_weight=0.20,
        propagation_exposure_weight=0.15,
        green_threshold=80.0,
        yellow_threshold=60.0,
        improving_threshold=5.0,
        degrading_threshold=-5.0
    ),

    # Evaluation windows
    evaluation_windows=[7, 30, 90],

    # Behavior
    verbose=False,
    summary_only=False,

    # CI/CD
    fail_on_breach=False,
    fail_on_at_risk=False
)
```

---

## JSON Schema

### Output Report Schema

```json
{
  "report_id": "sla_intelligence_20250108_120000",
  "generated_at": "2025-01-08T12:00:00.000Z",
  "report_version": "1.0.0",

  "org_report_path": "./org-health-report.json",
  "alerts_report_path": "./org-alerts.json",
  "sla_policy_path": "./sla-policies.yaml",

  "summary": {
    "total_slas_evaluated": 4,
    "compliant_slas": 2,
    "at_risk_slas": 1,
    "breached_slas": 1,
    "total_breaches": 1,
    "critical_breaches": 1,
    "overall_compliance_rate": 50.0,
    "avg_sla_health": 75.0,
    "executive_readiness_score": 65.0,
    "readiness_tier": "yellow"
  },

  "executive_readiness": {
    "readiness_score": 65.0,
    "readiness_tier": "yellow",
    "sla_compliance_score": 75.0,
    "trend_health_score": 70.0,
    "temporal_risk_score": 80.0,
    "propagation_exposure_score": 85.0,
    "risk_outlook": "stable",
    "outlook_confidence": 0.7,
    "compliant_slas": 2,
    "at_risk_slas": 1,
    "breached_slas": 1,
    "executive_summary": "Some service level commitments require attention...",
    "key_concerns": ["Availability SLA is breached"],
    "positive_highlights": ["2 SLAs fully compliant"]
  },

  "scorecards": [
    {
      "policy_id": "sla_availability",
      "policy_name": "Service Availability SLA",
      "sla_type": "availability",
      "status": "breached",
      "status_icon": "red_circle",
      "current_value": 85.0,
      "target_value": 99.0,
      "unit": "%",
      "trend_indicator": "arrow_down",
      "trend_description": "Trend: degrading",
      "risk_level": "high",
      "plain_english_status": "Target missed - action required"
    }
  ],

  "compliance_results": [
    {
      "policy_id": "sla_availability",
      "policy_name": "Service Availability SLA",
      "sla_type": "availability",
      "overall_status": "breached",
      "compliance_percentage": 33.3,
      "window_results": [
        {
          "window_size": 7,
          "window_label": "7-interval",
          "actual_value": 85.0,
          "target_value": 99.0,
          "variance": -14.0,
          "status": "breached"
        }
      ],
      "trend_direction": "degrading",
      "affected_repos": ["failing-repo"],
      "breach_probability": 0.8,
      "days_until_breach": null
    }
  ],

  "breaches": [
    {
      "breach_id": "breach_20250108_0001",
      "policy_id": "sla_availability",
      "policy_name": "Service Availability SLA",
      "severity": "critical",
      "breach_timestamp": "2025-01-08T12:00:00",
      "actual_value": 85.0,
      "target_value": 99.0,
      "breach_magnitude": 14.0,
      "status": "active",
      "root_causes": [
        {
          "cause_id": "repo_cause_failing-repo",
          "cause_type": "repo_degradation",
          "title": "Repository Degradation: failing-repo",
          "confidence_score": 0.9,
          "contribution_percentage": 55.0
        }
      ],
      "primary_cause": "Repository Degradation: failing-repo",
      "recommended_actions": [
        "IMMEDIATE: Escalate to leadership",
        "Prioritize remediation of failing-repo"
      ]
    }
  ],

  "risk_narrative": {
    "headline": "Service Levels: Attention Required",
    "summary_paragraph": "Some service level commitments require attention...",
    "current_status": "Readiness Score: 65/100 (YELLOW)",
    "key_risks": ["Availability SLA is breached"],
    "mitigating_factors": ["2 SLAs fully compliant"],
    "recommended_focus_areas": ["Restore breached SLAs to compliance"]
  },

  "recommendations": [
    {
      "id": "rec_001",
      "priority": "critical",
      "title": "Address Critical SLA Breaches",
      "message": "1 critical SLA breach requires immediate attention",
      "actions": ["Escalate to leadership", "Investigate root causes"],
      "affected_slas": ["Service Availability SLA"]
    }
  ],

  "org_health_status": "yellow",
  "org_health_score": 85.0,
  "total_repos": 5,
  "analysis_duration_ms": 150.5
}
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: SLA Intelligence Analysis

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Run Org Health Engine (Task 1)
        run: |
          python -m analytics.run_org_health \
            --root ./repos \
            --output ./org-health-report.json

      - name: Run SLA Intelligence Analysis
        run: |
          python -m analytics.run_org_sla_intelligence \
            --org-report ./org-health-report.json \
            --output ./sla-intelligence-report.json \
            --fail-on-breach

      - name: Upload Report
        uses: actions/upload-artifact@v4
        with:
          name: sla-intelligence-report
          path: sla-intelligence-report.json
```

### GitLab CI

```yaml
sla-intelligence:
  stage: analyze
  image: python:3.11
  script:
    - python -m analytics.run_org_health --root ./repos --output ./org-health-report.json
    - python -m analytics.run_org_sla_intelligence --org-report ./org-health-report.json --output ./sla-report.json --fail-on-breach
  artifacts:
    paths:
      - sla-report.json
    expire_in: 30 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

### Exit Code Handling

```bash
#!/bin/bash

python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --output ./sla-report.json

EXIT_CODE=$?

case $EXIT_CODE in
    140)
        echo "SUCCESS: All SLAs compliant"
        ;;
    141)
        echo "WARNING: SLAs at risk - review recommended"
        ;;
    142)
        echo "CRITICAL: SLA breach detected - immediate action required"
        exit 1
        ;;
    143)
        echo "ERROR: Configuration error"
        exit 1
        ;;
    144)
        echo "ERROR: Failed to parse input"
        exit 1
        ;;
    *)
        echo "ERROR: Unexpected error"
        exit 1
        ;;
esac
```

---

## Interpreting Results

### Reading the Executive Summary

```
================================================================================
SLA REPORTING & EXECUTIVE READINESS DASHBOARD
================================================================================

[YELLOW] EXECUTIVE READINESS: 65/100
----------------------------------------
Risk Outlook: stable

----------------------------------------
SLA COMPLIANCE SUMMARY
----------------------------------------
Total SLAs Evaluated: 4
  - Compliant: 2
  - At Risk: 1
  - Breached: 1
Overall Compliance Rate: 50.0%
```

**Key metrics to watch:**
- **Executive Readiness Score**: Single number for board reporting
- **Readiness Tier**: Traffic light status (GREEN/YELLOW/RED)
- **Breached SLAs**: Require immediate attention
- **At Risk SLAs**: May breach soon if trends continue

### Reading Scorecards

Scorecards provide at-a-glance SLA status:

| Icon | Status | Action |
|------|--------|--------|
| [OK] | Compliant | Maintain |
| [!] | At Risk | Monitor/Prevent |
| [X] | Breached | Immediate action |

### Breach Attribution

When reviewing breaches, focus on:
1. **Primary Cause**: Most likely root cause
2. **Confidence Score**: How certain is the attribution
3. **Contribution Percentage**: How much this cause contributes
4. **Related Repos**: Which repos are involved

---

## Best Practices

### 1. Regular Analysis Schedule

Run SLA intelligence analysis:
- Daily for production environments
- Weekly for development environments
- After major deployments
- After incident resolution

### 2. Custom SLA Policies

Define policies that match your business commitments:
- External customer SLAs
- Internal team objectives
- Platform reliability targets
- DORA metrics (if applicable)

### 3. Integrate with Upstream Tasks

For best results, run the full pipeline:
```bash
# Task 1: Org Health
python -m analytics.run_org_health --output ./org-health-report.json

# Task 2: Alerting
python -m analytics.run_org_alerts --output ./org-alerts.json

# Task 3: Correlation
python -m analytics.run_org_trend_correlation --output ./correlation.json

# Task 4: Temporal Intelligence
python -m analytics.run_org_temporal_intelligence --output ./temporal.json

# Task 5: SLA Intelligence (uses all above)
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --alerts-report ./org-alerts.json \
    --trend-correlation-report ./correlation.json \
    --temporal-intelligence-report ./temporal.json \
    --output ./sla-report.json
```

### 4. Monitor Trends

Watch for:
- SLAs moving from COMPLIANT to AT_RISK
- Increasing breach probability
- Degrading trend direction
- Rising temporal risk exposure

### 5. Action on Breaches

When breaches occur:
1. **Immediate**: Address primary root cause
2. **Short-term**: Implement fixes for contributing factors
3. **Long-term**: Review SLA targets and architecture

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-08 | Initial release - Phase 14.8 Task 5 |

---

## Related Documentation

- [Org Health Governance Engine](./ORG_HEALTH_GOVERNANCE_GUIDE.md) - Task 1
- [Org Alerting & Escalation Engine](./ORG_ALERTING_AND_ESCALATION_ENGINE.md) - Task 2
- [Multi-Repository Trend Correlation Engine](./ORG_TREND_CORRELATION_ENGINE.md) - Task 3
- [Temporal Intelligence Engine](./ORG_TEMPORAL_INTELLIGENCE_ENGINE.md) - Task 4
- [Phase 14.8 Task 5 Completion Summary](./PHASE14_8_TASK5_COMPLETION_SUMMARY.md)

# Multi-Repository Trend Correlation Engine

## Overview

The **Multi-Repository Trend Correlation Engine** is a cross-repository analytics component that analyzes trend patterns across multiple repositories to identify:

- **Synchronized Declines**: Multiple repos declining together
- **Correlated Metric Movements**: Repos with similar health trajectories
- **Clustered Risk Escalations**: Groups of correlated high-risk repos
- **Predictive Trend Indicators**: Leading indicators that precede changes in other repos

This module is part of **Phase 14.8 - Operations Excellence** and consumes output from Task 1 (Org Health Governance Engine).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Programmatic API](#programmatic-api)
6. [Correlation Analysis](#correlation-analysis)
7. [Cluster Detection](#cluster-detection)
8. [Anomaly Detection](#anomaly-detection)
9. [Exit Codes](#exit-codes)
10. [Configuration Reference](#configuration-reference)
11. [JSON Schema](#json-schema)
12. [CI/CD Integration](#cicd-integration)
13. [Interpreting Results](#interpreting-results)
14. [Troubleshooting](#troubleshooting)
15. [Best Practices](#best-practices)

---

## Architecture

```
org-health-report.json (Task 1)
         │
         ▼
┌──────────────────────────────────────────────────┐
│         TrendCorrelationEngine                   │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │             TrendLoader                    │  │
│  │                                            │  │
│  │  Load org report → Extract repo series    │  │
│  └─────────────────┬──────────────────────────┘  │
│                    │                             │
│                    ▼                             │
│  ┌────────────────────────────────────────────┐  │
│  │      CorrelationMatrixBuilder              │  │
│  │                                            │  │
│  │  Pairwise Pearson/Spearman correlations   │  │
│  │  Classify: positive/negative/synchronized │  │
│  └─────────────────┬──────────────────────────┘  │
│                    │                             │
│                    ▼                             │
│  ┌────────────────────────────────────────────┐  │
│  │          ClusterDetector                   │  │
│  │                                            │  │
│  │  Connected components → Risk clusters     │  │
│  └─────────────────┬──────────────────────────┘  │
│                    │                             │
│                    ▼                             │
│  ┌────────────────────────────────────────────┐  │
│  │          AnomalyDetector                   │  │
│  │                                            │  │
│  │  Synchronized decline │ Risk clusters      │  │
│  │  Leading indicators │ Volatility          │  │
│  └─────────────────────────────────────────────┘  │
│                                                  │
└──────────────────────────────────────────────────┘
         │
         ▼
   trend-correlation-report.json + Exit Code (120-129)
```

---

## Installation

### Requirements

- Python 3.10+
- No external dependencies (standard library only)

### File Structure

```
analytics/
├── org_trend_correlation.py    # Core module (~1,500 LOC)
├── run_org_trend_correlation.py # CLI tool (~300 LOC)
├── __init__.py

tests/integration/
├── test_org_trend_correlation.py  # Test suite (~900 LOC)

docs/
├── ORG_TREND_CORRELATION_ENGINE.md  # This documentation
```

---

## Quick Start

### 1. Basic Analysis

```bash
python -m analytics.run_org_trend_correlation \
    --org-report ./org-health-report.json
```

### 2. Save Output to File

```bash
python -m analytics.run_org_trend_correlation \
    --org-report ./org-health-report.json \
    --output ./trend-correlation-report.json
```

### 3. JSON Output to Stdout

```bash
python -m analytics.run_org_trend_correlation \
    --org-report ./report.json \
    --json
```

### 4. Custom Correlation Threshold

```bash
python -m analytics.run_org_trend_correlation \
    --org-report ./report.json \
    --min-correlation-threshold 0.7
```

### 5. CI/CD Mode - Fail on Critical

```bash
python -m analytics.run_org_trend_correlation \
    --org-report ./report.json \
    --fail-on-critical
```

### 6. Summary Only (Minimal Output)

```bash
python -m analytics.run_org_trend_correlation \
    --org-report ./report.json \
    --summary-only --json
```

### 7. Skip Specific Analysis Steps

```bash
python -m analytics.run_org_trend_correlation \
    --org-report ./report.json \
    --skip-clusters \
    --skip-anomalies
```

---

## CLI Reference

```
usage: run_org_trend_correlation [-h] --org-report ORG_REPORT
                                  [--output OUTPUT] [--json] [--summary-only]
                                  [--verbose] [--min-correlation-threshold THRESHOLD]
                                  [--min-cluster-size SIZE]
                                  [--synchronized-decline-threshold THRESHOLD]
                                  [--skip-clusters] [--skip-anomalies]
                                  [--skip-leading-indicators]
                                  [--fail-on-critical] [--fail-on-any-correlations]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--org-report` | Path to org-health-report.json from Task 1 |

### Output Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Path to write trend-correlation-report.json |
| `--json` | Output full report as JSON to stdout |
| `--summary-only` | Only output summary statistics |
| `--verbose, -v` | Enable verbose logging |

### Threshold Options

| Option | Default | Description |
|--------|---------|-------------|
| `--min-correlation-threshold` | 0.5 | Minimum correlation coefficient for significance |
| `--min-cluster-size` | 2 | Minimum repos to form a cluster |
| `--synchronized-decline-threshold` | 0.20 | Minimum % declining to trigger anomaly |

### Analysis Options

| Option | Description |
|--------|-------------|
| `--skip-clusters` | Skip cluster detection |
| `--skip-anomalies` | Skip anomaly detection |
| `--skip-leading-indicators` | Skip leading indicator detection |

### CI/CD Options

| Option | Description |
|--------|-------------|
| `--fail-on-critical` | Exit 122 if critical anomalies detected |
| `--fail-on-any-correlations` | Exit 121 if any correlations found |

---

## Programmatic API

### Basic Usage

```python
from pathlib import Path
from analytics.org_trend_correlation import (
    TrendCorrelationEngine,
    TrendCorrelationConfig,
    CorrelationThresholds
)

# Configure engine
config = TrendCorrelationConfig(
    org_report_path=Path("./org-health-report.json"),
    output_path=Path("./trend-correlation-report.json"),
    thresholds=CorrelationThresholds(
        min_correlation_strong=0.7,
        min_cluster_size=3
    ),
    compute_clusters=True,
    detect_anomalies=True
)

# Run analysis
engine = TrendCorrelationEngine(config)
report, exit_code = engine.run()

# Access results
print(f"Repos analyzed: {report.total_repos}")
print(f"Significant correlations: {report.summary.significant_correlations}")
print(f"Clusters found: {report.summary.total_clusters}")
print(f"Anomalies detected: {report.summary.total_anomalies}")
```

### Using Individual Components

```python
from analytics.org_trend_correlation import (
    TrendLoader,
    CorrelationMatrixBuilder,
    ClusterDetector,
    AnomalyDetector,
    TrendCorrelationConfig
)

config = TrendCorrelationConfig(org_report_path=Path("./report.json"))

# Load trend data
loader = TrendLoader(config)
org_report = loader.load_org_report()
repo_series = loader.extract_trend_series()

# Build correlation matrix
builder = CorrelationMatrixBuilder(config)
matrix = builder.build_matrix(repo_series)
correlations = builder.get_significant_correlations()

# Detect clusters
detector = ClusterDetector(config)
clusters = detector.detect_clusters(repo_series, matrix, correlations)

# Detect anomalies
anomaly_detector = AnomalyDetector(config)
anomalies = anomaly_detector.detect_anomalies(repo_series, correlations, clusters)
```

---

## Correlation Analysis

### Correlation Types

| Type | Description | Detection |
|------|-------------|-----------|
| `POSITIVE` | Repos trend in same direction | Pearson > 0.5 |
| `NEGATIVE` | Repos trend in opposite directions | Pearson < -0.5 |
| `SYNCHRONIZED_DECLINE` | Both declining together | Positive + shared decline periods |
| `SYNCHRONIZED_IMPROVEMENT` | Both improving together | Positive + shared improvement periods |
| `SHARED_VOLATILITY` | Similar volatility patterns | High volatility similarity ratio |
| `ISSUE_SPIKE_CORRELATED` | Issue counts correlated | Issue correlation > threshold |

### Correlation Strength

| Strength | Coefficient Range |
|----------|-------------------|
| Strong | |r| ≥ 0.7 |
| Moderate | 0.5 ≤ |r| < 0.7 |
| Weak | 0.3 ≤ |r| < 0.5 |
| None | |r| < 0.3 |

### Correlation Matrix

The engine builds a full NxN correlation matrix where:
- Diagonal elements are 1.0 (self-correlation)
- Off-diagonal elements are Pearson coefficients
- Matrix is symmetric

---

## Cluster Detection

### Algorithm

1. Build adjacency graph from significant correlations
2. Find connected components using BFS
3. Filter components by minimum cluster size
4. Compute cluster metrics

### Cluster Properties

| Property | Description |
|----------|-------------|
| `repo_ids` | List of repos in cluster |
| `avg_internal_correlation` | Mean correlation within cluster |
| `cluster_density` | Ratio of strong correlations |
| `dominant_trend` | Most common trend direction |
| `is_risk_cluster` | True if majority high-risk |
| `cluster_risk_tier` | Overall cluster risk level |

### Risk Cluster Detection

A cluster is marked as a **risk cluster** when:
- ≥ 50% of member repos are HIGH or CRITICAL risk
- Majority of repos are declining

---

## Anomaly Detection

### Anomaly Types

| Type | Description | Severity |
|------|-------------|----------|
| `SYNCHRONIZED_DECLINE` | Multiple repos declining together | MEDIUM-CRITICAL |
| `EMERGING_RISK_CLUSTER` | Correlated high-risk repos | HIGH-CRITICAL |
| `LEADING_INDICATOR` | One repo predicts another | MEDIUM |
| `SHARED_VOLATILITY` | Correlated high volatility | MEDIUM |
| `CORRELATED_ISSUE_SPIKE` | Issue counts correlated | HIGH |

### Severity Levels

| Severity | Numeric | Description |
|----------|---------|-------------|
| LOW | 0 | Informational, no action required |
| MEDIUM | 1 | Monitor, may require attention |
| HIGH | 2 | Action required soon |
| CRITICAL | 3 | Immediate action required |

### Synchronized Decline Thresholds

| % Repos Declining | Severity |
|-------------------|----------|
| ≥ 40% | CRITICAL |
| ≥ 30% | HIGH |
| ≥ 20% | MEDIUM |

---

## Exit Codes

| Code | Constant | Description |
|------|----------|-------------|
| 120 | `EXIT_CORRELATION_SUCCESS` | No concerning correlations |
| 121 | `EXIT_CORRELATIONS_FOUND` | Correlations found (non-critical) |
| 122 | `EXIT_CRITICAL_ANOMALY` | Critical cross-repo anomaly |
| 123 | `EXIT_CORRELATION_CONFIG_ERROR` | Configuration error |
| 124 | `EXIT_CORRELATION_PARSE_ERROR` | Failed to parse org report |
| 199 | `EXIT_GENERAL_CORRELATION_ERROR` | General error |

---

## Configuration Reference

### Full YAML Configuration

```yaml
# Correlation thresholds
thresholds:
  # Correlation coefficient thresholds
  min_correlation_weak: 0.3
  min_correlation_moderate: 0.5
  min_correlation_strong: 0.7
  significance_threshold: 0.5

  # Clustering
  min_cluster_correlation: 0.6
  min_cluster_size: 2
  max_cluster_size: 50

  # Anomaly detection
  synchronized_decline_threshold: 0.20
  synchronized_decline_min_repos: 3
  volatility_similarity_threshold: 0.8
  issue_spike_correlation_threshold: 0.6

  # Leading indicators
  leading_indicator_lag: 1
  leading_indicator_correlation: 0.7

# Analysis options
compute_clusters: true
detect_anomalies: true
compute_leading_indicators: true

# Output options
verbose: false
summary_only: false

# CI/CD options
fail_on_critical_anomaly: false
fail_on_any_correlations: false
```

---

## JSON Schema

### Output Report Schema

```json
{
  "report_id": "trend_correlation_20250107_120000",
  "generated_at": "2025-01-07T12:00:00.000Z",
  "org_report_path": "./org-health-report.json",

  "summary": {
    "total_repo_pairs": 10,
    "significant_correlations": 3,
    "positive_correlations": 2,
    "negative_correlations": 1,
    "synchronized_decline_pairs": 1,
    "total_clusters": 1,
    "risk_clusters": 1,
    "total_anomalies": 2,
    "critical_anomalies": 1,
    "predictive_indicators": 1,
    "avg_inter_repo_correlation": 0.65,
    "correlation_density": 0.30
  },

  "correlations": [
    {
      "repo_a_id": "repo-gamma",
      "repo_b_id": "repo-delta",
      "pearson_coefficient": 0.85,
      "spearman_coefficient": 0.82,
      "correlation_type": "synchronized_decline",
      "correlation_strength": "strong",
      "is_significant": true,
      "shared_decline_periods": 2
    }
  ],

  "clusters": [
    {
      "cluster_id": "cluster_001",
      "cluster_name": "Correlation Cluster 1",
      "repo_ids": ["repo-gamma", "repo-delta", "repo-epsilon"],
      "repo_count": 3,
      "avg_internal_correlation": 0.78,
      "dominant_trend": "declining",
      "is_risk_cluster": true
    }
  ],

  "anomalies": [
    {
      "anomaly_id": "anomaly_001",
      "anomaly_type": "synchronized_decline",
      "severity": "critical",
      "title": "Synchronized Decline Across Repositories",
      "message": "3 repos (60%) exhibit simultaneous declining trends",
      "affected_repos": ["repo-gamma", "repo-delta", "repo-epsilon"],
      "affected_count": 3,
      "is_predictive": false,
      "recommended_actions": [
        "Investigate common factors across declining repos",
        "Review recent org-wide changes"
      ]
    }
  ],

  "predictive_indicators": [
    {
      "anomaly_id": "anomaly_002",
      "type": "leading_indicator",
      "affected_repos": ["repo-gamma", "repo-delta"],
      "predicted_impact": "Changes in repo-gamma may predict future changes in repo-delta",
      "confidence": 0.7
    }
  ],

  "recommendations": [
    {
      "id": "rec_001",
      "priority": "critical",
      "title": "Address Critical Cross-Repo Anomalies",
      "message": "1 critical anomaly requiring immediate attention"
    }
  ],

  "org_health_status": "yellow",
  "org_health_score": 72.5,
  "total_repos": 5,
  "analysis_duration_ms": 150.5
}
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Trend Correlation Analysis

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

      - name: Run Org Health Engine
        run: |
          python -m analytics.run_org_health \
            --root ./org-health \
            --output ./org-health-report.json

      - name: Run Trend Correlation Analysis
        run: |
          python -m analytics.run_org_trend_correlation \
            --org-report ./org-health-report.json \
            --output ./trend-correlation-report.json \
            --fail-on-critical

      - name: Upload Report
        uses: actions/upload-artifact@v4
        with:
          name: correlation-report
          path: trend-correlation-report.json
```

### GitLab CI

```yaml
trend-correlation:
  stage: analyze
  image: python:3.11
  script:
    - python -m analytics.run_org_health --root ./org-health --output ./org-health-report.json
    - python -m analytics.run_org_trend_correlation --org-report ./org-health-report.json --output ./trend-correlation-report.json --fail-on-critical
  artifacts:
    paths:
      - trend-correlation-report.json
    expire_in: 30 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

---

## Interpreting Results

### Reading the Summary

```
================================================================================
MULTI-REPOSITORY TREND CORRELATION ANALYSIS
================================================================================

Org Health: YELLOW | Score: 72.5
Repos Analyzed: 5
Analysis Time: 150.5ms

----------------------------------------
CORRELATION SUMMARY
----------------------------------------
Total Repo Pairs: 10
Significant Correlations: 3
  - Positive: 2
  - Negative: 1
  - Synchronized Decline: 1
Correlation Density: 30.0%
```

**Key metrics to watch:**
- **Correlation Density**: High density (>50%) indicates org-wide patterns
- **Synchronized Decline Pairs**: Indicates shared risk factors
- **Risk Clusters**: Groups of repos requiring coordinated remediation

### Anomaly Prioritization

1. **CRITICAL**: Immediate action required
   - Synchronized decline >40% of repos
   - Large risk clusters (5+ repos)

2. **HIGH**: Action required within 24-48 hours
   - Emerging risk clusters
   - Correlated issue spikes

3. **MEDIUM**: Monitor and plan remediation
   - Leading indicators
   - Shared volatility patterns

4. **LOW**: Informational, track for trends

---

## Troubleshooting

### Common Issues

**Issue: No correlations found**
- Check that repos have sufficient trend history (≥2 data points)
- Lower `--min-correlation-threshold` to 0.3
- Verify repos have varying scores (constant scores = 0 correlation)

**Issue: Parse error (124)**
- Verify org-health-report.json exists and is valid JSON
- Check required fields: `repositories`, `repos_loaded`
- Run org health engine first

**Issue: High memory usage with large orgs**
- Use `--skip-clusters` for initial analysis
- Increase `--min-correlation-threshold` to reduce matrix size

**Issue: Too many false positive anomalies**
- Increase `--synchronized-decline-threshold` to 0.30
- Increase `--min-cluster-size` to 3

---

## Best Practices

### 1. Regular Analysis Schedule

Run correlation analysis:
- Daily for active orgs (100+ repos)
- After major deployments
- When org health score drops

### 2. Threshold Tuning

Start with defaults, then tune:
```bash
# More sensitive (catches more patterns)
--min-correlation-threshold 0.4
--synchronized-decline-threshold 0.15

# Less sensitive (fewer false positives)
--min-correlation-threshold 0.7
--synchronized-decline-threshold 0.30
```

### 3. Cluster Investigation Workflow

When risk clusters are detected:
1. Identify common dependencies
2. Check shared configurations
3. Review recent coordinated changes
4. Plan cluster-wide remediation

### 4. Leading Indicator Monitoring

When leading indicators are found:
1. Set up alerts on leader repos
2. Use leader trends for prediction
3. Investigate causal relationships

### 5. Integration with Alerting

Chain with Task 2 alerting:
```bash
# Run correlation first
python -m analytics.run_org_trend_correlation --org-report ./report.json --output ./correlation.json

# Then alerting (can incorporate correlation findings)
python -m analytics.run_org_alerts --org-report ./report.json
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-07 | Initial release - Phase 14.8 Task 3 |

---

## Related Documentation

- [Org Health Governance Engine](./ORG_HEALTH_GOVERNANCE_GUIDE.md) - Task 1
- [Org Alerting & Escalation Engine](./ORG_ALERTING_AND_ESCALATION_ENGINE.md) - Task 2
- [Phase 14.8 Task 3 Completion Summary](./PHASE14_8_TASK3_COMPLETION_SUMMARY.md)

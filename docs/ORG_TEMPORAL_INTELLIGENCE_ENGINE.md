# Advanced Correlation & Temporal Intelligence Engine

## Overview

The **Advanced Correlation & Temporal Intelligence Engine** is a time-aware, direction-aware analytics component that extends cross-repository trend analysis with causality-oriented intelligence. It answers the critical question:

> "Not just what is correlated — but who leads, who follows, and how fast risk propagates across the organization."

This module is part of **Phase 14.8 - Task 4: Advanced Correlation & Temporal Intelligence** and builds upon:
- Task 1: Org Health Governance Engine (`org-health-report.json`)
- Task 3: Multi-Repository Trend Correlation Engine (`trend-correlation-report.json`)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Programmatic API](#programmatic-api)
6. [Lagged Correlation Analysis](#lagged-correlation-analysis)
7. [Influence Scoring](#influence-scoring)
8. [Propagation Path Detection](#propagation-path-detection)
9. [Temporal Anomaly Detection](#temporal-anomaly-detection)
10. [Exit Codes](#exit-codes)
11. [Configuration Reference](#configuration-reference)
12. [JSON Schema](#json-schema)
13. [CI/CD Integration](#cicd-integration)
14. [Interpreting Results](#interpreting-results)
15. [Troubleshooting](#troubleshooting)
16. [Best Practices](#best-practices)

---

## Architecture

```
org-health-report.json (Task 1)      trend-correlation-report.json (Task 3)
         │                                        │ (optional)
         └─────────────────┬──────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  TemporalIntelligenceEngine                              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Data Loader                                     │  │
│  │                                                                    │  │
│  │  Load org report → Extract time series → Validate data            │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │               LaggedCorrelationEngine                              │  │
│  │                                                                    │  │
│  │  Compute correlations at lags: -3, -2, -1, 0, +1, +2, +3          │  │
│  │  Identify optimal lag per pair                                    │  │
│  │  Determine leader → follower relationships                        │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │               InfluenceScoringEngine                               │  │
│  │                                                                    │  │
│  │  Score repos by leadership behavior (0-100)                       │  │
│  │  Rank by systemic importance                                      │  │
│  │  Identify early warning potential                                 │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │              PropagationGraphBuilder                               │  │
│  │                                                                    │  │
│  │  Build directed graph of influence                                │  │
│  │  Detect propagation paths (A → B → C)                             │  │
│  │  Classify path types (linear, branching)                          │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │             TemporalAnomalyDetector                                │  │
│  │                                                                    │  │
│  │  Rapid propagation │ Leader deterioration                         │  │
│  │  Systemic propagation │ Synchronized lag patterns                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
   temporal-intelligence-report.json + Exit Code (130-139)
```

---

## Installation

### Requirements

- Python 3.10+
- No external dependencies (standard library only)

### File Structure

```
analytics/
├── org_temporal_intelligence.py     # Core module (~1,600 LOC)
├── run_org_temporal_intelligence.py # CLI tool (~350 LOC)
├── __init__.py

tests/integration/
├── test_org_temporal_intelligence.py  # Test suite (~1,000 LOC)

docs/
├── ORG_TEMPORAL_INTELLIGENCE_ENGINE.md  # This documentation
├── PHASE14_8_TASK4_COMPLETION_SUMMARY.md
```

---

## Quick Start

### 1. Basic Analysis

```bash
python -m analytics.run_org_temporal_intelligence \
    --org-report ./org-health-report.json
```

### 2. Save Output to File

```bash
python -m analytics.run_org_temporal_intelligence \
    --org-report ./org-health-report.json \
    --output ./temporal-intelligence-report.json
```

### 3. JSON Output to Stdout

```bash
python -m analytics.run_org_temporal_intelligence \
    --org-report ./report.json \
    --json
```

### 4. Custom Lag Window

```bash
python -m analytics.run_org_temporal_intelligence \
    --org-report ./report.json \
    --max-lag 5
```

### 5. CI/CD Mode - Fail on Critical

```bash
python -m analytics.run_org_temporal_intelligence \
    --org-report ./report.json \
    --fail-on-critical
```

### 6. Summary Only (Minimal Output)

```bash
python -m analytics.run_org_temporal_intelligence \
    --org-report ./report.json \
    --summary-only --json
```

---

## CLI Reference

```
usage: run_org_temporal_intelligence [-h] --org-report ORG_REPORT
                                      [--trend-correlation-report REPORT]
                                      [--output OUTPUT] [--json] [--summary-only]
                                      [--verbose] [--max-lag LAG]
                                      [--min-influence-score SCORE]
                                      [--min-correlation CORR]
                                      [--min-causality-score SCORE]
                                      [--skip-lagged-correlations]
                                      [--skip-influence-scores]
                                      [--skip-propagation-paths]
                                      [--skip-anomalies]
                                      [--fail-on-critical]
                                      [--fail-on-any-temporal-patterns]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--org-report` | Path to org-health-report.json from Task 1 |

### Optional Input

| Option | Description |
|--------|-------------|
| `--trend-correlation-report` | Path to trend-correlation-report.json from Task 3 |

### Output Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Path to write temporal-intelligence-report.json |
| `--json` | Output full report as JSON to stdout |
| `--summary-only` | Only output summary statistics |
| `--verbose, -v` | Enable verbose logging |

### Threshold Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-lag` | 3 | Maximum lag to analyze (±max_lag intervals) |
| `--min-influence-score` | 30.0 | Minimum influence score to consider significant |
| `--min-correlation` | 0.5 | Minimum lagged correlation coefficient |
| `--min-causality-score` | 0.4 | Minimum causality score for propagation edges |

### Analysis Options

| Option | Description |
|--------|-------------|
| `--skip-lagged-correlations` | Skip lagged correlation computation |
| `--skip-influence-scores` | Skip influence score computation |
| `--skip-propagation-paths` | Skip propagation path detection |
| `--skip-anomalies` | Skip temporal anomaly detection |

### CI/CD Options

| Option | Description |
|--------|-------------|
| `--fail-on-critical` | Exit 132 if critical propagation risks detected |
| `--fail-on-any-temporal-patterns` | Exit 131 if any temporal patterns detected |

---

## Programmatic API

### Basic Usage

```python
from pathlib import Path
from analytics.org_temporal_intelligence import (
    TemporalIntelligenceEngine,
    TemporalIntelligenceConfig,
    TemporalThresholds
)

# Configure engine
config = TemporalIntelligenceConfig(
    org_report_path=Path("./org-health-report.json"),
    output_path=Path("./temporal-intelligence-report.json"),
    thresholds=TemporalThresholds(
        max_lag=5,
        min_influence_score=40.0
    ),
    compute_lagged_correlations=True,
    compute_influence_scores=True,
    compute_propagation_paths=True,
    detect_temporal_anomalies=True
)

# Run analysis
engine = TemporalIntelligenceEngine(config)
report, exit_code = engine.run()

# Access results
print(f"Repos analyzed: {report.total_repos}")
print(f"Lagged correlations: {report.summary.significant_lagged_correlations}")
print(f"Leader repos: {report.summary.leader_repos}")
print(f"Propagation paths: {report.summary.propagation_paths_detected}")
print(f"Anomalies: {report.summary.total_anomalies}")
```

### Using Individual Components

```python
from analytics.org_temporal_intelligence import (
    LaggedCorrelationEngine,
    InfluenceScoringEngine,
    PropagationGraphBuilder,
    TemporalAnomalyDetector,
    TemporalIntelligenceConfig
)

config = TemporalIntelligenceConfig(org_report_path=Path("./report.json"))

# Extract time series (after loading org report)
repo_series = {...}  # Dict[str, RepoTimeSeries]

# Compute lagged correlations
lag_engine = LaggedCorrelationEngine(config)
lagged_correlations = lag_engine.compute_all_lagged_correlations(repo_series)

# Compute influence scores
influence_engine = InfluenceScoringEngine(config)
influence_scores = influence_engine.compute_influence_scores(repo_series, lagged_correlations)

# Build propagation graph
builder = PropagationGraphBuilder(config)
graph = builder.build_propagation_graph(lagged_correlations, influence_engine.get_influence_scores())
paths = builder.detect_propagation_paths(repo_series)

# Detect anomalies
anomaly_detector = TemporalAnomalyDetector(config)
anomalies = anomaly_detector.detect_anomalies(repo_series, lagged_correlations, influence_scores, paths)
```

---

## Lagged Correlation Analysis

### Concept

Lagged correlation measures the correlation between two time series where one is shifted by `k` intervals.

- **lag=0**: Synchronous correlation (standard Pearson)
- **lag=+k**: Repo A leads Repo B by k intervals
- **lag=-k**: Repo B leads Repo A by k intervals

### Mathematical Formula

For time series X and Y with lag k:

```
corr(X, Y, k) = Σ[(X[t] - μX)(Y[t+k] - μY)] / (σX * σY * n)
```

### Example

```
Time:    t1    t2    t3    t4    t5
Repo A:  80 →  75 →  70 →  65 →  60   (declining)
Repo B:  90    90    85    80    75   (declines 1 interval later)

Correlation at lag=1: HIGH (A leads B)
```

### Configuration

```yaml
thresholds:
  max_lag: 3                    # Analyze lags: -3, -2, -1, 0, +1, +2, +3
  min_lagged_correlation: 0.5   # Minimum |r| to consider
  min_significant_correlation: 0.6  # Threshold for significance
```

---

## Influence Scoring

### Concept

Influence score (0-100) measures how much a repo "leads" other repos in the organization.

### Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Leadership Count | 40 pts max | Number of repos this repo leads |
| Leadership Strength | 30 pts max | Average correlation when leading |
| Consistency | 20 pts max | Leads more than follows |
| Early Warning | 10 pts max | Average lead lag (earlier = better) |

### Influence Direction

| Direction | Criteria |
|-----------|----------|
| LEADER | Leads ≥2 repos, leads more than follows |
| FOLLOWER | Follows ≥2 repos, follows more than leads |
| BIDIRECTIONAL | Both leads and follows significantly |
| INDEPENDENT | No significant temporal relationships |

### Example Output

```
Top Influencers:
  #1 core-service: Score 78.5 [LEADER]
      Leads 5 repo(s): auth-service, api-gateway, ...
  #2 database-service: Score 65.2 [LEADER]
      Leads 3 repo(s): cache-service, ...
```

---

## Propagation Path Detection

### Concept

Propagation paths are chains showing how changes/issues spread through repositories:

```
source-repo → mid-repo → terminal-repo
     │          │            │
     └─ lag 1 ──┴── lag 1 ───┘
           Total lag: 2 intervals
```

### Path Types

| Type | Pattern | Description |
|------|---------|-------------|
| LINEAR | A → B → C | Single chain propagation |
| BRANCHING | A → B, A → C | One source, multiple targets |
| CONVERGING | A → C, B → C | Multiple sources, one target |

### Causality Score

Each edge is assigned a causality score (0-1) based on:

1. **Temporal Precedence** (0.3 max): Leader's changes precede follower's
2. **Correlation Strength** (0.3 max): Strong correlation coefficient
3. **Consistent Leadership** (0.2 max): Leader consistently leads others
4. **Asymmetry** (0.2 max): Leader leads more than follows

---

## Temporal Anomaly Detection

### Anomaly Types

| Type | Description | Severity |
|------|-------------|----------|
| `RAPID_PROPAGATION` | Changes spread quickly (low lag) | HIGH-CRITICAL |
| `LEADER_DETERIORATION` | High-influence repo declining | HIGH-CRITICAL |
| `SYSTEMIC_PROPAGATION` | Many repos affected through propagation | HIGH-CRITICAL |
| `SYNCHRONIZED_LAG_PATTERN` | Multiple pairs at same lag | MEDIUM |
| `DELAYED_CASCADE` | Slow but wide-reaching propagation | MEDIUM |

### Severity Levels

| Severity | Numeric | Description |
|----------|---------|-------------|
| LOW | 0 | Informational |
| MEDIUM | 1 | Monitor, may require attention |
| HIGH | 2 | Action required soon |
| CRITICAL | 3 | Immediate action required |

### Detection Thresholds

| Threshold | Default | Description |
|-----------|---------|-------------|
| `rapid_propagation_threshold` | 1 | Max lag for rapid propagation |
| `systemic_propagation_threshold` | 3 | Min repos for systemic risk |
| `leader_deterioration_threshold` | 0.15 | Min score drop percentage |

---

## Exit Codes

| Code | Constant | Description |
|------|----------|-------------|
| 130 | `EXIT_TEMPORAL_SUCCESS` | No temporal risks |
| 131 | `EXIT_TEMPORAL_CORRELATIONS_FOUND` | Temporal correlations found (non-critical) |
| 132 | `EXIT_CRITICAL_PROPAGATION_RISK` | Critical propagation risk |
| 133 | `EXIT_TEMPORAL_CONFIG_ERROR` | Configuration error |
| 134 | `EXIT_TEMPORAL_PARSE_ERROR` | Failed to parse input report |
| 199 | `EXIT_GENERAL_TEMPORAL_ERROR` | General error |

---

## Configuration Reference

### Full YAML Configuration

```yaml
# Lag configuration
thresholds:
  max_lag: 3
  min_lag: -3

  # Correlation thresholds
  min_lagged_correlation: 0.5
  min_significant_correlation: 0.6
  strong_correlation_threshold: 0.7

  # Influence scoring
  min_influence_score: 30.0
  high_influence_threshold: 70.0
  min_repos_led: 2

  # Propagation detection
  min_path_confidence: 0.5
  max_path_length: 5
  min_causality_score: 0.4

  # Anomaly detection
  rapid_propagation_threshold: 1
  systemic_propagation_threshold: 3
  leader_deterioration_threshold: 0.15

# Analysis options
compute_lagged_correlations: true
compute_influence_scores: true
compute_propagation_paths: true
detect_temporal_anomalies: true

# Output options
verbose: false
summary_only: false

# CI/CD options
fail_on_critical_propagation: false
fail_on_any_temporal_patterns: false
```

---

## JSON Schema

### Output Report Schema

```json
{
  "report_id": "temporal_intelligence_20250108_120000",
  "generated_at": "2025-01-08T12:00:00.000Z",
  "org_report_path": "./org-health-report.json",
  "correlation_report_path": "",

  "summary": {
    "total_repo_pairs": 10,
    "lagged_correlations_computed": 70,
    "significant_lagged_correlations": 8,
    "leader_follower_pairs": 5,
    "repos_with_influence": 4,
    "high_influence_repos": 2,
    "leader_repos": 2,
    "follower_repos": 2,
    "propagation_paths_detected": 3,
    "linear_paths": 2,
    "branching_paths": 1,
    "longest_path_length": 3,
    "repos_in_paths": 5,
    "total_anomalies": 2,
    "critical_anomalies": 1,
    "high_anomalies": 1,
    "systemic_risks": 1,
    "avg_propagation_lag": 1.5,
    "org_interconnectedness": 0.8
  },

  "lagged_correlations": [
    {
      "repo_a_id": "core-service",
      "repo_b_id": "auth-service",
      "lag": 1,
      "lag_description": "core-service leads auth-service by 1 interval(s)",
      "correlation_coefficient": 0.85,
      "leader_repo_id": "core-service",
      "follower_repo_id": "auth-service",
      "lag_intervals": 1,
      "sample_size": 5,
      "is_significant": true,
      "confidence": 0.9
    }
  ],

  "optimal_lag_matrix": {
    "core-service": {
      "auth-service": 1,
      "api-gateway": 2
    }
  },

  "influence_scores": [
    {
      "repo_id": "core-service",
      "repo_name": "Core Service",
      "influence_score": 78.5,
      "influence_rank": 1,
      "influence_direction": "leader",
      "repos_led": 5,
      "repos_following": 0,
      "avg_lead_lag": 1.4,
      "leadership_strength": 0.82,
      "systemic_importance": 85.0,
      "early_warning_potential": 70.0,
      "led_repos": ["auth-service", "api-gateway", "cache-service"]
    }
  ],

  "leader_ranking": ["core-service", "database-service", "api-gateway"],

  "propagation_paths": [
    {
      "path_id": "path_001",
      "path_type": "linear",
      "repo_sequence": ["core-service", "auth-service", "user-service"],
      "total_lag": 2,
      "path_length": 2,
      "avg_edge_strength": 0.78,
      "path_confidence": 0.72,
      "source_repo_id": "core-service",
      "terminal_repos": ["user-service"],
      "affected_repo_count": 3,
      "involves_critical_repos": false,
      "estimated_propagation_time": "2 interval(s)"
    }
  ],

  "propagation_graph": {
    "core-service": ["auth-service", "api-gateway"],
    "auth-service": ["user-service"]
  },

  "anomalies": [
    {
      "anomaly_id": "temporal_20250108_001",
      "anomaly_type": "leader_deterioration",
      "severity": "critical",
      "title": "High-Influence Repo Declining: core-service",
      "message": "core-service (influence score: 78.5) is declining and leads 5 other repos",
      "timestamp": "2025-01-08T12:00:00",
      "affected_repos": ["core-service", "auth-service", "api-gateway"],
      "affected_count": 6,
      "trigger_metric": "leader_score_drop",
      "trigger_value": 25.0,
      "trigger_threshold": 15.0,
      "recommended_actions": [
        "Prioritize stabilization of core-service",
        "Monitor downstream repos"
      ]
    }
  ],

  "recommendations": [
    {
      "id": "rec_001",
      "priority": "critical",
      "title": "Address Critical Temporal Risks",
      "message": "1 critical temporal anomaly requiring immediate attention"
    }
  ],

  "monitoring_priorities": ["core-service", "database-service"],

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
name: Temporal Intelligence Analysis

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

      - name: Run Temporal Intelligence Analysis
        run: |
          python -m analytics.run_org_temporal_intelligence \
            --org-report ./org-health-report.json \
            --output ./temporal-intelligence-report.json \
            --fail-on-critical

      - name: Upload Report
        uses: actions/upload-artifact@v4
        with:
          name: temporal-intelligence-report
          path: temporal-intelligence-report.json
```

### GitLab CI

```yaml
temporal-intelligence:
  stage: analyze
  image: python:3.11
  script:
    - python -m analytics.run_org_health --root ./org-health --output ./org-health-report.json
    - python -m analytics.run_org_temporal_intelligence --org-report ./org-health-report.json --output ./temporal-report.json --fail-on-critical
  artifacts:
    paths:
      - temporal-report.json
    expire_in: 30 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

---

## Interpreting Results

### Reading the Summary

```
================================================================================
ADVANCED CORRELATION & TEMPORAL INTELLIGENCE ANALYSIS
================================================================================

Org Health: YELLOW | Score: 72.5
Repos Analyzed: 5
Analysis Time: 150.5ms

----------------------------------------
LAGGED CORRELATION SUMMARY
----------------------------------------
Total Repo Pairs: 10
Lagged Correlations Computed: 70
Significant Correlations: 8
Leader-Follower Pairs: 5

----------------------------------------
INFLUENCE SCORING
----------------------------------------
Repos with Influence: 4
High-Influence Repos: 2
Leader Repos: 2
Follower Repos: 2

  Top Influencers:
    #1 core-service: Score 78.5 [LEADER]
        Leads 5 repo(s): auth-service, api-gateway, cache-service
```

**Key metrics to watch:**
- **High-Influence Repos**: These are early warning indicators
- **Leader Repos**: Changes here propagate to others
- **Propagation Path Length**: Longer paths = more indirect risk
- **Critical Anomalies**: Require immediate attention

### Anomaly Prioritization

1. **CRITICAL**: Immediate action required
   - High-influence leader declining
   - Systemic propagation affecting 5+ repos

2. **HIGH**: Action required within 24-48 hours
   - Rapid propagation patterns
   - Multiple propagation paths from single source

3. **MEDIUM**: Monitor and plan remediation
   - Synchronized lag patterns
   - Moderate influence changes

---

## Troubleshooting

### Common Issues

**Issue: No lagged correlations found**
- Check that repos have sufficient trend history (≥3 data points)
- Lower `--min-correlation` to 0.4
- Increase `--max-lag` to 5

**Issue: All repos have zero influence**
- Verify trend patterns are distinct (not constant)
- Check for leader-follower relationships
- Review lag window settings

**Issue: No propagation paths detected**
- Lower `--min-causality-score` to 0.3
- Check that lagged correlations exist with non-zero lag
- Verify graph connectivity

**Issue: Too many false positive anomalies**
- Increase `--min-influence-score` to 50.0
- Increase `--min-correlation` to 0.6
- Review anomaly severity thresholds

---

## Best Practices

### 1. Regular Analysis Schedule

Run temporal intelligence analysis:
- Daily for active orgs with 20+ repos
- After significant incidents
- When org health score drops significantly

### 2. Monitor High-Influence Repos

Set up dedicated monitoring for repos with:
- Influence score > 70
- Repos that lead 3+ others
- Early warning potential > 60

### 3. Propagation Prevention

When propagation paths are detected:
1. Identify source repos
2. Implement circuit breakers at intermediate nodes
3. Increase monitoring on terminal repos
4. Consider architectural decoupling

### 4. Integration with Correlation Engine

Chain with Task 3 correlation analysis:
```bash
# Run correlation first
python -m analytics.run_org_trend_correlation \
    --org-report ./report.json \
    --output ./correlation.json

# Then temporal intelligence (can use both reports)
python -m analytics.run_org_temporal_intelligence \
    --org-report ./report.json \
    --trend-correlation-report ./correlation.json \
    --output ./temporal.json
```

### 5. Leader Deterioration Response

When leader deterioration is detected:
1. **Immediate**: Prioritize stabilization of leader repo
2. **Short-term**: Monitor all follower repos for cascade
3. **Long-term**: Review dependencies to reduce coupling

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-08 | Initial release - Phase 14.8 Task 4 |

---

## Related Documentation

- [Org Health Governance Engine](./ORG_HEALTH_GOVERNANCE_GUIDE.md) - Task 1
- [Org Alerting & Escalation Engine](./ORG_ALERTING_AND_ESCALATION_ENGINE.md) - Task 2
- [Multi-Repository Trend Correlation Engine](./ORG_TREND_CORRELATION_ENGINE.md) - Task 3
- [Phase 14.8 Task 4 Completion Summary](./PHASE14_8_TASK4_COMPLETION_SUMMARY.md)

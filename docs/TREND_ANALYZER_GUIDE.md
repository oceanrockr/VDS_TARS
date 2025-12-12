# T.A.R.S. Repository Health Trend Analyzer

**Phase:** 14.7 Task 10
**Version:** 1.0.0
**Author:** T.A.R.S. Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Statistical Methods](#statistical-methods)
7. [CLI Reference](#cli-reference)
8. [Programmatic API](#programmatic-api)
9. [Exit Codes](#exit-codes)
10. [Configuration](#configuration)
11. [JSON Schema](#json-schema)
12. [Integration with Pipeline](#integration-with-pipeline)
13. [CI/CD Integration](#cicd-integration)
14. [Interpreting Early Warnings](#interpreting-early-warnings)
15. [Example Plots](#example-plots)
16. [Troubleshooting](#troubleshooting)
17. [Best Practices](#best-practices)

---

## Overview

The Repository Health Trend Analyzer is a time-series analysis engine that operates on historical dashboard snapshots to provide:

- **Score Trend Analysis**: Linear regression, gradient, and direction detection
- **Moving Averages**: 3, 7, and 14-snapshot windows
- **Volatility Metrics**: Standard deviation and coefficient of variation
- **Issue Trends**: Total and critical issue trajectory
- **Version Stability**: Per-version health tracking over time
- **Anomaly Detection**: Z-score based detection of outliers
- **Predictive Scoring**: Extrapolated future scores with confidence intervals
- **Early Warnings**: Proactive alerts for degradation patterns

### Use Cases

- **Proactive Monitoring**: Detect degradation before it becomes critical
- **Capacity Planning**: Forecast repository health for release planning
- **Pattern Detection**: Identify recurring issues or seasonal patterns
- **Regression Detection**: Catch slow regressions across releases
- **Compliance Reporting**: Historical health trends for audits

### When to Use the Trend Analyzer

- After generating health dashboards over time
- As part of scheduled monitoring jobs
- Before major releases to assess health trajectory
- When investigating recurring issues
- For building health prediction models

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │       Dashboard Snapshots           │
                    │     (dashboard-history/*.json)      │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       HealthHistoryStore            │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │ Index Management             │    │
                    │  │ (timestamps, versions)       │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │ Snapshot Loading             │    │
                    │  │ (date range, last N)         │    │
                    │  └─────────────────────────────┘    │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         TrendAnalyzer               │
                    │                                     │
                    │  ┌───────────┐  ┌───────────┐       │
                    │  │  Score    │  │  Issue    │       │
                    │  │  Trend    │  │  Trend    │       │
                    │  └───────────┘  └───────────┘       │
                    │  ┌───────────┐  ┌───────────┐       │
                    │  │  Anomaly  │  │ Prediction│       │
                    │  │ Detection │  │  Engine   │       │
                    │  └───────────┘  └───────────┘       │
                    │  ┌───────────┐  ┌───────────┐       │
                    │  │  Early    │  │  Version  │       │
                    │  │ Warnings  │  │  Trends   │       │
                    │  └───────────┘  └───────────┘       │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │        TrendReport (JSON)           │
                    │    + Charts (PNG/ASCII)             │
                    │    + Exit Code (80-89)              │
                    └─────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **HealthHistoryStore** | Manages historical snapshot storage and retrieval |
| **StatisticsCalculator** | Pure-Python statistical computations |
| **TrendAnalyzer** | Core time-series analysis engine |
| **TrendChartGenerator** | Chart visualization (matplotlib/ASCII fallback) |
| **TrendEngine** | Main orchestrator for the analysis pipeline |

---

## Key Features

### 1. Score Trend Analysis

Computes repository score trajectory using:
- **Linear Regression**: Ordinary Least Squares (OLS) for trend line
- **Gradient**: Points change per snapshot
- **R-Squared**: Trend confidence/fit quality
- **Direction**: IMPROVING, STABLE, or DECLINING

### 2. Moving Averages

Smooths score fluctuations:
- **MA-3**: 3-snapshot window (short-term)
- **MA-7**: 7-snapshot window (medium-term)
- **MA-14**: 14-snapshot window (long-term)

### 3. Volatility Analysis

Measures score stability:
- **Standard Deviation**: Absolute volatility
- **Coefficient of Variation**: Normalized volatility
- **Volatility Trend**: INCREASING, STABLE, DECREASING

### 4. Anomaly Detection

Identifies outliers using:
- **Z-Score**: Deviations > configurable threshold
- **Sudden Drops**: Score decreases > threshold
- **Issue Spikes**: Sudden increase in issues

### 5. Predictive Scoring

Forecasts future health:
- **Next Score**: Linear extrapolation
- **Confidence Intervals**: 95% CI by default
- **Probability Estimates**: P(YELLOW) and P(RED)

### 6. Early Warning System

Proactive alerts for:
- Slow degradation trends
- Increasing volatility
- Approaching thresholds
- Consistent version degradation

---

## Installation

### Prerequisites

- Python 3.8+
- Repository Health Dashboard (Task 8) for generating snapshots
- Alerting Engine (Task 9) for integrated monitoring

### Setup

The trend analyzer is part of the analytics module:

```bash
# Verify installation
python -c "from analytics.trend_analyzer import TrendEngine; print('OK')"

# Check CLI
python -m analytics.run_trends --help
```

---

## Quick Start

### Example 1: Add Snapshot to History

```bash
# Add current dashboard to history
python -m analytics.run_trends \
  --history-dir ./dashboard-history \
  --add-snapshot ./dashboard/health-dashboard.json
```

### Example 2: Basic Trend Analysis

```bash
# Run trend analysis on history
python -m analytics.run_trends \
  --history-dir ./dashboard-history \
  --output ./trend-report.json
```

### Example 3: With Chart Generation

```bash
# Generate charts alongside analysis
python -m analytics.run_trends \
  --history-dir ./dashboard-history \
  --output ./trend-report.json \
  --generate-charts \
  --chart-dir ./charts
```

### Example 4: Quick Summary

```bash
# Get quick trend summary without full analysis
python -m analytics.run_trends \
  --history-dir ./dashboard-history \
  --summary-only
```

### Example 5: Full Pipeline Integration

```bash
# Complete workflow: dashboard + history + trends
python scripts/prepare_release_artifacts.py \
  --generate-dashboard \
  --update-history \
  --run-trends \
  --trend-history-dir ./dashboard-history \
  --trend-charts
```

---

## Statistical Methods

### Linear Regression

The analyzer uses Ordinary Least Squares (OLS) regression:

```
y = mx + b

where:
  y = repository score
  x = snapshot index (0, 1, 2, ...)
  m = slope (gradient)
  b = intercept
```

**Metrics**:
- **R-Squared (R²)**: Coefficient of determination (0-1)
  - R² > 0.7: Strong trend
  - R² > 0.5: Moderate trend
  - R² < 0.3: Weak/no trend

- **P-Value**: Statistical significance
  - p < 0.05: Significant trend
  - p >= 0.05: May be random

### Moving Averages

Simple Moving Average (SMA):

```
SMA_n = (1/n) * Σ(values in window)
```

Windows:
- **MA-3**: Captures short-term fluctuations
- **MA-7**: Smooths weekly patterns
- **MA-14**: Reveals long-term trends

### Z-Score Anomaly Detection

Z-score calculation:

```
z = (x - μ) / σ

where:
  x = observed value
  μ = mean
  σ = standard deviation
```

Thresholds:
- |z| > 2.0: Potential anomaly (default)
- |z| > 3.0: Strong anomaly

### Volatility Metrics

**Standard Deviation**:
```
σ = sqrt( Σ(x_i - μ)² / (n-1) )
```

**Coefficient of Variation**:
```
CV = σ / μ
```

### Prediction with Confidence Intervals

Extrapolation:
```
ŷ = m * x + b
```

Confidence interval:
```
CI = ŷ ± t * SE

where:
  t = t-distribution value for confidence level
  SE = standard error of prediction
```

---

## CLI Reference

### Standalone CLI

```bash
python -m analytics.run_trends [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--history-dir PATH` | Directory containing historical snapshots |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output PATH` | None | Output path for trend report JSON |
| `--generate-charts` | False | Generate trend visualization charts |
| `--chart-dir PATH` | history-dir/charts | Directory for chart output |
| `--min-snapshots INT` | 3 | Minimum snapshots for analysis |
| `--max-snapshots INT` | 100 | Maximum snapshots to analyze |
| `--zscore-threshold FLOAT` | 2.0 | Z-score threshold for anomalies |
| `--score-drop-threshold FLOAT` | 15.0 | Score drop threshold for anomalies |
| `--issue-spike-threshold INT` | 5 | Issue spike threshold |
| `--prediction-horizon INT` | 3 | Snapshots to predict ahead |
| `--confidence-level FLOAT` | 0.95 | Confidence level for predictions |
| `--warning-threshold FLOAT` | 60.0 | Warning score threshold |
| `--critical-threshold FLOAT` | 40.0 | Critical score threshold |
| `--add-snapshot PATH` | None | Add snapshot before analysis |
| `--snapshot-version STR` | None | Version for added snapshot |
| `--rebuild-index` | False | Rebuild history index |
| `--validate-index` | False | Validate index integrity |
| `--summary-only` | False | Quick summary only |
| `--quiet` | False | Suppress output |
| `--verbose` | False | Enable debug output |
| `--json` | False | Output as JSON to stdout |

---

## Programmatic API

### Basic Usage

```python
from pathlib import Path
from analytics.trend_analyzer import (
    TrendConfig,
    TrendEngine,
    HealthHistoryStore,
    add_snapshot_to_history,
)

# Step 1: Add snapshots to history
add_snapshot_to_history(
    history_dir=Path("./dashboard-history"),
    dashboard_path=Path("./dashboard/health-dashboard.json"),
    version="1.0.0"
)

# Step 2: Configure analysis
config = TrendConfig(
    history_dir=Path("./dashboard-history"),
    output_path=Path("./trend-report.json"),
    min_snapshots=3,
    prediction_horizon=5,
    generate_charts=True,
    chart_output_dir=Path("./charts"),
)

# Step 3: Run analysis
engine = TrendEngine(config)
report, exit_code = engine.run()

# Step 4: Use results
print(f"Overall Trend: {report.overall_trend}")
print(f"Current Score: {report.current_score}")
print(f"Predicted Next: {report.predicted_next_score}")
print(f"Anomalies: {report.total_anomalies}")
print(f"Warnings: {report.total_warnings}")
```

### Advanced: Direct Analyzer Usage

```python
from analytics.trend_analyzer import (
    TrendConfig,
    TrendAnalyzer,
    HealthHistoryStore,
)

# Load history
store = HealthHistoryStore(Path("./dashboard-history"))
store.initialize()
snapshots = store.get_last_n_snapshots(30)

# Analyze
config = TrendConfig(history_dir=Path("./dashboard-history"))
analyzer = TrendAnalyzer(config)
analyzer.load_snapshots(snapshots)

# Get specific analysis
score_trend = analyzer.compute_score_trend()
issue_trend = analyzer.compute_issue_trend()
anomalies = analyzer.detect_anomalies()
predictions = analyzer.generate_prediction()
warnings = analyzer.generate_early_warnings()
graph_data = analyzer.generate_graph_data()

# Access results
print(f"Regression Slope: {score_trend.regression.slope}")
print(f"R-Squared: {score_trend.regression.r_squared}")
print(f"MA-7: {score_trend.moving_averages[7].current_value}")
print(f"Volatility: {score_trend.volatility.standard_deviation}")
```

---

## Exit Codes

### Exit Code Table (80-89)

| Code | Constant | Description |
|------|----------|-------------|
| 80 | `EXIT_TREND_SUCCESS` | Analysis completed successfully |
| 81 | `EXIT_INSUFFICIENT_HISTORY` | Not enough snapshots |
| 82 | `EXIT_INVALID_SNAPSHOT` | Corrupted or malformed snapshot |
| 83 | `EXIT_COMPUTATION_ERROR` | Time series computation failed |
| 84 | `EXIT_PREDICTION_ERROR` | Prediction model failed |
| 85 | `EXIT_HISTORY_STORE_ERROR` | History read/write error |
| 86 | `EXIT_CHART_GENERATION_ERROR` | Chart generation failed |
| 89 | `EXIT_GENERAL_TREND_ERROR` | General error |

---

## Configuration

### TrendConfig Options

```python
@dataclass
class TrendConfig:
    # Input
    history_dir: Path                       # Required
    output_path: Optional[Path] = None

    # Analysis parameters
    min_snapshots: int = 3
    max_snapshots: int = 100
    ma_windows: List[int] = [3, 7, 14]

    # Anomaly detection
    zscore_threshold: float = 2.0
    score_drop_threshold: float = 15.0
    issue_spike_threshold: int = 5

    # Prediction
    prediction_horizon: int = 3
    confidence_level: float = 0.95

    # Early warning thresholds
    warning_score_threshold: float = 60.0
    critical_score_threshold: float = 40.0
    volatility_threshold: float = 10.0

    # Charts
    generate_charts: bool = False
    chart_output_dir: Optional[Path] = None

    # Behavior
    verbose: bool = False
```

---

## JSON Schema

### TrendReport Schema

```json
{
  "report_id": "string",
  "generated_at": "ISO8601 timestamp",
  "history_dir": "string",
  "snapshots_analyzed": "integer",
  "first_snapshot": "ISO8601 timestamp",
  "last_snapshot": "ISO8601 timestamp",
  "analysis_window_days": "integer",

  "overall_trend": "improving|stable|declining",
  "trend_confidence": "float (0-1)",

  "current_score": "float",
  "current_health": "green|yellow|red",

  "predicted_next_score": "float",
  "confidence_interval": ["lower", "upper"],

  "ma_3": "float",
  "ma_7": "float",
  "ma_14": "float",

  "score_volatility": "float",
  "volatility_trend": "increasing|stable|decreasing",

  "total_anomalies": "integer",
  "anomalies": [
    {
      "anomaly_id": "string",
      "anomaly_type": "sudden_score_drop|issue_spike|...",
      "severity": "low|medium|high|critical",
      "timestamp": "ISO8601 timestamp",
      "actual_value": "float",
      "expected_value": "float",
      "zscore": "float",
      "description": "string"
    }
  ],

  "total_warnings": "integer",
  "early_warnings": [
    {
      "warning_id": "string",
      "warning_type": "slow_degradation|high_volatility|...",
      "level": "low|medium|high|critical",
      "title": "string",
      "message": "string",
      "recommendations": ["string"]
    }
  ],

  "degrading_versions": ["version_strings"],

  "regression_slope": "float",
  "regression_intercept": "float",
  "regression_r_squared": "float",

  "analysis_duration_ms": "float"
}
```

---

## Integration with Pipeline

### Complete Pipeline Workflow

```
┌──────────────┐
│   Generate   │ ──→ Health Dashboard (Task 8)
│   Dashboard  │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Run        │ ──→ Alert Rules (Task 9)
│   Alerts     │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Update     │ ──→ Add snapshot to history
│   History    │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Analyze    │ ──→ Time-series analysis ◀── TASK 10
│   Trends     │
└──────────────┘
        │
        ▼
┌──────────────┐
│   Generate   │ ──→ Charts (optional)
│   Charts     │
└──────────────┘
        │
        ▼
┌──────────────┐
│   CI/CD      │ ──→ Act on warnings/predictions
│   Decision   │
└──────────────┘
```

### Integrated Command

```bash
python scripts/prepare_release_artifacts.py \
  --generate-dashboard \
  --run-alerts \
  --update-history \
  --run-trends \
  --trend-history-dir ./dashboard-history \
  --trend-output ./release/trend-report.json \
  --trend-charts
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Repository Health Trends

on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  workflow_dispatch:

jobs:
  trend-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Download History
        uses: actions/download-artifact@v4
        with:
          name: dashboard-history
          path: ./dashboard-history
        continue-on-error: true

      - name: Generate Dashboard
        run: |
          python -m analytics.repository_health_dashboard \
            --repository-path ./artifact-repository \
            --output-dir ./dashboard

      - name: Run Trend Analysis
        run: |
          python -m analytics.run_trends \
            --history-dir ./dashboard-history \
            --add-snapshot ./dashboard/health-dashboard.json \
            --output ./trend-report.json \
            --generate-charts \
            --chart-dir ./charts

      - name: Upload History
        uses: actions/upload-artifact@v4
        with:
          name: dashboard-history
          path: ./dashboard-history/

      - name: Upload Reports
        uses: actions/upload-artifact@v4
        with:
          name: trend-analysis
          path: |
            ./trend-report.json
            ./charts/

      - name: Check Trend Health
        run: |
          python -c "
          import json
          with open('trend-report.json') as f:
              report = json.load(f)
          if report['overall_trend'] == 'declining':
              print('::warning::Repository health is declining!')
          if report['total_warnings'] > 0:
              print('::warning::Early warnings detected!')
          "
```

### GitLab CI

```yaml
trend-analysis:
  stage: analyze
  script:
    - python -m analytics.run_trends
        --history-dir ./dashboard-history
        --add-snapshot ./dashboard/health-dashboard.json
        --output ./trend-report.json
  artifacts:
    paths:
      - trend-report.json
      - dashboard-history/
    expire_in: 30 days
  cache:
    key: dashboard-history
    paths:
      - dashboard-history/
```

---

## Interpreting Early Warnings

### Warning Types

| Warning Type | Level | Meaning | Action |
|--------------|-------|---------|--------|
| `slow_degradation` | MEDIUM | Score declining steadily | Investigate root cause |
| `high_volatility` | MEDIUM | Score fluctuating | Review recent changes |
| `approaching_threshold` | MEDIUM | Score near warning level | Preventive maintenance |
| `approaching_critical` | HIGH | Score near critical level | Immediate action required |

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| LOW | Informational | Next sprint |
| MEDIUM | Attention needed | This week |
| HIGH | Urgent | This day |
| CRITICAL | Emergency | Immediate |

### Recommended Responses

1. **Slow Degradation**
   - Review recent releases for patterns
   - Check for accumulating technical debt
   - Consider maintenance release

2. **High Volatility**
   - Investigate intermittent issues
   - Check for flaky tests/scans
   - Review CI/CD stability

3. **Approaching Threshold**
   - Prioritize issue backlog
   - Plan preventive fixes
   - Increase monitoring frequency

4. **Approaching Critical**
   - Emergency triage
   - Consider rollback
   - Escalate to team leads

---

## Example Plots

### Score Trend Chart (ASCII)

```
Repository Health Score Trend
=============================

 95.0 |*
      |  *
 90.0 |    *
      |      *
 85.0 |        *
      |          * * *
 80.0 |              * - - - (prediction)
      |
 75.0 |
      +---------------------------
        Start              End

Points: 10
Min: 78.5
Max: 95.0
Latest: 82.3
Predicted: 79.8 [76.2, 83.4]
```

### Volatility Over Time

```
Score Volatility (std dev)
==========================

 12.0 |
      |                    *
 10.0 |              *
      |        *
  8.0 |  *
      |
  6.0 | *  *
      +---------------------------
        Increasing volatility detected!
```

---

## Troubleshooting

### Common Issues

#### 1. "Not enough history"

**Cause**: Fewer snapshots than `min_snapshots`.

**Solution**:
```bash
# Lower minimum or add more snapshots
python -m analytics.run_trends \
  --history-dir ./dashboard-history \
  --min-snapshots 2

# Or add snapshots over time
python -m analytics.run_trends \
  --history-dir ./dashboard-history \
  --add-snapshot ./dashboard.json
```

#### 2. "Invalid snapshot"

**Cause**: Malformed JSON or missing required fields.

**Solution**:
```bash
# Validate snapshot
python -c "
import json
with open('dashboard.json') as f:
    d = json.load(f)
    print('Score:', d.get('repository_score'))
    print('Health:', d.get('overall_health'))
"

# Rebuild index if needed
python -m analytics.run_trends \
  --history-dir ./dashboard-history \
  --rebuild-index
```

#### 3. "Chart generation failed"

**Cause**: Matplotlib not installed.

**Solution**:
```bash
# Install matplotlib
pip install matplotlib

# Or use ASCII fallback (automatic)
```

#### 4. Trend seems wrong

**Cause**: Insufficient data or outliers.

**Solution**:
```bash
# Increase minimum snapshots
--min-snapshots 10

# Increase z-score threshold
--zscore-threshold 3.0
```

---

## Best Practices

### 1. Regular Snapshots

Add snapshots at consistent intervals:
```bash
# Daily cron job
0 0 * * * python -m analytics.run_trends \
  --history-dir /data/history \
  --add-snapshot /data/dashboard/latest.json
```

### 2. Retain History

Keep sufficient history for analysis:
```bash
# At least 14 snapshots for MA-14
--min-snapshots 14

# Limit old snapshots
--max-snapshots 90  # ~3 months daily
```

### 3. Monitor Predictions

Act on predictions, not just current state:
```python
if report.prediction.probability_red > 0.5:
    alert("High probability of critical health in next 3 snapshots")
```

### 4. Archive Reports

Keep historical trend reports:
```bash
# Archive with timestamp
mv trend-report.json "trends/trend-$(date +%Y%m%d).json"
```

### 5. Correlate with Releases

Track trends across releases:
```python
# Tag snapshots with versions
--add-snapshot ./dashboard.json \
--snapshot-version "v1.2.3"
```

---

## Summary

The Repository Health Trend Analyzer provides:

- **Time-Series Analysis**: Linear regression, moving averages, volatility
- **Anomaly Detection**: Z-score based outlier identification
- **Predictive Scoring**: Future health forecasting with confidence intervals
- **Early Warnings**: Proactive degradation detection
- **Version Tracking**: Per-version health trends
- **Charts**: Visual trend representation
- **Exit Codes (80-89)**: CI/CD integration

For related documentation:
- [Alerting Engine Guide](./ALERTING_ENGINE_GUIDE.md) - Alert rules and dispatch
- [Repository Health Dashboard Guide](./REPOSITORY_HEALTH_DASHBOARD_GUIDE.md) - Dashboard generation

---

**Version:** 1.0.0
**Phase:** 14.7 Task 10
**Last Updated:** 2025-01-06

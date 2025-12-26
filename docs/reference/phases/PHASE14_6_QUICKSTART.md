# Phase 14.6 Quickstart Guide

**T.A.R.S. v1.0.1 - Post-GA 7-Day Stabilization & Retrospective**

This guide provides step-by-step instructions for running Phase 14.6 monitoring and retrospective generation for the T.A.R.S. system.

---

## Overview

Phase 14.6 implements a comprehensive 7-day post-GA monitoring and retrospective framework consisting of:

1. **GA Day Monitoring** - Baseline KPI collection on launch day
2. **7-Day Stability Monitoring** - Daily metric collection and drift detection
3. **Anomaly Detection** - EWMA-based anomaly detection with Z-score analysis
4. **Health Reporting** - Daily health scores and mitigation recommendations
5. **Regression Analysis** - Multi-baseline comparison and rollback recommendations
6. **Retrospective Generation** - Comprehensive post-mortem with v1.0.2 recommendations

---

## Prerequisites

### System Requirements
- Python 3.8+
- Prometheus metrics endpoint (or compatible metrics source)
- Sufficient disk space for 7 days of metric storage (~100MB)

### Python Dependencies
```bash
pip install -r requirements-dev.txt
```

Required packages:
- `prometheus-client` - Metrics collection
- `pytest` - Testing framework
- `numpy` - Statistical analysis (for anomaly detection)

---

## Production Usage (Real Data)

### Phase 1: GA Day Monitoring

On GA Day (launch day), collect baseline KPI metrics:

```bash
# Start GA Day monitoring (runs for 24 hours)
python observability/ga_kpi_collector.py \
  --prometheus-url http://localhost:9090 \
  --output-dir ga_kpis \
  --ga-timestamp "2025-11-18T00:00:00Z"
```

**Output:**
- `ga_kpis/ga_kpi_summary.json` - GA Day baseline metrics

**What it collects:**
- Availability, error rate, P99 latency
- CPU, memory, Redis, database metrics
- Cost estimation

---

### Phase 2: 7-Day Stability Monitoring

Run daily for 7 days post-GA:

```bash
# Run daily (Day 1-7)
python observability/stability_monitor_7day.py \
  --prometheus-url http://localhost:9090 \
  --output-dir stability \
  --day-number 1 \
  --ga-baseline ga_kpis/ga_kpi_summary.json
```

**Output:**
- `stability/day_01_summary.json` through `day_07_summary.json`
- Daily drift comparisons vs GA baseline
- Rollback recommendations (if severe regressions detected)

**Automation:**
Set up a cron job or scheduled task:
```bash
# Example cron (run at 11:59 PM daily)
59 23 * * * cd /path/to/tars && python observability/stability_monitor_7day.py --day-number $(( ($(date +\%s) - $(date -d "2025-11-18" +\%s)) / 86400 ))
```

---

### Phase 3: Anomaly Detection

Run continuously or on-demand to detect anomalies:

```bash
# Run anomaly detection
python observability/anomaly_detector_lightweight.py \
  --prometheus-url http://localhost:9090 \
  --output-file anomaly_events.json \
  --baseline ga_kpis/ga_kpi_summary.json \
  --z-threshold 3.0
```

**Output:**
- `anomaly_events.json` - All detected anomalies with Z-scores

**Options:**
- `--z-threshold`: Z-score threshold for anomaly detection (default: 3.0)
- `--ewma-alpha`: EWMA smoothing factor (default: 0.3)

---

### Phase 4: Daily Health Reporting

Generate daily health reports:

```bash
# Run at end of each day
python observability/daily_health_reporter.py \
  --stability-data stability/day_01_summary.json \
  --anomaly-data anomaly_events.json \
  --output-file stability/day_01_HEALTH.json \
  --day-number 1
```

**Output:**
- `stability/day_XX_HEALTH.json` - Health scores and mitigation plans

---

### Phase 5: Regression Analysis

Run on Day 7 to analyze regressions vs GA baseline:

```bash
# Run regression analysis
python observability/regression_analyzer.py \
  --ga-baseline ga_kpis/ga_kpi_summary.json \
  --7day-summaries stability \
  --output regression_summary.json
```

**Output:**
- `regression_summary.json` - Identified regressions with severity
- Rollback recommendation (if critical regressions detected)

---

### Phase 6: Retrospective Generation

Generate comprehensive retrospective on Day 7:

```bash
# Auto-detect all files (recommended)
python scripts/generate_retrospective.py --auto

# Or specify paths manually
python scripts/generate_retrospective.py \
  --ga-data ga_kpis \
  --7day-data stability \
  --regression regression_summary.json \
  --anomalies anomaly_events.json \
  --output docs/final/GA_7DAY_RETROSPECTIVE.md
```

**Output:**
- `docs/final/GA_7DAY_RETROSPECTIVE.md` - Human-readable retrospective
- `docs/final/GA_7DAY_RETROSPECTIVE.json` - Machine-readable data

**What's included:**
- Executive summary with overall status
- Success metrics (what went well)
- Degradations (what could be improved)
- Unexpected drifts (emerging issues)
- Cost analysis and optimization recommendations
- SLO burn-down analysis
- Prioritized v1.0.2 recommendations (P0/P1/P2/P3)
- Process improvement suggestions
- Action item checklist

---

## Test Mode Usage (Test Data)

For development, testing, or demonstration purposes, use the provided test data:

### Run Unit Tests

```bash
# Run all unit tests
pytest tests/test_retrospective_generator.py -v

# Run specific test classes
pytest tests/test_retrospective_generator.py::TestSuccessAnalyzer -v
pytest tests/test_retrospective_generator.py::TestDriftAnalyzer -v

# Run with coverage
pytest tests/test_retrospective_generator.py --cov=scripts.generate_retrospective --cov-report=html
```

**Expected output:**
- All tests should pass (40+ test cases)
- Coverage report in `htmlcov/index.html`

---

### Run End-to-End Smoke Test

```bash
# Run complete pipeline on test data
./scripts/test_phase14_6_pipeline.sh
```

**What it does:**
1. Verifies test data exists (GA KPI, 7-day summaries, regression, anomalies)
2. Runs retrospective generator on test data
3. Validates Markdown and JSON outputs
4. Checks for expected sections and structure
5. Displays quick stats

**Expected output:**
```
============================================================
Phase 14.6 - End-to-End Pipeline Smoke Test
============================================================

Step 1: Verifying test data...
  ✓ GA KPI summary found
  ✓ All 7 daily summaries found
  ✓ Regression summary found
  ✓ Anomaly events found

Step 2: Running retrospective generator...
...

Step 3: Verifying outputs...
  ✓ Markdown report generated
  ✓ JSON report generated
  ✓ JSON structure validated

Step 4: Validating Markdown sections...
  ✓ All expected Markdown sections found

============================================================
SMOKE TEST PASSED
============================================================

Reports generated:
  - Markdown: test_output/GA_7DAY_RETROSPECTIVE.md
  - JSON: test_output/GA_7DAY_RETROSPECTIVE.json

Quick Stats:
  - Successes: 8
  - Degradations: 5
  - Unexpected Drifts: 3
  - Cost Trend: stable
  - Recommendations: 10
  - Process Improvements: 5
  - Action Items: 7
```

---

### Generate Retrospective on Test Data

```bash
# Using auto-detection (test_data/ directory)
python scripts/generate_retrospective.py \
  --ga-data test_data/ga_kpis \
  --7day-data test_data/stability \
  --regression test_data/regression/regression_summary.json \
  --anomalies test_data/anomalies/anomaly_events.json \
  --output test_output/retrospective.md
```

**Output:**
- `test_output/retrospective.md` - Test retrospective (Markdown)
- `test_output/retrospective.json` - Test retrospective (JSON)

---

## Makefile Shortcuts

Use the provided Makefile for common tasks:

```bash
# Run all Phase 14.6 tests
make test-phase14_6

# Run unit tests only
make test-retro-unit

# Run smoke test only
make test-retro-smoke

# Run retrospective on test data
make retro-test

# Clean test outputs
make clean-test
```

---

## Interpreting Outputs

### Markdown Retrospective Sections

1. **Executive Summary**
   - High-level overview of 7-day period
   - Overall status: EXCELLENT / STABLE / NEEDS ATTENTION / CRITICAL
   - Key metrics at a glance

2. **What Went Well ✅**
   - Success metrics table (category, metric, target, actual, achievement)
   - Top success highlights

3. **What Could Be Improved ⚠️**
   - Degradations grouped by severity (Critical / High / Medium)
   - Day occurred, description, impact, resolution status

4. **Unexpected Drifts 📊**
   - Metrics that drifted 10-30% (not regressions, but significant)
   - Trend analysis (increasing/decreasing/volatile)
   - Potential causes (3 per drift)
   - Investigation flag for drifts > 15%

5. **Cost Analysis 💰**
   - GA Day cost vs 7-day average
   - Cost trend (increasing/stable/decreasing)
   - Cost breakdown by resource type
   - Optimization recommendations

6. **SLO Compliance Summary**
   - SLO table (Availability, P99 Latency, Error Rate)
   - Budget consumed percentage
   - Days to exhaustion (if applicable)
   - Status indicators: 🚨 Critical / ⚠️ At Risk / ⚡ Moderate / ✅ Healthy

7. **Recommendations for v1.0.2 🚀**
   - Top 10 prioritized recommendations
   - Priority tags: [P0] Critical, [P1] High, [P2] Medium, [P3] Low
   - Actionable next steps

8. **Process Improvements 🔧**
   - 3-5 process improvement suggestions
   - Monitoring, testing, incident response enhancements

9. **Action Items**
   - **P0:** Immediate action required (critical issues)
   - **P1:** High priority (24-48h resolution)
   - **P2:** Medium priority (within sprint)
   - Checkbox format for tracking

---

### JSON Retrospective Structure

```json
{
  "generation_timestamp": "2025-11-25T14:30:00Z",
  "ga_day_timestamp": "2025-11-18T00:00:00Z",
  "seven_day_end_timestamp": "2025-11-25T23:59:59Z",

  "successes": [
    {
      "category": "availability",
      "description": "Maintained high availability",
      "metric_name": "overall_availability",
      "target_value": "99.9%",
      "actual_value": "99.950%",
      "achievement_percent": 100.05
    }
  ],

  "degradations": [
    {
      "day_occurred": 2,
      "category": "performance",
      "description": "P99 latency spike",
      "severity": "high",
      "impact": "Performance degradation",
      "resolution_status": "resolved",
      "resolution_details": "Recovered on Day 3"
    }
  ],

  "unexpected_drifts": [
    {
      "metric_name": "avg_cpu_percent",
      "baseline_value": 45.2,
      "final_value": 52.3,
      "drift_percent": 15.7,
      "trend": "increasing",
      "potential_causes": ["Increased workload", "..."],
      "investigation_needed": true
    }
  ],

  "cost_analysis": {
    "ga_day_cost": 300.0,
    "seven_day_total_cost": 2352.0,
    "daily_average_cost": 336.0,
    "cost_trend": "stable",
    "cost_breakdown": {
      "compute": 201.6,
      "storage": 67.2,
      "network": 50.4,
      "other": 16.8
    },
    "cost_optimization_recommendations": [...]
  },

  "slo_burn_downs": [
    {
      "slo_name": "Availability",
      "target": 99.9,
      "budget": 0.1,
      "budget_consumed_percent": 12.5,
      "days_to_exhaustion": 52.8,
      "compliance_by_day": [
        {
          "day": 0,
          "value": 99.95,
          "compliant": true,
          "budget_consumed": 0.0
        },
        ...
      ]
    }
  ],

  "recommendations_v1_0_2": [
    "[P1] Investigate and resolve 1 high-severity degradation(s)",
    "[P2] Investigate 2 unexpected metric drift(s)",
    ...
  ],

  "process_improvements": [...],

  "action_items": [
    {
      "priority": "P1",
      "description": "Investigate degradations",
      "status": "open"
    }
  ]
}
```

**Use cases:**
- **compliance_by_day:** Full SLO time-series for charting/analysis
- **action_items:** Import into issue tracker (Jira, GitHub Issues)
- **cost_breakdown:** Feed into cost dashboards
- **recommendations:** Automated v1.0.2 planning

---

## Common Issues & Troubleshooting

### Issue: "FileNotFoundError: ga_kpi_summary.json not found"

**Cause:** GA Day monitoring hasn't been run yet, or output path is incorrect.

**Solution:**
1. Verify GA Day monitoring completed successfully
2. Check `--ga-data` path points to correct directory
3. If using `--auto`, ensure files are in default locations:
   - `ga_kpis/ga_kpi_summary.json`
   - `stability/day_XX_summary.json`

---

### Issue: "No regressions detected but expected some"

**Cause:** Metrics are within acceptable drift range (< 30%).

**Solution:**
- This is expected if system is stable
- Check `unexpected_drifts` section for 10-30% drifts
- Review `degradations` for day-specific issues

---

### Issue: "SLO burn-down shows 100% budget consumed"

**Cause:** Multiple SLO violations occurred during 7-day period.

**Solution:**
1. Review degradations for availability/latency/error rate issues
2. Check action items for immediate remediation steps
3. Implement P0/P1 recommendations before next release

---

### Issue: "Cost trend is 'increasing' but want to understand why"

**Cause:** Second-half average cost > first-half by 10%+.

**Solution:**
1. Review `cost_breakdown` for largest cost component
2. Check `cost_optimization_recommendations` for specific actions
3. Cross-reference with `unexpected_drifts` (CPU/memory increases)
4. Review daily summaries for autoscaling events

---

### Issue: "Test suite fails on Windows"

**Cause:** Path separator differences (`\` vs `/`).

**Solution:**
- Use `pathlib.Path` for cross-platform compatibility
- Or run tests in WSL (Windows Subsystem for Linux)

---

## Next Steps

After generating the retrospective:

1. **Review with team** - Schedule retrospective meeting to discuss findings
2. **Prioritize action items** - Assign owners to P0/P1 items
3. **Create v1.0.2 issues** - Import action items into issue tracker
4. **Implement recommendations** - Start on highest-priority items
5. **Update monitoring** - Apply process improvements to monitoring stack
6. **Document learnings** - Add to team knowledge base

---

## Additional Resources

### Documentation
- [Phase 14.6 Implementation Report](PHASE14_6_PHASE6_COMPLETION_SUMMARY.md) - Complete implementation details
- Retrospective Generator Source: `scripts/generate_retrospective.py`
- Test Suite: `tests/` directory

### Related Phases
- **Phase 11:** Multi-Agent RL System
- **Phase 12:** Security & Production Deployment
- **Phase 13:** Evaluation Engine
- **Phase 14.1-14.5:** GA Day monitoring, anomaly detection, health reporting, regression analysis

### Support
- GitHub Issues: https://github.com/veleron-dev/tars/issues
- Documentation: `docs/` directory
- Runbooks: `docs/runbooks/` (coming soon)

---

**Generated:** 2025-11-25
**Phase:** 14.6 - Quickstart Guide
**Version:** v1.0.1

---

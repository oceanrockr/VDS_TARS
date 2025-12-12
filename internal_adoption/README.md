# What is Phase 14.6?

**TL;DR:** Automated 7-day post-GA monitoring that tells you what broke, what improved, and what to fix next.

---

## The Problem

After launching a major release (GA Day), teams struggle to:

- ❌ **Track stability trends** - Is performance getting better or worse?
- ❌ **Detect subtle regressions** - Small degradations compound over time
- ❌ **Prioritize fixes** - Which issues matter most for v1.0.2?
- ❌ **Learn from data** - Manual analysis is time-consuming and error-prone

**Result:** Teams react to incidents instead of proactively improving.

---

## The Solution

**Phase 14.6** automates post-GA monitoring for 7 days, providing:

✅ **Baseline Capture** - GA Day metrics become your north star
✅ **Daily Drift Detection** - Automatic comparison vs baseline (10%+ drift flagged)
✅ **Anomaly Detection** - EWMA + Z-score catch spikes in real-time
✅ **Health Scoring** - 0-100 score tells you if things are OK (90+) or critical (<50)
✅ **Regression Analysis** - Day 7 report identifies what degraded
✅ **Automated Retrospective** - Markdown + JSON report with prioritized v1.0.2 roadmap

---

## How It Works (5 Steps)

### 1. GA Day (Day 0)
Run once on launch day to capture baseline:

```bash
tars-ga-kpi --prometheus-url http://prometheus:9090
```

**Output:** `ga_kpi_summary.json` (availability, latency, error rate, costs)

---

### 2. Days 1-7: Daily Monitoring
Run automatically at 11:59 PM (cron/K8s):

```bash
tars-stability-monitor --day-number 1
```

**Output:** `day_01_summary.json` with drift % vs GA baseline

---

### 3. Continuous Anomaly Detection (Optional)
Catch spikes as they happen:

```bash
tars-anomaly-detector --z-threshold 3.0
```

**Output:** `anomaly_events.json` with Z-scores and timestamps

---

### 4. Day 7: Regression Analysis
Analyze full 7-day period:

```bash
tars-regression-analyzer
```

**Output:** `regression_summary.json` with severity rankings

---

### 5. Day 7: Generate Retrospective
Auto-generate comprehensive report:

```bash
tars-retro --auto
```

**Output:**
- `GA_7DAY_RETROSPECTIVE.md` (human-readable, Markdown)
- `GA_7DAY_RETROSPECTIVE.json` (machine-readable, for automation)

---

## What You Get (Day 7 Retrospective)

The retrospective includes:

1. **What Went Well ✅**
   - Successes (e.g., "Maintained 99.95% availability, target 99.9%")
   - Achievement percentages

2. **What Could Be Improved ⚠️**
   - Degradations (Critical/High/Medium severity)
   - Day occurred, impact, resolution status

3. **Unexpected Drifts 📊**
   - Metrics that drifted 10-30% (not regressions, but notable)
   - Trend analysis (increasing/decreasing/volatile)
   - Potential causes

4. **Cost Analysis 💰**
   - GA Day cost vs 7-day average
   - Cost breakdown by resource type
   - Optimization recommendations

5. **SLO Compliance Summary**
   - SLO burn-down (% budget consumed)
   - Days to exhaustion
   - Compliance by day (time-series)

6. **Recommendations for v1.0.2 🚀**
   - Top 10 prioritized fixes (P0/P1/P2/P3)
   - Actionable next steps

7. **Action Items**
   - Checkbox format for tracking
   - Grouped by priority (P0 = immediate, P1 = 24-48h, P2 = within sprint)

---

## Real-World Example (Test Data)

Based on test data, here's what the retrospective found:

### Successes (8 total)
- ✅ Availability: 99.950% (target: 99.9%) → **100.05% achievement**
- ✅ Cost stability: $336/day average, only 12% above GA Day

### Degradations (5 total)
- ⚠️ **High:** P99 latency spike on Day 2 (150ms → 180ms, +20%)
- ⚠️ **Medium:** Error rate increase on Day 4 (0.05% → 0.08%)

### Unexpected Drifts (3 total)
- 📊 CPU usage increased 15.7% (45.2% → 52.3%)
- 📊 Memory usage volatile (±18% fluctuation)

### Recommendations (10 total)
- [P1] Investigate and resolve 1 high-severity degradation
- [P2] Investigate 2 unexpected metric drifts
- [P3] Optimize cost (projected $9,840/month)

---

## Who Should Use This?

- **SRE Teams** - Automate post-GA monitoring
- **Engineering Teams** - Data-driven v1.0.2 planning
- **Product Teams** - Understand production health
- **On-Call Engineers** - Early warning system for degradations

---

## Quick Start (10 Minutes)

### 1. Install
```bash
pip install tars-observability
```

### 2. Verify
```bash
tars-retro --version
# Output: T.A.R.S. Observability v1.0.2-pre (Phase 14.6)
```

### 3. Run Test
```bash
bash internal_adoption/onboard.sh
```

### 4. View Sample Output
```bash
cat test_output/GA_7DAY_RETROSPECTIVE.md
```

---

## Integration Examples

### Slack Notifications
```bash
# Send retrospective to Slack on Day 7
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK \
  -d '{"text":"🎉 GA +7 Retrospective Ready!\nView: /data/output/GA_7DAY_RETROSPECTIVE.md"}'
```

### GitHub Issues (Automated)
```python
# Create issues from action items
import json, requests

with open('GA_7DAY_RETROSPECTIVE.json') as f:
    retro = json.load(f)

for item in retro['action_items']:
    if item['priority'] in ['P0', 'P1']:
        requests.post('https://api.github.com/repos/ORG/REPO/issues', json={
            'title': f"{item['priority']}: {item['description']}",
            'labels': [item['priority'], 'retrospective']
        })
```

### Cost Dashboard
```python
# Push cost data to dashboard
import json

with open('GA_7DAY_RETROSPECTIVE.json') as f:
    retro = json.load(f)

cost_data = retro['cost_analysis']
# POST to Grafana, Datadog, etc.
```

---

## FAQ

**Q: Do I need to run this for every release?**
A: Recommended for major releases (GA, major version bumps). Optional for minor releases.

**Q: What if I don't have Prometheus?**
A: You can adapt collectors to other metrics sources (CloudWatch, Datadog, etc.). See [Customization Guide](../docs/CUSTOMIZATION.md).

**Q: Can I run this on test data?**
A: Yes! We provide test data in `test_data/` for safe experimentation.

**Q: How much disk space is required?**
A: ~100 MB for 7 days of data (varies by metric volume).

**Q: Can I extend beyond 7 days?**
A: Yes, modify `DAY_NUMBER` parameter. We recommend 7 days for most teams (balances signal vs noise).

---

## Next Steps

1. ✅ Run onboarding script: `bash internal_adoption/onboard.sh`
2. ✅ Read [Quickstart Guide](../docs/PHASE14_6_QUICKSTART.md)
3. ✅ Review [Production Runbook](../docs/PHASE14_6_PRODUCTION_RUNBOOK.md)
4. ✅ Set up automation (cron/K8s) for your next GA
5. ✅ Customize alerts and dashboards

---

## Support

- **Documentation:** `docs/` directory
- **Issues:** https://github.com/veleron-dev/tars/issues
- **Slack:** #tars-support (internal)
- **Email:** tars@veleron.dev

---

**Generated:** 2025-11-26
**Version:** v1.0.2-pre
**Phase:** 14.6 - Internal Adoption Guide

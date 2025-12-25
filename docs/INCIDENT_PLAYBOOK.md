# T.A.R.S. Incident Response & Troubleshooting Playbook

**Version:** 1.0.7
**Phase:** 17 - Post-GA Observability
**Status:** Production
**Last Updated:** December 24, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Golden Incident Path (SEV-1 SLA Breach)](#golden-incident-path-sev-1-sla-breach)
3. [Incident Classification](#incident-classification)
4. [Decision Tree](#decision-tree)
5. [Triage Procedures](#triage-procedures)
6. [Containment Actions](#containment-actions)
7. [Evidence Collection](#evidence-collection)
8. [Escalation Procedures](#escalation-procedures)
9. [Post-Incident Review](#post-incident-review)

---

## Overview

This playbook provides structured guidance for responding to incidents detected by T.A.R.S. Organization Health Governance tools. It covers triage, containment, evidence collection, and escalation procedures.

### When to Use This Playbook

Use this playbook when any of the following occur:

- Exit code **102** (Critical alerts present)
- Exit code **122** (Critical cross-repo anomaly)
- Exit code **132** (Critical propagation risk)
- Exit code **142** (SLA breach detected)
- Exit code **92** (High org risk tier)
- Manual escalation from on-call engineer

---

## Golden Incident Path (SEV-1 SLA Breach)

**Use this path when Exit Code 142 is detected. Complete within 15 minutes.**

This is the fastest path to contain an SLA breach incident.

### Minute 0-2: Acknowledge

```bash
# Create incident ID
INCIDENT_ID="INC-$(date -u +%Y%m%d%H%M)"
echo "Incident $INCIDENT_ID: SLA Breach Detected"

# Notify team immediately
# [YOUR NOTIFICATION COMMAND HERE - Slack, PagerDuty, etc.]
```

### Minute 2-5: Identify Breach

```bash
# Find which SLAs are breached
python -m analytics.run_org_sla_intelligence \
    --org-report ./reports/runs/tars-run-*/org-health-report.json \
    --json | python -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('compliance_results', []):
    if r.get('status') == 'BREACHED':
        print(f\"BREACHED: {r.get('sla_id')} - {r.get('sla_type')}\")
"
```

**Record:** Which SLA(s) breached? ____________________

### Minute 5-8: Determine Root Cause

```bash
# Check breach attribution
python -m analytics.run_org_sla_intelligence \
    --org-report ./reports/runs/tars-run-*/org-health-report.json \
    --json | python -c "
import json, sys
data = json.load(sys.stdin)
for b in data.get('breaches', []):
    print(f\"SLA: {b.get('sla_id')}\")
    for rc in b.get('root_causes', [])[:3]:
        print(f\"  Cause: {rc.get('cause')} (confidence: {rc.get('confidence', 'N/A')})\")
"
```

**Record:** Primary root cause? ____________________

### Minute 8-12: Contain

| Root Cause Type | Containment Action |
|-----------------|-------------------|
| Single repo degradation | Freeze deployments to that repo |
| Cascading failure | Enable circuit breakers |
| Infrastructure issue | Contact infrastructure team |
| External dependency | Activate fallback/cache |

**Action Taken:** ____________________

### Minute 12-15: Escalate & Document

```bash
# Collect evidence
python scripts/run_full_org_governance_pipeline.py \
    --root ./org-health \
    --outdir "./incidents/${INCIDENT_ID}" \
    --print-paths

# Package for leadership
python scripts/package_executive_bundle.py \
    --run-dir "./incidents/${INCIDENT_ID}/tars-run-"*
```

**Notify:**
- [ ] Team Lead
- [ ] Engineering Manager
- [ ] VP Engineering (if customer-facing)

**Next Update ETA:** ____________________

---

## Incident Classification

### Severity Levels

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **SEV-1** | Critical - Business impact | 15 minutes | SLA breach, critical propagation |
| **SEV-2** | High - Degradation | 1 hour | At-risk SLAs, high org risk |
| **SEV-3** | Medium - Anomaly | 4 hours | Correlations found, non-critical alerts |
| **SEV-4** | Low - Informational | Next business day | Trend patterns, baseline drift |

### Incident Type Mapping

| Exit Code | Incident Type | Default Severity |
|-----------|---------------|------------------|
| 92 | High Org Risk | SEV-2 |
| 102 | Critical Alerts | SEV-1 |
| 122 | Critical Anomaly | SEV-1 |
| 132 | Propagation Risk | SEV-1 |
| 142 | SLA Breach | SEV-1 |

---

## Decision Tree

Use this decision tree to determine the appropriate response path:

```
START: Incident Detected
│
├─► Exit Code 142 (SLA Breach)?
│   └─► YES → [SEV-1] Go to: SLA Breach Response
│
├─► Exit Code 132 (Critical Propagation)?
│   └─► YES → [SEV-1] Go to: Propagation Containment
│
├─► Exit Code 122 (Critical Anomaly)?
│   └─► YES → [SEV-1] Go to: Cross-Repo Anomaly Response
│
├─► Exit Code 102 (Critical Alerts)?
│   └─► YES → [SEV-1] Go to: Critical Alert Triage
│
├─► Exit Code 92 (High Org Risk)?
│   └─► YES → [SEV-2] Go to: Org Risk Assessment
│
├─► Exit Code 141 (At-Risk SLAs)?
│   └─► YES → [SEV-2] Go to: At-Risk Monitoring
│
├─► Exit Code 121/131 (Correlations Found)?
│   └─► YES → [SEV-3] Go to: Correlation Investigation
│
└─► Other Exit Code
    └─► [SEV-4] Document and monitor
```

---

## Triage Procedures

### Step 1: Gather Context (5 minutes)

```bash
# Generate summary reports in JSON mode
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --json \
    --summary-only > incident-context.json

# Check for temporal patterns
python -m analytics.run_org_temporal_intelligence \
    --org-report ./org-health-report.json \
    --json \
    --summary-only > temporal-context.json
```

### Step 2: Identify Primary Cause

Analyze the reports to determine the root cause:

| Cause Type | Indicator | Report Section |
|------------|-----------|----------------|
| **Repo Degradation** | Single repo health decline | `org_health.repository_scores` |
| **Correlation Cluster** | Multiple repos declining together | `trend_correlation.clusters` |
| **Propagation Path** | Leader→Follower pattern | `temporal_intelligence.propagation_paths` |
| **Alert Pattern** | Repeated critical alerts | `org_alerts.alerts[severity=critical]` |

### Step 3: Assess Impact

| Question | Where to Find Answer |
|----------|---------------------|
| Which SLAs are affected? | `sla_intelligence.compliance_results` |
| Which repos are impacted? | `org_health.repository_reports` |
| What's the executive readiness tier? | `sla_intelligence.executive_readiness.tier` |
| Is propagation active? | `temporal_intelligence.temporal_anomalies` |

---

## Triage by Incident Type

### SLA Breach Response (Exit Code 142)

**Time to Contain:** 15 minutes

1. **Identify Breached SLAs**
   ```bash
   python -m analytics.run_org_sla_intelligence \
       --org-report ./org-health-report.json \
       --json | jq '.compliance_results[] | select(.status == "BREACHED")'
   ```

2. **Determine Root Cause**
   ```bash
   # Check breach attribution
   python -m analytics.run_org_sla_intelligence \
       --org-report ./org-health-report.json \
       --json | jq '.breaches[] | {sla_id, root_causes}'
   ```

3. **Assess Business Impact**
   - Review `executive_readiness.readiness_score`
   - Check `risk_narrative` for board-ready summary

4. **Immediate Actions**
   - Notify stakeholders per escalation matrix
   - Begin remediation per root cause
   - Update status page if customer-facing

### Propagation Containment (Exit Code 132)

**Time to Contain:** 15 minutes

1. **Identify Leader Repositories**
   ```bash
   python -m analytics.run_org_temporal_intelligence \
       --org-report ./org-health-report.json \
       --json | jq '.influence_scores[] | select(.classification == "LEADER")'
   ```

2. **Map Propagation Paths**
   ```bash
   python -m analytics.run_org_temporal_intelligence \
       --org-report ./org-health-report.json \
       --json | jq '.propagation_paths'
   ```

3. **Containment Strategy**
   - Isolate leader repository from downstream dependencies
   - Freeze deployments to affected repos
   - Enable enhanced monitoring on follower repos

4. **Monitor Propagation**
   - Run hourly temporal intelligence checks
   - Track `propagation_paths.impact_score` for improvement

### Cross-Repo Anomaly Response (Exit Code 122)

**Time to Contain:** 30 minutes

1. **Identify Anomaly Cluster**
   ```bash
   python -m analytics.run_org_trend_correlation \
       --org-report ./org-health-report.json \
       --json | jq '.clusters[] | select(.is_anomalous == true)'
   ```

2. **Determine Common Factor**
   - Shared infrastructure (DB, cache, queue)
   - Shared dependency (library, service)
   - Shared team/deployment pipeline
   - External factor (network, cloud provider)

3. **Containment Actions**
   - Rollback recent changes to cluster members
   - Isolate shared infrastructure if applicable
   - Enable feature flags to disable affected functionality

### Critical Alert Triage (Exit Code 102)

**Time to Contain:** 30 minutes

1. **List Critical Alerts**
   ```bash
   python -m analytics.run_org_alerts \
       --org-report ./org-health-report.json \
       --json | jq '.alerts[] | select(.severity == "CRITICAL")'
   ```

2. **Group by Alert Type**
   - Health degradation alerts
   - SLO violation alerts
   - Trend anomaly alerts
   - Escalation timeout alerts

3. **Prioritize by Impact**
   - Customer-facing issues first
   - Data integrity issues second
   - Performance degradation third

### Org Risk Assessment (Exit Code 92)

**Time to Respond:** 1 hour

1. **Calculate Risk Score**
   ```bash
   python -m analytics.run_org_health \
       --root-dir ./org-health \
       --json | jq '.risk_tier, .aggregate_score'
   ```

2. **Identify Risk Contributors**
   - Sort repos by health score (ascending)
   - Identify repos below threshold

3. **Mitigation Plan**
   - Create improvement tickets for low-health repos
   - Schedule remediation sprints
   - Update SLO targets if unrealistic

---

## Containment Actions

### Operational Containment (No Code Changes)

| Action | When to Use | Command/Process |
|--------|-------------|-----------------|
| **Freeze Deployments** | Active propagation | Block CI/CD pipelines |
| **Enable Feature Flags** | Isolate functionality | Toggle flags in config |
| **Scale Horizontally** | Performance degradation | Increase replica count |
| **Enable Circuit Breakers** | Cascading failures | Activate existing breakers |
| **Redirect Traffic** | Single point of failure | Update load balancer |
| **Enable Read-Only Mode** | Data integrity risk | Database configuration |

### Monitoring Escalation

| Action | When to Use | Implementation |
|--------|-------------|----------------|
| **Increase Check Frequency** | At-risk state | Run checks every 15 min instead of hourly |
| **Enable Verbose Logging** | Investigation | Set `TARS_LOG_LEVEL=DEBUG` |
| **Add Custom Alerts** | Specific threshold | Update alerting config |
| **Enable Tracing** | Performance issues | Enable distributed tracing |

### Communication Actions

| Action | Audience | Template |
|--------|----------|----------|
| **Status Page Update** | Customers | "We are investigating..." |
| **Slack Notification** | Engineering | "@channel Incident in progress..." |
| **Email to Leadership** | Executives | Executive summary from SLA report |
| **Bridge Call** | Cross-functional | Conference bridge for coordination |

---

## Evidence Collection

### Standard Evidence Bundle

Collect the following for every incident using the pipeline orchestrator:

**Linux/macOS:**
```bash
# Create incident-specific run using orchestrator (recommended)
INCIDENT_ID="INC-$(date +%Y%m%d%H%M)"
python scripts/run_full_org_governance_pipeline.py \
    --root ./org-health \
    --outdir "./incidents/${INCIDENT_ID}" \
    --print-paths

# Package evidence bundle
python scripts/package_executive_bundle.py \
    --run-dir "./incidents/${INCIDENT_ID}/tars-run-"* \
    --bundle-name "incident-evidence-${INCIDENT_ID}"
```

**Windows (PowerShell):**
```powershell
# Create incident-specific run using orchestrator (recommended)
$IncidentId = "INC-$((Get-Date).ToString('yyyyMMddHHmm'))"
python scripts/run_full_org_governance_pipeline.py `
    --root ./org-health `
    --outdir "./incidents/$IncidentId" `
    --print-paths

# Get the run directory and package evidence bundle
$RunDir = Get-ChildItem -Path "./incidents/$IncidentId" -Directory | Select-Object -First 1
python scripts/package_executive_bundle.py `
    --run-dir $RunDir.FullName `
    --bundle-name "incident-evidence-$IncidentId"
```

**Manual Evidence Collection (Alternative):**
```bash
# Create evidence directory
INCIDENT_ID="INC-$(date +%Y%m%d%H%M)"
mkdir -p "./incidents/${INCIDENT_ID}"
cd "./incidents/${INCIDENT_ID}"

# Collect all reports
python -m analytics.run_org_health \
    --root-dir ../../org-health \
    --output ./org-health-report.json

python -m analytics.run_org_alerts \
    --org-report ./org-health-report.json \
    --output ./org-alerts.json

python -m analytics.run_org_trend_correlation \
    --org-report ./org-health-report.json \
    --output ./trend-correlation.json

python -m analytics.run_org_temporal_intelligence \
    --org-report ./org-health-report.json \
    --trend-correlation-report ./trend-correlation.json \
    --output ./temporal-intelligence.json

python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --alerts-report ./org-alerts.json \
    --trend-correlation-report ./trend-correlation.json \
    --temporal-intelligence-report ./temporal-intelligence.json \
    --output ./sla-intelligence.json
```

### JSON and Summary-Only Modes

For incident communication, use the summary modes:

```bash
# JSON output for automated processing
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --json > sla-summary.json

# Summary-only for human communication
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --summary-only > sla-summary.txt
```

### Leadership Evidence Package

For executive escalation, prepare:

1. **Executive Summary** (from `sla_intelligence.executive_readiness`)
2. **Risk Narrative** (from `sla_intelligence.risk_narrative`)
3. **SLA Scorecard** (from `sla_intelligence.scorecard`)
4. **Trend Visualization** (if available)

---

## Escalation Procedures

### Escalation Matrix

| Severity | Initial Contact | 15 min | 30 min | 1 hour |
|----------|-----------------|--------|--------|--------|
| SEV-1 | On-call engineer | Team lead | Engineering manager | VP Engineering |
| SEV-2 | On-call engineer | Team lead | Engineering manager | - |
| SEV-3 | On-call engineer | Team lead | - | - |
| SEV-4 | Create ticket | - | - | - |

### Escalation Communication Template

```markdown
**Incident ID:** INC-YYYYMMDDHHMM
**Severity:** SEV-X
**Status:** Investigating / Mitigating / Resolved

**Summary:**
[One sentence description]

**Impact:**
- SLAs affected: [list]
- Repositories affected: [list]
- Customer impact: [description]

**Root Cause:**
[If known, otherwise "Under investigation"]

**Current Actions:**
1. [Action 1]
2. [Action 2]

**ETA to Resolution:**
[Estimate or "Unknown"]

**Next Update:**
[Time]
```

### When to Escalate

| Condition | Escalate To | Method |
|-----------|-------------|--------|
| SLA breach | VP Engineering | Phone + Email |
| Customer impact | Customer Success | Slack + Email |
| Data loss risk | Security team | Phone + Slack |
| No progress 30 min | Next tier | Phone |
| Regulatory impact | Legal + Compliance | Email (documented) |

---

## Post-Incident Review

### Review Timeline

| Activity | When | Participants |
|----------|------|--------------|
| Initial debrief | Within 24 hours | Responders |
| Blameless postmortem | Within 5 days | Team + stakeholders |
| Action item review | Weekly | Team lead |
| Trend review | Monthly | Engineering leadership |

### Postmortem Template

```markdown
# Incident Postmortem: INC-YYYYMMDDHHMM

## Summary
[Brief description]

## Timeline
| Time (UTC) | Event |
|------------|-------|
| HH:MM | [Event] |

## Root Cause
[Detailed analysis]

## Impact
- Duration: [X hours]
- SLAs breached: [list]
- Customers affected: [estimate]

## What Went Well
1. [Item]

## What Could Be Improved
1. [Item]

## Action Items
| ID | Action | Owner | Due Date | Status |
|----|--------|-------|----------|--------|
| 1 | [Action] | [Name] | [Date] | Open |

## Lessons Learned
[Summary]
```

### Metrics to Track

| Metric | Target | How to Measure |
|--------|--------|----------------|
| MTTD (Mean Time to Detect) | < 5 minutes | Alert timestamp - Issue start |
| MTTR (Mean Time to Resolve) | < 2 hours | Resolution time - Alert time |
| Recurrence Rate | < 10% | Same root cause incidents / Total |
| SLA Recovery Time | < 4 hours | SLA compliance restored |

---

## Related Documentation

- [Operator Runbook](OPS_RUNBOOK.md) - Daily and weekly operations
- [Post-GA Governance](POST_GA_GOVERNANCE.md) - Change management policy
- [SLA Intelligence Engine Guide](ORG_SLA_INTELLIGENCE_ENGINE.md) - Detailed SLA engine docs
- [Temporal Intelligence Engine Guide](ORG_TEMPORAL_INTELLIGENCE_ENGINE.md) - Temporal analysis docs

---

**Document Version:** 1.0.0
**Maintained By:** T.A.R.S. Operations Team

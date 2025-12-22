# T.A.R.S. SLA Policy Templates

**Version:** 1.0.5
**Phase:** 15 - Post-GA Operations Enablement

---

## Overview

This directory contains example SLA policy templates for use with the T.A.R.S. SLA Intelligence Engine. Each policy defines targets, thresholds, and metadata for evaluating SLA compliance.

## Available Templates

| Template | Use Case | Availability Target | Error Rate Target |
|----------|----------|---------------------|-------------------|
| `availability_default.yaml` | Standard production services | 99.5% | N/A |
| `incident_response_default.yaml` | Incident response times | N/A | N/A |
| `reliability_default.yaml` | Error rates and latency | 99.9% success | 0.1% |
| `dora_metrics_default.yaml` | Engineering performance | N/A | N/A |
| `internal_platform_strict.yaml` | Critical infrastructure | 99.95% | 0.01% |
| `startup_lenient.yaml` | MVPs and beta services | 99.0% | 1.0% |

## Quick Start

### Using a Single Policy

```bash
# Apply availability policy
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --sla-policy ./policies/examples/availability_default.yaml \
    --output ./sla-report.json
```

### Using Multiple Policies

Create a combined policy file that references multiple templates:

```yaml
# combined-policy.yaml
policies:
  - include: ./policies/examples/availability_default.yaml
  - include: ./policies/examples/reliability_default.yaml
  - include: ./policies/examples/dora_metrics_default.yaml
```

Then run:

```bash
python -m analytics.run_org_sla_intelligence \
    --org-report ./org-health-report.json \
    --sla-policy ./combined-policy.yaml
```

## Policy Structure

Each policy YAML file contains:

```yaml
# Required metadata
policy_id: "unique-policy-id"
name: "Human Readable Name"
type: "availability|reliability|incident_response|dora|platform|development"
version: "1.0.0"
description: "What this policy covers"

# Target thresholds
targets:
  metric_name:
    target: 99.5          # The target value
    at_risk_threshold: 99.0   # Threshold for "at risk" status
    breach_threshold: 98.0    # Threshold for "breach" status
    unit: "percent"           # Unit of measurement
    lower_is_better: false    # Optional: invert threshold comparison
    description: "Description of this metric"

# Evaluation windows (in days)
windows:
  - 7
  - 30
  - 90

# Severity mapping
severity:
  compliant: "info"
  at_risk: "warning"
  breached: "critical"

# Business context
business_impact:
  category: "operations|reliability|engineering_excellence|infrastructure|development"
  customer_facing: true|false
  revenue_impact: "low|medium|high|very_high"
  description: "Business context for this SLA"

# Remediation guidance
remediation:
  at_risk:
    - "Action 1"
    - "Action 2"
  breach:
    - "Action 1"
    - "Action 2"

# Categorization
tags:
  - "tag1"
  - "tag2"
```

## Choosing a Policy

### Decision Matrix

| Scenario | Recommended Policy |
|----------|-------------------|
| New MVP/beta service | `startup_lenient.yaml` |
| Standard production service | `availability_default.yaml` + `reliability_default.yaml` |
| Customer-facing API | `availability_default.yaml` + `incident_response_default.yaml` |
| Internal platform/database | `internal_platform_strict.yaml` |
| Engineering team metrics | `dora_metrics_default.yaml` |

### Graduation Path

Services should graduate through policies as they mature:

```
startup_lenient → availability_default → internal_platform_strict
```

Criteria for graduation:
- Service has been stable for 30+ days
- User base has grown beyond beta
- Service is customer-facing
- Revenue depends on availability

## Customizing Policies

### Creating Custom Policies

1. Copy the closest template:
   ```bash
   cp policies/examples/availability_default.yaml policies/custom/my-service.yaml
   ```

2. Modify targets for your requirements:
   ```yaml
   targets:
     uptime:
       target: 99.9        # Increase from 99.5
       at_risk_threshold: 99.5
       breach_threshold: 99.0
   ```

3. Update metadata:
   ```yaml
   policy_id: "my-service-v1"
   name: "My Service SLA"
   ```

4. Use your custom policy:
   ```bash
   python -m analytics.run_org_sla_intelligence \
       --sla-policy ./policies/custom/my-service.yaml
   ```

### Best Practices

1. **Start lenient, tighten over time** - Begin with achievable targets
2. **Use multiple windows** - 7/30/90 day windows catch different patterns
3. **Set realistic breach thresholds** - Breaches should be rare but meaningful
4. **Include remediation guidance** - Make policies actionable
5. **Tag appropriately** - Tags help with filtering and reporting
6. **Document business impact** - Helps prioritization during incidents

## Validation

Validate policy YAML syntax:

```bash
python -c "import yaml; yaml.safe_load(open('policies/examples/availability_default.yaml'))"
```

Validate all policies:

```bash
for f in policies/examples/*.yaml; do
    python -c "import yaml; yaml.safe_load(open('$f'))" && echo "OK: $f"
done
```

## Related Documentation

- [SLA Intelligence Engine Guide](../../docs/ORG_SLA_INTELLIGENCE_ENGINE.md)
- [Operator Runbook](../../docs/OPS_RUNBOOK.md)
- [Incident Playbook](../../docs/INCIDENT_PLAYBOOK.md)

---

**Maintained By:** T.A.R.S. Operations Team

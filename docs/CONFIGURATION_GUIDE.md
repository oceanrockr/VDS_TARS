# T.A.R.S. Configuration Guide

**Version:** 1.0.0
**Phase:** 18 - Ops Integrations, Config Management
**Status:** Production

---

## Overview

T.A.R.S. v1.0.8 introduces unified configuration file support for all governance tools. This allows operators to define default settings in a single file, reducing command-line complexity and ensuring consistent configurations across runs.

**Key Benefits:**
- Reduce long CLI command lines to simple invocations
- Ensure consistent settings across daily/weekly operations
- Enable config-as-code for GitOps workflows
- Support multiple environments (dev, staging, prod)

---

## Configuration Precedence

T.A.R.S. configuration follows this precedence order (highest to lowest):

1. **CLI Arguments** - Explicit command-line flags always win
2. **--config <path>** - Explicit config file path
3. **TARS_CONFIG** - Environment variable pointing to config file
4. **./tars.yml** or **./tars.yaml** - Default YAML config in working directory
5. **./tars.json** - Default JSON config in working directory
6. **Built-in Defaults** - Hardcoded safe defaults

---

## File Formats

### YAML Format (Recommended)

Requires `PyYAML` package (`pip install pyyaml`).

```yaml
# tars.yml - T.A.R.S. Configuration
# Minimal example for daily operations

orchestrator:
  root: "./org-health"
  outdir: "./reports/runs"
  format: "flat"
  print_paths: true

packager:
  output_dir: "./release/executive"
  tar: true
  checksums: true
```

### JSON Format

Works out of the box, no extra dependencies.

```json
{
  "orchestrator": {
    "root": "./org-health",
    "outdir": "./reports/runs",
    "format": "flat",
    "print_paths": true
  },
  "packager": {
    "output_dir": "./release/executive",
    "tar": true,
    "checksums": true
  }
}
```

---

## Configuration Sections

### orchestrator

Controls the pipeline orchestrator behavior.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `root` | string | `"./org-health"` | Root directory for org health data |
| `outdir` | string | `"./reports/runs"` | Base output directory for reports |
| `format` | string | `"flat"` | Output format: `flat` or `structured` |
| `print_paths` | boolean | `false` | Print artifact paths after run |
| `fail_on_breach` | boolean | `false` | Exit 142 on SLA breach |
| `fail_on_critical` | boolean | `false` | Exit non-zero on critical conditions |
| `with_narrative` | boolean | `false` | Generate executive narrative |
| `sla_policy` | string | `null` | Path to SLA policy YAML file |
| `windows` | array | `[]` | Evaluation windows in days |

### packager

Controls the executive bundle packager.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | string | `"./release/executive"` | Output directory for bundles |
| `bundle_name_template` | string | `"tars-exec-bundle-{version}-{timestamp}"` | Bundle naming template |
| `tar` | boolean | `false` | Create tar.gz archive |
| `zip` | boolean | `true` | Create ZIP archive |
| `checksums` | boolean | `true` | Generate SHA-256 checksums |
| `manifest` | boolean | `true` | Generate manifest JSON |
| `compliance_index` | boolean | `true` | Generate compliance index |
| `signing.enabled` | boolean | `false` | Enable GPG signing |
| `signing.gpg_key_id` | string | `null` | GPG key ID for signing |

### retention

Controls artifact retention and cleanup.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `false` | Enable retention management |
| `days_hot` | integer | `30` | Days to keep in hot storage |
| `days_warm` | integer | `90` | Days to keep in warm storage |
| `days_archive` | integer | `365` | Days to keep in archive |
| `compress_after` | integer | `30` | Days before compressing to tar.gz |

### notify

Controls notification hooks.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `false` | Enable notifications |
| `exit_codes` | array | `[92, 102, 122, 132, 142]` | Exit codes that trigger notifications |
| `webhook_url` | string | `null` | Generic webhook URL |
| `slack_webhook_url` | string | `null` | Slack webhook URL |
| `pagerduty_routing_key` | string | `null` | PagerDuty routing key |

---

## Example Configurations

### Minimal Daily Operations

```yaml
# tars.yml - Minimal config for daily runs
orchestrator:
  root: "./org-health"
  print_paths: true
```

Run with:
```bash
python scripts/run_full_org_governance_pipeline.py
```

### CI/CD Pipeline Config

```yaml
# tars.yml - CI/CD configuration with strict settings
orchestrator:
  root: "./org-health"
  outdir: "./reports/runs"
  format: "structured"
  print_paths: true
  fail_on_breach: true
  fail_on_critical: true
  with_narrative: true

packager:
  output_dir: "./release/executive"
  tar: true
  zip: true
  checksums: true
  compliance_index: true

notify:
  enabled: true
  exit_codes: [92, 102, 122, 132, 142]
  webhook_url: "${NOTIFICATION_WEBHOOK_URL}"
```

### Incident Response Config

```yaml
# tars-incident.yml - Config for incident response mode
orchestrator:
  root: "./org-health"
  format: "structured"
  print_paths: true
  with_narrative: true
  # Use strict SLA policy during incidents
  sla_policy: "./policies/examples/internal_platform_strict.yaml"

packager:
  tar: true
  checksums: true
  compliance_index: true
  signing:
    enabled: true
    gpg_key_id: "INCIDENT_RESPONSE_KEY"

retention:
  # Extend retention during incident
  enabled: true
  days_hot: 90
  days_archive: 730  # 2 years
```

Use with:
```bash
python scripts/run_full_org_governance_pipeline.py --config ./tars-incident.yml
```

### Multi-Environment Setup

**Development (tars.dev.yml):**
```yaml
orchestrator:
  root: "./test-data/org-health"
  outdir: "./test-reports"
  format: "flat"
  print_paths: true
```

**Staging (tars.staging.yml):**
```yaml
orchestrator:
  root: "./staging-org-health"
  outdir: "./staging-reports"
  format: "structured"
  print_paths: true
  with_narrative: true
```

**Production (tars.prod.yml):**
```yaml
orchestrator:
  root: "/data/org-health"
  outdir: "/data/reports/runs"
  format: "structured"
  print_paths: true
  fail_on_breach: true
  with_narrative: true

packager:
  output_dir: "/data/release/executive"
  tar: true
  checksums: true
  signing:
    enabled: true
    gpg_key_id: "PROD_SIGNING_KEY"

retention:
  enabled: true
  days_hot: 30
  days_warm: 90
  days_archive: 365

notify:
  enabled: true
  webhook_url: "${PROD_WEBHOOK_URL}"
  pagerduty_routing_key: "${PAGERDUTY_KEY}"
```

---

## Environment Variable Support

Configuration values can reference environment variables:

```yaml
notify:
  webhook_url: "${NOTIFICATION_WEBHOOK_URL}"
  pagerduty_routing_key: "${PAGERDUTY_KEY}"

packager:
  signing:
    gpg_key_id: "${GPG_KEY_ID}"
```

**Note:** Environment variable expansion is handled at the application level, not during config parsing.

---

## Usage Examples

### Basic Config-Driven Run

```bash
# Uses ./tars.yml automatically
python scripts/run_full_org_governance_pipeline.py
```

### Explicit Config Path

```bash
# Use specific config file
python scripts/run_full_org_governance_pipeline.py --config ./config/prod.yml
```

### Environment Variable Config

```bash
# Set config via environment
export TARS_CONFIG="./config/staging.yml"
python scripts/run_full_org_governance_pipeline.py
```

### Config + CLI Override

CLI arguments always override config file values:

```bash
# Config says format: flat, CLI overrides to structured
python scripts/run_full_org_governance_pipeline.py \
    --config ./tars.yml \
    --format structured \
    --fail-on-breach
```

### Packager with Config

```bash
# Packager respects same config file
python scripts/package_executive_bundle.py \
    --config ./tars.yml \
    --run-dir ./reports/runs/tars-run-20251225-080000
```

---

## Validation and Errors

The config loader performs validation:

1. **Unknown Keys** - Warns but continues (allows future expansion)
2. **Invalid Values** - Errors on clearly invalid values (e.g., negative days)
3. **Missing Files** - Warns and uses defaults (does not fail)
4. **Parse Errors** - Warns and uses defaults (does not fail)

### Error Messages

```
# YAML not installed
WARNING: YAML config file found (tars.yml) but PyYAML is not installed.
Install with: pip install pyyaml

# Parse error
WARNING: Config parse failed, using defaults: YAML parse error in tars.yml: ...

# Invalid value
ERROR: Invalid orchestrator.format: invalid. Must be 'flat' or 'structured'.

# Unknown key (warning only)
WARNING: Unknown config key (ignored): custom_setting
```

---

## Best Practices

1. **Version Control Your Config** - Keep `tars.yml` in git for reproducibility
2. **Use Environment-Specific Files** - `tars.dev.yml`, `tars.staging.yml`, `tars.prod.yml`
3. **Don't Hardcode Secrets** - Use environment variables for webhook URLs and keys
4. **Start Minimal** - Add config options as needed
5. **Document Changes** - Comment your config file for team clarity

---

## Troubleshooting

### YAML Not Recognized

```bash
pip install pyyaml
```

### Config Not Loading

```bash
# Enable debug logging
export TARS_LOG_LEVEL=DEBUG
python scripts/run_full_org_governance_pipeline.py
```

### Verify Config Path

```bash
# Print resolved config path
python -c "from scripts.tars_config import TarsConfigLoader; l=TarsConfigLoader(); l.load(); print(l.resolved_path)"
```

---

## Related Documentation

- [Operator Runbook](OPS_RUNBOOK.md) - Daily operations guide
- [Incident Playbook](INCIDENT_PLAYBOOK.md) - Incident response procedures
- [SLA Policy Templates](../policies/examples/) - Ready-to-use SLA policies

---

**Document Version:** 1.0.0
**Last Updated:** December 25, 2025

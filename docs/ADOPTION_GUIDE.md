# T.A.R.S. Adoption Guide

**Version:** 1.0.0
**Phase:** 19 - Production Ops Maturity
**Status:** Production

---

## Overview

This guide provides a minimal rollout checklist for adopting T.A.R.S. organization health governance in your environment. It covers secrets management, signing posture, retention policies, and common deployment patterns.

---

## Quick Start Checklist

### Phase 1: Initial Setup (Day 1)

- [ ] Clone the T.A.R.S. repository
- [ ] Install Python dependencies: `pip install -r requirements-dev.txt`
- [ ] Create org-health data directory: `mkdir -p ./org-health`
- [ ] Run a test daily check: `python scripts/tars_ops.py daily`
- [ ] Verify output in `./reports/runs/`

### Phase 2: Configuration (Day 1-2)

- [ ] Copy example config: `cp examples/configs/tars.ci.yml .github/config/tars.ci.yml`
- [ ] Customize orchestrator settings for your environment
- [ ] Set up environment variable placeholders for secrets
- [ ] Test with: `python scripts/tars_ops.py daily --config .github/config/tars.ci.yml`

### Phase 3: CI/CD Integration (Day 2-3)

- [ ] Copy workflow: `cp .github/workflows/tars_daily_ops.yml` to your repo
- [ ] Configure GitHub Actions secrets (see below)
- [ ] Run workflow manually to verify
- [ ] Enable scheduled runs

### Phase 4: Notifications (Optional, Day 3)

- [ ] Set up webhook or Slack integration
- [ ] Add secrets to CI/CD
- [ ] Test notification with: `python scripts/notify_ops.py --dry-run`

### Phase 5: Production Hardening (Week 2)

- [ ] Review and customize SLA policies
- [ ] Configure GPG signing for evidence bundles
- [ ] Set up retention management
- [ ] Document on-call procedures with team

---

## Secrets Management

### GitHub Actions Secrets

Set these secrets in your repository (Settings > Secrets and variables > Actions):

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `NOTIFICATION_WEBHOOK_URL` | Generic webhook for alerts | Optional |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL | Optional |
| `PAGERDUTY_ROUTING_KEY` | PagerDuty service routing key | Optional |
| `GPG_PRIVATE_KEY` | GPG private key for signing | Optional |
| `GPG_KEY_ID` | GPG key ID for signing | Optional |

### Environment Variable Syntax

In config files, reference secrets with `${VAR_NAME}` syntax:

```yaml
notify:
  webhook_url: "${NOTIFICATION_WEBHOOK_URL}"
  slack_webhook_url: "${SLACK_WEBHOOK_URL}"
```

The config loader expands these at runtime from `os.environ`.

### Security Best Practices

1. **Never commit secrets** - Use `.gitignore` for `.env` files
2. **Rotate regularly** - Update API keys and webhooks quarterly
3. **Limit scope** - Use read-only tokens where possible
4. **Audit access** - Review secret access logs monthly

---

## GPG Signing Posture

### When to Sign

| Scenario | Sign? | Rationale |
|----------|-------|-----------|
| Daily health checks | No | Speed over formality |
| Weekly trend reports | Optional | Team preference |
| Incident evidence | Yes | Chain of custody |
| Compliance audits | Yes | Regulatory requirement |
| External sharing | Yes | Integrity verification |

### Setting Up GPG

1. **Generate a key** (if you don't have one):
   ```bash
   gpg --full-generate-key
   # Choose RSA and RSA, 4096 bits, no expiration for automation
   ```

2. **Export for CI/CD**:
   ```bash
   gpg --export-secret-keys --armor YOUR_KEY_ID > gpg-private.key
   # Add gpg-private.key content to GPG_PRIVATE_KEY secret
   ```

3. **Import in workflow**:
   ```yaml
   - name: Import GPG Key
     run: |
       echo "${{ secrets.GPG_PRIVATE_KEY }}" | gpg --import
   ```

4. **Configure in tars.yml**:
   ```yaml
   packager:
     signing:
       enabled: true
       gpg_key_id: "${GPG_KEY_ID}"
   ```

### Verification Commands

Verify a signed bundle:

```bash
# List files in bundle
unzip -l tars-exec-bundle-*.zip

# Verify signature
gpg --verify tars-exec-bundle-*.zip.sig tars-exec-bundle-*.zip

# Check checksums
sha256sum -c SHA256SUMS.txt
```

---

## Retention Tiers

### Recommended Configurations

**Standard Operations:**
```yaml
retention:
  enabled: true
  days_hot: 30    # Quick access for recent runs
  days_warm: 90   # Compressed, accessible
  days_archive: 365 # Long-term storage
```

**Compliance-Heavy (SOC 2, ISO 27001):**
```yaml
retention:
  enabled: true
  days_hot: 90
  days_warm: 365
  days_archive: 2555  # 7 years
```

**Minimal (Dev/Test):**
```yaml
retention:
  enabled: true
  days_hot: 7
  days_warm: 30
  days_archive: 90
```

### Running Retention Management

```bash
# Dry-run (safe, shows what would happen)
python scripts/retention_manage.py --root ./reports/runs --summary-only

# Apply changes (careful!)
python scripts/retention_manage.py --root ./reports/runs --force
```

---

## Common Deployment Patterns

### Pattern 1: Daily Health Check Only

Simplest adoption - run daily checks and review manually.

```yaml
# .github/workflows/tars-health.yml
on:
  schedule:
    - cron: '0 8 * * *'
jobs:
  health:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements-dev.txt
      - run: python scripts/tars_ops.py daily
```

### Pattern 2: Daily + Weekly + Alerts

Full operational monitoring with notifications.

```yaml
# Daily at 08:00 UTC
on:
  schedule:
    - cron: '0 8 * * *'
    - cron: '0 10 * * 1'  # Weekly on Monday
```

### Pattern 3: Incident-Driven

Only run during active incidents for evidence collection.

```bash
# Manual trigger during incident
python scripts/tars_ops.py incident \
  --config examples/configs/tars.incident.yml \
  --incident-id INC-12345 \
  --sign
```

---

## Troubleshooting

### Common Issues

**Config not loading:**
```bash
# Check config path
export TARS_LOG_LEVEL=DEBUG
python scripts/tars_ops.py daily --config ./path/to/config.yml
```

**Environment variables not expanding:**
```bash
# Verify variable is set
echo $NOTIFICATION_WEBHOOK_URL

# Check expansion in config
python -c "
from scripts.tars_config import TarsConfigLoader
loader = TarsConfigLoader(config_path='./tars.yml')
config = loader.load()
print(config.get('notify', {}).get('webhook_url'))
"
```

**GPG signing fails:**
```bash
# Check GPG keys
gpg --list-secret-keys

# Test signing manually
echo "test" | gpg --sign --armor
```

### Getting Help

- Review [docs/OPS_RUNBOOK.md](OPS_RUNBOOK.md) for operational guidance
- Check [docs/INCIDENT_PLAYBOOK.md](INCIDENT_PLAYBOOK.md) for incident response
- See [docs/CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for config reference

---

## Next Steps

After completing the adoption checklist:

1. **Customize SLA policies** - Review `policies/examples/` and create org-specific policies
2. **Train team** - Share the OPS_RUNBOOK with on-call engineers
3. **Set up dashboards** - Import Grafana dashboards for visualization
4. **Iterate** - Start with daily checks, add complexity as needed

---

**Document Version:** 1.0.0
**Last Updated:** December 26, 2025

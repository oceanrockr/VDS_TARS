# T.A.R.S. Support Bundle Guide

**Version:** v1.0.11 (GA)
**Phase:** 23 - Field Debugging Support
**Target:** Home Network Operators & Support Personnel

---

## Overview

The Support Bundle Generator creates a comprehensive diagnostic package for troubleshooting T.A.R.S. issues. All sensitive information (passwords, tokens, secrets) is automatically redacted, making the bundle safe to share.

---

## Quick Start

```bash
# Generate support bundle
./deploy/generate-support-bundle.sh

# Output: support/tars-support-bundle-YYYYMMDD_HHMMSS.tar.gz
```

---

## What's Included

| Category | Files | Purpose |
|----------|-------|---------|
| System Info | `system-info.txt` | Host specs, RAM, disk, GPU |
| Container Status | `container-status.txt` | Docker container health |
| Logs | `logs/*.log` | Container logs (redacted) |
| Configuration | `config/` | Env vars, compose files (redacted) |
| API Health | `api-health.txt` | Endpoint responses |
| Validation | `validation/` | Script output results |
| Mounts | `mount-status.txt` | NAS mount info |
| Network | `network-status.txt` | Ports, firewall |
| Manifest | `MANIFEST.txt` | Bundle contents list |

---

## Security Guarantees

The bundle generator automatically redacts:

| Pattern | Example | Becomes |
|---------|---------|---------|
| Passwords | `PASSWORD=abc123` | `PASSWORD=<REDACTED>` |
| JWT Tokens | `Bearer eyJhbG...` | `Bearer <REDACTED>` |
| API Keys | `API_KEY=sk-xxx` | `API_KEY=<REDACTED>` |
| Connection Strings | `postgres://user:pass@host` | `postgres://user:<REDACTED>@host` |
| Long Hashes | `abc123def456...` (32+ chars) | `<HASH_REDACTED>` |

**Safe to share:** The bundle contains NO recoverable secrets.

---

## Usage Options

### Basic Generation

```bash
./deploy/generate-support-bundle.sh
```

Creates bundle in `./support/` directory.

### Custom Output Directory

```bash
./deploy/generate-support-bundle.sh --output-dir /tmp
```

Creates bundle in specified directory.

### Extended Logs

```bash
./deploy/generate-support-bundle.sh --include-all-logs
```

Includes 1000 log lines per container instead of 200.

---

## Bundle Contents Detail

### system-info.txt

```
=== Host Information ===
Hostname: tars-home
OS: Ubuntu 22.04.3 LTS
Kernel: 5.15.0-91-generic

=== CPU Information ===
model name: AMD Ryzen 9 5900X
cpu cores: 12

=== Memory Information ===
              total        used        free
Mem:           31Gi       8.2Gi        18Gi

=== GPU Information ===
NVIDIA GeForce RTX 3080, 10240 MiB, 8192 MiB free, 0%, 45C
```

### container-status.txt

```
=== Container Health Status ===
tars-home-backend: Status=running, Health=healthy, Restarts=0
tars-home-ollama: Status=running, Health=healthy, Restarts=0
tars-home-chromadb: Status=running, Health=healthy, Restarts=0
tars-home-redis: Status=running, Health=healthy, Restarts=0
tars-home-postgres: Status=running, Health=healthy, Restarts=0
```

### logs/tars-home-backend.log

```
=== tars-home-backend logs (last 200 lines) ===
2025-12-27 10:15:32 INFO Starting T.A.R.S. Backend v0.3.0-alpha
2025-12-27 10:15:33 INFO Redis cache connected - Status: healthy
2025-12-27 10:15:34 INFO PostgreSQL database initialized
...
```

### config/tars-home.env.redacted

```
TARS_POSTGRES_PASSWORD=<REDACTED>
TARS_JWT_SECRET=<REDACTED>
TARS_POSTGRES_DB=tars_home
TARS_POSTGRES_USER=tars
OLLAMA_MODEL=mistral:7b-instruct
LOG_LEVEL=INFO
```

---

## When to Generate a Bundle

Generate a support bundle when:

1. **Services won't start** - Bundle captures container status and logs
2. **Slow inference** - Bundle includes GPU status and resource usage
3. **RAG failures** - Bundle includes ChromaDB stats and mount status
4. **Authentication issues** - Bundle includes API health (tokens redacted)
5. **Before contacting support** - Always include a bundle

---

## Sharing the Bundle

### With Support Team

1. Generate the bundle
2. Share both files:
   - `tars-support-bundle-YYYYMMDD_HHMMSS.tar.gz`
   - `tars-support-bundle-YYYYMMDD_HHMMSS.tar.gz.sha256`
3. Include a description of your issue

### Verifying Integrity

Recipient can verify the bundle wasn't corrupted:

```bash
sha256sum -c tars-support-bundle-*.tar.gz.sha256
```

### Extracting the Bundle

```bash
tar -xzf tars-support-bundle-*.tar.gz
cd tars-support-bundle-*/
```

---

## Troubleshooting the Generator

### "Permission denied"

```bash
chmod +x deploy/generate-support-bundle.sh
```

### "Docker not running"

```bash
sudo systemctl start docker
```

### "jq: command not found"

```bash
sudo apt install jq
```

### Bundle too large

Use default settings (not `--include-all-logs`) for smaller bundles.

---

## Reading the Bundle

### Quick Diagnosis Workflow

1. **Check container status first:**
   ```bash
   cat container-status.txt
   ```
   Look for: unhealthy, restarting, high restart count

2. **Check API health:**
   ```bash
   cat api-health.txt
   ```
   Look for: error responses, missing endpoints

3. **Review validation results:**
   ```bash
   cat validation/config-doctor.txt
   ```
   Look for: [FAIL] entries

4. **Examine relevant logs:**
   ```bash
   cat logs/tars-home-backend.log | grep -i error
   ```

### Common Patterns

| Symptom in Bundle | Likely Cause | Fix |
|-------------------|--------------|-----|
| Container "unhealthy" | Service crashed | Check logs, restart |
| High restart count | Configuration issue | Run config-doctor |
| GPU "not detected" | Driver issue | Reinstall NVIDIA toolkit |
| API "could not reach" | Port conflict | Check network-status.txt |
| NAS "not mounted" | Network/auth issue | Run mount-nas.sh setup |

---

## Privacy Considerations

### What IS Collected

- System specifications (CPU, RAM, GPU)
- Container names and status
- Log messages (errors, warnings)
- Configuration structure (not values)
- API response structure
- Port and network configuration

### What IS NOT Collected

- Passwords, tokens, or secrets
- User credentials
- Private documents
- Chat history or conversations
- Personal identifiable information
- Network credentials

---

## Retention

Support bundles are stored in `./support/` by default. Consider:

- Deleting old bundles after issues are resolved
- Not committing bundles to version control
- Encrypting bundles if storing long-term

```bash
# Clean up old bundles
rm -f support/tars-support-bundle-*.tar.gz*
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [CONFIG_DOCTOR.md](CONFIG_DOCTOR.md) | Configuration validation |
| [GO_NO_GO_HOME.md](GO_NO_GO_HOME.md) | Daily operation checklist |
| [INSTALL_HOME.md](INSTALL_HOME.md) | Installation guide |

---

**Last Updated:** December 27, 2025
**Version:** v1.0.11 (GA)
**Phase:** 23 - Field Debugging Support

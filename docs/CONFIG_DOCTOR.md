# T.A.R.S. Configuration Doctor Guide

**Version:** v1.0.11 (GA)
**Phase:** 23 - Configuration UX Hardening
**Target:** Home Network Operators

---

## Overview

The Configuration Doctor is a diagnostic tool that validates your T.A.R.S. configuration and provides actionable fixes. It's designed for non-technical operators who need to quickly identify and resolve issues.

**Key Features:**
- Validates all required environment variables
- Checks NAS connectivity and mount status
- Verifies port availability
- Confirms GPU detection and container access
- Tests LLM model availability
- Provides exact fix commands for each issue

---

## Quick Start

```bash
# Run all configuration checks
./deploy/config-doctor.sh

# Attempt automatic fixes
./deploy/config-doctor.sh --fix

# Only show errors (quiet mode)
./deploy/config-doctor.sh --quiet
```

---

## What It Checks

### 1. Environment File

| Check | Pass Criteria | Fix |
|-------|---------------|-----|
| File exists | `tars-home.env` present | Copy from template |
| Syntax valid | No parse errors | Check file format |

### 2. Required Variables

| Variable | Requirement | Fix |
|----------|-------------|-----|
| `TARS_POSTGRES_PASSWORD` | Set, not placeholder, 16+ chars | Generate with `openssl rand -base64 32` |
| `TARS_JWT_SECRET` | Set, not placeholder, 32+ chars | Generate with `openssl rand -hex 64` |
| `TARS_POSTGRES_DB` | Set (default: tars_home) | Optional |
| `TARS_POSTGRES_USER` | Set (default: tars) | Optional |
| `OLLAMA_MODEL` | Set (default: mistral:7b-instruct) | Optional |

### 3. NAS Configuration

| Check | Pass Criteria | Fix |
|-------|---------------|-----|
| Mount point exists | Directory exists | `sudo mkdir -p /mnt/llm_docs` |
| NAS mounted | `mountpoint` returns true | `./deploy/mount-nas.sh setup` |
| NAS reachable | Ping succeeds | Check network/power |

### 4. Port Availability

| Port | Service | Fix if Busy |
|------|---------|-------------|
| 8000 | Backend API | Stop conflicting service or change port |
| 8001 | ChromaDB | Stop conflicting service or change port |
| 11434 | Ollama | Stop conflicting service or change port |
| 6379 | Redis | Stop conflicting service or change port |
| 5432 | PostgreSQL | Stop conflicting service or change port |
| 3000 | Frontend | Optional, only if using React UI |

### 5. GPU Configuration

| Check | Pass Criteria | Impact if Missing |
|-------|---------------|-------------------|
| Host GPU | `nvidia-smi` shows GPU | CPU inference (slower) |
| NVIDIA Toolkit | Docker runtime has nvidia | GPU unavailable in containers |
| Container access | Container can run nvidia-smi | CPU inference in container |

### 6. LLM Models

| Check | Pass Criteria | Fix |
|-------|---------------|-----|
| Ollama running | Container is up | Start services |
| Models installed | At least one model | `docker exec tars-home-ollama ollama pull <model>` |
| Configured model | Expected model present | Pull the configured model |

### 7. Service Health

| Check | Pass Criteria | Fix |
|-------|---------------|-----|
| Containers healthy | Health check passes | Check logs, restart |
| API responding | /health returns 200 | Check backend logs |
| All services ready | /ready returns "ready" | Check individual services |

### 8. Security

| Check | Pass Criteria | Fix |
|-------|---------------|-----|
| Non-root user | Not running as root | Use regular user |
| Docker group | User in docker group | `sudo usermod -aG docker $USER` |
| Env file perms | 600 or 640 | `chmod 600 deploy/tars-home.env` |

---

## Understanding Output

### Status Indicators

| Indicator | Meaning | Action |
|-----------|---------|--------|
| `[OK]` | Check passed | None needed |
| `[WARN]` | Non-critical issue | Review, fix if convenient |
| `[FAIL]` | Critical issue | Must fix before running |
| `[INFO]` | Informational | No action needed |
| `Fix:` | Suggested command | Copy and run |

### Example Output

```
=== Required Environment Variables ===
[OK] TARS_POSTGRES_PASSWORD is set (32 chars)
[OK] TARS_JWT_SECRET is set (128 chars)
[WARN] OLLAMA_MODEL not set (default: mistral:7b-instruct)

=== Port Availability ===
[OK] Port 8000 (Backend API): Available
[WARN] Port 5432 (PostgreSQL): In use by another process
       Fix: ss -tlnp | grep :5432  # Find process
       Fix: Change port in deploy/tars-home.env if needed
```

---

## Common Scenarios

### Scenario 1: Fresh Install

Run after cloning the repository:

```bash
./deploy/config-doctor.sh --fix
```

Expected: Creates environment file, generates secrets, identifies missing components.

### Scenario 2: Services Won't Start

Run when `start-tars-home.sh` fails:

```bash
./deploy/config-doctor.sh
```

Look for: Port conflicts, missing variables, Docker issues.

### Scenario 3: Slow Inference

Run when queries take too long:

```bash
./deploy/config-doctor.sh | grep -A5 "GPU Configuration"
```

Look for: GPU not detected, container GPU access failed.

### Scenario 4: RAG Returns No Results

Run when document search fails:

```bash
./deploy/config-doctor.sh | grep -A10 "NAS Configuration"
```

Look for: NAS not mounted, no documents found.

---

## Fix Mode

The `--fix` flag attempts to automatically resolve common issues:

```bash
./deploy/config-doctor.sh --fix
```

**Auto-fixable issues:**
- Create environment file from template
- Create NAS mount point directory
- Set correct permissions on env file
- Pull missing LLM models

**Not auto-fixable (requires manual action):**
- Install Docker or Docker Compose
- Install NVIDIA drivers
- Resolve port conflicts
- Configure NAS credentials
- Generate secure secrets (prompted)

---

## Integration with Other Tools

### Before Installation

```bash
# Check prerequisites before installing
./deploy/config-doctor.sh

# If all checks pass, proceed with install
./deploy/install-tars-home.sh
```

### After Service Issues

```bash
# Services misbehaving?
./deploy/config-doctor.sh

# Generate support bundle if needed
./deploy/generate-support-bundle.sh
```

### Regular Maintenance

```bash
# Weekly health check
./deploy/config-doctor.sh --quiet

# Only shows warnings and errors
```

---

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | All checks passed | Ready to run |
| 1 | One or more failures | Fix before running |

Use in scripts:

```bash
if ./deploy/config-doctor.sh --quiet; then
    ./deploy/start-tars-home.sh start
else
    echo "Configuration issues detected"
    exit 1
fi
```

---

## Troubleshooting the Doctor Itself

### "Permission denied"

```bash
# Make script executable
chmod +x deploy/config-doctor.sh
```

### "command not found: jq"

```bash
# Install jq for JSON parsing
sudo apt install jq
```

### "Cannot connect to Docker daemon"

```bash
# Start Docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [INSTALL_HOME.md](INSTALL_HOME.md) | Complete installation guide |
| [GO_NO_GO_HOME.md](GO_NO_GO_HOME.md) | Daily operation checklist |
| [SUPPORT_BUNDLE.md](SUPPORT_BUNDLE.md) | Generating support bundles |

---

**Last Updated:** December 27, 2025
**Version:** v1.0.11 (GA)
**Phase:** 23 - Configuration UX Hardening

# T.A.R.S. Home Network Installation Guide

**Version:** v1.0.11 (GA)
**Phase:** 23 - Local Machine Rollout
**Target:** Ubuntu 22.04 LTS with optional NVIDIA GPU

---

## Quick Start (One Command)

```bash
# Clone and install
git clone https://github.com/oceanrockr/VDS_TARS.git
cd VDS_TARS
chmod +x deploy/install-tars-home.sh
./deploy/install-tars-home.sh
```

That's it. The installer handles everything automatically.

---

## What the Installer Does

The one-command installer performs these steps:

| Step | Description | Duration |
|------|-------------|----------|
| 1 | Verify prerequisites (Docker, Compose, GPU) | ~10s |
| 2 | Generate secure secrets and create env file | ~5s |
| 3 | Configure NAS mount (optional) | ~30s |
| 4 | Pull Docker images | 2-5 min |
| 5 | Start all services | ~30s |
| 6 | Wait for services to initialize | 1-2 min |
| 7 | Pull LLM model (first run only) | 5-15 min |
| 8 | Run validation suite | ~60s |

**Total first-run time:** 10-25 minutes (mostly model download)
**Subsequent runs:** 2-5 minutes

---

## Prerequisites

### Required

| Component | Minimum | Recommended | Check Command |
|-----------|---------|-------------|---------------|
| Ubuntu | 20.04 LTS | 22.04 LTS | `lsb_release -a` |
| Docker | 20.10+ | 24.0+ | `docker --version` |
| Docker Compose | v2.0+ | v2.20+ | `docker compose version` |
| RAM | 8 GB | 16 GB | `free -h` |
| Disk Space | 20 GB | 50 GB | `df -h` |

### Optional (Recommended)

| Component | Purpose | Check Command |
|-----------|---------|---------------|
| NVIDIA GPU | Fast LLM inference | `nvidia-smi` |
| NVIDIA Container Toolkit | GPU in Docker | `docker info \| grep nvidia` |
| Synology NAS | Document storage for RAG | `ping synology-nas.local` |

---

## Installation Modes

### Interactive (Default)

```bash
./deploy/install-tars-home.sh
```

Prompts for confirmation at each step. Best for first-time setup.

### Non-Interactive

```bash
./deploy/install-tars-home.sh --yes
```

Accepts all defaults. Best for automation or scripted deployments.

### CPU-Only Mode

```bash
./deploy/install-tars-home.sh --cpu-only
```

Skips GPU verification. LLM runs on CPU (slower but works everywhere).

### Skip NAS

```bash
./deploy/install-tars-home.sh --skip-nas
```

Skips NAS mount setup. RAG works without documents initially.

### With Frontend

```bash
./deploy/install-tars-home.sh --with-frontend
```

Includes the React web UI (available at http://localhost:3000).

### Combined Flags

```bash
./deploy/install-tars-home.sh --yes --cpu-only --skip-nas
```

---

## Post-Installation

### Verify Installation

```bash
# Check service status
./deploy/start-tars-home.sh status

# Quick health check
curl http://localhost:8000/health

# Full readiness check
curl http://localhost:8000/ready
```

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| API | http://localhost:8000 | REST API |
| Swagger UI | http://localhost:8000/docs | Interactive API docs |
| Health | http://localhost:8000/health | Service health |
| Ready | http://localhost:8000/ready | Readiness status |
| Ollama | http://localhost:11434 | LLM engine |
| ChromaDB | http://localhost:8001 | Vector store |

### First Query

```bash
# Get an auth token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" | jq -r '.access_token')

# Send a test query
curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, are you working?"}'
```

---

## Troubleshooting

### Installation Failed at Prerequisites

**Docker not installed:**
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in
```

**Docker daemon not running:**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Installation Failed at Service Start

**Port already in use:**
```bash
# Find what's using the port
sudo ss -tlnp | grep :8000

# Stop the conflicting service or change port in tars-home.env
```

**Out of memory:**
```bash
# Check memory usage
free -h

# Increase swap (temporary fix)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Model Download Failed

```bash
# Retry model pull manually
docker exec tars-home-ollama ollama pull mistral:7b-instruct

# Or try a smaller model
docker exec tars-home-ollama ollama pull phi:2.7b
```

### Services Keep Restarting

```bash
# Check logs
docker logs tars-home-backend --tail 50
docker logs tars-home-ollama --tail 50

# Common causes:
# - Missing environment variables (check tars-home.env)
# - GPU access denied (add user to docker group)
# - Insufficient memory
```

---

## Updating T.A.R.S.

```bash
# Stop current deployment
./deploy/start-tars-home.sh stop

# Pull latest code
git pull origin main

# Re-run installer (preserves your data)
./deploy/install-tars-home.sh --yes
```

---

## Uninstalling

### Keep Data (Recommended)

```bash
# Stop services only
./deploy/start-tars-home.sh stop
```

### Remove Everything

```bash
# Stop and remove containers + volumes (DATA LOSS!)
docker compose -f deploy/docker-compose.home.yml down -v

# Remove images
docker compose -f deploy/docker-compose.home.yml down --rmi all

# Remove env file (contains secrets)
rm deploy/tars-home.env
```

---

## Daily Operations

After installation, use the start script for daily operations:

```bash
# Start T.A.R.S.
./deploy/start-tars-home.sh start

# Check status
./deploy/start-tars-home.sh status

# View logs
./deploy/start-tars-home.sh logs backend

# Stop T.A.R.S.
./deploy/start-tars-home.sh stop

# Restart
./deploy/start-tars-home.sh restart
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [GO_NO_GO_HOME.md](GO_NO_GO_HOME.md) | Daily operation checklist |
| [CONFIG_DOCTOR.md](CONFIG_DOCTOR.md) | Configuration troubleshooting |
| [SUPPORT_BUNDLE.md](SUPPORT_BUNDLE.md) | Generating support bundles |
| [DEPLOYMENT_VALIDATION.md](DEPLOYMENT_VALIDATION.md) | Validation script details |

---

**Last Updated:** December 27, 2025
**Version:** v1.0.11 (GA)
**Phase:** 23 - Local Machine Rollout

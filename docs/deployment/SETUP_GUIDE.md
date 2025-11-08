# T.A.R.S. Setup Guide - Phase 1

**Version:** v0.1.0-alpha
**Phase:** 1 - Infrastructure Foundation
**Date:** November 7, 2025

## Overview

This guide covers the complete setup process for T.A.R.S. (Temporal Augmented Retrieval System) Phase 1, establishing the infrastructure foundation with GPU-accelerated Ollama, FastAPI backend, and ChromaDB vector database.

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 250 GB SSD
- GPU: NVIDIA GTX 1660 (6GB VRAM) or equivalent
- Network: Gigabit Ethernet

**Recommended (Dell XPS 8950 specs):**
- CPU: Intel Core i7-12700
- RAM: 32 GB DDR5
- Storage: 512 GB NVMe SSD
- GPU: NVIDIA RTX 3060 Ti (8GB VRAM) or better
- Network: Gigabit Ethernet (wired)

### Software Requirements

- **OS:** Windows 11 or Ubuntu 22.04+
- **Docker Desktop:** 4.25+ with GPU support
- **NVIDIA Driver:** 535+ (for GPU inference)
- **Docker Compose:** 2.0+
- **Git:** 2.30+ (for cloning repository)

## Installation Steps

### Step 1: Install Prerequisites

#### On Windows 11:

1. Install NVIDIA Drivers from [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
2. Install Docker Desktop with WSL2 backend
3. Enable GPU support in Docker Desktop settings
4. Install Git from [git-scm.com](https://git-scm.com/download/win)

### Step 2: Verify GPU Access

```bash
# Test NVIDIA driver
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Step 3: Clone Repository

```bash
git clone https://github.com/yourusername/VDS_TARS.git
cd VDS_TARS
```

### Step 4: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
notepad .env  # Windows
```

### Step 5: Run Build Harness

```bash
# Run build validation
bash scripts/setup/build_harness.sh
```

### Step 6: Start the Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### Step 7: Verify Installation

```bash
# Backend health check
curl http://localhost:8000/health

# Verify GPU in Ollama
docker exec tars-ollama nvidia-smi

# Check ChromaDB
curl http://localhost:8001/api/v1/heartbeat
```

### Step 8: Pull LLM Model

```bash
# Pull Mistral 7B model
docker exec tars-ollama ollama pull mistral:7b-instruct

# Test inference
docker exec tars-ollama ollama run mistral:7b-instruct "Hello!"
```

## Phase 1 Validation Checklist

- [ ] Docker and Docker Compose installed
- [ ] NVIDIA drivers installed
- [ ] GPU accessible in Docker containers
- [ ] All services started successfully
- [ ] Health endpoints responding
- [ ] Mistral 7B model operational
- [ ] API documentation accessible

## Troubleshooting

### GPU Not Detected
- Verify NVIDIA drivers with `nvidia-smi`
- Restart Docker Desktop
- Check NVIDIA Container Toolkit

### Services Won't Start
- Check logs with `docker-compose logs`
- Verify ports not in use
- Check .env configuration

## Performance Benchmarking

Target metrics:
- **Token Generation:** e 20 tokens/second
- **GPU Utilization:** 70-90% during inference
- **Build Time:** < 10 minutes

## Next Steps

Phase 1 complete! Coming in Phase 2:
- WebSocket streaming interface
- JWT authentication
- Real-time responses

---

**Last Updated:** November 7, 2025

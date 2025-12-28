# T.A.R.S. Home Network Deployment

**Version:** v1.0.10 (GA)
**Phase:** 21 - User/Production Testing
**Target:** Home network Chatbot/RAG with Synology NAS

## Quick Start

### Prerequisites

- Ubuntu 22.04 LTS
- Docker & Docker Compose
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit
- Synology NAS accessible on LAN

### 1. Set Up Environment

```bash
# Copy environment template
cp deploy/tars-home.env.template deploy/tars-home.env

# Edit with your secrets
nano deploy/tars-home.env
```

Required settings in `tars-home.env`:
```bash
# Generate these:
TARS_POSTGRES_PASSWORD=$(openssl rand -base64 32)
TARS_JWT_SECRET=$(openssl rand -hex 64)
```

### 2. Mount NAS

```bash
# Full setup (mount + add to fstab)
sudo ./deploy/mount-nas.sh setup

# Or just mount temporarily
sudo ./deploy/mount-nas.sh mount

# Check status
./deploy/mount-nas.sh status
```

### 3. Start T.A.R.S.

```bash
# Start all services
./deploy/start-tars-home.sh start

# Or with React frontend
./deploy/start-tars-home.sh start --with-frontend
```

### 4. Pull LLM Model

```bash
# Pull default model (mistral:7b-instruct)
docker exec tars-home-ollama ollama pull mistral:7b-instruct

# Or pull alternatives
docker exec tars-home-ollama ollama pull llama3:8b
docker exec tars-home-ollama ollama pull codellama:7b-instruct
```

### 5. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check readiness (all services)
curl http://localhost:8000/ready

# View API docs
open http://localhost:8000/docs
```

## Service Endpoints

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| Docs | http://localhost:8000/docs | Swagger UI |
| Health | http://localhost:8000/health | Health check |
| Ready | http://localhost:8000/ready | Readiness check |
| Metrics | http://localhost:8000/metrics/prometheus | Prometheus metrics |
| Ollama | http://localhost:11434 | LLM inference |
| ChromaDB | http://localhost:8001 | Vector database |
| Frontend | http://localhost:3000 | React UI (if enabled) |

## Commands Reference

```bash
# Start services
./deploy/start-tars-home.sh start

# Stop services
./deploy/start-tars-home.sh stop

# Restart services
./deploy/start-tars-home.sh restart

# View status
./deploy/start-tars-home.sh status

# View logs
./deploy/start-tars-home.sh logs backend
./deploy/start-tars-home.sh logs ollama

# Health check
./deploy/start-tars-home.sh health
```

## File Structure

```
deploy/
├── README.md                    # This file
├── tars-home.yml               # Application configuration
├── docker-compose.home.yml     # Docker services
├── tars-home.env.template      # Environment template
├── tars-home.env               # Your environment (DO NOT COMMIT)
├── mount-nas.sh                # NAS mount script
└── start-tars-home.sh          # Deployment script
```

## Environment Details

| Component | Specification |
|-----------|---------------|
| Host OS | Ubuntu 22.04 LTS |
| RAM | 64 GB |
| GPU | NVIDIA RTX (12-24 GB VRAM) |
| NAS | Synology (SMB/CIFS) |
| LLM | mistral:7b-instruct (default) |
| Network | LAN-only, static IP |

## Troubleshooting

### NAS Mount Issues

```bash
# Check mount status
./deploy/mount-nas.sh status

# Verify NAS connectivity
ping synology-nas.local

# Check SMB port
nc -zv synology-nas.local 445
```

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Reinstall NVIDIA Container Toolkit if needed
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Container Health Issues

```bash
# Check container status
docker compose -f deploy/docker-compose.home.yml ps

# View specific container logs
docker logs tars-home-backend
docker logs tars-home-ollama

# Restart unhealthy container
docker compose -f deploy/docker-compose.home.yml restart backend
```

## Security Notes

- **LAN-only:** No external/public access configured
- **HSTS disabled:** HTTP-only deployment (no TLS)
- **Security headers enabled:** XSS protection, frame options, CSP
- **Rate limiting:** 200 requests/minute (relaxed for home use)
- **JWT tokens:** 7-day expiration for home convenience

## Next Steps

1. Verify all services are healthy
2. Index documents from NAS
3. Test RAG queries
4. Test WebSocket chat
5. Test from mobile devices on LAN

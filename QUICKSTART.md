# T.A.R.S. Quick Start

**Version:** v0.1.0-alpha | **Phase:** 1

## 5-Minute Setup

### Prerequisites Check
```bash
docker --version          # Need 4.25+
docker-compose --version  # Need 2.0+
nvidia-smi               # Verify GPU
```

### Installation
```bash
# 1. Clone and navigate
git clone https://github.com/yourusername/VDS_TARS.git
cd VDS_TARS

# 2. Configure environment
cp .env.example .env
# Edit .env: Change JWT_SECRET_KEY and HOST_IP

# 3. Start services
docker-compose up -d

# 4. Verify health
curl http://localhost:8000/health

# 5. Pull model
docker exec tars-ollama ollama pull mistral:7b-instruct
```

## Common Commands

### Service Management
```bash
docker-compose up -d              # Start
docker-compose down               # Stop
docker-compose restart backend    # Restart one service
docker-compose logs -f           # View logs
docker-compose ps                # Check status
```

### Health Checks
```bash
curl http://localhost:8000/health    # Backend health
curl http://localhost:8000/ready     # Readiness check
curl http://localhost:8000/docs      # Open in browser
```

### GPU & Models
```bash
docker exec tars-ollama nvidia-smi           # GPU status
docker exec tars-ollama ollama list          # Installed models
docker exec tars-ollama ollama run mistral   # Test inference
```

### Troubleshooting
```bash
docker-compose logs backend      # Backend logs
docker stats                     # Resource usage
docker exec -it tars-backend bash  # Shell access
```

## Service URLs

- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Ollama:** http://localhost:11434
- **ChromaDB:** http://localhost:8001

## Next Steps

1. Review [SETUP_GUIDE.md](docs/deployment/SETUP_GUIDE.md) for detailed instructions
2. Read [PHASE1_IMPLEMENTATION_REPORT.md](PHASE1_IMPLEMENTATION_REPORT.md) for architecture details
3. Check [TASKS.md](TASKS.md) for Phase 1 completion status

## Phase Status

- **Phase 1:** ✅ Infrastructure Foundation (COMPLETE)
- **Phase 2:** 📅 WebSocket Gateway (NEXT)

---

For full documentation, see [README.md](README.md)

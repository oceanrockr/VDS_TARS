# T.A.R.S. Go / No-Go Operator Checklist

**Version:** v1.0.10 (GA)
**Phase:** 22 - Deployment Validation
**Question:** Is this safe and ready to run daily on my home machine?

---

## Quick Answer

Run all validation scripts and check the results:

```bash
./deploy/validate-deployment.sh && \
./deploy/validate-rag.sh && \
./deploy/validate-security.sh
```

**If all pass:** GREEN - Ready to run daily
**If warnings only:** YELLOW - Review warnings, likely OK
**If any fail:** RED - Fix issues before daily use

---

## Decision Criteria

### GREEN - Go

All of the following must be true:

| Check | Command | Expected |
|-------|---------|----------|
| All containers healthy | `docker compose -f deploy/docker-compose.home.yml ps` | All show `healthy` |
| GPU detected | `nvidia-smi` | Shows GPU with free VRAM |
| Health endpoint | `curl http://localhost:8000/health` | `{"status":"healthy"}` |
| Ready endpoint | `curl http://localhost:8000/ready` | `{"status":"ready"}` |
| Model available | `docker exec tars-home-ollama ollama list` | At least one model |
| Test inference | Quick query returns response | Response within 60s |
| No public exposure | `ss -tlnp \| grep 8000` | Only local/LAN binding |

### YELLOW - Caution

Acceptable for home use, but review:

| Condition | Implication | Action |
|-----------|-------------|--------|
| `/ready` shows `degraded` | Some service unhealthy | Check logs, restart if needed |
| NAS not mounted | No document ingestion | Run `mount-nas.sh` if needed |
| High memory usage | May slow over time | Monitor, restart weekly |
| Cold start slow (>60s) | First query delay | Normal, GPU warming up |
| Rate limit triggered | Too many requests | Wait 60s, normal operation |

### RED - No-Go

Stop and fix before daily use:

| Condition | Risk | Fix |
|-----------|------|-----|
| Container crash loops | Unstable system | Check logs, verify config |
| GPU not detected | Slow inference (CPU) | Fix NVIDIA drivers |
| Auth bypass (401 not enforced) | Security issue | Check JWT secret, rebuild |
| XSS in responses | Security issue | Verify sanitization module |
| Port exposed to WAN | Security risk | Check router, firewall |
| Secrets in logs | Credential leak | Rotate secrets, update config |
| Persistent 500 errors | Backend broken | Check backend logs |

---

## One-Page Checklist

Print and use for daily operation verification:

```
T.A.R.S. GO/NO-GO CHECKLIST v1.0.10
====================================
Date: _______________

PRE-FLIGHT
[ ] Host powered on, Ubuntu logged in
[ ] Network connected to LAN
[ ] GPU visible: nvidia-smi shows GPU
[ ] NAS accessible: ping synology-nas.local

SERVICES
[ ] Start: ./deploy/start-tars-home.sh start
[ ] Wait for "All services ready" message
[ ] Health: curl localhost:8000/health → healthy
[ ] Ready: curl localhost:8000/ready → ready

VERIFICATION
[ ] Swagger UI loads: http://localhost:8000/docs
[ ] Quick test query responds
[ ] No error spam in logs

DAILY OPERATION
[ ] Keep terminal open for logs
[ ] Access via: http://localhost:8000
[ ] For issues: docker logs tars-home-backend

SHUTDOWN
[ ] ./deploy/start-tars-home.sh stop
[ ] Or: docker compose -f deploy/docker-compose.home.yml down

STATUS: [ ] GO    [ ] NO-GO
Notes: _________________________________
```

---

## Common Failure Symptoms

### Symptom: Containers keep restarting

```bash
docker compose -f deploy/docker-compose.home.yml ps
# Shows: "Restarting" or exit codes

# Fix: Check logs for root cause
docker logs tars-home-backend --tail 50
docker logs tars-home-ollama --tail 50
```

**Common causes:**
- Missing environment variables
- Port conflict
- GPU access denied
- Out of memory

### Symptom: Slow inference (>2 minutes)

```bash
nvidia-smi
# Check GPU utilization and memory
```

**Common causes:**
- Model running on CPU (no GPU access)
- VRAM exhausted (too many models loaded)
- Cold start (first query after restart)

**Fix:**
```bash
# Restart Ollama to free VRAM
docker restart tars-home-ollama
```

### Symptom: RAG returns no results

```bash
curl http://localhost:8000/rag/stats
# Check total_chunks
```

**Common causes:**
- No documents indexed
- ChromaDB collection empty
- NAS not mounted during ingestion

**Fix:**
```bash
# Verify NAS mount
./deploy/mount-nas.sh status

# Re-index documents
curl -X POST http://localhost:8000/rag/index/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["/mnt/nas/doc1.txt", "/mnt/nas/doc2.pdf"]}'
```

### Symptom: 401 Unauthorized on all requests

```bash
# Try getting a new token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"
```

**Common causes:**
- JWT secret changed
- Token expired (after 7 days)
- Wrong credentials

### Symptom: 503 Service Unavailable

```bash
curl http://localhost:8000/ready
# Check which service is unhealthy
```

**Common causes:**
- Ollama not started
- ChromaDB not responding
- PostgreSQL connection failed

**Fix:**
```bash
# Restart the unhealthy service
docker restart tars-home-<service>
```

---

## Rollback Steps

If things go wrong, here's how to recover:

### Level 1: Restart Services

```bash
./deploy/start-tars-home.sh restart
```

### Level 2: Full Stop and Start

```bash
./deploy/start-tars-home.sh stop
sleep 10
./deploy/start-tars-home.sh start
```

### Level 3: Clean Container Restart

```bash
docker compose -f deploy/docker-compose.home.yml down
docker compose -f deploy/docker-compose.home.yml up -d
```

### Level 4: Reset Volumes (DATA LOSS)

**Warning:** This deletes all indexed documents and conversation history.

```bash
# Stop everything
docker compose -f deploy/docker-compose.home.yml down -v

# Remove volumes
docker volume rm tars-home_chroma_data tars-home_postgres_data

# Start fresh
./deploy/start-tars-home.sh start
```

### Level 5: Full Reinstall

```bash
# Stop and remove everything
docker compose -f deploy/docker-compose.home.yml down -v --rmi all

# Remove volumes
docker volume prune -f

# Pull fresh images and rebuild
./deploy/start-tars-home.sh start
```

---

## Monitoring Commands

### Quick Status

```bash
# One-liner status check
./deploy/start-tars-home.sh status
```

### Watch Logs

```bash
# Follow backend logs
./deploy/start-tars-home.sh logs backend

# Follow all logs
docker compose -f deploy/docker-compose.home.yml logs -f
```

### Resource Usage

```bash
# Container stats
docker stats --no-stream

# GPU usage
nvidia-smi

# Memory
free -h
```

---

## Weekly Maintenance

Recommended weekly tasks for stable operation:

```bash
# 1. Check for stuck processes
docker compose -f deploy/docker-compose.home.yml ps

# 2. Review logs for errors
docker logs tars-home-backend --since 7d 2>&1 | grep -i error | head -20

# 3. Check disk space
df -h /var/lib/docker

# 4. Restart to free resources (optional)
./deploy/start-tars-home.sh restart
```

---

## Final Go/No-Go Decision Tree

```
START
  │
  ├─ All containers healthy? ─── NO ──→ RED: Fix containers
  │       │
  │      YES
  │       │
  ├─ GPU detected? ─── NO ──→ YELLOW: Slow but usable
  │       │
  │      YES
  │       │
  ├─ /health returns 200? ─── NO ──→ RED: Backend broken
  │       │
  │      YES
  │       │
  ├─ /ready returns ready? ─── NO ──→ YELLOW: Check degraded services
  │       │
  │      YES
  │       │
  ├─ Test query works? ─── NO ──→ YELLOW: Check model, try restart
  │       │
  │      YES
  │       │
  ├─ No WAN exposure? ─── NO ──→ RED: Security risk, fix firewall
  │       │
  │      YES
  │       │
  └─────────────────────────────→ GREEN: Ready for daily use
```

---

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Infrastructure | Stable | Docker Compose, GPU, NAS |
| Security | Acceptable | LAN-only, auth enforced |
| Performance | Good | 7B model on RTX GPU |
| Reliability | Good | Health checks, auto-restart |
| Maintenance | Low | Weekly restart recommended |

**Verdict:** T.A.R.S. v1.0.10 is ready for daily home network operation.

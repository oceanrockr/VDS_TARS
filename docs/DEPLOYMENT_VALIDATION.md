# T.A.R.S. Deployment Validation Checklist

**Version:** v1.0.10 (GA)
**Phase:** 22 - Deployment Validation
**Target:** Ubuntu 22.04 LTS, Home LAN

---

## 1. Container Health Validation

### Check All Containers Running

```bash
docker compose -f deploy/docker-compose.home.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"
```

**Expected:** All 5 containers (ollama, chromadb, redis, postgres, backend) show `Up` and `healthy`

### Individual Container Status

```bash
# Ollama
docker inspect tars-home-ollama --format '{{.State.Health.Status}}'
# Expected: healthy

# ChromaDB
docker inspect tars-home-chromadb --format '{{.State.Health.Status}}'
# Expected: healthy

# Redis
docker inspect tars-home-redis --format '{{.State.Health.Status}}'
# Expected: healthy

# PostgreSQL
docker inspect tars-home-postgres --format '{{.State.Health.Status}}'
# Expected: healthy

# Backend
docker inspect tars-home-backend --format '{{.State.Health.Status}}'
# Expected: healthy
```

### Quick Health Summary Script

```bash
for c in tars-home-ollama tars-home-chromadb tars-home-redis tars-home-postgres tars-home-backend; do
  status=$(docker inspect $c --format '{{.State.Health.Status}}' 2>/dev/null || echo "not_found")
  printf "%-25s %s\n" "$c" "$status"
done
```

---

## 2. GPU Detection & Usage

### Verify Host GPU Available

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

**Expected:** GPU name, total VRAM, current usage displayed

### Verify Docker GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Expected:** Same GPU info from within container

### Verify Ollama GPU Usage

```bash
docker exec tars-home-ollama nvidia-smi
```

**Expected:** nvidia-smi output with Ollama process using GPU memory

### Check GPU Memory Under Load

```bash
# Start a query then check GPU
curl -s http://localhost:11434/api/generate -d '{"model":"mistral:7b-instruct","prompt":"Hello","stream":false}' &
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

**Expected:** Memory usage increases during inference (~4-6GB for mistral:7b)

---

## 3. NAS Mount Validation

### Verify Mount Point

```bash
mountpoint -q /mnt/llm_docs && echo "MOUNTED" || echo "NOT MOUNTED"
```

**Expected:** `MOUNTED`

### Verify Read Access

```bash
ls -la /mnt/llm_docs | head -5
```

**Expected:** Directory listing with files/folders

### Verify Write Access (if applicable)

```bash
touch /mnt/llm_docs/.write_test && rm /mnt/llm_docs/.write_test && echo "WRITABLE" || echo "READ-ONLY"
```

**Note:** Read-only is acceptable for document ingestion

### Count Available Documents

```bash
find /mnt/llm_docs -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" \) | wc -l
```

### Verify NAS Accessible from Container

```bash
docker exec tars-home-backend ls -la /mnt/nas | head -5
```

**Expected:** Same files visible inside container

---

## 4. ChromaDB Persistence Validation

### Verify ChromaDB Responding

```bash
curl -s http://localhost:8001/api/v1/heartbeat
```

**Expected:** `{"nanosecond heartbeat":<timestamp>}`

### List Collections

```bash
curl -s http://localhost:8001/api/v1/collections
```

**Expected:** JSON array (may be empty initially)

### Verify Persistence Volume

```bash
docker volume inspect tars-home_chroma_data --format '{{.Mountpoint}}'
ls -la $(docker volume inspect tars-home_chroma_data --format '{{.Mountpoint}}')
```

**Expected:** Volume exists with chroma data files

### Restart Persistence Test

```bash
# Create test collection
curl -s -X POST http://localhost:8001/api/v1/collections -H "Content-Type: application/json" \
  -d '{"name":"_validation_test"}'

# Restart ChromaDB
docker restart tars-home-chromadb
sleep 10

# Verify collection persisted
curl -s http://localhost:8001/api/v1/collections | grep "_validation_test"

# Cleanup
curl -s -X DELETE http://localhost:8001/api/v1/collections/_validation_test
```

**Expected:** Collection survives restart

---

## 5. Backend Health & Readiness

### Basic Health Check

```bash
curl -s http://localhost:8000/health | jq .
```

**Expected:**
```json
{
  "status": "healthy",
  "service": "T.A.R.S. Backend",
  "version": "v0.3.0-alpha"
}
```

### Full Readiness Check

```bash
curl -s http://localhost:8000/ready | jq .
```

**Expected:**
```json
{
  "status": "ready",
  "checks": {
    "ollama": "healthy",
    "chromadb": "healthy",
    "embedding_model": "healthy",
    "conversation_service": "healthy",
    "nas_watcher": "enabled",
    "redis_cache": "healthy",
    "postgres": "connected"
  }
}
```

### Degraded State Detection

If any check shows `unhealthy` or `unknown`, the overall status should be `degraded`.

---

## 6. Service Dependency Failure Behavior

### Test: Ollama Down → Backend Degraded

```bash
# Stop Ollama
docker stop tars-home-ollama

# Check readiness (should show degraded)
curl -s http://localhost:8000/ready | jq '.status, .checks.ollama'

# Restart Ollama
docker start tars-home-ollama
```

**Expected:** Status becomes `degraded`, ollama shows `unhealthy`

### Test: ChromaDB Down → Backend Degraded

```bash
docker stop tars-home-chromadb
curl -s http://localhost:8000/ready | jq '.status, .checks.chromadb'
docker start tars-home-chromadb
```

### Test: Redis Down → Backend Still Works (Degraded)

```bash
docker stop tars-home-redis
curl -s http://localhost:8000/ready | jq '.status, .checks.redis_cache'
docker start tars-home-redis
```

**Expected:** Backend continues operating (cache is optional)

---

## 7. Endpoint Connectivity Matrix

| Endpoint | Command | Expected |
|----------|---------|----------|
| API Root | `curl -s http://localhost:8000/` | JSON with service info |
| Health | `curl -s http://localhost:8000/health` | `{"status":"healthy"}` |
| Ready | `curl -s http://localhost:8000/ready` | `{"status":"ready"}` |
| Docs | `curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs` | `200` |
| Metrics | `curl -s http://localhost:8000/metrics/prometheus` | Prometheus format |
| Ollama | `curl -s http://localhost:11434/api/tags` | Model list |
| ChromaDB | `curl -s http://localhost:8001/api/v1/heartbeat` | Heartbeat |
| Redis | `docker exec tars-home-redis redis-cli ping` | `PONG` |
| Postgres | `docker exec tars-home-postgres pg_isready -U tars` | `accepting connections` |

### Run Full Connectivity Check

```bash
echo "=== Endpoint Connectivity ==="
printf "%-20s %s\n" "API Root:" "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/)"
printf "%-20s %s\n" "Health:" "$(curl -s http://localhost:8000/health | jq -r '.status')"
printf "%-20s %s\n" "Ready:" "$(curl -s http://localhost:8000/ready | jq -r '.status')"
printf "%-20s %s\n" "Docs:" "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/docs)"
printf "%-20s %s\n" "Prometheus:" "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/metrics/prometheus)"
printf "%-20s %s\n" "Ollama:" "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:11434/api/tags)"
printf "%-20s %s\n" "ChromaDB:" "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/api/v1/heartbeat)"
printf "%-20s %s\n" "Redis:" "$(docker exec tars-home-redis redis-cli ping 2>/dev/null || echo 'FAIL')"
printf "%-20s %s\n" "Postgres:" "$(docker exec tars-home-postgres pg_isready -U tars -q && echo 'OK' || echo 'FAIL')"
```

---

## 8. Model Availability

### List Available Models

```bash
docker exec tars-home-ollama ollama list
```

**Expected:** At minimum `mistral:7b-instruct`

### Verify Default Model Responds

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "mistral:7b-instruct",
  "prompt": "Say hello in one word.",
  "stream": false
}' | jq -r '.response'
```

**Expected:** A greeting response

---

## 9. Quick Validation Script

Save as `deploy/validate-deployment.sh`:

```bash
#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "=== T.A.R.S. Deployment Validation ==="
echo ""

# Container health
echo "--- Container Health ---"
for c in tars-home-ollama tars-home-chromadb tars-home-redis tars-home-postgres tars-home-backend; do
  status=$(docker inspect $c --format '{{.State.Health.Status}}' 2>/dev/null || echo "not_found")
  if [ "$status" = "healthy" ]; then
    pass "$c"
  else
    fail "$c ($status)"
  fi
done

echo ""
echo "--- GPU ---"
if nvidia-smi &>/dev/null; then
  pass "nvidia-smi available"
else
  warn "nvidia-smi not available"
fi

echo ""
echo "--- NAS Mount ---"
if mountpoint -q /mnt/llm_docs 2>/dev/null; then
  pass "NAS mounted at /mnt/llm_docs"
else
  warn "NAS not mounted"
fi

echo ""
echo "--- API Endpoints ---"
health=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null)
if [ "$health" = "healthy" ]; then
  pass "/health: $health"
else
  fail "/health: $health"
fi

ready=$(curl -s http://localhost:8000/ready | jq -r '.status' 2>/dev/null)
if [ "$ready" = "ready" ]; then
  pass "/ready: $ready"
else
  warn "/ready: $ready (degraded services)"
fi

echo ""
echo "--- Model ---"
model_count=$(docker exec tars-home-ollama ollama list 2>/dev/null | wc -l)
if [ "$model_count" -gt 1 ]; then
  pass "Models available: $((model_count - 1))"
else
  fail "No models found"
fi

echo ""
echo "=== Validation Complete ==="
```

Make executable:
```bash
chmod +x deploy/validate-deployment.sh
```

Run:
```bash
./deploy/validate-deployment.sh
```

---

## Validation Criteria Summary

| Check | Status | Required for Ready |
|-------|--------|-------------------|
| All containers healthy | ✅ | Yes |
| GPU detected | ✅ | Yes (for performance) |
| NAS mounted | ⚠️ | Optional (can run without) |
| /health returns 200 | ✅ | Yes |
| /ready returns ready | ✅ | Yes |
| At least one model available | ✅ | Yes |
| ChromaDB persistence works | ✅ | Yes |
| Backend survives dep restart | ✅ | Yes |

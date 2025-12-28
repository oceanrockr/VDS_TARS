#!/bin/bash
# ==============================================================================
# T.A.R.S. Deployment Validation Script
# Version: v1.0.10 (GA) - Phase 22 Validation
# Target: Ubuntu 22.04 LTS, Home Network Deployment
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }
header() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# ==============================================================================
# Container Health
# ==============================================================================
validate_containers() {
    header "Container Health"

    for container in tars-home-ollama tars-home-chromadb tars-home-redis tars-home-postgres tars-home-backend; do
        status=$(docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "not_found")
        if [ "$status" = "healthy" ]; then
            pass "$container"
        elif [ "$status" = "not_found" ]; then
            fail "$container (container not found)"
        else
            fail "$container (status: $status)"
        fi
    done
}

# ==============================================================================
# GPU Validation
# ==============================================================================
validate_gpu() {
    header "GPU Detection"

    # Host GPU
    if command -v nvidia-smi &>/dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$gpu_name" ]; then
            pass "Host GPU: $gpu_name"
        else
            fail "nvidia-smi available but no GPU detected"
        fi
    else
        warn "nvidia-smi not available on host"
    fi

    # Container GPU access
    if docker exec tars-home-ollama nvidia-smi &>/dev/null; then
        vram=$(docker exec tars-home-ollama nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        pass "Ollama container GPU access: $vram"
    else
        warn "Ollama container cannot access GPU"
    fi
}

# ==============================================================================
# NAS Mount
# ==============================================================================
validate_nas() {
    header "NAS Mount"

    NAS_PATH="${NAS_MOUNT_POINT:-/mnt/llm_docs}"

    if mountpoint -q "$NAS_PATH" 2>/dev/null; then
        pass "NAS mounted at $NAS_PATH"

        # Count documents
        doc_count=$(find "$NAS_PATH" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l)
        echo "     Documents found: $doc_count"

        # Check container access
        if docker exec tars-home-backend ls /mnt/nas &>/dev/null; then
            pass "Container can access NAS mount"
        else
            fail "Container cannot access NAS mount"
        fi
    else
        warn "NAS not mounted at $NAS_PATH (optional for basic operation)"
    fi
}

# ==============================================================================
# API Endpoints
# ==============================================================================
validate_api() {
    header "API Endpoints"

    # Health
    health_resp=$(curl -s --max-time 5 http://localhost:8000/health 2>/dev/null)
    health_status=$(echo "$health_resp" | jq -r '.status' 2>/dev/null)
    if [ "$health_status" = "healthy" ]; then
        pass "/health: $health_status"
    else
        fail "/health: ${health_status:-no response}"
    fi

    # Ready
    ready_resp=$(curl -s --max-time 10 http://localhost:8000/ready 2>/dev/null)
    ready_status=$(echo "$ready_resp" | jq -r '.status' 2>/dev/null)
    if [ "$ready_status" = "ready" ]; then
        pass "/ready: $ready_status"
    elif [ "$ready_status" = "degraded" ]; then
        warn "/ready: $ready_status"
        echo "     Degraded checks:"
        echo "$ready_resp" | jq -r '.checks | to_entries[] | select(.value != "healthy" and .value != "enabled" and .value != "connected") | "       - \(.key): \(.value)"' 2>/dev/null
    else
        fail "/ready: ${ready_status:-no response}"
    fi

    # Docs
    docs_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8000/docs 2>/dev/null)
    if [ "$docs_code" = "200" ]; then
        pass "/docs: HTTP $docs_code"
    else
        fail "/docs: HTTP ${docs_code:-timeout}"
    fi

    # Prometheus metrics
    metrics_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8000/metrics/prometheus 2>/dev/null)
    if [ "$metrics_code" = "200" ]; then
        pass "/metrics/prometheus: HTTP $metrics_code"
    else
        warn "/metrics/prometheus: HTTP ${metrics_code:-timeout}"
    fi
}

# ==============================================================================
# Service Backends
# ==============================================================================
validate_backends() {
    header "Backend Services"

    # Ollama
    ollama_resp=$(curl -s --max-time 5 http://localhost:11434/api/tags 2>/dev/null)
    if [ -n "$ollama_resp" ]; then
        pass "Ollama API responding"
    else
        fail "Ollama API not responding"
    fi

    # ChromaDB
    chroma_resp=$(curl -s --max-time 5 http://localhost:8001/api/v1/heartbeat 2>/dev/null)
    if echo "$chroma_resp" | grep -q "nanosecond"; then
        pass "ChromaDB heartbeat OK"
    else
        fail "ChromaDB not responding"
    fi

    # Redis
    redis_ping=$(docker exec tars-home-redis redis-cli ping 2>/dev/null)
    if [ "$redis_ping" = "PONG" ]; then
        pass "Redis PING/PONG OK"
    else
        fail "Redis not responding"
    fi

    # PostgreSQL
    if docker exec tars-home-postgres pg_isready -U tars -q 2>/dev/null; then
        pass "PostgreSQL accepting connections"
    else
        fail "PostgreSQL not ready"
    fi
}

# ==============================================================================
# Model Availability
# ==============================================================================
validate_models() {
    header "LLM Models"

    models=$(docker exec tars-home-ollama ollama list 2>/dev/null | tail -n +2)
    model_count=$(echo "$models" | grep -c "." || echo "0")

    if [ "$model_count" -gt 0 ]; then
        pass "Models available: $model_count"
        echo "$models" | while read -r line; do
            echo "     - $line"
        done
    else
        fail "No models found (run: docker exec tars-home-ollama ollama pull mistral:7b-instruct)"
    fi

    # Quick inference test
    echo ""
    echo "Testing inference..."
    inference_start=$(date +%s%3N)
    inference_resp=$(curl -s --max-time 60 http://localhost:11434/api/generate \
        -d '{"model":"mistral:7b-instruct","prompt":"Say hello.","stream":false}' 2>/dev/null)
    inference_end=$(date +%s%3N)

    if echo "$inference_resp" | jq -e '.response' &>/dev/null; then
        inference_time=$((inference_end - inference_start))
        pass "Inference successful (${inference_time}ms)"
    else
        fail "Inference failed or timed out"
    fi
}

# ==============================================================================
# ChromaDB Persistence
# ==============================================================================
validate_chroma_persistence() {
    header "ChromaDB Persistence"

    # Check volume exists
    volume_path=$(docker volume inspect tars-home_chroma_data --format '{{.Mountpoint}}' 2>/dev/null)
    if [ -n "$volume_path" ]; then
        pass "ChromaDB volume exists at $volume_path"
    else
        warn "ChromaDB volume not found (may be using different naming)"
    fi

    # List collections
    collections=$(curl -s --max-time 5 http://localhost:8001/api/v1/collections 2>/dev/null)
    coll_count=$(echo "$collections" | jq 'length' 2>/dev/null || echo "0")
    pass "ChromaDB collections: $coll_count"
}

# ==============================================================================
# Summary
# ==============================================================================
print_summary() {
    header "Validation Summary"

    total=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))

    echo -e "${GREEN}Passed:${NC}  $PASS_COUNT"
    echo -e "${YELLOW}Warnings:${NC} $WARN_COUNT"
    echo -e "${RED}Failed:${NC}  $FAIL_COUNT"
    echo ""

    if [ "$FAIL_COUNT" -eq 0 ]; then
        echo -e "${GREEN}✓ Deployment validation PASSED${NC}"
        if [ "$WARN_COUNT" -gt 0 ]; then
            echo -e "${YELLOW}  (with $WARN_COUNT non-critical warnings)${NC}"
        fi
        exit 0
    else
        echo -e "${RED}✗ Deployment validation FAILED${NC}"
        echo "  Review failed checks above and fix before proceeding."
        exit 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${CYAN}=================================================================${NC}"
    echo -e "${CYAN} T.A.R.S. Deployment Validation - v1.0.10 (Phase 22)${NC}"
    echo -e "${CYAN}=================================================================${NC}"

    validate_containers
    validate_gpu
    validate_nas
    validate_api
    validate_backends
    validate_models
    validate_chroma_persistence
    print_summary
}

main "$@"

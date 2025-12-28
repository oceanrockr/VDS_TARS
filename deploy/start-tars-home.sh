#!/bin/bash
# ==============================================================================
# T.A.R.S. Home Network Deployment Script
# Version: v1.0.10 (GA) - Phase 21 User Testing
# Target: Ubuntu 22.04 LTS, Synology NAS, NVIDIA RTX GPU
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.home.yml"
ENV_FILE="$SCRIPT_DIR/tars-home.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# ==============================================================================
# Pre-flight Checks
# ==============================================================================
preflight_checks() {
    log_header "Pre-flight Checks"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_success "Docker installed: $(docker --version)"

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_success "Docker Compose installed: $(docker compose version --short)"

    # Check NVIDIA Docker
    if docker info 2>/dev/null | grep -q "nvidia"; then
        log_success "NVIDIA Docker runtime available"
    else
        log_warn "NVIDIA Docker runtime not detected - GPU acceleration may not work"
    fi

    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        log_warn "nvidia-smi not found - GPU support unclear"
    fi

    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
        log_info "Create it from template:"
        log_info "  cp $SCRIPT_DIR/tars-home.env.template $ENV_FILE"
        exit 1
    fi
    log_success "Environment file found"

    # Validate required environment variables
    source "$ENV_FILE"
    if [[ "$TARS_POSTGRES_PASSWORD" == "CHANGE_ME"* ]] || [[ -z "$TARS_POSTGRES_PASSWORD" ]]; then
        log_error "TARS_POSTGRES_PASSWORD not set in $ENV_FILE"
        exit 1
    fi
    if [[ "$TARS_JWT_SECRET" == "CHANGE_ME"* ]] || [[ -z "$TARS_JWT_SECRET" ]]; then
        log_error "TARS_JWT_SECRET not set in $ENV_FILE"
        exit 1
    fi
    log_success "Required environment variables set"

    # Check NAS mount
    NAS_MOUNT="${NAS_MOUNT_POINT:-/mnt/llm_docs}"
    if mountpoint -q "$NAS_MOUNT" 2>/dev/null; then
        log_success "NAS mounted at $NAS_MOUNT"
        DOC_COUNT=$(find "$NAS_MOUNT" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l)
        log_info "  Documents found: $DOC_COUNT"
    else
        log_warn "NAS not mounted at $NAS_MOUNT"
        log_info "  Run: sudo ./deploy/mount-nas.sh setup"
        read -p "Continue without NAS? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    log_success "All pre-flight checks passed"
}

# ==============================================================================
# Start Services
# ==============================================================================
start_services() {
    local profile=""
    if [[ "$1" == "--with-frontend" ]]; then
        profile="--profile with-frontend"
    fi

    log_header "Starting T.A.R.S. Services"

    cd "$PROJECT_ROOT"

    # Pull latest images
    log_info "Pulling Docker images..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull

    # Build custom images
    log_info "Building T.A.R.S. images..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build

    # Start services
    log_info "Starting containers..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profile up -d

    log_success "Containers started"
}

# ==============================================================================
# Wait for Services
# ==============================================================================
wait_for_services() {
    log_header "Waiting for Services"

    local max_wait=120
    local waited=0

    # Wait for Ollama
    log_info "Waiting for Ollama..."
    while ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -ge $max_wait ]]; then
            log_error "Ollama did not start in time"
            return 1
        fi
    done
    log_success "Ollama ready"

    # Wait for ChromaDB
    log_info "Waiting for ChromaDB..."
    waited=0
    while ! curl -sf http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -ge $max_wait ]]; then
            log_error "ChromaDB did not start in time"
            return 1
        fi
    done
    log_success "ChromaDB ready"

    # Wait for Backend
    log_info "Waiting for Backend API..."
    waited=0
    while ! curl -sf http://localhost:8000/health > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -ge $max_wait ]]; then
            log_error "Backend did not start in time"
            return 1
        fi
    done
    log_success "Backend API ready"

    log_success "All services ready"
}

# ==============================================================================
# Pull LLM Model
# ==============================================================================
pull_model() {
    log_header "Checking LLM Model"

    source "$ENV_FILE"
    local model="${OLLAMA_MODEL:-mistral:7b-instruct}"

    # Check if model exists
    if docker exec tars-home-ollama ollama list 2>/dev/null | grep -q "$model"; then
        log_success "Model already available: $model"
    else
        log_info "Pulling model: $model (this may take a while)..."
        docker exec tars-home-ollama ollama pull "$model"
        log_success "Model pulled: $model"
    fi
}

# ==============================================================================
# Health Check
# ==============================================================================
health_check() {
    log_header "Service Health Check"

    # Basic health
    log_info "Checking /health..."
    curl -s http://localhost:8000/health | jq .

    echo ""

    # Readiness with all services
    log_info "Checking /ready..."
    curl -s http://localhost:8000/ready | jq .

    echo ""

    # Container status
    log_info "Container status:"
    docker compose -f "$COMPOSE_FILE" ps
}

# ==============================================================================
# Show Status
# ==============================================================================
show_status() {
    log_header "T.A.R.S. Home Deployment Status"

    echo ""
    echo "Service Endpoints:"
    echo "  - API:        http://localhost:8000"
    echo "  - Docs:       http://localhost:8000/docs"
    echo "  - Health:     http://localhost:8000/health"
    echo "  - Ready:      http://localhost:8000/ready"
    echo "  - Metrics:    http://localhost:8000/metrics/prometheus"
    echo "  - Ollama:     http://localhost:11434"
    echo "  - ChromaDB:   http://localhost:8001"
    echo ""

    health_check
}

# ==============================================================================
# Stop Services
# ==============================================================================
stop_services() {
    log_header "Stopping T.A.R.S. Services"

    cd "$PROJECT_ROOT"
    docker compose -f "$COMPOSE_FILE" down

    log_success "All services stopped"
}

# ==============================================================================
# View Logs
# ==============================================================================
view_logs() {
    local service="${1:-backend}"

    cd "$PROJECT_ROOT"
    docker compose -f "$COMPOSE_FILE" logs -f "$service"
}

# ==============================================================================
# Main
# ==============================================================================
usage() {
    echo "T.A.R.S. Home Network Deployment Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start             Start all services (API only)"
    echo "  start --with-frontend  Start all services including React frontend"
    echo "  stop              Stop all services"
    echo "  restart           Restart all services"
    echo "  status            Show status and health"
    echo "  logs [service]    View logs (default: backend)"
    echo "  pull-model        Download the LLM model"
    echo "  health            Run health checks"
    echo "  preflight         Run pre-flight checks only"
    echo ""
    echo "Examples:"
    echo "  $0 start                  # Start T.A.R.S. (API only)"
    echo "  $0 start --with-frontend  # Start with React UI"
    echo "  $0 logs ollama            # View Ollama logs"
    echo "  $0 status                 # Check all services"
}

case "${1:-}" in
    start)
        preflight_checks
        start_services "$2"
        wait_for_services
        pull_model
        show_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services "$2"
        wait_for_services
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        view_logs "${2:-backend}"
        ;;
    pull-model)
        pull_model
        ;;
    health)
        health_check
        ;;
    preflight)
        preflight_checks
        ;;
    *)
        usage
        exit 1
        ;;
esac

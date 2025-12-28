#!/bin/bash
# ==============================================================================
# T.A.R.S. Home Network One-Command Installer
# Version: v1.0.11 (GA) - Phase 23 Local Machine Rollout
# Target: Ubuntu 22.04 LTS, Synology NAS, NVIDIA RTX GPU
# ==============================================================================
#
# This script provides a complete, repeatable installation experience for
# deploying T.A.R.S. on a fresh Ubuntu home network machine.
#
# Features:
#   - Prerequisite verification (Docker, Compose, NVIDIA toolkit)
#   - Environment file setup from template
#   - Optional NAS mount configuration
#   - Service startup and health waiting
#   - Automatic Phase 22 validation
#   - Final GO/NO-GO summary
#
# Usage:
#   ./install-tars-home.sh                    # Interactive install
#   ./install-tars-home.sh --yes              # Non-interactive (accept defaults)
#   ./install-tars-home.sh --skip-nas         # Skip NAS mount step
#   ./install-tars-home.sh --cpu-only         # Skip GPU verification
#   ./install-tars-home.sh --with-frontend    # Include React frontend
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.home.yml"
ENV_FILE="$SCRIPT_DIR/tars-home.env"
ENV_TEMPLATE="$SCRIPT_DIR/tars-home.env.template"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Counters for summary
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# CLI Flags
NON_INTERACTIVE=false
SKIP_NAS=false
CPU_ONLY=false
WITH_FRONTEND=false

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; ((PASS_COUNT++)); }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; ((FAIL_COUNT++)); }
log_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${MAGENTA}>>> $1${NC}"
}

pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }

# ==============================================================================
# Parse CLI Arguments
# ==============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --yes|--non-interactive|-y)
                NON_INTERACTIVE=true
                shift
                ;;
            --skip-nas)
                SKIP_NAS=true
                shift
                ;;
            --cpu-only)
                CPU_ONLY=true
                shift
                ;;
            --with-frontend)
                WITH_FRONTEND=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "T.A.R.S. Home Network Installer"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --yes, --non-interactive, -y  Accept all defaults, no prompts"
    echo "  --skip-nas                    Skip NAS mount verification/setup"
    echo "  --cpu-only                    Skip GPU verification (CPU inference)"
    echo "  --with-frontend               Include React frontend service"
    echo "  --help, -h                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                            Interactive installation"
    echo "  $0 --yes                      Non-interactive with defaults"
    echo "  $0 --yes --skip-nas           Non-interactive, no NAS"
    echo "  $0 --cpu-only --with-frontend CPU mode with React UI"
}

# ==============================================================================
# Prompt Helper
# ==============================================================================
prompt_user() {
    local prompt="$1"
    local default="$2"

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        echo "$default"
        return
    fi

    read -p "$prompt [$default]: " response
    echo "${response:-$default}"
}

confirm() {
    local prompt="$1"

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        return 0
    fi

    read -p "$prompt (y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# ==============================================================================
# Step 1: Verify Prerequisites
# ==============================================================================
verify_prerequisites() {
    log_header "Step 1: Verifying Prerequisites"

    local prereq_ok=true

    # Check Docker
    log_step "Checking Docker..."
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version)
        pass "Docker installed: $docker_version"
    else
        fail "Docker is not installed"
        log_info "Install Docker: https://docs.docker.com/engine/install/ubuntu/"
        prereq_ok=false
    fi

    # Check Docker Compose
    log_step "Checking Docker Compose..."
    if docker compose version &> /dev/null; then
        compose_version=$(docker compose version --short 2>/dev/null || echo "unknown")
        pass "Docker Compose installed: $compose_version"
    else
        fail "Docker Compose is not installed"
        log_info "Install Docker Compose: https://docs.docker.com/compose/install/"
        prereq_ok=false
    fi

    # Check Docker daemon
    log_step "Checking Docker daemon..."
    if docker info &> /dev/null; then
        pass "Docker daemon is running"
    else
        fail "Docker daemon is not running"
        log_info "Start Docker: sudo systemctl start docker"
        prereq_ok=false
    fi

    # Check NVIDIA GPU (unless --cpu-only)
    if [[ "$CPU_ONLY" != "true" ]]; then
        log_step "Checking NVIDIA GPU..."

        # Check nvidia-smi
        if command -v nvidia-smi &> /dev/null; then
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            if [ -n "$gpu_name" ]; then
                pass "NVIDIA GPU detected: $gpu_name"

                vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
                log_info "  VRAM: $vram"
            else
                warn "nvidia-smi available but no GPU detected"
            fi
        else
            warn "nvidia-smi not found - GPU acceleration unavailable"
            log_info "Install NVIDIA drivers: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi

        # Check NVIDIA Container Toolkit
        log_step "Checking NVIDIA Container Toolkit..."
        if docker info 2>/dev/null | grep -q "nvidia"; then
            pass "NVIDIA Docker runtime available"
        else
            warn "NVIDIA Docker runtime not detected"
            log_info "Models will run on CPU (slower). Install nvidia-container-toolkit for GPU support."
        fi
    else
        log_info "Skipping GPU checks (--cpu-only mode)"
    fi

    # Check available disk space
    log_step "Checking disk space..."
    available_gb=$(df -BG "$PROJECT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ -n "$available_gb" ] && [ "$available_gb" -ge 20 ]; then
        pass "Disk space available: ${available_gb}GB (minimum 20GB)"
    elif [ -n "$available_gb" ]; then
        warn "Low disk space: ${available_gb}GB (recommended 20GB+)"
    else
        warn "Could not determine disk space"
    fi

    # Check RAM
    log_step "Checking system memory..."
    total_ram_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
    if [ -n "$total_ram_kb" ]; then
        total_ram_gb=$((total_ram_kb / 1024 / 1024))
        if [ "$total_ram_gb" -ge 16 ]; then
            pass "System RAM: ${total_ram_gb}GB (recommended 16GB+)"
        elif [ "$total_ram_gb" -ge 8 ]; then
            warn "System RAM: ${total_ram_gb}GB (16GB+ recommended for optimal performance)"
        else
            warn "System RAM: ${total_ram_gb}GB (may experience performance issues)"
        fi
    fi

    # Check required ports
    log_step "Checking port availability..."
    local ports_ok=true
    for port in 8000 8001 11434 6379 5432; do
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            warn "Port $port is already in use"
            ports_ok=false
        fi
    done
    if [ "$ports_ok" = true ]; then
        pass "Required ports available (8000, 8001, 11434, 6379, 5432)"
    fi

    if [ "$prereq_ok" = false ]; then
        log_error "Prerequisites not met. Please install missing components and retry."
        exit 1
    fi

    log_success "All critical prerequisites verified"
}

# ==============================================================================
# Step 2: Setup Environment File
# ==============================================================================
setup_environment() {
    log_header "Step 2: Setting Up Environment Configuration"

    if [[ -f "$ENV_FILE" ]]; then
        log_info "Environment file already exists: $ENV_FILE"

        # Validate existing env file
        source "$ENV_FILE" 2>/dev/null || true

        local env_valid=true
        if [[ "$TARS_POSTGRES_PASSWORD" == "CHANGE_ME"* ]] || [[ -z "$TARS_POSTGRES_PASSWORD" ]]; then
            warn "TARS_POSTGRES_PASSWORD not properly set"
            env_valid=false
        fi
        if [[ "$TARS_JWT_SECRET" == "CHANGE_ME"* ]] || [[ -z "$TARS_JWT_SECRET" ]]; then
            warn "TARS_JWT_SECRET not properly set"
            env_valid=false
        fi

        if [ "$env_valid" = true ]; then
            pass "Environment file validated"
            return
        else
            if confirm "Environment file has placeholder values. Regenerate secrets?"; then
                generate_env_secrets
            else
                log_error "Cannot proceed with placeholder secrets"
                exit 1
            fi
        fi
    else
        log_info "Creating environment file from template..."

        if [[ ! -f "$ENV_TEMPLATE" ]]; then
            log_error "Template not found: $ENV_TEMPLATE"
            exit 1
        fi

        cp "$ENV_TEMPLATE" "$ENV_FILE"
        generate_env_secrets
    fi
}

generate_env_secrets() {
    log_step "Generating secure secrets..."

    # Generate PostgreSQL password
    local pg_password=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)
    sed -i "s/TARS_POSTGRES_PASSWORD=.*/TARS_POSTGRES_PASSWORD=$pg_password/" "$ENV_FILE"

    # Generate JWT secret
    local jwt_secret=$(openssl rand -hex 64)
    sed -i "s/TARS_JWT_SECRET=.*/TARS_JWT_SECRET=$jwt_secret/" "$ENV_FILE"

    pass "Secure secrets generated and saved"
    log_info "Environment file: $ENV_FILE"
    log_warn "IMPORTANT: Back up this file securely. Secrets are not recoverable."
}

# ==============================================================================
# Step 3: NAS Mount (Optional)
# ==============================================================================
setup_nas_mount() {
    log_header "Step 3: NAS Mount Configuration"

    if [[ "$SKIP_NAS" == "true" ]]; then
        log_info "Skipping NAS mount (--skip-nas flag)"
        return
    fi

    source "$ENV_FILE" 2>/dev/null || true
    NAS_MOUNT="${NAS_MOUNT_POINT:-/mnt/llm_docs}"

    if mountpoint -q "$NAS_MOUNT" 2>/dev/null; then
        pass "NAS already mounted at $NAS_MOUNT"

        # Count documents
        doc_count=$(find "$NAS_MOUNT" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l)
        log_info "  Documents found: $doc_count"
        return
    fi

    log_info "NAS not currently mounted at $NAS_MOUNT"

    if [[ -f "$SCRIPT_DIR/mount-nas.sh" ]]; then
        if confirm "Would you like to set up NAS mount now?"; then
            log_info "Running NAS mount script..."
            if sudo "$SCRIPT_DIR/mount-nas.sh" setup 2>/dev/null; then
                pass "NAS mount configured"
            else
                warn "NAS mount setup failed - continuing without NAS"
                log_info "You can run 'sudo ./deploy/mount-nas.sh setup' later"
            fi
        else
            log_info "Skipping NAS mount. RAG will work without document ingestion."
        fi
    else
        warn "NAS mount script not found: $SCRIPT_DIR/mount-nas.sh"
        log_info "NAS mount is optional - RAG can function without it"
    fi
}

# ==============================================================================
# Step 4: Start Services
# ==============================================================================
start_services() {
    log_header "Step 4: Starting T.A.R.S. Services"

    cd "$PROJECT_ROOT"

    local profile_flag=""
    if [[ "$WITH_FRONTEND" == "true" ]]; then
        profile_flag="--profile with-frontend"
        log_info "Including React frontend service"
    fi

    # Pull images
    log_step "Pulling Docker images..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
    pass "Docker images pulled"

    # Build custom images
    log_step "Building T.A.R.S. images..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build
    pass "T.A.R.S. images built"

    # Start services
    log_step "Starting containers..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profile_flag up -d
    pass "Containers started"
}

# ==============================================================================
# Step 5: Wait for Services
# ==============================================================================
wait_for_services() {
    log_header "Step 5: Waiting for Services to Initialize"

    local max_wait=180
    local interval=5

    # Wait for Ollama
    log_step "Waiting for Ollama (LLM engine)..."
    local waited=0
    while ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
        sleep $interval
        waited=$((waited + interval))
        echo -n "."
        if [[ $waited -ge $max_wait ]]; then
            echo ""
            fail "Ollama did not start within ${max_wait}s"
            log_info "Check logs: docker logs tars-home-ollama"
            return 1
        fi
    done
    echo ""
    pass "Ollama ready"

    # Wait for ChromaDB
    log_step "Waiting for ChromaDB (vector store)..."
    waited=0
    while ! curl -sf http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; do
        sleep $interval
        waited=$((waited + interval))
        echo -n "."
        if [[ $waited -ge $max_wait ]]; then
            echo ""
            fail "ChromaDB did not start within ${max_wait}s"
            log_info "Check logs: docker logs tars-home-chromadb"
            return 1
        fi
    done
    echo ""
    pass "ChromaDB ready"

    # Wait for Backend API
    log_step "Waiting for Backend API..."
    waited=0
    while ! curl -sf http://localhost:8000/health > /dev/null 2>&1; do
        sleep $interval
        waited=$((waited + interval))
        echo -n "."
        if [[ $waited -ge $max_wait ]]; then
            echo ""
            fail "Backend API did not start within ${max_wait}s"
            log_info "Check logs: docker logs tars-home-backend"
            return 1
        fi
    done
    echo ""
    pass "Backend API ready"

    log_success "All core services are running"
}

# ==============================================================================
# Step 6: Pull LLM Model
# ==============================================================================
pull_model() {
    log_header "Step 6: Ensuring LLM Model is Available"

    source "$ENV_FILE" 2>/dev/null || true
    local model="${OLLAMA_MODEL:-mistral:7b-instruct}"

    # Check if model exists
    log_step "Checking for model: $model"
    if docker exec tars-home-ollama ollama list 2>/dev/null | grep -q "$model"; then
        pass "Model already available: $model"
    else
        log_info "Model not found. Pulling $model (this may take 5-15 minutes)..."

        if docker exec tars-home-ollama ollama pull "$model"; then
            pass "Model pulled successfully: $model"
        else
            fail "Failed to pull model: $model"
            log_info "Try manually: docker exec tars-home-ollama ollama pull $model"
            return 1
        fi
    fi
}

# ==============================================================================
# Step 7: Run Validation Scripts
# ==============================================================================
run_validation() {
    log_header "Step 7: Running Phase 22 Validation Suite"

    cd "$PROJECT_ROOT"

    local validation_failed=false

    # Run deployment validation
    log_step "Running deployment validation..."
    if [[ -x "$SCRIPT_DIR/validate-deployment.sh" ]]; then
        if "$SCRIPT_DIR/validate-deployment.sh"; then
            pass "Deployment validation passed"
        else
            fail "Deployment validation failed"
            validation_failed=true
        fi
    else
        warn "Deployment validation script not found or not executable"
    fi

    # Run RAG validation
    log_step "Running RAG validation..."
    if [[ -x "$SCRIPT_DIR/validate-rag.sh" ]]; then
        if "$SCRIPT_DIR/validate-rag.sh"; then
            pass "RAG validation passed"
        else
            warn "RAG validation had issues (may be normal without documents)"
        fi
    else
        warn "RAG validation script not found or not executable"
    fi

    # Run security validation
    log_step "Running security validation..."
    if [[ -x "$SCRIPT_DIR/validate-security.sh" ]]; then
        if "$SCRIPT_DIR/validate-security.sh"; then
            pass "Security validation passed"
        else
            fail "Security validation failed"
            validation_failed=true
        fi
    else
        warn "Security validation script not found or not executable"
    fi

    if [ "$validation_failed" = true ]; then
        return 1
    fi
}

# ==============================================================================
# Step 8: Print Final Summary
# ==============================================================================
print_summary() {
    log_header "Installation Summary"

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    T.A.R.S. INSTALLATION COMPLETE              ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Statistics
    echo -e "  ${GREEN}Passed:${NC}   $PASS_COUNT"
    echo -e "  ${YELLOW}Warnings:${NC} $WARN_COUNT"
    echo -e "  ${RED}Failed:${NC}   $FAIL_COUNT"
    echo ""

    # GO/NO-GO Decision
    if [ "$FAIL_COUNT" -eq 0 ]; then
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                                                               ║${NC}"
        echo -e "${GREEN}║   █████   ███████       ██████   ██████                       ║${NC}"
        echo -e "${GREEN}║  ██   ██  ██           ██       ██    ██                      ║${NC}"
        echo -e "${GREEN}║  ███████  █████   ███  ██   ███ ██    ██                      ║${NC}"
        echo -e "${GREEN}║  ██   ██  ██           ██    ██ ██    ██                      ║${NC}"
        echo -e "${GREEN}║  ██   ██  ███████       ██████   ██████                       ║${NC}"
        echo -e "${GREEN}║                                                               ║${NC}"
        echo -e "${GREEN}║           T.A.R.S. IS READY FOR DAILY USE                    ║${NC}"
        echo -e "${GREEN}║                                                               ║${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"

        if [ "$WARN_COUNT" -gt 0 ]; then
            echo ""
            echo -e "${YELLOW}Note: $WARN_COUNT non-critical warnings above. Review if needed.${NC}"
        fi
    else
        echo -e "${RED}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║                                                               ║${NC}"
        echo -e "${RED}║   ███    ██  ██████        ██████   ██████                    ║${NC}"
        echo -e "${RED}║   ████   ██ ██    ██      ██       ██    ██                   ║${NC}"
        echo -e "${RED}║   ██ ██  ██ ██    ██ ███  ██   ███ ██    ██                   ║${NC}"
        echo -e "${RED}║   ██  ██ ██ ██    ██      ██    ██ ██    ██                   ║${NC}"
        echo -e "${RED}║   ██   ████  ██████        ██████   ██████                    ║${NC}"
        echo -e "${RED}║                                                               ║${NC}"
        echo -e "${RED}║           FIX ISSUES BEFORE DAILY USE                         ║${NC}"
        echo -e "${RED}║                                                               ║${NC}"
        echo -e "${RED}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "Review failed checks above and run: ${CYAN}$0${NC}"
    fi

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Quick reference
    echo "Quick Reference:"
    echo "  - API:        http://localhost:8000"
    echo "  - Swagger UI: http://localhost:8000/docs"
    echo "  - Health:     http://localhost:8000/health"
    echo "  - Ready:      http://localhost:8000/ready"
    echo ""
    echo "Common Commands:"
    echo "  - Status:     ./deploy/start-tars-home.sh status"
    echo "  - Logs:       ./deploy/start-tars-home.sh logs backend"
    echo "  - Stop:       ./deploy/start-tars-home.sh stop"
    echo "  - Restart:    ./deploy/start-tars-home.sh restart"
    echo ""
    echo "Documentation:"
    echo "  - Install:    docs/INSTALL_HOME.md"
    echo "  - Go/No-Go:   docs/GO_NO_GO_HOME.md"
    echo "  - Config:     docs/CONFIG_DOCTOR.md"
    echo ""

    # Exit code
    if [ "$FAIL_COUNT" -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}     T.A.R.S. Home Network Installer - v1.0.11 (Phase 23)      ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "This installer will set up T.A.R.S. on your home network machine."
    echo ""

    parse_args "$@"

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        log_info "Running in non-interactive mode (--yes)"
    fi

    verify_prerequisites
    setup_environment
    setup_nas_mount
    start_services
    wait_for_services
    pull_model
    run_validation
    print_summary
}

main "$@"

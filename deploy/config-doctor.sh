#!/bin/bash
# ==============================================================================
# T.A.R.S. Configuration Doctor
# Version: v1.0.11 (GA) - Phase 23 Configuration UX Hardening
# Target: Ubuntu 22.04 LTS, Home Network Deployment
# ==============================================================================
#
# This script validates T.A.R.S. configuration and provides actionable fixes.
# Designed for operators to quickly diagnose and resolve configuration issues.
#
# Features:
#   - Required environment variable validation
#   - NAS path existence check
#   - Port availability verification
#   - GPU detection and configuration
#   - Model presence verification with auto-pull suggestion
#   - Actionable fix commands for each issue
#
# Usage:
#   ./config-doctor.sh              # Run all checks
#   ./config-doctor.sh --fix        # Attempt automatic fixes (experimental)
#   ./config-doctor.sh --quiet      # Only show errors
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$SCRIPT_DIR/tars-home.env"
ENV_TEMPLATE="$SCRIPT_DIR/tars-home.env.template"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.home.yml"
CONFIG_FILE="$SCRIPT_DIR/tars-home.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# CLI Flags
QUIET_MODE=false
FIX_MODE=false

# Logging functions
log_info() { [[ "$QUIET_MODE" != "true" ]] && echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; ((PASS_COUNT++)); }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARN_COUNT++)); }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
log_fix() { echo -e "       ${CYAN}Fix:${NC} $1"; }
log_header() {
    [[ "$QUIET_MODE" != "true" ]] && echo -e "\n${CYAN}=== $1 ===${NC}"
}

# ==============================================================================
# Parse CLI Arguments
# ==============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --fix)
                FIX_MODE=true
                shift
                ;;
            --quiet|-q)
                QUIET_MODE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "T.A.R.S. Configuration Doctor"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --fix       Attempt automatic fixes for common issues"
    echo "  --quiet, -q Only show errors and warnings"
    echo "  --help, -h  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Run all configuration checks"
    echo "  $0 --fix        # Run checks and attempt fixes"
    echo "  $0 --quiet      # Only show problems"
}

# ==============================================================================
# Check: Environment File Exists
# ==============================================================================
check_env_file() {
    log_header "Environment File"

    if [[ -f "$ENV_FILE" ]]; then
        log_success "Environment file exists: $ENV_FILE"
        return 0
    else
        log_error "Environment file not found: $ENV_FILE"
        log_fix "cp $ENV_TEMPLATE $ENV_FILE"

        if [[ "$FIX_MODE" == "true" ]] && [[ -f "$ENV_TEMPLATE" ]]; then
            cp "$ENV_TEMPLATE" "$ENV_FILE"
            echo -e "       ${GREEN}Fixed:${NC} Created environment file from template"
        fi
        return 1
    fi
}

# ==============================================================================
# Check: Required Environment Variables
# ==============================================================================
check_required_vars() {
    log_header "Required Environment Variables"

    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Cannot check variables - environment file missing"
        return 1
    fi

    source "$ENV_FILE" 2>/dev/null

    local all_ok=true

    # PostgreSQL Password
    if [[ -z "$TARS_POSTGRES_PASSWORD" ]]; then
        log_error "TARS_POSTGRES_PASSWORD is not set"
        log_fix "Add to $ENV_FILE: TARS_POSTGRES_PASSWORD=\$(openssl rand -base64 32)"
        all_ok=false
    elif [[ "$TARS_POSTGRES_PASSWORD" == "CHANGE_ME"* ]]; then
        log_error "TARS_POSTGRES_PASSWORD still has placeholder value"
        log_fix "Generate new password: openssl rand -base64 32"
        all_ok=false
    elif [[ ${#TARS_POSTGRES_PASSWORD} -lt 16 ]]; then
        log_warn "TARS_POSTGRES_PASSWORD is short (recommended: 32+ characters)"
        all_ok=false
    else
        log_success "TARS_POSTGRES_PASSWORD is set (${#TARS_POSTGRES_PASSWORD} chars)"
    fi

    # JWT Secret
    if [[ -z "$TARS_JWT_SECRET" ]]; then
        log_error "TARS_JWT_SECRET is not set"
        log_fix "Add to $ENV_FILE: TARS_JWT_SECRET=\$(openssl rand -hex 64)"
        all_ok=false
    elif [[ "$TARS_JWT_SECRET" == "CHANGE_ME"* ]]; then
        log_error "TARS_JWT_SECRET still has placeholder value"
        log_fix "Generate new secret: openssl rand -hex 64"
        all_ok=false
    elif [[ ${#TARS_JWT_SECRET} -lt 32 ]]; then
        log_warn "TARS_JWT_SECRET is short (recommended: 64+ hex chars)"
        all_ok=false
    else
        log_success "TARS_JWT_SECRET is set (${#TARS_JWT_SECRET} chars)"
    fi

    # PostgreSQL Database
    if [[ -z "$TARS_POSTGRES_DB" ]]; then
        log_warn "TARS_POSTGRES_DB not set (default: tars_home)"
    else
        log_success "TARS_POSTGRES_DB: $TARS_POSTGRES_DB"
    fi

    # PostgreSQL User
    if [[ -z "$TARS_POSTGRES_USER" ]]; then
        log_warn "TARS_POSTGRES_USER not set (default: tars)"
    else
        log_success "TARS_POSTGRES_USER: $TARS_POSTGRES_USER"
    fi

    # Ollama Model
    if [[ -z "$OLLAMA_MODEL" ]]; then
        log_warn "OLLAMA_MODEL not set (default: mistral:7b-instruct)"
    else
        log_success "OLLAMA_MODEL: $OLLAMA_MODEL"
    fi

    return 0
}

# ==============================================================================
# Check: NAS Paths
# ==============================================================================
check_nas_paths() {
    log_header "NAS Configuration"

    source "$ENV_FILE" 2>/dev/null || true

    local nas_mount="${NAS_MOUNT_POINT:-/mnt/llm_docs}"

    # Check if NAS is configured
    if [[ -z "$NAS_HOSTNAME" ]] && [[ -z "$NAS_IP" ]]; then
        log_info "NAS not configured (optional for basic operation)"
        return 0
    fi

    # Check mount point exists
    if [[ ! -d "$nas_mount" ]]; then
        log_warn "NAS mount point does not exist: $nas_mount"
        log_fix "sudo mkdir -p $nas_mount"

        if [[ "$FIX_MODE" == "true" ]]; then
            sudo mkdir -p "$nas_mount" 2>/dev/null && \
                echo -e "       ${GREEN}Fixed:${NC} Created mount point" || \
                echo -e "       ${RED}Failed:${NC} Could not create mount point (try with sudo)"
        fi
    else
        log_success "NAS mount point exists: $nas_mount"
    fi

    # Check if mounted
    if mountpoint -q "$nas_mount" 2>/dev/null; then
        log_success "NAS is mounted at $nas_mount"

        # Count documents
        doc_count=$(find "$nas_mount" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l)
        log_info "Documents found: $doc_count"
    else
        log_warn "NAS is not currently mounted at $nas_mount"
        log_fix "./deploy/mount-nas.sh setup"
    fi

    # Check NAS host reachability
    local nas_host="${NAS_HOSTNAME:-$NAS_IP}"
    if [[ -n "$nas_host" ]]; then
        if ping -c 1 -W 2 "$nas_host" &>/dev/null; then
            log_success "NAS host reachable: $nas_host"
        else
            log_warn "Cannot reach NAS host: $nas_host"
            log_fix "Check NAS is powered on and on the same network"
        fi
    fi
}

# ==============================================================================
# Check: Port Availability
# ==============================================================================
check_ports() {
    log_header "Port Availability"

    local ports=(
        "8000:Backend API"
        "8001:ChromaDB"
        "11434:Ollama"
        "6379:Redis"
        "5432:PostgreSQL"
        "3000:Frontend (optional)"
    )

    local all_ok=true

    for port_info in "${ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info#*:}"

        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            # Check if it's our container using the port
            local container=$(docker ps --format '{{.Names}}' 2>/dev/null | xargs -I {} docker port {} 2>/dev/null | grep "$port" | head -1)

            if [[ -n "$container" ]] || docker ps 2>/dev/null | grep -q "tars-home"; then
                log_success "Port $port ($service): In use by T.A.R.S."
            else
                log_warn "Port $port ($service): In use by another process"
                log_fix "ss -tlnp | grep :$port  # Find process"
                log_fix "Change port in $ENV_FILE if needed"
                all_ok=false
            fi
        else
            log_success "Port $port ($service): Available"
        fi
    done
}

# ==============================================================================
# Check: GPU Availability
# ==============================================================================
check_gpu() {
    log_header "GPU Configuration"

    # Check nvidia-smi on host
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)

        if [[ -n "$gpu_name" ]]; then
            log_success "Host GPU: $gpu_name ($gpu_memory)"
            log_info "Current utilization: $gpu_util"
        else
            log_warn "nvidia-smi available but no GPU detected"
        fi
    else
        log_info "No NVIDIA GPU detected on host (CPU inference mode)"
    fi

    # Check NVIDIA Container Toolkit
    if docker info 2>/dev/null | grep -q "nvidia"; then
        log_success "NVIDIA Container Toolkit: Available"
    else
        log_info "NVIDIA Container Toolkit: Not detected"
        log_fix "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    fi

    # Check GPU access from container
    if docker ps 2>/dev/null | grep -q "tars-home-ollama"; then
        if docker exec tars-home-ollama nvidia-smi &>/dev/null; then
            local container_gpu=$(docker exec tars-home-ollama nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            log_success "Container GPU access: $container_gpu"
        else
            log_warn "Ollama container cannot access GPU"
            log_fix "Ensure NVIDIA Container Toolkit is installed and restart Docker"
        fi
    fi
}

# ==============================================================================
# Check: Model Availability
# ==============================================================================
check_models() {
    log_header "LLM Models"

    source "$ENV_FILE" 2>/dev/null || true
    local expected_model="${OLLAMA_MODEL:-mistral:7b-instruct}"

    # Check if Ollama container is running
    if ! docker ps 2>/dev/null | grep -q "tars-home-ollama"; then
        log_warn "Ollama container not running - cannot check models"
        log_fix "./deploy/start-tars-home.sh start"
        return
    fi

    # List available models
    local models=$(docker exec tars-home-ollama ollama list 2>/dev/null | tail -n +2)

    if [[ -z "$models" ]]; then
        log_warn "No models installed"
        log_fix "docker exec tars-home-ollama ollama pull $expected_model"

        if [[ "$FIX_MODE" == "true" ]]; then
            echo -e "       ${BLUE}Pulling model (this may take several minutes)...${NC}"
            docker exec tars-home-ollama ollama pull "$expected_model" && \
                echo -e "       ${GREEN}Fixed:${NC} Model pulled successfully" || \
                echo -e "       ${RED}Failed:${NC} Could not pull model"
        fi
    else
        log_success "Models installed:"
        echo "$models" | while read -r line; do
            echo "       - $line"
        done

        # Check if expected model is present
        if echo "$models" | grep -q "$expected_model"; then
            log_success "Configured model available: $expected_model"
        else
            log_warn "Configured model not found: $expected_model"
            log_fix "docker exec tars-home-ollama ollama pull $expected_model"
        fi
    fi
}

# ==============================================================================
# Check: Docker Compose Configuration
# ==============================================================================
check_compose_config() {
    log_header "Docker Compose Configuration"

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        return 1
    fi

    log_success "Docker Compose file exists"

    # Validate compose file syntax
    if docker compose -f "$COMPOSE_FILE" config --quiet 2>/dev/null; then
        log_success "Docker Compose syntax is valid"
    else
        log_error "Docker Compose file has syntax errors"
        log_fix "docker compose -f $COMPOSE_FILE config"
    fi

    # Check if env file is referenced
    if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" config --quiet 2>/dev/null; then
        log_success "Environment file integrates correctly"
    else
        log_warn "Issues integrating environment file"
        log_fix "Check variable references in compose file"
    fi
}

# ==============================================================================
# Check: Service Health (if running)
# ==============================================================================
check_service_health() {
    log_header "Service Health (if running)"

    # Check if any services are running
    if ! docker ps 2>/dev/null | grep -q "tars-home"; then
        log_info "No T.A.R.S. services currently running"
        log_fix "./deploy/start-tars-home.sh start"
        return
    fi

    # Check each container
    local containers=(
        "tars-home-ollama:Ollama"
        "tars-home-chromadb:ChromaDB"
        "tars-home-redis:Redis"
        "tars-home-postgres:PostgreSQL"
        "tars-home-backend:Backend API"
    )

    for container_info in "${containers[@]}"; do
        local container="${container_info%%:*}"
        local service="${container_info#*:}"

        local status=$(docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "not_found")

        case "$status" in
            "healthy")
                log_success "$service: Healthy"
                ;;
            "unhealthy")
                log_error "$service: Unhealthy"
                log_fix "docker logs $container --tail 20"
                ;;
            "starting")
                log_info "$service: Starting..."
                ;;
            "not_found")
                log_warn "$service: Container not found"
                ;;
            *)
                log_warn "$service: Status unknown ($status)"
                ;;
        esac
    done

    # Check API endpoints
    if curl -sf http://localhost:8000/health &>/dev/null; then
        log_success "API /health endpoint responding"
    fi

    if curl -sf http://localhost:8000/ready &>/dev/null; then
        local ready_status=$(curl -s http://localhost:8000/ready | jq -r '.status' 2>/dev/null)
        if [[ "$ready_status" == "ready" ]]; then
            log_success "API /ready: All services ready"
        elif [[ "$ready_status" == "degraded" ]]; then
            log_warn "API /ready: Degraded (some services unhealthy)"
        fi
    fi
}

# ==============================================================================
# Check: Security Configuration
# ==============================================================================
check_security() {
    log_header "Security Configuration"

    source "$ENV_FILE" 2>/dev/null || true

    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root user"
        log_fix "Run as regular user with Docker group membership"
    else
        log_success "Running as non-root user: $(whoami)"
    fi

    # Check Docker group membership
    if groups | grep -q docker; then
        log_success "User is in docker group"
    else
        log_warn "User not in docker group"
        log_fix "sudo usermod -aG docker \$USER && newgrp docker"
    fi

    # Check if API is exposed externally
    if ss -tlnp 2>/dev/null | grep ":8000 " | grep -q "0.0.0.0\|::"; then
        log_warn "API port 8000 is bound to all interfaces"
        log_fix "For LAN-only: Bind to specific IP in docker-compose"
    else
        log_info "API port binding: Check manually with 'ss -tlnp | grep 8000'"
    fi

    # Check env file permissions
    if [[ -f "$ENV_FILE" ]]; then
        local env_perms=$(stat -c '%a' "$ENV_FILE" 2>/dev/null || echo "unknown")
        if [[ "$env_perms" == "600" ]] || [[ "$env_perms" == "640" ]]; then
            log_success "Environment file permissions: $env_perms"
        else
            log_warn "Environment file permissions: $env_perms (recommended: 600)"
            log_fix "chmod 600 $ENV_FILE"

            if [[ "$FIX_MODE" == "true" ]]; then
                chmod 600 "$ENV_FILE" && \
                    echo -e "       ${GREEN}Fixed:${NC} Set permissions to 600" || \
                    echo -e "       ${RED}Failed:${NC} Could not change permissions"
            fi
        fi
    fi
}

# ==============================================================================
# Summary
# ==============================================================================
print_summary() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}               T.A.R.S. Configuration Doctor Summary            ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}Passed:${NC}   $PASS_COUNT"
    echo -e "  ${YELLOW}Warnings:${NC} $WARN_COUNT"
    echo -e "  ${RED}Failed:${NC}   $FAIL_COUNT"
    echo ""

    if [ "$FAIL_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
        echo -e "${GREEN}Configuration is healthy. T.A.R.S. should work correctly.${NC}"
    elif [ "$FAIL_COUNT" -eq 0 ]; then
        echo -e "${YELLOW}Configuration has warnings. Review items above.${NC}"
    else
        echo -e "${RED}Configuration has errors. Fix the failed items before running T.A.R.S.${NC}"
    fi

    echo ""
    echo "Next steps:"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "  1. Fix the failed items listed above"
        echo "  2. Re-run: $0"
    elif [ "$WARN_COUNT" -gt 0 ]; then
        echo "  1. Review warnings (optional but recommended)"
        echo "  2. Start T.A.R.S.: ./deploy/start-tars-home.sh start"
    else
        echo "  1. Start T.A.R.S.: ./deploy/start-tars-home.sh start"
        echo "  2. Check status: ./deploy/start-tars-home.sh status"
    fi
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
    echo -e "${CYAN}       T.A.R.S. Configuration Doctor - v1.0.11 (Phase 23)       ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

    parse_args "$@"

    if [[ "$FIX_MODE" == "true" ]]; then
        echo -e "${YELLOW}Running in FIX mode - will attempt automatic fixes${NC}"
    fi

    check_env_file
    check_required_vars
    check_nas_paths
    check_ports
    check_gpu
    check_compose_config
    check_models
    check_service_health
    check_security

    print_summary
}

main "$@"

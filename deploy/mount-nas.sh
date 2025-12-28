#!/bin/bash
# ==============================================================================
# T.A.R.S. NAS Mount Script (SMB/CIFS)
# Target: Synology NAS at synology-nas.local / 192.168.1.20
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
NAS_HOSTNAME="${NAS_HOSTNAME:-synology-nas.local}"
NAS_IP="${NAS_IP:-192.168.1.20}"
NAS_SHARE="${NAS_SHARE:-LLM_docs}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/llm_docs}"
NAS_USER="${NAS_USER:-}"
NAS_PASSWORD="${NAS_PASSWORD:-}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/etc/tars/nas-credentials}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# Pre-flight Checks
# ------------------------------------------------------------------------------
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check for cifs-utils
    if ! command -v mount.cifs &> /dev/null; then
        log_warn "cifs-utils not installed. Installing..."
        apt-get update && apt-get install -y cifs-utils
    fi

    # Check NAS connectivity
    log_info "Checking NAS connectivity..."
    if ping -c 1 -W 3 "$NAS_HOSTNAME" &> /dev/null; then
        log_success "NAS reachable at $NAS_HOSTNAME"
    elif ping -c 1 -W 3 "$NAS_IP" &> /dev/null; then
        log_warn "Hostname $NAS_HOSTNAME not resolving, using IP $NAS_IP"
        NAS_HOSTNAME="$NAS_IP"
    else
        log_error "Cannot reach NAS at $NAS_HOSTNAME or $NAS_IP"
        exit 1
    fi

    # Check SMB port
    if timeout 3 bash -c "echo > /dev/tcp/$NAS_HOSTNAME/445" 2>/dev/null; then
        log_success "SMB port 445 is open"
    else
        log_error "SMB port 445 is not accessible on $NAS_HOSTNAME"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# Create Credentials File
# ------------------------------------------------------------------------------
setup_credentials() {
    log_info "Setting up credentials..."

    # Create config directory
    mkdir -p /etc/tars
    chmod 700 /etc/tars

    if [[ -f "$CREDENTIALS_FILE" ]]; then
        log_info "Using existing credentials file: $CREDENTIALS_FILE"
        return
    fi

    if [[ -z "$NAS_USER" ]]; then
        read -p "Enter NAS username: " NAS_USER
    fi

    if [[ -z "$NAS_PASSWORD" ]]; then
        read -s -p "Enter NAS password: " NAS_PASSWORD
        echo
    fi

    # Create credentials file
    cat > "$CREDENTIALS_FILE" <<EOF
username=$NAS_USER
password=$NAS_PASSWORD
EOF

    chmod 600 "$CREDENTIALS_FILE"
    log_success "Credentials file created at $CREDENTIALS_FILE"
}

# ------------------------------------------------------------------------------
# Mount NAS Share
# ------------------------------------------------------------------------------
mount_nas() {
    log_info "Mounting NAS share..."

    # Create mount point if it doesn't exist
    if [[ ! -d "$MOUNT_POINT" ]]; then
        log_info "Creating mount point: $MOUNT_POINT"
        mkdir -p "$MOUNT_POINT"
    fi

    # Check if already mounted
    if mountpoint -q "$MOUNT_POINT"; then
        log_warn "$MOUNT_POINT is already mounted"
        return
    fi

    # Mount the share
    mount -t cifs \
        "//${NAS_HOSTNAME}/${NAS_SHARE}" \
        "$MOUNT_POINT" \
        -o credentials="$CREDENTIALS_FILE",uid=1000,gid=1000,file_mode=0644,dir_mode=0755,vers=3.0,ro

    if mountpoint -q "$MOUNT_POINT"; then
        log_success "NAS mounted successfully at $MOUNT_POINT"
        log_info "Contents:"
        ls -la "$MOUNT_POINT" | head -10
    else
        log_error "Failed to mount NAS"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# Unmount NAS Share
# ------------------------------------------------------------------------------
unmount_nas() {
    log_info "Unmounting NAS share..."

    if ! mountpoint -q "$MOUNT_POINT"; then
        log_warn "$MOUNT_POINT is not mounted"
        return
    fi

    umount "$MOUNT_POINT"
    log_success "NAS unmounted from $MOUNT_POINT"
}

# ------------------------------------------------------------------------------
# Add to /etc/fstab for Persistence
# ------------------------------------------------------------------------------
add_to_fstab() {
    log_info "Adding mount to /etc/fstab for persistence..."

    FSTAB_ENTRY="//${NAS_HOSTNAME}/${NAS_SHARE} ${MOUNT_POINT} cifs credentials=${CREDENTIALS_FILE},uid=1000,gid=1000,file_mode=0644,dir_mode=0755,vers=3.0,ro,_netdev,x-systemd.automount 0 0"

    if grep -q "${NAS_SHARE}" /etc/fstab; then
        log_warn "Entry already exists in /etc/fstab"
        return
    fi

    # Backup fstab
    cp /etc/fstab /etc/fstab.backup.$(date +%Y%m%d%H%M%S)

    # Add entry
    echo "" >> /etc/fstab
    echo "# T.A.R.S. NAS Mount (Synology)" >> /etc/fstab
    echo "$FSTAB_ENTRY" >> /etc/fstab

    log_success "Added to /etc/fstab"
    log_info "Mount will persist across reboots"
}

# ------------------------------------------------------------------------------
# Create systemd Mount Unit (Alternative to fstab)
# ------------------------------------------------------------------------------
create_systemd_unit() {
    log_info "Creating systemd mount unit..."

    UNIT_FILE="/etc/systemd/system/mnt-llm_docs.mount"

    cat > "$UNIT_FILE" <<EOF
[Unit]
Description=T.A.R.S. NAS Mount (Synology LLM Documents)
After=network-online.target
Wants=network-online.target

[Mount]
What=//${NAS_HOSTNAME}/${NAS_SHARE}
Where=${MOUNT_POINT}
Type=cifs
Options=credentials=${CREDENTIALS_FILE},uid=1000,gid=1000,file_mode=0644,dir_mode=0755,vers=3.0,ro

[Install]
WantedBy=multi-user.target
EOF

    # Create automount unit
    cat > "/etc/systemd/system/mnt-llm_docs.automount" <<EOF
[Unit]
Description=T.A.R.S. NAS Automount (Synology LLM Documents)
After=network-online.target
Wants=network-online.target

[Automount]
Where=${MOUNT_POINT}
TimeoutIdleSec=300

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable mnt-llm_docs.automount
    log_success "Systemd mount units created and enabled"
}

# ------------------------------------------------------------------------------
# Status Check
# ------------------------------------------------------------------------------
status() {
    echo "========================================"
    echo "T.A.R.S. NAS Mount Status"
    echo "========================================"
    echo "NAS Host:     $NAS_HOSTNAME"
    echo "NAS IP:       $NAS_IP"
    echo "Share:        $NAS_SHARE"
    echo "Mount Point:  $MOUNT_POINT"
    echo "----------------------------------------"

    if mountpoint -q "$MOUNT_POINT"; then
        echo -e "Status:       ${GREEN}MOUNTED${NC}"
        echo "----------------------------------------"
        echo "Disk Usage:"
        df -h "$MOUNT_POINT"
        echo "----------------------------------------"
        echo "Document Count:"
        find "$MOUNT_POINT" -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l
    else
        echo -e "Status:       ${RED}NOT MOUNTED${NC}"
    fi
    echo "========================================"
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
usage() {
    echo "Usage: $0 {mount|unmount|status|fstab|systemd|setup}"
    echo ""
    echo "Commands:"
    echo "  mount     Mount the NAS share"
    echo "  unmount   Unmount the NAS share"
    echo "  status    Show mount status"
    echo "  fstab     Add mount to /etc/fstab for persistence"
    echo "  systemd   Create systemd mount units for persistence"
    echo "  setup     Full setup (mount + fstab)"
    echo ""
    echo "Environment Variables:"
    echo "  NAS_HOSTNAME   NAS hostname (default: synology-nas.local)"
    echo "  NAS_IP         NAS IP address (default: 192.168.1.20)"
    echo "  NAS_SHARE      Share name (default: LLM_docs)"
    echo "  MOUNT_POINT    Local mount path (default: /mnt/llm_docs)"
    echo "  NAS_USER       NAS username"
    echo "  NAS_PASSWORD   NAS password"
}

case "${1:-}" in
    mount)
        check_root
        preflight_checks
        setup_credentials
        mount_nas
        ;;
    unmount)
        check_root
        unmount_nas
        ;;
    status)
        status
        ;;
    fstab)
        check_root
        setup_credentials
        add_to_fstab
        ;;
    systemd)
        check_root
        setup_credentials
        create_systemd_unit
        ;;
    setup)
        check_root
        preflight_checks
        setup_credentials
        mount_nas
        add_to_fstab
        ;;
    *)
        usage
        exit 1
        ;;
esac

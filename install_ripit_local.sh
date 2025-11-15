#!/bin/bash
# RiPIT Local Installation Script
# Installs RiPIT framework and ACE stub locally to a single project

set -e

# Configuration
PROJECT_ROOT="$(pwd)"
RIPIT_LOCAL="${PROJECT_ROOT}/.ripit"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  RiPIT Local Installation (Project-Only)        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Installation Location:${NC} $PROJECT_ROOT"
echo -e "${BLUE}RiPIT Directory:${NC} $RIPIT_LOCAL"
echo -e "${BLUE}Virtual Environment:${NC} $VENV_DIR"
echo ""

# Step 1: Check prerequisites
echo -e "${BLUE}[1/8]${NC} Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ git not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Step 2: Create virtual environment
echo -e "${BLUE}[2/8]${NC} Creating virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment exists, skipping creation${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Step 3: Activate virtual environment
echo -e "${BLUE}[3/8]${NC} Activating virtual environment..."

# Detect platform and activate accordingly
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Git Bash on Windows
    source "${VENV_DIR}/Scripts/activate"
else
    # Linux/macOS
    source "${VENV_DIR}/bin/activate"
fi

ACTIVE_PYTHON=$(which python3)
echo -e "${GREEN}✅ Active Python: $ACTIVE_PYTHON${NC}"

# Step 4: Clone/update RiPIT framework
echo -e "${BLUE}[4/8]${NC} Setting up RiPIT framework..."

if [ -d "$RIPIT_LOCAL" ]; then
    echo -e "${YELLOW}⚠️  RiPIT directory exists${NC}"
    read -p "Update existing RiPIT installation? [y/N]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$RIPIT_LOCAL"
        git pull origin main
        cd "$PROJECT_ROOT"
        echo -e "${GREEN}✅ RiPIT framework updated${NC}"
    else
        echo -e "${YELLOW}⚠️  Using existing RiPIT installation${NC}"
    fi
else
    git clone https://github.com/Veleron-Dev-Studios-LLC/VDS_RiPIT-Agent-Coding-Workflow.git "$RIPIT_LOCAL"
    echo -e "${GREEN}✅ RiPIT framework cloned${NC}"
fi

# Step 5: Install ACE stub to virtual environment
echo -e "${BLUE}[5/8]${NC} Installing ACE stub to virtual environment..."

pip3 install -e "${RIPIT_LOCAL}/ace_stub" --quiet
echo -e "${GREEN}✅ ACE stub installed${NC}"

# Step 6: Verify installation
echo -e "${BLUE}[6/8]${NC} Verifying installation..."

ACE_VERSION=$(python3 -c "import ace; print(ace.__version__)" 2>&1)
ACE_STATUS=$(python3 -c "import ace; print(ace.__status__)" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ ACE stub verified${NC}"
    echo -e "   Version: $ACE_VERSION"
    echo -e "   Status: $ACE_STATUS"
else
    echo -e "${RED}❌ ACE stub verification failed${NC}"
    exit 1
fi

# Step 7: Create Claude Code commands
echo -e "${BLUE}[7/8]${NC} Setting up Claude Code commands..."

CLAUDE_DIR="${PROJECT_ROOT}/.claude/commands"
mkdir -p "$CLAUDE_DIR"

# Copy RiPIT commands if they don't exist
if [ -d "${RIPIT_LOCAL}/.claude/commands" ]; then
    cp -n "${RIPIT_LOCAL}/.claude/commands/"*.md "$CLAUDE_DIR/" 2>/dev/null || true
    echo -e "${GREEN}✅ Claude Code commands installed${NC}"
else
    echo -e "${YELLOW}⚠️  No Claude Code commands found in RiPIT${NC}"
fi

# Step 8: Create activation helper
echo -e "${BLUE}[8/8]${NC} Creating activation helper..."

cat > "${PROJECT_ROOT}/activate_ripit.sh" << 'ACTIVATE_SCRIPT'
#!/bin/bash
# RiPIT Environment Activation Script
# Source this file to activate RiPIT for this project

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect platform
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Git Bash on Windows
    VENV_ACTIVATE="${PROJECT_ROOT}/.venv/Scripts/activate"
else
    # Linux/macOS
    VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
fi

# Activate virtual environment
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
else
    echo "❌ Virtual environment not found: $VENV_ACTIVATE"
    return 1
fi

# Set RiPIT environment variables
export RIPIT_HOME="${PROJECT_ROOT}/.ripit"
export PYTHONPATH="${RIPIT_HOME}/ace_stub:${PYTHONPATH}"

echo "✅ RiPIT environment activated"
echo "   Project: $(basename $PROJECT_ROOT)"
echo "   RiPIT Home: $RIPIT_HOME"
echo "   Python: $(which python3)"
echo ""
echo "To deactivate: deactivate"
ACTIVATE_SCRIPT

chmod +x "${PROJECT_ROOT}/activate_ripit.sh"
echo -e "${GREEN}✅ Activation helper created: activate_ripit.sh${NC}"

# Step 9: Update .gitignore
echo -e "${BLUE}[9/9]${NC} Updating .gitignore..."

if ! grep -q ".venv/" "${PROJECT_ROOT}/.gitignore" 2>/dev/null; then
    cat >> "${PROJECT_ROOT}/.gitignore" << 'GITIGNORE'

# RiPIT local installation
.venv/
.ripit/.git/
.ripit/playbooks/*.json
GITIGNORE
    echo -e "${GREEN}✅ .gitignore updated${NC}"
else
    echo -e "${YELLOW}⚠️  .gitignore already configured${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ RiPIT Installation Complete!                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Installation Summary:${NC}"
echo -e "  ${GREEN}✅${NC} Virtual environment: .venv/"
echo -e "  ${GREEN}✅${NC} RiPIT framework: .ripit/"
echo -e "  ${GREEN}✅${NC} ACE stub: installed to venv"
echo -e "  ${GREEN}✅${NC} Claude commands: .claude/commands/"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Activate RiPIT environment:"
echo -e "     ${YELLOW}source ./activate_ripit.sh${NC}"
echo -e ""
echo -e "  2. Use RiPIT in Claude Code:"
echo -e "     ${YELLOW}/workflow${NC} - Full autonomous workflow"
echo -e "     ${YELLOW}/research${NC} - Research phase"
echo -e "     ${YELLOW}/plan${NC} - Planning phase"
echo -e "     ${YELLOW}/implement${NC} - Implementation phase"
echo -e "     ${YELLOW}/test${NC} - Testing phase"
echo -e ""
echo -e "  3. Deactivate when done:"
echo -e "     ${YELLOW}deactivate${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC} .ripit/README.md"
echo ""

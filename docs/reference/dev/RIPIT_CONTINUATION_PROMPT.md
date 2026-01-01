# T.A.R.S. RiPIT Integration Continuation Prompt

**Version:** 1.0.0
**Date:** December 27, 2025
**Project:** VDS_TARS
**RiPIT Version:** 1.6

---

## New Session Initialization Prompt

Copy and paste the following prompt to continue development with RiPIT integration:

---

```markdown
# T.A.R.S. Phase 21 - User/Production Testing with RiPIT

## Project Context

**Repository:** VDS_TARS (Temporal Augmented Retrieval System)
**Branch:** main
**Version:** v1.0.10 (GA)
**MVP Status:** 100% Complete
**Current Phase:** Phase 21 - User/Production Testing

## RiPIT Framework

RiPIT v1.6 is installed and configured for this project.

### Activation
```bash
source ./activate_ripit.sh
```

### Framework Location
- **RiPIT Home:** `.ripit/`
- **Virtual Environment:** `.venv/`
- **ACE Stub:** v1.0.0-stub

### Available Agent Playbooks (13 total)
1. architecture_agent
2. backend_implementation_agent
3. context_manager_agent
4. data_layer_agent
5. domain_expert_agent
6. frontend_implementation_agent
7. implementation_planner_agent
8. integration_agent
9. qa_review_agent
10. research_agent
11. spec_writer_agent
12. test_agent_playbook
13. test_generation_agent

### RiPIT Integration Pattern
```python
import sys
sys.path.insert(0, '.ripit')

from ace_integration.agent_wrapper import AgentWrapper

# Create deployment agent
agent = AgentWrapper(
    name="deployment_agent",
    role="Home network deployment and configuration specialist",
    ripit_home=".ripit"
)

# Playbooks auto-save to .ripit/playbooks/
```

## Current Sprint Goals

### Primary Objective
Deploy T.A.R.S. as a home network Chatbot/RAG system with:
- NAS as knowledge base
- Desktops and mobile devices as clients
- Ollama for local LLM inference
- ChromaDB for vector storage

### Required User Information
Before proceeding, collect:
1. NAS configuration (type, IP, path, protocol)
2. Host machine specs (RAM, GPU, OS)
3. Network topology (LAN-only vs external)
4. LLM preference (default: mistral:7b-instruct)

### Phase 21 Tasks

#### HIGH PRIORITY
1. Security middleware integration
2. Create home network configuration (tars-home.yml)
3. NAS mount configuration
4. First deployment validation

#### MEDIUM PRIORITY
5. Client device testing (desktop/mobile browsers)
6. Documentation cleanup
7. Performance benchmarks

## Development Protocol

Follow the Confidence-Driven Development Protocol:
- See: `docs/reference/dev/CONFIDENCE_DRIVEN_DEVELOPMENT.md`

### Quick Reference
- **≥95% confidence:** Implement immediately
- **90-94% confidence:** Implement with noted uncertainties
- **<90% confidence:** STOP - Present options to user

### Mandatory Test Structure
Every change requires:
- Unit test
- Edge case test
- Regression test

## Key Files

### Backend
- `backend/app/main.py` - FastAPI entry point
- `backend/app/services/rag_service.py` - RAG orchestration
- `backend/app/services/nas_watcher.py` - NAS integration
- `backend/app/core/sanitize.py` - XSS protection
- `backend/app/middleware/security_headers.py` - Security headers

### Configuration
- `docker-compose.yaml` - Container orchestration
- `docs/CONFIGURATION_GUIDE.md` - Config reference

### Security
- `security/certificate_monitor.py` - TLS monitoring
- `cognition/shared/auth.py` - JWT authentication
- `cognition/shared/rate_limiter.py` - Rate limiting

### Documentation
- `docs/reference/dev/DEV_NOTES_20251227.md` - Latest notes
- `docs/reference/dev/HANDOFF_PHASE21_USER_TESTING.md` - Sprint handoff
- `scripts/handoff/HANDOFF_NEXT_SPRINT.md` - Task priorities

## Verification Commands

```bash
# Test collection
python -m pytest --collect-only  # Expect 1,313 tests

# Security tests
python -m pytest tests/security/ -v

# Syntax validation
python -m py_compile backend/app/core/sanitize.py
python -m py_compile backend/app/middleware/security_headers.py
python -m py_compile security/certificate_monitor.py

# RiPIT verification
.venv/Scripts/python.exe -c "import ace; print(ace.__version__)"
```

## Session Objectives

1. Collect environment information from user
2. Create deployment configuration for home network
3. Validate NAS connectivity
4. Execute first deployment test
5. Begin client device testing

## RiPIT Agent Delegation Strategy

For Phase 21, delegate tasks to these agents:

| Task | Agent | Purpose |
|------|-------|---------|
| Architecture review | architecture_agent | Validate deployment design |
| Backend integration | backend_implementation_agent | Security middleware |
| Frontend testing | frontend_implementation_agent | Client compatibility |
| Test generation | test_generation_agent | Deployment tests |
| QA validation | qa_review_agent | UAT checklist |
| Integration | integration_agent | Component connectivity |

## Notes from Previous Session

- MVP reached 100% completion
- Phase 20 security hardening completed
- Security modules created but not yet integrated into main app
- Awaiting user environment details for home network deployment
- RiPIT framework ready with 13 agent playbooks

## GitHub References

- **T.A.R.S. Repository:** https://github.com/oceanrockr/VDS_TARS.git
- **RiPIT Framework:** https://github.com/Veleron-Dev-Studios-LLC/VDS_RiPIT-Agent-Coding-Workflow

---

Begin by asking the user for their NAS and host machine configuration details.
```

---

## RiPIT Agent Initialization Scripts

### Deployment Agent Template

```python
#!/usr/bin/env python3
"""
T.A.R.S. Deployment Agent - RiPIT Integration
Handles home network deployment configuration and validation
"""

import sys
import os

# Add RiPIT to path
sys.path.insert(0, '.ripit')

from ace_integration.agent_wrapper import AgentWrapper
from ace_integration.playbook_manager import PlaybookManager

def create_deployment_agent():
    """Initialize the deployment specialist agent"""
    agent = AgentWrapper(
        name="deployment_agent",
        role="""Home network deployment and configuration specialist.
        Responsible for:
        - NAS mount configuration
        - Docker Compose setup
        - Network configuration
        - Health check validation
        - Client device compatibility""",
        ripit_home=".ripit"
    )
    return agent

def create_testing_agent():
    """Initialize the UAT testing agent"""
    agent = AgentWrapper(
        name="uat_testing_agent",
        role="""User acceptance testing specialist.
        Responsible for:
        - Desktop browser testing
        - Mobile browser testing
        - RAG query validation
        - WebSocket chat testing
        - Performance benchmarking""",
        ripit_home=".ripit"
    )
    return agent

def main():
    # Initialize agents
    deployment_agent = create_deployment_agent()
    testing_agent = create_testing_agent()

    # Initialize playbook manager
    pm = PlaybookManager(ripit_home=".ripit")

    print("RiPIT Agents Initialized:")
    print(f"  - Deployment Agent: {deployment_agent.name}")
    print(f"  - Testing Agent: {testing_agent.name}")
    print(f"\nPlaybooks available: {len(pm.list_playbooks())}")

if __name__ == "__main__":
    main()
```

### Save as: `.ripit/agents/deployment_agents.py`

---

## RiPIT Workflow Integration

### Step 1: Activate Environment

```bash
# Windows
source ./activate_ripit.sh

# Or directly
.venv/Scripts/activate
export RIPIT_HOME=".ripit"
```

### Step 2: Run Agent Scripts

```bash
# Run deployment agent
.venv/Scripts/python.exe .ripit/agents/deployment_agents.py

# Run integration tests
.venv/Scripts/python.exe .ripit/ace_integration/test_integration.py
```

### Step 3: Monitor Playbook Learning

```bash
# Check playbook updates
ls -la .ripit/playbooks/

# View specific playbook
cat .ripit/playbooks/deployment_agent.json
```

---

## Quick Start Commands

```bash
# 1. Activate RiPIT
source ./activate_ripit.sh

# 2. Verify installation
.venv/Scripts/python.exe -c "import ace; print(f'ACE v{ace.__version__}')"

# 3. Run tests
python -m pytest tests/security/ -v

# 4. Start API server (for testing)
python scripts/run_api_server.py

# 5. Check Docker status
docker-compose ps
```

---

## Troubleshooting

### RiPIT Import Errors

```bash
# Ensure RIPIT_HOME is set
export RIPIT_HOME=".ripit"

# Add to Python path
export PYTHONPATH=".:$PYTHONPATH"
```

### Windows Path Issues

```bash
# Use forward slashes
.venv/Scripts/python.exe ./script.py

# Or PowerShell
.\.venv\Scripts\python.exe .\script.py
```

### ACE Stub Not Found

```bash
# Reinstall ACE stub
pip install -e .ripit/ace_stub/
```

---

## Session Checklist

- [ ] RiPIT environment activated
- [ ] ACE stub verified (v1.0.0-stub)
- [ ] User environment info collected
- [ ] tars-home.yml created
- [ ] NAS mount configured
- [ ] First deployment successful
- [ ] Health checks passing
- [ ] Client testing initiated

---

**Document Status:** Active
**Last Updated:** December 27, 2025
**Author:** Claude Opus 4.5

---

*RiPIT Integration Guide for T.A.R.S. Phase 21*

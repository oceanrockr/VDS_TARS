# CLAUDE INIT — T.A.R.S. LocalLLM Desktop MVP

**Project Name:** T.A.R.S. (Temporal Augmented Retrieval System)  
**Codename:** ABRV  
**Version:** v0.1.0-alpha  
**Primary Compute Node:** Dell XPS 8950 (GPU-accelerated)  
**Storage Node:** Synology DS1515+ NAS  
**Core Mission:** Build a distributed, on-premises LLM system that operates as a private, LAN-based inference and retrieval hub with real-time WebSocket streaming and multi-device accessibility.

## Project Context

T.A.R.S. is a privacy-preserving language model platform designed to deliver conversational AI and document reasoning capabilities entirely within a local network. The system eliminates cloud dependencies while maintaining professional-grade performance through GPU acceleration and intelligent document indexing.

The architecture fuses local inference, retrieval-augmented generation (RAG), and temporal context memory through WebSocket-streamed responses. Data is indexed from both the central NAS and client machines, stored as vector embeddings in ChromaDB, and served to users through an Electron-based desktop UI or browser interface.

This MVP establishes personal AI infrastructure that runs on consumer hardware without sacrificing privacy, performance, or functionality, proving that sophisticated AI systems can exist outside cloud platforms while maintaining enterprise-level capabilities.

## Core Objectives

1. **Privacy-Preserving AI:** All computation and storage remain within the local network; zero external API dependencies
2. **GPU-Accelerated Inference:** Run Ollama or LM Studio models (Mistral 7B default) on XPS 8950 hardware
3. **Temporal Augmented Retrieval:** Index and semantically query documents from NAS and client devices
4. **Low-Latency Communication:** FastAPI WebSocket gateway delivers sub-200ms token streaming
5. **Multi-Client Ecosystem:** Lightweight agents monitor local folders and synchronize embeddings
6. **Unified Interface:** React-based Electron UI with chat, dashboard, and administration capabilities
7. **LAN Security:** JWT authentication, HTTPS encryption, and optional VPN integration

## Technical Foundation

### Core Infrastructure
- **LLM Runtime:** Ollama (primary) / LM Studio (alternative)
- **Recommended Models:** Mistral 7B Instruct, Llama 3.1 (8B), Phi-3 Mini (3.8B)
- **Backend Framework:** FastAPI with async WebSocket support
- **Retrieval System:** LangChain + ChromaDB for vector storage and semantic search
- **Containerization:** Docker Compose for orchestrated deployment
- **UI Layer:** Electron 28 with React 18 + Tailwind CSS 3
- **Document Storage:** NAS-mounted volumes via NFS/SMB protocols

### Network Architecture
- **Switch:** TP-Link TL-SG108E (Gigabit backbone)
- **Mesh Network:** Google Nest WiFi Pro 6E
- **Gateway Router:** CenturyLink C4000XG
- **Static IP Allocation:** 192.168.0.10 (NAS), 192.168.0.11 (Desktop)
- **Local DNS:** llm.local → Desktop Hub, nas.local → NAS
- **Security Layer:** JWT authentication, local HTTPS, optional Tailscale VPN

## System Capabilities

### Deliverables & Success Metrics
- ✅ Live WebSocket streaming with ≤ 300ms token latency over LAN
- ✅ Concurrent document ingestion from NAS and client directories
- ✅ Persistent vector memory with automatic ChromaDB backups to Synology
- ✅ One-command Docker Compose deployment
- ✅ 24-hour continuous operation stability with zero external API calls
- ✅ 90% backend unit test coverage, 85% E2E coverage
- ✅ Multi-device support: Windows PC, Surface tablet, Android clients

### Hub-and-Spoke Architecture

**Central Hub (XPS 8950):**
- Hosts Ollama LLM inference engine
- Runs FastAPI WebSocket gateway
- Manages global ChromaDB vector index
- Serves admin dashboard and monitoring

**Storage Layer (DS1515+ NAS):**
- Shared document repository (personal + business)
- Vector database backups and versioning
- Conversation history archives
- System configuration persistence

**Client Agents (PC/Surface/Android):**
- Local document monitoring and ingestion
- Distributed embedding generation
- Query routing to central inference engine
- Local caching for offline access

## Development Phases

**Phase 1 (Weeks 1-2):** Infrastructure setup — Docker, Ollama, FastAPI, network configuration  
**Phase 2 (Weeks 3-4):** WebSocket gateway implementation with token streaming  
**Phase 3 (Weeks 5-6):** Document indexing, RAG pipeline, NAS integration  
**Phase 4 (Weeks 7-8):** Client agent development for distributed monitoring  
**Phase 5 (Weeks 9-10):** UI and admin dashboard creation  
**Phase 6 (Weeks 11-12):** Testing, optimization, documentation, and v0.1.0-alpha release

## Repository Structure

```
LocalLLM_Desktop/
├── claude.md                  # This file - project initialization
├── PRD.md                     # Product requirements document
├── RULES.md                   # Governance and methodology
├── PLANNING.md                # Detailed roadmap and phases
├── TASKS.md                   # Granular task breakdown
├── AGENTS.md                  # Agent roles and responsibilities
├── UI_FRAMEWORK.md            # Design system and UI specs
├── docker/
│   ├── docker-compose.yml
│   ├── ollama/
│   ├── backend/
│   └── chromadb/
├── backend/
│   ├── api/                   # FastAPI endpoints
│   ├── rag/                   # LangChain integration
│   ├── auth/                  # JWT authentication
│   └── websocket/             # Streaming handlers
├── client-agents/
│   ├── python/                # Python file watcher agent
│   └── nodejs/                # Alternative Node.js implementation
├── ui/
│   ├── electron/              # Desktop application
│   └── web/                   # Browser-accessible interface
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── api/                   # OpenAPI specifications
    ├── deployment/            # Setup and deployment guides
    └── architecture/          # System diagrams
```

## Integration Notes

T.A.R.S. is designed as the foundational local AI infrastructure component that can integrate with:
- GridSync project (distributed synchronization capabilities)
- LMS platform (learning management and knowledge organization)
- Additional GPU nodes for horizontal scaling
- Multi-model routing for specialized task delegation
- Future voice input/output capabilities

All documentation artifacts follow standardized Claude ingestion format for cross-project agent coordination and knowledge transfer.

## Governance Principles

1. **Local-Only Architecture:** No external APIs, telemetry, or cloud dependencies
2. **Modular Design:** Loosely coupled components with well-defined interfaces
3. **Security-First:** All network communication authenticated and encrypted
4. **Deterministic Builds:** Docker images pinned by hash for reproducibility
5. **Semantic Versioning:** Clear communication of compatibility and changes
6. **Test-Driven Quality:** Comprehensive automated testing at all levels
7. **Privacy Compliance:** Data encryption at rest and in transit, user control
8. **Agent Autonomy:** Distributed agents operate within defined boundaries
9. **Documentation Standards:** Every feature documented before deployment
10. **Continuous Improvement:** Regular retrospectives and process refinement

## Expected Outcome

At v1.0 release, T.A.R.S. will function as a self-sustaining personal AI infrastructure capable of:
- Context-aware dialogue with temporal memory across conversations
- Secure, LAN-only knowledge retrieval across all connected devices
- Seamless integration with personal and business document repositories
- Real-time performance comparable to cloud-based AI services
- Complete privacy preservation through local-only operation
- Modular expansion path for future capabilities and scaling

This document serves as the foundation for development, onboarding, and handoff to both human developers and AI agents responsible for delivering the T.A.R.S. LocalLLM Desktop MVP.
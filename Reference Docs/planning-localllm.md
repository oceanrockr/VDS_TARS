# PROJECT PLANNING — T.A.R.S. LocalLLM Desktop MVP

**Project:** T.A.R.S. (Temporal Augmented Retrieval System)  
**Timeline:** 12 weeks (3 months)  
**Methodology:** Phased development with iterative delivery  
**Team:** Multi-agent distributed development

## Overview

This document outlines the complete development roadmap for T.A.R.S., organized into six strategic phases delivering incremental value. Each phase builds upon previous work and includes defined deliverables, success criteria, and risk mitigation strategies.

---

## Phase 1: Foundation & Infrastructure (Weeks 1-2)

### Objectives
Establish core infrastructure stack and validate architectural approach with minimal working system. Create stable foundation for all subsequent development.

### Key Deliverables
- ✅ Docker Compose configuration (Ollama, FastAPI, ChromaDB)
- ✅ GPU passthrough validation and optimization
- ✅ Network configuration (static IPs, local DNS, firewall)
- ✅ SSL/TLS certificate generation and installation
- ✅ Health check endpoints and monitoring
- ✅ Development environment setup documentation
- ✅ Base CI/CD pipeline configuration

### Technical Tasks

**Infrastructure Setup (T-001 to T-005):**
1. Install Docker Desktop with NVIDIA GPU support on XPS 8950
2. Create `docker-compose.yml` with core services:
   - Ollama container with GPU passthrough configuration
   - FastAPI backend container with volume mounts
   - ChromaDB vector database with persistent storage
   - Optional Nginx/Caddy reverse proxy for HTTPS termination
3. Validate GPU access within containers using `nvidia-smi`
4. Configure container networking and service discovery
5. Set up container auto-restart policies

**Network Configuration (T-006 to T-011):**
1. Configure Synology NAS:
   - Create shared folder `/volume1/LLM_docs`
   - Set up NFS export with appropriate permissions
   - Configure SMB share as Windows fallback
   - Test read/write access from Docker host
2. Network setup:
   - Assign static IP 192.168.0.11 to XPS 8950 (via router DHCP reservation)
   - Assign static IP 192.168.0.10 to DS1515+ NAS
   - Configure local DNS entries (`llm.local`, `nas.local`) in router or via hosts file
   - Open required firewall ports: 8000 (API), 443 (HTTPS), 11434 (Ollama)
   - Verify connectivity between all network nodes

**Model Setup (T-012 to T-015):**
1. Pull Mistral 7B Instruct model via Ollama CLI
2. Test inference with sample prompts
3. Benchmark token generation speed (target: 20+ tokens/sec)
4. Measure GPU memory usage and VRAM allocation
5. Document cold start vs warm inference latency

### Success Criteria
- ✅ Ollama responds to API queries with < 2 second cold start
- ✅ Docker containers auto-restart on system reboot
- ✅ GPU utilization visible and > 70% during active inference
- ✅ NAS mount accessible from Docker container with read/write
- ✅ Network latency between nodes < 5ms (ping test)

### Risks & Mitigation Strategies
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GPU passthrough failures | High | Medium | Test with `docker run --gpus all` first, prepare CPU fallback |
| Network connectivity issues | Medium | Low | Benchmark with `iperf3`, validate gigabit speeds |
| Docker compatibility on Windows | Medium | Medium | Use WSL2 backend, validate with test containers |
| NFS mount permission errors | Low | Medium | Document exact permission requirements, use SMB fallback |

### Exit Criteria
- All infrastructure components running and communicating
- GPU inference working at acceptable speed
- Documentation complete for environment setup
- Team can independently reproduce environment

---

## Phase 2: WebSocket Gateway (Weeks 3-4)

### Objectives
Implement real-time bidirectional communication layer for low-latency token streaming. Create foundation for responsive chat experience.

### Key Deliverables
- ✅ FastAPI WebSocket endpoint (`/ws/chat`)
- ✅ Async token streaming from Ollama
- ✅ Client reconnection and error handling logic
- ✅ Message queue for request sequencing
- ✅ Load testing results and performance baseline
- ✅ WebSocket client example code

### Technical Tasks

**Backend WebSocket Implementation (T-020 to T-024):**
1. Create FastAPI WebSocket route with connection manager
2. Implement connection lifecycle handlers (connect, disconnect, error)
3. Build async generator for token streaming from Ollama
4. Add connection heartbeat/keepalive (ping/pong every 30 seconds)
5. Implement message routing and queuing system
6. Add graceful shutdown handling
7. Create connection pool for managing multiple clients

**Ollama Integration (T-023, T-029):**
1. Integrate `ollama` Python SDK
2. Configure streaming parameters (temperature, max_tokens, top_p)
3. Handle model loading and unloading
4. Implement timeout and cancellation logic
5. Add error handling for model failures
6. Create fallback logic for overloaded GPU

**Authentication Layer (T-025 to T-027):**
1. Implement JWT token generation endpoint (`/auth/token`)
2. Add WebSocket handshake validation middleware
3. Create token refresh mechanism with sliding expiration
4. Build device registration system
5. Add rate limiting per client
6. Implement token blacklist for logout

**Testing & Validation (T-030 to T-033):**
1. Write unit tests for WebSocket handlers
2. Create integration tests with mock Ollama responses
3. Build load test scenarios with k6:
   - 10 concurrent connections
   - 100 messages per minute per connection
   - Connection churn (connect/disconnect cycles)
4. Measure and document latency distribution
5. Profile memory usage under load

### Success Criteria
- ✅ Token latency < 100ms between backend receipt and client delivery
- ✅ Support 10+ concurrent WebSocket connections without degradation
- ✅ Graceful reconnection within 5 seconds of disconnect
- ✅ < 1% packet loss under normal load
- ✅ Zero connection leaks or memory growth
- ✅ Error messages clear and actionable

### Risks & Mitigation Strategies
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| WebSocket stability issues | High | Medium | Implement heartbeat, test with network simulators |
| Token streaming lag | High | Medium | Profile async code, optimize buffer sizes |
| Connection leak under load | Medium | Low | Use connection pooling, add monitoring |
| Authentication bypass | Critical | Low | Security review, penetration testing |

### Exit Criteria
- WebSocket endpoint functional and performant
- Load testing confirms capacity targets
- Client reconnection reliable
- Authentication working correctly

---

## Phase 3: Document Indexing & RAG (Weeks 5-6)

### Objectives
Build retrieval-augmented generation pipeline with document ingestion and semantic search capabilities. Enable context-aware responses from personal documents.

### Key Deliverables
- ✅ NAS document indexing service
- ✅ LangChain RAG pipeline integration
- ✅ ChromaDB persistence configuration
- ✅ Document format parsers (PDF, DOCX, TXT, MD, CSV)
- ✅ Embedding generation pipeline
- ✅ Query-document relevance scoring system

### Technical Tasks

**Document Ingestion (T-040 to T-045):**
1. Design chunking strategy (512 tokens, 50-token overlap)
2. Mount `/volume1/LLM_docs` to backend container
3. Create file watcher service using `watchdog` library
4. Implement document loaders:
   - PDF: PyPDF2 or pdfplumber (with OCR fallback)
   - DOCX: python-docx
   - TXT/MD: Standard file readers with encoding detection
   - CSV: Pandas with automatic delimiter detection
5. Build text extraction and cleaning pipeline
6. Handle edge cases (corrupted files, unsupported formats)

**Embedding Pipeline (T-046 to T-050):**
1. Deploy `sentence-transformers/all-MiniLM-L6-v2` locally
2. Create batch embedding generation system
3. Implement chunking with semantic boundaries (preserve sentences)
4. Generate embeddings with progress tracking
5. Store vectors in ChromaDB with rich metadata:
   - Document path and filename
   - Creation and modification timestamps
   - File type and size
   - Chunk position and context
6. Build index rebuild functionality
7. Add incremental update support

**Retrieval System (T-051 to T-056):**
1. Implement similarity search with configurable top-k
2. Add reranking with cross-encoder model (optional enhancement)
3. Build context window manager:
   - Calculate token counts for retrieved chunks
   - Prioritize most relevant content
   - Ensure context fits within model limits
4. Create citation extraction logic (track source documents)
5. Format retrieved context for LLM prompting
6. Handle edge cases:
   - No relevant documents found
   - Context too large for window
   - Multiple equally relevant sources

**RAG Integration (T-055 to T-058):**
1. Integrate retrieval into chat pipeline
2. Inject context into prompts with clear demarcation
3. Format citations in model responses
4. Add metadata passthrough for source tracking
5. Implement relevance threshold filtering
6. Create fallback behavior when no relevant docs

### Success Criteria
- ✅ Successfully index 1,000+ documents from NAS
- ✅ Query retrieval time < 200ms for 10K document corpus
- ✅ Embedding generation rate > 100 docs/minute
- ✅ Relevance score > 0.7 for correct retrievals in test set
- ✅ Citations accurate and traceable to source documents
- ✅ Zero data loss or corruption during indexing

### Risks & Mitigation Strategies
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Slow embedding generation | High | Medium | Batch processing, GPU acceleration for embeddings |
| Poor retrieval quality | High | Medium | Evaluate with test query set, tune chunk size and overlap |
| ChromaDB corruption | Medium | Low | Regular backups, validation checks |
| Out of memory errors | Medium | Medium | Stream processing, memory monitoring |

### Exit Criteria
- Document indexing reliable and fast
- Retrieval quality meets acceptance criteria
- RAG responses include relevant context
- System handles large document sets

---

## Phase 4: Client Agent Development (Weeks 7-8)

### Objectives
Create lightweight client applications enabling distributed document access and local embedding generation. Enable seamless multi-device document ingestion.

### Key Deliverables
- ✅ Python client agent (Windows/Linux primary)
- ✅ Node.js client agent (alternative implementation)
- ✅ Client registration and authentication flow
- ✅ Local folder monitoring with configurable paths
- ✅ Incremental sync protocol
- ✅ Installation packages (Windows installer, portable executable)
- ✅ Client agent documentation

### Technical Tasks

**Agent Architecture (T-070 to T-074):**
1. Design client-server protocol specification
2. Create Python agent project scaffold:
   - Configuration management (YAML/JSON)
   - Logging system (rotating files)
   - Service wrapper for background operation
3. Implement client registration handshake:
   - Device identification
   - Capability negotiation
   - Authentication token exchange
4. Build local configuration UI or CLI
5. Create status monitoring and diagnostics

**File Monitoring (T-075 to T-077):**
1. Implement file watcher using `watchdog` (Python) or `chokidar` (Node.js)
2. Configure watched directories from config file
3. Filter by supported file extensions
4. Implement debounce logic (wait 5 seconds after last change)
5. Handle large file batches efficiently
6. Detect file moves and renames
7. Support multiple watched directories

**Local Embedding (T-078 to T-080):**
1. Deploy lightweight embedding model to client
2. Create local embedding generation pipeline
3. Implement batch processing for efficiency
4. Compress vectors before transmission (gzip)
5. Track sync status per document
6. Store local cache of embeddings
7. Handle model updates gracefully

**Network Communication (T-081 to T-084):**
1. Build sync protocol over WebSocket or HTTP/2
2. Implement retry logic with exponential backoff
3. Handle network interruptions gracefully:
   - Queue pending updates
   - Resume from last successful sync
   - Validate sync completion
4. Add bandwidth throttling (configurable)
5. Compress payloads for efficiency
6. Implement chunk-based upload for large files

**Packaging & Distribution (T-085 to T-088):**
1. Create Windows installer using NSIS or WiX Toolset
2. Build portable executables with PyInstaller or pkg
3. Include auto-update mechanism
4. Write installation and configuration guide
5. Create uninstall process
6. Test on multiple Windows versions
7. Validate installation on clean systems

### Success Criteria
- ✅ Agent installs in < 5 minutes with clear instructions
- ✅ Detects and syncs new documents within 30 seconds
- ✅ Memory footprint < 200 MB per agent
- ✅ CPU usage < 5% when idle, < 20% during sync
- ✅ Successful reconnection after network outage
- ✅ Zero document loss during sync failures
- ✅ Runs reliably as background service

### Risks & Mitigation Strategies
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| High resource usage on client devices | High | Medium | Optimize embedding model, add rate limiting |
| Sync conflicts with concurrent edits | Medium | Low | Last-write-wins strategy, conflict detection |
| Installation failures on diverse systems | Medium | Medium | Comprehensive testing, detailed troubleshooting guide |
| Security vulnerabilities in client | High | Low | Code review, security testing, signed executables |

### Exit Criteria
- Client agents operational on 3+ test devices
- Document sync reliable and efficient
- Installation process smooth and well-documented
- Resource usage within acceptable limits

---

## Phase 5: UI & Dashboard (Weeks 9-10)

### Objectives
Build intuitive user interfaces for chat interaction and system administration. Create polished user experience matching cloud-based alternatives.

### Key Deliverables
- ✅ Electron chat application (Windows primary)
- ✅ React/Tailwind component library
- ✅ Admin dashboard web interface
- ✅ System monitoring visualizations
- ✅ Settings and configuration panels
- ✅ User documentation and tutorials

### Technical Tasks

**Electron Application Setup (T-100 to T-102):**
1. Scaffold with Electron Forge + Vite + React + TypeScript
2. Configure build pipeline and hot module replacement
3. Set up Tailwind CSS with custom design system
4. Integrate ShadCN/UI component library
5. Configure Electron window management
6. Add system tray integration
7. Implement app menu structure

**Chat Interface (T-103 to T-113):**
1. Build message list component with virtual scrolling
2. Create message bubble components (user vs assistant)
3. Implement markdown renderer with `react-markdown`
4. Add code highlighting with Prism.js
5. Build WebSocket client with reconnection logic
6. Create typing indicators and loading states
7. Implement message input field with multiline support
8. Add file attachment UI with drag-and-drop
9. Build conversation history with local storage
10. Add full-text search across conversations
11. Create conversation management (new, rename, delete)

**Admin Dashboard (T-115 to T-122):**
1. Design dashboard layout and navigation
2. Build system health cards:
   - GPU temperature, utilization, VRAM
   - CPU usage and temperature
   - Memory usage (system and Docker)
   - Disk usage and I/O stats
3. Create connected clients table:
   - Device name and type
   - Connection status (online/offline/error)
   - Last activity timestamp
   - Data sync status
4. Build query logs view:
   - Recent queries with timestamps
   - Response times and token counts
   - Filtering and search
5. Add document index statistics:
   - Total documents and chunks
   - Storage usage
   - Indexing queue status
6. Create model management interface:
   - List available models
   - Load/unload/download models
   - Model configuration (temperature, etc.)
7. Implement real-time metric charts with Recharts
8. Add backup management UI

**Settings & Configuration (T-114, T-122):**
1. Build tabbed settings panel
2. Create general settings (theme, language)
3. Add model configuration section
4. Build network settings (endpoint URLs)
5. Create privacy settings (data retention)
6. Add about section (version, licenses)
7. Implement error notification system (toasts)

**Testing & Polish (T-123 to T-124):**
1. Test on Surface Pro with touch input
2. Optimize bundle size with code splitting
3. Add loading states and skeletons
4. Implement accessibility features (keyboard nav, ARIA)
5. Create onboarding flow for first-time users
6. Add tooltips and help text
7. Polish animations and transitions

### Success Criteria
- ✅ App launches in < 3 seconds cold start
- ✅ Chat messages render in < 50ms
- ✅ Dashboard loads all metrics in < 1 second
- ✅ Zero UI blocking during inference
- ✅ Responsive on Surface Pro touchscreen
- ✅ Keyboard shortcuts functional
- ✅ No accessibility violations (automated testing)

### Risks & Mitigation Strategies
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Poor Electron performance | High | Medium | Lazy load components, optimize bundle size |
| Complex state management | Medium | Medium | Use Zustand for simplicity, avoid over-engineering |
| UI/UX not intuitive | Medium | Medium | User testing, iterate based on feedback |
| Accessibility issues | Low | Medium | Automated testing, manual keyboard testing |

### Exit Criteria
- UI functional and polished
- Dashboard provides actionable insights
- User feedback positive on usability
- Performance meets targets

---

## Phase 6: Testing, Optimization & Release (Weeks 11-12)

### Objectives
Validate system stability, optimize performance, finalize documentation, and prepare for production deployment.

### Key Deliverables
- ✅ Comprehensive test suite (unit, integration, E2E)
- ✅ Performance optimization report
- ✅ Production deployment guide
- ✅ User documentation and tutorials
- ✅ v0.1.0-alpha release package
- ✅ Known issues and roadmap document

### Technical Tasks

**Testing (T-140 to T-146):**
1. Write backend unit tests for all modules
   - FastAPI routes and WebSocket handlers
   - RAG pipeline components
   - Authentication and authorization
   - Document parsers and processors
2. Achieve 90%+ code coverage
3. Create API integration tests
   - All endpoints with valid/invalid inputs
   - Authentication flows
   - WebSocket lifecycle
4. Build E2E test suite with Playwright
   - User registration and login
   - Chat conversation flows
   - Document upload and retrieval
   - Settings modification
5. Write load test scenarios with k6
   - Concurrent users
   - Sustained query load
   - Stress testing (breaking point)
6. Execute security scanning
   - Python: Bandit, Safety
   - JavaScript: npm audit, Snyk
   - Docker: Trivy image scanning
   - Penetration testing (basic)

**Performance Optimization (T-147 to T-151):**
1. Profile backend with cProfile and memory_profiler
2. Identify and optimize slow queries/operations
3. Reduce Docker image sizes (multi-stage builds)
4. Tune ChromaDB parameters for performance
5. Optimize embedding batch sizes
6. Add caching where appropriate
7. Database query optimization
8. Memory leak detection and fixing

**Documentation (T-152 to T-156):**
1. Write comprehensive installation guide
   - System requirements
   - Step-by-step setup instructions
   - Common installation issues
2. Create configuration reference
   - All environment variables
   - Docker Compose options
   - Client agent configuration
3. Document all API endpoints (OpenAPI spec)
4. Build troubleshooting guide
   - Common error messages
   - Diagnostic procedures
   - Recovery steps
5. Record setup and usage video tutorials
6. Create quick start guide (5-minute setup)

**Release Preparation (T-157 to T-166):**
1. Finalize version numbers (v0.1.0-alpha)
2. Update CHANGELOG.md with all changes
3. Write release notes highlighting features
4. Create distribution packages
   - Docker images (multi-arch)
   - Client installers (Windows, Linux)
   - Source code archives
5. Tag Git release with signed tag
6. Deploy to test environment
7. Conduct user acceptance testing
   - 3+ test users
   - Diverse hardware configurations
   - Real-world usage scenarios
8. Execute 72-hour stability test
   - Continuous operation
   - Monitoring for errors/crashes
   - Resource usage tracking
9. Test rollback procedures
10. Final deployment to production
11. Post-deployment verification

### Success Criteria
- ✅ All tests passing with >90% backend coverage
- ✅ Zero critical bugs or security issues
- ✅ 72-hour continuous uptime achieved
- ✅ Performance benchmarks met or exceeded
- ✅ Documentation complete and validated
- ✅ Installation success rate 95%+ in user testing
- ✅ Release artifacts built and verified

### Risks & Mitigation Strategies
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Late-discovered critical bugs | High | Medium | Start testing early, maintain bug triage process |
| Performance regressions | Medium | Medium | Continuous benchmarking throughout development |
| Incomplete documentation | Medium | High | Parallel documentation with development |
| User acceptance issues | Medium | Medium | Early user testing, iterate on feedback |

### Exit Criteria
- All tests passing
- Performance validated
- Documentation complete
- User acceptance positive
- Release artifacts ready
- v0.1.0-alpha deployed

---

## Post-MVP Roadmap

### Phase 7: Mobile & Voice (Q2 2026)
- Native Android application
- iOS feasibility study and potential development
- Voice input integration (Whisper)
- Text-to-speech output (Piper/Coqui)
- Mobile-optimized UI

### Phase 8: Advanced Features (Q3 2026)
- Multi-user support with role-based access control
- Custom model fine-tuning interface
- Plugin/extension system for integrations
- Advanced analytics and insights
- Collaborative features

### Phase 9: Scaling & Distribution (Q4 2026)
- Multi-node inference clustering
- Load balancing across multiple GPUs
- Distributed vector database (scale to millions of docs)
- Edge deployment options
- Cloud-optional hybrid mode

---

## Resource Allocation

### Development Team Structure
| Role | Allocation | Primary Phases |
|------|------------|----------------|
| **Backend Developer** | 60% | 1, 2, 3, 6 |
| **Frontend Developer** | 40% | 5, 6 |
| **DevOps Engineer** | 20% | 1, 6 |
| **Data Engineer** | 30% | 3, 6 |
| **QA Engineer** | 30% | All phases (testing) |

### Hardware Requirements
- **Development:** Dell XPS 8950 desktop (already owned)
- **Storage:** Synology DS1515+ NAS (already owned)
- **Network:** Existing infrastructure (sufficient)
- **Test Devices:** Windows PC, Surface Pro, Android phone/tablet

---

## Milestones & Checkpoints

| Week | Phase | Milestone | Checkpoint |
|------|-------|-----------|------------|
| 2 | Phase 1 | Infrastructure Complete | Ollama running, GPU validated |
| 4 | Phase 2 | WebSocket Functional | Token streaming operational |
| 6 | Phase 3 | RAG Operational | Documents indexed, retrieval working |
| 8 | Phase 4 | Agents Deployed | Multi-device sync functional |
| 10 | Phase 5 | UI Complete | Chat and dashboard operational |
| 12 | Phase 6 | Alpha Release | v0.1.0-alpha ready for use |

---

## Budget & Timeline Considerations

### Timeline Flexibility
- **Best Case:** 10 weeks (aggressive, all goes smoothly)
- **Expected Case:** 12 weeks (realistic with minor setbacks)
- **Worst Case:** 16 weeks (significant blockers or scope expansion)

### Development Effort
- **Total Estimated Hours:** 300-400 hours
- **Part-time Development:** 20-25 hours/week
- **Full-time Development:** 40-50 hours/week

### Cost Analysis
- **Software Licenses:** $0 (all open source)
- **Hardware:** $0 (already owned)
- **Cloud Services:** $0 (local-only architecture)
- **Total Cost:** Developer time only

---

## Communication & Reporting

### Weekly Sync (Every Monday)
- Review TASKS.md progress
- Identify blockers and dependencies
- Adjust priorities as needed
- Next week planning

### Sprint Reviews (Bi-weekly, End of Phase)
- Demo completed features to stakeholders
- Gather feedback and iterate
- Backlog grooming for next sprint
- Retrospective: what went well, what to improve

### Documentation Updates
- CLAUDE LOG updated after major decisions
- CHANGELOG.md updated for each release
- README.md updated for setup changes
- API docs auto-generated from code

---

## Success Definition

T.A.R.S. will be considered successful when:
- ✅ Running continuously for 7+ days without intervention
- ✅ Handling 50+ queries per week across devices
- ✅ Maintaining <300ms latency for 95th percentile
- ✅ User reports preference over cloud alternatives
- ✅ Zero privacy concerns or data leaks
- ✅ System becomes essential part of daily workflow

This planning document is a living artifact and will be updated as the project progresses and new information emerges.
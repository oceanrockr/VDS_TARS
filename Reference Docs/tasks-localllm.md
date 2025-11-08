# TASKS — T.A.R.S. LocalLLM Desktop

**Project:** T.A.R.S. (Temporal Augmented Retrieval System)  
**Status Legend:** ☐ Not Started | 🔄 In Progress | ✅ Complete | 🚫 Blocked | ⏸️ Paused

---

## Phase 1: Foundation & Infrastructure

| ID | Task | Owner | Priority | Status | Dependencies | Est. Hours | Notes |
|----|------|-------|----------|--------|--------------|------------|-------|
| T-001 | Install Docker Desktop with GPU support | Infra Agent | 🔥 Critical | ☐ | - | 2h | Verify NVIDIA drivers first |
| T-002 | Create base `docker-compose.yml` | Infra Agent | 🔥 Critical | ☐ | T-001 | 4h | Include Ollama, FastAPI, ChromaDB |
| T-003 | Configure NVIDIA GPU passthrough | Infra Agent | 🔥 Critical | ☐ | T-001 | 3h | Test with `nvidia-smi` in container |
| T-004 | Pull Mistral 7B model via Ollama | Infra Agent | 🔥 Critical | ☐ | T-002 | 1h | Network bandwidth dependent |
| T-005 | Benchmark GPU inference speed | QA Agent | 🔥 Critical | ☐ | T-004 | 2h | Document tokens/sec baseline |
| T-006 | Set up NAS shared folder structure | Data Agent | 🔥 Critical | ☐ | - | 1h | `/volume1/LLM_docs` |
| T-007 | Configure NFS export on Synology | Data Agent | 🔥 Critical | ☐ | T-006 | 2h | Test permissions, fallback to SMB |
| T-008 | Assign static IP to XPS 8950 | Infra Agent | ✅ High | ☐ | - | 0.5h | 192.168.0.11 via DHCP reservation |
| T-009 | Assign static IP to DS1515+ | Infra Agent | ✅ High | ☐ | - | 0.5h | 192.168.0.10 |
| T-010 | Configure local DNS entries | Infra Agent | ✅ High | ☐ | T-008, T-009 | 1h | llm.local, nas.local |
| T-011 | Generate SSL certificates | Infra Agent | ✅ High | ☐ | T-010 | 1h | Self-signed or Synology certs |
| T-012 | Configure firewall rules | Infra Agent | ✅ High | ☐ | - | 1h | Open ports 8000, 443, 11434 |
| T-013 | Mount NAS to Docker container | Infra Agent | 🔥 Critical | ☐ | T-007 | 2h | Verify read/write access |
| T-014 | Create health check endpoints | Backend Agent | ✅ High | ☐ | T-002 | 2h | `/health`, `/ready`, `/metrics` |
| T-015 | Write infrastructure setup docs | PM Agent | ✅ High | ☐ | T-001-T-014 | 4h | Step-by-step with screenshots |

**Phase 1 Total: 26.5 hours**

---

## Phase 2: WebSocket Gateway

| ID | Task | Owner | Priority | Status | Dependencies | Est. Hours | Notes |
|----|------|-------|----------|--------|--------------|------------|-------|
| T-020 | Design WebSocket protocol spec | Backend Agent | 🔥 Critical | ☐ | - | 3h | Document message formats |
| T-021 | Create FastAPI WebSocket endpoint | Backend Agent | 🔥 Critical | ☐ | T-020 | 4h | `/ws/chat` with connection manager |
| T-022 | Implement connection lifecycle | Backend Agent | 🔥 Critical | ☐ | T-021 | 3h | Connect, disconnect, error handlers |
| T-023 | Build async token streaming | Backend Agent | 🔥 Critical | ☐ | T-021 | 5h | Async generator from Ollama |
| T-024 | Integrate Ollama Python SDK | Backend Agent | 🔥 Critical | ☐ | T-004 | 3h | Streaming API configuration |
| T-025 | Add WebSocket heartbeat | Backend Agent | ✅ High | ☐ | T-021 | 2h | Ping/pong every 30 seconds |
| T-026 | Implement JWT authentication | Backend Agent | 🔥 Critical | ☐ | - | 6h | Token generation endpoint |
| T-027 | Add WebSocket auth validation | Backend Agent | 🔥 Critical | ☐ | T-026 | 3h | Verify JWT on handshake |
| T-028 | Create token refresh mechanism | Backend Agent | ✅ High | ☐ | T-026 | 2h | 24-hour expiry, sliding window |
| T-029 | Build message routing system | Backend Agent | ✅ High | ☐ | T-021 | 4h | Queue for request handling |
| T-030 | Add timeout and error handling | Backend Agent | ✅ High | ☐ | T-023 | 3h | Graceful degradation |
| T-031 | Write WebSocket unit tests | QA Agent | 🔥 Critical | ☐ | T-021-T-030 | 6h | Mock client connections |
| T-032 | Create WebSocket integration tests | QA Agent | 🔥 Critical | ☐ | T-031 | 5h | Real Ollama API calls |
| T-033 | Build load test scenarios | QA Agent | ✅ High | ☐ | T-032 | 4h | k6 scripts for concurrency |
| T-034 | Execute load testing | QA Agent | ✅ High | ☐ | T-033 | 3h | 10+ concurrent connections |
| T-035 | Measure and document latency | QA Agent | ✅ High | ☐ | T-034 | 2h | Baseline performance metrics |

**Phase 2 Total: 58 hours**

---

## Phase 3: Document Indexing & RAG

| ID | Task | Owner | Priority | Status | Dependencies | Est. Hours | Notes |
|----|------|-------|----------|--------|--------------|------------|-------|
| T-040 | Design document chunking strategy | Data Agent | 🔥 Critical | ☐ | - | 3h | 512 tokens, 50 overlap |
| T-041 | Implement PDF parser | Data Agent | 🔥 Critical | ☐ | T-040 | 5h | PyPDF2 with OCR fallback |
| T-042 | Implement DOCX parser | Data Agent | 🔥 Critical | ☐ | T-040 | 3h | python-docx library |
| T-043 | Implement TXT/MD parser | Data Agent | ✅ High | ☐ | T-040 | 2h | Encoding detection |
| T-044 | Add CSV parser | Data Agent | ✅ Medium | ☐ | T-040 | 2h | Pandas integration |
| T-045 | Create file watcher service | Data Agent | 🔥 Critical | ☐ | T-013 | 4h | Monitor NAS folder with watchdog |
| T-046 | Deploy embedding model locally | Data Agent | 🔥 Critical | ☐ | - | 2h | sentence-transformers/all-MiniLM-L6-v2 |
| T-047 | Build embedding generation pipeline | Data Agent | 🔥 Critical | ☐ | T-046 | 6h | Batch processing with progress |
| T-048 | Configure ChromaDB persistence | Data Agent | 🔥 Critical | ☐ | T-002 | 3h | Mount volume for vector data |
| T-049 | Implement vector storage | Data Agent | 🔥 Critical | ☐ | T-048 | 4h | Store embeddings with metadata |
| T-050 | Create document metadata index | Data Agent | ✅ High | ☐ | T-049 | 3h | Paths, timestamps, file types |
| T-051 | Build similarity search function | Backend Agent | 🔥 Critical | ☐ | T-049 | 5h | Top-k retrieval with scoring |
| T-052 | Add reranking with cross-encoder | Backend Agent | ✅ High | ☐ | T-051 | 4h | Improve relevance accuracy |
| T-053 | Implement context window manager | Backend Agent | 🔥 Critical | ☐ | T-051 | 4h | 8K token limit handling |
| T-054 | Build citation extraction | Backend Agent | ✅ High | ☐ | T-051 | 3h | Track source documents |
| T-055 | Integrate RAG into chat pipeline | Backend Agent | 🔥 Critical | ☐ | T-023, T-051 | 6h | Context injection in prompts |
| T-056 | Handle RAG edge cases | Backend Agent | ✅ High | ☐ | T-055 | 3h | No docs, oversized context |
| T-057 | Create test document corpus | QA Agent | ✅ High | ☐ | - | 3h | 1000+ diverse documents |
| T-058 | Test with large document set | QA Agent | 🔥 Critical | ☐ | T-040-T-057 | 4h | Performance validation |
| T-059 | Measure retrieval latency | QA Agent | ✅ High | ☐ | T-058 | 2h | Baseline metrics |
| T-060 | Validate retrieval accuracy | QA Agent | 🔥 Critical | ☐ | T-058 | 4h | Test query set evaluation |

**Phase 3 Total: 75 hours**

---

## Phase 4: Client Agent Development

| ID | Task | Owner | Priority | Status | Dependencies | Est. Hours | Notes |
|----|------|-------|----------|--------|--------------|------------|-------|
| T-070 | Design client-server protocol | Agent Ops | 🔥 Critical | ☐ | T-020 | 4h | Registration, sync, query specs |
| T-071 | Create Python agent scaffold | Agent Ops | 🔥 Critical | ☐ | T-070 | 5h | Project structure, logging |
| T-072 | Implement client registration | Agent Ops | 🔥 Critical | ☐ | T-026, T-071 | 5h | Device ID, JWT exchange |
| T-073 | Build local config management | Agent Ops | ✅ High | ☐ | T-071 | 3h | YAML/JSON config files |
| T-074 | Add logging and diagnostics | Agent Ops | ✅ High | ☐ | T-071 | 2h | Rotating log files |
| T-075 | Implement file watcher | Agent Ops | 🔥 Critical | ☐ | T-071 | 4h | watchdog library integration |
| T-076 | Filter supported file types | Agent Ops | ✅ High | ☐ | T-075 | 2h | PDF, DOCX, TXT, MD, CSV |
| T-077 | Add debounce for rapid changes | Agent Ops | ✅ Medium | ☐ | T-075 | 2h | Wait 5s after last change |
| T-078 | Deploy local embedding model | Agent Ops | 🔥 Critical | ☐ | T-046 | 3h | Client-side model deployment |
| T-079 | Generate embeddings locally | Agent Ops | 🔥 Critical | ☐ | T-078 | 5h | Local inference pipeline |
| T-080 | Compress vectors for transmission | Agent Ops | ✅ High | ☐ | T-079 | 2h | gzip compression |
| T-081 | Implement sync protocol | Agent Ops | 🔥 Critical | ☐ | T-021, T-079 | 6h | Send vectors to central node |
| T-082 | Add retry logic | Agent Ops | ✅ High | ☐ | T-081 | 3h | Exponential backoff |
| T-083 | Handle network interruptions | Agent Ops | ✅ High | ☐ | T-082 | 4h | Graceful degradation |
| T-084 | Queue offline updates | Agent Ops | ✅ High | ☐ | T-083 | 3h | Persist to local SQLite |
| T-085 | Create Node.js agent (alternative) | Agent Ops | ✅ Medium | ☐ | T-070 | 12h | TypeScript implementation |
| T-086 | Create Windows installer | Agent Ops | 🔥 Critical | ☐ | T-071-T-084 | 6h | NSIS or WiX Toolset |
| T-087 | Build portable executable | Agent Ops | ✅ High | ☐ | T-086 | 3h | PyInstaller with dependencies |
| T-088 | Write agent installation docs | PM Agent | 🔥 Critical | ☐ | T-086 | 4h | Step-by-step guide |
| T-089 | Create agent configuration UI | Agent Ops | ✅ Medium | ☐ | T-073 | 5h | Simple Tkinter/Qt interface |
| T-090 | Test agent on Windows PC | QA Agent | 🔥 Critical | ☐ | T-086 | 2h | Primary platform validation |
| T-091 | Test agent on Surface tablet | QA Agent | ✅ High | ☐ | T-086 | 2h | Touch interface validation |
| T-092 | Test agent on Android | QA Agent | ✅ Medium | ☐ | T-086 | 3h | Termux or similar environment |

**Phase 4 Total: 90 hours**

---

## Phase 5: UI & Dashboard

| ID | Task | Owner | Priority | Status | Dependencies | Est. Hours | Notes |
|----|------|-------|----------|--------|--------------|------------|-------|
| T-100 | Scaffold Electron app | UI Agent | 🔥 Critical | ☐ | - | 4h | Electron Forge + Vite + React |
| T-101 | Set up React + Tailwind | UI Agent | 🔥 Critical | ☐ | T-100 | 3h | Configure Tailwind with ShadCN |
| T-102 | Create design system | UI Agent | ✅ High | ☐ | T-101 | 5h | Colors, typography, components |
| T-103 | Build chat message list | UI Agent | 🔥 Critical | ☐ | T-101 | 6h | Virtual scrolling for performance |
| T-104 | Create message bubble components | UI Agent | 🔥 Critical | ☐ | T-103 | 4h | User vs assistant styling |
| T-105 | Add markdown renderer | UI Agent | ✅ High | ☐ | T-104 | 3h | react-markdown with plugins |
| T-106 | Implement code highlighting | UI Agent | ✅ High | ☐ | T-105 | 2h | Prism.js integration |
| T-107 | Build WebSocket client | UI Agent | 🔥 Critical | ☐ | T-021 | 5h | Native WebSocket API wrapper |
| T-108 | Add reconnection logic | UI Agent | 🔥 Critical | ☐ | T-107 | 4h | Auto-reconnect with backoff |
| T-109 | Implement typing indicators | UI Agent | ✅ High | ☐ | T-107 | 2h | Animated dots during streaming |
| T-110 | Create message input field | UI Agent | 🔥 Critical | ☐ | T-103 | 3h | Multiline textarea with send |
| T-111 | Add file attachment UI | UI Agent | ✅ High | ☐ | T-110 | 4h | Drag-and-drop support |
| T-112 | Build conversation history | UI Agent | ✅ High | ☐ | T-103 | 5h | IndexedDB persistence |
| T-113 | Add history search | UI Agent | ✅ Medium | ☐ | T-112 | 3h | Full-text search with filters |
| T-114 | Create settings panel | UI Agent | 🔥 Critical | ☐ | T-100 | 6h | Tabbed interface with forms |
| T-115 | Build admin dashboard layout | UI Agent | 🔥 Critical | ☐ | T-100 | 5h | Grid layout with navigation |
| T-116 | Add system health cards | UI Agent | 🔥 Critical | ☐ | T-115 | 6h | GPU, RAM, disk metrics |
| T-117 | Create connected clients table | UI Agent | ✅ High | ☐ | T-115 | 4h | Real-time status updates |
| T-118 | Build query logs view | UI Agent | ✅ High | ☐ | T-115 | 5h | Filterable table with search |
| T-119 | Add document stats display | UI Agent | ✅ High | ☐ | T-115 | 3h | Index size, counts, growth |
| T-120 | Create model management UI | UI Agent | ✅ High | ☐ | T-115 | 5h | Load/unload/configure models |
| T-121 | Add real-time metric charts | UI Agent | ✅ Medium | ☐ | T-116 | 6h | Recharts line/area charts |
| T-122 | Implement error notifications | UI Agent | ✅ High | ☐ | T-107 | 2h | Toast/snackbar system |
| T-123 | Add loading states and skeletons | UI Agent | ✅ Medium | ☐ | T-103 | 3h | Skeleton screens for async |
| T-124 | Implement dark/light mode | UI Agent | ✅ Medium | ☐ | T-102 | 3h | Theme toggle with persistence |
| T-125 | Add keyboard shortcuts | UI Agent | ✅ Medium | ☐ | T-100 | 3h | Global hotkeys for navigation |
| T-126 | Create onboarding flow | UI Agent | ✅ Medium | ☐ | T-100 | 4h | First-time user tutorial |
| T-127 | Test on Surface Pro touch | QA Agent | 🔥 Critical | ☐ | T-100-T-126 | 4h | Touch input validation |
| T-128 | Optimize bundle size | UI Agent | ✅ Medium | ☐ | T-100 | 3h | Code splitting, tree shaking |
| T-129 | Add accessibility features | UI Agent | ✅ High | ☐ | T-100 | 5h | ARIA labels, keyboard nav |
| T-130 | Write UI component docs | PM Agent | ✅ Medium | ☐ | T-101-T-126 | 4h | Storybook or similar |

**Phase 5 Total: 119 hours**

---

## Phase 6: Testing, Optimization & Release

| ID | Task | Owner | Priority | Status | Dependencies | Est. Hours | Notes |
|----|------|-------|----------|--------|--------------|------------|-------|
| T-140 | Write backend unit tests | QA Agent | 🔥 Critical | ☐ | All phases | 15h | Pytest suite for all modules |
| T-141 | Achieve 90% test coverage | QA Agent | 🔥 Critical | ☐ | T-140 | 10h | Fill coverage gaps |
| T-142 | Create API integration tests | QA Agent | 🔥 Critical | ☐ | T-140 | 12h | Test all endpoints |
| T-143 | Build E2E test suite | QA Agent | 🔥 Critical | ☐ | T-142 | 15h | Playwright automated tests |
| T-144 | Write load test scenarios | QA Agent | ✅ High | ☐ | T-143 | 6h | k6 comprehensive scripts |
| T-145 | Execute load testing | QA Agent | ✅ High | ☐ | T-144 | 4h | 100 queries/hour sustained |
| T-146 | Run security scan | QA Agent | 🔥 Critical | ☐ | All phases | 4h | Bandit, npm audit, Trivy |
| T-147 | Profile backend performance | Backend Agent | ✅ High | ☐ | T-140 | 6h | cProfile + memory_profiler |
| T-148 | Optimize slow queries | Backend Agent | ✅ High | ☐ | T-147 | 8h | Target identified bottlenecks |
| T-149 | Reduce Docker image sizes | Infra Agent | ✅ Medium | ☐ | T-002 | 4h | Multi-stage builds |
| T-150 | Tune ChromaDB parameters | Data Agent | ✅ High | ☐ | T-048 | 4h | Performance optimization |
| T-151 | Optimize embedding batching | Data Agent | ✅ High | ☐ | T-047 | 3h | Batch size tuning |
| T-152 | Write installation guide | PM Agent | 🔥 Critical | ☐ | All phases | 8h | Comprehensive step-by-step |
| T-153 | Create configuration reference | PM Agent | 🔥 Critical | ☐ | T-152 | 6h | Document all variables |
| T-154 | Document API endpoints | PM Agent | 🔥 Critical | ☐ | T-153 | 8h | OpenAPI specification |
| T-155 | Build troubleshooting guide | PM Agent | ✅ High | ☐ | T-152 | 6h | Common issues and solutions |
| T-156 | Record setup video | PM Agent | ✅ Medium | ☐ | T-152 | 4h | Screencast walkthrough |
| T-157 | Create quick start guide | PM Agent | ✅ High | ☐ | T-152 | 3h | 5-minute setup doc |
| T-158 | Finalize version numbers | PM Agent | 🔥 Critical | ☐ | All phases | 1h | Semver v0.1.0-alpha |
| T-159 | Update CHANGELOG.md | PM Agent | 🔥 Critical | ☐ | T-158 | 3h | User-facing changes |
| T-160 | Write release notes | PM Agent | 🔥 Critical | ☐ | T-159 | 4h | Highlight features |
| T-161 | Create distribution packages | Infra Agent | 🔥 Critical | ☐ | All phases | 8h | Docker images, installers |
| T-162 | Build multi-arch Docker images | Infra Agent | ✅ Medium | ☐ | T-161 | 4h | AMD64, ARM64 if possible |
| T-163 | Tag Git release | PM Agent | 🔥 Critical | ☐ | T-161 | 1h | v0.1.0-alpha signed tag |
| T-164 | Deploy to test environment | Infra Agent | 🔥 Critical | ☐ | T-161 | 3h | Staging validation |
| T-165 | Conduct user acceptance testing | QA Agent | 🔥 Critical | ☐ | T-164 | 12h | 3+ test users |
| T-166 | Execute 72-hour stability test | QA Agent | 🔥 Critical | ☐ | T-164 | 8h | Continuous monitoring |
| T-167 | Test rollback procedures | Infra Agent | ✅ High | ☐ | T-164 | 3h | Disaster recovery |
| T-168 | Fix critical bugs from UAT | All Agents | 🔥 Critical | ☐ | T-165 | 15h | Priority bug fixes |
| T-169 | Deploy to production | Infra Agent | 🔥 Critical | ☐ | T-166, T-168 | 2h | Final release |
| T-170 | Post-deployment verification | QA Agent | 🔥 Critical | ☐ | T-169 | 2h | Smoke tests |

**Phase 6 Total: 193 hours**

---

## Backlog (Post-MVP Features)

| ID | Task | Priority | Estimated Effort | Target Phase |
|----|------|----------|------------------|--------------|
| T-200 | Native Android app development | ✅ High | 80h | Phase 7 |
| T-201 | iOS app feasibility study | ✅ Medium | 16h | Phase 7 |
| T-202 | Whisper voice input integration | ✅ High | 24h | Phase 7 |
| T-203 | Piper TTS voice output | ✅ High | 20h | Phase 7 |
| T-204 | Multi-user access control | ✅ High | 40h | Phase 8 |
| T-205 | Role-based permissions | ✅ High | 32h | Phase 8 |
| T-206 | Fine-tuning interface | ✅ Medium | 48h | Phase 8 |
| T-207 | Plugin system architecture | ✅ Medium | 40h | Phase 8 |
| T-208 | Multi-node clustering | ✅ Low | 60h | Phase 9 |
| T-209 | Load balancing across GPUs | ✅ Low | 32h | Phase 9 |
| T-210 | Distributed vector DB | ✅ Low | 56h | Phase 9 |
| T-211 | Advanced analytics dashboard | ✅ Medium | 28h | Phase 8 |
| T-212 | Conversation branching | ✅ Low | 24h | Future |
| T-213 | Custom prompt templates | ✅ Medium | 16h | Phase 8 |
| T-214 | Backup/restore UI | ✅ High | 12h | Phase 7 |
| T-215 | Model quantization tools | ✅ Medium | 20h | Phase 8 |

---

## Task Management Guidelines

### Task Status Definitions
- **☐ Not Started:** Task not yet begun, waiting for dependencies or assignment
- **🔄 In Progress:** Actively being worked on by assigned agent
- **✅ Complete:** Finished, tested, and merged to main branch
- **🚫 Blocked:** Cannot proceed due to blocker (document in notes)
- **⏸️ Paused:** Temporarily suspended, will resume later

### Priority Levels
- **🔥 Critical:** Blocks other work or core MVP functionality
- **✅ High:** Important for MVP, complete within phase
- **✅ Medium:** Nice-to-have for MVP, can defer if needed
- **✅ Low:** Future enhancement, backlog item

### Time Tracking
- **Estimated Hours:** Initial estimate before work begins
- **Actual Hours:** Log actual time spent (update in notes)
- **Variance:** Track for future estimation accuracy

### Update Frequency
- **Daily:** Update status for tasks in progress
- **Weekly:** Review blockers and adjust assignments
- **End of Phase:** Comprehensive review and planning

### Dependencies
- Must complete dependency tasks before starting
- Flag dependency blockers immediately
- Consider parallel work where possible

### Notes Field Usage
- Document important decisions
- Track actual hours spent
- Note blockers and resolutions
- Link related PRs or issues

---

## Sprint Planning

### Sprint Duration
- 2-week sprints aligned with phase boundaries
- Sprint 1-2: Phase 1-2
- Sprint 3-4: Phase 3-4
- Sprint 5-6: Phase 5-6

### Sprint Ceremonies
- **Planning:** Monday, Week 1 (2h)
- **Daily Standups:** Async via TASKS.md updates
- **Review:** Friday, Week 2 (1h)
- **Retrospective:** Friday, Week 2 (1h)

### Capacity Planning
- Backend Agent: 20h/sprint
- UI Agent: 16h/sprint
- Infra Agent: 8h/sprint
- Data Agent: 12h/sprint
- Agent Ops: 16h/sprint
- QA Agent: 12h/sprint
- PM Agent: 10h/sprint

**Total Capacity:** ~94 hours per 2-week sprint

---

## Progress Tracking

| Phase | Total Tasks | Completed | In Progress | Blocked | Completion % |
|-------|-------------|-----------|-------------|---------|--------------|
| Phase 1 | 15 | 0 | 0 | 0 | 0% |
| Phase 2 | 16 | 0 | 0 | 0 | 0% |
| Phase 3 | 21 | 0 | 0 | 0 | 0% |
| Phase 4 | 23 | 0 | 0 | 0 | 0% |
| Phase 5 | 31 | 0 | 0 | 0 | 0% |
| Phase 6 | 31 | 0 | 0 | 0 | 0% |
| **Total** | **137** | **0** | **0** | **0** | **0%** |

**Total Estimated Effort:** 561 hours  
**Expected Timeline:** 12 weeks at ~47 hours/week

---

This task list is a living document and will be updated throughout the project lifecycle. All agents should reference and update this file regularly.
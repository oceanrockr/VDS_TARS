# AGENT DIRECTORY — T.A.R.S. LocalLLM Desktop

**Project:** T.A.R.S. (Temporal Augmented Retrieval System)  
**Development Methodology:** Multi-agent distributed development  
**Agent Count:** 7 specialized roles

## Overview

T.A.R.S. follows a multi-agent development methodology where specialized agents own distinct domains and operate autonomously within their scope. This structure enables parallel development, clear ownership, and efficient collaboration through well-defined interfaces and communication protocols.

---

## Agent Roster

### 1. PM Agent (Project Manager)

**Core Responsibilities:**
- Sprint planning, timeline management, and milestone tracking
- Stakeholder communication and project status reporting
- PRD maintenance and requirements documentation
- Risk identification, assessment, and mitigation planning
- Resource allocation and workload balancing across agents
- Cross-agent coordination and conflict resolution
- Budget and timeline management
- Decision documentation and CLAUDE LOG maintenance

**Key Deliverables:**
- Sprint plans with task assignments and timelines
- PLANNING.md updates reflecting current roadmap
- Project status reports (weekly, monthly, quarterly)
- Risk registers with mitigation strategies
- Timeline adjustments based on blockers
- Meeting agendas, notes, and action items
- CLAUDE LOG entries for major decisions
- Stakeholder communications

**Tools & Systems:**
- Project tracker (Notion, Linear, or GitHub Projects)
- TASKS.md for work breakdown and tracking
- CLAUDE LOG for decision documentation
- PLANNING.md for roadmap management
- Miro or FigJam for planning sessions
- Google Workspace for documentation

**Communication Protocols:**
- **Daily:** Review TASKS.md for blockers and progress updates
- **Weekly:** Sprint planning and team sync meetings
- **Bi-weekly:** Sprint reviews and retrospectives
- **Monthly:** Roadmap updates and stakeholder reports
- **Ad-hoc:** Immediate escalation for critical blockers

**Success Metrics:**
- On-time delivery rate for sprints (target: >80%)
- Blocker resolution speed (target: <24 hours)
- Stakeholder satisfaction (target: 4/5 or higher)
- Documentation completeness (target: 100%)
- Resource utilization efficiency (target: 80-90%)

---

### 2. Infra Agent (Infrastructure Engineer)

**Core Responsibilities:**
- Docker containerization and orchestration
- Network architecture design and configuration
- GPU passthrough setup and optimization
- SSL/TLS certificate generation and management
- System monitoring, alerting, and observability
- Deployment automation and CI/CD pipeline management
- Disaster recovery planning and backup systems
- Performance tuning and capacity planning

**Key Deliverables:**
- `docker-compose.yml` and related container configurations
- Network architecture diagrams and documentation
- Deployment scripts and automation playbooks
- Monitoring dashboards (Grafana, custom UI)
- Backup and restore procedures with testing
- CI/CD pipeline configuration (GitHub Actions)
- Infrastructure documentation and runbooks
- Security hardening and compliance reports

**Tools & Systems:**
- Docker Desktop with GPU support
- Docker Compose for multi-container orchestration
- Nginx or Caddy for reverse proxy
- Prometheus + Grafana for monitoring (optional)
- GitHub Actions for CI/CD automation
- Synology DSM CLI for NAS management
- Tailscale for optional VPN access
- k6 for load testing and performance validation

**Communication Protocols:**
- **Infrastructure Change Proposals:** Document and review before implementation
- **Incident Response:** Immediate notification and coordination
- **Capacity Planning:** Monthly reviews with PM and Backend agents
- **Security Alerts:** Immediate escalation for vulnerabilities
- **Deployment Windows:** Scheduled with advance notice

**Success Metrics:**
- System uptime (target: >99% over 30 days)
- Deployment success rate (target: >95%)
- Mean time to recovery (MTTR) (target: <30 minutes)
- Security scan pass rate (target: zero critical findings)
- Docker image build time (target: <10 minutes)

---

### 3. Backend Agent (FastAPI Developer)

**Core Responsibilities:**
- FastAPI application architecture and development
- WebSocket implementation for real-time streaming
- LLM integration with Ollama Python SDK
- Authentication and authorization (JWT)
- API endpoint design, implementation, and documentation
- Message routing, queuing, and request handling
- Error handling, logging, and observability
- Performance optimization and code quality

**Key Deliverables:**
- FastAPI backend application with RESTful and WebSocket APIs
- WebSocket server with connection management
- API documentation (OpenAPI/Swagger)
- Authentication middleware and JWT handling
- Integration with Ollama for model inference
- Integration with ChromaDB for vector search
- Comprehensive unit and integration tests
- Performance benchmarks and optimization reports

**Tools & Systems:**
- Python 3.11+ with FastAPI framework
- Ollama Python SDK for LLM communication
- Uvicorn or Hypercorn ASGI server
- Pydantic for data validation and serialization
- Redis for caching and message queuing (optional)
- Pytest for testing with pytest-asyncio
- Black, isort, mypy for code quality
- cProfile and memory_profiler for performance analysis

**Communication Protocols:**
- **API Contract Reviews:** Collaborate with UI Agent on endpoint specs
- **Performance Metrics:** Share latency and throughput data with Infra Agent
- **Data Schema Coordination:** Align with Data Agent on vector metadata
- **Integration Testing:** Partner with QA Agent on test coverage
- **Code Reviews:** Peer review with other development agents

**Success Metrics:**
- API response time (target: <100ms for standard requests)
- WebSocket token latency (target: <50ms between tokens)
- Test coverage (target: >90%)
- Error rate (target: <1% of requests)
- Code quality score (target: >8/10 via linters)

---

### 4. Data Agent (Data Operations)

**Core Responsibilities:**
- Document ingestion pipeline development and optimization
- ChromaDB configuration, tuning, and maintenance
- Embedding generation and vector management
- Vector storage, indexing, and retrieval optimization
- RAG pipeline implementation and enhancement
- Data quality monitoring and validation
- NAS integration and document synchronization
- Backup automation and data integrity verification

**Key Deliverables:**
- Document parsers for multiple formats (PDF, DOCX, TXT, MD, CSV)
- Embedding generation pipeline with batch processing
- ChromaDB setup, configuration, and optimization
- Vector backup and restore scripts
- Data quality monitoring dashboards
- Retrieval performance optimization reports
- Document metadata indexing system
- RAG pipeline integration documentation

**Tools & Systems:**
- LangChain for RAG orchestration and document loading
- ChromaDB for vector storage and similarity search
- sentence-transformers for local embedding generation
- PyPDF2, pdfplumber, python-docx for document parsing
- Pandas for structured data processing
- Synology Drive API for NAS integration
- watchdog library for file system monitoring
- NumPy for vector operations

**Communication Protocols:**
- **Data Schema Definitions:** Collaborate with Backend Agent
- **Storage Requirements:** Coordinate with Infra Agent on capacity
- **Retrieval Quality:** Feedback loops with QA Agent on accuracy
- **Model Selection:** Consult on embedding model choices
- **Performance Tuning:** Share metrics on indexing and retrieval speed

**Success Metrics:**
- Document indexing speed (target: >100 docs/minute)
- Retrieval accuracy/relevance (target: >0.7 score)
- Embedding generation latency (target: <1s per document)
- Storage efficiency (target: <10MB per 1000 documents)
- Data integrity (target: zero corruption or loss)

---

### 5. UI Agent (Frontend Developer)

**Core Responsibilities:**
- Electron desktop application development
- React component library creation and maintenance
- Tailwind CSS styling, theming, and design system
- WebSocket client implementation and state management
- Admin dashboard development and data visualization
- User experience optimization and accessibility
- Responsive design for multiple device types
- Performance optimization and bundle management

**Key Deliverables:**
- Electron chat application for Windows/Linux
- React component library with reusable UI elements
- Admin dashboard with real-time monitoring
- User settings and configuration panels
- WebSocket client with reconnection handling
- UI/UX documentation and design guidelines
- Accessibility compliance (WCAG 2.1 AA)
- Performance optimization reports

**Tools & Systems:**
- React 18 with TypeScript
- Vite 5 for fast build and HMR
- Electron 28 for desktop shell
- Tailwind CSS 3 + ShadCN/UI components
- Zustand for lightweight state management
- React Query (TanStack Query) for data fetching
- Vitest for component testing
- Playwright for E2E testing
- Framer Motion for animations
- Recharts for data visualization

**Communication Protocols:**
- **Design Reviews:** Weekly reviews with PM Agent
- **API Contract Validation:** Align with Backend Agent on endpoints
- **User Feedback Sessions:** Conduct and incorporate feedback
- **Component Documentation:** Maintain Storybook or similar
- **Accessibility Testing:** Collaborate with QA Agent

**Success Metrics:**
- App startup time (target: <3 seconds)
- UI render time for messages (target: <50ms)
- Bundle size (target: <500KB gzipped)
- User satisfaction scores (target: >4/5)
- Accessibility compliance (target: zero critical violations)

---

### 6. Agent Ops (Client Agent Developer)

**Core Responsibilities:**
- Client agent application development (Python/Node.js)
- File system monitoring and event handling
- Local embedding generation pipeline
- Network synchronization protocols and implementation
- Client-server communication and authentication
- Installation package creation and distribution
- Cross-platform compatibility (Windows, Linux, Android)
- Client troubleshooting and support documentation

**Key Deliverables:**
- Python client agent with watchdog integration
- Node.js client agent (alternative implementation)
- File watcher service with configurable paths
- Local embedding generation system
- Client registration and authentication flow
- Sync protocol with retry and error handling
- Windows installer (NSIS/WiX) and portable executable
- Cross-platform testing and validation
- Client agent documentation and user guides

**Tools & Systems:**
- Python 3.11+ with asyncio
- Node.js 20+ with TypeScript
- watchdog library (Python) or chokidar (Node.js)
- sentence-transformers for local embeddings
- PyInstaller or pkg for executable packaging
- NSIS or WiX Toolset for Windows installers
- WebSocket or HTTP/2 for communication
- SQLite for local data persistence

**Communication Protocols:**
- **Protocol Design:** Collaborate with Backend Agent on sync specs
- **Deployment Coordination:** Work with Infra Agent on rollout
- **User Testing:** Partner with QA Agent for device validation
- **Resource Optimization:** Share performance metrics with team
- **Installation Support:** Document common issues and solutions

**Success Metrics:**
- Client resource usage (target: <200MB RAM, <5% CPU idle)
- Sync latency (target: <30 seconds for new documents)
- Installation success rate (target: >95%)
- Cross-platform compatibility (target: 100% on supported platforms)
- Uptime as background service (target: >99%)

---

### 7. QA Agent (Quality Assurance Engineer)

**Core Responsibilities:**
- Test strategy development and planning
- Unit test development and maintenance
- Integration test creation and execution
- End-to-end test automation
- Performance testing and benchmarking
- Security testing and vulnerability scanning
- Bug triage, tracking, and verification
- Quality metrics reporting and analysis

**Key Deliverables:**
- Comprehensive test suites (unit, integration, E2E)
- Test coverage reports and gap analysis
- Performance benchmarks and load test results
- Bug reports with reproduction steps
- Security scan reports and remediation plans
- QA documentation and testing guides
- Automated test pipelines in CI/CD
- Quality metrics dashboards

**Tools & Systems:**
- Pytest for Python backend testing
- pytest-cov for coverage reporting
- Vitest for JavaScript/TypeScript testing
- Playwright for E2E browser automation
- k6 for load and performance testing
- Bandit for Python security scanning
- npm audit for JavaScript vulnerability checks
- Trivy for Docker image scanning
- Coverage.py for detailed code coverage
- GitHub Actions for CI integration

**Communication Protocols:**
- **Daily Bug Triage:** Review and prioritize new issues
- **Weekly Quality Reports:** Share metrics with PM Agent
- **Performance Feedback:** Provide data to Backend and Infra agents
- **Security Findings:** Immediate escalation to Infra Agent
- **Test Coverage Reviews:** Collaborate with development agents

**Success Metrics:**
- Test coverage (target: >90% backend, >85% frontend)
- Bug escape rate (target: <5% to production)
- Test execution time (target: <15 minutes for full suite)
- Critical bug resolution time (target: <24 hours)
- Security scan pass rate (target: zero critical findings)

---

## Agent Interaction Matrix

| From/To | PM | Infra | Backend | Data | UI | Agent Ops | QA |
|---------|-----|-------|---------|------|----|-----------|----|
| **PM** | - | Planning, Status | Planning, Status | Planning, Status | Planning, Status | Planning, Status | Planning, Status |
| **Infra** | Reports | - | Deploy, Monitor | Storage | Deploy | Deploy | Security |
| **Backend** | Status | Deploy | - | API, Schema | API Contracts | Protocol | Tests |
| **Data** | Status | Storage | API, Schema | - | Data Models | Sync | Tests |
| **UI** | Status | Deploy | API Contracts | Data Models | - | Install | Tests |
| **Agent Ops** | Status | Deploy | Protocol | Sync | Install | - | Tests |
| **QA** | Metrics | Security, Perf | Tests | Tests | Tests | Tests | - |

---

## Collaboration Protocols

### 1. Interface Contracts
When agents must integrate their work:
- **Define Explicit Contracts:** API schemas, data formats, message protocols
- **Document in Shared Repository:** Store in `/contracts` or `/schemas` folder
- **Version All Contracts:** Use semantic versioning for interface changes
- **Breaking Changes:** Require cross-agent approval and major version bump
- **Validation:** Automated contract testing in CI pipeline

### 2. Asynchronous Communication (Default)
Primary communication mode for non-urgent matters:
- **GitHub Issues/PRs:** Technical discussions and code reviews
- **TASKS.md Comments:** Progress updates and blockers
- **CLAUDE LOG:** Major decisions and architectural changes
- **Tagged Notifications:** Use @mentions for specific agent attention
- **Response SLA:** 24 hours for non-urgent, 4 hours for blockers

### 3. Synchronous Meetings (When Needed)
Schedule real-time coordination when:
- **Complex Integration:** Multiple agents need aligned understanding
- **Blocking Issues:** Immediate resolution required
- **Design Reviews:** Architectural decisions need discussion
- **Sprint Ceremonies:** Planning, reviews, retrospectives

**Meeting Best Practices:**
- Schedule with 24-hour notice minimum
- Share agenda in advance
- Limit to 60 minutes maximum
- Record decisions and action items
- Follow up with written summary in CLAUDE LOG

### 4. Escalation Path
```
Issue/Blocker → Owning Agent (24h resolution attempt) →
Related Agents (48h collaborative resolution) →
PM Agent (prioritization and resource allocation) →
Stakeholder (for scope or timeline changes)
```

**Escalation Triggers:**
- Blocker unresolved after 24 hours
- Cross-agent dependencies causing delays
- Technical decisions impacting multiple domains
- Security or privacy concerns
- Resource constraints

### 5. Knowledge Sharing
Continuous learning and cross-pollination:
- **Weekly Demo Sessions:** Show completed work (30 min)
- **Shared Documentation:** Maintain `/docs` with up-to-date info
- **Code Reviews:** Cross-agent reviews for learning
- **Pair Programming:** Scheduled sessions for complex integrations
- **Retrospectives:** Share lessons learned every 2 weeks

---

## Agent Autonomy Guidelines

### Decision Authority by Domain

Each agent has full authority within their primary domain:

| Agent | Decision Authority |
|-------|-------------------|
| **PM Agent** | Timeline adjustments, resource allocation, priority changes |
| **Infra Agent** | Infrastructure choices, deployment methods, monitoring tools |
| **Backend Agent** | API design, implementation details, library choices |
| **Data Agent** | Storage schemas, embedding strategies, indexing approaches |
| **UI Agent** | Component structure, styling decisions, UX patterns |
| **Agent Ops** | Client architecture, sync protocols, packaging methods |
| **QA Agent** | Testing strategies, quality gates, bug priority |

### Escalation Triggers

Agents **must escalate** when decisions:
- Impact other agent domains or dependencies
- Have budget or timeline implications
- Require cross-functional input or expertise
- Involve security, privacy, or compliance
- Contradict requirements or project goals
- Need architectural review

### Documentation Requirements

All agents must:
- **ADRs:** Document architectural decisions with rationale
- **README Files:** Maintain up-to-date setup and usage docs
- **Code Comments:** Explain complex logic and edge cases
- **Project Docs:** Update PRD, PLANNING, TASKS as changes occur
- **Changelog:** Record all user-facing and API changes

---

## Agent Onboarding Process

### New Agent Checklist

**Technical Setup (Week 1):**
- [ ] Repository access granted (GitHub, GitLab, etc.)
- [ ] Development environment configured and tested
- [ ] Required tools installed (Docker, IDEs, CLI tools)
- [ ] Sample data and test accounts available
- [ ] Access to monitoring and logging systems

**Knowledge Transfer (Week 1-2):**
- [ ] Project overview presentation completed
- [ ] Review PRD, PLANNING, RULES, TASKS documents
- [ ] Understand T.A.R.S. architecture and design decisions
- [ ] Shadow existing agent for 1 week (pair programming)
- [ ] Complete first small task successfully

**Communication Setup (Week 1):**
- [ ] Added to team communication channels (Slack, Discord, etc.)
- [ ] Introduced to other agents in team meeting
- [ ] Calendar access for meetings and ceremonies
- [ ] Notification preferences configured

**Quality Standards (Week 1-2):**
- [ ] Code style guide reviewed and understood
- [ ] Testing requirements and coverage goals clear
- [ ] Security practices training completed
- [ ] Documentation standards learned and practiced

---

## Agent Performance Reviews

### Quarterly Review Areas

**Delivery:**
- Task completion rate vs. committed work
- On-time delivery percentage
- Quality of deliverables (bug rate, test coverage)

**Quality:**
- Code review feedback and resolution
- Test coverage and quality
- Documentation completeness

**Collaboration:**
- Responsiveness to requests and blockers
- Cross-agent communication effectiveness
- Constructive participation in reviews and planning

**Growth:**
- Technical skill development
- Learning new tools or techniques
- Sharing knowledge with team

**Innovation:**
- Problem-solving creativity
- Process improvements suggested/implemented
- Proactive issue identification

### Metrics by Agent Role

| Agent | Key Performance Indicators |
|-------|---------------------------|
| **PM Agent** | On-time delivery rate, blocker resolution speed, stakeholder satisfaction |
| **Infra Agent** | System uptime, deployment success rate, MTTR |
| **Backend Agent** | API latency, test coverage, code quality score |
| **Data Agent** | Indexing speed, retrieval accuracy, storage efficiency |
| **UI Agent** | User satisfaction, performance metrics, accessibility compliance |
| **Agent Ops** | Client stability, resource usage, installation success rate |
| **QA Agent** | Bug detection rate, test coverage, security findings |

---

## Cross-Functional Initiatives

Some projects require multi-agent collaboration:

### Security Audit
- **Lead:** Infra Agent
- **Support:** Backend, Agent Ops, QA Agents
- **Deliverable:** Comprehensive security assessment
- **Timeline:** 1 sprint (2 weeks)

### Performance Optimization
- **Lead:** Backend Agent
- **Support:** Data, Infra, QA Agents
- **Deliverable:** Performance improvement plan and implementation
- **Timeline:** 1-2 sprints

### User Onboarding Experience
- **Lead:** UI Agent
- **Support:** Agent Ops, PM, QA Agents
- **Deliverable:** Polished first-time user experience
- **Timeline:** 1 sprint

---

## Tool Access Matrix

| Tool/System | PM | Infra | Backend | Data | UI | Agent Ops | QA |
|-------------|-----|-------|---------|------|----|-----------|----|
| GitHub/GitLab | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Docker Host | Admin | ✅ | ✅ | ✅ | Read | Read | ✅ |
| Synology NAS | Read | ✅ | Read | ✅ | - | Read | Read |
| CI/CD Pipeline | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Monitoring | ✅ | ✅ | ✅ | ✅ | Read | Read | ✅ |
| Project Tracker | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Production Deploy | - | ✅ | - | - | - | - | Read |

**Legend:**
- ✅ Full Access
- Read: Read-only access
- Admin: Administrative privileges
- `-` No access needed

---

## Communication Channels

### Synchronous
- **Team Meetings:** Weekly sprint planning and reviews
- **Stand-ups:** Async via TASKS.md updates (optional sync call)
- **Pair Programming:** Scheduled sessions for complex work
- **Emergency Calls:** Critical incident response

### Asynchronous
- **GitHub:** Issues, PRs, code reviews, discussions
- **Project Tracker:** Task updates, blockers, progress
- **CLAUDE LOG:** Architectural decisions and rationale
- **Email/Slack:** General updates and announcements
- **Documentation:** Shared knowledge base

---

This agent directory is a living document and will evolve as the project grows and team structure changes. Regular reviews ensure roles and responsibilities remain clear and effective.
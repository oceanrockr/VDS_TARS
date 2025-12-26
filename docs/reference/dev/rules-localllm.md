# PROJECT RULES — T.A.R.S. LocalLLM Desktop

## Core Principles

### 1. Local-Only Architecture
**Rule:** All computation, storage, and network traffic must remain within the local network.

**Rationale:** Privacy and data sovereignty are fundamental to the T.A.R.S. mission. Users choose this system specifically to avoid cloud dependencies and maintain complete control over their data.

**Implementation:**
- No external API calls permitted in production code
- All dependencies must be vendored, self-hosted, or locally cached
- Telemetry and analytics disabled by default
- No crash reporting to external services
- **Exception:** Optional Tailscale VPN for secure remote access (user opt-in only)

**Enforcement:**
- Static analysis checks for external network calls in CI pipeline
- Code review must explicitly verify no cloud service dependencies
- Integration tests run in network-isolated Docker environment
- Pre-commit hooks scan for common external API endpoints

**Violation Response:** Immediate code rejection, security review, architectural discussion

---

### 2. Modular Architecture
**Rule:** All components must be loosely coupled with well-defined, versioned interfaces.

**Rationale:** Enable independent development, testing, and replacement of system components. Support future scaling and extensibility without major refactoring.

**Implementation:**
- Backend exposes REST/WebSocket API contracts documented in OpenAPI/AsyncAPI
- Client agents communicate only via documented endpoints
- UI components use standardized TypeScript interfaces
- Database access abstracted behind repository pattern
- Message formats defined in shared schema files

**Enforcement:**
- API contracts must be documented before implementation
- Breaking changes require major version bump (semver)
- Interface changes reviewed by Architecture Lead
- Integration tests validate contract compliance

**Violation Response:** Refactoring required before merge, design review meeting

---

### 3. Security-First Development
**Rule:** Every network connection must be authenticated and encrypted by default.

**Rationale:** Protect sensitive personal and business data from unauthorized access, even within the trusted LAN environment. Defense in depth principle applies.

**Implementation:**
- HTTPS/WSS mandatory for all client-server communication
- JWT tokens with expiration and refresh mechanism
- No credentials stored in plaintext, environment variables, or logs
- Security headers on all HTTP responses (CSP, HSTS, X-Frame-Options)
- Regular dependency vulnerability scanning
- Input validation and sanitization at all boundaries
- Rate limiting on authentication endpoints

**Enforcement:**
- Pre-commit hooks scan for hardcoded secrets and credentials
- Security review required for all authentication/authorization changes
- Automated security testing (Bandit, npm audit) in CI pipeline
- Penetration testing before major releases

**Violation Response:** Critical security finding triggers immediate fix, post-mortem required

---

### 4. Deterministic Builds
**Rule:** All Docker images and dependencies must be pinned to specific versions or SHA256 digests.

**Rationale:** Ensure reproducible builds across environments and prevent supply chain attacks. Enable rollback to known-good states.

**Implementation:**
- All `FROM` statements in Dockerfiles use SHA256 digests
- Python dependencies locked with `requirements.txt.lock` and hash verification
- npm dependencies locked with `package-lock.json` and integrity checks
- Build artifacts checksummed in release manifests
- Multi-stage builds for minimal attack surface
- Base images regularly updated and re-pinned

**Enforcement:**
- CI fails on unpinned dependencies or loose version specifications
- Monthly security updates with new pinned versions
- Build reproducibility verified in clean test environment
- Dockerfile linting with hadolint

**Violation Response:** Build fails, must pin versions before merge

---

### 5. Semantic Versioning
**Rule:** Follow semver (vMAJOR.MINOR.PATCH) strictly with pre-release tags for non-production code.

**Rationale:** Clear communication of compatibility, change impact, and stability to users and developers.

**Version Format:**
```
vMAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
v1.0.0          - Production release
v0.1.0-alpha    - Early development
v0.5.0-beta     - Feature complete, testing
v1.2.3-rc.1     - Release candidate
v2.0.0          - Breaking changes
```

**Increment Rules:**
- **MAJOR:** Breaking API changes, architecture changes, data format changes
- **MINOR:** New features, non-breaking additions, significant improvements
- **PATCH:** Bug fixes, performance improvements, documentation updates
- **Pre-release:** `-alpha`, `-beta`, `-rc.N` for development versions

**Enforcement:**
- Automated version validation in CI/CD pipeline
- CHANGELOG.md must be updated for every release
- Version numbers embedded in all artifact filenames
- Git tags required for all releases

**Violation Response:** Release blocked until version corrected

---

### 6. Change Management Process
**Rule:** All changes follow documented workflow from planning to deployment.

**Standard Workflow:**
```
Requirement → PLANNING.md Design → TASKS.md Assignment → 
Branch Creation → Development → Self-Review → PR Creation → 
Code Review → CI Checks → Approval → Merge → Version Bump → 
Documentation Update → Deployment → Verification
```

**Branch Strategy:**
- `main` - production-ready code only
- `develop` - integration branch for features
- `feature/TASK-ID-description` - feature development
- `fix/TASK-ID-description` - bug fixes
- `docs/TASK-ID-description` - documentation only

**Pull Request Requirements:**
- Linked to task in TASKS.md
- Description explains what and why
- All tests passing (unit, integration, E2E)
- Code coverage maintained or improved
- Documentation updated
- Self-review checklist completed
- At least 1 reviewer approval

**Enforcement:**
- Branch protection rules on `main` and `develop`
- PR template enforces checklist completion
- Merge blocked if any CI check fails
- Automated deployment only from `main` branch

**Violation Response:** PR rejected, must follow process

---

### 7. Testing Standards
**Rule:** Maintain comprehensive automated testing at multiple levels.

**Coverage Requirements:**
- **Unit Tests:** ≥ 90% code coverage for backend services
- **Integration Tests:** ≥ 85% API endpoint coverage
- **E2E Tests:** ≥ 85% critical user flow coverage
- **Performance Tests:** Latency benchmarks must not regress

**Testing Framework:**
- Backend: `pytest` with `pytest-cov`, `pytest-asyncio`
- Frontend: `vitest` with `@testing-library/react`
- E2E: Playwright for browser automation
- Load: k6 for performance and stress testing
- Security: Bandit (Python), npm audit (JavaScript)

**Test Quality Standards:**
- Tests must be deterministic (no flaky tests)
- Tests must be fast (unit < 1s, integration < 10s each)
- Tests must be isolated (no shared state)
- Tests must be readable (clear arrange-act-assert)

**Enforcement:**
- CI blocks merge if coverage drops below threshold
- Flaky tests must be fixed or disabled within 24 hours
- Performance regression fails build automatically
- Test results published on every PR

**Violation Response:** PR blocked until tests added/fixed

---

### 8. Documentation Requirements
**Rule:** Documentation is not optional—it's part of the definition of "done."

**Required Artifacts per Sprint:**
- **CLAUDE LOG:** Agent conversations, decisions, and rationale
- **CHANGELOG.md:** User-facing changes in each release
- **API Documentation:** OpenAPI/AsyncAPI specs auto-generated from code
- **User Guides:** Step-by-step instructions for new features
- **Architecture Diagrams:** System design updates using Mermaid or PlantUML
- **README Updates:** Reflect current state of project

**Documentation Standards:**
- Written in clear, accessible language (avoid unnecessary jargon)
- Examples provided for all public APIs and features
- Screenshots/GIFs for UI features
- Troubleshooting sections for common issues
- Updated simultaneously with code changes

**Enforcement:**
- PR template includes documentation checklist
- Documentation build failures block merge
- Quarterly documentation audit and gap analysis
- "Docs-or-it-didn't-happen" culture

**Violation Response:** PR marked incomplete until docs added

---

### 9. Privacy & Data Governance
**Rule:** User data must be protected, minimized, and deletable on demand.

**Data Protection:**
- All data at rest encrypted with AES-256
- Conversation history opt-in only with configurable retention
- Clear data deletion procedures available to users
- No analytics, telemetry, or usage tracking by default
- Audit logs for all data access and modifications
- PII redaction in all logs and error messages

**User Rights:**
- Right to export all personal data
- Right to delete all personal data (hard delete)
- Right to know what data is stored and how it's used
- Right to opt-out of non-essential features

**Implementation:**
- Encryption keys managed locally, not in code
- Automated data retention policy enforcement
- Data export functionality in UI
- One-click data deletion with confirmation

**Enforcement:**
- Encryption verified in integration tests
- Privacy impact assessment for all new features
- Annual privacy compliance review
- User consent flows for data collection

**Violation Response:** Feature disabled until privacy issues resolved

---

### 10. Agent Autonomy & Isolation
**Rule:** Development agents operate within defined boundaries with verified communication channels.

**Agent Isolation:**
- Each agent runs under separate service account with minimal permissions
- Inter-agent communication through documented API or message queue only
- Rate limiting on agent API calls to prevent resource exhaustion
- Comprehensive audit trail for all agent actions
- Sandbox environments for agent testing and experimentation

**Communication Protocols:**
- Agents use versioned API contracts
- Message authentication and validation required
- Timeout and retry logic for resilience
- Circuit breakers to prevent cascade failures

**Enforcement:**
- Agent permissions reviewed quarterly
- Anomaly detection on agent behavior patterns
- Incident response plan for compromised agents
- Regular security training for agent developers

**Violation Response:** Agent access revoked, security investigation

---

## Development Workflow Rules

### Git Workflow

**Branch Naming Convention:**
```
feature/TASK-ID-short-description
fix/TASK-ID-short-description
docs/TASK-ID-short-description
refactor/TASK-ID-short-description
test/TASK-ID-short-description
```

**Commit Message Format (Conventional Commits):**
```
<type>(<scope>): <subject>

<body>

<footer>

Types: feat, fix, docs, style, refactor, test, chore
Example: feat(websocket): add connection heartbeat mechanism
```

**Commit Requirements:**
- All commits must be GPG signed for authenticity
- Commit messages must be clear and descriptive
- One logical change per commit (atomic commits)
- No force push on shared branches (`main`, `develop`)

### Code Review Standards

**Review Timeline:**
- Initial review within 24 hours of PR creation
- Final approval within 48 hours for standard changes
- Urgent fixes reviewed within 4 hours

**Reviewer Responsibilities:**
- Verify functionality meets requirements
- Check test coverage and quality
- Validate documentation updates
- Identify security concerns
- Assess performance implications
- Ensure code style compliance

**Approval Requirements:**
- Minimum 1 maintainer approval required
- All conversations must be resolved
- No "LGTM" without actual review
- Constructive feedback encouraged

### CI/CD Pipeline

**Pipeline Stages:**
```
Trigger → Lint → Unit Tests → Build → Integration Tests → 
Security Scan → Performance Tests → Deploy (if main) → Verify
```

**Required Checks (all must pass):**
- Code linting (pylint, eslint)
- Type checking (mypy, TypeScript)
- Unit test execution and coverage
- Integration test execution
- Security vulnerability scan
- Docker image build
- Documentation build

**Artifacts:**
- Test reports and coverage
- Build logs
- Docker images
- Documentation site
- Retention: 90 days

**Deployment:**
- `main` branch → automatic deployment to production
- `develop` branch → automatic deployment to staging
- Feature branches → manual deployment to test environments
- Rollback available within 5 minutes

---

## Exception Process

### Temporary Exceptions

For urgent hotfixes or time-sensitive experimental features:

**Requirements:**
- Written justification in PR description
- Architecture Lead approval
- Created tracking issue for proper remediation
- Maximum 7-day exception period
- Must not compromise security or privacy

**Approval Authority:** PM Agent + Infra Agent

### Permanent Exceptions

For architectural decisions requiring permanent rule modifications:

**Process:**
1. Create RFC (Request for Comments) document
2. Present to all agents in design review
3. Allow 1-week comment period
4. Team discussion and vote (majority required)
5. Update RULES.md via dedicated PR
6. Notify all stakeholders
7. Update related documentation

**Approval Authority:** Majority vote of all agents

---

## Compliance Verification

### Automated Checks
- **Pre-commit hooks:** Linting, secret scanning, formatting
- **PR checks:** Tests, coverage, security scan, documentation build
- **Scheduled scans:** Nightly dependency vulnerability checks, license compliance

### Manual Reviews
- **Weekly:** TASKS.md progress review, blocker identification
- **Monthly:** Security audit, performance profiling
- **Quarterly:** Architecture review, dependency updates, technical debt assessment
- **Annually:** Privacy compliance audit, disaster recovery testing, process retrospective

---

## Violation Handling

### Process
1. **Detection:** Automated tool or manual identification
2. **Documentation:** Create issue with severity label and details
3. **Triage:** PM Agent assigns priority and owner
4. **Remediation:** Fix implemented and verified
5. **Review:** Post-mortem for recurring violations
6. **Prevention:** Process/tooling improvements to prevent recurrence

### Severity Levels
- **Critical:** Security vulnerability, data breach risk, production down
  - Response time: Immediate (< 1 hour)
  - All hands on deck until resolved
  
- **High:** Production degradation, major functionality broken
  - Response time: Same day
  - Prioritized above all other work
  
- **Medium:** Feature degradation, minor security issue, performance regression
  - Response time: Within 3 days
  - Scheduled in current sprint
  
- **Low:** Code quality, documentation gaps, technical debt
  - Response time: Within 2 weeks
  - Backlog item for future sprint

### Escalation Path
```
Developer → Team Lead → PM Agent → Architecture Review → External Audit (if needed)
```

---

## Continuous Improvement

This RULES.md document is a living artifact. Quarterly reviews ensure rules remain relevant and effective. Suggestions for improvements welcome through RFC process. Culture of learning from mistakes and adapting processes accordingly.
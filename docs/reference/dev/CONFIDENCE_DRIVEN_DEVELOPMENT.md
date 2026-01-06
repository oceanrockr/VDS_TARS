# Confidence-Driven Development Guide

**Version:** 1.2
**Project:** T.A.R.S.
**Methodology:** RiPIT Agent Coding Workflow
**Date:** January 3, 2026
**Status:** Active
**Last Session:** Phase 25 - Backup & Recovery (Analysis Complete)

---

## Core Principles

1. **Always calculate confidence before implementing**
2. **Never code without tests**
3. **Analyze before implementing**
4. **Ask when uncertain**

---

## CONFIDENCE SCORING (Required for ALL Changes)

### Before ANY implementation, state:

```
CONFIDENCE: X%
Reasoning: [brief explanation]
```

### Scoring Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| API Documentation | 30% | Official docs, type definitions, examples |
| Similar Patterns | 25% | Existing codebase patterns, prior implementations |
| Data Flow | 20% | Understanding of input/output/state flow |
| Complexity | 15% | Task complexity and scope |
| Impact | 10% | Blast radius of changes |

### Confidence Score Calculation

```
CONFIDENCE = (API_DOCS × 0.30) + (PATTERNS × 0.25) + (DATA_FLOW × 0.20) +
             (COMPLEXITY × 0.15) + (IMPACT × 0.10)
```

### Actions by Score

| Score | Action |
|-------|--------|
| **≥95%** | Implement immediately |
| **90-94%** | Implement with noted uncertainties |
| **<90%** | **STOP** - Present multiple-choice options |

### Multiple Choice Format (for <90% confidence)

```
CONFIDENCE: X% - [uncertainty reason]

Options:

A: [approach] - Best if [condition]
B: [approach] - Best if [condition]
C: [approach] - Best if [condition]

Which fits your needs?
```

---

## TWO-PHASE WORKFLOW (Required for ALL Fixes)

### PHASE 1: ANALYZE (No Code Yet)

```
╔═══════════════════════════════════════════════════════════════╗
║                         ANALYSIS                               ║
╠═══════════════════════════════════════════════════════════════╣
║ Issue:           [what's broken]                               ║
║ Evidence:        [errors/logs]                                 ║
║ Location:        [file/function]                               ║
║ Root Cause:      [underlying problem]                          ║
║ Recommended Fix: [approach]                                    ║
║ Risk:            [potential issues]                            ║
╠═══════════════════════════════════════════════════════════════╣
║ CONFIDENCE: X%                                                 ║
╠═══════════════════════════════════════════════════════════════╣
║ AWAITING APPROVAL - Proceed?                                   ║
╚═══════════════════════════════════════════════════════════════╝
```

### PHASE 2: IMPLEMENT (After Approval)

```
╔═══════════════════════════════════════════════════════════════╗
║                      IMPLEMENTATION                            ║
╠═══════════════════════════════════════════════════════════════╣
║ TESTS (write first):                                           ║
║   [Unit + edge case + regression tests]                        ║
║                                                                 ║
║ IMPLEMENTATION:                                                 ║
║   [The fix]                                                     ║
║                                                                 ║
║ VALIDATION:                                                     ║
║   [Confirm tests pass]                                          ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## MANDATORY TEST STRUCTURE

Every fix requires:

- **Unit test** (specific function)
- **Edge case test** (boundaries)
- **Regression test** (original bug scenario)

### Python Test Template

```python
import pytest
from typing import Any

class TestFixDescription:
    """Test suite for: [Issue Description]"""

    def test_expected_behavior(self):
        """Unit test: should [expected behavior]"""
        # Arrange
        input_data = ...
        expected = ...

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected

    def test_edge_case(self):
        """Edge case: should handle [boundary condition]"""
        # Arrange
        edge_input = ...

        # Act
        result = function_under_test(edge_input)

        # Assert
        assert result is not None  # or appropriate assertion

    def test_regression(self):
        """Regression: should not regress [original bug]"""
        # Arrange - reproduce original failing case
        failing_input = ...

        # Act
        result = function_under_test(failing_input)

        # Assert - verify bug is fixed
        assert result != broken_behavior
```

### JavaScript Test Template

```javascript
describe('Fix: [Issue Description]', () => {
  it('should [expected behavior]', () => {
    // Arrange
    const input = {};
    const expected = {};

    // Act
    const result = functionUnderTest(input);

    // Assert
    expect(result).toEqual(expected);
  });

  it('should handle [edge case]', () => {
    // Boundary test
    const edgeInput = {};

    const result = functionUnderTest(edgeInput);

    expect(result).toBeDefined();
  });

  it('should not regress [original bug]', () => {
    // Original failing case
    const failingInput = {};

    const result = functionUnderTest(failingInput);

    expect(result).not.toBe(brokenBehavior);
  });
});
```

---

## CRITICAL RULES

### Never Do

| Rule | Description |
|------|-------------|
| ❌ | Skip confidence scoring |
| ❌ | Implement without tests |
| ❌ | Code and analyze simultaneously |
| ❌ | Guess at requirements |
| ❌ | Make assumptions about user intent |

### Always Do

| Rule | Description |
|------|-------------|
| ✅ | Ask when <90% confident |
| ✅ | Give user checkpoint before coding |
| ✅ | Test root cause, not symptoms |
| ✅ | Document uncertainties |
| ✅ | Verify with user before significant changes |

---

## WORKFLOW EXAMPLES

### Example 1: Bug Fix Request

**User:** "Fix the email validation"

#### Wrong Response ❌
```
"Here's the updated regex: /^[^\s@]+@[^\s@]+\.[^\s@]+$/"
```

#### Correct Response ✅

**Phase 1: Analysis**
```
╔═══════════════════════════════════════════════════════════════╗
║                         ANALYSIS                               ║
╠═══════════════════════════════════════════════════════════════╣
║ Issue:           Email validation failing                      ║
║ Evidence:        [error message from logs]                     ║
║ Location:        validators/email.py:42                        ║
║ Root Cause:      Current regex too permissive for edge cases   ║
║ Recommended Fix: Replace with RFC 5322 compliant pattern       ║
║ Risk:            May reject some valid but unusual addresses   ║
╠═══════════════════════════════════════════════════════════════╣
║ CONFIDENCE: 85%                                                ║
║ Below 90% because: Multiple valid approaches exist             ║
╠═══════════════════════════════════════════════════════════════╣
║ Options:                                                       ║
║   A: RFC 5322 compliant - Best if: need full standard          ║
║   B: Simple pattern - Best if: basic validation ok             ║
║   C: Review requirements - Best if: spec exists                ║
║                                                                 ║
║ Which approach?                                                 ║
╚═══════════════════════════════════════════════════════════════╝
```

**After user chooses A:**

**Phase 2: Implementation**
```python
# TESTS (written first):

def test_validates_standard_emails():
    """Unit test: standard email formats"""
    assert is_valid_email('user@example.com') is True
    assert is_valid_email('user.name@example.com') is True

def test_rejects_invalid_emails():
    """Edge case: invalid formats"""
    assert is_valid_email('invalid') is False
    assert is_valid_email('@example.com') is False
    assert is_valid_email('user@') is False

def test_handles_edge_cases():
    """Edge case: unusual but valid formats"""
    assert is_valid_email('user+tag@example.co.uk') is True
    assert is_valid_email('user.name+tag@subdomain.example.com') is True

def test_regression_original_bug():
    """Regression: original failing case"""
    # The email that was incorrectly rejected
    assert is_valid_email('valid.but.rejected@example.org') is True

# IMPLEMENTATION:
[RFC 5322 validator code]

# VALIDATION:
All tests pass including original failing case.
```

---

### Example 2: Feature Implementation

**User:** "Add a logout button to the dashboard"

#### Correct Response ✅

**Phase 1: Analysis**
```
╔═══════════════════════════════════════════════════════════════╗
║                         ANALYSIS                               ║
╠═══════════════════════════════════════════════════════════════╣
║ Feature:         Logout button on dashboard                    ║
║ Current State:   No logout mechanism in dashboard              ║
║ Location:        dashboard/frontend/src/components/            ║
║ Implementation:                                                 ║
║   1. Add logout button to header component                     ║
║   2. Call /auth/revoke API on click                            ║
║   3. Clear local storage                                       ║
║   4. Redirect to login page                                    ║
║ Dependencies:    Auth service, React Router                    ║
╠═══════════════════════════════════════════════════════════════╣
║ CONFIDENCE: 92%                                                ║
║ Reasoning: Clear pattern from existing auth flow               ║
╠═══════════════════════════════════════════════════════════════╣
║ Proceeding with implementation...                              ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## CONFIDENCE SCORE TEMPLATES

### High Confidence (≥95%)

```
CONFIDENCE: 97%
Reasoning:
  - API docs fully reviewed (30/30)
  - Identical pattern exists in codebase (25/25)
  - Clear data flow documented (20/20)
  - Low complexity single-file change (15/15)
  - Minimal impact, isolated change (7/10)

Proceeding with implementation.
```

### Medium Confidence (90-94%)

```
CONFIDENCE: 92%
Reasoning:
  - API docs reviewed but some gaps (25/30)
  - Similar pattern exists, needs adaptation (23/25)
  - Data flow understood (18/20)
  - Moderate complexity (13/15)
  - Limited impact scope (8/10)

Noted uncertainties:
  - [specific uncertainty 1]
  - [specific uncertainty 2]

Proceeding with caution.
```

### Low Confidence (<90%)

```
CONFIDENCE: 78%
Reasoning:
  - Limited API documentation (20/30)
  - No similar pattern in codebase (15/25)
  - Data flow unclear in some areas (12/20)
  - High complexity (10/15)
  - Potential wide impact (6/10)

Cannot proceed without clarification.

Options:
A: [approach] - Best if [condition]
B: [approach] - Best if [condition]
C: [approach] - Best if [condition]

Which approach fits your needs?
```

---

## T.A.R.S. PROJECT-SPECIFIC GUIDELINES

### File Modification Protocol

1. **Read before edit** - Always use Read tool before Edit
2. **One change at a time** - Atomic, reviewable changes
3. **Test after each change** - Verify with pytest

### Security-Sensitive Changes

For changes to:
- `backend/app/core/sanitize.py`
- `backend/app/middleware/security_headers.py`
- `security/certificate_monitor.py`
- `cognition/shared/auth.py`
- `cognition/shared/rate_limiter.py`

**Require:**
- Minimum 95% confidence
- Security test coverage
- Code review checkpoint

### API Changes

For changes to:
- `backend/app/api/*`
- `enterprise_api/*`

**Require:**
- OpenAPI spec update
- Integration test
- Client compatibility check

---

## WHY THIS PROTOCOL WORKS

### The Problem

Asking for immediate fixes forces simultaneous:
- Problem understanding
- Solution generation
- Code writing

**Result:** High error rate, wasted effort, frustrated users

### The Solution

Separate phases:
1. **Analyze** → User reviews → Approve approach
2. **Tests first** → Validates fix → Documents behavior
3. **Implement** → With confidence → Minimal rework

**Result:** Fewer bugs, better code, clearer communication

### Key Benefits

| Benefit | Description |
|---------|-------------|
| Reduced Rework | User approves approach before coding |
| Better Quality | Tests written before implementation |
| Clear Communication | Uncertainties surfaced early |
| Audit Trail | Analysis documented for future reference |
| User Alignment | Multiple options when approach unclear |

---

## QUICK REFERENCE CARD

```
┌─────────────────────────────────────────────────────────────┐
│           CONFIDENCE-DRIVEN DEVELOPMENT QUICK REF           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BEFORE CODING:                                             │
│    1. State CONFIDENCE: X%                                  │
│    2. If <90%: Present options, ask user                    │
│    3. If ≥90%: Proceed with noted uncertainties             │
│                                                             │
│  PHASE 1 - ANALYZE:                                         │
│    • Issue, Evidence, Location                              │
│    • Root Cause, Recommended Fix                            │
│    • Risk assessment                                        │
│    • AWAIT APPROVAL                                         │
│                                                             │
│  PHASE 2 - IMPLEMENT:                                       │
│    • Write tests FIRST                                      │
│    • Implement fix                                          │
│    • Validate tests pass                                    │
│                                                             │
│  TESTS REQUIRED:                                            │
│    • Unit test (expected behavior)                          │
│    • Edge case test (boundaries)                            │
│    • Regression test (original bug)                         │
│                                                             │
│  NEVER:                                                     │
│    ❌ Skip confidence scoring                               │
│    ❌ Code without tests                                    │
│    ❌ Analyze and code simultaneously                       │
│                                                             │
│  ALWAYS:                                                    │
│    ✅ Ask when <90% confident                               │
│    ✅ Checkpoint before coding                              │
│    ✅ Test root cause, not symptoms                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## COMMON PITFALLS AND SOLUTIONS

### Pitfall 1: Premature Implementation
**Problem:** Starting to code before fully understanding the issue.

**Solution:** Always complete PHASE 1 (ANALYZE) and get user approval before writing any code.

**Example:**
```
❌ BAD: "I'll fix the auth bug by adding a check..."
✅ GOOD: "ANALYSIS: Auth token expires at 1h but refresh at 30m..."
```

### Pitfall 2: Overconfidence
**Problem:** Claiming 95%+ confidence without solid evidence.

**Solution:** Be honest about uncertainties. It's better to present options than to implement the wrong solution.

**Example:**
```
❌ BAD: CONFIDENCE: 95% (but haven't checked the auth flow)
✅ GOOD: CONFIDENCE: 82% - Uncertain if middleware applies to websockets
```

### Pitfall 3: Symptom Patching
**Problem:** Fixing the visible error without addressing root cause.

**Solution:** Trace back to the actual source of the problem.

**Example:**
```
❌ BAD: "Added try/catch to hide the error"
✅ GOOD: "Root cause: race condition in async initialization"
```

### Pitfall 4: Testing Implementation Instead of Behavior
**Problem:** Tests that verify code structure instead of functionality.

**Solution:** Test the behavior and outcomes, not the implementation details.

**Example:**
```
❌ BAD: assert function_called_twice()
✅ GOOD: assert result == expected_value
```

---

## T.A.R.S. ARCHITECTURE CONFIDENCE FACTORS

### Backend (FastAPI)
```
High Confidence (95%+):
- Standard CRUD endpoints
- Pydantic model validation
- JWT authentication patterns
- OpenAPI documentation

Medium Confidence (85-94%):
- Async background tasks
- WebSocket connections
- Multi-service orchestration
- Custom middleware

Low Confidence (<85%):
- Performance optimization
- Complex async chains
- New external integrations
- Database schema migrations
```

### RAG System (ChromaDB + Ollama)
```
High Confidence (95%+):
- Document ingestion
- Basic similarity search
- Collection management
- Standard embeddings

Medium Confidence (85-94%):
- Query optimization
- Multi-collection searches
- Custom metadata filters
- Embedding model changes

Low Confidence (<85%):
- Performance tuning at scale
- Advanced retrieval strategies
- Hybrid search implementations
- Custom distance metrics
```

### Deployment (Docker Compose)
```
High Confidence (95%+):
- Service definitions
- Volume mounts
- Network configuration
- Environment variables

Medium Confidence (85-94%):
- GPU passthrough
- Multi-stage builds
- Health check strategies
- Resource limits

Low Confidence (<85%):
- Swarm mode orchestration
- Custom network plugins
- Advanced security policies
- Cross-platform compatibility
```

---

## ESCALATION GUIDELINES

### When to Ask for Clarification

1. **Ambiguous Requirements**
   - Multiple valid interpretations
   - Unclear success criteria
   - Missing business context

2. **Technical Uncertainty**
   - Unfamiliar technology
   - Undocumented patterns
   - Complex architectural decisions

3. **Risk Assessment**
   - Potential breaking changes
   - Security implications
   - Performance impact unknown

### How to Present Options

```
CONFIDENCE: X% - [specific uncertainty]

I need clarification on: [specific question]

Options I'm considering:
A: [approach]
   Pros: [benefits]
   Cons: [drawbacks]
   Best if: [condition]

B: [approach]
   Pros: [benefits]
   Cons: [drawbacks]
   Best if: [condition]

C: [approach]
   Pros: [benefits]
   Cons: [drawbacks]
   Best if: [condition]

Which aligns with your goals?
```

---

## TESTING STRATEGIES BY COMPONENT

### API Endpoints
```python
# Test structure for API endpoints
def test_endpoint_success_case():
    """Happy path with valid data."""
    response = client.post("/api/endpoint", json=valid_data)
    assert response.status_code == 200

def test_endpoint_validation():
    """Invalid data should return 422."""
    response = client.post("/api/endpoint", json=invalid_data)
    assert response.status_code == 422

def test_endpoint_authentication():
    """Unauthenticated request should return 401."""
    response = client.post("/api/endpoint", headers={})
    assert response.status_code == 401

def test_endpoint_authorization():
    """Insufficient permissions should return 403."""
    response = client.post("/api/endpoint", headers=user_token)
    assert response.status_code == 403
```

### Middleware
```python
# Test structure for middleware
def test_middleware_processes_request():
    """Middleware should process valid requests."""
    response = client.get("/test", headers=valid_headers)
    assert "X-Custom-Header" in response.headers

def test_middleware_rejects_invalid():
    """Middleware should reject invalid requests."""
    response = client.get("/test", headers=invalid_headers)
    assert response.status_code == 400

def test_middleware_chain_order():
    """Middleware should execute in correct order."""
    response = client.get("/test")
    assert response.headers["X-Process-Order"] == "1,2,3"
```

### Background Tasks
```python
# Test structure for async tasks
@pytest.mark.asyncio
async def test_task_completes():
    """Task should complete successfully."""
    result = await background_task(valid_input)
    assert result.status == "completed"

@pytest.mark.asyncio
async def test_task_handles_errors():
    """Task should handle errors gracefully."""
    result = await background_task(invalid_input)
    assert result.status == "failed"
    assert result.error_message is not None

@pytest.mark.asyncio
async def test_task_timeout():
    """Task should respect timeout."""
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(long_running_task(), timeout=1.0)
```

### Deployment Scripts
```bash
# Test structure for bash scripts
test_script_prerequisites() {
    # Verify required commands exist
    command -v docker >/dev/null 2>&1
    assertTrue "Docker should be installed" $?
}

test_script_creates_required_files() {
    # Verify script creates expected output
    ./script.sh
    assertTrue "Config file should exist" "[ -f /path/to/config ]"
}

test_script_handles_errors() {
    # Verify script fails gracefully
    ./script.sh --invalid-option 2>&1 | grep -q "Error"
    assertTrue "Should show error message" $?
}
```

---

## DOCUMENTATION REQUIREMENTS

Every significant change requires:

1. **Code Comments**
   - Why (not what) for non-obvious logic
   - Assumptions and constraints
   - Edge cases handled

2. **API Documentation**
   - Endpoint description
   - Request/response schemas
   - Error codes and meanings
   - Example usage

3. **README Updates**
   - New features in relevant README
   - Configuration changes
   - Breaking changes highlighted

4. **CHANGELOG Entry**
   - Follow Keep a Changelog format
   - Link to related issues/PRs
   - Migration steps if needed

---

## REAL-WORLD T.A.R.S. EXAMPLES

### Example 3: Security Enhancement

**User:** "Add rate limiting to the API"

**Correct Response:**
```
ANALYSIS

Issue: No rate limiting protection
Evidence: User request for security hardening
Location: backend/app/middleware/ (new middleware needed)
Root Cause: Missing rate limiting middleware
Recommended Fix: Implement SlowAPI middleware with Redis backend

CONFIDENCE: 87%
Below 90% because:
- Uncertain on desired rate limits per endpoint
- Need to confirm Redis already available for shared state

Options:
A: SlowAPI with Redis - Best if: Multiple backend instances (production)
B: SlowAPI in-memory - Best if: Single instance development
C: Nginx rate limiting - Best if: Prefer infrastructure-level control

Questions:
1. What rate limits? (e.g., 100 req/min per IP?)
2. Per-endpoint or global limits?
3. Redis available for state sharing?

Which approach and what limits?
```

### Example 4: Performance Optimization

**User:** "The RAG search is slow with large documents"

**Correct Response:**
```
ANALYSIS

Issue: Slow RAG search performance
Evidence: User report (need metrics)
Location: backend/app/services/rag_service.py
Root Cause: [Need to investigate - multiple possibilities]

Before recommending fix, I need data:
1. Current response time? (avg/p95/p99)
2. Document count and sizes?
3. Embedding model used?
4. ChromaDB collection size?

CONFIDENCE: 65%
Too low to recommend specific fix without metrics

Next steps:
1. Add timing instrumentation
2. Profile the search pipeline
3. Analyze bottleneck (embedding/retrieval/reranking)

May I add logging to measure performance first?
```

### Example 5: Bug Fix with Migration

**User:** "User sessions aren't persisting across restarts"

**Correct Response:**
```
ANALYSIS

Issue: Sessions lost on restart
Evidence: User report
Location: backend/app/core/session.py
Root Cause: Sessions stored in memory, not persisted to Redis

Recommended Fix:
1. Update SessionMiddleware to use Redis backend
2. Migrate existing session schema if needed
3. Add session TTL configuration

CONFIDENCE: 92%
High because: Clear pattern, Redis already available

Risks:
- Requires Redis connection (add to health checks)
- Existing in-memory sessions will be lost (acceptable?)
- Performance impact minimal (Redis is fast)

TESTS:
- Session persists after server restart
- Session expires after TTL
- Multiple servers share session state

Data migration needed:
- No migration (sessions are ephemeral)
- Users will need to re-login once

Proceed with implementation?
```

---

## VERSION CONTROL INTEGRATION

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

CONFIDENCE: X%
Tests: [list of test files]
```

**Types:**
- feat: New feature
- fix: Bug fix
- refactor: Code restructuring
- test: Adding tests
- docs: Documentation
- chore: Maintenance

**Example:**
```
fix(api): resolve session persistence on restart

Update SessionMiddleware to use Redis backend instead of in-memory
storage. Sessions now persist across service restarts.

CONFIDENCE: 92%
Tests: tests/test_session_persistence.py
Risk: Requires Redis connection
Breaking: Users must re-login once after deployment
```

### Branch Naming
```
<type>/<issue-number>-<short-description>

Examples:
- fix/123-session-persistence
- feat/456-rate-limiting
- refactor/789-api-cleanup
```

---

## METRICS AND RETROSPECTIVES

### Track Confidence Accuracy
```
Initial Confidence: X%
Actual Outcome: [success/partial/failure]
Lessons Learned: [what was missed]
```

### Common Confidence Errors
1. **Overestimation:** Claiming 95% but missing edge cases
2. **Underestimation:** Being too cautious on well-known patterns
3. **Wrong Factors:** Focusing on code complexity vs. requirement clarity

### Improvement Actions
- Review past confidence scores vs. outcomes
- Document patterns that increase/decrease confidence
- Build confidence calibration over time

---

## REFERENCE

- **RiPIT Methodology:** https://github.com/Veleron-Dev-Studios-LLC/VDS_RiPIT-Agent-Coding-Workflow
- **T.A.R.S. Repository:** https://github.com/oceanrockr/VDS_TARS
- **Current Version:** v1.0.11 (GA)
- **Keep a Changelog:** https://keepachangelog.com/
- **Semantic Versioning:** https://semver.org/

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-27 | Initial protocol definition |
| 1.0 | 2025-12-28 | Comprehensive guide with RiPIT methodology |
| 1.1 | 2026-01-01 | Phase 24 updates, enhanced quick reference |
| 1.2 | 2026-01-03 | Phase 25 analysis, session learnings added |

---

## SESSION CONTINUATION PROMPT

Use this prompt to continue development in a new Claude Code session:

```markdown
# T.A.R.S. RiPIT Continuation Prompt

## Project Context
- **Repository:** https://github.com/oceanrockr/VDS_TARS.git
- **Version:** v1.0.11 (GA) → v1.0.12
- **Last Phase:** 24 - Field Testing & Codebase Hygiene (COMPLETE)
- **Current Phase:** 25 - Backup & Recovery (Analysis Complete - IMPLEMENT)

## RiPIT Status
- **Confidence:** 92.50% (APPROVED)
- **Action:** Implement with noted uncertainties
- **Tests First:** Required

## Phase 25 Deliverables
1. tests/test_backup_restore.py (WRITE FIRST)
2. deploy/backup-tars.sh
3. deploy/restore-tars.sh
4. docs/BACKUP_RECOVERY.md

## Key Files
- docs/reference/dev/CONFIDENCE_DRIVEN_DEVELOPMENT.md
- docs/reference/dev/HANDOFF_PHASE25_BACKUP_RECOVERY.md
- deploy/generate-support-bundle.sh (pattern template)

## Docker MCP
docker mcp client connect claude-code --global
```

---

## JAVASCRIPT TEST TEMPLATE (Enhanced)

```javascript
describe('Fix: [Issue Description]', () => {
  // Unit test - expected behavior
  it('should [expected behavior]', () => {
    // Arrange
    const input = {};
    const expected = {};

    // Act
    const result = functionUnderTest(input);

    // Assert
    expect(result).toEqual(expected);
  });

  // Edge case test - boundary conditions
  it('should handle [edge case]', () => {
    const edgeInput = {};
    const result = functionUnderTest(edgeInput);
    expect(result).toBeDefined();
  });

  // Regression test - original bug scenario
  it('should not regress [original bug]', () => {
    const failingInput = {};
    const result = functionUnderTest(failingInput);
    expect(result).not.toBe(brokenBehavior);
  });

  // Security test - for security-sensitive code
  it('should sanitize [malicious input]', () => {
    const maliciousInput = '<script>alert("xss")</script>';
    const result = functionUnderTest(maliciousInput);
    expect(result).not.toContain('<script>');
  });
});
```

---

## PHASE 24 SESSION LEARNINGS

### Confidence Score Applied
```
Session Confidence: 91.75%

Breakdown:
- API Documentation:     95% × 0.30 = 28.5
- Similar Patterns:      90% × 0.25 = 22.5
- Data Flow:             95% × 0.20 = 19.0
- Complexity:            85% × 0.15 = 12.75
- Impact:                90% × 0.10 = 9.0
                         ─────────────────
                         TOTAL: 91.75%

Result: Proceeded with noted uncertainties
Outcome: SUCCESS - All deliverables completed
```

### Key Decisions Made
1. **Categorize before commit** - Analyzed all untracked files before action
2. **Remove redundant docs** - 9 duplicate files deleted to reduce maintenance
3. **Commit valid tests** - All security module tests now tracked
4. **Update exports** - Module `__init__.py` files properly export public APIs

---

## PHASE 25 SESSION LEARNINGS

### Confidence Score Applied
```
Session Confidence: 92.50%

Breakdown:
- API Documentation:     95% × 0.30 = 28.50
- Similar Patterns:      95% × 0.25 = 23.75
- Data Flow:             90% × 0.20 = 18.00
- Complexity:            85% × 0.15 = 12.75
- Impact:                95% × 0.10 = 9.50
                         ─────────────────
                         TOTAL: 92.50%

Result: Proceeded with noted uncertainties
Outcome: Analysis complete, implementation pending
```

### Key Decisions Made
1. **TDD Required** - Tests must be written before implementation
2. **Pattern Reuse** - Use generate-support-bundle.sh as script template
3. **Modular Backup** - Separate functions for each data source
4. **Integrity First** - SHA-256 checksums for all backup components

### Noted Uncertainties
1. ChromaDB API bulk export version compatibility
2. Large Ollama model backup handling
3. Restore sequencing with running services

---

**Document Status:** Active
**Applies To:** All T.A.R.S. development work
**Enforcement:** Required for all code changes

---

**Last Updated:** January 3, 2026
**Maintainer:** T.A.R.S. Development Team
**License:** MIT

# Contributing to T.A.R.S.

Thank you for your interest in contributing to T.A.R.S. (Temporal Augmented Retrieval System). This document provides guidelines and instructions for contributing to the project.

## Development Workflow

This project follows the **VDS RiPIT Agent Coding Workflow v2.9** conventions for development.

### Prerequisites

- Python 3.9+
- Redis 6.0+ (for API server and rate limiting)
- Git

### Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/oceanrockr/VDS_TARS.git
cd VDS_TARS

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=scripts --cov=analytics --cov-report=html

# Run specific test suite
pytest tests/integration/ -v

# Run a specific test file
pytest tests/integration/test_org_sla_intelligence.py -v
```

### Code Style

We use the following tools for code formatting and linting:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Run both
black . && isort .
```

### Pre-Commit Checklist

Before submitting a PR:

1. **Run tests:** `pytest tests/ -v`
2. **Format code:** `black . && isort .`
3. **Check for type hints:** Ensure new code includes type annotations
4. **Update documentation:** If adding new features, update relevant docs
5. **Add tests:** New features should include corresponding test cases

## Branching Strategy

- `main` - Production-ready code (protected)
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Creating a Branch

```bash
# Feature branch
git checkout -b feature/my-new-feature

# Bug fix branch
git checkout -b fix/issue-123-description

# Documentation branch
git checkout -b docs/update-configuration-guide
```

## Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code style (formatting, no logic changes)
- `refactor` - Code refactoring
- `test` - Adding/updating tests
- `chore` - Maintenance tasks

**Examples:**
```
feat(analytics): add temporal intelligence correlation engine

fix(api): resolve JWT token refresh race condition

docs(readme): update documentation map section
```

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** with appropriate commits
3. **Run tests** and ensure all pass
4. **Update documentation** as needed
5. **Submit a PR** with a clear description
6. **Address review feedback** promptly
7. **Squash and merge** when approved

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] README updated (if applicable)
- [ ] API docs updated (if applicable)
- [ ] Configuration guide updated (if applicable)
```

## Project Structure

```
VDS_TARS/
├── analytics/           # Analytics engines
├── cognition/           # Cognitive services
├── docs/                # Production documentation
│   ├── reference/       # Development/historical docs
│   └── runbooks/        # Operational runbooks
├── examples/            # Ready-to-use examples
├── observability/       # Observability tools
├── policies/            # SLA policy templates
├── scripts/             # CLI tools and utilities
├── tests/               # Test suites
│   ├── integration/     # Integration tests
│   └── unit/            # Unit tests (if applicable)
└── ...
```

## Adding New Features

### Analytics Engines

New analytics engines should:

1. Live in `analytics/` directory
2. Include a corresponding CLI in `analytics/run_*.py`
3. Have integration tests in `tests/integration/`
4. Be documented in `docs/`

### CLI Tools

New CLI tools should:

1. Use argparse for argument parsing
2. Support `--help` with clear descriptions
3. Define exit codes for CI/CD integration
4. Include a `--dry-run` mode where applicable

### Exit Code Conventions

| Range | Category |
|-------|----------|
| 0 | Success |
| 1-79 | General errors |
| 80-99 | Repository health errors |
| 100-119 | Alerting errors |
| 120-139 | Trend/correlation errors |
| 140-159 | SLA/compliance errors |
| 199 | General/unknown error |

## Getting Help

- **Documentation:** See [docs/](docs/) for guides
- **Issues:** Open a GitHub issue for bugs or feature requests
- **Reference Docs:** See [docs/reference/](docs/reference/) for implementation details

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Follow project conventions

---

**Thank you for contributing to T.A.R.S.!**

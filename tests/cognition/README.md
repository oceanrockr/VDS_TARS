# T.A.R.S. Cognition Module Test Infrastructure

This directory contains the test infrastructure for the T.A.R.S. cognition module.

## Directory Structure

```
tests/cognition/
├── __init__.py                    # Package initializer
├── README.md                       # This file
├── conftest.py                     # Shared pytest fixtures
├── shared/                         # Tests for shared utilities
│   ├── __init__.py
│   └── test_rate_limiter.py       # Rate limiter tests
└── eval_engine/                    # Tests for evaluation engine
    └── __init__.py
```

## Running Tests

### Prerequisites

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

### Running All Cognition Tests

From the project root directory:

```bash
# Set PYTHONPATH and run all cognition tests
PYTHONPATH=. python -m pytest tests/cognition/ -v --no-cov

# Or with coverage
PYTHONPATH=. python -m pytest tests/cognition/ -v
```

### Running Specific Test Files

```bash
# Rate limiter tests
PYTHONPATH=. python -m pytest tests/cognition/shared/test_rate_limiter.py -v --no-cov

# Run specific test class
PYTHONPATH=. python -m pytest tests/cognition/shared/test_rate_limiter.py::TestInMemoryRateLimiter -v --no-cov

# Run specific test
PYTHONPATH=. python -m pytest tests/cognition/shared/test_rate_limiter.py::TestInMemoryRateLimiter::test_allows_requests_under_limit -v --no-cov
```

### Windows Users

On Windows, use the Windows-style path separator:

```powershell
$env:PYTHONPATH="."; python -m pytest tests/cognition/ -v --no-cov
```

Or in Git Bash:

```bash
PYTHONPATH=. python -m pytest tests/cognition/ -v --no-cov
```

## Available Fixtures

### Redis Fixtures

- `mock_redis`: Async mock Redis client with full sorted set support
- `mock_redis_sync`: Synchronous mock Redis client for testing sync code

### Database Fixtures

- `mock_database`: Mock database with user, session, and audit log support

### Authentication Fixtures

- `jwt_manager`: JWT manager for creating and verifying tokens
- `test_user`: Standard test user with developer role
- `test_admin_user`: Admin user for testing admin functionality
- `test_viewer_user`: Viewer user for testing limited permissions
- `access_token`: Pre-generated access token for test user
- `admin_token`: Pre-generated admin access token

### Request Fixtures

- `mock_request`: Mock FastAPI Request object for testing endpoints

### Utility Fixtures

- `freeze_time`: Freeze time for deterministic testing
- `reset_rate_limiter`: Auto-cleanup for rate limiter state
- `cleanup_env`: Auto-cleanup for environment variables

## Test Organization

### Test Classes

Tests are organized into classes by functionality:

- `TestInMemoryRateLimiter`: Tests for in-memory rate limiting
- `TestRedisRateLimiter`: Tests for Redis-backed rate limiting
- `TestRateLimiter`: Tests for main RateLimiter class
- `TestRateLimitConfig`: Tests for configuration
- `TestRateLimiterIntegration`: Integration tests
- `TestRateLimiterEdgeCases`: Edge cases and error handling

### Test Methods

Test methods follow the naming convention:
- `test_<functionality>_<expected_behavior>`

Example: `test_allows_requests_under_limit`

## Markers

Tests can be marked with pytest markers:

```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.integration
@pytest.mark.redis
def test_redis_integration():
    pass
```

Available markers:
- `unit`: Fast, isolated unit tests
- `integration`: Integration tests requiring services
- `redis`: Tests requiring Redis connection
- `auth`: Authentication/authorization tests
- `slow`: Slow-running tests (>30s)

## Coverage

To generate coverage reports:

```bash
# HTML coverage report
PYTHONPATH=. python -m pytest tests/cognition/ --cov=cognition --cov-report=html

# Terminal coverage report
PYTHONPATH=. python -m pytest tests/cognition/ --cov=cognition --cov-report=term-missing
```

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError: No module named 'cognition'`:

1. Ensure you're running from the project root directory
2. Set PYTHONPATH explicitly: `PYTHONPATH=. python -m pytest ...`
3. Verify `__init__.py` files exist in `cognition/` and `cognition/shared/`

### Redis Connection Errors

Tests use mock Redis by default. If you see Redis connection errors, ensure:

1. The `USE_REDIS` environment variable is set to `false` for testing
2. Mock fixtures are being used correctly

### Windows Path Issues

If you encounter path-related issues on Windows:

1. Use forward slashes in paths: `tests/cognition/`
2. Use Git Bash or PowerShell
3. Set environment variables appropriately for your shell

## Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Add appropriate fixtures to `conftest.py` if needed
3. Document complex test scenarios
4. Ensure tests are isolated and don't depend on external state
5. Use appropriate markers for test categorization

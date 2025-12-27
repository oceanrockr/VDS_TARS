"""
T.A.R.S. Shared Cognition Utilities

Shared components for cognition services:
- Authentication and authorization (JWT, API keys)
- Rate limiting
- Audit logging
- JWT key rotation
- API key management
"""

__all__ = [
    'auth',
    'rate_limiter',
    'audit_logger',
    'api_key_store',
    'jwt_key_store',
]

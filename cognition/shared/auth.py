"""
T.A.R.S. Authentication Module
JWT-based authentication with RBAC support and API key system
Phase 11.5
"""

import os
import jwt
import hashlib
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from functools import wraps
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Import persistent API key store
try:
    from cognition.shared.api_key_store import api_key_store, APIKeyRecord
    PERSISTENT_STORE_AVAILABLE = True
except ImportError:
    PERSISTENT_STORE_AVAILABLE = False
    logging.warning("API key persistent store not available, using in-memory fallback")

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles for RBAC"""
    VIEWER = "viewer"
    DEVELOPER = "developer"
    ADMIN = "admin"


class User(BaseModel):
    """User model"""
    user_id: str
    username: str
    roles: List[Role]
    email: Optional[str] = None


class TokenData(BaseModel):
    """JWT token payload"""
    user_id: str
    username: str
    roles: List[Role]
    exp: int
    iss: str = "tars-auth"


class APIKey(BaseModel):
    """API key for service-to-service authentication"""
    key_id: str
    service_name: str
    key_hash: str
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True


class AuthConfig:
    """Authentication configuration"""

    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", self._generate_secret())
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_minutes = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))
        self.refresh_expiry_days = int(os.getenv("REFRESH_EXPIRY_DAYS", "7"))
        self.issuer = "tars-auth"

        # API key configuration
        self.api_keys: Dict[str, APIKey] = {}
        self._load_api_keys()

    def _generate_secret(self) -> str:
        """Generate a secure random secret"""
        return secrets.token_urlsafe(32)

    def _load_api_keys(self):
        """Load API keys from environment or storage"""
        # For now, load from environment
        # In production, load from secure key storage
        automl_key = os.getenv("AUTOML_API_KEY")
        hypersync_key = os.getenv("HYPERSYNC_API_KEY")
        orchestration_key = os.getenv("ORCHESTRATION_API_KEY")

        if automl_key:
            self.api_keys["automl"] = APIKey(
                key_id="automl",
                service_name="AutoML Service",
                key_hash=self._hash_api_key(automl_key),
                created_at=datetime.utcnow()
            )

        if hypersync_key:
            self.api_keys["hypersync"] = APIKey(
                key_id="hypersync",
                service_name="HyperSync Service",
                key_hash=self._hash_api_key(hypersync_key),
                created_at=datetime.utcnow()
            )

        if orchestration_key:
            self.api_keys["orchestration"] = APIKey(
                key_id="orchestration",
                service_name="Orchestration Service",
                key_hash=self._hash_api_key(orchestration_key),
                created_at=datetime.utcnow()
            )

    @staticmethod
    def _hash_api_key(key: str) -> str:
        """Hash an API key using SHA-256"""
        return hashlib.sha256(key.encode()).hexdigest()


# Global auth config
auth_config = AuthConfig()


class AuthService:
    """Authentication service for JWT and API key management"""

    def __init__(self, config: AuthConfig = auth_config):
        self.config = config

    def create_access_token(self, user: User) -> str:
        """Create a JWT access token"""
        expiry = datetime.utcnow() + timedelta(minutes=self.config.jwt_expiry_minutes)

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "exp": int(expiry.timestamp()),
            "iss": self.config.issuer,
            "iat": int(datetime.utcnow().timestamp())
        }

        token = jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )

        return token

    def create_refresh_token(self, user: User) -> str:
        """Create a JWT refresh token"""
        expiry = datetime.utcnow() + timedelta(days=self.config.refresh_expiry_days)

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "exp": int(expiry.timestamp()),
            "iss": self.config.issuer,
            "type": "refresh"
        }

        token = jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )

        return token

    def verify_token(self, token: str) -> TokenData:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                issuer=self.config.issuer
            )

            return TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                roles=[Role(role) for role in payload["roles"]],
                exp=payload["exp"]
            )

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidIssuerError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )

    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify an API key (checks persistent store first, then in-memory fallback)"""
        key_hash = AuthConfig._hash_api_key(api_key)

        # Try persistent store first
        if PERSISTENT_STORE_AVAILABLE:
            try:
                record = api_key_store.get_by_hash(key_hash)
                if record and not record.revoked:
                    # Update last used timestamp (async in Redis)
                    api_key_store.update_last_used(record.key_id)

                    # Convert to APIKey model for compatibility
                    return APIKey(
                        key_id=record.key_id,
                        service_name=record.service_name,
                        key_hash=record.key_hash,
                        created_at=record.created_at,
                        last_used=record.last_used_at,
                        is_active=not record.revoked
                    )
            except Exception as e:
                logger.error(f"Failed to verify API key from persistent store: {e}")
                # Fall through to in-memory fallback

        # Fallback to in-memory store
        for key_id, stored_key in self.config.api_keys.items():
            if stored_key.key_hash == key_hash and stored_key.is_active:
                # Update last used timestamp
                stored_key.last_used = datetime.utcnow()
                return stored_key

        return None

    def generate_api_key(self, service_name: str) -> tuple[str, APIKey]:
        """Generate a new API key for a service (stores in persistent store if available)"""
        raw_key = secrets.token_urlsafe(32)
        key_id = service_name.lower().replace(" ", "_")

        api_key = APIKey(
            key_id=key_id,
            service_name=service_name,
            key_hash=AuthConfig._hash_api_key(raw_key),
            created_at=datetime.utcnow()
        )

        # Store in persistent store if available
        if PERSISTENT_STORE_AVAILABLE:
            try:
                record = APIKeyRecord(
                    key_id=key_id,
                    service_name=service_name,
                    key_hash=api_key.key_hash,
                    created_at=api_key.created_at,
                    revoked=False
                )
                api_key_store.create(record)
                logger.info(f"Stored API key {key_id} in persistent store")
            except Exception as e:
                logger.error(f"Failed to store API key in persistent store: {e}")
                # Fall through to in-memory storage

        # Always store in memory for backward compatibility
        self.config.api_keys[key_id] = api_key

        return raw_key, api_key

    def rotate_api_key(self, key_id: str) -> Optional[tuple[str, APIKey]]:
        """Rotate an existing API key (persists changes if persistent store available)"""
        # Check persistent store first
        old_key_data = None
        if PERSISTENT_STORE_AVAILABLE:
            try:
                record = api_key_store.get_by_id(key_id)
                if record:
                    old_key_data = record
            except Exception as e:
                logger.error(f"Failed to check persistent store for key rotation: {e}")

        # Check in-memory store if not found in persistent
        if not old_key_data and key_id not in self.config.api_keys:
            return None

        # Get service name
        service_name = old_key_data.service_name if old_key_data else self.config.api_keys[key_id].service_name

        # Generate new key
        raw_key, new_key = self.generate_api_key(service_name)

        # Revoke old key in persistent store
        if PERSISTENT_STORE_AVAILABLE:
            try:
                api_key_store.revoke(key_id)
                logger.info(f"Revoked old API key {key_id} in persistent store")
            except Exception as e:
                logger.error(f"Failed to revoke old key in persistent store: {e}")

        # Deactivate old key in memory
        if key_id in self.config.api_keys:
            self.config.api_keys[key_id].is_active = False

        return raw_key, new_key

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key (persists to store if available)"""
        revoked = False

        # Revoke in persistent store
        if PERSISTENT_STORE_AVAILABLE:
            try:
                if api_key_store.revoke(key_id):
                    revoked = True
                    logger.info(f"Revoked API key {key_id} in persistent store")
            except Exception as e:
                logger.error(f"Failed to revoke API key in persistent store: {e}")

        # Revoke in memory
        if key_id in self.config.api_keys:
            self.config.api_keys[key_id].is_active = False
            revoked = True

        return revoked

    def list_api_keys(self) -> List[APIKey]:
        """List all API keys (from persistent store if available, else in-memory)"""
        if PERSISTENT_STORE_AVAILABLE:
            try:
                records = api_key_store.list_all()
                return [
                    APIKey(
                        key_id=r.key_id,
                        service_name=r.service_name,
                        key_hash=r.key_hash,
                        created_at=r.created_at,
                        last_used=r.last_used_at,
                        is_active=not r.revoked
                    )
                    for r in records
                ]
            except Exception as e:
                logger.error(f"Failed to list API keys from persistent store: {e}")

        # Fallback to in-memory
        return list(self.config.api_keys.values())


# Global auth service
auth_service = AuthService()

# Security scheme for FastAPI
security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    """FastAPI dependency to get current authenticated user"""
    token = credentials.credentials
    token_data = auth_service.verify_token(token)

    return User(
        user_id=token_data.user_id,
        username=token_data.username,
        roles=token_data.roles
    )


def verify_service_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> APIKey:
    """FastAPI dependency to verify service API key"""
    api_key = credentials.credentials

    # Try to verify as API key first
    service_key = auth_service.verify_api_key(api_key)

    if service_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return service_key


def require_roles(*required_roles: Role):
    """Decorator to require specific roles for an endpoint"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (injected by FastAPI dependency)
            user = kwargs.get("current_user")

            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="No user context found"
                )

            # Check if user has any of the required roles
            user_roles = set(user.roles)
            required = set(required_roles)

            if not user_roles.intersection(required):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires one of roles: {[r.value for r in required_roles]}"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_admin(func):
    """Decorator to require admin role"""
    return require_roles(Role.ADMIN)(func)


def require_developer(func):
    """Decorator to require developer or admin role"""
    return require_roles(Role.DEVELOPER, Role.ADMIN)(func)


# Demo users for testing (in production, use database)
DEMO_USERS = {
    "admin": User(
        user_id="admin-001",
        username="admin",
        roles=[Role.ADMIN],
        email="admin@tars.ai"
    ),
    "developer": User(
        user_id="dev-001",
        username="developer",
        roles=[Role.DEVELOPER],
        email="dev@tars.ai"
    ),
    "viewer": User(
        user_id="view-001",
        username="viewer",
        roles=[Role.VIEWER],
        email="viewer@tars.ai"
    )
}


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user (demo implementation)
    In production, use proper password hashing and database lookup
    """
    # Demo: accept any password for demo users
    if username in DEMO_USERS:
        return DEMO_USERS[username]

    return None

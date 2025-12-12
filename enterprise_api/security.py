"""
Security and Authentication for Enterprise API

Features:
- JWT authentication
- API key authentication
- RBAC (Role-Based Access Control)
- Rate limiting
"""

from typing import Optional, Dict, Annotated
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import secrets

from enterprise_config.schema import RBACRole, AuthMode


# JWT Configuration
ALGORITHM = "HS256"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TokenData(BaseModel):
    """JWT token payload data."""
    username: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model for authentication."""
    username: str
    role: RBACRole
    disabled: bool = False


class SecurityManager:
    """
    Manages authentication and authorization for Enterprise API.
    """

    def __init__(
        self,
        secret_key: str,
        auth_mode: AuthMode = AuthMode.API_KEY,
        api_keys: Optional[Dict[str, str]] = None,
        token_expiration_minutes: int = 60,
    ):
        """
        Initialize security manager.

        Args:
            secret_key: Secret key for JWT signing
            auth_mode: Authentication mode (jwt, api_key, token)
            api_keys: Dictionary mapping API keys to roles
            token_expiration_minutes: JWT token expiration time
        """
        self.secret_key = secret_key
        self.auth_mode = auth_mode
        self.api_keys = api_keys or {}
        self.token_expiration_minutes = token_expiration_minutes

        # In-memory user store (for demo; use database in production)
        self.users: Dict[str, User] = {
            "admin": User(username="admin", role=RBACRole.ADMIN),
            "sre": User(username="sre", role=RBACRole.SRE),
            "viewer": User(username="viewer", role=RBACRole.READONLY),
        }

        # Password hashes (for demo; use database in production)
        self.password_hashes: Dict[str, str] = {
            "admin": pwd_context.hash("admin123"),  # Change in production!
            "sre": pwd_context.hash("sre123"),
            "viewer": pwd_context.hash("viewer123"),
        }

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username/password.

        Args:
            username: Username
            password: Password

        Returns:
            User if authenticated, None otherwise
        """
        if username not in self.users:
            return None

        user = self.users[username]
        if user.disabled:
            return None

        password_hash = self.password_hashes.get(username)
        if not password_hash:
            return None

        if not self.verify_password(password, password_hash):
            return None

        return user

    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.

        Args:
            user: User to create token for
            expires_delta: Token expiration delta

        Returns:
            JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.token_expiration_minutes)

        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "sub": user.username,
            "role": user.role.value,
            "exp": expire,
        }

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_jwt_token(self, token: str) -> Optional[TokenData]:
        """
        Verify JWT token.

        Args:
            token: JWT token string

        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            exp: int = payload.get("exp")

            if username is None or role is None:
                return None

            return TokenData(
                username=username,
                role=role,
                exp=datetime.fromtimestamp(exp) if exp else None
            )
        except JWTError:
            return None

    def verify_api_key(self, api_key: str) -> Optional[RBACRole]:
        """
        Verify API key and return associated role.

        Args:
            api_key: API key string

        Returns:
            RBACRole if valid, None otherwise
        """
        role_str = self.api_keys.get(api_key)
        if role_str:
            try:
                return RBACRole(role_str)
            except ValueError:
                return None
        return None

    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(32)

    def check_permission(
        self,
        user_role: RBACRole,
        required_role: RBACRole
    ) -> bool:
        """
        Check if user has required permission.

        Role hierarchy: admin > sre > readonly

        Args:
            user_role: User's role
            required_role: Required role for operation

        Returns:
            True if user has permission, False otherwise
        """
        role_hierarchy = {
            RBACRole.READONLY: 1,
            RBACRole.SRE: 2,
            RBACRole.ADMIN: 3,
        }

        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 99)

        return user_level >= required_level


# Dependency injection for FastAPI

def get_security_manager() -> SecurityManager:
    """
    Get SecurityManager instance (dependency injection).

    In production, this should load config from enterprise_config.
    """
    # For now, use hardcoded values
    # TODO: Load from enterprise_config
    return SecurityManager(
        secret_key="your-secret-key-change-in-production",  # CHANGE THIS!
        auth_mode=AuthMode.API_KEY,
        api_keys={
            "dev-key-readonly": "readonly",
            "dev-key-sre": "sre",
            "dev-key-admin": "admin",
        },
        token_expiration_minutes=60,
    )


async def get_current_user_jwt(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
    security_manager: Annotated[SecurityManager, Depends(get_security_manager)],
) -> User:
    """
    Get current user from JWT token (dependency).

    Args:
        credentials: HTTP Bearer credentials
        security_manager: SecurityManager instance

    Returns:
        Authenticated User

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = security_manager.verify_jwt_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token expired
    if token_data.exp and token_data.exp < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    user = security_manager.users.get(token_data.username)
    if user is None or user.disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    return user


async def get_current_user_api_key(
    api_key: Annotated[Optional[str], Depends(api_key_header)],
    security_manager: Annotated[SecurityManager, Depends(get_security_manager)],
) -> User:
    """
    Get current user from API key (dependency).

    Args:
        api_key: API key from X-API-Key header
        security_manager: SecurityManager instance

    Returns:
        User with role from API key

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    role = security_manager.verify_api_key(api_key)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Create anonymous user with API key role
    return User(username=f"api_key_{api_key[:8]}", role=role)


async def get_current_user(
    security_manager: Annotated[SecurityManager, Depends(get_security_manager)],
    jwt_user: Annotated[Optional[User], Depends(get_current_user_jwt)] = None,
    api_key_user: Annotated[Optional[User], Depends(get_current_user_api_key)] = None,
) -> User:
    """
    Get current user from either JWT or API key.

    Args:
        security_manager: SecurityManager instance
        jwt_user: User from JWT (if provided)
        api_key_user: User from API key (if provided)

    Returns:
        Authenticated User

    Raises:
        HTTPException: If authentication fails
    """
    # Try JWT first, then API key
    if jwt_user:
        return jwt_user
    elif api_key_user:
        return api_key_user
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )


def require_role(required_role: RBACRole):
    """
    Dependency factory for role-based access control.

    Usage:
        @app.get("/admin-only")
        async def admin_endpoint(user: User = Depends(require_role(RBACRole.ADMIN))):
            ...

    Args:
        required_role: Required role for access

    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: Annotated[User, Depends(get_current_user_api_key)],
        security_manager: Annotated[SecurityManager, Depends(get_security_manager)],
    ) -> User:
        if not security_manager.check_permission(current_user.role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}",
            )
        return current_user

    return role_checker

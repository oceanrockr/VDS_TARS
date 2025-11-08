"""
T.A.R.S. Authentication Models
Pydantic models for authentication requests and responses
"""

from typing import Optional
from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    """Request model for token generation"""
    client_id: str = Field(..., description="Unique client identifier")
    device_name: Optional[str] = Field(None, description="Device name for tracking")
    device_type: Optional[str] = Field(None, description="Device type (windows, android, etc.)")


class TokenResponse(BaseModel):
    """Response model for token generation"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class TokenRefreshRequest(BaseModel):
    """Request model for token refresh"""
    refresh_token: str = Field(..., description="Refresh token to exchange")


class TokenValidationResponse(BaseModel):
    """Response model for token validation"""
    valid: bool = Field(..., description="Whether the token is valid")
    client_id: Optional[str] = Field(None, description="Client ID from token")
    expires_at: Optional[str] = Field(None, description="Token expiration timestamp")


class TokenRevokeRequest(BaseModel):
    """Request model for token revocation"""
    token: str = Field(..., description="Token to revoke")

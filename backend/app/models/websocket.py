"""
T.A.R.S. WebSocket Models
Pydantic models for WebSocket messages and events
"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: str = Field(..., description="Message type")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChatMessage(WebSocketMessage):
    """Chat message from client"""
    type: Literal["chat"] = "chat"
    content: str = Field(..., description="Message content")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for threading")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(None, description="Model temperature override")
    max_tokens: Optional[int] = Field(None, description="Max tokens override")
    use_rag: bool = Field(default=False, description="Enable RAG for this message (Phase 3)")
    rag_top_k: Optional[int] = Field(None, description="Number of documents to retrieve (Phase 3)")
    rag_threshold: Optional[float] = Field(None, description="Relevance threshold (Phase 3)")


class TokenStreamMessage(WebSocketMessage):
    """Token stream message to client"""
    type: Literal["token"] = "token"
    token: str = Field(..., description="Generated token")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")


class StreamCompleteMessage(WebSocketMessage):
    """Stream completion message"""
    type: Literal["complete"] = "complete"
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    total_tokens: int = Field(..., description="Total tokens generated")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")


class ErrorMessage(WebSocketMessage):
    """Error message to client"""
    type: Literal["error"] = "error"
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class PingMessage(WebSocketMessage):
    """Ping message for heartbeat"""
    type: Literal["ping"] = "ping"


class PongMessage(WebSocketMessage):
    """Pong response for heartbeat"""
    type: Literal["pong"] = "pong"


class ConnectionAckMessage(WebSocketMessage):
    """Connection acknowledgment message"""
    type: Literal["connection_ack"] = "connection_ack"
    client_id: str = Field(..., description="Client ID from token")
    session_id: str = Field(..., description="Unique session ID")


class SystemMessage(WebSocketMessage):
    """System message to client"""
    type: Literal["system"] = "system"
    message: str = Field(..., description="System message content")
    level: Literal["info", "warning", "error"] = Field(default="info")


class MetricsMessage(WebSocketMessage):
    """Metrics update message"""
    type: Literal["metrics"] = "metrics"
    active_connections: int = Field(..., description="Number of active connections")
    queue_size: int = Field(..., description="Message queue size")
    avg_latency_ms: float = Field(..., description="Average response latency")


# ==============================================================================
# RAG-SPECIFIC MESSAGES (Phase 3)
# ==============================================================================

class RAGSourcesMessage(WebSocketMessage):
    """RAG sources message - sent before streaming tokens"""
    type: Literal["rag_sources"] = "rag_sources"
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    sources: list = Field(..., description="List of source references")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")


class RAGTokenMessage(WebSocketMessage):
    """RAG token stream message"""
    type: Literal["rag_token"] = "rag_token"
    token: str = Field(..., description="Generated token")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    has_sources: bool = Field(default=False, description="Whether sources were used")


class RAGCompleteMessage(WebSocketMessage):
    """RAG stream completion message"""
    type: Literal["rag_complete"] = "rag_complete"
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    total_tokens: int = Field(..., description="Total tokens generated")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    total_time_ms: float = Field(..., description="Total time in milliseconds")
    sources_count: int = Field(..., description="Number of sources used")

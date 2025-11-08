"""
Conversation Management Models

Data models for conversation history and message management.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Individual message in a conversation"""

    id: str = Field(..., description="Unique message ID")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    sources: Optional[List[Dict[str, Any]]] = Field(
        None, description="RAG sources if applicable"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata (timing, model, etc.)"
    )


class Conversation(BaseModel):
    """Complete conversation with messages"""

    id: str = Field(..., description="Unique conversation ID")
    client_id: str = Field(..., description="Client who owns this conversation")
    messages: List[ConversationMessage] = Field(
        default_factory=list, description="List of messages"
    )
    created_at: str = Field(..., description="Conversation creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    title: Optional[str] = Field(None, description="Conversation title")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )


class ConversationListResponse(BaseModel):
    """Response for listing conversations"""

    conversations: List[Conversation] = Field(
        default_factory=list, description="List of conversations"
    )
    total: int = Field(..., description="Total number of conversations")
    limit: int = Field(..., description="Number of results returned")


class SaveMessageRequest(BaseModel):
    """Request to save a message to a conversation"""

    conversation_id: str = Field(..., description="Conversation ID")
    message: ConversationMessage = Field(..., description="Message to save")


class SaveMessageResponse(BaseModel):
    """Response after saving a message"""

    success: bool = Field(..., description="Whether save was successful")
    message_id: str = Field(..., description="Saved message ID")
    conversation_id: str = Field(..., description="Conversation ID")


class DeleteConversationResponse(BaseModel):
    """Response after deleting a conversation"""

    success: bool = Field(..., description="Whether deletion was successful")
    conversation_id: str = Field(..., description="Deleted conversation ID")


class ConversationStatsResponse(BaseModel):
    """Statistics for conversations"""

    total_conversations: int = Field(..., description="Total conversations")
    total_messages: int = Field(..., description="Total messages across all conversations")
    average_messages_per_conversation: float = Field(
        ..., description="Average messages per conversation"
    )
    oldest_conversation: Optional[str] = Field(
        None, description="Timestamp of oldest conversation"
    )
    newest_conversation: Optional[str] = Field(
        None, description="Timestamp of newest conversation"
    )

"""
Conversation Management API

REST endpoints for conversation history management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from app.core.security import get_current_client_id
from app.services.conversation_service import conversation_service
from app.models.conversation import (
    Conversation,
    ConversationListResponse,
    SaveMessageRequest,
    SaveMessageResponse,
    DeleteConversationResponse,
    ConversationStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.get("/health")
async def health_check():
    """
    Check conversation service health.

    Returns:
        Service health status
    """
    health = conversation_service.health_check()
    status_code = 200 if health['status'] == 'healthy' else 503

    return JSONResponse(
        content=health,
        status_code=status_code
    )


@router.get("/stats")
async def get_stats() -> ConversationStatsResponse:
    """
    Get conversation statistics.

    Returns:
        Conversation statistics
    """
    try:
        stats = await conversation_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_conversations(
    limit: int = Query(50, ge=1, le=100, description="Maximum conversations to return"),
    client_id: Optional[str] = Depends(get_current_client_id)
) -> ConversationListResponse:
    """
    List conversations for the authenticated client.

    Args:
        limit: Maximum number of conversations to return
        client_id: Authenticated client ID

    Returns:
        List of conversations
    """
    try:
        conversations = await conversation_service.list_conversations(
            client_id=client_id,
            limit=limit
        )

        return ConversationListResponse(
            conversations=conversations,
            total=len(conversations),
            limit=limit
        )

    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    client_id: Optional[str] = Depends(get_current_client_id)
) -> Conversation:
    """
    Get a specific conversation by ID.

    Args:
        conversation_id: Conversation ID
        client_id: Authenticated client ID

    Returns:
        Conversation details
    """
    try:
        conversation = await conversation_service.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Verify ownership
        if client_id and conversation.client_id != client_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    client_id: Optional[str] = Depends(get_current_client_id)
) -> DeleteConversationResponse:
    """
    Delete a conversation.

    Args:
        conversation_id: Conversation ID to delete
        client_id: Authenticated client ID

    Returns:
        Deletion confirmation
    """
    try:
        # Verify ownership before deletion
        conversation = await conversation_service.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if client_id and conversation.client_id != client_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Delete conversation
        success = await conversation_service.delete_conversation(conversation_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete conversation")

        return DeleteConversationResponse(
            success=True,
            conversation_id=conversation_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message")
async def save_message(
    request: SaveMessageRequest,
    client_id: Optional[str] = Depends(get_current_client_id)
) -> SaveMessageResponse:
    """
    Save a message to a conversation.

    Args:
        request: Message save request
        client_id: Authenticated client ID

    Returns:
        Save confirmation
    """
    try:
        if not client_id:
            raise HTTPException(status_code=401, detail="Authentication required")

        success = await conversation_service.add_message(
            conversation_id=request.conversation_id,
            message=request.message,
            client_id=client_id
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save message")

        return SaveMessageResponse(
            success=True,
            message_id=request.message.id,
            conversation_id=request.conversation_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

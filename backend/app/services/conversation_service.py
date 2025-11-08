"""
Conversation Management Service

Manages conversation history storage and retrieval using ChromaDB.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationListResponse,
    ConversationStatsResponse,
)

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Service for managing conversation history in ChromaDB.
    """

    def __init__(self):
        self.client: Optional[chromadb.HttpClient] = None
        self.collection = None
        self.collection_name = settings.CHROMA_CONVERSATION_COLLECTION
        self.max_messages_per_conversation = 100  # Trim old messages

    async def connect(self) -> bool:
        """
        Connect to ChromaDB and get/create conversation collection.

        Returns:
            True if successful
        """
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST.replace('http://', '').replace(':8000', ''),
                port=8000,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "T.A.R.S. conversation history"},
            )

            logger.info(f"Connected to conversation collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB for conversations: {e}")
            return False

    async def save_conversation(self, conversation: Conversation) -> bool:
        """
        Save or update a complete conversation.

        Args:
            conversation: Conversation object to save

        Returns:
            True if successful
        """
        try:
            if not self.collection:
                await self.connect()

            # Serialize conversation to JSON
            conversation_data = conversation.model_dump_json()

            # Create metadata for ChromaDB
            metadata = {
                'client_id': conversation.client_id,
                'created_at': conversation.created_at,
                'updated_at': conversation.updated_at,
                'message_count': len(conversation.messages),
                'title': conversation.title or 'Untitled',
            }

            # Upsert to ChromaDB
            self.collection.upsert(
                ids=[conversation.id],
                documents=[conversation_data],
                metadatas=[metadata],
            )

            logger.info(f"Saved conversation {conversation.id} with {len(conversation.messages)} messages")
            return True

        except Exception as e:
            logger.error(f"Error saving conversation {conversation.id}: {e}", exc_info=True)
            return False

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation object or None if not found
        """
        try:
            if not self.collection:
                await self.connect()

            result = self.collection.get(
                ids=[conversation_id],
                include=['documents', 'metadatas']
            )

            if not result['ids']:
                return None

            # Deserialize conversation
            conversation_json = result['documents'][0]
            conversation = Conversation.model_validate_json(conversation_json)

            return conversation

        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return None

    async def list_conversations(
        self,
        client_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Conversation]:
        """
        List conversations, optionally filtered by client_id.

        Args:
            client_id: Filter by client ID (optional)
            limit: Maximum number of conversations to return

        Returns:
            List of conversations
        """
        try:
            if not self.collection:
                await self.connect()

            # Build query
            where = {'client_id': client_id} if client_id else None

            result = self.collection.get(
                where=where,
                limit=limit,
                include=['documents', 'metadatas']
            )

            conversations = []
            for doc in result['documents']:
                try:
                    conv = Conversation.model_validate_json(doc)
                    conversations.append(conv)
                except Exception as e:
                    logger.error(f"Error parsing conversation: {e}")

            # Sort by updated_at descending
            conversations.sort(
                key=lambda c: c.updated_at,
                reverse=True
            )

            return conversations

        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if successful
        """
        try:
            if not self.collection:
                await self.connect()

            self.collection.delete(ids=[conversation_id])

            logger.info(f"Deleted conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False

    async def add_message(
        self,
        conversation_id: str,
        message: ConversationMessage,
        client_id: str
    ) -> bool:
        """
        Add a message to an existing conversation or create new one.

        Args:
            conversation_id: Conversation ID
            message: Message to add
            client_id: Client ID who owns the conversation

        Returns:
            True if successful
        """
        try:
            # Try to get existing conversation
            conversation = await self.get_conversation(conversation_id)

            if conversation:
                # Add message to existing conversation
                conversation.messages.append(message)
                conversation.updated_at = datetime.now().isoformat()

                # Trim old messages if exceeds limit
                if len(conversation.messages) > self.max_messages_per_conversation:
                    conversation.messages = conversation.messages[
                        -self.max_messages_per_conversation:
                    ]

            else:
                # Create new conversation
                now = datetime.now().isoformat()
                conversation = Conversation(
                    id=conversation_id,
                    client_id=client_id,
                    messages=[message],
                    created_at=now,
                    updated_at=now,
                    title=self._generate_title(message.content),
                )

            # Save conversation
            return await self.save_conversation(conversation)

        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id}: {e}")
            return False

    def _generate_title(self, first_message: str) -> str:
        """
        Generate a title from the first message.

        Args:
            first_message: First message content

        Returns:
            Generated title
        """
        # Take first 50 characters
        title = first_message[:50]
        if len(first_message) > 50:
            title += '...'
        return title

    async def get_stats(self) -> ConversationStatsResponse:
        """
        Get conversation statistics.

        Returns:
            Statistics response
        """
        try:
            if not self.collection:
                await self.connect()

            # Get all conversations
            result = self.collection.get(include=['documents', 'metadatas'])

            total_conversations = len(result['ids'])
            total_messages = 0
            timestamps = []

            for doc in result['documents']:
                try:
                    conv = Conversation.model_validate_json(doc)
                    total_messages += len(conv.messages)
                    timestamps.append(conv.created_at)
                except Exception:
                    pass

            avg_messages = total_messages / total_conversations if total_conversations > 0 else 0

            return ConversationStatsResponse(
                total_conversations=total_conversations,
                total_messages=total_messages,
                average_messages_per_conversation=avg_messages,
                oldest_conversation=min(timestamps) if timestamps else None,
                newest_conversation=max(timestamps) if timestamps else None,
            )

        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return ConversationStatsResponse(
                total_conversations=0,
                total_messages=0,
                average_messages_per_conversation=0.0,
            )

    def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status dictionary
        """
        try:
            connected = self.client is not None and self.collection is not None

            return {
                'status': 'healthy' if connected else 'unhealthy',
                'connected': connected,
                'collection': self.collection_name if connected else None,
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
            }


# Global service instance
conversation_service = ConversationService()

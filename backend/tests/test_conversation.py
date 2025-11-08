"""
Tests for Conversation Service

Basic tests for conversation management functionality.
"""

import pytest
from datetime import datetime

from app.models.conversation import (
    Conversation,
    ConversationMessage,
)
from app.services.conversation_service import ConversationService


@pytest.fixture
def conversation_service():
    """Create a conversation service instance for testing"""
    return ConversationService()


@pytest.fixture
def sample_message():
    """Create a sample message for testing"""
    return ConversationMessage(
        id="msg_test_123",
        role="user",
        content="Hello, this is a test message",
        timestamp=datetime.now().isoformat(),
    )


@pytest.fixture
def sample_conversation(sample_message):
    """Create a sample conversation for testing"""
    now = datetime.now().isoformat()
    return Conversation(
        id="conv_test_123",
        client_id="test_client",
        messages=[sample_message],
        created_at=now,
        updated_at=now,
        title="Test Conversation",
    )


class TestConversationModels:
    """Test conversation data models"""

    def test_conversation_message_creation(self, sample_message):
        """Test creating a conversation message"""
        assert sample_message.id == "msg_test_123"
        assert sample_message.role == "user"
        assert sample_message.content == "Hello, this is a test message"

    def test_conversation_creation(self, sample_conversation):
        """Test creating a conversation"""
        assert sample_conversation.id == "conv_test_123"
        assert sample_conversation.client_id == "test_client"
        assert len(sample_conversation.messages) == 1
        assert sample_conversation.title == "Test Conversation"


class TestConversationService:
    """Test conversation service methods"""

    def test_service_initialization(self, conversation_service):
        """Test service can be initialized"""
        assert conversation_service.collection_name == "tars_conversations"
        assert conversation_service.max_messages_per_conversation == 100

    def test_generate_title(self, conversation_service):
        """Test title generation from message"""
        short_message = "Hello"
        title = conversation_service._generate_title(short_message)
        assert title == "Hello"

        long_message = "This is a very long message that exceeds fifty characters in total length"
        title = conversation_service._generate_title(long_message)
        assert len(title) <= 53  # 50 chars + "..."
        assert title.endswith("...")

    def test_health_check(self, conversation_service):
        """Test health check method"""
        health = conversation_service.health_check()
        assert 'status' in health
        assert 'connected' in health


@pytest.mark.asyncio
class TestConversationServiceAsync:
    """Test async conversation service methods"""

    @pytest.mark.skip(reason="Requires ChromaDB connection")
    async def test_save_conversation(self, conversation_service, sample_conversation):
        """Test saving a conversation"""
        # This would require a real ChromaDB instance
        success = await conversation_service.save_conversation(sample_conversation)
        assert isinstance(success, bool)

    @pytest.mark.skip(reason="Requires ChromaDB connection")
    async def test_get_conversation(self, conversation_service):
        """Test retrieving a conversation"""
        # This would require a real ChromaDB instance
        conversation = await conversation_service.get_conversation("conv_test_123")
        # Will be None if not found or not connected
        assert conversation is None or isinstance(conversation, Conversation)

    @pytest.mark.skip(reason="Requires ChromaDB connection")
    async def test_add_message(self, conversation_service, sample_message):
        """Test adding a message to a conversation"""
        # This would require a real ChromaDB instance
        success = await conversation_service.add_message(
            conversation_id="conv_test_123",
            message=sample_message,
            client_id="test_client"
        )
        assert isinstance(success, bool)

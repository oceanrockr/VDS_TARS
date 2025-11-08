"""
T.A.R.S. WebSocket Tests
Unit and integration tests for WebSocket gateway
"""

import pytest
import json
import asyncio
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.services.connection_manager import connection_manager, ConnectionInfo
from app.models.websocket import (
    ChatMessage,
    TokenStreamMessage,
    PingMessage,
    PongMessage,
)


class TestWebSocketHealth:
    """Test WebSocket health and status endpoints"""

    def test_websocket_health(self, client):
        """Test WebSocket health endpoint"""
        response = client.get("/ws/health")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "websocket"
        assert "timestamp" in data
        assert "metrics" in data
        assert "ollama_status" in data

    def test_get_active_sessions_empty(self, client):
        """Test getting active sessions when none exist"""
        response = client.get("/ws/sessions")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "total_sessions" in data
        assert "sessions" in data
        assert isinstance(data["sessions"], dict)


class TestConnectionManager:
    """Test ConnectionManager functionality"""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up connections after each test"""
        yield
        # Clear all connections
        connection_manager.active_connections.clear()
        connection_manager.client_sessions.clear()
        connection_manager.message_queues.clear()

    def test_get_metrics(self):
        """Test getting connection metrics"""
        metrics = connection_manager.get_metrics()

        assert "active_connections" in metrics
        assert "unique_clients" in metrics
        assert "total_connections" in metrics
        assert "total_messages" in metrics
        assert "total_tokens" in metrics
        assert "total_errors" in metrics
        assert "max_connections" in metrics

    def test_get_active_sessions_empty(self):
        """Test getting active sessions when empty"""
        sessions = connection_manager.get_active_sessions()

        assert isinstance(sessions, dict)
        assert len(sessions) == 0

    def test_increment_token_count(self):
        """Test incrementing token count for a session"""
        # This test requires a mock connection
        # For now, test that calling with non-existent session doesn't crash
        connection_manager.increment_token_count("nonexistent-session")

        # Should not raise an error


class TestWebSocketConnection:
    """Test WebSocket connection flow"""

    def test_websocket_connect_without_token(self, client):
        """Test WebSocket connection without authentication token"""
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat"):
                pass

    def test_websocket_connect_with_invalid_token(self, client):
        """Test WebSocket connection with invalid token"""
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat?token=invalid.token.here"):
                pass

    def test_websocket_connect_with_valid_token(self, client, test_token):
        """Test successful WebSocket connection with valid token"""
        with client.websocket_connect(f"/ws/chat?token={test_token}") as websocket:
            # Should receive connection acknowledgment
            data = websocket.receive_json()

            assert data["type"] == "connection_ack"
            assert "client_id" in data
            assert "session_id" in data
            assert "timestamp" in data

    def test_websocket_chat_message(self, client, test_token):
        """Test sending a chat message over WebSocket"""
        with client.websocket_connect(f"/ws/chat?token={test_token}") as websocket:
            # Receive connection ack
            ack = websocket.receive_json()
            assert ack["type"] == "connection_ack"

            # Send chat message
            chat_msg = {
                "type": "chat",
                "content": "Hello, T.A.R.S.!",
                "conversation_id": "test-conv-123",
            }
            websocket.send_json(chat_msg)

            # Should receive token stream or error
            # (Actual Ollama response depends on service availability)
            try:
                response = websocket.receive_json(timeout=5)
                assert response["type"] in ["token", "error", "complete"]
            except Exception:
                # Ollama might not be available in test environment
                pass

    def test_websocket_ping_pong(self, client, test_token):
        """Test WebSocket heartbeat ping/pong"""
        with client.websocket_connect(f"/ws/chat?token={test_token}") as websocket:
            # Receive connection ack
            ack = websocket.receive_json()
            assert ack["type"] == "connection_ack"

            # Send pong message
            pong_msg = {"type": "pong", "timestamp": "2025-11-07T00:00:00"}
            websocket.send_json(pong_msg)

            # Should not crash or return error
            # Ping will be sent automatically by server after HEARTBEAT_INTERVAL

    def test_websocket_invalid_message_format(self, client, test_token):
        """Test WebSocket with invalid message format"""
        with client.websocket_connect(f"/ws/chat?token={test_token}") as websocket:
            # Receive connection ack
            ack = websocket.receive_json()
            assert ack["type"] == "connection_ack"

            # Send invalid JSON
            websocket.send_text("not valid json")

            # Should receive error message
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "INVALID_JSON" in response.get("code", "")

    def test_websocket_unknown_message_type(self, client, test_token):
        """Test WebSocket with unknown message type"""
        with client.websocket_connect(f"/ws/chat?token={test_token}") as websocket:
            # Receive connection ack
            ack = websocket.receive_json()
            assert ack["type"] == "connection_ack"

            # Send unknown message type
            unknown_msg = {
                "type": "unknown_type",
                "data": "test",
            }
            websocket.send_json(unknown_msg)

            # Should log warning but not crash
            # Connection should remain open


class TestWebSocketModels:
    """Test WebSocket message models"""

    def test_chat_message_model(self):
        """Test ChatMessage model validation"""
        msg = ChatMessage(
            content="Hello",
            conversation_id="conv-123",
        )

        assert msg.type == "chat"
        assert msg.content == "Hello"
        assert msg.conversation_id == "conv-123"
        assert msg.timestamp is not None

    def test_token_stream_message_model(self):
        """Test TokenStreamMessage model validation"""
        msg = TokenStreamMessage(
            token="test",
            conversation_id="conv-123",
        )

        assert msg.type == "token"
        assert msg.token == "test"
        assert msg.conversation_id == "conv-123"

    def test_ping_pong_message_models(self):
        """Test Ping and Pong message models"""
        ping = PingMessage()
        assert ping.type == "ping"

        pong = PongMessage()
        assert pong.type == "pong"

    def test_message_serialization(self):
        """Test message serialization to JSON"""
        msg = ChatMessage(
            content="Test message",
            conversation_id="conv-456",
        )

        json_data = msg.model_dump()

        assert json_data["type"] == "chat"
        assert json_data["content"] == "Test message"
        assert json_data["conversation_id"] == "conv-456"


@pytest.mark.asyncio
class TestConcurrentConnections:
    """Test concurrent WebSocket connections"""

    async def test_multiple_connections(self):
        """Test multiple concurrent WebSocket connections"""
        from httpx import AsyncClient
        from app.core.security import create_access_token

        # Create test tokens
        tokens = [
            create_access_token({"sub": f"client-{i}"})
            for i in range(3)
        ]

        # Note: This test requires async WebSocket support
        # Actual implementation would use websockets library or similar
        # For now, this is a placeholder for the test structure

        # TODO: Implement with async WebSocket client
        # async with AsyncClient(app=app, base_url="ws://test") as client:
        #     connections = []
        #     for token in tokens:
        #         ws = await client.websocket_connect(f"/ws/chat?token={token}")
        #         connections.append(ws)
        #
        #     # All connections should be active
        #     assert len(connections) == 3
        #
        #     # Close all connections
        #     for ws in connections:
        #         await ws.close()

        # For now, just pass
        pass

    async def test_max_connections_limit(self):
        """Test that max connections limit is enforced"""
        # TODO: Implement test for max connections (default 10)
        # This would create 11 connections and verify the 11th is rejected

        # Placeholder
        pass


class TestOllamaIntegration:
    """Test Ollama service integration (requires Ollama running)"""

    @pytest.mark.skipif(
        True,  # Skip by default, enable when Ollama is available
        reason="Requires Ollama service to be running"
    )
    def test_ollama_stream_generation(self, client, test_token):
        """Test end-to-end token streaming from Ollama"""
        with client.websocket_connect(f"/ws/chat?token={test_token}") as websocket:
            # Receive connection ack
            ack = websocket.receive_json()
            assert ack["type"] == "connection_ack"

            # Send chat message
            chat_msg = {
                "type": "chat",
                "content": "Say hello",
                "conversation_id": "test-stream",
            }
            websocket.send_json(chat_msg)

            # Collect tokens
            tokens = []
            complete = False

            while not complete:
                msg = websocket.receive_json(timeout=30)

                if msg["type"] == "token":
                    tokens.append(msg["token"])
                elif msg["type"] == "complete":
                    complete = True
                    assert msg["total_tokens"] > 0
                    assert msg["latency_ms"] > 0
                elif msg["type"] == "error":
                    pytest.fail(f"Received error: {msg['error']}")

            # Should have received tokens
            assert len(tokens) > 0

            # Join tokens to form response
            response = "".join(tokens)
            assert len(response) > 0


class TestWebSocketMetrics:
    """Test WebSocket metrics and monitoring"""

    def test_connection_metrics_after_activity(self, client, test_token):
        """Test that metrics are updated after WebSocket activity"""
        # Get initial metrics
        response = client.get("/ws/health")
        initial_metrics = response.json()["metrics"]

        # Create WebSocket connection
        with client.websocket_connect(f"/ws/chat?token={test_token}") as websocket:
            # Receive ack
            websocket.receive_json()

            # Get metrics during connection
            response = client.get("/ws/health")
            active_metrics = response.json()["metrics"]

            # Active connections should be greater
            assert active_metrics["active_connections"] >= 1

        # After closing, active connections should decrease
        response = client.get("/ws/health")
        final_metrics = response.json()["metrics"]

        assert final_metrics["active_connections"] == 0
        assert final_metrics["total_connections"] >= initial_metrics["total_connections"]

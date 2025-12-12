"""
WebSocket Reconnection Tests

TARS-1001: WebSocket Reconnection Issue Test Suite
-------------------------------------------------
Validates:
- Heartbeat ping/pong mechanism
- Exponential backoff reconnection
- Auto-resubscription after reconnect
- Silent disconnect detection
- Connection state management
- Message replay correctness

Author: T.A.R.S. Engineering Team
Version: 1.0.1
Date: 2025-11-20
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
from websockets.server import serve, WebSocketServerProtocol

from websocket_client_patch import (
    ConnectionState,
    HeartbeatConfig,
    ReconnectionConfig,
    ReconnectingWebSocketClient,
    WebSocketMessage,
)

logger = logging.getLogger(__name__)


# Mock WebSocket server for testing
class MockWebSocketServer:
    """
    Mock WebSocket server for testing

    Features:
    - Simulate connection drops
    - Heartbeat ping/pong
    - Channel subscription tracking
    - Message broadcast
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.server = None
        self.clients: List[WebSocketServerProtocol] = []
        self.subscriptions: dict = {}  # client -> set of channels
        self.message_log: List[dict] = []
        self.should_disconnect_next: bool = False
        self.disconnect_after_messages: int = 0
        self.message_count: int = 0

    async def start(self):
        """Start mock server"""
        self.server = await serve(self._handle_client, self.host, self.port)
        logger.info(f"Mock server started at {self.uri}")

    async def stop(self):
        """Stop mock server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("Mock server stopped")

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle client connection"""
        self.clients.append(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"Client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                self.message_count += 1

                # Handle heartbeat ping
                if message == "ping":
                    await websocket.send("pong")
                    logger.debug("Sent pong")
                    continue

                # Parse JSON message
                try:
                    data = json.loads(message)
                    self.message_log.append(
                        {
                            "timestamp": time.time(),
                            "client": websocket.remote_address,
                            "data": data,
                        }
                    )

                    # Handle subscription
                    if data.get("type") == "subscribe":
                        channel = data.get("channel")
                        self.subscriptions[websocket].add(channel)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "subscribed",
                                    "channel": channel,
                                    "timestamp": time.time(),
                                }
                            )
                        )
                        logger.info(f"Client subscribed to {channel}")

                    # Handle unsubscription
                    elif data.get("type") == "unsubscribe":
                        channel = data.get("channel")
                        self.subscriptions[websocket].discard(channel)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "unsubscribed",
                                    "channel": channel,
                                    "timestamp": time.time(),
                                }
                            )
                        )
                        logger.info(f"Client unsubscribed from {channel}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message}")

                # Simulate disconnect
                if self.should_disconnect_next:
                    logger.info("Simulating disconnect")
                    await websocket.close()
                    break

                if (
                    self.disconnect_after_messages > 0
                    and self.message_count >= self.disconnect_after_messages
                ):
                    logger.info(
                        f"Simulating disconnect after {self.message_count} messages"
                    )
                    await websocket.close()
                    break

        except Exception as e:
            logger.error(f"Client error: {e}", exc_info=True)
        finally:
            self.clients.remove(websocket)
            del self.subscriptions[websocket]
            logger.info(f"Client disconnected: {websocket.remote_address}")

    async def broadcast(self, channel: str, payload: dict):
        """Broadcast message to all clients subscribed to channel"""
        message = json.dumps(
            {"type": "message", "channel": channel, "payload": payload}
        )

        for client, channels in self.subscriptions.items():
            if channel in channels:
                try:
                    await client.send(message)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")

    def force_disconnect_next(self):
        """Force disconnect on next message"""
        self.should_disconnect_next = True

    def force_disconnect_after(self, count: int):
        """Force disconnect after N messages"""
        self.disconnect_after_messages = count
        self.message_count = 0


# Fixtures
@pytest.fixture
async def mock_server():
    """Fixture for mock WebSocket server"""
    server = MockWebSocketServer()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def reconnect_config():
    """Fast reconnection config for testing"""
    return ReconnectionConfig(
        enabled=True,
        initial_delay_seconds=0.1,
        max_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_attempts=None,
        jitter=False,  # Disable jitter for deterministic tests
    )


@pytest.fixture
def heartbeat_config():
    """Fast heartbeat config for testing"""
    return HeartbeatConfig(
        enabled=True,
        interval_seconds=1.0,
        timeout_seconds=0.5,
        max_missed_heartbeats=2,
    )


# Tests
@pytest.mark.asyncio
async def test_basic_connection(mock_server, reconnect_config, heartbeat_config):
    """Test basic connection establishment"""
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    # Track state changes
    states = []
    client.on_state_change(lambda s: states.append(s))

    # Connect
    await client.connect()
    await asyncio.sleep(0.5)

    # Verify connection
    assert client.is_connected
    assert client.state == ConnectionState.CONNECTED
    assert ConnectionState.CONNECTING in states
    assert ConnectionState.CONNECTED in states

    # Cleanup
    await client.close()
    assert client.state == ConnectionState.CLOSED


@pytest.mark.asyncio
async def test_heartbeat_mechanism(mock_server, reconnect_config, heartbeat_config):
    """Test heartbeat ping/pong mechanism"""
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    await client.connect()
    await asyncio.sleep(0.5)

    # Wait for multiple heartbeats
    initial_pong_time = client._last_pong_time
    await asyncio.sleep(2.5)  # Should trigger 2 heartbeats

    # Verify heartbeats occurred
    assert client._last_pong_time > initial_pong_time
    assert client._missed_heartbeats == 0
    assert client.is_connected

    await client.close()


@pytest.mark.asyncio
async def test_subscription_management(mock_server, reconnect_config, heartbeat_config):
    """Test channel subscription and unsubscription"""
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    await client.connect()
    await asyncio.sleep(0.5)

    # Subscribe to channels
    await client.subscribe("channel1")
    await client.subscribe("channel2")
    await asyncio.sleep(0.2)

    # Verify subscriptions
    assert "channel1" in client.subscribed_channels
    assert "channel2" in client.subscribed_channels

    # Verify server received subscriptions
    channel1_found = any(
        msg["data"].get("channel") == "channel1" for msg in mock_server.message_log
    )
    channel2_found = any(
        msg["data"].get("channel") == "channel2" for msg in mock_server.message_log
    )
    assert channel1_found
    assert channel2_found

    # Unsubscribe
    await client.unsubscribe("channel1")
    await asyncio.sleep(0.2)

    assert "channel1" not in client.subscribed_channels
    assert "channel2" in client.subscribed_channels

    await client.close()


@pytest.mark.asyncio
async def test_message_receiving(mock_server, reconnect_config, heartbeat_config):
    """Test message receiving and callbacks"""
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    # Track received messages
    received_messages = []
    client.on_message(lambda msg: received_messages.append(msg))

    await client.connect()
    await client.subscribe("test-channel")
    await asyncio.sleep(0.5)

    # Broadcast message from server
    await mock_server.broadcast(
        "test-channel", {"data": "test-payload", "count": 1}
    )
    await asyncio.sleep(0.5)

    # Verify message received
    assert len(received_messages) >= 1
    msg = next(
        (m for m in received_messages if m.type == "message"), None
    )
    assert msg is not None
    assert msg.channel == "test-channel"
    assert msg.payload["data"] == "test-payload"

    await client.close()


@pytest.mark.asyncio
async def test_reconnection_after_disconnect(
    mock_server, reconnect_config, heartbeat_config
):
    """
    Test automatic reconnection after disconnect

    This is the core test for TARS-1001
    """
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    # Track state changes
    states = []
    client.on_state_change(lambda s: states.append(s))

    # Connect
    await client.connect()
    await asyncio.sleep(0.5)
    assert client.is_connected

    # Force disconnect
    mock_server.force_disconnect_next()
    await client.send({"type": "test", "trigger": "disconnect"})
    await asyncio.sleep(0.5)

    # Verify disconnection detected
    assert ConnectionState.DISCONNECTED in states or ConnectionState.RECONNECTING in states

    # Wait for reconnection (should happen within 30s per requirement)
    reconnect_start = time.time()
    max_wait = 30.0
    while time.time() - reconnect_start < max_wait:
        if client.is_connected:
            break
        await asyncio.sleep(0.1)

    reconnect_time = time.time() - reconnect_start

    # Verify reconnection
    assert client.is_connected, f"Failed to reconnect within {max_wait}s"
    assert reconnect_time < 30.0, f"Reconnection took {reconnect_time:.1f}s (requirement: <30s)"
    assert ConnectionState.CONNECTED in states[-2:]

    logger.info(f"✓ Reconnection successful in {reconnect_time:.2f}s")

    await client.close()


@pytest.mark.asyncio
async def test_auto_resubscription(mock_server, reconnect_config, heartbeat_config):
    """
    Test automatic resubscription after reconnect

    Validates that channels are automatically resubscribed after disconnect
    """
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    # Connect and subscribe
    await client.connect()
    await client.subscribe("channel1")
    await client.subscribe("channel2")
    await asyncio.sleep(0.5)

    initial_sub_count = len(
        [msg for msg in mock_server.message_log if msg["data"].get("type") == "subscribe"]
    )
    assert initial_sub_count == 2

    # Force disconnect
    mock_server.force_disconnect_after(1)
    await client.send({"type": "test"})
    await asyncio.sleep(0.5)

    # Wait for reconnection
    max_wait = 5.0
    reconnect_start = time.time()
    while time.time() - reconnect_start < max_wait:
        if client.is_connected:
            break
        await asyncio.sleep(0.1)

    assert client.is_connected
    await asyncio.sleep(0.5)

    # Verify resubscription
    final_sub_count = len(
        [msg for msg in mock_server.message_log if msg["data"].get("type") == "subscribe"]
    )
    assert final_sub_count == 4, "Should have 2 initial + 2 resubscriptions"

    # Verify channels still tracked
    assert "channel1" in client.subscribed_channels
    assert "channel2" in client.subscribed_channels

    logger.info("✓ Auto-resubscription successful")

    await client.close()


@pytest.mark.asyncio
async def test_exponential_backoff(mock_server, reconnect_config):
    """Test exponential backoff reconnection timing"""
    # Configure specific backoff parameters
    config = ReconnectionConfig(
        enabled=True,
        initial_delay_seconds=0.5,
        max_delay_seconds=4.0,
        backoff_multiplier=2.0,
        max_attempts=None,
        jitter=False,
    )

    # Create server that immediately closes connections
    failing_server = MockWebSocketServer(port=8766)
    await failing_server.start()

    client = ReconnectingWebSocketClient(
        uri=failing_server.uri,
        reconnect_config=config,
        heartbeat_config=HeartbeatConfig(enabled=False),
    )

    # Track reconnection attempts
    attempt_times = []

    original_do_connect = client._do_connect

    async def track_attempts():
        attempt_times.append(time.time())
        await original_do_connect()

    client._do_connect = track_attempts

    # Trigger connection (will fail)
    await failing_server.stop()  # Stop server to force failures
    await client.connect()
    await asyncio.sleep(0.1)

    # Wait for multiple attempts
    await asyncio.sleep(5.0)

    # Verify exponential backoff
    assert len(attempt_times) >= 3

    # Calculate delays between attempts
    delays = [attempt_times[i] - attempt_times[i - 1] for i in range(1, len(attempt_times))]

    # Verify delays follow exponential pattern
    # First delay: ~0.5s, second: ~1.0s, third: ~2.0s, fourth: ~4.0s (capped)
    assert 0.4 < delays[0] < 0.7, f"First delay {delays[0]:.2f}s should be ~0.5s"
    assert 0.9 < delays[1] < 1.3, f"Second delay {delays[1]:.2f}s should be ~1.0s"

    if len(delays) >= 3:
        assert 1.8 < delays[2] < 2.5, f"Third delay {delays[2]:.2f}s should be ~2.0s"

    logger.info(f"✓ Exponential backoff validated: {[f'{d:.2f}s' for d in delays]}")

    await client.close()


@pytest.mark.asyncio
async def test_silent_disconnect_detection(mock_server, reconnect_config):
    """
    Test detection of silent disconnects via heartbeat timeout

    Simulates network issue where connection appears open but server isn't responding
    """
    # Configure aggressive heartbeat for faster testing
    heartbeat_config = HeartbeatConfig(
        enabled=True,
        interval_seconds=0.5,
        timeout_seconds=0.3,
        max_missed_heartbeats=2,
    )

    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    states = []
    client.on_state_change(lambda s: states.append(s))

    await client.connect()
    await asyncio.sleep(0.5)
    assert client.is_connected

    # Simulate silent disconnect by preventing pong responses
    # (in real scenario, we'd patch the websocket to drop pongs)
    client._last_pong_time = time.time() - 10.0  # Pretend last pong was 10s ago

    # Wait for heartbeat to detect silent disconnect
    await asyncio.sleep(2.0)

    # Verify disconnect was detected
    assert ConnectionState.DISCONNECTED in states or ConnectionState.RECONNECTING in states

    logger.info("✓ Silent disconnect detected via heartbeat timeout")

    await client.close()


@pytest.mark.asyncio
async def test_message_replay_after_reconnect(mock_server, reconnect_config, heartbeat_config):
    """
    Test message continuity after reconnect

    Validates no messages are lost or duplicated during reconnection
    """
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    received_messages = []
    client.on_message(lambda msg: received_messages.append(msg))

    await client.connect()
    await client.subscribe("test-channel")
    await asyncio.sleep(0.5)

    # Send messages before disconnect
    await mock_server.broadcast("test-channel", {"seq": 1})
    await asyncio.sleep(0.2)

    # Force disconnect
    mock_server.force_disconnect_after(1)
    await client.send({"type": "test"})
    await asyncio.sleep(0.5)

    # Wait for reconnection
    max_wait = 5.0
    reconnect_start = time.time()
    while time.time() - reconnect_start < max_wait:
        if client.is_connected:
            break
        await asyncio.sleep(0.1)

    assert client.is_connected
    await asyncio.sleep(0.5)

    # Send messages after reconnect
    await mock_server.broadcast("test-channel", {"seq": 2})
    await asyncio.sleep(0.5)

    # Verify both messages received
    message_msgs = [m for m in received_messages if m.type == "message"]
    sequences = [m.payload.get("seq") for m in message_msgs if m.payload]

    assert 1 in sequences, "Message before disconnect should be received"
    assert 2 in sequences, "Message after reconnect should be received"
    assert len([s for s in sequences if s == 1]) == 1, "No duplicates"
    assert len([s for s in sequences if s == 2]) == 1, "No duplicates"

    logger.info("✓ Message continuity validated")

    await client.close()


@pytest.mark.asyncio
async def test_concurrent_reconnections(mock_server, reconnect_config, heartbeat_config):
    """Test multiple clients reconnecting simultaneously"""
    clients = []
    for i in range(3):
        client = ReconnectingWebSocketClient(
            uri=mock_server.uri,
            reconnect_config=reconnect_config,
            heartbeat_config=heartbeat_config,
        )
        await client.connect()
        await client.subscribe(f"channel-{i}")
        clients.append(client)

    await asyncio.sleep(0.5)

    # All connected
    assert all(c.is_connected for c in clients)

    # Force disconnect all
    mock_server.force_disconnect_after(1)
    for client in clients:
        await client.send({"type": "test"})

    await asyncio.sleep(0.5)

    # Wait for all to reconnect
    max_wait = 10.0
    start = time.time()
    while time.time() - start < max_wait:
        if all(c.is_connected for c in clients):
            break
        await asyncio.sleep(0.1)

    # Verify all reconnected
    assert all(c.is_connected for c in clients), "All clients should reconnect"

    logger.info("✓ Concurrent reconnections successful")

    # Cleanup
    for client in clients:
        await client.close()


@pytest.mark.asyncio
async def test_graceful_shutdown_during_reconnect(mock_server, reconnect_config):
    """Test graceful shutdown while reconnection is in progress"""
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=HeartbeatConfig(enabled=False),
    )

    await client.connect()
    await asyncio.sleep(0.5)

    # Force disconnect
    mock_server.force_disconnect_after(1)
    await client.send({"type": "test"})
    await asyncio.sleep(0.2)

    # Close while reconnecting
    close_start = time.time()
    await client.close()
    close_time = time.time() - close_start

    # Verify clean shutdown
    assert client.state == ConnectionState.CLOSED
    assert close_time < 5.0, f"Shutdown took {close_time:.1f}s (should be <5s)"

    logger.info(f"✓ Graceful shutdown completed in {close_time:.2f}s")


# Performance benchmarks
@pytest.mark.asyncio
async def test_reconnect_performance_benchmark(mock_server, reconnect_config, heartbeat_config):
    """
    Benchmark reconnection performance

    Success criteria: Reconnect within 30 seconds (per TARS-1001)
    """
    client = ReconnectingWebSocketClient(
        uri=mock_server.uri,
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
    )

    await client.connect()
    await asyncio.sleep(0.5)

    reconnect_times = []

    # Test multiple reconnections
    for i in range(5):
        # Force disconnect
        mock_server.force_disconnect_after(1)
        await client.send({"type": f"test-{i}"})
        await asyncio.sleep(0.2)

        # Measure reconnection time
        start = time.time()
        max_wait = 30.0
        while time.time() - start < max_wait:
            if client.is_connected:
                break
            await asyncio.sleep(0.05)

        reconnect_time = time.time() - start
        reconnect_times.append(reconnect_time)

        assert client.is_connected, f"Reconnection {i+1} failed"
        await asyncio.sleep(0.5)

    # Performance analysis
    avg_reconnect = sum(reconnect_times) / len(reconnect_times)
    max_reconnect = max(reconnect_times)
    min_reconnect = min(reconnect_times)

    logger.info(f"Reconnection Performance:")
    logger.info(f"  Average: {avg_reconnect:.2f}s")
    logger.info(f"  Min: {min_reconnect:.2f}s")
    logger.info(f"  Max: {max_reconnect:.2f}s")
    logger.info(f"  All times: {[f'{t:.2f}s' for t in reconnect_times]}")

    # Verify all reconnections within requirement
    assert all(
        t < 30.0 for t in reconnect_times
    ), f"All reconnections must be <30s (max: {max_reconnect:.2f}s)"

    assert avg_reconnect < 5.0, f"Average reconnection should be <5s (got {avg_reconnect:.2f}s)"

    logger.info("✓ Reconnection performance validated")

    await client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pytest.main([__file__, "-v", "-s"])

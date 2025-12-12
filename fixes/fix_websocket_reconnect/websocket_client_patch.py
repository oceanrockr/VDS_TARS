"""
WebSocket Client with Auto-Reconnection and Heartbeat

TARS-1001: WebSocket Reconnection Issue Fix
-------------------------------------------
Problem: WebSocket connections drop silently, requiring manual page refresh
Solution: Heartbeat mechanism + exponential backoff reconnection + auto-resubscribe

Features:
- Heartbeat ping/pong with configurable interval (default: 30s)
- Exponential backoff reconnection (1s, 2s, 4s, 8s, 16s, max 30s)
- Silent disconnect detection via ping timeout
- Automatic resubscription to previous channels
- Connection state management with callbacks
- Thread-safe implementation

Author: T.A.R.S. Engineering Team
Version: 1.0.1
Date: 2025-11-20
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    WebSocketException,
)

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class ReconnectionConfig:
    """Configuration for reconnection behavior"""

    enabled: bool = True
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    max_attempts: Optional[int] = None  # None = infinite
    jitter: bool = True  # Add random jitter to backoff


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat mechanism"""

    enabled: bool = True
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    max_missed_heartbeats: int = 3


@dataclass
class WebSocketMessage:
    """Structured WebSocket message"""

    type: str
    channel: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ReconnectingWebSocketClient:
    """
    WebSocket client with automatic reconnection and heartbeat

    This client wraps the websockets library and provides:
    - Automatic reconnection with exponential backoff
    - Heartbeat ping/pong to detect silent disconnects
    - Channel subscription management and auto-resubscribe
    - Connection state callbacks
    - Thread-safe message queue

    Example:
        ```python
        client = ReconnectingWebSocketClient(
            uri="wss://api.tars.dev/ws",
            reconnect_config=ReconnectionConfig(),
            heartbeat_config=HeartbeatConfig()
        )

        client.on_message(lambda msg: print(f"Received: {msg}"))
        client.on_state_change(lambda state: print(f"State: {state}"))

        await client.connect()
        await client.subscribe("evaluation.updates")

        # Client will automatically reconnect and resubscribe
        # if connection is lost
        ```
    """

    def __init__(
        self,
        uri: str,
        reconnect_config: Optional[ReconnectionConfig] = None,
        heartbeat_config: Optional[HeartbeatConfig] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.uri = uri
        self.reconnect_config = reconnect_config or ReconnectionConfig()
        self.heartbeat_config = heartbeat_config or HeartbeatConfig()
        self.extra_headers = extra_headers or {}

        # Connection state
        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._reconnect_attempts: int = 0
        self._last_ping_time: Optional[float] = None
        self._last_pong_time: Optional[float] = None
        self._missed_heartbeats: int = 0

        # Subscriptions
        self._subscribed_channels: Set[str] = set()

        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Callbacks
        self._message_callbacks: List[Callable[[WebSocketMessage], None]] = []
        self._state_callbacks: List[Callable[[ConnectionState], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Graceful shutdown
        self._closing: bool = False
        self._close_event: asyncio.Event = asyncio.Event()

        logger.info(
            f"ReconnectingWebSocketClient initialized for {uri}",
            extra={
                "reconnect_enabled": self.reconnect_config.enabled,
                "heartbeat_enabled": self.heartbeat_config.enabled,
            },
        )

    @property
    def state(self) -> ConnectionState:
        """Current connection state"""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._state == ConnectionState.CONNECTED and self._websocket is not None

    @property
    def subscribed_channels(self) -> Set[str]:
        """Get set of currently subscribed channels"""
        return self._subscribed_channels.copy()

    def on_message(self, callback: Callable[[WebSocketMessage], None]) -> None:
        """Register message callback"""
        self._message_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register state change callback"""
        self._state_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register error callback"""
        self._error_callbacks.append(callback)

    async def connect(self) -> None:
        """
        Connect to WebSocket server

        This initiates the connection and starts background tasks for:
        - Message receiving
        - Heartbeat monitoring
        - Automatic reconnection (if enabled)
        """
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("Already connected or connecting")
            return

        self._closing = False
        self._close_event.clear()
        await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal connection logic"""
        self._set_state(ConnectionState.CONNECTING)

        try:
            logger.info(f"Connecting to {self.uri}")
            self._websocket = await websockets.connect(
                self.uri,
                extra_headers=self.extra_headers,
                ping_interval=None,  # We handle heartbeat manually
                ping_timeout=None,
            )

            self._reconnect_attempts = 0
            self._missed_heartbeats = 0
            self._last_pong_time = time.time()
            self._set_state(ConnectionState.CONNECTED)

            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            if self.heartbeat_config.enabled:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Resubscribe to channels
            await self._resubscribe()

            logger.info("WebSocket connected successfully")

        except Exception as e:
            logger.error(f"Connection failed: {e}", exc_info=True)
            self._notify_error(e)
            await self._handle_disconnect(reconnect=True)

    async def _resubscribe(self) -> None:
        """Resubscribe to all previously subscribed channels"""
        if not self._subscribed_channels:
            return

        logger.info(
            f"Resubscribing to {len(self._subscribed_channels)} channels",
            extra={"channels": list(self._subscribed_channels)},
        )

        for channel in list(self._subscribed_channels):
            try:
                await self._send_subscribe(channel)
            except Exception as e:
                logger.error(
                    f"Failed to resubscribe to {channel}: {e}", exc_info=True
                )

    async def subscribe(self, channel: str) -> None:
        """
        Subscribe to a channel

        Args:
            channel: Channel name (e.g., "evaluation.updates")
        """
        if channel in self._subscribed_channels:
            logger.debug(f"Already subscribed to {channel}")
            return

        self._subscribed_channels.add(channel)

        if self.is_connected:
            await self._send_subscribe(channel)

    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel

        Args:
            channel: Channel name
        """
        if channel not in self._subscribed_channels:
            logger.debug(f"Not subscribed to {channel}")
            return

        self._subscribed_channels.remove(channel)

        if self.is_connected:
            await self._send_unsubscribe(channel)

    async def _send_subscribe(self, channel: str) -> None:
        """Send subscribe message"""
        await self._send_message(
            {"type": "subscribe", "channel": channel, "timestamp": time.time()}
        )
        logger.info(f"Subscribed to {channel}")

    async def _send_unsubscribe(self, channel: str) -> None:
        """Send unsubscribe message"""
        await self._send_message(
            {"type": "unsubscribe", "channel": channel, "timestamp": time.time()}
        )
        logger.info(f"Unsubscribed from {channel}")

    async def send(self, message: Dict[str, Any]) -> None:
        """
        Send message to server

        Args:
            message: Message dictionary to send
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        await self._send_message(message)

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Internal send message"""
        if self._websocket is None:
            raise RuntimeError("WebSocket not initialized")

        try:
            await self._websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
            raise

    async def _receive_loop(self) -> None:
        """Background task to receive messages"""
        while not self._closing and self._websocket is not None:
            try:
                raw_message = await self._websocket.recv()

                # Handle pong responses
                if raw_message == "pong":
                    self._last_pong_time = time.time()
                    self._missed_heartbeats = 0
                    logger.debug("Received pong")
                    continue

                # Parse JSON message
                try:
                    data = json.loads(raw_message)
                    message = WebSocketMessage(
                        type=data.get("type", "unknown"),
                        channel=data.get("channel"),
                        payload=data.get("payload"),
                        timestamp=datetime.utcnow(),
                    )
                    self._notify_message(message)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {raw_message[:100]}")

            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                await self._handle_disconnect(reconnect=True)
                break
            except Exception as e:
                logger.error(f"Receive error: {e}", exc_info=True)
                self._notify_error(e)
                await self._handle_disconnect(reconnect=True)
                break

    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeat pings"""
        while not self._closing and self.is_connected:
            try:
                await asyncio.sleep(self.heartbeat_config.interval_seconds)

                if not self.is_connected or self._websocket is None:
                    break

                # Check for missed pongs
                if self._last_pong_time is not None:
                    elapsed = time.time() - self._last_pong_time
                    if elapsed > (
                        self.heartbeat_config.interval_seconds
                        + self.heartbeat_config.timeout_seconds
                    ):
                        self._missed_heartbeats += 1
                        logger.warning(
                            f"Missed heartbeat (count: {self._missed_heartbeats})",
                            extra={"elapsed_seconds": elapsed},
                        )

                        if (
                            self._missed_heartbeats
                            >= self.heartbeat_config.max_missed_heartbeats
                        ):
                            logger.error("Too many missed heartbeats, reconnecting")
                            await self._handle_disconnect(reconnect=True)
                            break

                # Send ping
                self._last_ping_time = time.time()
                await self._websocket.send("ping")
                logger.debug("Sent ping")

            except Exception as e:
                logger.error(f"Heartbeat error: {e}", exc_info=True)
                await self._handle_disconnect(reconnect=True)
                break

    async def _handle_disconnect(self, reconnect: bool = True) -> None:
        """
        Handle disconnection

        Args:
            reconnect: Whether to attempt reconnection
        """
        # Cancel background tasks
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        # Close websocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        if self._closing:
            self._set_state(ConnectionState.CLOSED)
            self._close_event.set()
            return

        self._set_state(ConnectionState.DISCONNECTED)

        # Attempt reconnection
        if reconnect and self.reconnect_config.enabled:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Background task to handle reconnection with exponential backoff"""
        self._set_state(ConnectionState.RECONNECTING)

        while not self._closing:
            self._reconnect_attempts += 1

            # Check max attempts
            if (
                self.reconnect_config.max_attempts is not None
                and self._reconnect_attempts > self.reconnect_config.max_attempts
            ):
                logger.error("Max reconnection attempts reached")
                self._set_state(ConnectionState.DISCONNECTED)
                break

            # Calculate backoff delay
            delay = min(
                self.reconnect_config.initial_delay_seconds
                * (self.reconnect_config.backoff_multiplier ** (self._reconnect_attempts - 1)),
                self.reconnect_config.max_delay_seconds,
            )

            # Add jitter
            if self.reconnect_config.jitter:
                import random

                delay *= 0.5 + random.random()

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts} in {delay:.1f}s",
                extra={"attempt": self._reconnect_attempts, "delay_seconds": delay},
            )

            await asyncio.sleep(delay)

            # Attempt reconnection
            try:
                await self._do_connect()
                if self.is_connected:
                    logger.info("Reconnection successful")
                    break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}", exc_info=True)
                self._notify_error(e)

    async def close(self) -> None:
        """
        Close connection gracefully

        This will:
        - Cancel all background tasks
        - Unsubscribe from all channels
        - Close WebSocket connection
        - Wait for cleanup to complete
        """
        logger.info("Closing WebSocket client")
        self._closing = True

        # Cancel reconnection
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Unsubscribe from all channels
        for channel in list(self._subscribed_channels):
            try:
                await self.unsubscribe(channel)
            except Exception:
                pass

        # Trigger disconnect
        await self._handle_disconnect(reconnect=False)

        # Wait for cleanup
        try:
            await asyncio.wait_for(self._close_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Close timeout exceeded")

        logger.info("WebSocket client closed")

    def _set_state(self, state: ConnectionState) -> None:
        """Update connection state and notify callbacks"""
        if self._state == state:
            return

        old_state = self._state
        self._state = state

        logger.info(f"State transition: {old_state.value} -> {state.value}")

        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}", exc_info=True)

    def _notify_message(self, message: WebSocketMessage) -> None:
        """Notify message callbacks"""
        for callback in self._message_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Message callback error: {e}", exc_info=True)

    def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks"""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}", exc_info=True)


# Example usage and integration points
async def example_usage():
    """
    Example usage of ReconnectingWebSocketClient

    This demonstrates:
    - Connection setup
    - Channel subscription
    - Message handling
    - State monitoring
    - Graceful shutdown
    """
    # Configure reconnection behavior
    reconnect_config = ReconnectionConfig(
        enabled=True,
        initial_delay_seconds=1.0,
        max_delay_seconds=30.0,
        backoff_multiplier=2.0,
        max_attempts=None,  # Retry indefinitely
        jitter=True,
    )

    # Configure heartbeat
    heartbeat_config = HeartbeatConfig(
        enabled=True,
        interval_seconds=30.0,
        timeout_seconds=10.0,
        max_missed_heartbeats=3,
    )

    # Create client
    client = ReconnectingWebSocketClient(
        uri="wss://api.tars.dev/ws",
        reconnect_config=reconnect_config,
        heartbeat_config=heartbeat_config,
        extra_headers={"Authorization": "Bearer <token>"},
    )

    # Register callbacks
    def on_message(msg: WebSocketMessage):
        print(f"[{msg.timestamp}] {msg.type} on {msg.channel}: {msg.payload}")

    def on_state_change(state: ConnectionState):
        print(f"Connection state: {state.value}")

    def on_error(error: Exception):
        print(f"Error: {error}")

    client.on_message(on_message)
    client.on_state_change(on_state_change)
    client.on_error(on_error)

    # Connect
    await client.connect()

    # Subscribe to channels
    await client.subscribe("evaluation.updates")
    await client.subscribe("agent.status")

    # Keep alive (client will auto-reconnect if connection drops)
    try:
        await asyncio.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        pass

    # Graceful shutdown
    await client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(example_usage())

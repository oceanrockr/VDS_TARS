"""
T.A.R.S. Connection Manager
Manages WebSocket connections, sessions, and broadcasts
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any
from collections import defaultdict
import uuid

from fastapi import WebSocket
from ..models.websocket import (
    WebSocketMessage,
    SystemMessage,
    MetricsMessage,
)

logger = logging.getLogger(__name__)


class ConnectionInfo:
    """Information about a WebSocket connection"""

    def __init__(
        self,
        websocket: WebSocket,
        client_id: str,
        session_id: str,
        ip_address: str,
        connected_at: datetime,
    ):
        self.websocket = websocket
        self.client_id = client_id
        self.session_id = session_id
        self.ip_address = ip_address
        self.connected_at = connected_at
        self.last_activity = datetime.utcnow()
        self.message_count = 0
        self.token_count = 0
        self.error_count = 0

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "client_id": self.client_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "token_count": self.token_count,
            "error_count": self.error_count,
        }


class ConnectionManager:
    """
    Manages WebSocket connections with session tracking and broadcasting.

    Features:
    - Connection tracking with metadata
    - Broadcast messaging
    - Heartbeat monitoring
    - Connection limits and queuing
    - Metrics collection
    """

    def __init__(self, max_connections: int = 10):
        """
        Initialize the connection manager.

        Args:
            max_connections: Maximum number of concurrent connections
        """
        self.max_connections = max_connections

        # Active connections: session_id -> ConnectionInfo
        self.active_connections: Dict[str, ConnectionInfo] = {}

        # Client mapping: client_id -> Set[session_id]
        self.client_sessions: Dict[str, Set[str]] = defaultdict(set)

        # Message queues: session_id -> asyncio.Queue
        self.message_queues: Dict[str, asyncio.Queue] = {}

        # Metrics
        self.total_connections = 0
        self.total_messages = 0
        self.total_tokens = 0
        self.total_errors = 0

        logger.info(f"ConnectionManager initialized (max_connections={max_connections})")

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        ip_address: str,
    ) -> Optional[str]:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Client identifier from JWT
            ip_address: Client IP address

        Returns:
            Session ID if connected successfully, None if rejected
        """
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            logger.warning(
                f"Connection rejected for {client_id}: max connections reached "
                f"({self.max_connections})"
            )
            return None

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Create connection info
        connection_info = ConnectionInfo(
            websocket=websocket,
            client_id=client_id,
            session_id=session_id,
            ip_address=ip_address,
            connected_at=datetime.utcnow(),
        )

        # Store connection
        self.active_connections[session_id] = connection_info
        self.client_sessions[client_id].add(session_id)

        # Create message queue
        self.message_queues[session_id] = asyncio.Queue(maxsize=100)

        # Update metrics
        self.total_connections += 1

        logger.info(
            f"Connection accepted: session={session_id}, client={client_id}, "
            f"ip={ip_address}, active={len(self.active_connections)}"
        )

        return session_id

    async def disconnect(self, session_id: str) -> None:
        """
        Disconnect a WebSocket connection.

        Args:
            session_id: Session identifier
        """
        if session_id not in self.active_connections:
            return

        connection_info = self.active_connections[session_id]
        client_id = connection_info.client_id

        # Remove from client sessions
        if client_id in self.client_sessions:
            self.client_sessions[client_id].discard(session_id)
            if not self.client_sessions[client_id]:
                del self.client_sessions[client_id]

        # Remove message queue
        if session_id in self.message_queues:
            del self.message_queues[session_id]

        # Remove connection
        del self.active_connections[session_id]

        logger.info(
            f"Connection closed: session={session_id}, client={client_id}, "
            f"active={len(self.active_connections)}"
        )

    async def send_message(
        self,
        session_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            session_id: Session identifier
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        if session_id not in self.active_connections:
            return False

        try:
            connection_info = self.active_connections[session_id]
            await connection_info.websocket.send_json(message.model_dump())

            connection_info.update_activity()
            connection_info.message_count += 1
            self.total_messages += 1

            return True

        except Exception as e:
            logger.error(f"Failed to send message to {session_id}: {e}")
            connection_info.error_count += 1
            self.total_errors += 1
            return False

    async def send_text(
        self,
        session_id: str,
        text: str,
    ) -> bool:
        """
        Send raw text to a specific connection.

        Args:
            session_id: Session identifier
            text: Text to send

        Returns:
            True if sent successfully, False otherwise
        """
        if session_id not in self.active_connections:
            return False

        try:
            connection_info = self.active_connections[session_id]
            await connection_info.websocket.send_text(text)

            connection_info.update_activity()
            return True

        except Exception as e:
            logger.error(f"Failed to send text to {session_id}: {e}")
            return False

    async def broadcast(
        self,
        message: WebSocketMessage,
        exclude_sessions: Optional[Set[str]] = None,
    ) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast
            exclude_sessions: Set of session IDs to exclude

        Returns:
            Number of successful sends
        """
        exclude_sessions = exclude_sessions or set()
        success_count = 0

        for session_id in list(self.active_connections.keys()):
            if session_id in exclude_sessions:
                continue

            if await self.send_message(session_id, message):
                success_count += 1

        logger.debug(f"Broadcast message to {success_count} clients")
        return success_count

    async def broadcast_to_client(
        self,
        client_id: str,
        message: WebSocketMessage,
    ) -> int:
        """
        Broadcast a message to all sessions of a specific client.

        Args:
            client_id: Client identifier
            message: Message to broadcast

        Returns:
            Number of successful sends
        """
        if client_id not in self.client_sessions:
            return 0

        success_count = 0
        for session_id in self.client_sessions[client_id]:
            if await self.send_message(session_id, message):
                success_count += 1

        return success_count

    def get_connection_info(self, session_id: str) -> Optional[ConnectionInfo]:
        """
        Get connection information.

        Args:
            session_id: Session identifier

        Returns:
            ConnectionInfo if found, None otherwise
        """
        return self.active_connections.get(session_id)

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active sessions with their information.

        Returns:
            Dictionary of session information
        """
        return {
            session_id: info.to_dict()
            for session_id, info in self.active_connections.items()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connection manager metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "active_connections": len(self.active_connections),
            "unique_clients": len(self.client_sessions),
            "total_connections": self.total_connections,
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "max_connections": self.max_connections,
        }

    def increment_token_count(self, session_id: str) -> None:
        """Increment token count for a session"""
        if session_id in self.active_connections:
            self.active_connections[session_id].token_count += 1
            self.total_tokens += 1


# Global connection manager instance
connection_manager = ConnectionManager(max_connections=10)

"""
T.A.R.S. WebSocket Router
WebSocket endpoints for real-time chat and streaming
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, Query
from fastapi.responses import JSONResponse

from ..models.websocket import (
    ChatMessage,
    TokenStreamMessage,
    StreamCompleteMessage,
    ErrorMessage,
    PingMessage,
    PongMessage,
    ConnectionAckMessage,
    SystemMessage,
    RAGSourcesMessage,
    RAGTokenMessage,
    RAGCompleteMessage,
)
from ..core.middleware import verify_websocket_token
from ..services.connection_manager import connection_manager
from ..services.ollama_service import ollama_service
from ..services.rag_service import rag_service
from ..models.rag import RAGQueryRequest
from ..core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ws", tags=["WebSocket"])


async def handle_ping_pong(websocket: WebSocket, session_id: str) -> None:
    """
    Handle ping/pong heartbeat in the background.

    Args:
        websocket: WebSocket connection
        session_id: Session identifier
    """
    try:
        while True:
            await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)

            # Check if connection is still active
            connection_info = connection_manager.get_connection_info(session_id)
            if not connection_info:
                break

            # Send ping
            try:
                ping_message = PingMessage()
                await connection_manager.send_message(session_id, ping_message)
                logger.debug(f"Sent ping to session {session_id}")
            except Exception as e:
                logger.error(f"Failed to send ping to {session_id}: {e}")
                break

    except asyncio.CancelledError:
        logger.debug(f"Heartbeat task cancelled for session {session_id}")
    except Exception as e:
        logger.error(f"Heartbeat error for session {session_id}: {e}")


async def handle_chat_stream(
    session_id: str,
    message: ChatMessage,
) -> None:
    """
    Handle a chat message and stream tokens back.

    Args:
        session_id: Session identifier
        message: Chat message from client
    """
    start_time = time.time()
    token_count = 0

    try:
        # Generate streaming response from Ollama
        async for chunk in ollama_service.generate_stream(
            prompt=message.content,
            model=message.model,
            temperature=message.temperature,
            max_tokens=message.max_tokens,
        ):
            token = chunk.get("token", "")
            done = chunk.get("done", False)

            if token:
                # Send token to client
                token_message = TokenStreamMessage(
                    token=token,
                    conversation_id=message.conversation_id,
                )

                success = await connection_manager.send_message(session_id, token_message)

                if success:
                    token_count += 1
                    connection_manager.increment_token_count(session_id)
                else:
                    logger.warning(f"Failed to send token to session {session_id}")
                    break

            if done:
                break

        # Send completion message
        elapsed_ms = (time.time() - start_time) * 1000
        complete_message = StreamCompleteMessage(
            conversation_id=message.conversation_id,
            total_tokens=token_count,
            latency_ms=elapsed_ms,
        )

        await connection_manager.send_message(session_id, complete_message)

        logger.info(
            f"Stream complete for session {session_id}: {token_count} tokens in "
            f"{elapsed_ms:.0f}ms ({token_count / (elapsed_ms / 1000):.1f} tokens/s)"
        )

    except Exception as e:
        logger.error(f"Error in chat stream for session {session_id}: {e}")

        # Send error message to client
        error_message = ErrorMessage(
            error=f"Generation failed: {str(e)}",
            code="GENERATION_ERROR",
        )
        await connection_manager.send_message(session_id, error_message)


async def handle_rag_chat_stream(
    session_id: str,
    message: ChatMessage,
) -> None:
    """
    Handle a RAG-enabled chat message and stream tokens back.
    (Phase 3)

    Args:
        session_id: Session identifier
        message: Chat message from client with RAG enabled
    """
    total_start = time.time()
    token_count = 0
    retrieval_time_ms = 0
    sources_count = 0

    try:
        # Create RAG query request
        rag_request = RAGQueryRequest(
            query=message.content,
            conversation_id=message.conversation_id,
            top_k=message.rag_top_k or settings.RAG_TOP_K,
            relevance_threshold=message.rag_threshold or settings.RAG_RELEVANCE_THRESHOLD,
            include_sources=True,
            rerank=settings.RAG_RERANK_ENABLED
        )

        # Stream RAG response
        async for stream_message in rag_service.query_stream(
            request=rag_request,
            conversation_id=message.conversation_id
        ):
            message_type = stream_message.get('type')

            if message_type == 'rag_sources':
                # Send sources to client
                retrieval_time_ms = stream_message.get('retrieval_time_ms', 0)
                sources = stream_message.get('sources', [])
                sources_count = len(sources)

                sources_msg = RAGSourcesMessage(
                    conversation_id=message.conversation_id,
                    sources=[s.dict() if hasattr(s, 'dict') else s for s in sources],
                    retrieval_time_ms=retrieval_time_ms
                )

                await connection_manager.send_message(session_id, sources_msg)
                logger.debug(f"Sent {sources_count} sources to session {session_id}")

            elif message_type == 'rag_token':
                # Send token to client
                token = stream_message.get('token', '')

                if token:
                    rag_token_msg = RAGTokenMessage(
                        token=token,
                        conversation_id=message.conversation_id,
                        has_sources=sources_count > 0
                    )

                    success = await connection_manager.send_message(session_id, rag_token_msg)

                    if success:
                        token_count += 1
                        connection_manager.increment_token_count(session_id)
                    else:
                        logger.warning(f"Failed to send RAG token to session {session_id}")
                        break

            elif message_type == 'rag_complete':
                # Send completion message
                generation_time_ms = stream_message.get('generation_time_ms', 0)
                total_time_ms = stream_message.get('total_time_ms', 0)

                complete_msg = RAGCompleteMessage(
                    conversation_id=message.conversation_id,
                    total_tokens=token_count,
                    retrieval_time_ms=retrieval_time_ms,
                    generation_time_ms=generation_time_ms,
                    total_time_ms=total_time_ms,
                    sources_count=sources_count
                )

                await connection_manager.send_message(session_id, complete_msg)

                logger.info(
                    f"RAG stream complete for session {session_id}: "
                    f"{token_count} tokens, {sources_count} sources, {total_time_ms:.0f}ms"
                )

    except Exception as e:
        logger.error(f"Error in RAG chat stream for session {session_id}: {e}")

        # Send error message to client
        error_message = ErrorMessage(
            error=f"RAG generation failed: {str(e)}",
            code="RAG_GENERATION_ERROR",
        )
        await connection_manager.send_message(session_id, error_message)


@router.websocket("/chat")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
):
    """
    WebSocket endpoint for real-time chat with streaming.

    Connection Flow:
    1. Client connects with JWT token in query parameter
    2. Server verifies token and accepts connection
    3. Server sends connection acknowledgment
    4. Server starts heartbeat (ping/pong every 30s)
    5. Client sends chat messages
    6. Server streams tokens back in real-time
    7. Server sends completion message when done

    Message Types:
    - Client → Server: "chat" (ChatMessage)
    - Client → Server: "pong" (PongMessage)
    - Server → Client: "connection_ack" (ConnectionAckMessage)
    - Server → Client: "token" (TokenStreamMessage)
    - Server → Client: "complete" (StreamCompleteMessage)
    - Server → Client: "error" (ErrorMessage)
    - Server → Client: "ping" (PingMessage)
    - Server → Client: "system" (SystemMessage)

    Args:
        websocket: WebSocket connection
        token: JWT access token from query parameter
    """
    session_id = None
    heartbeat_task = None

    try:
        # Verify authentication
        payload = await verify_websocket_token(websocket)

        if not payload:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            logger.warning("WebSocket connection rejected: invalid token")
            return

        client_id = payload.get("sub")
        ip_address = websocket.client.host if websocket.client else "unknown"

        # Accept the WebSocket connection
        await websocket.accept()

        # Register connection
        session_id = await connection_manager.connect(
            websocket=websocket,
            client_id=client_id,
            ip_address=ip_address,
        )

        if not session_id:
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
            logger.warning(f"Connection rejected for {client_id}: max connections reached")
            return

        # Send connection acknowledgment
        ack_message = ConnectionAckMessage(
            client_id=client_id,
            session_id=session_id,
        )
        await connection_manager.send_message(session_id, ack_message)

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(handle_ping_pong(websocket, session_id))

        logger.info(f"WebSocket connected: session={session_id}, client={client_id}")

        # Main message loop
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.WS_TIMEOUT_SECONDS,
                )

                # Parse message
                try:
                    message_dict = json.loads(data)
                    message_type = message_dict.get("type")

                    if message_type == "chat":
                        # Handle chat message
                        chat_message = ChatMessage(**message_dict)
                        logger.info(
                            f"Received chat message from session {session_id}: "
                            f"{len(chat_message.content)} chars, RAG={chat_message.use_rag}"
                        )

                        # Choose handler based on RAG mode (Phase 3)
                        if chat_message.use_rag:
                            asyncio.create_task(handle_rag_chat_stream(session_id, chat_message))
                        else:
                            asyncio.create_task(handle_chat_stream(session_id, chat_message))

                    elif message_type == "pong":
                        # Handle pong response
                        logger.debug(f"Received pong from session {session_id}")

                    else:
                        logger.warning(f"Unknown message type: {message_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from session {session_id}: {e}")
                    error_message = ErrorMessage(
                        error="Invalid JSON format",
                        code="INVALID_JSON",
                    )
                    await connection_manager.send_message(session_id, error_message)

                except Exception as e:
                    logger.error(f"Error parsing message from session {session_id}: {e}")
                    error_message = ErrorMessage(
                        error=f"Invalid message format: {str(e)}",
                        code="INVALID_MESSAGE",
                    )
                    await connection_manager.send_message(session_id, error_message)

            except asyncio.TimeoutError:
                logger.warning(f"Session {session_id} timed out")
                system_message = SystemMessage(
                    message="Connection timeout - please reconnect",
                    level="warning",
                )
                await connection_manager.send_message(session_id, system_message)
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")

    finally:
        # Cancel heartbeat task
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Disconnect
        if session_id:
            await connection_manager.disconnect(session_id)


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="WebSocket Service Health",
    description="Check WebSocket service health and metrics",
)
async def websocket_health() -> Dict[str, Any]:
    """WebSocket service health check with metrics"""
    metrics = connection_manager.get_metrics()
    ollama_healthy = await ollama_service.health_check()

    return {
        "status": "healthy",
        "service": "websocket",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "ollama_status": "healthy" if ollama_healthy else "unhealthy",
    }


@router.get(
    "/sessions",
    status_code=status.HTTP_200_OK,
    summary="Active Sessions",
    description="Get information about active WebSocket sessions",
)
async def get_active_sessions() -> Dict[str, Any]:
    """Get active WebSocket sessions"""
    sessions = connection_manager.get_active_sessions()

    return {
        "total_sessions": len(sessions),
        "sessions": sessions,
        "timestamp": datetime.utcnow().isoformat(),
    }

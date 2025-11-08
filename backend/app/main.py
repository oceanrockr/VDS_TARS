"""
T.A.R.S. Backend - Main Application Entry Point
FastAPI application with WebSocket, Authentication, RAG, and Conversation History (Phase 4)
"""

import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routers
from .api.auth import router as auth_router
from .api.websocket import router as websocket_router
from .api.rag import router as rag_router
from .api.conversation import router as conversation_router
from .api.metrics import router as metrics_router
from .api.analytics import router as analytics_router  # Phase 5
from .services.ollama_service import ollama_service
from .services.rag_service import rag_service
from .services.chromadb_service import chromadb_service
from .services.embedding_service import embedding_service
from .services.conversation_service import conversation_service
from .services.nas_watcher import nas_watcher
from .core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Application metadata
APP_VERSION = "v0.3.0-alpha"
APP_NAME = "T.A.R.S. Backend"
APP_DESCRIPTION = "Temporal Augmented Retrieval System - Local LLM Platform with Advanced RAG, Hybrid Search & Analytics"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting {APP_NAME} {APP_VERSION}")
    logger.info("Phase 5: Advanced RAG & Semantic Chunking")

    # Startup: Initialize services
    startup_time = datetime.utcnow()
    app.state.startup_time = startup_time

    # Check Ollama connectivity
    ollama_healthy = await ollama_service.health_check()
    if ollama_healthy:
        logger.info(f"Ollama service is healthy at {settings.OLLAMA_HOST}")
    else:
        logger.warning(f"Ollama service is not available at {settings.OLLAMA_HOST}")

    # Initialize RAG components (Phase 3)
    logger.info("Initializing RAG components...")
    rag_initialized = await rag_service.initialize()
    if rag_initialized:
        logger.info("RAG service initialized successfully")
    else:
        logger.warning("RAG service initialization incomplete")

    # Initialize conversation service (Phase 4)
    logger.info("Initializing conversation service...")
    conv_initialized = await conversation_service.connect()
    if conv_initialized:
        logger.info("Conversation service initialized successfully")
    else:
        logger.warning("Conversation service initialization incomplete")

    # Start NAS watcher if enabled (Phase 4)
    if settings.NAS_WATCH_ENABLED:
        logger.info("Starting NAS file watcher...")
        nas_watcher.start()

        # Start background tasks for NAS watcher
        app.state.nas_tasks = [
            asyncio.create_task(nas_watcher.process_pending_files()),
            asyncio.create_task(nas_watcher.periodic_scan())
        ]
        logger.info("NAS file watcher started")
    else:
        logger.info("NAS watcher disabled in configuration")
        app.state.nas_tasks = []

    yield

    # Shutdown: Cleanup resources
    logger.info("Shutting down services...")

    # Stop NAS watcher
    nas_watcher.stop()
    for task in app.state.nas_tasks:
        task.cancel()

    await ollama_service.close()
    await embedding_service.close()
    await chromadb_service.close()
    logger.info(f"Shutting down {APP_NAME}")


# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
cors_origins = settings.CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# ROUTERS - Phase 4 & Phase 5
# ==============================================================================

# Include authentication router
app.include_router(auth_router)

# Include WebSocket router
app.include_router(websocket_router)

# Include RAG router (Phase 3)
app.include_router(rag_router)

# Include conversation router (Phase 4)
app.include_router(conversation_router)

# Include metrics router (Phase 4)
app.include_router(metrics_router)

# Include analytics router (Phase 5)
app.include_router(analytics_router)


# ==============================================================================
# HEALTH CHECK ENDPOINTS - Phase 1 Requirement
# ==============================================================================

@app.get(
    "/health",
    tags=["Health"],
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Basic health check endpoint - returns 200 if service is running",
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for Docker healthcheck and load balancers.
    Returns basic service status.
    """
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get(
    "/ready",
    tags=["Health"],
    response_model=Dict[str, any],
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Readiness check - returns 200 when service is ready to accept traffic",
)
async def readiness_check() -> Dict[str, any]:
    """
    Readiness check endpoint for Kubernetes and orchestration systems.
    Returns detailed service readiness status.

    Phase 4: Checks Ollama, ChromaDB, Embedding model, Conversation service, and NAS watcher
    """
    uptime = datetime.utcnow() - app.state.startup_time

    # Check Ollama connectivity
    ollama_status = "healthy" if await ollama_service.health_check() else "unhealthy"

    # Check ChromaDB (Phase 3)
    chroma_health = chromadb_service.health_check()
    chroma_status = chroma_health.get('status', 'unknown')

    # Check Embedding Model (Phase 3)
    embed_health = embedding_service.health_check()
    embed_status = embed_health.get('status', 'unknown')

    # Check Conversation Service (Phase 4)
    conv_health = conversation_service.health_check()
    conv_status = conv_health.get('status', 'unknown')

    # Check NAS Watcher (Phase 4)
    nas_stats = nas_watcher.get_stats()
    nas_status = "enabled" if nas_stats['enabled'] else "disabled"

    # Overall status
    all_healthy = all([
        ollama_status == "healthy",
        chroma_status == "healthy",
        embed_status == "healthy",
        conv_status == "healthy"
    ])

    return {
        "status": "ready" if all_healthy else "degraded",
        "service": APP_NAME,
        "version": APP_VERSION,
        "uptime_seconds": int(uptime.total_seconds()),
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "ollama": ollama_status,
            "chromadb": chroma_status,
            "embedding_model": embed_status,
            "conversation_service": conv_status,
            "nas_watcher": nas_status,
        },
    }


@app.get(
    "/metrics",
    tags=["Monitoring"],
    response_model=Dict[str, any],
    summary="Basic Metrics",
    description="Basic application metrics endpoint",
)
async def metrics() -> Dict[str, any]:
    """
    Basic metrics endpoint for monitoring.
    Will be enhanced with Prometheus metrics in Phase 6.
    """
    uptime = datetime.utcnow() - app.state.startup_time
    
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "uptime_seconds": int(uptime.total_seconds()),
        "startup_time": app.state.startup_time.isoformat(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get(
    "/",
    tags=["Root"],
    response_model=Dict[str, str],
    summary="Root Endpoint",
    description="API root - provides basic service information",
)
async def root() -> Dict[str, str]:
    """Root endpoint with API information"""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "phase": "Phase 4 - Client UI & NAS Monitoring",
        "status": "in_development",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "auth": "/auth/token",
        "websocket": "/ws/chat",
        "rag": "/rag/query",
        "conversation": "/conversation/list",
        "metrics": "/metrics/system",
        "nas_stats": "/rag/stats",
    }


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "service": APP_NAME,
            "version": APP_VERSION,
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "service": APP_NAME,
            "version": APP_VERSION,
        },
    )


# ==============================================================================
# APPLICATION STARTUP
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

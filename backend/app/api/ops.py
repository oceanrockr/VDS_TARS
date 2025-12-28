"""
T.A.R.S. Operations API Router
Production-in-Use Observability Endpoints
Phase 23 - Local Machine Rollout + Operator UX
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, status, Depends

from ..core.middleware import verify_token
from ..services.chromadb_service import chromadb_service
from ..services.ollama_service import ollama_service
from ..services.redis_cache import redis_cache
from ..services.conversation_service import conversation_service
from ..services.nas_watcher import nas_watcher

logger = logging.getLogger(__name__)


# ==============================================================================
# Pydantic Models
# ==============================================================================

class ServiceHealthSnapshot(BaseModel):
    """Health status for a single service."""
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Health status: healthy, unhealthy, degraded, unknown")
    latency_ms: Optional[float] = Field(None, description="Response latency in milliseconds")
    message: Optional[str] = Field(None, description="Additional status message")


class ModelInfo(BaseModel):
    """Information about loaded LLM model."""
    name: str = Field(..., description="Model name")
    loaded: bool = Field(..., description="Whether model is loaded")
    size: Optional[str] = Field(None, description="Model size")


class ChromaStats(BaseModel):
    """ChromaDB collection statistics."""
    collection_name: str = Field(..., description="Collection name")
    collection_count: int = Field(..., description="Number of collections")
    chunk_count: int = Field(..., description="Total number of chunks/embeddings")


class IngestionInfo(BaseModel):
    """Document ingestion information."""
    last_ingestion_time: Optional[str] = Field(None, description="ISO timestamp of last ingestion")
    documents_indexed: int = Field(0, description="Total documents indexed")
    pending_files: int = Field(0, description="Files pending ingestion")


class QueryInfo(BaseModel):
    """Query activity information."""
    last_successful_query_time: Optional[str] = Field(None, description="ISO timestamp of last successful query")
    queries_today: int = Field(0, description="Number of queries today")


class OpsSummaryResponse(BaseModel):
    """
    Comprehensive operational summary for home deployment monitoring.

    This endpoint provides a single-call overview of system health,
    enabling operators to quickly assess if T.A.R.S. is functioning correctly.
    """
    timestamp: str = Field(..., description="ISO timestamp of this summary")
    version: str = Field(..., description="T.A.R.S. version")
    uptime_seconds: int = Field(..., description="Seconds since service start")

    # Overall status
    overall_status: str = Field(..., description="Overall system status: healthy, degraded, unhealthy")

    # Service health
    services: List[ServiceHealthSnapshot] = Field(..., description="Health status of each service")

    # Model information
    model: Optional[ModelInfo] = Field(None, description="Loaded LLM model information")

    # ChromaDB stats
    chroma_stats: Optional[ChromaStats] = Field(None, description="ChromaDB collection statistics")

    # Ingestion info
    ingestion: Optional[IngestionInfo] = Field(None, description="Document ingestion status")

    # Query info
    query_activity: Optional[QueryInfo] = Field(None, description="Query activity summary")

    # NAS status
    nas_mounted: bool = Field(..., description="Whether NAS is mounted and accessible")
    nas_document_count: int = Field(0, description="Documents found on NAS")


# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(
    prefix="/ops",
    tags=["Operations"],
    responses={
        401: {"description": "Unauthorized - Valid token required"},
        403: {"description": "Forbidden - Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)


# ==============================================================================
# Helper Functions
# ==============================================================================

async def check_service_health(name: str, check_func, timeout: float = 5.0) -> ServiceHealthSnapshot:
    """Check health of a service with timing."""
    import time
    start = time.time()

    try:
        if asyncio.iscoroutinefunction(check_func):
            result = await asyncio.wait_for(check_func(), timeout=timeout)
        else:
            result = check_func()

        latency = (time.time() - start) * 1000

        if isinstance(result, dict):
            status = result.get('status', 'unknown')
            message = result.get('message')
        elif isinstance(result, bool):
            status = 'healthy' if result else 'unhealthy'
            message = None
        else:
            status = 'healthy' if result else 'unhealthy'
            message = None

        return ServiceHealthSnapshot(
            name=name,
            status=status,
            latency_ms=round(latency, 2),
            message=message
        )
    except asyncio.TimeoutError:
        return ServiceHealthSnapshot(
            name=name,
            status='unhealthy',
            latency_ms=timeout * 1000,
            message='Health check timed out'
        )
    except Exception as e:
        return ServiceHealthSnapshot(
            name=name,
            status='unhealthy',
            latency_ms=None,
            message=str(e)[:100]
        )


import asyncio


async def get_model_info() -> Optional[ModelInfo]:
    """Get information about the loaded LLM model."""
    try:
        models = await ollama_service.list_models()
        if models:
            # Get the first/primary model
            model = models[0] if isinstance(models, list) else models
            model_name = model.get('name', 'unknown') if isinstance(model, dict) else str(model)
            return ModelInfo(
                name=model_name,
                loaded=True,
                size=model.get('size') if isinstance(model, dict) else None
            )
        return ModelInfo(name='none', loaded=False, size=None)
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        return None


async def get_chroma_stats() -> Optional[ChromaStats]:
    """Get ChromaDB collection statistics."""
    try:
        stats = await chromadb_service.get_stats()
        return ChromaStats(
            collection_name=stats.collection_name if hasattr(stats, 'collection_name') else 'default',
            collection_count=1,  # We typically use one collection
            chunk_count=stats.total_chunks if hasattr(stats, 'total_chunks') else 0
        )
    except Exception as e:
        logger.warning(f"Could not get ChromaDB stats: {e}")
        return None


def get_ingestion_info() -> IngestionInfo:
    """Get document ingestion information."""
    try:
        nas_stats = nas_watcher.get_stats()
        return IngestionInfo(
            last_ingestion_time=nas_stats.get('last_ingestion_time'),
            documents_indexed=nas_stats.get('documents_indexed', 0),
            pending_files=nas_stats.get('pending_files', 0)
        )
    except Exception as e:
        logger.warning(f"Could not get ingestion info: {e}")
        return IngestionInfo()


async def get_query_activity() -> QueryInfo:
    """Get query activity information."""
    try:
        # Try to get from Redis cache if available
        last_query_time = await redis_cache.get('tars:last_successful_query_time')
        queries_today = await redis_cache.get('tars:queries_today')

        return QueryInfo(
            last_successful_query_time=last_query_time,
            queries_today=int(queries_today) if queries_today else 0
        )
    except Exception as e:
        logger.warning(f"Could not get query activity: {e}")
        return QueryInfo()


def get_nas_status() -> tuple:
    """Get NAS mount status and document count."""
    try:
        nas_stats = nas_watcher.get_stats()
        mounted = nas_stats.get('enabled', False) and nas_stats.get('nas_accessible', False)
        doc_count = nas_stats.get('document_count', 0)
        return mounted, doc_count
    except Exception as e:
        logger.warning(f"Could not get NAS status: {e}")
        return False, 0


# ==============================================================================
# Endpoints
# ==============================================================================

@router.get(
    "/summary",
    response_model=OpsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Operational Summary",
    description="""
    Get a comprehensive operational summary for monitoring T.A.R.S. home deployment.

    This endpoint provides:
    - Overall system health status
    - Individual service health with latency
    - LLM model information
    - ChromaDB collection statistics
    - Document ingestion status
    - Query activity summary
    - NAS mount status

    **Authentication Required:** Yes (Bearer token)

    **Use Case:** Quick system health check for operators
    """,
)
async def get_ops_summary(
    current_user: dict = Depends(verify_token)
) -> OpsSummaryResponse:
    """
    Get comprehensive operational summary.

    Returns a single-call overview of all system components,
    enabling operators to quickly assess system health.
    """
    from ..main import app, APP_VERSION

    # Calculate uptime
    uptime_seconds = 0
    if hasattr(app.state, 'startup_time'):
        uptime = datetime.utcnow() - app.state.startup_time
        uptime_seconds = int(uptime.total_seconds())

    # Check all services concurrently
    service_checks = await asyncio.gather(
        check_service_health('ollama', ollama_service.health_check),
        check_service_health('chromadb', lambda: chromadb_service.health_check()),
        check_service_health('redis', redis_cache.health_check),
        check_service_health('conversation', lambda: conversation_service.health_check()),
        return_exceptions=True
    )

    # Process service checks
    services = []
    for check in service_checks:
        if isinstance(check, Exception):
            services.append(ServiceHealthSnapshot(
                name='unknown',
                status='error',
                message=str(check)[:100]
            ))
        else:
            services.append(check)

    # Determine overall status
    unhealthy_count = sum(1 for s in services if s.status == 'unhealthy')
    if unhealthy_count == 0:
        overall_status = 'healthy'
    elif unhealthy_count < len(services):
        overall_status = 'degraded'
    else:
        overall_status = 'unhealthy'

    # Get additional info concurrently
    model_info, chroma_stats, query_activity = await asyncio.gather(
        get_model_info(),
        get_chroma_stats(),
        get_query_activity(),
        return_exceptions=True
    )

    # Handle any exceptions
    if isinstance(model_info, Exception):
        model_info = None
    if isinstance(chroma_stats, Exception):
        chroma_stats = None
    if isinstance(query_activity, Exception):
        query_activity = QueryInfo()

    # Get sync info
    ingestion_info = get_ingestion_info()
    nas_mounted, nas_doc_count = get_nas_status()

    return OpsSummaryResponse(
        timestamp=datetime.utcnow().isoformat(),
        version=APP_VERSION,
        uptime_seconds=uptime_seconds,
        overall_status=overall_status,
        services=services,
        model=model_info,
        chroma_stats=chroma_stats,
        ingestion=ingestion_info,
        query_activity=query_activity,
        nas_mounted=nas_mounted,
        nas_document_count=nas_doc_count
    )


@router.get(
    "/health-snapshot",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Quick Health Snapshot",
    description="Get a minimal health snapshot without detailed stats.",
)
async def get_health_snapshot(
    current_user: dict = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Get a minimal health snapshot.

    Faster than /summary, useful for frequent polling.
    """
    from ..main import APP_VERSION

    # Quick health checks
    ollama_ok = await ollama_service.health_check()
    chroma_health = chromadb_service.health_check()
    redis_health = await redis_cache.health_check()

    all_healthy = (
        ollama_ok and
        chroma_health.get('status') == 'healthy' and
        redis_health.get('status') == 'healthy'
    )

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "version": APP_VERSION,
        "status": "healthy" if all_healthy else "degraded",
        "services": {
            "ollama": "up" if ollama_ok else "down",
            "chromadb": chroma_health.get('status', 'unknown'),
            "redis": redis_health.get('status', 'unknown'),
        }
    }

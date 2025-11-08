"""
System Metrics API

REST endpoints for system resource monitoring and metrics.
"""

import logging
import time
from typing import Optional, List
from datetime import datetime, timedelta

import psutil
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services.chromadb_service import chromadb_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


# Metrics cache to avoid excessive system calls
_metrics_cache = {
    'data': None,
    'timestamp': 0,
    'cache_duration': 5.0,  # Cache for 5 seconds
}


def get_gpu_info() -> Optional[dict]:
    """
    Get NVIDIA GPU information using nvidia-smi.

    Returns:
        GPU info dict or None if unavailable
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                parts = [p.strip() for p in output.split(',')]
                if len(parts) >= 4:
                    return {
                        'name': parts[0],
                        'utilization': float(parts[1]),
                        'memory_used_mb': float(parts[2]),
                        'memory_total_mb': float(parts[3]),
                        'memory_percent': (float(parts[2]) / float(parts[3])) * 100
                    }
    except Exception as e:
        logger.debug(f"GPU info not available: {e}")

    return None


def collect_system_metrics() -> dict:
    """
    Collect current system metrics.

    Returns:
        System metrics dictionary
    """
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Memory metrics
    memory = psutil.virtual_memory()
    memory_used_mb = memory.used / (1024 * 1024)
    memory_total_mb = memory.total / (1024 * 1024)
    memory_percent = memory.percent

    # GPU metrics (if available)
    gpu_info = get_gpu_info()

    # ChromaDB stats
    try:
        stats = chromadb_service.collection.count() if chromadb_service.collection else 0
        documents_indexed = stats
        chunks_stored = stats
    except Exception:
        documents_indexed = 0
        chunks_stored = 0

    metrics = {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'memory_used_mb': memory_used_mb,
        'memory_total_mb': memory_total_mb,
        'documents_indexed': documents_indexed,
        'chunks_stored': chunks_stored,
        'queries_processed': 0,  # TODO: Track this in RAG service
        'average_retrieval_time_ms': 0.0,  # TODO: Track this in RAG service
        'timestamp': datetime.now().isoformat(),
    }

    # Add GPU metrics if available
    if gpu_info:
        metrics.update({
            'gpu_name': gpu_info['name'],
            'gpu_percent': gpu_info['utilization'],
            'gpu_memory_percent': gpu_info['memory_percent'],
            'gpu_memory_used_mb': gpu_info['memory_used_mb'],
            'gpu_memory_total_mb': gpu_info['memory_total_mb'],
        })

    return metrics


@router.get("/system")
async def get_system_metrics():
    """
    Get current system metrics.

    Returns:
        System resource metrics including CPU, memory, GPU, and document stats
    """
    try:
        current_time = time.time()

        # Check cache
        if (_metrics_cache['data'] is not None and
            current_time - _metrics_cache['timestamp'] < _metrics_cache['cache_duration']):
            return _metrics_cache['data']

        # Collect fresh metrics
        metrics = collect_system_metrics()

        # Update cache
        _metrics_cache['data'] = metrics
        _metrics_cache['timestamp'] = current_time

        return metrics

    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@router.get("/historical")
async def get_historical_metrics(
    start_time: str = Query(..., description="Start time (ISO 8601)"),
    end_time: str = Query(..., description="End time (ISO 8601)"),
    interval: int = Query(60, ge=10, le=3600, description="Interval in seconds")
) -> dict:
    """
    Get historical metrics (stub implementation).

    In a production system, this would query a time-series database.
    For Phase 4, we return the current metrics only.

    Args:
        start_time: Start timestamp
        end_time: End timestamp
        interval: Sampling interval in seconds

    Returns:
        Historical metrics
    """
    try:
        # For now, return current metrics only
        # In production, implement with Prometheus/InfluxDB
        current_metrics = collect_system_metrics()

        return {
            'metrics': [current_metrics],
            'start_time': start_time,
            'end_time': end_time,
            'interval': interval,
            'note': 'Historical metrics not yet implemented - showing current metrics only'
        }

    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@router.get("/health")
async def metrics_health():
    """
    Check metrics service health.

    Returns:
        Health status
    """
    try:
        # Test metric collection
        metrics = collect_system_metrics()

        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'cpu_available': metrics['cpu_percent'] >= 0,
            'memory_available': metrics['memory_percent'] >= 0,
            'gpu_available': 'gpu_percent' in metrics,
        }

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
        )

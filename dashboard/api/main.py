"""
T.A.R.S. Cognitive Orchestration Dashboard API
FastAPI backend with WebSocket support for real-time multi-agent monitoring

Provides REST and WebSocket endpoints for dashboard visualization.
"""
import os
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import httpx
import numpy as np

# Import Redis backend
try:
    from redis_backend import RedisBackend
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Redis backend not available. Using in-memory storage.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="T.A.R.S. Dashboard API",
    description="Real-time multi-agent cognitive orchestration dashboard",
    version="1.0.0-rc1",
)

# Import and register admin routes
try:
    from admin_routes import router as admin_router
    app.include_router(admin_router, prefix="/api/v1")
    logger.info("Admin routes registered successfully")
except ImportError as e:
    logger.warning(f"Admin routes not available: {e}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
dashboard_requests_total = Counter(
    'tars_dashboard_requests_total',
    'Total dashboard API requests',
    ['endpoint', 'method']
)
websocket_connections = Gauge(
    'tars_dashboard_websocket_connections',
    'Active WebSocket connections'
)
simulation_latency_seconds = Histogram(
    'tars_dashboard_simulation_latency_seconds',
    'What-if simulation latency'
)

# Configuration
ORCHESTRATION_SERVICE_URL = os.getenv("ORCHESTRATION_SERVICE_URL", "http://localhost:8094")
AUTOML_SERVICE_URL = os.getenv("AUTOML_SERVICE_URL", "http://localhost:8097")
DASHBOARD_REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "5"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
USE_REDIS = os.getenv("USE_REDIS", "true").lower() == "true" and REDIS_AVAILABLE

# Data storage backend
redis_backend: Optional[RedisBackend] = None

# Fallback in-memory storage (if Redis unavailable)
agent_history: Dict[str, deque] = {
    "policy": deque(maxlen=1000),
    "consensus": deque(maxlen=1000),
    "ethical": deque(maxlen=1000),
    "resource": deque(maxlen=1000),
}
nash_equilibrium_history: deque = deque(maxlen=500)
conflict_history: deque = deque(maxlen=500)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        websocket_connections.set(len(self.active_connections))
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        websocket_connections.set(len(self.active_connections))
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")

manager = ConnectionManager()


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Redis backend on startup."""
    global redis_backend

    if USE_REDIS:
        try:
            logger.info("Initializing Redis backend...")
            redis_backend = RedisBackend(redis_url=REDIS_URL)
            await redis_backend.connect()
            logger.info("Redis backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Redis backend: {e}")
            logger.warning("Falling back to in-memory storage")
            redis_backend = None
    else:
        logger.info("Redis backend disabled. Using in-memory storage.")


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect Redis on shutdown."""
    global redis_backend

    if redis_backend:
        try:
            await redis_backend.disconnect()
            logger.info("Redis backend disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting Redis: {e}")


# ============================================================================
# Request/Response Models
# ============================================================================

class SimulationRequest(BaseModel):
    scenario: str = Field(..., description="Scenario type: high_load, ethical_conflict, resource_constraint")
    agent_states: Dict[str, List[float]] = Field(..., description="Initial states for each agent")
    steps: int = Field(10, description="Number of simulation steps", ge=1, le=100)
    interventions: Optional[List[Dict[str, Any]]] = Field(None, description="Manual interventions")

class AgentStateUpdate(BaseModel):
    agent_id: str
    state: List[float]
    action: Any
    reward: float
    episode: int

class NashEquilibriumRequest(BaseModel):
    agent_payoffs: Dict[str, float]
    candidate_actions: List[Dict[str, Any]]


# ============================================================================
# Agent State Endpoints
# ============================================================================

@app.get("/api/v1/agents")
async def get_all_agents() -> Dict[str, Any]:
    """Get current state of all agents."""
    dashboard_requests_total.labels(endpoint="/agents", method="GET").inc()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ORCHESTRATION_SERVICE_URL}/api/v1/orchestration/agents/state")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get agent states: {e}")
        # Return mock data if service unavailable
        return _get_mock_agent_states()


@app.get("/api/v1/agents/{agent_id}")
async def get_agent(agent_id: str) -> Dict[str, Any]:
    """Get detailed state of a specific agent."""
    dashboard_requests_total.labels(endpoint=f"/agents/{agent_id}", method="GET").inc()

    if agent_id not in agent_history:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Get recent history
    history = list(agent_history[agent_id])

    if not history:
        return _get_mock_agent_state(agent_id)

    return {
        "agent_id": agent_id,
        "current_state": history[-1] if history else None,
        "history": history[-100:],  # Last 100 datapoints
        "statistics": _calculate_agent_stats(history),
    }


@app.get("/api/v1/agents/{agent_id}/history")
async def get_agent_history(
    agent_id: str,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Get historical data for an agent."""
    if agent_id not in agent_history:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    history = list(agent_history[agent_id])
    total = len(history)

    return {
        "agent_id": agent_id,
        "history": history[offset:offset+limit],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ============================================================================
# Multi-Agent Coordination Endpoints
# ============================================================================

@app.get("/api/v1/nash-equilibrium")
async def get_nash_equilibrium() -> Dict[str, Any]:
    """Get recent Nash equilibrium computations."""
    dashboard_requests_total.labels(endpoint="/nash-equilibrium", method="GET").inc()

    try:
        async with httpx.AsyncClient() as client:
            # Get from orchestration service
            response = await client.get(f"{ORCHESTRATION_SERVICE_URL}/api/v1/orchestration/nash")
            response.raise_for_status()
            data = response.json()

            # Store in history
            nash_equilibrium_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            })

            return data
    except Exception as e:
        logger.error(f"Failed to get Nash equilibrium: {e}")
        return _get_mock_nash_equilibrium()


@app.get("/api/v1/conflicts")
async def get_conflicts() -> Dict[str, Any]:
    """Get recent conflict history."""
    dashboard_requests_total.labels(endpoint="/conflicts", method="GET").inc()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ORCHESTRATION_SERVICE_URL}/api/v1/orchestration/conflicts")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get conflicts: {e}")
        return {"conflicts": list(conflict_history)}


@app.get("/api/v1/metrics")
async def get_metrics_summary() -> Dict[str, Any]:
    """Get aggregated metrics summary."""
    dashboard_requests_total.labels(endpoint="/metrics", method="GET").inc()

    # Calculate aggregate metrics from history
    metrics = {
        "global_reward": {
            "current": 0.0,
            "mean_100": 0.0,
            "std_100": 0.0,
            "trend": "stable",
        },
        "agent_rewards": {},
        "conflicts": {
            "total": len(conflict_history),
            "recent_10": len([c for c in conflict_history if True]),  # Last 10 minutes
            "rate_per_step": 0.0,
        },
        "nash_convergence": {
            "avg_iterations": 0.0,
            "convergence_rate": 0.0,
        },
    }

    # Calculate per-agent metrics
    for agent_id, history in agent_history.items():
        if history:
            rewards = [h.get("reward", 0) for h in history if isinstance(h, dict)]
            if rewards:
                metrics["agent_rewards"][agent_id] = {
                    "current": rewards[-1] if rewards else 0.0,
                    "mean": float(np.mean(rewards)),
                    "std": float(np.std(rewards)),
                    "min": float(np.min(rewards)),
                    "max": float(np.max(rewards)),
                }

    return metrics


# ============================================================================
# What-If Simulation Endpoints
# ============================================================================

@app.post("/api/v1/simulate")
async def run_simulation(request: SimulationRequest) -> Dict[str, Any]:
    """Run a what-if simulation scenario."""
    dashboard_requests_total.labels(endpoint="/simulate", method="POST").inc()

    logger.info(f"Running simulation: {request.scenario} for {request.steps} steps")

    import time
    start_time = time.time()

    try:
        # Run simulation (simplified version)
        results = await _execute_simulation(request)

        elapsed_time = time.time() - start_time
        simulation_latency_seconds.observe(elapsed_time)

        return {
            "scenario": request.scenario,
            "steps": request.steps,
            "results": results,
            "elapsed_time_seconds": elapsed_time,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_simulation(request: SimulationRequest) -> Dict[str, Any]:
    """Execute a multi-step simulation."""
    trajectory = []

    current_states = request.agent_states.copy()

    for step in range(request.steps):
        # Simulate one step
        step_result = {
            "step": step,
            "states": current_states.copy(),
            "actions": {},
            "rewards": {},
            "conflicts": [],
        }

        # Mock agent actions (in production, call orchestration service)
        for agent_id in current_states.keys():
            if agent_id == "policy":
                action = np.random.randint(0, 10)
            elif agent_id == "consensus":
                action = np.random.randint(0, 5)
            elif agent_id == "ethical":
                action = np.random.randint(0, 8)
            else:  # resource
                action = float(np.random.uniform(0, 1))

            step_result["actions"][agent_id] = action
            step_result["rewards"][agent_id] = float(np.random.uniform(0, 1))

            # Update state (simplified dynamics)
            current_states[agent_id] = [
                x + np.random.randn() * 0.1 for x in current_states[agent_id]
            ]

        # Check for conflicts
        if np.random.rand() < 0.1:  # 10% chance of conflict
            step_result["conflicts"].append({
                "type": "policy_ethical_mismatch",
                "agents": ["policy", "ethical"],
                "resolution": "nash_equilibrium",
            })

        trajectory.append(step_result)

    return {
        "trajectory": trajectory,
        "final_states": current_states,
        "total_conflicts": sum(len(s["conflicts"]) for s in trajectory),
        "avg_reward": float(np.mean([
            np.mean(list(s["rewards"].values())) for s in trajectory
        ])),
    }


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "data": {
                "agents": _get_mock_agent_states(),
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

        # Keep connection alive and send periodic updates
        while True:
            # Wait for messages from client (for ping/pong)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=DASHBOARD_REFRESH_INTERVAL)
                # Client sent a message (e.g., ping)
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # No message received, send periodic update
                update = await _get_live_update()
                await websocket.send_json(update)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def _get_live_update() -> Dict[str, Any]:
    """Get current system state for live updates."""
    return {
        "type": "update",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "agents": _get_mock_agent_states(),
            "metrics": await get_metrics_summary(),
            "recent_conflicts": list(conflict_history)[-10:],
        }
    }


# ============================================================================
# Explainability Endpoints
# ============================================================================

@app.get("/api/v1/explain/{agent_id}/decision")
async def explain_decision(
    agent_id: str,
    step: int,
) -> Dict[str, Any]:
    """Get explanation for an agent's decision at a specific step."""
    dashboard_requests_total.labels(endpoint="/explain", method="GET").inc()

    # Mock explainability data (in production, integrate with causal inference)
    return {
        "agent_id": agent_id,
        "step": step,
        "decision": {
            "action": 5,
            "confidence": 0.87,
        },
        "explanation": {
            "top_features": [
                {"name": "slo_violation_rate", "importance": 0.45},
                {"name": "avg_latency", "importance": 0.32},
                {"name": "throughput", "importance": 0.23},
            ],
            "counterfactuals": [
                {
                    "scenario": "If SLO violation rate was 0.01 instead of 0.05",
                    "predicted_action": 3,
                    "predicted_reward": 0.92,
                },
                {
                    "scenario": "If avg_latency was 50ms instead of 100ms",
                    "predicted_action": 7,
                    "predicted_reward": 0.85,
                },
            ],
            "causal_chain": [
                "High SLO violation rate (0.05)",
                "→ Increased scaling priority",
                "→ Action 5 (scale up 2 replicas)",
                "→ Expected reward: 0.87",
            ],
        },
    }


# ============================================================================
# Health & Metrics
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "dashboard-api",
        "timestamp": datetime.utcnow().isoformat(),
        "websocket_connections": len(manager.active_connections),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Helper Functions
# ============================================================================

def _get_mock_agent_states() -> Dict[str, Any]:
    """Generate mock agent states for testing."""
    return {
        "policy": {
            "agent_id": "policy",
            "agent_type": "DQN",
            "state_dim": 32,
            "action_dim": 10,
            "current_episode": 150,
            "total_steps": 15000,
            "epsilon": 0.15,
            "recent_reward_mean": 0.78,
            "loss": 0.023,
        },
        "consensus": {
            "agent_id": "consensus",
            "agent_type": "A2C",
            "state_dim": 16,
            "action_dim": 5,
            "current_episode": 150,
            "total_steps": 15000,
            "entropy": 0.045,
            "recent_reward_mean": 0.82,
            "value_loss": 0.015,
        },
        "ethical": {
            "agent_id": "ethical",
            "agent_type": "PPO",
            "state_dim": 24,
            "action_dim": 8,
            "current_episode": 150,
            "total_steps": 15000,
            "kl_divergence": 0.012,
            "recent_reward_mean": 0.75,
            "clip_fraction": 0.18,
        },
        "resource": {
            "agent_id": "resource",
            "agent_type": "DDPG",
            "state_dim": 20,
            "action_dim": 1,
            "current_episode": 150,
            "total_steps": 15000,
            "noise_sigma": 0.12,
            "recent_reward_mean": 0.88,
            "critic_loss": 0.031,
        },
    }


def _get_mock_agent_state(agent_id: str) -> Dict[str, Any]:
    """Get mock state for a specific agent."""
    states = _get_mock_agent_states()
    return states.get(agent_id, {})


def _get_mock_nash_equilibrium() -> Dict[str, Any]:
    """Generate mock Nash equilibrium data."""
    return {
        "converged": True,
        "iterations": 32,
        "strategy_profile": {
            "policy": [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01],
            "consensus": [0.4, 0.35, 0.15, 0.07, 0.03],
            "ethical": [0.25, 0.20, 0.18, 0.12, 0.10, 0.08, 0.05, 0.02],
            "resource": [0.7, 0.3],
        },
        "expected_payoffs": {
            "policy": 0.76,
            "consensus": 0.81,
            "ethical": 0.72,
            "resource": 0.87,
        },
        "is_pure": False,
        "method": "iterative_br",
        "timestamp": datetime.utcnow().isoformat(),
    }


def _calculate_agent_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics from agent history."""
    if not history:
        return {}

    rewards = [h.get("reward", 0) for h in history if isinstance(h, dict)]

    if not rewards:
        return {}

    return {
        "total_steps": len(history),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "recent_trend": "improving" if len(rewards) > 10 and np.mean(rewards[-10:]) > np.mean(rewards[-20:-10]) else "stable",
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "3001"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

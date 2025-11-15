"""
Meta-Consensus Optimizer Service
FastAPI service for consensus parameter optimization
"""
import os
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from optimizer import MetaConsensusOptimizer, ConsensusState, ConsensusAction

# Configuration
AGENT_PATH = os.getenv("AGENT_PATH", "/app/agent.pkl")
LATENCY_TARGET_MS = float(os.getenv("LATENCY_TARGET_MS", "300.0"))
SUCCESS_RATE_TARGET = float(os.getenv("SUCCESS_RATE_TARGET", "0.97"))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
CONSENSUS_REWARD = Gauge(
    'tars_consensus_reward',
    'Current consensus reward signal',
    ['component']
)

CONSENSUS_TIMEOUT_MS = Gauge(
    'tars_consensus_timeout_ms',
    'Current consensus timeout in milliseconds',
    ['algorithm']
)

OPTIMIZATION_ACTIONS = Counter(
    'tars_consensus_optimization_actions_total',
    'Total optimization actions taken',
    ['action_type']
)

Q_TABLE_SIZE = Gauge(
    'tars_consensus_q_table_size',
    'Size of Q-learning table'
)

# Global optimizer
optimizer: Optional[MetaConsensusOptimizer] = None

app = FastAPI(
    title="T.A.R.S. Meta-Consensus Optimizer",
    description="Reinforcement learning based consensus parameter optimization",
    version="0.8.0-alpha"
)


@app.on_event("startup")
async def startup():
    """Initialize optimizer on startup"""
    global optimizer

    logger.info("Starting Meta-Consensus Optimizer")

    optimizer = MetaConsensusOptimizer(
        agent_path=AGENT_PATH,
        latency_target_ms=LATENCY_TARGET_MS,
        success_rate_target=SUCCESS_RATE_TARGET
    )

    logger.info("Meta-Consensus Optimizer ready")


@app.on_event("shutdown")
async def shutdown():
    """Save agent on shutdown"""
    if optimizer:
        optimizer.save_agent()
        logger.info("Saved optimizer agent")


class ConsensusStateRequest(BaseModel):
    """Request with current consensus state"""
    avg_latency_ms: float
    p95_latency_ms: float
    success_rate: float
    quorum_failures: int
    total_votes: int
    algorithm: str
    current_timeout_ms: int


class OptimizationResponse(BaseModel):
    """Response with optimization recommendation"""
    action_type: str
    parameter: str
    current_value: float
    new_value: float
    delta: float
    reward_signal: float
    timestamp: str


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "meta-consensus-optimizer",
        "version": "0.8.0-alpha"
    }


@app.post("/api/v1/consensus/optimize", response_model=OptimizationResponse)
async def optimize_consensus(request: ConsensusStateRequest):
    """Optimize consensus parameters based on current state"""

    try:
        # Convert request to ConsensusState
        state = ConsensusState(
            avg_latency_ms=request.avg_latency_ms,
            p95_latency_ms=request.p95_latency_ms,
            success_rate=request.success_rate,
            quorum_failures=request.quorum_failures,
            total_votes=request.total_votes,
            algorithm=request.algorithm,
            current_timeout_ms=request.current_timeout_ms,
            timestamp=datetime.utcnow()
        )

        # Run optimization
        action = optimizer.optimize(state)

        if not action:
            raise HTTPException(status_code=500, detail="Optimization failed")

        # Calculate reward
        reward = optimizer.calculate_reward(state)

        # Update metrics
        CONSENSUS_REWARD.labels(component="latency").set(reward.latency_component)
        CONSENSUS_REWARD.labels(component="accuracy").set(reward.accuracy_component)
        CONSENSUS_REWARD.labels(component="total").set(reward.total_reward)
        CONSENSUS_TIMEOUT_MS.labels(algorithm=state.algorithm).set(action.new_value)
        OPTIMIZATION_ACTIONS.labels(action_type=action.action_type).inc()

        stats = optimizer.get_statistics()
        Q_TABLE_SIZE.set(stats["q_table_size"])

        logger.info(f"Optimization: {action.action_type} → {action.new_value:.0f}ms (reward: {reward.total_reward:.3f})")

        return OptimizationResponse(
            action_type=action.action_type,
            parameter=action.parameter,
            current_value=state.current_timeout_ms,
            new_value=action.new_value,
            delta=action.delta,
            reward_signal=reward.total_reward,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/consensus/statistics")
async def get_statistics():
    """Get optimization statistics"""

    try:
        stats = optimizer.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error fetching statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/consensus/save")
async def save_agent():
    """Manually save RL agent"""

    try:
        optimizer.save_agent()
        return {"saved": True, "path": AGENT_PATH}

    except Exception as e:
        logger.error(f"Error saving agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8092, log_level="info")

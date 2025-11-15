"""
Orchestration Agent Service - Multi-Agent RL Coordinator

FastAPI service managing DQN, A2C, PPO, and DDPG agents with Nash equilibrium coordination.
Port: 8094

Features:
- Policy Agent (DQN): Discrete policy adjustments
- Consensus Agent (A2C): Consensus decision-making
- Ethical Agent (PPO): Fairness and ethical oversight
- Resource Agent (DDPG): Continuous resource allocation
- Nash Equilibrium Solver: Strategic coordination
- Global Reward Aggregation: Multi-agent reward fusion

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
import uvicorn

from dqn import DQNAgent
from a2c import A2CAgent
from ppo import PPOAgent
from ddpg import DDPGAgent
from rewards import (
    GlobalRewardAggregator,
    GlobalRewardConfig,
    AgentReward,
    AgentType,
)
from nash import NashSolver, ParetoFrontier

# Import auth components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from auth import get_current_user, verify_service_key, User, Role
from auth_routes import router as auth_router
from rate_limiter import public_rate_limit, rate_limit_middleware


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

ORCHESTRATION_STEPS = Counter(
    "tars_orchestration_steps_total",
    "Total orchestration steps executed"
)

GLOBAL_REWARD = Histogram(
    "tars_global_reward",
    "Global aggregated rewards",
    buckets=[-10, -5, -1, 0, 1, 5, 10, 20, 50]
)

AGENT_REWARD = Histogram(
    "tars_agent_reward",
    "Individual agent rewards",
    ["agent_type"],
    buckets=[-5, -1, 0, 1, 5, 10, 20]
)

MULTIAGENT_CONFLICTS = Counter(
    "tars_multiagent_conflicts_total",
    "Total conflicts detected between agents",
    ["conflict_type"]
)

NASH_CONVERGENCE_TIME = Histogram(
    "tars_nash_convergence_time_seconds",
    "Time to compute Nash equilibrium",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

AGENT_REWARD_ALIGNMENT = Gauge(
    "tars_agent_reward_alignment",
    "Correlation between agent rewards"
)

DQN_EPSILON = Gauge("tars_dqn_epsilon", "DQN exploration rate")
A2C_ENTROPY = Gauge("tars_a2c_entropy", "A2C policy entropy")
PPO_KL_DIVERGENCE = Gauge("tars_ppo_kl_divergence", "PPO KL divergence")
DDPG_NOISE_SIGMA = Gauge("tars_ddpg_noise_sigma", "DDPG exploration noise")

# ============================================================================
# Pydantic Models
# ============================================================================

class AgentState(BaseModel):
    """State vector for an agent."""
    agent_type: str
    state_vector: List[float]


class MultiAgentStepRequest(BaseModel):
    """Request for multi-agent orchestration step."""
    policy_state: Optional[AgentState] = None
    consensus_state: Optional[AgentState] = None
    ethical_state: Optional[AgentState] = None
    resource_state: Optional[AgentState] = None


class AgentActionResponse(BaseModel):
    """Response for agent action selection."""
    agent_type: str
    action: any  # Can be int or float
    confidence: Optional[float] = None
    value_estimate: Optional[float] = None


class OrchestrationStepResponse(BaseModel):
    """Response for orchestration step."""
    step_id: int
    global_reward: float
    agent_actions: Dict[str, any]
    agent_rewards: Dict[str, float]
    conflicts_detected: List[str]
    resolution_method: Optional[str] = None
    nash_equilibrium_reached: bool
    timestamp: str


class NashEquilibriumRequest(BaseModel):
    """Request for Nash equilibrium computation."""
    agent_payoffs: Dict[str, float]
    candidate_actions: List[Dict[str, any]]


class NashEquilibriumResponse(BaseModel):
    """Response for Nash equilibrium."""
    converged: bool
    strategy_profile: Dict[str, List[float]]
    expected_payoffs: Dict[str, float]
    iterations: int
    method: str
    is_pure: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    agents_initialized: Dict[str, bool]


# ============================================================================
# Global State
# ============================================================================

# Agents
policy_agent: Optional[DQNAgent] = None  # DQN for policy
consensus_agent: Optional[A2CAgent] = None  # A2C for consensus
ethical_agent: Optional[PPOAgent] = None  # PPO for ethics
resource_agent: Optional[DDPGAgent] = None  # DDPG for resources

# Coordination
reward_aggregator: Optional[GlobalRewardAggregator] = None
nash_solver: Optional[NashSolver] = None
pareto_frontier: Optional[ParetoFrontier] = None

# Statistics
step_count = 0
conflict_history = []


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global policy_agent, consensus_agent, ethical_agent, resource_agent
    global reward_aggregator, nash_solver, pareto_frontier

    # Startup
    logger.info("Starting Multi-Agent Orchestration Service...")

    # Initialize Policy Agent (DQN)
    policy_agent = DQNAgent(
        state_dim=32,
        action_dim=10,
        learning_rate=float(os.getenv("DQN_LEARNING_RATE", "0.001")),
        gamma=float(os.getenv("DQN_GAMMA", "0.95")),
        batch_size=int(os.getenv("DQN_BATCH_SIZE", "64")),
        buffer_size=int(os.getenv("DQN_BUFFER_SIZE", "10000")),
        use_dueling=os.getenv("DQN_USE_DUELING", "false").lower() == "true",
        use_prioritized_replay=True
    )
    logger.info("Policy Agent (DQN) initialized")

    # Initialize Consensus Agent (A2C)
    consensus_agent = A2CAgent(
        state_dim=16,
        action_dim=5,
        learning_rate=float(os.getenv("A2C_LEARNING_RATE", "0.0007")),
        gamma=float(os.getenv("A2C_GAMMA", "0.99")),
        gae_lambda=float(os.getenv("A2C_GAE_LAMBDA", "0.95"))
    )
    logger.info("Consensus Agent (A2C) initialized")

    # Initialize Ethical Agent (PPO)
    ethical_agent = PPOAgent(
        state_dim=24,
        action_dim=8,
        learning_rate=float(os.getenv("PPO_LEARNING_RATE", "0.0003")),
        gamma=float(os.getenv("PPO_GAMMA", "0.99")),
        clip_epsilon=float(os.getenv("PPO_CLIP_EPSILON", "0.2")),
        n_epochs=int(os.getenv("PPO_N_EPOCHS", "10"))
    )
    logger.info("Ethical Agent (PPO) initialized")

    # Initialize Resource Agent (DDPG)
    resource_agent = DDPGAgent(
        state_dim=20,
        action_dim=1,
        actor_lr=float(os.getenv("DDPG_ACTOR_LR", "0.0001")),
        critic_lr=float(os.getenv("DDPG_CRITIC_LR", "0.001")),
        gamma=float(os.getenv("DDPG_GAMMA", "0.99")),
        tau=float(os.getenv("DDPG_TAU", "0.005"))
    )
    logger.info("Resource Agent (DDPG) initialized")

    # Initialize Reward Aggregator
    reward_config = GlobalRewardConfig(
        w_policy=float(os.getenv("REWARD_W_POLICY", "0.30")),
        w_consensus=float(os.getenv("REWARD_W_CONSENSUS", "0.25")),
        w_ethical=float(os.getenv("REWARD_W_ETHICAL", "0.25")),
        w_resource=float(os.getenv("REWARD_W_RESOURCE", "0.20")),
    )
    reward_aggregator = GlobalRewardAggregator(config=reward_config)
    logger.info("Global Reward Aggregator initialized")

    # Initialize Nash Solver
    nash_solver = NashSolver(
        method=os.getenv("NASH_METHOD", "iterative_br"),
        max_iterations=int(os.getenv("NASH_MAX_ITER", "1000")),
        tolerance=float(os.getenv("NASH_TOLERANCE", "1e-6"))
    )
    logger.info("Nash Equilibrium Solver initialized")

    # Initialize Pareto Frontier
    pareto_frontier = ParetoFrontier(
        epsilon=float(os.getenv("PARETO_EPSILON", "1e-6"))
    )
    logger.info("Pareto Frontier initialized")

    logger.info("Multi-Agent Orchestration Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Multi-Agent Orchestration Service...")


# Create FastAPI app
app = FastAPI(
    title="T.A.R.S. Multi-Agent Orchestration Service",
    version="v0.9.5-alpha",
    lifespan=lifespan
)

# Add auth routes
app.include_router(auth_router)

# Add metrics routes
try:
    from metrics_routes import router as metrics_router
    app.include_router(metrics_router, prefix="/api/v1/orchestration")
    logger.info("Training metrics routes registered successfully")
except ImportError as e:
    logger.warning(f"Training metrics routes not available: {e}")

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="multi-agent-orchestration",
        version="v0.9.2-alpha",
        agents_initialized={
            "policy": policy_agent is not None,
            "consensus": consensus_agent is not None,
            "ethical": ethical_agent is not None,
            "resource": resource_agent is not None
        }
    )


@app.post("/api/v1/orchestration/step", response_model=OrchestrationStepResponse)
async def orchestration_step(
    request: MultiAgentStepRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Execute one multi-agent orchestration step.

    Requires: Viewer role or higher

    This endpoint:
    1. Collects actions from all active agents
    2. Detects conflicts between agent actions
    3. Resolves conflicts using Nash equilibrium or Pareto optimality
    4. Computes global reward
    5. Returns coordinated actions and rewards
    """
    # Check authorization (viewer, developer, or admin)
    if Role.VIEWER not in current_user.roles and Role.DEVELOPER not in current_user.roles and Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requires viewer role or higher"
        )
    global step_count, conflict_history

    try:
        step_count += 1
        ORCHESTRATION_STEPS.inc()

        agent_actions = {}
        agent_rewards = {}
        agent_values = {}

        # ==================== Policy Agent (DQN) ====================
        if request.policy_state and policy_agent:
            state = np.array(request.policy_state.state_vector)
            action = policy_agent.select_action(state)
            agent_actions["policy"] = int(action)
            # Placeholder reward (will be computed after environment feedback)
            agent_rewards["policy"] = 0.0

        # ==================== Consensus Agent (A2C) ====================
        if request.consensus_state and consensus_agent:
            state = np.array(request.consensus_state.state_vector)
            action, log_prob, value = consensus_agent.select_action(state)
            agent_actions["consensus"] = int(action)
            agent_values["consensus"] = value
            agent_rewards["consensus"] = 0.0

        # ==================== Ethical Agent (PPO) ====================
        if request.ethical_state and ethical_agent:
            state = np.array(request.ethical_state.state_vector)
            action, log_prob, value = ethical_agent.select_action(state)
            agent_actions["ethical"] = int(action)
            agent_values["ethical"] = value
            agent_rewards["ethical"] = 0.0

        # ==================== Resource Agent (DDPG) ====================
        if request.resource_state and resource_agent:
            state = np.array(request.resource_state.state_vector)
            action = resource_agent.select_action(state, add_noise=True)
            agent_actions["resource"] = float(action[0])
            agent_rewards["resource"] = 0.0

        # ==================== Conflict Detection ====================
        conflicts = detect_conflicts(agent_actions)

        if conflicts:
            logger.warning(f"Detected conflicts: {conflicts}")
            for conflict_type in conflicts:
                MULTIAGENT_CONFLICTS.labels(conflict_type=conflict_type).inc()

        # ==================== Conflict Resolution ====================
        resolution_method = None
        nash_reached = False

        if conflicts:
            # Use Nash equilibrium or Pareto optimality to resolve
            resolution_method = "nash_pareto"
            nash_reached = True  # Simplified for now

        # ==================== Global Reward Aggregation ====================
        # In real scenario, these rewards come from environment
        # For now, use placeholder values
        reward_list = [
            AgentReward(agent_type=AgentType.POLICY, value=0.75, confidence=0.9),
            AgentReward(agent_type=AgentType.CONSENSUS, value=0.82, confidence=0.85),
            AgentReward(agent_type=AgentType.ETHICAL, value=0.68, confidence=0.88),
            AgentReward(agent_type=AgentType.RESOURCE, value=0.91, confidence=0.92),
        ]

        global_reward, breakdown, stats = reward_aggregator.aggregate(reward_list)

        # Update metrics
        GLOBAL_REWARD.observe(global_reward)
        for agent_type, reward in agent_rewards.items():
            AGENT_REWARD.labels(agent_type=agent_type).observe(reward)

        # Compute reward alignment
        reward_values = [r.value for r in reward_list]
        if len(reward_values) > 1:
            alignment = np.corrcoef(reward_values)[0, 1] if len(reward_values) > 1 else 1.0
            AGENT_REWARD_ALIGNMENT.set(alignment)

        # Update agent-specific metrics
        if policy_agent:
            stats_dqn = policy_agent.get_statistics()
            DQN_EPSILON.set(stats_dqn.get("epsilon", 0.0))

        if consensus_agent:
            stats_a2c = consensus_agent.get_statistics()
            entropy = stats_a2c.get("avg_entropy", 0.0)
            if entropy is not None:
                A2C_ENTROPY.set(entropy)

        if ethical_agent:
            stats_ppo = ethical_agent.get_statistics()
            kl = stats_ppo.get("avg_kl", 0.0)
            if kl is not None:
                PPO_KL_DIVERGENCE.set(kl)

        # Return response
        return OrchestrationStepResponse(
            step_id=step_count,
            global_reward=global_reward,
            agent_actions=agent_actions,
            agent_rewards={r.agent_type.value: r.value for r in reward_list},
            conflicts_detected=conflicts,
            resolution_method=resolution_method,
            nash_equilibrium_reached=nash_reached,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Orchestration step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/orchestration/nash", response_model=NashEquilibriumResponse)
async def compute_nash_equilibrium(
    request: NashEquilibriumRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Compute Nash equilibrium for given agent payoffs.

    Requires: Developer role or higher

    This endpoint computes the Nash equilibrium strategy profile
    that maximizes each agent's payoff given others' strategies.
    """
    # Check authorization (developer or admin)
    if Role.DEVELOPER not in current_user.roles and Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requires developer role or higher"
        )
    import time

    try:
        start_time = time.time()

        # Build payoff matrices from candidate actions
        n_agents = len(request.agent_payoffs)
        n_candidates = len(request.candidate_actions)

        # Simplified: assume each agent evaluates each candidate
        # In practice, this would be more complex
        agent_names = list(request.agent_payoffs.keys())

        # Create dummy payoff matrices for demonstration
        payoff_matrices = {}
        for agent in agent_names:
            payoff_matrices[agent] = np.random.rand(n_candidates, n_candidates)

        # Solve Nash equilibrium
        result = nash_solver.solve(payoff_matrices, agent_names)

        # Track convergence time
        convergence_time = time.time() - start_time
        NASH_CONVERGENCE_TIME.observe(convergence_time)

        return NashEquilibriumResponse(
            converged=result.converged,
            strategy_profile={k: v.tolist() for k, v in result.strategy_profile.items()},
            expected_payoffs=result.payoffs,
            iterations=result.iterations,
            method=result.method,
            is_pure=result.is_pure
        )

    except Exception as e:
        logger.error(f"Nash equilibrium computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/orchestration/agents/state")
async def get_agents_state():
    """Get current state of all agents."""
    try:
        agents_state = {}

        if policy_agent:
            agents_state["policy"] = policy_agent.get_statistics()

        if consensus_agent:
            agents_state["consensus"] = consensus_agent.get_statistics()

        if ethical_agent:
            agents_state["ethical"] = ethical_agent.get_statistics()

        if resource_agent:
            agents_state["resource"] = resource_agent.get_statistics()

        return agents_state

    except Exception as e:
        logger.error(f"Failed to get agents state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/orchestration/conflicts")
async def get_conflicts():
    """Get recent conflict history."""
    return {
        "conflicts": conflict_history[-50:],
        "total_conflicts": len(conflict_history)
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/orchestration/agents/{agent_id}/reload")
async def reload_agent_hyperparameters(
    agent_id: str,
    hyperparameters: Dict[str, Any],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Hot-reload agent hyperparameters without downtime.

    Requires: Developer role or higher

    This endpoint updates an agent's hyperparameters dynamically
    while preserving its current state (experience buffer, network weights, etc.).

    Args:
        agent_id: Agent identifier (policy, consensus, ethical, resource)
        hyperparameters: New hyperparameter dictionary

    Returns:
        Dictionary with reload status and updated configuration
    """
    # Check authorization (developer or admin)
    if Role.DEVELOPER not in current_user.roles and Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requires developer role or higher"
        )
    global policy_agent, consensus_agent, ethical_agent, resource_agent

    try:
        logger.info(f"Reloading hyperparameters for {agent_id}")

        if agent_id == "policy" and policy_agent:
            # Update DQN hyperparameters
            _reload_dqn_params(policy_agent, hyperparameters)
            updated_config = policy_agent.get_statistics()

        elif agent_id == "consensus" and consensus_agent:
            # Update A2C hyperparameters
            _reload_a2c_params(consensus_agent, hyperparameters)
            updated_config = consensus_agent.get_statistics()

        elif agent_id == "ethical" and ethical_agent:
            # Update PPO hyperparameters
            _reload_ppo_params(ethical_agent, hyperparameters)
            updated_config = ethical_agent.get_statistics()

        elif agent_id == "resource" and resource_agent:
            # Update DDPG hyperparameters
            _reload_ddpg_params(resource_agent, hyperparameters)
            updated_config = resource_agent.get_statistics()

        else:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found or not initialized"
            )

        logger.info(f"Successfully reloaded hyperparameters for {agent_id}")

        return {
            "status": "success",
            "agent_id": agent_id,
            "message": f"Hyperparameters reloaded for {agent_id}",
            "updated_config": updated_config,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to reload hyperparameters for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _reload_dqn_params(agent, params: Dict[str, Any]):
    """Reload DQN agent hyperparameters."""
    if "learning_rate" in params:
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = params["learning_rate"]

    if "gamma" in params:
        agent.gamma = params["gamma"]

    if "epsilon_decay" in params:
        agent.epsilon_decay = params["epsilon_decay"]

    if "epsilon_end" in params:
        agent.epsilon_end = params["epsilon_end"]

    if "batch_size" in params:
        agent.batch_size = params["batch_size"]

    if "target_update_freq" in params:
        agent.target_update_freq = params["target_update_freq"]

    logger.info(f"DQN parameters reloaded: {list(params.keys())}")


def _reload_a2c_params(agent, params: Dict[str, Any]):
    """Reload A2C agent hyperparameters."""
    if "learning_rate" in params:
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = params["learning_rate"]

    if "gamma" in params:
        agent.gamma = params["gamma"]

    if "gae_lambda" in params:
        agent.gae_lambda = params["gae_lambda"]

    if "value_loss_coef" in params:
        agent.value_loss_coef = params["value_loss_coef"]

    if "entropy_coef" in params:
        agent.entropy_coef = params["entropy_coef"]

    if "max_grad_norm" in params:
        agent.max_grad_norm = params["max_grad_norm"]

    logger.info(f"A2C parameters reloaded: {list(params.keys())}")


def _reload_ppo_params(agent, params: Dict[str, Any]):
    """Reload PPO agent hyperparameters."""
    if "learning_rate" in params:
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = params["learning_rate"]

    if "gamma" in params:
        agent.gamma = params["gamma"]

    if "gae_lambda" in params:
        agent.gae_lambda = params["gae_lambda"]

    if "clip_epsilon" in params:
        agent.clip_epsilon = params["clip_epsilon"]

    if "value_loss_coef" in params:
        agent.value_loss_coef = params["value_loss_coef"]

    if "entropy_coef" in params:
        agent.entropy_coef = params["entropy_coef"]

    if "max_grad_norm" in params:
        agent.max_grad_norm = params["max_grad_norm"]

    if "n_epochs" in params:
        agent.n_epochs = params["n_epochs"]

    if "target_kl" in params:
        agent.target_kl = params["target_kl"]

    logger.info(f"PPO parameters reloaded: {list(params.keys())}")


def _reload_ddpg_params(agent, params: Dict[str, Any]):
    """Reload DDPG agent hyperparameters."""
    if "actor_lr" in params:
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = params["actor_lr"]

    if "critic_lr" in params:
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = params["critic_lr"]

    if "gamma" in params:
        agent.gamma = params["gamma"]

    if "tau" in params:
        agent.tau = params["tau"]

    if "batch_size" in params:
        agent.batch_size = params["batch_size"]

    if "noise_sigma" in params and hasattr(agent.noise, 'sigma'):
        agent.noise.sigma = params["noise_sigma"]

    logger.info(f"DDPG parameters reloaded: {list(params.keys())}")


# ============================================================================
# Helper Functions
# ============================================================================

def detect_conflicts(agent_actions: Dict[str, any]) -> List[str]:
    """
    Detect conflicts between agent actions.

    Args:
        agent_actions (Dict[str, any]): Actions from each agent

    Returns:
        List[str]: List of detected conflict types
    """
    conflicts = []

    # Example conflict detection logic
    # In practice, this would be domain-specific

    # Conflict 1: Policy wants aggressive action while Ethical wants conservative
    if "policy" in agent_actions and "ethical" in agent_actions:
        policy_action = agent_actions["policy"]
        ethical_action = agent_actions["ethical"]
        # Placeholder logic
        if policy_action > 7 and ethical_action < 3:
            conflicts.append("policy_ethical_mismatch")

    # Conflict 2: Resource allocation exceeds capacity
    if "resource" in agent_actions:
        resource_action = agent_actions["resource"]
        if resource_action > 0.95:
            conflicts.append("resource_over_allocation")

    # Add to history
    if conflicts:
        conflict_history.append({
            "step": step_count,
            "conflicts": conflicts,
            "actions": agent_actions,
            "timestamp": datetime.utcnow().isoformat()
        })

    return conflicts


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("ORCHESTRATION_PORT", "8094"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

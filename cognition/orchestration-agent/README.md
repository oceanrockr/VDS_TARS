# T.A.R.S. Cognitive Orchestration Agent

Multi-Agent Reinforcement Learning coordinator for T.A.R.S. Phase 11.

## Overview

The Orchestration Agent coordinates 4 specialized RL agents:

1. **Policy Agent** (DQN): Optimizes policy parameters
2. **Consensus Agent** (A2C): Tunes consensus timeouts and quorum
3. **Ethical Agent** (PPO): Maximizes fairness across demographics
4. **Resource Agent** (DDPG): Optimizes resource allocation and cost

## Architecture

```
┌─────────────────────────────────────────┐
│     Orchestration Agent (Port 8094)      │
│  ┌───────────────────────────────────┐  │
│  │  Nash Equilibrium Solver          │  │
│  │  Conflict Resolution Engine       │  │
│  │  Shared Reward Aggregator         │  │
│  └───────────────────────────────────┘  │
└────────┬────────┬────────┬────────┬─────┘
         │        │        │        │
         ↓        ↓        ↓        ↓
    ┌────────┐┌────────┐┌────────┐┌────────┐
    │ Policy ││Consensus││Ethical││Resource│
    │  DQN   ││  A2C   ││  PPO   ││  DDPG  │
    └────────┘└────────┘└────────┘└────────┘
```

## API Endpoints

### POST `/api/v1/orchestration/step`

Execute one orchestration step with agent states.

**Request**:
```json
{
  "policy_state": {
    "agent_id": "policy-001",
    "agent_type": "policy",
    "state_vector": [0.25, 0.12, ...],
    "last_reward": 0.75
  },
  "consensus_state": {...},
  "ethical_state": {...},
  "resource_state": {...}
}
```

**Response**:
```json
{
  "step_id": 42,
  "global_reward": 0.79,
  "agent_actions": {
    "policy": "adjust_threshold_5%",
    "consensus": "decrease_timeout_10",
    "ethical": "increase_diversity",
    "resource": "scale_replicas:0.85"
  },
  "conflicts_detected": ["policy_resource_scaling_conflict"],
  "resolution_method": "pareto_optimal",
  "nash_equilibrium_reached": true
}
```

### POST `/api/v1/orchestration/nash`

Compute Nash equilibrium for agent strategy profile.

### GET `/api/v1/orchestration/agents`

Get current agent states and configurations.

### GET `/api/v1/orchestration/statistics`

Get orchestration statistics.

## Environment Variables

```bash
PORT=8094
LOG_LEVEL=info

# RL Hyperparameters
POLICY_LEARNING_RATE=0.0003
CONSENSUS_LEARNING_RATE=0.0007
ETHICAL_LEARNING_RATE=0.0003
RESOURCE_LEARNING_RATE=0.0001

# Orchestration Parameters
REWARD_WEIGHT_POLICY=0.3
REWARD_WEIGHT_CONSENSUS=0.25
REWARD_WEIGHT_ETHICAL=0.3
REWARD_WEIGHT_RESOURCE=0.15
CONFLICT_PENALTY=0.5
```

## Development Status

**Phase 11.1** (Current):
- ✅ FastAPI skeleton
- ✅ API endpoint stubs
- ✅ Prometheus metrics
- 🚧 DQN Policy Agent (TODO)
- 🚧 Nash equilibrium solver (TODO)
- 🚧 Conflict resolution (TODO)

**Phase 11.2** (Next):
- A2C Consensus Agent
- PPO Ethical Agent
- DDPG Resource Agent
- Full multi-agent integration

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py

# Test endpoint
curl http://localhost:8094/api/v1/orchestration/agents
```

## Running in Docker

```bash
# Build image
docker build -t tars-orchestration-agent:0.9.0 .

# Run container
docker run -p 8094:8094 tars-orchestration-agent:0.9.0
```

## Metrics

- `tars_orchestration_steps_total`: Total orchestration steps executed
- `tars_nash_equilibrium_convergence_seconds`: Time to Nash equilibrium
- `tars_agent_rewards{agent_type}`: Individual agent rewards
- `tars_conflicts_detected_total{conflict_type}`: Conflicts detected
- `tars_global_reward`: Global orchestration reward signal

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Nash Equilibrium (NashPy)](https://nashpy.readthedocs.io/)
- [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)

# T.A.R.S. Phase 11.3 Quickstart Guide
## AutoML Pipeline & Cognitive Orchestration Dashboard

**Version**: v0.9.3-alpha
**Date**: November 14, 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [AutoML Pipeline Usage](#automl-pipeline-usage)
5. [Dashboard Usage](#dashboard-usage)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.9+ (3.10 recommended)
- **Node.js**: 18+ (for dashboard frontend)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for MLflow artifacts

### Python Dependencies

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
fastapi>=0.115.0
uvicorn>=0.32.0

# AutoML
optuna>=3.4.0
featuretools>=1.28.0
mlflow>=2.9.0
scikit-learn>=1.5.2

# Dashboard API
websockets>=12.0
httpx>=0.25.0
```

### Node.js Dependencies

```bash
# Frontend
react>=18.2.0
typescript>=5.0.0
@mui/material>=5.15.0
recharts>=2.10.0
d3>=7.8.5
```

---

## Installation

### 1. Install Python Dependencies

```bash
# Navigate to AutoML pipeline directory
cd cognition/automl-pipeline
pip install -r requirements.txt

# Install dashboard API dependencies
cd ../../dashboard/api
pip install fastapi uvicorn websockets httpx prometheus-client
```

### 2. Install Frontend Dependencies

```bash
# Navigate to dashboard frontend
cd ../frontend
npm install
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and configure Phase 11.3 variables
nano .env
```

**Required Environment Variables**:
```bash
# AutoML
AUTOML_ENABLED=true
AUTOML_PORT=8097
OPTUNA_N_TRIALS=100
FEATURETOOLS_DEPTH=2
MLFLOW_TRACKING_URI=/mnt/data/mlflow

# Dashboard
DASHBOARD_ENABLED=true
DASHBOARD_PORT=3000
DASHBOARD_API_PORT=3001
DASHBOARD_REFRESH_INTERVAL=5
ORCHESTRATION_SERVICE_URL=http://localhost:8094
AUTOML_SERVICE_URL=http://localhost:8097
```

---

## Quick Start

### Terminal 1: Start Orchestration Service (Phase 11.2)

```bash
cd cognition/orchestration-agent
python main.py
# Listening on http://localhost:8094
```

### Terminal 2: Start AutoML Pipeline

```bash
cd cognition/automl-pipeline
python service.py
# Listening on http://localhost:8097
```

### Terminal 3: Start Dashboard API

```bash
cd dashboard/api
python main.py
# Listening on http://localhost:3001
```

### Terminal 4: Start Dashboard Frontend

```bash
cd dashboard/frontend
npm start
# Listening on http://localhost:3000
# Opens browser automatically
```

### Verify Services

```bash
# Check orchestration service
curl http://localhost:8094/health

# Check AutoML service
curl http://localhost:8097/health

# Check dashboard API
curl http://localhost:3001/health
```

---

## AutoML Pipeline Usage

### 1. Optimize DQN Agent Hyperparameters

**Start Optimization**:
```bash
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 100,
    "timeout_seconds": 600,
    "sampler_type": "tpe",
    "pruner_type": "median",
    "register_model": true
  }'

# Response:
# {
#   "optimization_id": "dqn_1731600000",
#   "status": "pending",
#   "message": "Optimization task started for dqn"
# }
```

**Check Status**:
```bash
curl http://localhost:8097/api/v1/optimize/dqn_1731600000

# Response (when completed):
# {
#   "status": "completed",
#   "result": {
#     "best_params": {
#       "learning_rate": 0.0015,
#       "gamma": 0.98,
#       "epsilon_decay": 0.996,
#       ...
#     },
#     "best_score": 0.89,
#     "n_trials": 100,
#     "mlflow_run_id": "abc123..."
#   }
# }
```

### 2. Optimize All Agents

```bash
# A2C Consensus Agent
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "a2c", "n_trials": 50}'

# PPO Ethical Agent
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "ppo", "n_trials": 50}'

# DDPG Resource Agent
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "ddpg", "n_trials": 50}'
```

### 3. Generate Features

```bash
curl -X POST http://localhost:8097/api/v1/features \
  -H "Content-Type: application/json" \
  -d '{
    "feature_type": "temporal",
    "data_source": "agent_rewards.csv",
    "max_depth": 2,
    "max_features": 100
  }'

# Response:
# {
#   "feature_type": "temporal",
#   "n_features": 87,
#   "feature_names": [
#     "reward_rolling_mean_5",
#     "reward_rolling_std_10",
#     "reward_momentum_20",
#     ...
#   ],
#   "elapsed_time_seconds": 2.34
# }
```

### 4. Model Registry Operations

**List Best Models**:
```bash
curl http://localhost:8097/api/v1/models?agent_type=dqn
```

**Register Model**:
```bash
curl -X POST http://localhost:8097/api/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "abc123...",
    "model_name": "dqn_policy_agent",
    "description": "Optimized DQN for policy orchestration"
  }'
```

**Promote to Production**:
```bash
curl -X POST http://localhost:8097/api/v1/models/promote \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "dqn_policy_agent",
    "version": "1",
    "stage": "Production"
  }'
```

---

## Dashboard Usage

### 1. Access Dashboard

Navigate to: **http://localhost:3000**

### 2. Overview Page

- View all 4 agents (Policy, Consensus, Ethical, Resource)
- Real-time reward plots
- Live metrics updates (via WebSocket)
- Agent status indicators (connected/training/idle)

### 3. Run What-If Simulation

```bash
curl -X POST http://localhost:3001/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "high_load",
    "agent_states": {
      "policy": [0.1, 0.2, ..., 0.5],
      "consensus": [0.3, 0.1, ..., 0.4],
      "ethical": [0.2, 0.3, ..., 0.6],
      "resource": [0.5, 0.7, ..., 0.8]
    },
    "steps": 10,
    "interventions": []
  }'

# Response:
# {
#   "scenario": "high_load",
#   "results": {
#     "trajectory": [...],
#     "total_conflicts": 2,
#     "avg_reward": 0.78
#   },
#   "elapsed_time_seconds": 0.15
# }
```

### 4. Get Agent Explanation

```bash
curl http://localhost:3001/api/v1/explain/policy/decision?step=100

# Response:
# {
#   "agent_id": "policy",
#   "step": 100,
#   "decision": {"action": 5, "confidence": 0.87},
#   "explanation": {
#     "top_features": [
#       {"name": "slo_violation_rate", "importance": 0.45},
#       {"name": "avg_latency", "importance": 0.32}
#     ],
#     "counterfactuals": [...],
#     "causal_chain": [...]
#   }
# }
```

### 5. WebSocket Live Updates

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:3001/ws/live');

ws.onopen = () => {
  console.log('Connected to T.A.R.S. Dashboard');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'init') {
    // Initial agent states
    console.log('Initial state:', message.data);
  } else if (message.type === 'update') {
    // Periodic updates every 5 seconds
    console.log('Live update:', message.data);
  }
};

// Send ping to keep connection alive
setInterval(() => {
  ws.send('ping');
}, 30000);
```

---

## API Reference

### AutoML Service (Port 8097)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/optimize` | POST | Start hyperparameter optimization |
| `/api/v1/optimize/{id}` | GET | Get optimization status |
| `/api/v1/features` | POST | Generate features |
| `/api/v1/models` | GET | List models |
| `/api/v1/models/register` | POST | Register model |
| `/api/v1/models/promote` | POST | Promote model to stage |
| `/api/v1/stats` | GET | Service statistics |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### Dashboard API (Port 3001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/agents` | GET | Get all agent states |
| `/api/v1/agents/{id}` | GET | Get agent details |
| `/api/v1/nash-equilibrium` | GET | Nash equilibrium |
| `/api/v1/conflicts` | GET | Conflict history |
| `/api/v1/metrics` | GET | Metrics summary |
| `/api/v1/simulate` | POST | Run what-if simulation |
| `/api/v1/explain/{id}/decision` | GET | Explain decision |
| `/ws/live` | WebSocket | Real-time updates |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

---

## Troubleshooting

### AutoML Service Won't Start

**Error**: `ModuleNotFoundError: No module named 'optuna'`

**Solution**:
```bash
cd cognition/automl-pipeline
pip install -r requirements.txt
```

---

### Dashboard WebSocket Disconnects

**Error**: `WebSocket connection failed`

**Solution**:
1. Check dashboard API is running on port 3001
2. Verify `REACT_APP_WS_URL` in frontend `.env`:
   ```bash
   REACT_APP_WS_URL=ws://localhost:3001/ws/live
   ```
3. Check CORS settings in dashboard API

---

### MLflow Tracking URI Error

**Error**: `Failed to create MLflow experiment`

**Solution**:
1. Create MLflow directory:
   ```bash
   mkdir -p /mnt/data/mlflow
   ```
2. Or use local directory in `.env`:
   ```bash
   MLFLOW_TRACKING_URI=./mlruns
   ```

---

### Optimization Stuck in "Pending"

**Error**: Optimization never completes

**Solution**:
1. Check AutoML service logs for errors
2. Verify objective function is not timing out
3. Reduce `n_trials` or increase `timeout_seconds`

---

### Dashboard Shows No Agents

**Error**: Dashboard displays empty agent cards

**Solution**:
1. Verify orchestration service is running (port 8094)
2. Check `ORCHESTRATION_SERVICE_URL` in dashboard API `.env`
3. Curl orchestration service directly:
   ```bash
   curl http://localhost:8094/api/v1/orchestration/agents/state
   ```

---

## Performance Tips

### 1. Optimize Hyperparameter Search

- Use **TPE sampler** for most agents (default)
- Use **CMA-ES** for continuous optimization (DDPG)
- Enable **pruning** to stop bad trials early:
  ```json
  {"pruner_type": "median"}  // or "hyperband"
  ```
- Reduce `n_trials` for faster results (50 vs 100)

### 2. Speed Up Feature Generation

- Reduce `max_depth` to 1 for faster DFS
- Limit `max_features` to 50 for testing
- Use smaller data samples for feature engineering

### 3. Improve Dashboard Performance

- Increase `DASHBOARD_REFRESH_INTERVAL` to 10 seconds
- Limit agent history to 100 datapoints
- Use Redis backend for state storage (future)

---

## Example Workflow

### Complete AutoML → Dashboard Pipeline

```bash
# 1. Start all services
tmux new-session -d -s tars "cd cognition/orchestration-agent && python main.py"
tmux new-window -t tars "cd cognition/automl-pipeline && python service.py"
tmux new-window -t tars "cd dashboard/api && python main.py"
tmux new-window -t tars "cd dashboard/frontend && npm start"

# 2. Wait for services to start
sleep 10

# 3. Optimize DQN agent
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "dqn", "n_trials": 50}'

# 4. Monitor optimization status
watch -n 5 'curl -s http://localhost:8097/api/v1/optimize/dqn_* | jq .status'

# 5. View results in dashboard
open http://localhost:3000

# 6. Run what-if simulation
curl -X POST http://localhost:3001/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d @simulation_config.json

# 7. View Prometheus metrics
curl http://localhost:8097/metrics
curl http://localhost:3001/metrics
```

---

## Next Steps

1. **Integrate with Orchestration Service**:
   - Implement objective function with real agent training
   - Add automatic hyperparameter updates

2. **Deploy to Kubernetes**:
   - Create Helm charts for AutoML + Dashboard
   - Configure persistent volumes for MLflow

3. **Add Authentication**:
   - Implement OAuth2/JWT for dashboard
   - Secure AutoML API endpoints

4. **Run Benchmarks**:
   - Validate hyperparameter optimization gains
   - Test dashboard latency under load
   - Measure feature generation speed

---

## Support

**Documentation**: [PHASE11_3_IMPLEMENTATION_REPORT.md](PHASE11_3_IMPLEMENTATION_REPORT.md)
**Issues**: GitHub Issues
**Logs**: Check `./logs/` directory for service logs

---

**Status**: ✅ **Phase 11.3 Ready for Deployment**

🚀 **Enjoy optimizing your multi-agent RL system with T.A.R.S. AutoML + Dashboard!**

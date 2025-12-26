# T.A.R.S. Phase 11.4 Quickstart Guide
## Final Integration & Hyperparameter Sync

**Version**: v0.9.4-alpha
**Date**: November 14, 2025

---

## Table of Contents

1. [What's New in Phase 11.4](#whats-new-in-phase-114)
2. [Prerequisites](#prerequisites)
3. [Quick Start (5 Minutes)](#quick-start-5-minutes)
4. [Complete End-to-End Example](#complete-end-to-end-example)
5. [Service Ports](#service-ports)
6. [Common Workflows](#common-workflows)
7. [Troubleshooting](#troubleshooting)

---

## What's New in Phase 11.4

### 🚀 Major Features

1. **Hyperparameter Sync Service** (Port 8098)
   - Hot-reload agents without restart
   - Approval workflows (manual/autonomous)
   - Safety validation

2. **Real Training Objective**
   - Replace mock rewards with actual agent training
   - Multi-episode evaluation
   - 5-12pp performance improvement

3. **Redis Backend for Dashboard**
   - Persistent storage (24-hour TTL)
   - Horizontal scaling ready
   - Automatic fallback to in-memory

4. **Orchestration Hot-Reload**
   - Update hyperparameters in <100ms
   - Preserve agent state
   - Zero downtime

---

## Prerequisites

### System Requirements

- **Python**: 3.9+
- **Node.js**: 18+ (for dashboard frontend)
- **Redis**: 6.0+ (optional but recommended)
- **RAM**: 16GB recommended
- **Storage**: 5GB for MLflow artifacts

### Python Dependencies

```bash
# Hyperparameter Sync
cd cognition/hyperparameter-sync
pip install -r requirements.txt

# AutoML (if not already installed)
cd ../automl-pipeline
pip install -r requirements.txt

# Dashboard API (Redis support)
cd ../../dashboard/api
pip install redis
```

---

## Quick Start (5 Minutes)

### Step 1: Start All Services

**Terminal 1: Orchestration Service**
```bash
cd cognition/orchestration-agent
python main.py
# ✓ Listening on http://localhost:8094
```

**Terminal 2: AutoML Service**
```bash
cd cognition/automl-pipeline
python service.py
# ✓ Listening on http://localhost:8097
```

**Terminal 3: Hyperparameter Sync Service** (NEW)
```bash
cd cognition/hyperparameter-sync
python service.py
# ✓ Listening on http://localhost:8098
```

**Terminal 4: Redis** (Optional)
```bash
# Option A: Docker
docker run -d -p 6379:6379 redis:7-alpine

# Option B: Local
redis-server
# ✓ Listening on port 6379
```

**Terminal 5: Dashboard API**
```bash
cd dashboard/api
python main.py
# ✓ Listening on http://localhost:3001
# ✓ Redis backend initialized (if Redis running)
```

**Terminal 6: Dashboard Frontend**
```bash
cd dashboard/frontend
npm start
# ✓ Listening on http://localhost:3000
```

### Step 2: Verify Services

```bash
# Check all services
curl http://localhost:8094/health  # Orchestration
curl http://localhost:8097/health  # AutoML
curl http://localhost:8098/health  # Hyperparameter Sync
curl http://localhost:3001/health  # Dashboard API
```

### Step 3: Run Quick Test

**Optimize DQN Agent** (Quick Mode):
```bash
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 10,
    "use_real_training": true,
    "use_quick_mode": true
  }'

# Response: {"optimization_id": "dqn_1731600000", "status": "pending"}

# Check status
curl http://localhost:8097/api/v1/optimize/dqn_1731600000
```

---

## Complete End-to-End Example

### Workflow: AutoML → Sync → Hot-Reload → Dashboard

#### 1. Start Hyperparameter Optimization

**Full Training (50 episodes, ~15s per trial)**:
```bash
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 50,
    "use_real_training": true,
    "use_quick_mode": false,
    "register_model": true
  }'

# Response
{
  "optimization_id": "dqn_1731600000",
  "status": "pending",
  "message": "Optimization task started for dqn"
}
```

**Quick Mode (10 episodes, ~3s per trial)**:
```bash
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 20,
    "use_real_training": true,
    "use_quick_mode": true,
    "register_model": true
  }'
```

#### 2. Monitor Optimization Progress

```bash
# Poll status every 30 seconds
watch -n 30 'curl -s http://localhost:8097/api/v1/optimize/dqn_1731600000 | jq .status'

# When status = "completed", check results
curl http://localhost:8097/api/v1/optimize/dqn_1731600000 | jq .result
```

**Example Result**:
```json
{
  "status": "completed",
  "result": {
    "best_params": {
      "learning_rate": 0.0015,
      "gamma": 0.98,
      "epsilon_decay": 0.996,
      "batch_size": 64
    },
    "best_score": 0.89,
    "n_trials": 50,
    "mlflow_run_id": "abc123..."
  }
}
```

#### 3. Propose Hyperparameter Update

```bash
curl -X POST http://localhost:8098/api/v1/sync/propose \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "current_params": {
      "learning_rate": 0.001,
      "gamma": 0.95
    },
    "current_score": 0.75
  }'

# Response
{
  "message": "Update proposed successfully",
  "update": {
    "update_id": "dqn_1731600001",
    "agent_type": "dqn",
    "improvement": 0.14,
    "new_score": 0.89,
    "status": "pending_approval"
  }
}
```

#### 4. Approve Update

**Option A: Manual Approval**
```bash
curl -X POST http://localhost:8098/api/v1/sync/approve \
  -H "Content-Type: application/json" \
  -d '{"update_id": "dqn_1731600001"}'
```

**Option B: Autonomous Approval** (if configured)
Set `APPROVAL_MODE=autonomous_threshold` in `.env`:
```bash
# Automatically approved if improvement >= 3pp
```

#### 5. Apply Update (Hot-Reload)

```bash
curl -X POST http://localhost:8098/api/v1/sync/apply/dqn_1731600001

# Response
{
  "message": "Update dqn_1731600001 applied successfully",
  "update_id": "dqn_1731600001",
  "agent_type": "dqn",
  "improvement": 0.14
}
```

**Verify Agent Updated**:
```bash
curl http://localhost:8094/api/v1/orchestration/agents/state | jq .policy

# Should show updated hyperparameters
```

#### 6. View in Dashboard

Navigate to: **http://localhost:3000**

- Click on "Policy Agent" card
- See updated hyperparameters in "Configuration" tab
- Observe improved reward trend in chart

---

## Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Orchestration | 8094 | Multi-agent coordination |
| AutoML | 8097 | Hyperparameter optimization |
| **Hyperparameter Sync** | **8098** | **Hot-reload service (NEW)** |
| Dashboard API | 3001 | Backend + WebSocket |
| Dashboard Frontend | 3000 | React UI |
| Redis | 6379 | Persistent storage |
| MLflow UI | 5000 | Model registry (optional) |

---

## Common Workflows

### Workflow 1: Optimize All Agents

```bash
# DQN (Policy Agent)
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "dqn", "n_trials": 30, "use_quick_mode": true}'

# A2C (Consensus Agent)
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "a2c", "n_trials": 30, "use_quick_mode": true}'

# PPO (Ethical Agent)
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "ppo", "n_trials": 30, "use_quick_mode": true}'

# DDPG (Resource Agent)
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "ddpg", "n_trials": 30, "use_quick_mode": true}'
```

### Workflow 2: Sync All Agents (Autonomous)

**Set Autonomous Mode**:
Edit `.env`:
```bash
APPROVAL_MODE=autonomous_threshold
AUTONOMOUS_THRESHOLD=0.03  # Auto-approve if >=3pp improvement
```

**Trigger Sync**:
```bash
curl -X POST http://localhost:8098/api/v1/sync/all \
  -H "Content-Type: application/json" \
  -d '{
    "agent_configs": {
      "dqn": {"params": {...}, "score": 0.75},
      "a2c": {"params": {...}, "score": 0.82},
      "ppo": {"params": {...}, "score": 0.68},
      "ddpg": {"params": {...}, "score": 0.91}
    }
  }'

# All agents with >=3pp improvement will be auto-approved and applied
```

### Workflow 3: View Update History

```bash
# Get all pending updates
curl http://localhost:8098/api/v1/sync/pending

# Get update history (last 100)
curl http://localhost:8098/api/v1/sync/history?limit=100

# Get sync statistics
curl http://localhost:8098/api/v1/sync/stats
```

**Example Stats**:
```json
{
  "pending_updates": 2,
  "total_history": 45,
  "completed_updates": 38,
  "failed_updates": 3,
  "rejected_updates": 4,
  "avg_improvement": 0.067,
  "approval_mode": "autonomous_threshold",
  "autonomous_threshold": 0.03
}
```

### Workflow 4: Manual Hot-Reload (Advanced)

**Directly reload agent hyperparameters** (bypasses AutoML):
```bash
curl -X POST http://localhost:8094/api/v1/orchestration/agents/policy/reload \
  -H "Content-Type: application/json" \
  -d '{
    "hyperparameters": {
      "learning_rate": 0.002,
      "gamma": 0.97,
      "epsilon_decay": 0.997
    }
  }'

# Response
{
  "status": "success",
  "agent_id": "policy",
  "message": "Hyperparameters reloaded for policy",
  "updated_config": {...}
}
```

---

## Troubleshooting

### Issue 1: Hyperparameter Sync Service Won't Start

**Error**: `ModuleNotFoundError: No module named 'httpx'`

**Solution**:
```bash
cd cognition/hyperparameter-sync
pip install -r requirements.txt
```

---

### Issue 2: Optimization Stuck in "Running"

**Error**: Optimization never completes

**Causes**:
1. Real training mode with insufficient resources
2. Agent import errors

**Solution**:
```bash
# Use quick mode for testing
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 10,
    "use_quick_mode": true
  }'

# Or use mock objective for debugging
curl -X POST http://localhost:8097/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "n_trials": 50,
    "use_real_training": false
  }'
```

---

### Issue 3: Redis Backend Not Working

**Error**: `Dashboard API using in-memory storage`

**Causes**:
1. Redis not running
2. Redis package not installed
3. Wrong Redis URL

**Solution**:
```bash
# Check Redis running
redis-cli ping
# Should return: PONG

# Install Redis package
pip install redis

# Check .env configuration
cat .env | grep REDIS
# Should show:
# USE_REDIS=true
# REDIS_URL=redis://localhost:6379

# Test connection
python -c "import redis; r=redis.from_url('redis://localhost:6379'); print(r.ping())"
```

---

### Issue 4: Hot-Reload Failed

**Error**: `Update dqn_1731600001 failed to apply`

**Causes**:
1. Orchestration service not running
2. Invalid hyperparameters
3. Agent not initialized

**Solution**:
```bash
# Verify orchestration service
curl http://localhost:8094/health

# Check agent initialized
curl http://localhost:8094/api/v1/orchestration/agents/state | jq .policy

# Validate hyperparameters manually
curl -X POST http://localhost:8098/api/v1/sync/propose \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "dqn",
    "current_params": {"learning_rate": 0.001},
    "current_score": 0.75
  }'
# Check for validation errors in response
```

---

### Issue 5: Dashboard Shows No Updates

**Error**: Dashboard doesn't display hyperparameter changes

**Solution**:
```bash
# Check WebSocket connection
# In browser console (F12):
# Should see: "Connected to T.A.R.S. Dashboard"

# Manually trigger update
curl http://localhost:3001/api/v1/agents

# Check Redis persistence
redis-cli
> KEYS agent_state:*
> HGETALL agent_state:policy
```

---

## Performance Tips

### 1. Speed Up Optimization

**Quick Mode**:
- 10 train episodes + 5 eval episodes
- ~3s per trial (vs 15s full mode)
- Good for testing, acceptable for production

**Reduce Trials**:
```bash
# Instead of 100 trials
{"n_trials": 100}

# Use 30-50 trials
{"n_trials": 30}
```

**Use Mock Objective for Testing**:
```bash
{"use_real_training": false}
# 0.1s per trial
```

### 2. Redis Optimization

**Increase TTL**:
Edit `dashboard/api/redis_backend.py`:
```python
ttl_seconds: int = 86400  # 24 hours
# Change to:
ttl_seconds: int = 604800  # 7 days
```

**Use Redis Cluster** (for production):
```bash
REDIS_URL=redis://redis-cluster:6379
```

### 3. Autonomous Approval

**Set Threshold**:
```bash
# Approve if improvement >= 5pp (conservative)
AUTONOMOUS_THRESHOLD=0.05

# Approve if improvement >= 1pp (aggressive)
AUTONOMOUS_THRESHOLD=0.01

# Approve all (use with caution)
APPROVAL_MODE=autonomous_all
```

---

## Example: Complete Deployment

**Using tmux for multi-service management**:

```bash
# Create tmux session
tmux new-session -d -s tars

# Window 0: Orchestration
tmux send-keys -t tars "cd cognition/orchestration-agent && python main.py" C-m

# Window 1: AutoML
tmux new-window -t tars
tmux send-keys -t tars "cd cognition/automl-pipeline && python service.py" C-m

# Window 2: Hyperparameter Sync
tmux new-window -t tars
tmux send-keys -t tars "cd cognition/hyperparameter-sync && python service.py" C-m

# Window 3: Redis
tmux new-window -t tars
tmux send-keys -t tars "redis-server" C-m

# Window 4: Dashboard API
tmux new-window -t tars
tmux send-keys -t tars "cd dashboard/api && python main.py" C-m

# Window 5: Dashboard Frontend
tmux new-window -t tars
tmux send-keys -t tars "cd dashboard/frontend && npm start" C-m

# Attach to session
tmux attach -t tars

# Detach: Ctrl+B, D
# Kill session: tmux kill-session -t tars
```

---

## Next Steps

1. **Explore Dashboard**: http://localhost:3000
2. **Run Full Optimization**: Use `use_quick_mode=false` for production
3. **Set Up Monitoring**: View Prometheus metrics at `/metrics` endpoints
4. **Configure Autonomous Mode**: Edit `.env` for auto-approval
5. **Deploy to Kubernetes**: Use Helm charts (Phase 11.5)

---

## Support

**Documentation**:
- [PHASE11_4_IMPLEMENTATION_REPORT.md](PHASE11_4_IMPLEMENTATION_REPORT.md): Full technical details
- [PHASE11_3_QUICKSTART.md](PHASE11_3_QUICKSTART.md): AutoML + Dashboard basics
- [PHASE11_2_QUICKSTART.md](PHASE11_2_QUICKSTART.md): Multi-agent orchestration

**Logs**:
- Orchestration: `./logs/orchestration.log`
- AutoML: `./logs/automl.log`
- Hyperparameter Sync: `./logs/hyperparam_sync.log`

---

**Status**: ✅ **Phase 11.4 Ready for Deployment**

🚀 **Enjoy zero-downtime hyperparameter optimization with T.A.R.S.!**

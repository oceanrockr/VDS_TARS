"""
Meta-Consensus Optimizer
Uses reinforcement learning to tune consensus parameters
"""
import os
import json
import logging
import pickle
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConsensusState:
    """Current state of consensus system"""
    avg_latency_ms: float
    p95_latency_ms: float
    success_rate: float
    quorum_failures: int
    total_votes: int
    algorithm: str  # raft/pbft
    current_timeout_ms: int
    timestamp: datetime


@dataclass
class ConsensusAction:
    """Action to take on consensus parameters"""
    action_type: str  # increase_timeout/decrease_timeout/adjust_quorum
    parameter: str
    delta: float  # Multiplicative factor (0.8 = reduce by 20%, 1.2 = increase by 20%)
    new_value: float


@dataclass
class ConsensusReward:
    """Reward signal for RL agent"""
    latency_component: float
    accuracy_component: float
    violation_penalty: float
    total_reward: float
    timestamp: datetime


class SimpleRLAgent:
    """Simple Q-learning based RL agent for consensus optimization"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.2
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Q-table: state_discretization -> action -> Q-value
        self.q_table: Dict[str, Dict[str, float]] = {}

        # Action space
        self.actions = [
            "decrease_timeout_10",
            "decrease_timeout_20",
            "no_change",
            "increase_timeout_10",
            "increase_timeout_20"
        ]

        # History for learning
        self.history: List[Tuple[str, str, float, str]] = []  # (state, action, reward, next_state)

    def discretize_state(self, state: ConsensusState) -> str:
        """Convert continuous state to discrete representation"""

        # Bin latency
        if state.avg_latency_ms < 200:
            latency_bin = "low"
        elif state.avg_latency_ms < 400:
            latency_bin = "medium"
        else:
            latency_bin = "high"

        # Bin success rate
        if state.success_rate >= 0.98:
            success_bin = "high"
        elif state.success_rate >= 0.95:
            success_bin = "medium"
        else:
            success_bin = "low"

        # Bin quorum failures
        failure_rate = state.quorum_failures / state.total_votes if state.total_votes > 0 else 0
        if failure_rate < 0.02:
            failure_bin = "low"
        elif failure_rate < 0.05:
            failure_bin = "medium"
        else:
            failure_bin = "high"

        return f"{latency_bin}_{success_bin}_{failure_bin}"

    def get_q_value(self, state_key: str, action: str) -> float:
        """Get Q-value for state-action pair"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}

        return self.q_table[state_key][action]

    def select_action(self, state: ConsensusState) -> str:
        """Select action using epsilon-greedy policy"""

        state_key = self.discretize_state(state)

        # Exploration
        if np.random.random() < self.exploration_rate:
            action = np.random.choice(self.actions)
            logger.info(f"Exploration: selected action {action}")
            return action

        # Exploitation - choose best action
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}

        q_values = self.q_table[state_key]
        best_action = max(q_values, key=q_values.get)

        logger.info(f"Exploitation: selected action {best_action} (Q={q_values[best_action]:.3f})")
        return best_action

    def update_q_value(
        self,
        state: ConsensusState,
        action: str,
        reward: float,
        next_state: ConsensusState
    ):
        """Update Q-value using Q-learning update rule"""

        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        # Get current Q-value
        current_q = self.get_q_value(state_key, action)

        # Get max Q-value for next state
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}

        max_next_q = max(self.q_table[next_state_key].values())

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action] = new_q

        logger.info(f"Updated Q({state_key}, {action}): {current_q:.3f} → {new_q:.3f} (reward={reward:.3f})")

        # Store in history
        self.history.append((state_key, action, reward, next_state_key))

    def action_to_consensus_action(
        self,
        action: str,
        current_state: ConsensusState
    ) -> ConsensusAction:
        """Convert RL action to consensus parameter change"""

        current_timeout = current_state.current_timeout_ms

        if action == "decrease_timeout_10":
            delta = 0.9
            new_value = current_timeout * delta
        elif action == "decrease_timeout_20":
            delta = 0.8
            new_value = current_timeout * delta
        elif action == "increase_timeout_10":
            delta = 1.1
            new_value = current_timeout * delta
        elif action == "increase_timeout_20":
            delta = 1.2
            new_value = current_timeout * delta
        else:  # no_change
            delta = 1.0
            new_value = current_timeout

        # Clamp to reasonable bounds
        new_value = max(100, min(2000, new_value))

        return ConsensusAction(
            action_type=action,
            parameter="timeout_ms",
            delta=delta,
            new_value=new_value
        )

    def save(self, filepath: str):
        """Save agent to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'history': self.history,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate
            }, f)

        logger.info(f"Saved RL agent to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SimpleRLAgent':
        """Load agent from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        agent = cls(
            learning_rate=data['learning_rate'],
            discount_factor=data['discount_factor'],
            exploration_rate=data['exploration_rate']
        )
        agent.q_table = data['q_table']
        agent.history = data['history']

        logger.info(f"Loaded RL agent from {filepath}")
        return agent


class MetaConsensusOptimizer:
    """Meta-level optimizer for consensus parameters"""

    def __init__(
        self,
        agent_path: str = "/app/agent.pkl",
        latency_target_ms: float = 300.0,
        success_rate_target: float = 0.97
    ):
        self.agent_path = agent_path
        self.latency_target = latency_target_ms
        self.success_rate_target = success_rate_target

        # Try to load existing agent
        if os.path.exists(agent_path):
            try:
                self.agent = SimpleRLAgent.load(agent_path)
                logger.info("Loaded existing RL agent")
            except Exception as e:
                logger.warning(f"Failed to load agent: {e}. Creating new agent.")
                self.agent = SimpleRLAgent()
        else:
            self.agent = SimpleRLAgent()
            logger.info("Created new RL agent")

        # Track last state for learning updates
        self.last_state: Optional[ConsensusState] = None
        self.last_action: Optional[str] = None

    def calculate_reward(self, state: ConsensusState) -> ConsensusReward:
        """Calculate reward signal from consensus state"""

        # Latency component: reward for being close to target
        latency_diff = abs(state.avg_latency_ms - self.latency_target)
        latency_component = max(0, 1.0 - (latency_diff / self.latency_target))

        # Accuracy component: reward for high success rate
        accuracy_component = state.success_rate

        # Violation penalty: penalize quorum failures
        failure_rate = state.quorum_failures / state.total_votes if state.total_votes > 0 else 0
        violation_penalty = failure_rate * 10.0  # Heavy penalty for failures

        # Combined reward
        total_reward = latency_component + accuracy_component - violation_penalty

        return ConsensusReward(
            latency_component=latency_component,
            accuracy_component=accuracy_component,
            violation_penalty=violation_penalty,
            total_reward=total_reward,
            timestamp=datetime.utcnow()
        )

    def optimize(self, current_state: ConsensusState) -> Optional[ConsensusAction]:
        """Main optimization step"""

        # Calculate reward for current state
        reward = self.calculate_reward(current_state)

        # If we have a previous state, update Q-values
        if self.last_state and self.last_action:
            self.agent.update_q_value(
                self.last_state,
                self.last_action,
                reward.total_reward,
                current_state
            )

        # Select action for current state
        action = self.agent.select_action(current_state)

        # Convert to consensus action
        consensus_action = self.agent.action_to_consensus_action(action, current_state)

        # Store for next iteration
        self.last_state = current_state
        self.last_action = action

        # Save agent periodically
        if len(self.agent.history) % 10 == 0:
            self.agent.save(self.agent_path)

        logger.info(
            f"Optimization: {action} → timeout {current_state.current_timeout_ms:.0f}ms "
            f"→ {consensus_action.new_value:.0f}ms (reward: {reward.total_reward:.3f})"
        )

        return consensus_action

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""

        if not self.agent.history:
            return {
                "total_updates": 0,
                "avg_reward": 0.0,
                "exploration_rate": self.agent.exploration_rate,
                "q_table_size": 0
            }

        recent_history = self.agent.history[-100:]  # Last 100 updates
        rewards = [h[2] for h in recent_history]

        return {
            "total_updates": len(self.agent.history),
            "recent_avg_reward": np.mean(rewards) if rewards else 0.0,
            "recent_max_reward": np.max(rewards) if rewards else 0.0,
            "recent_min_reward": np.min(rewards) if rewards else 0.0,
            "exploration_rate": self.agent.exploration_rate,
            "q_table_size": len(self.agent.q_table),
            "actions_distribution": self._get_action_distribution(recent_history)
        }

    def _get_action_distribution(self, history: List[Tuple]) -> Dict[str, int]:
        """Get distribution of actions taken"""
        distribution = {}
        for _, action, _, _ in history:
            distribution[action] = distribution.get(action, 0) + 1
        return distribution

    def save_agent(self):
        """Manually save agent"""
        self.agent.save(self.agent_path)

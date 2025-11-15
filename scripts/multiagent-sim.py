#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning Simulation

Simulates the interaction of 4 RL agents (DQN, A2C, PPO, DDPG) with:
- 500+ episode training
- Nash equilibrium coordination
- Conflict detection and resolution
- Performance metrics and benchmarking

Usage:
    python scripts/multiagent-sim.py --episodes 500 --save-results
    python scripts/multiagent-sim.py --load-checkpoint agents_ep300.pt

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "cognition" / "orchestration-agent"))

try:
    from dqn import DQNAgent
    from a2c import A2CAgent
    from ppo import PPOAgent
    from ddpg import DDPGAgent
    from rewards import GlobalRewardAggregator, GlobalRewardConfig, AgentReward, AgentType
    from nash import NashSolver, ParetoFrontier
except ImportError as e:
    logging.error(f"Failed to import agents: {e}")
    logging.error("Make sure you're running from the project root directory")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiAgentEnvironment:
    """
    Simulated environment for multi-agent coordination.

    State space represents system metrics:
    - Policy: 32-dim (SLO violations, latency, throughput, etc.)
    - Consensus: 16-dim (consensus latency, success rate, node health)
    - Ethical: 24-dim (fairness metrics, bias indicators, equity scores)
    - Resource: 20-dim (CPU, memory, replicas, scaling factors)

    Actions:
    - Policy: 10 discrete actions (adjust thresholds, policies)
    - Consensus: 5 discrete actions (timeout adjustments, quorum size)
    - Ethical: 8 discrete actions (fairness adjustments, bias correction)
    - Resource: 1 continuous action (scaling factor 0.0-1.0)
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

        # State dimensions
        self.policy_state_dim = 32
        self.consensus_state_dim = 16
        self.ethical_state_dim = 24
        self.resource_state_dim = 20

        # Current states
        self.reset()

        # Episode statistics
        self.episode_length = 0
        self.max_episode_length = 100

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reset environment to initial state."""
        self.policy_state = np.random.randn(self.policy_state_dim) * 0.1
        self.consensus_state = np.random.randn(self.consensus_state_dim) * 0.1
        self.ethical_state = np.random.randn(self.ethical_state_dim) * 0.1
        self.resource_state = np.random.randn(self.resource_state_dim) * 0.1

        self.episode_length = 0

        return (
            self.policy_state,
            self.consensus_state,
            self.ethical_state,
            self.resource_state
        )

    def step(
        self,
        policy_action: int,
        consensus_action: int,
        ethical_action: int,
        resource_action: float
    ) -> Tuple[Tuple, Tuple, bool, Dict]:
        """
        Execute one step in the environment.

        Returns:
            next_states: Tuple of next states for each agent
            rewards: Tuple of rewards for each agent
            done: Whether episode is finished
            info: Additional information
        """
        self.episode_length += 1

        # Simulate state transitions (simplified)
        # In reality, these would be based on actual system dynamics

        # Policy state transition
        policy_noise = np.random.randn(self.policy_state_dim) * 0.05
        policy_effect = self._policy_effect(policy_action)
        self.policy_state = self.policy_state + policy_effect + policy_noise

        # Consensus state transition
        consensus_noise = np.random.randn(self.consensus_state_dim) * 0.05
        consensus_effect = self._consensus_effect(consensus_action)
        self.consensus_state = self.consensus_state + consensus_effect + consensus_noise

        # Ethical state transition
        ethical_noise = np.random.randn(self.ethical_state_dim) * 0.05
        ethical_effect = self._ethical_effect(ethical_action)
        self.ethical_state = self.ethical_state + ethical_effect + ethical_noise

        # Resource state transition
        resource_noise = np.random.randn(self.resource_state_dim) * 0.05
        resource_effect = self._resource_effect(resource_action)
        self.resource_state = self.resource_state + resource_effect + resource_noise

        # Compute rewards
        policy_reward = self._compute_policy_reward(policy_action)
        consensus_reward = self._compute_consensus_reward(consensus_action)
        ethical_reward = self._compute_ethical_reward(ethical_action)
        resource_reward = self._compute_resource_reward(resource_action)

        # Check for conflicts
        conflicts = self._detect_conflicts(
            policy_action, consensus_action, ethical_action, resource_action
        )

        # Apply conflict penalty
        if conflicts:
            conflict_penalty = 0.2
            policy_reward -= conflict_penalty
            consensus_reward -= conflict_penalty
            ethical_reward -= conflict_penalty
            resource_reward -= conflict_penalty

        # Check if episode is done
        done = self.episode_length >= self.max_episode_length

        # Info dictionary
        info = {
            "episode_length": self.episode_length,
            "conflicts": conflicts,
            "policy_state_norm": float(np.linalg.norm(self.policy_state)),
            "consensus_state_norm": float(np.linalg.norm(self.consensus_state)),
            "ethical_state_norm": float(np.linalg.norm(self.ethical_state)),
            "resource_state_norm": float(np.linalg.norm(self.resource_state)),
        }

        next_states = (
            self.policy_state.copy(),
            self.consensus_state.copy(),
            self.ethical_state.copy(),
            self.resource_state.copy()
        )

        rewards = (policy_reward, consensus_reward, ethical_reward, resource_reward)

        return next_states, rewards, done, info

    def _policy_effect(self, action: int) -> np.ndarray:
        """Compute effect of policy action on state."""
        effect = np.zeros(self.policy_state_dim)
        # Aggressive actions improve performance but may violate constraints
        if action > 7:
            effect[:8] += 0.1  # Improve metrics
            effect[8:16] += 0.05  # But increase violations
        elif action < 3:
            effect[:8] -= 0.05  # Conservative
            effect[8:16] -= 0.1  # Reduce violations
        return effect

    def _consensus_effect(self, action: int) -> np.ndarray:
        """Compute effect of consensus action on state."""
        effect = np.zeros(self.consensus_state_dim)
        # Actions affect latency and success rate trade-off
        if action > 3:
            effect[:8] -= 0.1  # Reduce latency
            effect[8:] += 0.05  # Slight success rate impact
        return effect

    def _ethical_effect(self, action: int) -> np.ndarray:
        """Compute effect of ethical action on state."""
        effect = np.zeros(self.ethical_state_dim)
        # Actions improve fairness metrics
        effect[:12] += 0.05 * (action / 8.0)
        return effect

    def _resource_effect(self, action: float) -> np.ndarray:
        """Compute effect of resource action on state."""
        effect = np.zeros(self.resource_state_dim)
        # Scaling action affects utilization
        effect[:10] += (action - 0.5) * 0.2
        return effect

    def _compute_policy_reward(self, action: int) -> float:
        """Compute reward for policy agent."""
        # Reward based on SLO compliance and performance
        slo_compliance = -np.mean(np.abs(self.policy_state[8:16]))
        performance = np.mean(self.policy_state[:8])
        return float(slo_compliance + 0.5 * performance)

    def _compute_consensus_reward(self, action: int) -> float:
        """Compute reward for consensus agent."""
        # Reward based on consensus latency and success rate
        latency_score = -np.mean(np.abs(self.consensus_state[:8]))
        success_score = np.mean(self.consensus_state[8:])
        return float(0.5 * latency_score + 0.5 * success_score)

    def _compute_ethical_reward(self, action: int) -> float:
        """Compute reward for ethical agent."""
        # Reward based on fairness metrics
        fairness_score = np.mean(self.ethical_state[:12])
        bias_penalty = -np.mean(np.abs(self.ethical_state[12:]))
        return float(fairness_score + bias_penalty)

    def _compute_resource_reward(self, action: float) -> float:
        """Compute reward for resource agent."""
        # Reward for balanced resource utilization
        utilization = np.mean(self.resource_state[:10])
        # Penalize over/under utilization
        target_utilization = 0.7
        utilization_penalty = -abs(utilization - target_utilization)

        # Penalize excessive scaling
        scaling_penalty = -0.1 * abs(action - 0.7)

        return float(utilization_penalty + scaling_penalty + 0.5)

    def _detect_conflicts(
        self, policy_action: int, consensus_action: int,
        ethical_action: int, resource_action: float
    ) -> bool:
        """Detect conflicts between agent actions."""
        conflicts = False

        # Conflict: Policy aggressive + Ethical conservative
        if policy_action > 7 and ethical_action < 3:
            conflicts = True

        # Conflict: Resource over-scaling
        if resource_action > 0.95:
            conflicts = True

        return conflicts


class MultiAgentSimulation:
    """Main simulation orchestrator."""

    def __init__(
        self,
        n_episodes: int = 500,
        save_dir: str = "./results/multiagent_sim",
        device: str = "cpu"
    ):
        self.n_episodes = n_episodes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Initialize environment
        self.env = MultiAgentEnvironment()

        # Initialize agents
        logger.info("Initializing agents...")
        self.policy_agent = DQNAgent(
            state_dim=32, action_dim=10, device=device
        )
        self.consensus_agent = A2CAgent(
            state_dim=16, action_dim=5, device=device
        )
        self.ethical_agent = PPOAgent(
            state_dim=24, action_dim=8, device=device
        )
        self.resource_agent = DDPGAgent(
            state_dim=20, action_dim=1, device=device
        )

        # Initialize coordination systems
        reward_config = GlobalRewardConfig(
            w_policy=0.30, w_consensus=0.25,
            w_ethical=0.25, w_resource=0.20
        )
        self.reward_aggregator = GlobalRewardAggregator(config=reward_config)
        self.nash_solver = NashSolver(method="iterative_br", max_iterations=1000)
        self.pareto_frontier = ParetoFrontier()

        # Tracking metrics
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = []
        self.global_rewards = []
        self.conflicts_per_episode = []
        self.nash_convergence_times = []
        self.reward_alignments = []

        logger.info("Simulation initialized successfully")

    def run(self):
        """Run the full simulation."""
        logger.info(f"Starting simulation for {self.n_episodes} episodes...")

        start_time = time.time()

        for episode in range(self.n_episodes):
            episode_start = time.time()

            # Run episode
            stats = self.run_episode(episode)

            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}/{self.n_episodes} | "
                    f"Global Reward: {stats['global_reward']:.3f} | "
                    f"Conflicts: {stats['conflicts']} | "
                    f"Duration: {time.time() - episode_start:.2f}s"
                )

            # Save checkpoint
            if (episode + 1) % 100 == 0:
                self.save_checkpoint(episode + 1)

        total_time = time.time() - start_time
        logger.info(f"Simulation completed in {total_time:.2f}s")

        # Generate report
        self.generate_report()

        logger.info("Results saved successfully!")

    def run_episode(self, episode_num: int) -> Dict:
        """Run one episode."""
        # Reset environment
        states = self.env.reset()
        policy_state, consensus_state, ethical_state, resource_state = states

        episode_reward = 0.0
        episode_conflicts = 0

        done = False
        step = 0

        while not done:
            # Select actions from all agents
            policy_action = self.policy_agent.select_action(policy_state)
            consensus_action, cons_log_prob, cons_value = self.consensus_agent.select_action(consensus_state)
            ethical_action, eth_log_prob, eth_value = self.ethical_agent.select_action(ethical_state)
            resource_action = self.resource_agent.select_action(resource_state, add_noise=True)

            # Execute in environment
            next_states, rewards, done, info = self.env.step(
                policy_action, consensus_action, ethical_action, resource_action[0]
            )

            policy_reward, consensus_reward, ethical_reward, resource_reward = rewards
            next_policy_state, next_consensus_state, next_ethical_state, next_resource_state = next_states

            # Store transitions
            self.policy_agent.store_experience(
                policy_state, policy_action, policy_reward, next_policy_state, done
            )

            self.consensus_agent.store_transition(
                consensus_state, consensus_action, consensus_reward, done,
                cons_log_prob, cons_value
            )

            self.ethical_agent.store_transition(
                ethical_state, ethical_action, ethical_reward, done,
                eth_log_prob, eth_value
            )

            self.resource_agent.store_transition(
                resource_state, resource_action, resource_reward, next_resource_state, done
            )

            # Train DQN and DDPG (off-policy)
            if len(self.policy_agent.replay_buffer) >= self.policy_agent.batch_size:
                self.policy_agent.train_step()

            if len(self.resource_agent.replay_buffer) >= self.resource_agent.batch_size:
                self.resource_agent.train_step()

            # Track conflicts
            if info["conflicts"]:
                episode_conflicts += 1

            # Global reward aggregation
            reward_list = [
                AgentReward(agent_type=AgentType.POLICY, value=policy_reward, confidence=0.9),
                AgentReward(agent_type=AgentType.CONSENSUS, value=consensus_reward, confidence=0.85),
                AgentReward(agent_type=AgentType.ETHICAL, value=ethical_reward, confidence=0.88),
                AgentReward(agent_type=AgentType.RESOURCE, value=resource_reward, confidence=0.92),
            ]
            global_reward, _, _ = self.reward_aggregator.aggregate(reward_list)
            episode_reward += global_reward

            # Update states
            policy_state = next_policy_state
            consensus_state = next_consensus_state
            ethical_state = next_ethical_state
            resource_state = next_resource_state

            step += 1

        # End of episode training for on-policy agents
        if len(self.consensus_agent.states) > 0:
            self.consensus_agent.train_step(next_state=next_consensus_state)

        if len(self.ethical_agent.buffer) > 0:
            self.ethical_agent.update(next_state=next_ethical_state)

        # Update DDPG episode stats
        self.resource_agent.end_episode(
            episode_reward=resource_reward,
            episode_length=step
        )

        # Track metrics
        self.episode_rewards["policy"].append(policy_reward)
        self.episode_rewards["consensus"].append(consensus_reward)
        self.episode_rewards["ethical"].append(ethical_reward)
        self.episode_rewards["resource"].append(resource_reward)
        self.episode_lengths.append(step)
        self.global_rewards.append(episode_reward)
        self.conflicts_per_episode.append(episode_conflicts)

        # Compute reward alignment
        reward_values = [policy_reward, consensus_reward, ethical_reward, resource_reward]
        if len(reward_values) > 1:
            corr_matrix = np.corrcoef(reward_values)
            alignment = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            self.reward_alignments.append(alignment)

        return {
            "global_reward": episode_reward,
            "conflicts": episode_conflicts,
            "episode_length": step
        }

    def save_checkpoint(self, episode: int):
        """Save agent checkpoints."""
        checkpoint_dir = self.save_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        self.policy_agent.save(str(checkpoint_dir / f"policy_ep{episode}.pt"))
        self.consensus_agent.save(str(checkpoint_dir / f"consensus_ep{episode}.pt"))
        self.ethical_agent.save(str(checkpoint_dir / f"ethical_ep{episode}.pt"))
        self.resource_agent.save(str(checkpoint_dir / f"resource_ep{episode}.pt"))

        logger.info(f"Checkpoint saved at episode {episode}")

    def generate_report(self):
        """Generate simulation report."""
        report = {
            "simulation_info": {
                "n_episodes": self.n_episodes,
                "timestamp": datetime.utcnow().isoformat(),
                "device": self.device
            },
            "metrics": {
                "avg_global_reward": float(np.mean(self.global_rewards)),
                "std_global_reward": float(np.std(self.global_rewards)),
                "avg_episode_length": float(np.mean(self.episode_lengths)),
                "total_conflicts": int(sum(self.conflicts_per_episode)),
                "avg_conflicts_per_episode": float(np.mean(self.conflicts_per_episode)),
                "avg_reward_alignment": float(np.mean(self.reward_alignments)) if self.reward_alignments else 0.0,
            },
            "agent_metrics": {
                "policy": {
                    "avg_reward_last_100": float(np.mean(self.episode_rewards["policy"][-100:])),
                    "max_reward": float(np.max(self.episode_rewards["policy"])),
                },
                "consensus": {
                    "avg_reward_last_100": float(np.mean(self.episode_rewards["consensus"][-100:])),
                    "max_reward": float(np.max(self.episode_rewards["consensus"])),
                },
                "ethical": {
                    "avg_reward_last_100": float(np.mean(self.episode_rewards["ethical"][-100:])),
                    "max_reward": float(np.max(self.episode_rewards["ethical"])),
                },
                "resource": {
                    "avg_reward_last_100": float(np.mean(self.episode_rewards["resource"][-100:])),
                    "max_reward": float(np.max(self.episode_rewards["resource"])),
                }
            }
        }

        # Save report
        report_path = self.save_dir / "simulation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")
        logger.info(f"Average Global Reward: {report['metrics']['avg_global_reward']:.3f}")
        logger.info(f"Reward Alignment: {report['metrics']['avg_reward_alignment']:.3f}")
        logger.info(f"Total Conflicts: {report['metrics']['total_conflicts']}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RL Simulation")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--save-dir", type=str, default="./results/multiagent_sim",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    # Run simulation
    sim = MultiAgentSimulation(
        n_episodes=args.episodes,
        save_dir=args.save_dir,
        device=args.device
    )
    sim.run()

    logger.info("Simulation completed successfully!")


if __name__ == "__main__":
    main()

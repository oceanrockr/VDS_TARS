"""
Global Reward Aggregation for Multi-Agent Coordination

Combines individual agent rewards into a global objective with
conflict penalties and cooperation bonuses.

Author: T.A.R.S. Cognitive Team
Version: v0.9.1-alpha
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the orchestration system."""
    POLICY = "policy"
    CONSENSUS = "consensus"
    ETHICAL = "ethical"
    RESOURCE = "resource"


@dataclass
class AgentReward:
    """
    Individual agent reward with metadata.

    Attributes:
        agent_type: Type of agent
        reward: Reward value
        confidence: Confidence score (0-1)
        metadata: Additional context
    """
    agent_type: AgentType
    reward: float
    confidence: float = 1.0
    metadata: Optional[Dict] = None


@dataclass
class GlobalRewardConfig:
    """
    Configuration for global reward aggregation.

    Attributes:
        w_policy: Weight for policy agent reward
        w_consensus: Weight for consensus agent reward
        w_ethical: Weight for ethical agent reward
        w_resource: Weight for resource agent reward
        conflict_penalty_base: Base penalty for agent conflicts
        cooperation_bonus: Bonus multiplier for cooperative behavior
        normalization_enabled: Enable reward normalization
    """
    w_policy: float = 0.30
    w_consensus: float = 0.25
    w_ethical: float = 0.25
    w_resource: float = 0.20
    conflict_penalty_base: float = 0.1
    cooperation_bonus: float = 1.2
    normalization_enabled: bool = True

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total_weight = self.w_policy + self.w_consensus + self.w_ethical + self.w_resource
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")


class GlobalRewardAggregator:
    """
    Aggregates individual agent rewards into a global objective.

    Implements cooperative reward shaping with conflict detection
    and cooperation bonuses.
    """

    def __init__(self, config: Optional[GlobalRewardConfig] = None):
        """
        Initialize reward aggregator.

        Args:
            config: Reward aggregation configuration
        """
        self.config = config or GlobalRewardConfig()

        # Reward history for normalization
        self.reward_history: Dict[AgentType, List[float]] = {
            agent_type: [] for agent_type in AgentType
        }

        # Statistics
        self.aggregation_count = 0
        self.conflict_count = 0
        self.cooperation_count = 0

    def aggregate(
        self,
        agent_rewards: List[AgentReward],
        causal_impact: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Aggregate individual agent rewards into global reward.

        Args:
            agent_rewards: List of agent rewards
            causal_impact: Optional causal inference impact estimate

        Returns:
            Tuple of (global_reward, breakdown_dict)
        """
        # Normalize rewards if enabled
        if self.config.normalization_enabled:
            normalized_rewards = self._normalize_rewards(agent_rewards)
        else:
            normalized_rewards = agent_rewards

        # Weighted sum of individual rewards
        weighted_sum = 0.0
        breakdown = {}

        for agent_reward in normalized_rewards:
            weight = self._get_weight(agent_reward.agent_type)
            contribution = weight * agent_reward.reward * agent_reward.confidence
            weighted_sum += contribution

            breakdown[f"{agent_reward.agent_type.value}_reward"] = agent_reward.reward
            breakdown[f"{agent_reward.agent_type.value}_contribution"] = contribution

        # Detect conflicts and cooperation
        conflict_penalty = self._compute_conflict_penalty(agent_rewards)
        cooperation_bonus = self._compute_cooperation_bonus(agent_rewards)

        # Apply causal impact modifier if available
        causal_modifier = 1.0
        if causal_impact is not None:
            causal_modifier = 1.0 + causal_impact  # Impact is in range [-0.5, 0.5]
            breakdown["causal_modifier"] = causal_modifier

        # Global reward formula
        global_reward = (
            weighted_sum * cooperation_bonus - conflict_penalty
        ) * causal_modifier

        # Update statistics
        self.aggregation_count += 1
        breakdown["global_reward"] = global_reward
        breakdown["weighted_sum"] = weighted_sum
        breakdown["conflict_penalty"] = conflict_penalty
        breakdown["cooperation_bonus"] = cooperation_bonus

        logger.debug(
            f"Global reward: {global_reward:.3f} "
            f"(weighted={weighted_sum:.3f}, conflict={conflict_penalty:.3f}, "
            f"coop={cooperation_bonus:.3f}, causal={causal_modifier:.3f})"
        )

        return global_reward, breakdown

    def _get_weight(self, agent_type: AgentType) -> float:
        """Get weight for agent type."""
        weight_map = {
            AgentType.POLICY: self.config.w_policy,
            AgentType.CONSENSUS: self.config.w_consensus,
            AgentType.ETHICAL: self.config.w_ethical,
            AgentType.RESOURCE: self.config.w_resource,
        }
        return weight_map[agent_type]

    def _normalize_rewards(
        self,
        agent_rewards: List[AgentReward]
    ) -> List[AgentReward]:
        """
        Normalize agent rewards using running statistics.

        Args:
            agent_rewards: Raw agent rewards

        Returns:
            Normalized agent rewards
        """
        normalized = []

        for agent_reward in agent_rewards:
            # Update history
            self.reward_history[agent_reward.agent_type].append(agent_reward.reward)

            # Keep last 100 rewards
            if len(self.reward_history[agent_reward.agent_type]) > 100:
                self.reward_history[agent_reward.agent_type].pop(0)

            # Compute statistics
            history = self.reward_history[agent_reward.agent_type]
            if len(history) < 2:
                # Not enough data for normalization
                normalized.append(agent_reward)
            else:
                mean = np.mean(history)
                std = np.std(history)

                # Z-score normalization
                if std > 1e-6:
                    normalized_reward = (agent_reward.reward - mean) / std
                else:
                    normalized_reward = agent_reward.reward

                # Create normalized copy
                normalized.append(
                    AgentReward(
                        agent_type=agent_reward.agent_type,
                        reward=normalized_reward,
                        confidence=agent_reward.confidence,
                        metadata=agent_reward.metadata
                    )
                )

        return normalized

    def _compute_conflict_penalty(self, agent_rewards: List[AgentReward]) -> float:
        """
        Compute penalty for conflicting agent objectives.

        Conflicts occur when agents have opposing reward signs or
        when high-confidence agents disagree.

        Args:
            agent_rewards: List of agent rewards

        Returns:
            Conflict penalty (≥ 0)
        """
        if len(agent_rewards) < 2:
            return 0.0

        penalty = 0.0
        conflict_detected = False

        # Check for opposing reward signs
        reward_signs = [np.sign(r.reward) for r in agent_rewards]
        if len(set(reward_signs)) > 1:
            # Agents disagree on improvement direction
            conflict_detected = True

            # Penalty proportional to disagreement magnitude
            rewards_array = np.array([r.reward for r in agent_rewards])
            penalty += self.config.conflict_penalty_base * np.std(rewards_array)

        # Check for high-variance in high-confidence rewards
        high_conf_rewards = [
            r.reward for r in agent_rewards if r.confidence > 0.7
        ]
        if len(high_conf_rewards) >= 2:
            variance = np.var(high_conf_rewards)
            if variance > 0.5:  # Threshold for significant disagreement
                conflict_detected = True
                penalty += self.config.conflict_penalty_base * variance

        if conflict_detected:
            self.conflict_count += 1

        return penalty

    def _compute_cooperation_bonus(self, agent_rewards: List[AgentReward]) -> float:
        """
        Compute bonus for cooperative agent behavior.

        Cooperation detected when all agents agree on improvement
        direction and have consistent rewards.

        Args:
            agent_rewards: List of agent rewards

        Returns:
            Cooperation bonus multiplier (≥ 1.0)
        """
        if len(agent_rewards) < 2:
            return 1.0

        # Check if all rewards are positive (all agree on improvement)
        all_positive = all(r.reward > 0 for r in agent_rewards)

        # Check if rewards are consistent (low variance)
        rewards_array = np.array([r.reward for r in agent_rewards])
        normalized_std = np.std(rewards_array) / (np.mean(np.abs(rewards_array)) + 1e-6)

        if all_positive and normalized_std < 0.3:
            # Strong cooperation detected
            self.cooperation_count += 1
            return self.config.cooperation_bonus
        elif all_positive:
            # Weak cooperation (aligned direction, but varying magnitude)
            return 1.1
        else:
            return 1.0

    def get_statistics(self) -> Dict[str, float]:
        """
        Get aggregator statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "aggregation_count": self.aggregation_count,
            "conflict_count": self.conflict_count,
            "cooperation_count": self.cooperation_count,
        }

        if self.aggregation_count > 0:
            stats["conflict_rate"] = self.conflict_count / self.aggregation_count
            stats["cooperation_rate"] = self.cooperation_count / self.aggregation_count

        # Add per-agent reward history stats
        for agent_type in AgentType:
            history = self.reward_history[agent_type]
            if history:
                stats[f"{agent_type.value}_reward_mean"] = np.mean(history)
                stats[f"{agent_type.value}_reward_std"] = np.std(history)

        return stats

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.aggregation_count = 0
        self.conflict_count = 0
        self.cooperation_count = 0
        for agent_type in AgentType:
            self.reward_history[agent_type] = []

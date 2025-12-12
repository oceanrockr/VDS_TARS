"""
Metrics Calculator - Compute evaluation metrics.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from models import EpisodeResult, MetricsResult


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""

    def compute_reward_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Compute reward statistics.

        Returns:
            {
                "mean_reward": float,
                "std_reward": float,
                "min_reward": float,
                "max_reward": float,
                "success_rate": float,
                "mean_steps": float
            }
        """
        rewards = [ep.total_reward for ep in episodes]
        steps = [ep.steps for ep in episodes]
        successes = [ep.success for ep in episodes]

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "success_rate": float(sum(successes) / len(episodes)),
            "mean_steps": float(np.mean(steps))
        }

    def compute_loss_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, Any]:
        """
        Compute loss metrics and trend.

        Returns:
            {
                "mean_loss": float | None,
                "loss_trend": "increasing" | "decreasing" | "stable" | None
            }
        """
        losses = [ep.loss for ep in episodes if ep.loss is not None]

        if not losses:
            return {"mean_loss": None, "loss_trend": None}

        trend = self.detect_loss_trend(losses)

        return {
            "mean_loss": float(np.mean(losses)),
            "loss_trend": trend
        }

    def detect_loss_trend(
        self,
        losses: List[float],
        window: int = 10
    ) -> str:
        """
        Detect if loss is increasing/decreasing/stable.

        Uses linear regression on last N episodes.

        Returns:
            "increasing" | "decreasing" | "stable"
        """
        if len(losses) < window:
            return "stable"

        # Take recent window
        recent = losses[-window:]
        x = np.arange(len(recent))

        # Fit linear regression
        slope, _ = np.polyfit(x, recent, 1)

        # Classify trend based on slope
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def compute_stability_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Compute reward variance and coefficient of variation.

        Returns:
            {
                "reward_variance": float,
                "coefficient_of_variation": float
            }
        """
        rewards = [ep.total_reward for ep in episodes]
        mean_reward = np.mean(rewards)
        variance = np.var(rewards)
        cv = np.std(rewards) / mean_reward if mean_reward != 0 else 0.0

        return {
            "reward_variance": float(variance),
            "coefficient_of_variation": float(cv)
        }

    def compute_action_entropy(
        self,
        action_distribution: np.ndarray
    ) -> float:
        """
        Compute Shannon entropy of action distribution.

        Args:
            action_distribution: Array of action counts

        Returns:
            Entropy value (higher = more exploration)
        """
        # Normalize to probabilities
        probs = action_distribution / action_distribution.sum()

        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]

        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log(probs))

        return float(entropy)

    def compute_variance_metrics(
        self,
        episodes: List[EpisodeResult]
    ) -> Dict[str, float]:
        """
        Compute reward variance metrics.

        Returns:
            {
                "reward_variance": float,
                "std_dev": float
            }
        """
        rewards = [ep.total_reward for ep in episodes]

        return {
            "reward_variance": float(np.var(rewards)),
            "std_dev": float(np.std(rewards))
        }

    def compute_all_metrics(
        self,
        episodes: List[EpisodeResult],
        action_distribution: Optional[np.ndarray] = None
    ) -> MetricsResult:
        """
        Compute all metrics and return MetricsResult.

        Args:
            episodes: List of episode results
            action_distribution: Optional action distribution for entropy

        Returns:
            MetricsResult with all computed metrics
        """
        # Compute reward metrics
        reward_metrics = self.compute_reward_metrics(episodes)

        # Compute loss metrics
        loss_metrics = self.compute_loss_metrics(episodes)

        # Compute stability metrics
        stability = self.compute_stability_metrics(episodes)

        # Compute action entropy
        if action_distribution is not None:
            entropy = self.compute_action_entropy(action_distribution)
        else:
            # Default entropy if no action distribution provided
            entropy = 0.0

        # Combine into MetricsResult
        return MetricsResult(
            mean_reward=reward_metrics["mean_reward"],
            std_reward=reward_metrics["std_reward"],
            min_reward=reward_metrics["min_reward"],
            max_reward=reward_metrics["max_reward"],
            success_rate=reward_metrics["success_rate"],
            mean_steps=reward_metrics["mean_steps"],
            mean_loss=loss_metrics["mean_loss"],
            loss_trend=loss_metrics["loss_trend"],
            action_entropy=entropy,
            reward_variance=stability["reward_variance"]
        )

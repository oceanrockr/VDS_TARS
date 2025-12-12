"""
Regression Detector - Detect performance degradation.
"""
from typing import List, Optional
from config import RegressionThresholds
from models import MetricsResult, RegressionResult, BaselineRecord


class RegressionDetector:
    """Detect performance regressions vs baseline."""

    def __init__(self, thresholds: RegressionThresholds):
        self.thresholds = thresholds

    def should_trigger_rollback(
        self,
        current_metrics: MetricsResult,
        baseline: BaselineRecord
    ) -> RegressionResult:
        """
        Check if current performance represents a regression.

        Regression Criteria:
        1. Failure rate > threshold (default 15%)
        2. Reward drop > threshold (default 10%)
        3. Loss trend is "increasing"
        4. Variance increased by >2.5x

        Returns:
            RegressionResult with detected flag and reason
        """
        failed_checks = []

        # Check 1: Failure rate
        failure_rate = 1 - current_metrics.success_rate
        if failure_rate > self.thresholds.failure_rate:
            failed_checks.append(
                f"Failure rate {failure_rate:.1%} exceeds threshold {self.thresholds.failure_rate:.1%}"
            )

        # Check 2: Reward drop
        if baseline.mean_reward > 0:
            reward_drop = (baseline.mean_reward - current_metrics.mean_reward) / baseline.mean_reward
            if reward_drop > self.thresholds.reward_drop_pct:
                failed_checks.append(
                    f"Reward dropped {reward_drop:.1%} (threshold: {self.thresholds.reward_drop_pct:.1%})"
                )

        # Check 3: Loss trend
        if current_metrics.loss_trend == "increasing":
            failed_checks.append("Loss trend is increasing")

        # Check 4: Variance increase
        baseline_variance = baseline.std_reward ** 2
        if baseline_variance > 0:
            variance_ratio = current_metrics.reward_variance / baseline_variance
            if variance_ratio > self.thresholds.variance_multiplier:
                failed_checks.append(
                    f"Variance increased {variance_ratio:.1f}x (threshold: {self.thresholds.variance_multiplier:.1f}x)"
                )

        detected = len(failed_checks) > 0
        score = self.compute_regression_score(current_metrics, baseline)
        reason = self.generate_rollback_reason(failed_checks)

        return RegressionResult(
            detected=detected,
            regression_score=score,
            reason=reason,
            should_rollback=detected,
            failed_checks=failed_checks
        )

    def compute_regression_score(
        self,
        current: MetricsResult,
        baseline: BaselineRecord
    ) -> float:
        """
        Compute regression score 0.0-1.0 (1.0 = severe regression).

        Formula:
            score = weighted_average([
                reward_drop_ratio,
                failure_rate_ratio,
                variance_ratio,
                loss_trend_penalty
            ])
        """
        # Calculate normalized scores for each factor

        # 1. Reward drop score
        if baseline.mean_reward > 0:
            reward_drop = (baseline.mean_reward - current.mean_reward) / baseline.mean_reward
            reward_drop_score = min(1.0, max(0.0, reward_drop / self.thresholds.reward_drop_pct))
        else:
            reward_drop_score = 0.0

        # 2. Failure rate score
        failure_rate = 1 - current.success_rate
        failure_score = min(1.0, max(0.0, failure_rate / self.thresholds.failure_rate))

        # 3. Variance score
        baseline_variance = baseline.std_reward ** 2
        if baseline_variance > 0:
            variance_ratio = current.reward_variance / baseline_variance
            variance_score = min(1.0, max(0.0, variance_ratio / self.thresholds.variance_multiplier))
        else:
            variance_score = 0.0

        # 4. Loss trend penalty
        if current.loss_trend == "increasing":
            loss_score = 1.0
        elif current.loss_trend == "stable":
            loss_score = 0.3
        else:
            loss_score = 0.0

        # Weighted average: 0.4 * reward + 0.3 * failure + 0.2 * variance + 0.1 * loss
        weighted_score = (
            0.4 * reward_drop_score +
            0.3 * failure_score +
            0.2 * variance_score +
            0.1 * loss_score
        )

        return min(1.0, max(0.0, weighted_score))

    def generate_rollback_reason(
        self,
        failed_checks: List[str]
    ) -> Optional[str]:
        """Generate human-readable rollback reason."""
        if not failed_checks:
            return None

        return f"Performance regression detected: {'; '.join(failed_checks)}"

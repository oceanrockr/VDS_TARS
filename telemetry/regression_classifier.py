"""
Regression Classifier for T.A.R.S.

ML-based regression detection using gradient boosting for production telemetry analysis.

Features:
- Gradient boosting classifier (XGBoost/LightGBM)
- Feature extraction from logs and metrics
- Offline batch predictions
- Streaming mode for real-time detection
- Confidence scoring
- Multi-class classification (benign, performance, environment)

Classification Categories:
- BENIGN: Normal behavior, no regression
- PERFORMANCE_REGRESSION: Code/config causing performance degradation
- ENVIRONMENT_REGRESSION: Infrastructure/dependency issues
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from telemetry.production_log_ingestor import LogEntry


logger = logging.getLogger(__name__)


class RegressionType(str, Enum):
    """Regression classification types."""
    BENIGN = "benign"
    PERFORMANCE_REGRESSION = "performance_regression"
    ENVIRONMENT_REGRESSION = "environment_regression"


@dataclass
class RegressionFeatures:
    """Feature vector for regression classification."""

    # Performance deltas (vs baseline)
    reward_delta: float = 0.0
    latency_delta: float = 0.0
    error_rate_delta: float = 0.0
    throughput_delta: float = 0.0

    # Queue metrics
    queue_depth_mean: float = 0.0
    queue_depth_max: float = 0.0
    queue_wait_time_p95: float = 0.0

    # Resource metrics
    cpu_utilization_mean: float = 0.0
    memory_utilization_mean: float = 0.0
    disk_io_rate: float = 0.0

    # Environment drift
    deployment_age_hours: float = 0.0
    config_change_count: int = 0
    dependency_change_count: int = 0

    # Hyperparameter drift
    hyperparameter_drift_score: float = 0.0
    policy_change_count: int = 0

    # Error patterns
    error_spike_detected: bool = False
    timeout_rate: float = 0.0
    retry_rate: float = 0.0

    # External dependencies
    external_api_latency_delta: float = 0.0
    database_latency_delta: float = 0.0
    cache_hit_rate_delta: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input.

        Returns:
            Feature array
        """
        return np.array([
            self.reward_delta,
            self.latency_delta,
            self.error_rate_delta,
            self.throughput_delta,
            self.queue_depth_mean,
            self.queue_depth_max,
            self.queue_wait_time_p95,
            self.cpu_utilization_mean,
            self.memory_utilization_mean,
            self.disk_io_rate,
            self.deployment_age_hours,
            self.config_change_count,
            self.dependency_change_count,
            self.hyperparameter_drift_score,
            self.policy_change_count,
            1.0 if self.error_spike_detected else 0.0,
            self.timeout_rate,
            self.retry_rate,
            self.external_api_latency_delta,
            self.database_latency_delta,
            self.cache_hit_rate_delta,
        ], dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        """Get feature names.

        Returns:
            List of feature names
        """
        return [
            'reward_delta',
            'latency_delta',
            'error_rate_delta',
            'throughput_delta',
            'queue_depth_mean',
            'queue_depth_max',
            'queue_wait_time_p95',
            'cpu_utilization_mean',
            'memory_utilization_mean',
            'disk_io_rate',
            'deployment_age_hours',
            'config_change_count',
            'dependency_change_count',
            'hyperparameter_drift_score',
            'policy_change_count',
            'error_spike_detected',
            'timeout_rate',
            'retry_rate',
            'external_api_latency_delta',
            'database_latency_delta',
            'cache_hit_rate_delta',
        ]


@dataclass
class RegressionPrediction:
    """Regression prediction result."""

    timestamp: datetime
    regression_type: RegressionType
    confidence: float  # 0-1
    features: RegressionFeatures

    # Supporting evidence
    contributing_factors: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommended_investigation: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'regression_type': self.regression_type.value,
            'confidence': round(self.confidence, 3),
            'features': asdict(self.features),
            'contributing_factors': self.contributing_factors,
            'evidence': self.evidence,
            'recommended_investigation': self.recommended_investigation,
        }

    def to_markdown(self) -> str:
        """Convert to markdown report.

        Returns:
            Markdown string
        """
        emoji = {
            RegressionType.BENIGN: '✅',
            RegressionType.PERFORMANCE_REGRESSION: '⚠️',
            RegressionType.ENVIRONMENT_REGRESSION: '🔧',
        }

        md = f"""
## {emoji.get(self.regression_type, '❓')} Regression Analysis

**Classification:** {self.regression_type.value.replace('_', ' ').title()}
**Confidence:** {self.confidence * 100:.1f}%
**Timestamp:** {self.timestamp.isoformat()}

### Top Contributing Factors
"""
        for i, factor in enumerate(self.contributing_factors[:5], 1):
            md += f"{i}. {factor}\n"

        md += "\n### Feature Analysis\n\n"
        md += "| Feature | Value |\n"
        md += "|---------|-------|\n"

        # Show top features
        feature_dict = asdict(self.features)
        for name, value in sorted(feature_dict.items(), key=lambda x: abs(float(x[1]) if isinstance(x[1], (int, float)) else 0), reverse=True)[:10]:
            if isinstance(value, bool):
                value_str = '✓' if value else '✗'
            elif isinstance(value, float):
                value_str = f"{value:.3f}"
            else:
                value_str = str(value)
            md += f"| {name} | {value_str} |\n"

        if self.recommended_investigation:
            md += "\n### Recommended Investigation Steps\n\n"
            for i, step in enumerate(self.recommended_investigation, 1):
                md += f"{i}. {step}\n"

        return md.strip()


class FeatureExtractor:
    """Extracts regression features from logs and metrics."""

    def __init__(self, baseline_window_hours: int = 24):
        """Initialize feature extractor.

        Args:
            baseline_window_hours: Hours of baseline data for delta calculations
        """
        self.baseline_window_hours = baseline_window_hours
        self._baseline_metrics: Dict[str, List[float]] = {}
        self._current_metrics: Dict[str, List[float]] = {}

    def extract_from_logs(
        self,
        recent_logs: List[LogEntry],
        baseline_logs: Optional[List[LogEntry]] = None,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> RegressionFeatures:
        """Extract features from log entries.

        Args:
            recent_logs: Recent log entries (current period)
            baseline_logs: Baseline log entries for comparison
            additional_metrics: Additional metrics (queue depth, resource usage, etc.)

        Returns:
            Extracted features
        """
        features = RegressionFeatures()

        # Extract from recent logs
        recent_latencies = [log.duration_ms for log in recent_logs if log.duration_ms is not None]
        recent_errors = sum(1 for log in recent_logs if log.error is not None)

        # Calculate current metrics
        if recent_latencies:
            current_latency_mean = np.mean(recent_latencies)
        else:
            current_latency_mean = 0.0

        current_error_rate = recent_errors / max(len(recent_logs), 1)

        # Calculate deltas if baseline provided
        if baseline_logs:
            baseline_latencies = [log.duration_ms for log in baseline_logs if log.duration_ms is not None]
            baseline_errors = sum(1 for log in baseline_logs if log.error is not None)

            if baseline_latencies:
                baseline_latency_mean = np.mean(baseline_latencies)
                features.latency_delta = (current_latency_mean - baseline_latency_mean) / max(baseline_latency_mean, 1.0)

            baseline_error_rate = baseline_errors / max(len(baseline_logs), 1)
            features.error_rate_delta = current_error_rate - baseline_error_rate

        # Extract additional metrics
        if additional_metrics:
            features.queue_depth_mean = additional_metrics.get('queue_depth_mean', 0.0)
            features.queue_depth_max = additional_metrics.get('queue_depth_max', 0.0)
            features.queue_wait_time_p95 = additional_metrics.get('queue_wait_time_p95', 0.0)

            features.cpu_utilization_mean = additional_metrics.get('cpu_utilization_mean', 0.0)
            features.memory_utilization_mean = additional_metrics.get('memory_utilization_mean', 0.0)
            features.disk_io_rate = additional_metrics.get('disk_io_rate', 0.0)

            features.deployment_age_hours = additional_metrics.get('deployment_age_hours', 0.0)
            features.config_change_count = additional_metrics.get('config_change_count', 0)
            features.dependency_change_count = additional_metrics.get('dependency_change_count', 0)

            features.hyperparameter_drift_score = additional_metrics.get('hyperparameter_drift_score', 0.0)
            features.policy_change_count = additional_metrics.get('policy_change_count', 0)

            features.error_spike_detected = additional_metrics.get('error_spike_detected', False)
            features.timeout_rate = additional_metrics.get('timeout_rate', 0.0)
            features.retry_rate = additional_metrics.get('retry_rate', 0.0)

            features.external_api_latency_delta = additional_metrics.get('external_api_latency_delta', 0.0)
            features.database_latency_delta = additional_metrics.get('database_latency_delta', 0.0)
            features.cache_hit_rate_delta = additional_metrics.get('cache_hit_rate_delta', 0.0)

            features.reward_delta = additional_metrics.get('reward_delta', 0.0)
            features.throughput_delta = additional_metrics.get('throughput_delta', 0.0)

        return features


class RegressionClassifier:
    """ML-based regression classifier using LightGBM."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.7,
    ):
        """Initialize classifier.

        Args:
            model_path: Path to pre-trained model
            confidence_threshold: Minimum confidence for positive classification
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, using rule-based fallback")
            self.model = None
        else:
            self.model = self._load_model(model_path) if model_path else None

        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureExtractor()

    def _load_model(self, model_path: Path) -> Optional[lgb.Booster]:
        """Load pre-trained model.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model or None
        """
        try:
            if model_path.exists():
                return lgb.Booster(model_file=str(model_path))
            else:
                logger.warning(f"Model not found at {model_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train classifier.

        Args:
            X_train: Training features
            y_train: Training labels (0=benign, 1=performance, 2=environment)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training metrics
        """
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM required for training")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)

        # Training parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
        }

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )

        # Get feature importance
        feature_importance = dict(zip(
            RegressionFeatures().feature_names,
            self.model.feature_importance(importance_type='gain'),
        ))

        return {
            'best_iteration': self.model.best_iteration,
            'feature_importance': feature_importance,
        }

    def save_model(self, model_path: Path):
        """Save trained model.

        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    def predict(
        self,
        features: RegressionFeatures,
        timestamp: Optional[datetime] = None,
    ) -> RegressionPrediction:
        """Predict regression type.

        Args:
            features: Input features
            timestamp: Prediction timestamp

        Returns:
            Regression prediction
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Use ML model if available
        if self.model is not None:
            X = features.to_array().reshape(1, -1)
            probs = self.model.predict(X)[0]

            # Get prediction
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            regression_types = [
                RegressionType.BENIGN,
                RegressionType.PERFORMANCE_REGRESSION,
                RegressionType.ENVIRONMENT_REGRESSION,
            ]
            regression_type = regression_types[pred_idx]

            # Get feature importance
            feature_importance = self.model.feature_importance(importance_type='gain')
            feature_names = features.feature_names

            # Top contributing factors
            top_indices = np.argsort(feature_importance)[::-1][:5]
            contributing_factors = [feature_names[i] for i in top_indices]

        else:
            # Rule-based fallback
            regression_type, confidence, contributing_factors = self._rule_based_prediction(features)

        # Generate recommendations
        recommended_investigation = self._generate_recommendations(
            regression_type,
            features,
            contributing_factors,
        )

        return RegressionPrediction(
            timestamp=timestamp,
            regression_type=regression_type,
            confidence=confidence,
            features=features,
            contributing_factors=contributing_factors,
            recommended_investigation=recommended_investigation,
        )

    def _rule_based_prediction(
        self,
        features: RegressionFeatures,
    ) -> Tuple[RegressionType, float, List[str]]:
        """Rule-based prediction fallback.

        Args:
            features: Input features

        Returns:
            Tuple of (regression_type, confidence, contributing_factors)
        """
        contributing_factors = []

        # Check for performance regression indicators
        perf_score = 0.0
        if features.latency_delta > 0.2:  # >20% increase
            perf_score += 0.3
            contributing_factors.append(f"latency_delta: +{features.latency_delta * 100:.1f}%")

        if features.reward_delta < -0.1:  # >10% decrease
            perf_score += 0.3
            contributing_factors.append(f"reward_delta: {features.reward_delta * 100:.1f}%")

        if features.throughput_delta < -0.15:  # >15% decrease
            perf_score += 0.2
            contributing_factors.append(f"throughput_delta: {features.throughput_delta * 100:.1f}%")

        if features.hyperparameter_drift_score > 0.3:
            perf_score += 0.2
            contributing_factors.append(f"hyperparameter_drift_score: {features.hyperparameter_drift_score:.2f}")

        # Check for environment regression indicators
        env_score = 0.0
        if features.error_rate_delta > 0.05:  # >5% increase
            env_score += 0.3
            contributing_factors.append(f"error_rate_delta: +{features.error_rate_delta * 100:.1f}%")

        if features.external_api_latency_delta > 0.3:  # >30% increase
            env_score += 0.3
            contributing_factors.append(f"external_api_latency_delta: +{features.external_api_latency_delta * 100:.1f}%")

        if features.database_latency_delta > 0.3:
            env_score += 0.2
            contributing_factors.append(f"database_latency_delta: +{features.database_latency_delta * 100:.1f}%")

        if features.cache_hit_rate_delta < -0.1:  # >10% decrease
            env_score += 0.2
            contributing_factors.append(f"cache_hit_rate_delta: {features.cache_hit_rate_delta * 100:.1f}%")

        # Determine classification
        if perf_score > env_score and perf_score > 0.5:
            return RegressionType.PERFORMANCE_REGRESSION, min(perf_score, 1.0), contributing_factors
        elif env_score > 0.5:
            return RegressionType.ENVIRONMENT_REGRESSION, min(env_score, 1.0), contributing_factors
        else:
            return RegressionType.BENIGN, max(1.0 - perf_score - env_score, 0.5), []

    def _generate_recommendations(
        self,
        regression_type: RegressionType,
        features: RegressionFeatures,
        contributing_factors: List[str],
    ) -> List[str]:
        """Generate investigation recommendations.

        Args:
            regression_type: Predicted regression type
            features: Input features
            contributing_factors: Top contributing factors

        Returns:
            List of recommended investigation steps
        """
        recommendations = []

        if regression_type == RegressionType.PERFORMANCE_REGRESSION:
            recommendations.extend([
                "Review recent code changes and deployments",
                "Check for algorithm or hyperparameter changes",
                "Profile CPU and memory usage during training",
                "Analyze training convergence patterns",
            ])

            if features.hyperparameter_drift_score > 0.3:
                recommendations.append("Investigate hyperparameter optimization results")

            if features.reward_delta < -0.2:
                recommendations.append("Review reward shaping and environment configuration")

        elif regression_type == RegressionType.ENVIRONMENT_REGRESSION:
            recommendations.extend([
                "Check external service health and latencies",
                "Verify database connection pool and query performance",
                "Review infrastructure changes (scaling, migrations, etc.)",
                "Check cache hit rates and invalidation patterns",
            ])

            if features.error_spike_detected:
                recommendations.append("Investigate error logs for failure patterns")

            if features.external_api_latency_delta > 0.3:
                recommendations.append("Contact external API providers for service status")

        else:
            recommendations.append("Continue monitoring - no immediate action required")

        return recommendations

    async def predict_batch(
        self,
        features_batch: List[RegressionFeatures],
    ) -> List[RegressionPrediction]:
        """Batch prediction.

        Args:
            features_batch: Batch of features

        Returns:
            List of predictions
        """
        predictions = []

        for features in features_batch:
            prediction = self.predict(features)
            predictions.append(prediction)

        return predictions


# Example usage
if __name__ == '__main__':
    # Example: Rule-based prediction
    classifier = RegressionClassifier(confidence_threshold=0.7)

    # Create sample features
    features = RegressionFeatures(
        reward_delta=-0.15,
        latency_delta=0.35,
        error_rate_delta=0.02,
        throughput_delta=-0.10,
        queue_depth_mean=45.0,
        cpu_utilization_mean=0.75,
        hyperparameter_drift_score=0.4,
        error_spike_detected=False,
    )

    # Predict
    prediction = classifier.predict(features)

    print(prediction.to_markdown())
    print("\nJSON:")
    print(json.dumps(prediction.to_dict(), indent=2))

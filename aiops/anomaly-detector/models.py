"""
Anomaly Detection Models
Implements STL decomposition, IsolationForest, and EWMA change-point detection.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, deque

# ML libraries
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    score: float
    confidence: float
    severity: str  # low, medium, high, critical
    reason: str
    method: str  # stl, isolation_forest, ewma
    timestamp: datetime
    recommendations: List[str]


class AnomalyDetector:
    """
    Multi-model anomaly detector using:
    - STL (Seasonal-Trend decomposition with LOESS) for seasonality
    - Isolation Forest for multivariate outlier detection
    - EWMA (Exponentially Weighted Moving Average) for change-point detection
    """

    def __init__(self, window_hours: int = 24, score_threshold: float = 0.8):
        """
        Initialize anomaly detector.

        Args:
            window_hours: Historical window for baseline calculation
            score_threshold: Threshold for anomaly classification (0-1)
        """
        self.window_hours = window_hours
        self.score_threshold = score_threshold

        # Model cache and history
        self.stl_models = {}  # signal -> STL model
        self.isolation_forests = {}  # service -> IsolationForest
        self.ewma_baselines = {}  # signal -> EWMA baseline
        self.anomaly_history = deque(maxlen=1000)  # Recent anomalies

        # Performance tracking
        self.stats = defaultdict(lambda: {
            "predictions": 0,
            "anomalies": 0,
            "false_positives": 0,
            "false_negatives": 0
        })

    def detect(
        self,
        signal_name: str,
        service_name: str,
        data: List[Tuple[datetime, float]]
    ) -> Dict:
        """
        Detect anomalies in a time series signal.

        Args:
            signal_name: Name of the metric/signal
            service_name: Name of the service
            data: List of (timestamp, value) tuples

        Returns:
            Dictionary with anomaly verdict and details
        """
        try:
            if not data or len(data) < 10:
                return self._no_anomaly_result("insufficient_data")

            # Convert to pandas series
            df = pd.DataFrame(data, columns=['timestamp', 'value'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Remove duplicates and NaNs
            df = df[~df.index.duplicated(keep='first')]
            df = df.dropna()

            if len(df) < 10:
                return self._no_anomaly_result("insufficient_clean_data")

            # Run multiple detection methods
            stl_result = self._detect_stl(signal_name, df)
            isolation_result = self._detect_isolation_forest(service_name, df)
            ewma_result = self._detect_ewma(signal_name, df)

            # Ensemble decision: combine scores with weights
            ensemble_score = (
                stl_result["score"] * 0.4 +
                isolation_result["score"] * 0.4 +
                ewma_result["score"] * 0.2
            )

            # Determine severity
            severity = self._calculate_severity(ensemble_score)

            # Is it an anomaly?
            is_anomaly = ensemble_score >= self.score_threshold

            # Confidence based on model agreement
            scores = [
                stl_result["score"],
                isolation_result["score"],
                ewma_result["score"]
            ]
            confidence = 1.0 - np.std(scores)  # Higher when models agree

            # Generate reason and recommendations
            reason = self._generate_reason(
                stl_result, isolation_result, ewma_result, ensemble_score
            )
            recommendations = self._generate_recommendations(
                signal_name, service_name, stl_result, isolation_result, ewma_result
            )

            # Record anomaly
            result = {
                "is_anomaly": is_anomaly,
                "score": float(ensemble_score),
                "confidence": float(confidence),
                "severity": severity,
                "reason": reason,
                "method": "ensemble",
                "timestamp": datetime.now(),
                "recommendations": recommendations,
                "model_scores": {
                    "stl": stl_result["score"],
                    "isolation_forest": isolation_result["score"],
                    "ewma": ewma_result["score"]
                }
            }

            # Update stats
            self.stats[signal_name]["predictions"] += 1
            if is_anomaly:
                self.stats[signal_name]["anomalies"] += 1
                self.anomaly_history.append({
                    "signal": signal_name,
                    "service": service_name,
                    "timestamp": datetime.now(),
                    **result
                })

            return result

        except Exception as e:
            logger.error(f"Anomaly detection failed for {signal_name}: {e}", exc_info=True)
            return self._no_anomaly_result(f"error: {str(e)}")

    def _detect_stl(self, signal_name: str, df: pd.DataFrame) -> Dict:
        """
        STL-based anomaly detection using seasonal decomposition.

        Returns:
            Dict with score and details
        """
        try:
            if len(df) < 24:  # Need at least 2 periods
                return {"score": 0.0, "details": "insufficient_data"}

            # Resample to regular intervals (1min) if needed
            df_resampled = df.resample('1min').mean().interpolate(method='linear')

            # STL decomposition (period = 60 for hourly seasonality)
            period = min(60, len(df_resampled) // 3)
            stl = STL(df_resampled['value'], period=period, seasonal=13)
            result = stl.fit()

            # Cache model
            self.stl_models[signal_name] = result

            # Anomaly score based on residuals (robust z-score)
            residuals = result.resid
            median = np.median(residuals)
            mad = np.median(np.abs(residuals - median))
            if mad == 0:
                return {"score": 0.0, "details": "no_variance"}

            # Modified z-score for last value
            last_residual = residuals.iloc[-1]
            z_score = abs(0.6745 * (last_residual - median) / mad)

            # Convert to 0-1 score (clamp at 5 sigma)
            score = min(z_score / 5.0, 1.0)

            return {
                "score": float(score),
                "details": {
                    "z_score": float(z_score),
                    "residual": float(last_residual),
                    "trend": float(result.trend.iloc[-1]),
                    "seasonal": float(result.seasonal.iloc[-1])
                }
            }

        except Exception as e:
            logger.warning(f"STL detection failed for {signal_name}: {e}")
            return {"score": 0.0, "details": f"error: {str(e)}"}

    def _detect_isolation_forest(self, service_name: str, df: pd.DataFrame) -> Dict:
        """
        Isolation Forest for multivariate outlier detection.

        Returns:
            Dict with score and details
        """
        try:
            if len(df) < 10:
                return {"score": 0.0, "details": "insufficient_data"}

            # Feature engineering
            df_features = df.copy()
            df_features['value_diff'] = df_features['value'].diff().fillna(0)
            df_features['value_rolling_mean'] = (
                df_features['value'].rolling(window=5, min_periods=1).mean()
            )
            df_features['value_rolling_std'] = (
                df_features['value'].rolling(window=5, min_periods=1).std().fillna(0)
            )

            # Prepare features
            features = df_features[[
                'value', 'value_diff', 'value_rolling_mean', 'value_rolling_std'
            ]].values

            # Train or use cached model
            if service_name not in self.isolation_forests or len(df) > 100:
                # Retrain with more data
                model = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                model.fit(features)
                self.isolation_forests[service_name] = model
            else:
                model = self.isolation_forests[service_name]

            # Predict on last point
            last_features = features[-1:, :]
            prediction = model.predict(last_features)[0]
            anomaly_score_raw = model.score_samples(last_features)[0]

            # Convert to 0-1 score (IsolationForest scores are negative)
            # Typical range: -0.5 to -0.1 for normal, < -0.5 for anomalies
            score = max(0.0, min(1.0, (-anomaly_score_raw - 0.1) / 0.4))

            is_outlier = prediction == -1

            return {
                "score": float(score) if is_outlier else 0.0,
                "details": {
                    "is_outlier": bool(is_outlier),
                    "raw_score": float(anomaly_score_raw)
                }
            }

        except Exception as e:
            logger.warning(f"IsolationForest detection failed for {service_name}: {e}")
            return {"score": 0.0, "details": f"error: {str(e)}"}

    def _detect_ewma(self, signal_name: str, df: pd.DataFrame) -> Dict:
        """
        EWMA-based change-point detection.

        Returns:
            Dict with score and details
        """
        try:
            if len(df) < 5:
                return {"score": 0.0, "details": "insufficient_data"}

            # Calculate EWMA
            alpha = 0.3  # Smoothing factor
            ewma = df['value'].ewm(alpha=alpha, adjust=False).mean()

            # Calculate deviation from EWMA
            deviation = abs(df['value'] - ewma)
            ewma_std = deviation.ewm(alpha=alpha, adjust=False).std()

            # Cache baseline
            self.ewma_baselines[signal_name] = {
                "ewma": ewma,
                "std": ewma_std
            }

            # Score based on deviation in standard deviations
            last_deviation = deviation.iloc[-1]
            last_std = ewma_std.iloc[-1]

            if last_std == 0 or np.isnan(last_std):
                return {"score": 0.0, "details": "no_variance"}

            z_score = last_deviation / last_std

            # Convert to 0-1 score (clamp at 4 sigma for EWMA)
            score = min(z_score / 4.0, 1.0)

            return {
                "score": float(score),
                "details": {
                    "z_score": float(z_score),
                    "deviation": float(last_deviation),
                    "ewma": float(ewma.iloc[-1]),
                    "std": float(last_std)
                }
            }

        except Exception as e:
            logger.warning(f"EWMA detection failed for {signal_name}: {e}")
            return {"score": 0.0, "details": f"error: {str(e)}"}

    def _calculate_severity(self, score: float) -> str:
        """Calculate severity level from anomaly score"""
        if score >= 0.95:
            return "critical"
        elif score >= 0.85:
            return "high"
        elif score >= 0.70:
            return "medium"
        else:
            return "low"

    def _generate_reason(
        self,
        stl_result: Dict,
        isolation_result: Dict,
        ewma_result: Dict,
        ensemble_score: float
    ) -> str:
        """Generate human-readable reason for anomaly"""
        reasons = []

        if stl_result["score"] > 0.7:
            details = stl_result.get("details", {})
            if isinstance(details, dict):
                z = details.get("z_score", 0)
                reasons.append(f"Significant deviation from seasonal pattern (z-score: {z:.2f})")

        if isolation_result["score"] > 0.7:
            reasons.append("Multivariate outlier detected")

        if ewma_result["score"] > 0.7:
            details = ewma_result.get("details", {})
            if isinstance(details, dict):
                z = details.get("z_score", 0)
                reasons.append(f"Rapid change detected (z-score: {z:.2f})")

        if not reasons:
            return f"Anomaly score {ensemble_score:.2f} above threshold"

        return "; ".join(reasons)

    def _generate_recommendations(
        self,
        signal_name: str,
        service_name: str,
        stl_result: Dict,
        isolation_result: Dict,
        ewma_result: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Generic recommendations based on signal type
        if "latency" in signal_name.lower() or "duration" in signal_name.lower():
            recommendations.append("Check for slow database queries or external API calls")
            recommendations.append("Review recent deployments or configuration changes")
            if stl_result["score"] > 0.8:
                recommendations.append("Consider scaling out backend replicas")

        if "error" in signal_name.lower() or "5xx" in signal_name.lower():
            recommendations.append("Check application logs for error stack traces")
            recommendations.append("Verify database connectivity and health")
            if ewma_result["score"] > 0.8:
                recommendations.append("Consider rolling back recent deployment")

        if "memory" in signal_name.lower():
            recommendations.append("Investigate for memory leaks")
            recommendations.append("Consider restarting affected pods")

        if "cpu" in signal_name.lower():
            recommendations.append("Check for CPU-intensive operations or infinite loops")
            recommendations.append("Review recent code changes for performance regressions")

        # Add monitoring recommendation
        recommendations.append(f"View detailed metrics in Grafana for {service_name}")

        return recommendations[:5]  # Limit to top 5

    def _no_anomaly_result(self, reason: str) -> Dict:
        """Return a no-anomaly result"""
        return {
            "is_anomaly": False,
            "score": 0.0,
            "confidence": 0.0,
            "severity": "none",
            "reason": reason,
            "method": "none",
            "timestamp": datetime.now(),
            "recommendations": []
        }

    def get_recent_anomalies(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get recent anomalies within time window"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [
            a for a in self.anomaly_history
            if a["timestamp"] >= cutoff
        ]
        return list(recent)[-limit:]

    def get_model_status(self) -> Dict:
        """Get model status and statistics"""
        return {
            "stl_models_cached": len(self.stl_models),
            "isolation_forests_cached": len(self.isolation_forests),
            "ewma_baselines_cached": len(self.ewma_baselines),
            "anomalies_in_history": len(self.anomaly_history),
            "signal_stats": dict(self.stats)
        }

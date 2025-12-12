#!/usr/bin/env python3
"""
Lightweight Anomaly Detector - Statistical Anomaly Detection

Uses Z-score and EWMA (Exponentially Weighted Moving Average) hybrid approach
for anomaly detection. No external ML libraries required.

Detects anomalies in:
- P95 latency
- Error rate
- CPU utilization
- Memory utilization

Classifies anomalies as:
- spike: sudden increase
- drop: sudden decrease
- drift: gradual trend change

Usage:
    python anomaly_detector_lightweight.py --data stability/ --duration 168
    python anomaly_detector_lightweight.py --data stability/ --duration 2 --test-mode

Author: T.A.R.S. Platform Team
Phase: 14.6 - Post-GA 7-Day Stabilization & Retrospective
"""

import json
import logging
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enterprise imports (Phase 14.6)
try:
    from enterprise_config import load_config
    from compliance.enforcer import ComplianceEnforcer
    from security.encryption import AESEncryption
    from security.signing import ReportSigner
    from telemetry import get_logger
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False
    print("Warning: Enterprise features not available. Running in legacy mode.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """
    A detected anomaly event.
    """
    timestamp: str
    metric_name: str
    actual_value: float
    expected_value: float  # EWMA prediction
    deviation: float  # Absolute deviation
    z_score: float  # Standardized deviation
    classification: str  # "spike", "drop", "drift"
    severity: str  # "low", "medium", "high"
    confidence: float  # 0-100%


@dataclass
class MetricTimeSeries:
    """
    Time series data for a single metric.
    """
    metric_name: str
    timestamps: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    ewma_values: List[float] = field(default_factory=list)
    z_scores: List[float] = field(default_factory=list)


@dataclass
class AnomalyReport:
    """
    Complete anomaly detection report.
    """
    start_time: str
    end_time: str
    duration_hours: float
    total_datapoints: int

    # Anomaly Counts
    total_anomalies: int = 0
    high_severity_anomalies: int = 0
    medium_severity_anomalies: int = 0
    low_severity_anomalies: int = 0

    # Classification Breakdown
    spike_count: int = 0
    drop_count: int = 0
    drift_count: int = 0

    # Metric Breakdown
    p95_latency_anomalies: int = 0
    error_rate_anomalies: int = 0
    cpu_anomalies: int = 0
    memory_anomalies: int = 0

    # Detected Anomalies
    anomalies: List[AnomalyEvent] = field(default_factory=list)

    # Summary Statistics
    avg_z_score: float = 0.0
    max_z_score: float = 0.0


class EWMACalculator:
    """
    Exponentially Weighted Moving Average calculator.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EWMA calculator.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Lower = more smoothing.
                   Default 0.3 provides good balance for hourly data.
        """
        self.alpha = alpha
        self.ewma_value: Optional[float] = None
        logger.info(f"EWMACalculator initialized with alpha={alpha}")

    def update(self, value: float) -> float:
        """
        Update EWMA with new value.

        Formula: EWMA_t = alpha * value + (1 - alpha) * EWMA_{t-1}

        Args:
            value: New data point

        Returns:
            Updated EWMA value
        """
        if self.ewma_value is None:
            # First value: initialize EWMA with the first data point
            self.ewma_value = value
        else:
            # Subsequent values: apply EWMA formula
            self.ewma_value = self.alpha * value + (1 - self.alpha) * self.ewma_value

        return self.ewma_value

    def predict_next(self) -> float:
        """
        Predict next value based on current EWMA.

        Returns:
            Predicted value (current EWMA)
        """
        if self.ewma_value is None:
            return 0.0
        return self.ewma_value

    def reset(self) -> None:
        """
        Reset EWMA state.
        """
        self.ewma_value = None


class ZScoreCalculator:
    """
    Z-score calculator for anomaly detection.
    """

    def __init__(self, window_size: int = 48):
        """
        Initialize Z-score calculator.

        Args:
            window_size: Number of recent points for mean/stddev calculation.
                         Default 48 = 24 hours @ 30min intervals.
        """
        self.window_size = window_size
        self.values_window: deque = deque(maxlen=window_size)
        logger.info(f"ZScoreCalculator initialized with window_size={window_size}")

    def add_value(self, value: float) -> None:
        """
        Add a value to the rolling window.

        Args:
            value: New data point
        """
        self.values_window.append(value)

    def calculate(self, value: float) -> float:
        """
        Calculate Z-score for a value.

        Formula: z = (value - mean) / stddev

        Args:
            value: Value to calculate Z-score for

        Returns:
            Z-score (standardized deviation from mean)
        """
        if len(self.values_window) < 2:
            # Not enough data for meaningful statistics
            return 0.0

        # Calculate mean and stddev from window
        mean = calculate_mean(list(self.values_window))
        stddev = calculate_stddev(list(self.values_window), mean)

        # Handle edge case: stddev = 0 (constant values)
        if stddev == 0:
            return 0.0

        # Calculate Z-score
        z_score = (value - mean) / stddev
        return z_score

    def get_statistics(self) -> Tuple[float, float]:
        """
        Get current mean and standard deviation.

        Returns:
            Tuple of (mean, stddev)
        """
        if len(self.values_window) < 2:
            return (0.0, 0.0)

        values_list = list(self.values_window)
        mean = calculate_mean(values_list)
        stddev = calculate_stddev(values_list, mean)

        return (mean, stddev)


class AnomalyClassifier:
    """
    Classifies anomalies based on Z-score and deviation direction.
    """

    # Z-score thresholds for severity
    Z_SCORE_THRESHOLDS = {
        "low": 2.0,      # 2 sigma (95% confidence)
        "medium": 2.5,   # 2.5 sigma (98.8% confidence)
        "high": 3.0      # 3 sigma (99.7% confidence)
    }

    def __init__(self):
        """
        Initialize anomaly classifier.
        """
        logger.info("AnomalyClassifier initialized")

    def classify(
        self,
        actual: float,
        expected: float,
        z_score: float,
        metric_name: str
    ) -> Tuple[str, str]:
        """
        Classify anomaly type and severity.

        Args:
            actual: Actual value
            expected: Expected value (EWMA)
            z_score: Z-score
            metric_name: Name of metric

        Returns:
            Tuple of (classification, severity)
        """
        abs_z = abs(z_score)

        # Classify severity based on abs(z_score)
        if abs_z >= self.Z_SCORE_THRESHOLDS["high"]:
            severity = "high"
        elif abs_z >= self.Z_SCORE_THRESHOLDS["medium"]:
            severity = "medium"
        elif abs_z >= self.Z_SCORE_THRESHOLDS["low"]:
            severity = "low"
        else:
            severity = "none"

        # Classify type based on direction and magnitude
        deviation_percent = abs((actual - expected) / expected) * 100 if expected != 0 else 0

        if actual > expected:
            # Value higher than expected
            if deviation_percent > 50:
                # Sudden large increase = spike
                classification = "spike"
            elif deviation_percent > 10:
                # Moderate increase over time = drift
                classification = "drift"
            else:
                classification = "spike"
        elif actual < expected:
            # Value lower than expected
            if deviation_percent > 50:
                # Sudden large decrease = drop
                classification = "drop"
            elif deviation_percent > 10:
                # Gradual decrease = drift
                classification = "drift"
            else:
                classification = "drop"
        else:
            # Shouldn't happen for anomalies, but handle it
            classification = "drift"

        return (classification, severity)

    def calculate_confidence(self, z_score: float) -> float:
        """
        Calculate confidence level for anomaly detection.

        Based on normal distribution percentiles:
        - Z=2.0 -> 95.45% confidence
        - Z=2.5 -> 98.76% confidence
        - Z=3.0 -> 99.73% confidence

        Args:
            z_score: Z-score value

        Returns:
            Confidence percentage (0-100)
        """
        abs_z = abs(z_score)

        # Empirical mapping of Z-score to confidence percentiles
        # Using error function (erf) approximation for normal distribution CDF
        # confidence = 100 * (1 - 2 * (1 - Φ(z))) where Φ is CDF

        # Simplified approximation using known values:
        if abs_z >= 3.0:
            return 99.73
        elif abs_z >= 2.5:
            return 98.76
        elif abs_z >= 2.0:
            return 95.45
        elif abs_z >= 1.5:
            return 86.64
        elif abs_z >= 1.0:
            return 68.27
        else:
            # Linear interpolation for values < 1.0
            return abs_z * 68.27


class AnomalyDetector:
    """
    Main lightweight anomaly detector.
    """

    # Metrics to monitor
    MONITORED_METRICS = [
        "p95_latency_ms",
        "error_rate",
        "avg_cpu_percent",
        "avg_memory_percent"
    ]

    def __init__(
        self,
        data_dir: str,
        duration_hours: int = 168,
        ewma_alpha: float = 0.3,
        z_score_window: int = 48,
        output_file: str = "anomaly_events.json",
        # Enterprise features
        compliance_enforcer: Optional[ComplianceEnforcer] = None,
        encryptor: Optional[AESEncryption] = None,
        signer: Optional[ReportSigner] = None,
    ):
        """
        Initialize anomaly detector.

        Args:
            data_dir: Path to stability data directory
            duration_hours: Monitoring duration (default: 168 = 7 days)
            ewma_alpha: EWMA smoothing factor (default: 0.3)
            z_score_window: Z-score rolling window size (default: 48)
            output_file: Output file for anomaly events
            compliance_enforcer: Optional compliance enforcer (enterprise)
            encryptor: Optional AES encryptor (enterprise)
            signer: Optional RSA signer (enterprise)
        """
        self.data_dir = Path(data_dir)
        self.duration_hours = duration_hours
        self.output_file = output_file

        # Enterprise features (Phase 14.6)
        self.compliance_enforcer = compliance_enforcer
        self.encryptor = encryptor
        self.signer = signer

        # Create calculators for each metric
        self.ewma_calculators: Dict[str, EWMACalculator] = {
            metric: EWMACalculator(ewma_alpha)
            for metric in self.MONITORED_METRICS
        }

        self.z_score_calculators: Dict[str, ZScoreCalculator] = {
            metric: ZScoreCalculator(z_score_window)
            for metric in self.MONITORED_METRICS
        }

        self.classifier = AnomalyClassifier()

        self.anomalies: List[AnomalyEvent] = []
        self.time_series: Dict[str, MetricTimeSeries] = {
            metric: MetricTimeSeries(metric_name=metric)
            for metric in self.MONITORED_METRICS
        }

        logger.info("AnomalyDetector initialized")

    def load_stability_snapshots(self) -> List[Dict[str, Any]]:
        """
        Load all stability snapshots from data directory.

        Returns:
            List of snapshot dictionaries, sorted by timestamp
        """
        snapshots = []

        # Load all day_XX.json files from data_dir
        day_files = sorted(self.data_dir.glob("day_*.json"))

        for day_file in day_files:
            try:
                with open(day_file, 'r') as f:
                    day_data = json.load(f)

                    # Check if file contains snapshots array
                    if isinstance(day_data, dict) and 'snapshots' in day_data:
                        snapshots.extend(day_data['snapshots'])
                    elif isinstance(day_data, list):
                        # File is an array of snapshots
                        snapshots.extend(day_data)
                    else:
                        # Single snapshot
                        snapshots.append(day_data)

                logger.info(f"Loaded snapshots from {day_file.name}")

            except Exception as e:
                logger.error(f"Error loading {day_file}: {e}")
                continue

        # Sort by timestamp
        snapshots.sort(key=lambda x: x.get('timestamp', ''))

        logger.info(f"Loaded {len(snapshots)} total snapshots")
        return snapshots

    def detect_anomalies_for_metric(
        self,
        metric_name: str,
        timestamps: List[str],
        values: List[float]
    ) -> List[AnomalyEvent]:
        """
        Detect anomalies for a single metric.

        Args:
            metric_name: Name of metric
            timestamps: List of timestamps
            values: List of metric values

        Returns:
            List of detected AnomalyEvent objects
        """
        anomalies = []

        # Get calculators for this metric
        ewma_calc = self.ewma_calculators[metric_name]
        z_score_calc = self.z_score_calculators[metric_name]

        # Reset calculators for fresh analysis
        ewma_calc.reset()

        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            # Update EWMA
            ewma_value = ewma_calc.update(value)

            # Add value to Z-score calculator's window
            z_score_calc.add_value(value)

            # Calculate Z-score
            z_score = z_score_calc.calculate(value)

            # Store time series data
            self.time_series[metric_name].timestamps.append(timestamp)
            self.time_series[metric_name].values.append(value)
            self.time_series[metric_name].ewma_values.append(ewma_value)
            self.time_series[metric_name].z_scores.append(z_score)

            # Check if this is an anomaly (Z-score exceeds low threshold)
            abs_z = abs(z_score)
            if abs_z >= self.classifier.Z_SCORE_THRESHOLDS["low"]:
                # Classify anomaly type and severity
                classification, severity = self.classifier.classify(
                    actual=value,
                    expected=ewma_value,
                    z_score=z_score,
                    metric_name=metric_name
                )

                # Calculate confidence
                confidence = self.classifier.calculate_confidence(z_score)

                # Create anomaly event
                anomaly = AnomalyEvent(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    actual_value=value,
                    expected_value=ewma_value,
                    deviation=value - ewma_value,
                    z_score=z_score,
                    classification=classification,
                    severity=severity,
                    confidence=confidence
                )

                anomalies.append(anomaly)

                logger.info(
                    f"Anomaly detected in {metric_name} at {timestamp}: "
                    f"{classification} ({severity}) - actual={value:.2f}, "
                    f"expected={ewma_value:.2f}, z={z_score:.2f}"
                )

        return anomalies

    def detect_all_anomalies(self) -> AnomalyReport:
        """
        Detect anomalies across all monitored metrics.

        Returns:
            AnomalyReport with all detected anomalies
        """
        logger.info("Starting anomaly detection across all metrics")

        # Load stability snapshots
        snapshots = self.load_stability_snapshots()

        if not snapshots:
            logger.warning("No snapshots found - returning empty report")
            return AnomalyReport(
                start_time=datetime.now(timezone.utc).isoformat(),
                end_time=datetime.now(timezone.utc).isoformat(),
                duration_hours=0.0,
                total_datapoints=0
            )

        # Extract timestamps and start/end times
        timestamps = [s.get('timestamp', '') for s in snapshots]
        start_time = snapshots[0].get('timestamp', '')
        end_time = snapshots[-1].get('timestamp', '')

        # Extract time series for each metric
        metric_series = {}
        for metric in self.MONITORED_METRICS:
            values = []
            for snapshot in snapshots:
                value = snapshot.get(metric, 0.0)
                values.append(float(value))
            metric_series[metric] = values

        logger.info(f"Extracted {len(snapshots)} snapshots across {len(self.MONITORED_METRICS)} metrics")

        # Detect anomalies for each metric
        all_anomalies = []
        for metric in self.MONITORED_METRICS:
            logger.info(f"Detecting anomalies for {metric}")
            metric_anomalies = self.detect_anomalies_for_metric(
                metric_name=metric,
                timestamps=timestamps,
                values=metric_series[metric]
            )
            all_anomalies.extend(metric_anomalies)
            logger.info(f"Found {len(metric_anomalies)} anomalies in {metric}")

        # Calculate summary statistics
        avg_z_score, max_z_score = self.generate_summary_stats(all_anomalies)

        # Count anomalies by severity
        high_count = sum(1 for a in all_anomalies if a.severity == "high")
        medium_count = sum(1 for a in all_anomalies if a.severity == "medium")
        low_count = sum(1 for a in all_anomalies if a.severity == "low")

        # Count anomalies by type
        spike_count = sum(1 for a in all_anomalies if a.classification == "spike")
        drop_count = sum(1 for a in all_anomalies if a.classification == "drop")
        drift_count = sum(1 for a in all_anomalies if a.classification == "drift")

        # Count anomalies by metric
        p95_count = sum(1 for a in all_anomalies if a.metric_name == "p95_latency_ms")
        error_count = sum(1 for a in all_anomalies if a.metric_name == "error_rate")
        cpu_count = sum(1 for a in all_anomalies if a.metric_name == "avg_cpu_percent")
        memory_count = sum(1 for a in all_anomalies if a.metric_name == "avg_memory_percent")

        # Create report
        report = AnomalyReport(
            start_time=start_time,
            end_time=end_time,
            duration_hours=self.duration_hours,
            total_datapoints=len(snapshots) * len(self.MONITORED_METRICS),
            total_anomalies=len(all_anomalies),
            high_severity_anomalies=high_count,
            medium_severity_anomalies=medium_count,
            low_severity_anomalies=low_count,
            spike_count=spike_count,
            drop_count=drop_count,
            drift_count=drift_count,
            p95_latency_anomalies=p95_count,
            error_rate_anomalies=error_count,
            cpu_anomalies=cpu_count,
            memory_anomalies=memory_count,
            anomalies=all_anomalies,
            avg_z_score=avg_z_score,
            max_z_score=max_z_score
        )

        logger.info(
            f"Anomaly detection complete: {len(all_anomalies)} total anomalies "
            f"({high_count} high, {medium_count} medium, {low_count} low)"
        )

        return report

    def save_anomaly_events(self, report: AnomalyReport) -> None:
        """
        Save anomaly events to JSON file.

        Args:
            report: Anomaly report
        """
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert anomaly events to dictionaries
        anomaly_dicts = [asdict(a) for a in report.anomalies]

        # Create output structure
        output_data = {
            "summary": {
                "start_time": report.start_time,
                "end_time": report.end_time,
                "duration_hours": report.duration_hours,
                "total_datapoints": report.total_datapoints,
                "total_anomalies": report.total_anomalies,
                "high_severity_anomalies": report.high_severity_anomalies,
                "medium_severity_anomalies": report.medium_severity_anomalies,
                "low_severity_anomalies": report.low_severity_anomalies,
                "spike_count": report.spike_count,
                "drop_count": report.drop_count,
                "drift_count": report.drift_count,
                "avg_z_score": report.avg_z_score,
                "max_z_score": report.max_z_score
            },
            "anomalies_by_metric": {
                "p95_latency_ms": report.p95_latency_anomalies,
                "error_rate": report.error_rate_anomalies,
                "avg_cpu_percent": report.cpu_anomalies,
                "avg_memory_percent": report.memory_anomalies
            },
            "anomalies": anomaly_dicts
        }

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Enterprise: Encrypt if enabled
        if self.encryptor:
            encrypted_file = output_path.with_suffix(".json.enc")
            self.encryptor.encrypt_file(output_path, encrypted_file)
            logger.info(f"Encrypted anomaly report: {encrypted_file}")

        # Enterprise: Sign if enabled
        if self.signer:
            signature = self.signer.sign_file(output_path)
            sig_file = output_path.with_suffix(".json.sig")
            with open(sig_file, "w") as f:
                f.write(f"RSA-PSS-SHA256\n{signature}\n")
            logger.info(f"Signed anomaly report: {sig_file}")

        logger.info(f"Saved {len(anomaly_dicts)} anomaly events to {output_path}")

    def generate_summary_stats(
        self,
        anomalies: List[AnomalyEvent]
    ) -> Tuple[float, float]:
        """
        Generate summary statistics for anomalies.

        Args:
            anomalies: List of detected anomalies

        Returns:
            Tuple of (avg_z_score, max_z_score)
        """
        if not anomalies:
            return (0.0, 0.0)

        # Calculate average Z-score (use absolute values)
        z_scores = [abs(a.z_score) for a in anomalies]
        avg_z_score = sum(z_scores) / len(z_scores)
        max_z_score = max(z_scores)

        return (avg_z_score, max_z_score)


def calculate_mean(values: List[float]) -> float:
    """
    Calculate mean of a list of values.

    Args:
        values: List of numeric values

    Returns:
        Mean value
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_stddev(values: List[float], mean: float) -> float:
    """
    Calculate standard deviation of a list of values.

    Args:
        values: List of numeric values
        mean: Pre-calculated mean

    Returns:
        Standard deviation
    """
    if not values or len(values) < 2:
        return 0.0

    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def main():
    """
    CLI entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight Anomaly Detector")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to stability data directory"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=168,
        help="Monitoring duration in hours (default: 168 = 7 days)"
    )
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=0.3,
        help="EWMA smoothing factor (default: 0.3)"
    )
    parser.add_argument(
        "--z-window",
        type=int,
        default=48,
        help="Z-score rolling window size (default: 48)"
    )
    parser.add_argument(
        "--output",
        default="anomaly_events.json",
        help="Output file for anomaly events (default: anomaly_events.json)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode (use with --duration 1-2 for quick testing)"
    )

    # Enterprise configuration arguments (Phase 14.6)
    if ENTERPRISE_AVAILABLE:
        parser.add_argument("--profile", type=str, default="local", help="Enterprise config profile (local, dev, staging, prod)")
        parser.add_argument("--config", type=str, help="Path to enterprise config file")
        parser.add_argument("--encrypt", action="store_true", help="Encrypt output files (requires AES key)")
        parser.add_argument("--sign", action="store_true", help="Sign output files (requires RSA key)")
        parser.add_argument("--no-compliance", action="store_true", help="Disable compliance enforcement")

    args = parser.parse_args()

    # Initialize enterprise features
    compliance_enforcer = None
    encryptor = None
    signer = None
    config = None

    if ENTERPRISE_AVAILABLE:
        try:
            # Load enterprise configuration
            config_file = Path(args.config) if hasattr(args, 'config') and args.config else None
            config = load_config(
                config_file=config_file,
                environment=args.profile if hasattr(args, 'profile') else "local",
            )

            logger.info(f"✓ Enterprise config loaded (profile: {config.environment.value})")

            # Override with CLI args if provided
            if args.output == "anomaly_events.json" and config.observability.output_dir != "output":
                args.output = str(Path(config.observability.output_dir) / "anomaly_events.json")

            # Initialize compliance enforcer
            if not args.no_compliance and config.compliance.enabled_standards:
                from pathlib import Path as P
                compliance_enforcer = ComplianceEnforcer(
                    enabled_standards=config.compliance.enabled_standards,
                    controls_dir=P("compliance/policies"),
                    audit_log_path=P(config.observability.output_dir) / "audit.log" if config.compliance.enable_audit_trail else None,
                    strict_mode=False,
                )
                logger.info(f"✓ Compliance enforcer initialized (standards: {', '.join(config.compliance.enabled_standards)})")

            # Initialize encryption
            if args.encrypt or config.security.enable_encryption:
                key_path = Path(config.security.aes_key_path) if config.security.aes_key_path else None
                if key_path and key_path.exists():
                    encryptor = AESEncryption(key_path=key_path)
                    logger.info(f"✓ AES encryption initialized")
                else:
                    logger.warning("⚠ Encryption requested but no valid AES key found")

            # Initialize signing
            if args.sign or config.security.enable_signing:
                private_key_path = Path(config.security.rsa_private_key_path) if config.security.rsa_private_key_path else None
                if private_key_path and private_key_path.exists():
                    signer = ReportSigner(private_key_path=private_key_path)
                    logger.info(f"✓ RSA signing initialized")
                else:
                    logger.warning("⚠ Signing requested but no valid RSA key found")

        except Exception as e:
            logger.warning(f"⚠ Failed to load enterprise config: {e}")
            logger.info("Falling back to legacy CLI configuration")
            config = None

    logger.info("=" * 60)
    logger.info("Lightweight Anomaly Detector - Starting")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Duration: {args.duration} hours")
    logger.info(f"EWMA alpha: {args.ewma_alpha}")
    logger.info(f"Z-score window: {args.z_window}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info("=" * 60)

    try:
        # Instantiate AnomalyDetector
        detector = AnomalyDetector(
            data_dir=args.data,
            duration_hours=args.duration,
            ewma_alpha=args.ewma_alpha,
            z_score_window=args.z_window,
            output_file=args.output,
            compliance_enforcer=compliance_enforcer,
            encryptor=encryptor,
            signer=signer,
        )

        # Run anomaly detection
        report = detector.detect_all_anomalies()

        # Save anomaly events
        detector.save_anomaly_events(report)

        # Print summary to console
        logger.info("=" * 60)
        logger.info("ANOMALY DETECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Period: {report.start_time} to {report.end_time}")
        logger.info(f"Duration: {report.duration_hours} hours")
        logger.info(f"Total Datapoints: {report.total_datapoints}")
        logger.info("")
        logger.info(f"Total Anomalies: {report.total_anomalies}")
        logger.info(f"  High Severity: {report.high_severity_anomalies}")
        logger.info(f"  Medium Severity: {report.medium_severity_anomalies}")
        logger.info(f"  Low Severity: {report.low_severity_anomalies}")
        logger.info("")
        logger.info(f"Classification Breakdown:")
        logger.info(f"  Spikes: {report.spike_count}")
        logger.info(f"  Drops: {report.drop_count}")
        logger.info(f"  Drifts: {report.drift_count}")
        logger.info("")
        logger.info(f"Metric Breakdown:")
        logger.info(f"  P95 Latency: {report.p95_latency_anomalies}")
        logger.info(f"  Error Rate: {report.error_rate_anomalies}")
        logger.info(f"  CPU Usage: {report.cpu_anomalies}")
        logger.info(f"  Memory Usage: {report.memory_anomalies}")
        logger.info("")
        logger.info(f"Statistics:")
        logger.info(f"  Average Z-score: {report.avg_z_score:.2f}")
        logger.info(f"  Max Z-score: {report.max_z_score:.2f}")
        logger.info("=" * 60)

        # Print top 5 most severe anomalies
        if report.anomalies:
            sorted_anomalies = sorted(
                report.anomalies,
                key=lambda a: abs(a.z_score),
                reverse=True
            )[:5]

            logger.info("TOP 5 MOST SEVERE ANOMALIES:")
            logger.info("=" * 60)
            for i, anomaly in enumerate(sorted_anomalies, 1):
                logger.info(
                    f"{i}. {anomaly.metric_name} at {anomaly.timestamp}"
                )
                logger.info(
                    f"   {anomaly.classification.upper()} ({anomaly.severity}) - "
                    f"Z-score: {anomaly.z_score:.2f}, Confidence: {anomaly.confidence:.1f}%"
                )
                logger.info(
                    f"   Actual: {anomaly.actual_value:.2f}, "
                    f"Expected: {anomaly.expected_value:.2f}, "
                    f"Deviation: {anomaly.deviation:.2f}"
                )
                logger.info("")

        logger.info("=" * 60)
        logger.info("Anomaly detection complete")
        logger.info(f"Results saved to: {args.output}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

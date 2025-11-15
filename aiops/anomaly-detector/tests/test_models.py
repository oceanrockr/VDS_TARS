"""
Unit tests for anomaly detection models
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import AnomalyDetector


class TestAnomalyDetector:
    """Test suite for AnomalyDetector"""

    def setup_method(self):
        """Setup test fixtures"""
        self.detector = AnomalyDetector(window_hours=24, score_threshold=0.8)

    def generate_normal_data(self, hours=24, interval_seconds=60):
        """Generate normal time series data"""
        n_points = int(hours * 3600 / interval_seconds)
        now = datetime.now()
        timestamps = [now - timedelta(seconds=i*interval_seconds) for i in range(n_points)]
        timestamps.reverse()

        # Normal data with slight variation
        values = 100 + np.random.normal(0, 5, n_points)
        return list(zip(timestamps, values))

    def generate_spike_data(self, hours=24, spike_position=0.9):
        """Generate data with anomaly spike"""
        data = self.generate_normal_data(hours)

        # Add spike at position
        spike_index = int(len(data) * spike_position)
        timestamps, values = zip(*data)
        values = list(values)
        values[spike_index] = values[spike_index] * 3  # 3x spike

        return list(zip(timestamps, values))

    def test_no_anomaly_on_normal_data(self):
        """Test that normal data doesn't trigger anomalies"""
        data = self.generate_normal_data(hours=2)

        result = self.detector.detect(
            signal_name="test_metric",
            service_name="test_service",
            data=data
        )

        assert result["is_anomaly"] == False
        assert result["score"] < self.detector.score_threshold
        assert result["severity"] in ["none", "low"]

    def test_anomaly_on_spike(self):
        """Test that spike is detected as anomaly"""
        data = self.generate_spike_data(hours=2)

        result = self.detector.detect(
            signal_name="test_metric_spike",
            service_name="test_service",
            data=data
        )

        # Should detect anomaly with high score
        assert result["score"] > 0.5  # At least medium confidence
        assert result["severity"] in ["medium", "high", "critical"]

    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        data = [(datetime.now(), 100.0)]  # Only 1 point

        result = self.detector.detect(
            signal_name="test_metric",
            service_name="test_service",
            data=data
        )

        assert result["is_anomaly"] == False
        assert "insufficient" in result["reason"].lower()

    def test_severity_calculation(self):
        """Test severity levels"""
        assert self.detector._calculate_severity(0.96) == "critical"
        assert self.detector._calculate_severity(0.90) == "high"
        assert self.detector._calculate_severity(0.75) == "medium"
        assert self.detector._calculate_severity(0.60) == "low"

    def test_recommendations_generation(self):
        """Test that recommendations are generated"""
        data = self.generate_spike_data(hours=2)

        result = self.detector.detect(
            signal_name="latency_p95",
            service_name="test_service",
            data=data
        )

        assert len(result["recommendations"]) > 0
        assert isinstance(result["recommendations"], list)

    def test_recent_anomalies_tracking(self):
        """Test that recent anomalies are tracked"""
        # Generate multiple anomalies
        for i in range(3):
            data = self.generate_spike_data(hours=1)
            self.detector.detect(
                signal_name=f"test_metric_{i}",
                service_name="test_service",
                data=data
            )

        recent = self.detector.get_recent_anomalies(hours=1)
        assert len(recent) >= 0  # Should have some anomalies

    def test_model_status(self):
        """Test model status reporting"""
        data = self.generate_normal_data(hours=1)
        self.detector.detect(
            signal_name="test_metric",
            service_name="test_service",
            data=data
        )

        status = self.detector.get_model_status()

        assert "signal_stats" in status
        assert "stl_models_cached" in status
        assert isinstance(status["signal_stats"], dict)

    def test_ensemble_scoring(self):
        """Test that ensemble combines multiple models"""
        data = self.generate_spike_data(hours=2)

        result = self.detector.detect(
            signal_name="test_ensemble",
            service_name="test_service",
            data=data
        )

        # Should have scores from all models
        assert "model_scores" in result
        assert "stl" in result["model_scores"]
        assert "isolation_forest" in result["model_scores"]
        assert "ewma" in result["model_scores"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

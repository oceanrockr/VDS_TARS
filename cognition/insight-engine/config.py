"""
Configuration for Cognitive Analytics Core
"""
import os
from typing import Optional


class CognitiveConfig:
    """Configuration settings for cognitive analytics"""

    # Database
    POSTGRES_URL: str = os.getenv(
        "POSTGRES_URL",
        "postgresql://tars:tars@postgres:5432/tars"
    )

    # Analysis parameters
    INSIGHT_REFRESH_INTERVAL: int = int(os.getenv("INSIGHT_REFRESH_INTERVAL", "60"))
    ANALYSIS_WINDOW_MINUTES: int = int(os.getenv("ANALYSIS_WINDOW_MINUTES", "60"))

    # Thresholds
    HIGH_VIOLATION_RATE_THRESHOLD: float = float(os.getenv("HIGH_VIOLATION_RATE_THRESHOLD", "0.3"))
    LOW_VIOLATION_RATE_THRESHOLD: float = float(os.getenv("LOW_VIOLATION_RATE_THRESHOLD", "0.05"))
    HIGH_CONSENSUS_LATENCY_THRESHOLD: float = float(os.getenv("HIGH_CONSENSUS_LATENCY_THRESHOLD", "400.0"))
    ETHICAL_FAIRNESS_THRESHOLD: float = float(os.getenv("ETHICAL_FAIRNESS_THRESHOLD", "0.75"))

    # Confidence parameters
    MIN_SAMPLE_SIZE_POLICY: int = int(os.getenv("MIN_SAMPLE_SIZE_POLICY", "50"))
    MIN_SAMPLE_SIZE_ETHICAL: int = int(os.getenv("MIN_SAMPLE_SIZE_ETHICAL", "30"))
    MIN_SAMPLE_SIZE_CONSENSUS: int = int(os.getenv("MIN_SAMPLE_SIZE_CONSENSUS", "20"))

    # Server
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8090"))

    # Feature flags
    COGNITION_ENABLED: bool = os.getenv("COGNITION_ENABLED", "true").lower() == "true"
    ADAPTIVE_POLICY_LEARNING: bool = os.getenv("ADAPTIVE_POLICY_LEARNING", "true").lower() == "true"
    META_CONSENSUS_OPTIMIZER: bool = os.getenv("META_CONSENSUS_OPTIMIZER", "true").lower() == "true"
    ETHICAL_TRAINER_ENABLED: bool = os.getenv("ETHICAL_TRAINER_ENABLED", "true").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate configuration"""
        assert 10 <= cls.INSIGHT_REFRESH_INTERVAL <= 600, "Refresh interval must be 10-600 seconds"
        assert 15 <= cls.ANALYSIS_WINDOW_MINUTES <= 1440, "Analysis window must be 15-1440 minutes"
        assert 0.0 <= cls.HIGH_VIOLATION_RATE_THRESHOLD <= 1.0, "Violation rate must be 0-1"
        assert 0.0 <= cls.ETHICAL_FAIRNESS_THRESHOLD <= 1.0, "Fairness threshold must be 0-1"

        return True


# Validate on import
CognitiveConfig.validate()

"""
validation module - Post-Release Validation Suite (PRVS) for T.A.R.S.

This module provides comprehensive post-release validation capabilities
for detecting regressions and ensuring release quality.
"""

from validation.post_release_validation import (
    # Main orchestrator
    ValidationOrchestrator,

    # Subsystems
    SBOMDeltaAnalyzer,
    SLSADeltaAnalyzer,
    APISurfaceComparator,
    PerformanceDriftAnalyzer,
    SecurityRegressionScanner,
    BehavioralRegressionChecker,

    # Data classes
    ValidationReport,
    SBOMDeltaResult,
    SLSADeltaResult,
    APICompatibilityResult,
    PerformanceDriftResult,
    SecurityRegressionResult,
    BehavioralRegressionResult,
    ComponentDelta,
    SLSADelta,
    APIChange,
    PerformanceMetric,
    SecurityFinding,
    BehavioralTest,

    # Enums
    ChangeType,
    Severity,
    APIChangeType,

    # Exceptions
    ValidationError,
    BehavioralRegressionError,
    SBOMDeltaError,
    SLSADeltaError,
    APICompatibilityError,
    PerformanceDriftError,
    SecurityRegressionError,
    BaselineMissingError,
    ValidationOrchestrationError,
    PolicyGateError
)

__all__ = [
    'ValidationOrchestrator',
    'SBOMDeltaAnalyzer',
    'SLSADeltaAnalyzer',
    'APISurfaceComparator',
    'PerformanceDriftAnalyzer',
    'SecurityRegressionScanner',
    'BehavioralRegressionChecker',
    'ValidationReport',
    'SBOMDeltaResult',
    'SLSADeltaResult',
    'APICompatibilityResult',
    'PerformanceDriftResult',
    'SecurityRegressionResult',
    'BehavioralRegressionResult',
    'ComponentDelta',
    'SLSADelta',
    'APIChange',
    'PerformanceMetric',
    'SecurityFinding',
    'BehavioralTest',
    'ChangeType',
    'Severity',
    'APIChangeType',
    'ValidationError',
    'BehavioralRegressionError',
    'SBOMDeltaError',
    'SLSADeltaError',
    'APICompatibilityError',
    'PerformanceDriftError',
    'SecurityRegressionError',
    'BaselineMissingError',
    'ValidationOrchestrationError',
    'PolicyGateError'
]

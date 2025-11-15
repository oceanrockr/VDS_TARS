"""
Stream Processor for Cognitive Analytics
Consumes logs from policy_audit, anomaly_logs, and consensus_metrics
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import asyncpg
from models import (
    PolicyMetrics, ConsensusMetrics, AnomalyCorrelation,
    EthicalMetrics, AdaptiveInsight, InsightType, InsightPriority
)

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Process data streams and generate cognitive insights"""

    def __init__(
        self,
        db_url: str,
        analysis_window_minutes: int = 60,
        insight_refresh_interval: int = 60
    ):
        self.db_url = db_url
        self.analysis_window = timedelta(minutes=analysis_window_minutes)
        self.refresh_interval = insight_refresh_interval
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        logger.info("StreamProcessor initialized with DB pool")

    async def shutdown(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()

    async def fetch_policy_metrics(self) -> List[PolicyMetrics]:
        """Fetch aggregated policy metrics from audit trail"""
        window_start = datetime.utcnow() - self.analysis_window

        query = """
        SELECT
            policy_id,
            policy_type,
            COUNT(*) as total_evaluations,
            SUM(CASE WHEN decision = 'allow' THEN 1 ELSE 0 END) as allow_count,
            SUM(CASE WHEN decision = 'deny' THEN 1 ELSE 0 END) as deny_count,
            SUM(CASE WHEN decision = 'warn' THEN 1 ELSE 0 END) as warn_count,
            AVG(EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp))) * 1000) as avg_latency_ms,
            (SUM(CASE WHEN decision = 'deny' THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0)) as violation_rate,
            MIN(timestamp) as window_start,
            MAX(timestamp) as window_end
        FROM policy_audit
        WHERE timestamp >= $1
        GROUP BY policy_id, policy_type
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, window_start)

        metrics = []
        for row in rows:
            metrics.append(PolicyMetrics(
                policy_id=row['policy_id'],
                policy_type=row['policy_type'],
                total_evaluations=row['total_evaluations'],
                allow_count=row['allow_count'],
                deny_count=row['deny_count'],
                warn_count=row['warn_count'],
                avg_evaluation_latency_ms=float(row['avg_latency_ms'] or 0),
                violation_rate=float(row['violation_rate'] or 0),
                window_start=row['window_start'],
                window_end=row['window_end']
            ))

        logger.info(f"Fetched {len(metrics)} policy metrics")
        return metrics

    async def fetch_consensus_metrics(self) -> List[ConsensusMetrics]:
        """Fetch consensus performance metrics"""
        # Simulated query - in production, this would query consensus_metrics table
        window_start = datetime.utcnow() - self.analysis_window

        # Mock data for demonstration
        metrics = [
            ConsensusMetrics(
                algorithm="raft",
                total_votes=150,
                avg_latency_ms=320.5,
                p95_latency_ms=485.2,
                p99_latency_ms=620.1,
                success_rate=0.973,
                quorum_failures=4,
                window_start=window_start,
                window_end=datetime.utcnow()
            )
        ]

        logger.info(f"Fetched {len(metrics)} consensus metrics")
        return metrics

    async def fetch_ethical_metrics(self) -> List[EthicalMetrics]:
        """Fetch ethical policy performance metrics"""
        window_start = datetime.utcnow() - self.analysis_window

        query = """
        SELECT
            policy_id,
            COUNT(*) as total_decisions,
            AVG((metadata->>'fairness_score')::float) as fairness_score
        FROM policy_audit
        WHERE policy_type = 'ethical'
          AND timestamp >= $1
          AND metadata->>'fairness_score' IS NOT NULL
        GROUP BY policy_id
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, window_start)

        metrics = []
        for row in rows:
            # Simplified ethical metrics
            metrics.append(EthicalMetrics(
                policy_id=row['policy_id'],
                fairness_score=float(row['fairness_score'] or 0.75),
                demographic_balance={},
                bias_detected=False,
                total_decisions=row['total_decisions'],
                window_start=window_start,
                window_end=datetime.utcnow()
            ))

        logger.info(f"Fetched {len(metrics)} ethical metrics")
        return metrics

    async def analyze_policy_optimization(
        self,
        policy_metrics: List[PolicyMetrics]
    ) -> List[AdaptiveInsight]:
        """Generate policy optimization insights"""
        insights = []

        for metric in policy_metrics:
            # High violation rate insight
            if metric.violation_rate > 0.3 and metric.total_evaluations > 50:
                insights.append(AdaptiveInsight(
                    id=f"insight-policy-{metric.policy_id}-{int(datetime.utcnow().timestamp())}",
                    type=InsightType.POLICY_OPTIMIZATION,
                    priority=InsightPriority.HIGH,
                    title=f"High violation rate in {metric.policy_id}",
                    description=f"Policy {metric.policy_id} has {metric.violation_rate:.1%} violation rate. Consider relaxing constraints.",
                    recommendation={
                        "policy": metric.policy_id,
                        "action": "relax_constraints",
                        "current_violation_rate": metric.violation_rate,
                        "target_violation_rate": 0.15
                    },
                    confidence_score=0.85,
                    evidence={
                        "sample_size": metric.total_evaluations,
                        "violation_rate": metric.violation_rate,
                        "deny_count": metric.deny_count
                    },
                    impact_estimate={
                        "violation_reduction_percent": 50.0,
                        "false_positive_risk": 5.0
                    }
                ))

            # Low violation rate - tighten policy
            elif metric.violation_rate < 0.05 and metric.total_evaluations > 100:
                insights.append(AdaptiveInsight(
                    id=f"insight-policy-{metric.policy_id}-{int(datetime.utcnow().timestamp())}",
                    type=InsightType.POLICY_OPTIMIZATION,
                    priority=InsightPriority.MEDIUM,
                    title=f"Low violation rate in {metric.policy_id}",
                    description=f"Policy {metric.policy_id} has only {metric.violation_rate:.1%} violation rate. Consider tightening for better security.",
                    recommendation={
                        "policy": metric.policy_id,
                        "action": "tighten_constraints",
                        "current_violation_rate": metric.violation_rate,
                        "target_violation_rate": 0.10
                    },
                    confidence_score=0.78,
                    evidence={
                        "sample_size": metric.total_evaluations,
                        "violation_rate": metric.violation_rate,
                        "allow_count": metric.allow_count
                    },
                    impact_estimate={
                        "security_improvement_percent": 25.0,
                        "false_negative_reduction": 15.0
                    }
                ))

        logger.info(f"Generated {len(insights)} policy optimization insights")
        return insights

    async def analyze_consensus_tuning(
        self,
        consensus_metrics: List[ConsensusMetrics]
    ) -> List[AdaptiveInsight]:
        """Generate consensus tuning insights"""
        insights = []

        for metric in consensus_metrics:
            # High latency insight
            if metric.avg_latency_ms > 400:
                insights.append(AdaptiveInsight(
                    id=f"insight-consensus-{metric.algorithm}-{int(datetime.utcnow().timestamp())}",
                    type=InsightType.CONSENSUS_TUNING,
                    priority=InsightPriority.HIGH,
                    title=f"High consensus latency in {metric.algorithm}",
                    description=f"Average consensus latency {metric.avg_latency_ms:.1f}ms exceeds target. Recommend timeout tuning.",
                    recommendation={
                        "algorithm": metric.algorithm,
                        "action": "reduce_timeout",
                        "current_avg_latency_ms": metric.avg_latency_ms,
                        "target_latency_ms": 300,
                        "timeout_reduction_percent": 20
                    },
                    confidence_score=0.82,
                    evidence={
                        "sample_size": metric.total_votes,
                        "avg_latency_ms": metric.avg_latency_ms,
                        "p95_latency_ms": metric.p95_latency_ms,
                        "success_rate": metric.success_rate
                    },
                    impact_estimate={
                        "latency_reduction_percent": 20.0,
                        "quorum_failure_risk_increase": 3.5
                    }
                ))

            # High quorum failure rate
            quorum_failure_rate = metric.quorum_failures / metric.total_votes if metric.total_votes > 0 else 0
            if quorum_failure_rate > 0.05:
                insights.append(AdaptiveInsight(
                    id=f"insight-consensus-failures-{int(datetime.utcnow().timestamp())}",
                    type=InsightType.CONSENSUS_TUNING,
                    priority=InsightPriority.CRITICAL,
                    title=f"High quorum failure rate: {quorum_failure_rate:.1%}",
                    description=f"Consensus failing to reach quorum {quorum_failure_rate:.1%} of the time. Increase timeout or check network.",
                    recommendation={
                        "algorithm": metric.algorithm,
                        "action": "increase_timeout",
                        "current_failure_rate": quorum_failure_rate,
                        "target_failure_rate": 0.02
                    },
                    confidence_score=0.91,
                    evidence={
                        "quorum_failures": metric.quorum_failures,
                        "total_votes": metric.total_votes,
                        "failure_rate": quorum_failure_rate
                    },
                    impact_estimate={
                        "failure_reduction_percent": 60.0,
                        "latency_increase_ms": 50.0
                    }
                ))

        logger.info(f"Generated {len(insights)} consensus tuning insights")
        return insights

    async def analyze_ethical_thresholds(
        self,
        ethical_metrics: List[EthicalMetrics]
    ) -> List[AdaptiveInsight]:
        """Generate ethical threshold adjustment insights"""
        insights = []

        for metric in ethical_metrics:
            # Low fairness score
            if metric.fairness_score < 0.75 and metric.total_decisions > 30:
                insights.append(AdaptiveInsight(
                    id=f"insight-ethical-{metric.policy_id}-{int(datetime.utcnow().timestamp())}",
                    type=InsightType.ETHICAL_THRESHOLD,
                    priority=InsightPriority.CRITICAL,
                    title=f"Low fairness score in {metric.policy_id}",
                    description=f"Fairness score {metric.fairness_score:.2f} below threshold 0.75. Review demographic balance.",
                    recommendation={
                        "policy": metric.policy_id,
                        "action": "adjust_fairness_threshold",
                        "current_fairness_score": metric.fairness_score,
                        "suggested_threshold": 0.70,
                        "review_training_data": True
                    },
                    confidence_score=0.88,
                    evidence={
                        "fairness_score": metric.fairness_score,
                        "sample_size": metric.total_decisions,
                        "bias_detected": metric.bias_detected
                    },
                    impact_estimate={
                        "fairness_improvement_percent": 10.0,
                        "false_negative_increase": 5.0
                    }
                ))

        logger.info(f"Generated {len(insights)} ethical insights")
        return insights

    async def generate_insights(self) -> List[AdaptiveInsight]:
        """Main analysis pipeline - generate all insights"""
        logger.info("Starting cognitive insight generation")

        # Fetch all metrics in parallel
        policy_metrics, consensus_metrics, ethical_metrics = await asyncio.gather(
            self.fetch_policy_metrics(),
            self.fetch_consensus_metrics(),
            self.fetch_ethical_metrics()
        )

        # Analyze and generate insights in parallel
        policy_insights, consensus_insights, ethical_insights = await asyncio.gather(
            self.analyze_policy_optimization(policy_metrics),
            self.analyze_consensus_tuning(consensus_metrics),
            self.analyze_ethical_thresholds(ethical_metrics)
        )

        all_insights = policy_insights + consensus_insights + ethical_insights

        logger.info(f"Generated {len(all_insights)} total insights")
        return all_insights

    async def save_insights(self, insights: List[AdaptiveInsight]):
        """Persist insights to database"""
        query = """
        INSERT INTO cognitive_insights
        (id, type, priority, title, description, recommendation, confidence_score, evidence, impact_estimate, status, timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (id) DO UPDATE SET
            status = EXCLUDED.status,
            timestamp = EXCLUDED.timestamp
        """

        async with self.pool.acquire() as conn:
            for insight in insights:
                await conn.execute(
                    query,
                    insight.id,
                    insight.type.value,
                    insight.priority.value,
                    insight.title,
                    insight.description,
                    insight.recommendation,
                    insight.confidence_score,
                    insight.evidence,
                    insight.impact_estimate,
                    insight.status,
                    insight.timestamp
                )

        logger.info(f"Saved {len(insights)} insights to database")

    async def run_continuous(self):
        """Run continuous insight generation loop"""
        logger.info(f"Starting continuous insight generation (interval: {self.refresh_interval}s)")

        while True:
            try:
                insights = await self.generate_insights()
                await self.save_insights(insights)

                logger.info(f"Generated and saved {len(insights)} insights. Sleeping {self.refresh_interval}s...")
                await asyncio.sleep(self.refresh_interval)

            except Exception as e:
                logger.error(f"Error in continuous insight generation: {e}", exc_info=True)
                await asyncio.sleep(10)  # Brief retry delay

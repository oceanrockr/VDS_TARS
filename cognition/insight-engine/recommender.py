"""
Recommender service for cognitive insights
Serves recommendations via REST API
"""
import logging
from typing import List, Optional
from datetime import datetime
import asyncpg
from models import (
    AdaptiveInsight, InsightType, InsightPriority,
    InsightRecommendationRequest, InsightRecommendationResponse,
    CognitiveState
)

logger = logging.getLogger(__name__)


class CognitiveRecommender:
    """Serve cognitive insights via API queries"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        logger.info("CognitiveRecommender initialized")

    async def shutdown(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()

    async def get_recommendations(
        self,
        request: InsightRecommendationRequest
    ) -> InsightRecommendationResponse:
        """Fetch recommendations based on request criteria"""

        # Build dynamic query
        conditions = ["status = 'pending'"]
        params = []
        param_idx = 1

        if request.insight_types:
            conditions.append(f"type = ANY(${param_idx})")
            params.append([t.value for t in request.insight_types])
            param_idx += 1

        if request.min_confidence:
            conditions.append(f"confidence_score >= ${param_idx}")
            params.append(request.min_confidence)
            param_idx += 1

        if request.priority:
            conditions.append(f"priority = ${param_idx}")
            params.append(request.priority.value)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT
            id, type, priority, title, description, recommendation,
            confidence_score, evidence, impact_estimate, status, timestamp
        FROM cognitive_insights
        WHERE {where_clause}
        ORDER BY
            CASE priority
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                WHEN 'low' THEN 4
            END,
            confidence_score DESC,
            timestamp DESC
        LIMIT ${param_idx}
        """
        params.append(request.limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        insights = []
        for row in rows:
            insights.append(AdaptiveInsight(
                id=row['id'],
                type=InsightType(row['type']),
                priority=InsightPriority(row['priority']),
                title=row['title'],
                description=row['description'],
                recommendation=row['recommendation'],
                confidence_score=row['confidence_score'],
                evidence=row['evidence'],
                impact_estimate=row['impact_estimate'],
                status=row['status'],
                timestamp=row['timestamp']
            ))

        avg_confidence = (
            sum(i.confidence_score for i in insights) / len(insights)
            if insights else 0.0
        )

        logger.info(f"Retrieved {len(insights)} recommendations (avg confidence: {avg_confidence:.2f})")

        return InsightRecommendationResponse(
            insights=insights,
            total_count=len(insights),
            avg_confidence=avg_confidence
        )

    async def get_cognitive_state(self) -> CognitiveState:
        """Get overall cognitive system state"""

        query = """
        SELECT
            MAX(timestamp) as last_analysis,
            COUNT(*) as total_insights,
            SUM(CASE WHEN status = 'applied' THEN 1 ELSE 0 END) as applied,
            SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
            AVG(confidence_score) as avg_confidence
        FROM cognitive_insights
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query)

            # Get active insights
            active_query = """
            SELECT id FROM cognitive_insights
            WHERE status = 'pending'
            ORDER BY timestamp DESC
            LIMIT 20
            """
            active_rows = await conn.fetch(active_query)

        active_ids = [r['id'] for r in active_rows]

        applied = row['applied'] or 0
        rejected = row['rejected'] or 0
        total = row['total_insights'] or 1  # Avoid division by zero

        success_rate = applied / (applied + rejected) if (applied + rejected) > 0 else 0.0

        state = CognitiveState(
            last_analysis_timestamp=row['last_analysis'] or datetime.utcnow(),
            total_insights_generated=total,
            insights_applied=applied,
            insights_rejected=rejected,
            avg_confidence_score=float(row['avg_confidence'] or 0.0),
            policy_adaptation_success_rate=success_rate,
            consensus_improvement_percent=0.0,  # Calculated separately
            ethical_fairness_delta=0.0,  # Calculated separately
            active_insights=active_ids
        )

        logger.info(f"Cognitive state: {applied}/{total} applied, success rate {success_rate:.1%}")
        return state

    async def update_insight_status(self, insight_id: str, new_status: str) -> bool:
        """Update insight status (approved/rejected/applied)"""

        query = """
        UPDATE cognitive_insights
        SET status = $1
        WHERE id = $2
        RETURNING id
        """

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(query, new_status, insight_id)

        if result:
            logger.info(f"Updated insight {insight_id} to status {new_status}")
            return True
        else:
            logger.warning(f"Insight {insight_id} not found for status update")
            return False

    async def get_recommendation_score(self, insight_id: str) -> float:
        """Get recommendation score for a specific insight"""

        query = """
        SELECT confidence_score, impact_estimate
        FROM cognitive_insights
        WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, insight_id)

        if not row:
            return 0.0

        # Calculate composite score
        confidence = row['confidence_score']
        impact = row['impact_estimate']

        # Extract positive impact metrics
        positive_impacts = [
            v for k, v in impact.items()
            if 'improvement' in k or 'reduction' in k
        ]

        avg_impact = sum(positive_impacts) / len(positive_impacts) if positive_impacts else 0.0

        # Composite score: 70% confidence + 30% impact
        score = 0.7 * confidence + 0.3 * (avg_impact / 100.0)

        return round(score, 3)

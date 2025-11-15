"""
Adaptive Policy Learner
Consumes cognitive insights and generates validated policy changes
"""
import os
import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from rego_patchgen import RegoPatchGenerator, RegoPatch

# Configuration
INSIGHT_ENGINE_URL = os.getenv(
    "INSIGHT_ENGINE_URL",
    "http://cognition-insight-engine:8090"
)
FEDERATION_HUB_URL = os.getenv(
    "FEDERATION_HUB_URL",
    "http://federation-hub:8080"
)
OPA_URL = os.getenv("OPA_URL", "http://opa:8181")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.8"))
ADAPTATION_APPROVAL_THRESHOLD = float(os.getenv("ADAPTATION_APPROVAL_THRESHOLD", "0.8"))
DRY_RUN_ENABLED = os.getenv("DRY_RUN_ENABLED", "true").lower() == "true"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
POLICY_ADAPTATION_SUCCESS = Counter(
    'tars_policy_adaptation_success_total',
    'Successful policy adaptations',
    ['policy_id', 'parameter']
)

POLICY_ADAPTATION_FAILURES = Counter(
    'tars_policy_adaptation_failures_total',
    'Failed policy adaptation attempts',
    ['policy_id', 'reason']
)

POLICY_VALIDATION_TIME = Histogram(
    'tars_policy_validation_seconds',
    'Time to validate policy changes',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ACTIVE_POLICY_PROPOSALS = Gauge(
    'tars_policy_proposals_active',
    'Active policy change proposals'
)

POLICY_ROLLBACK_TOTAL = Counter(
    'tars_policy_rollback_total',
    'Total policy rollbacks',
    ['policy_id', 'reason']
)


# Pydantic models
class PolicyProposal(BaseModel):
    """Policy change proposal"""
    id: str
    insight_id: str
    policy_id: str
    patch: Dict[str, Any]
    validation_status: str = "pending"  # pending/validated/rejected
    vote_id: Optional[str] = None
    approval_status: str = "pending"  # pending/approved/rejected
    created_at: datetime = Field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None


class AdaptationResult(BaseModel):
    """Result of policy adaptation attempt"""
    success: bool
    insight_id: str
    policy_id: str
    action: str
    details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Global state
patch_generator = RegoPatchGenerator()
http_client: Optional[httpx.AsyncClient] = None
learner_task: Optional[asyncio.Task] = None
active_proposals: Dict[str, PolicyProposal] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    global http_client, learner_task

    logger.info("Starting Adaptive Policy Learner")

    http_client = httpx.AsyncClient(timeout=30.0)

    # Start background learner
    learner_task = asyncio.create_task(run_adaptive_learning_loop())

    logger.info("Adaptive Policy Learner ready")

    yield

    # Shutdown
    logger.info("Shutting down Adaptive Policy Learner")
    if learner_task:
        learner_task.cancel()
        try:
            await learner_task
        except asyncio.CancelledError:
            pass

    if http_client:
        await http_client.aclose()


app = FastAPI(
    title="T.A.R.S. Adaptive Policy Learner",
    description="Autonomous policy adaptation based on cognitive insights",
    version="0.8.0-alpha",
    lifespan=lifespan
)


async def fetch_insights() -> List[Dict[str, Any]]:
    """Fetch pending insights from Insight Engine"""

    try:
        response = await http_client.post(
            f"{INSIGHT_ENGINE_URL}/api/v1/insights/recommendations",
            json={
                "insight_types": ["policy_optimization", "ethical_threshold"],
                "min_confidence": MIN_CONFIDENCE_THRESHOLD,
                "priority": "high",
                "limit": 10
            }
        )
        response.raise_for_status()

        data = response.json()
        insights = data.get("insights", [])

        logger.info(f"Fetched {len(insights)} insights from engine")
        return insights

    except Exception as e:
        logger.error(f"Error fetching insights: {e}", exc_info=True)
        return []


async def validate_policy_with_opa(policy_code: str, test_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate policy using OPA dry-run"""

    with POLICY_VALIDATION_TIME.time():
        try:
            # Upload policy to OPA
            policy_id = f"dryrun_{int(datetime.utcnow().timestamp())}"
            response = await http_client.put(
                f"{OPA_URL}/v1/policies/{policy_id}",
                content=policy_code,
                headers={"Content-Type": "text/plain"}
            )
            response.raise_for_status()

            # Evaluate policy
            eval_response = await http_client.post(
                f"{OPA_URL}/v1/data/tars/operational/scaling/allow",
                json={"input": test_input}
            )
            eval_response.raise_for_status()

            result = eval_response.json()

            # Clean up
            await http_client.delete(f"{OPA_URL}/v1/policies/{policy_id}")

            return {
                "valid": True,
                "result": result,
                "errors": []
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"OPA validation failed: {e.response.text}")
            return {
                "valid": False,
                "errors": [e.response.text]
            }
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return {
                "valid": False,
                "errors": [str(e)]
            }


async def submit_policy_proposal(proposal: PolicyProposal) -> bool:
    """Submit validated policy proposal to Federation Hub for voting"""

    try:
        response = await http_client.post(
            f"{FEDERATION_HUB_URL}/api/v1/policies/submit",
            json={
                "policy_id": proposal.policy_id,
                "policy_bundle": proposal.patch,
                "metadata": {
                    "source": "adaptive-policy-learner",
                    "insight_id": proposal.insight_id,
                    "automated": True
                }
            }
        )
        response.raise_for_status()

        vote_data = response.json()
        proposal.vote_id = vote_data.get("vote_id")

        logger.info(f"Submitted policy proposal {proposal.id} for voting (vote_id: {proposal.vote_id})")
        return True

    except Exception as e:
        logger.error(f"Error submitting proposal: {e}", exc_info=True)
        POLICY_ADAPTATION_FAILURES.labels(
            policy_id=proposal.policy_id,
            reason="submission_failed"
        ).inc()
        return False


async def check_vote_status(vote_id: str) -> Optional[str]:
    """Check status of federation vote"""

    try:
        response = await http_client.get(
            f"{FEDERATION_HUB_URL}/api/v1/votes/{vote_id}"
        )
        response.raise_for_status()

        vote_data = response.json()
        return vote_data.get("status")  # pending/approved/rejected

    except Exception as e:
        logger.error(f"Error checking vote status: {e}", exc_info=True)
        return None


async def update_insight_status(insight_id: str, status: str):
    """Update insight status in Insight Engine"""

    try:
        response = await http_client.post(
            f"{INSIGHT_ENGINE_URL}/api/v1/insights/{insight_id}/status?status={status}"
        )
        response.raise_for_status()

        logger.info(f"Updated insight {insight_id} status to {status}")

    except Exception as e:
        logger.error(f"Error updating insight status: {e}", exc_info=True)


async def process_insight(insight: Dict[str, Any]) -> Optional[PolicyProposal]:
    """Process a single insight and generate policy proposal"""

    insight_id = insight.get("id")
    recommendation = insight.get("recommendation", {})
    policy_id = recommendation.get("policy")

    if not policy_id:
        logger.warning(f"No policy_id in insight {insight_id}")
        return None

    # Generate patch
    patch = patch_generator.generate_patch(policy_id, recommendation)
    if not patch:
        logger.warning(f"Failed to generate patch for insight {insight_id}")
        POLICY_ADAPTATION_FAILURES.labels(
            policy_id=policy_id,
            reason="patch_generation_failed"
        ).inc()
        return None

    # Validate syntax
    validation = patch_generator.validate_patch_syntax(patch.rego_snippet)
    if not validation["valid"]:
        logger.warning(f"Invalid Rego syntax: {validation['errors']}")
        POLICY_ADAPTATION_FAILURES.labels(
            policy_id=policy_id,
            reason="syntax_validation_failed"
        ).inc()
        return None

    # OPA dry-run validation
    if DRY_RUN_ENABLED:
        test_input = {
            "action": "scale_out",
            "target_replicas": 5
        }

        opa_validation = await validate_policy_with_opa(
            patch_generator.generate_scaling_policy_patch(
                cooldown_seconds=int(patch.new_value) if patch.parameter == "cooldown_seconds" else 60,
                max_replicas=int(patch.new_value) if patch.parameter == "max_replicas" else 10,
                min_replicas=2
            ),
            test_input
        )

        if not opa_validation["valid"]:
            logger.warning(f"OPA validation failed: {opa_validation['errors']}")
            POLICY_ADAPTATION_FAILURES.labels(
                policy_id=policy_id,
                reason="opa_validation_failed"
            ).inc()
            return None

    # Create proposal
    proposal = PolicyProposal(
        id=f"proposal-{insight_id}",
        insight_id=insight_id,
        policy_id=policy_id,
        patch={
            "parameter": patch.parameter,
            "old_value": patch.old_value,
            "new_value": patch.new_value,
            "rego_snippet": patch.rego_snippet
        },
        validation_status="validated"
    )

    logger.info(f"Created policy proposal: {proposal.id}")
    return proposal


async def run_adaptive_learning_loop():
    """Main learning loop - poll insights and adapt policies"""

    logger.info(f"Starting adaptive learning loop (interval: {POLL_INTERVAL}s)")

    while True:
        try:
            # Fetch insights
            insights = await fetch_insights()

            if not insights:
                logger.debug("No insights to process")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # Process each insight
            for insight in insights:
                proposal = await process_insight(insight)

                if proposal:
                    # Store proposal
                    active_proposals[proposal.id] = proposal
                    ACTIVE_POLICY_PROPOSALS.set(len(active_proposals))

                    # Submit for voting
                    success = await submit_policy_proposal(proposal)

                    if success:
                        POLICY_ADAPTATION_SUCCESS.labels(
                            policy_id=proposal.policy_id,
                            parameter=proposal.patch["parameter"]
                        ).inc()
                    else:
                        await update_insight_status(insight["id"], "rejected")

            # Check status of pending proposals
            await check_pending_proposals()

            logger.info(f"Processed {len(insights)} insights. Sleeping {POLL_INTERVAL}s...")
            await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            logger.error(f"Error in learning loop: {e}", exc_info=True)
            await asyncio.sleep(10)


async def check_pending_proposals():
    """Check status of pending proposals and update accordingly"""

    completed_proposals = []

    for proposal_id, proposal in active_proposals.items():
        if not proposal.vote_id:
            continue

        status = await check_vote_status(proposal.vote_id)

        if status == "approved":
            proposal.approval_status = "approved"
            proposal.applied_at = datetime.utcnow()

            # Update insight status
            await update_insight_status(proposal.insight_id, "applied")

            logger.info(f"Proposal {proposal_id} approved and applied")
            completed_proposals.append(proposal_id)

        elif status == "rejected":
            proposal.approval_status = "rejected"

            # Update insight status
            await update_insight_status(proposal.insight_id, "rejected")

            logger.info(f"Proposal {proposal_id} rejected")
            completed_proposals.append(proposal_id)

    # Remove completed proposals
    for proposal_id in completed_proposals:
        del active_proposals[proposal_id]

    ACTIVE_POLICY_PROPOSALS.set(len(active_proposals))


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "adaptive-policy-learner",
        "version": "0.8.0-alpha"
    }


@app.get("/api/v1/proposals")
async def get_proposals():
    """Get all active proposals"""
    return {
        "proposals": list(active_proposals.values()),
        "total": len(active_proposals)
    }


@app.get("/api/v1/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get specific proposal"""
    if proposal_id not in active_proposals:
        raise HTTPException(status_code=404, detail="Proposal not found")

    return active_proposals[proposal_id]


@app.post("/api/v1/adapt/trigger")
async def trigger_adaptation():
    """Manually trigger adaptation cycle"""
    try:
        insights = await fetch_insights()

        processed = 0
        for insight in insights:
            proposal = await process_insight(insight)
            if proposal:
                active_proposals[proposal.id] = proposal
                await submit_policy_proposal(proposal)
                processed += 1

        ACTIVE_POLICY_PROPOSALS.set(len(active_proposals))

        return {
            "triggered": True,
            "insights_fetched": len(insights),
            "proposals_created": processed
        }

    except Exception as e:
        logger.error(f"Error triggering adaptation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091, log_level="info")

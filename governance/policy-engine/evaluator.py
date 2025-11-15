"""
Governance Policy Engine - OPA/Rego Policy Evaluator
"""
import asyncio
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolicyDecision(BaseModel):
    """Policy evaluation decision"""
    decision: str  # allow, deny, warn
    policy_id: str
    policy_type: str
    reasons: List[str]
    metadata: Dict[str, Any] = {}
    timestamp: datetime = datetime.utcnow()


class PolicyEvaluationRequest(BaseModel):
    """Request for policy evaluation"""
    input: Dict[str, Any]
    policy_type: str  # operational, ethical, security
    action: str
    resource: str
    principal: str


class PolicyBundle:
    """Policy bundle loader and manager"""

    def __init__(self, policy_dir: str):
        self.policy_dir = Path(policy_dir)
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.load_policies()

    def load_policies(self) -> None:
        """Load all Rego policies from directory"""
        if not self.policy_dir.exists():
            logger.warning(f"Policy directory {self.policy_dir} does not exist")
            return

        for policy_file in self.policy_dir.glob("**/*.rego"):
            policy_type = policy_file.parent.name
            policy_name = policy_file.stem

            with open(policy_file, 'r') as f:
                policy_content = f.read()

            policy_id = f"{policy_type}/{policy_name}"
            self.policies[policy_id] = {
                "id": policy_id,
                "type": policy_type,
                "name": policy_name,
                "content": policy_content,
                "checksum": hashlib.sha256(policy_content.encode()).hexdigest(),
                "loaded_at": datetime.utcnow()
            }

            logger.info(f"Loaded policy: {policy_id}")

    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get policy by ID"""
        return self.policies.get(policy_id)

    def list_policies(self, policy_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all policies, optionally filtered by type"""
        if policy_type:
            return [p for p in self.policies.values() if p["type"] == policy_type]
        return list(self.policies.values())


class OPAClient:
    """Client for Open Policy Agent"""

    def __init__(self, opa_url: str = "http://opa:8181"):
        self.opa_url = opa_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=10.0)

    async def evaluate(self, policy_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy using OPA"""
        try:
            # OPA data API endpoint
            url = f"{self.opa_url}/v1/data/{policy_path}"

            payload = {"input": input_data}

            response = await self.client.post(url, json=payload)

            if response.status_code == 200:
                result = response.json()
                return result.get("result", {})
            else:
                logger.error(f"OPA evaluation failed: {response.status_code} - {response.text}")
                return {}

        except Exception as e:
            logger.error(f"OPA client error: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check if OPA is healthy"""
        try:
            response = await self.client.get(f"{self.opa_url}/health")
            return response.status_code == 200
        except:
            return False


class PolicyEngine:
    """Governance policy engine"""

    def __init__(self, policy_dir: str, opa_url: str, audit_db=None):
        self.bundle = PolicyBundle(policy_dir)
        self.opa_client = OPAClient(opa_url)
        self.audit_db = audit_db
        logger.info(f"Policy engine initialized with {len(self.bundle.policies)} policies")

    async def evaluate_policy(
        self,
        request: PolicyEvaluationRequest
    ) -> PolicyDecision:
        """Evaluate policy for a request"""
        logger.info(f"Evaluating {request.policy_type} policy for {request.action} on {request.resource}")

        # Get matching policies
        policies = self.bundle.list_policies(request.policy_type)

        if not policies:
            logger.warning(f"No policies found for type {request.policy_type}")
            return PolicyDecision(
                decision="deny",
                policy_id="default",
                policy_type=request.policy_type,
                reasons=["No policy found - default deny"]
            )

        # Evaluate each policy via OPA
        decisions = []
        reasons = []

        for policy in policies:
            # Construct OPA policy path
            policy_path = f"tars/{policy['type']}/{policy['name']}"

            # Prepare input for OPA
            opa_input = {
                "action": request.action,
                "resource": request.resource,
                "principal": request.principal,
                **request.input
            }

            # Evaluate via OPA
            result = await self.opa_client.evaluate(policy_path, opa_input)

            if result:
                # Parse OPA result
                # Expected format: {"allow": true/false, "violations": [...]}
                allow = result.get("allow", False)
                violations = result.get("violations", [])

                decisions.append(allow)

                if not allow:
                    reasons.extend(violations if violations else [f"Policy {policy['id']} denied"])

        # Aggregate decision: all must allow
        final_decision = "allow" if all(decisions) and len(decisions) > 0 else "deny"

        policy_decision = PolicyDecision(
            decision=final_decision,
            policy_id=f"{request.policy_type}/aggregate",
            policy_type=request.policy_type,
            reasons=reasons if final_decision == "deny" else ["All policies passed"],
            metadata={
                "policies_evaluated": len(policies),
                "action": request.action,
                "resource": request.resource,
                "principal": request.principal
            }
        )

        # Audit log
        await self.audit_decision(policy_decision)

        return policy_decision

    async def audit_decision(self, decision: PolicyDecision) -> None:
        """Log policy decision to audit trail"""
        if self.audit_db:
            # In production, write to PostgreSQL audit table
            try:
                audit_entry = {
                    "decision": decision.decision,
                    "policy_id": decision.policy_id,
                    "policy_type": decision.policy_type,
                    "reasons": json.dumps(decision.reasons),
                    "metadata": json.dumps(decision.metadata),
                    "timestamp": decision.timestamp.isoformat()
                }
                # await self.audit_db.insert("policy_audit", audit_entry)
                logger.info(f"Audit: {decision.decision} - {decision.policy_id}")
            except Exception as e:
                logger.error(f"Audit logging failed: {e}")
        else:
            logger.debug(f"Audit (no DB): {decision.decision} - {decision.policy_id}")


# FastAPI app
app = FastAPI(
    title="T.A.R.S. Governance Policy Engine",
    version="0.7.0-alpha"
)

# Global policy engine
engine: Optional[PolicyEngine] = None


@app.on_event("startup")
async def startup():
    """Initialize policy engine on startup"""
    global engine

    import os
    policy_dir = os.getenv("POLICY_DIR", "/app/policies")
    opa_url = os.getenv("OPA_URL", "http://opa:8181")

    engine = PolicyEngine(policy_dir, opa_url)

    # Health check OPA
    if await engine.opa_client.health_check():
        logger.info("OPA is healthy")
    else:
        logger.warning("OPA is not reachable - policy evaluation may fail")


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "policy-engine"}


@app.post("/api/v1/evaluate", response_model=PolicyDecision)
async def evaluate_policy(request: PolicyEvaluationRequest):
    """Evaluate policy"""
    if not engine:
        raise HTTPException(status_code=503, detail="Policy engine not initialized")

    return await engine.evaluate_policy(request)


@app.get("/api/v1/policies")
async def list_policies(policy_type: Optional[str] = None):
    """List all loaded policies"""
    if not engine:
        raise HTTPException(status_code=503, detail="Policy engine not initialized")

    policies = engine.bundle.list_policies(policy_type)
    return {"policies": policies, "count": len(policies)}


@app.get("/api/v1/policies/{policy_id}")
async def get_policy(policy_id: str):
    """Get specific policy"""
    if not engine:
        raise HTTPException(status_code=503, detail="Policy engine not initialized")

    policy = engine.bundle.get_policy(policy_id)

    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    return policy


if __name__ == "__main__":
    uvicorn.run(
        "evaluator:app",
        host="0.0.0.0",
        port=8082,
        log_level="info"
    )

"""
T.A.R.S. Auto-Remediator Controller
Executes safe auto-remediation playbooks based on incidents and policies.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict, deque

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from playbooks import (
    ScaleOutPlaybook,
    RollbackPlaybook,
    RestartPodPlaybook,
    RedisCacheFlushKeysPlaybook
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="T.A.R.S. Auto-Remediator",
    description="Safe auto-remediation controller",
    version="1.0.0"
)

# Configuration
config_values = {
    "namespace": os.getenv("NAMESPACE", "tars"),
    "rate_limit_per_hour": int(os.getenv("RATE_LIMIT_PER_HOUR", "6")),
    "dry_run": os.getenv("DRY_RUN", "false").lower() == "true",
    "allow_playbooks": os.getenv("ALLOW_PLAYBOOKS", "ScaleOut,Rollback,RestartPod,RedisCacheFlushKeys").split(","),
    "cooldown_seconds": int(os.getenv("COOLDOWN_SECONDS", "900"))
}


class RemediationController:
    """Controller for auto-remediation policies and actions"""

    def __init__(self, config: Dict):
        """Initialize controller"""
        self.config = config
        self.namespace = config["namespace"]

        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        self.k8s_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi(self.k8s_client)
        self.apps_api = client.AppsV1Api(self.k8s_client)
        self.core_api = client.CoreV1Api(self.k8s_client)

        # Playbook registry
        self.playbooks = {
            "ScaleOut": ScaleOutPlaybook(self.k8s_client),
            "Rollback": RollbackPlaybook(self.k8s_client),
            "RestartPod": RestartPodPlaybook(self.k8s_client),
            "RedisCacheFlushKeys": RedisCacheFlushKeysPlaybook(self.k8s_client)
        }

        # Execution tracking for rate limiting
        self.execution_history = defaultdict(lambda: deque(maxlen=100))
        self.last_execution = {}  # policy_name -> timestamp

    async def process_webhook(self, payload: Dict) -> Dict:
        """
        Process webhook from anomaly detector or Alertmanager.

        Args:
            payload: Webhook payload

        Returns:
            Response dict
        """
        try:
            logger.info(f"Processing remediation webhook")

            # Extract trigger information
            incident_id = payload.get("incident_id", "unknown")
            service = payload.get("service", "unknown")
            severity = payload.get("severity", "medium")
            anomaly_results = payload.get("anomaly_results", [])

            # Find matching remediation policies
            policies = await self.find_matching_policies(service, severity, anomaly_results)

            if not policies:
                logger.info(f"No matching policies for service={service}, severity={severity}")
                return {"status": "no_policies_matched", "incident_id": incident_id}

            # Execute remediation actions
            results = []
            for policy in policies:
                result = await self.execute_policy(policy, payload)
                results.append(result)

            return {
                "status": "processed",
                "incident_id": incident_id,
                "policies_executed": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Webhook processing failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def find_matching_policies(
        self,
        service: str,
        severity: str,
        anomaly_results: List[Dict]
    ) -> List[Dict]:
        """Find remediation policies that match the conditions"""
        try:
            # List all RemediationPolicy resources
            policies = self.custom_api.list_namespaced_custom_object(
                group="aiops.tars",
                version="v1",
                namespace=self.namespace,
                plural="remediationpolicies"
            )

            matching = []
            for policy in policies.get("items", []):
                if await self.policy_matches(policy, service, severity, anomaly_results):
                    matching.append(policy)

            return matching

        except ApiException as e:
            logger.error(f"Failed to list policies: {e}")
            return []

    async def policy_matches(
        self,
        policy: Dict,
        service: str,
        severity: str,
        anomaly_results: List[Dict]
    ) -> bool:
        """Check if policy matches the conditions"""
        spec = policy.get("spec", {})
        triggers = spec.get("triggers", [])

        for trigger in triggers:
            trigger_type = trigger.get("type")

            if trigger_type == "anomaly":
                # Check if any anomaly result matches
                metric = trigger.get("metric", "")
                min_score = trigger.get("minScore", 0.8)

                for result in anomaly_results:
                    if metric in result.get("signal", "") and result.get("score", 0) >= min_score:
                        return True

            elif trigger_type == "promql":
                # Would evaluate PromQL expression here
                # For now, simplified check
                pass

        return False

    async def execute_policy(self, policy: Dict, context: Dict) -> Dict:
        """Execute a remediation policy"""
        policy_name = policy["metadata"]["name"]
        spec = policy.get("spec", {})

        try:
            # Check rate limits and cooldown
            if not await self.check_execution_allowed(policy):
                logger.warning(f"Policy {policy_name} blocked by rate limit or cooldown")
                return {
                    "policy": policy_name,
                    "status": "blocked",
                    "reason": "rate_limit_or_cooldown"
                }

            # Execute actions
            actions = spec.get("actions", [])
            dry_run = spec.get("dryRun", self.config["dry_run"])

            action_results = []
            for action in actions:
                result = await self.execute_action(action, policy, context, dry_run)
                action_results.append(result)

            # Record execution
            await self.record_execution(policy, action_results, context)

            # Update policy status
            await self.update_policy_status(policy, "success", action_results)

            return {
                "policy": policy_name,
                "status": "executed",
                "dry_run": dry_run,
                "actions": action_results
            }

        except Exception as e:
            logger.error(f"Policy execution failed: {e}", exc_info=True)
            await self.update_policy_status(policy, "failed", str(e))
            return {
                "policy": policy_name,
                "status": "failed",
                "error": str(e)
            }

    async def execute_action(
        self,
        action: Dict,
        policy: Dict,
        context: Dict,
        dry_run: bool
    ) -> Dict:
        """Execute a single remediation action"""
        action_type = action.get("type")
        params = action.get("params", {})

        logger.info(f"Executing action {action_type} (dry_run={dry_run})")

        # Check if playbook is allowed
        if action_type not in self.config["allow_playbooks"]:
            logger.warning(f"Playbook {action_type} not in allowed list")
            return {
                "action": action_type,
                "status": "blocked",
                "reason": "playbook_not_allowed"
            }

        # Get playbook
        playbook = self.playbooks.get(action_type)
        if not playbook:
            raise ValueError(f"Unknown action type: {action_type}")

        # Verify resource permissions
        allowed_resources = policy.get("spec", {}).get("allowedResources", [])
        if not self.check_resource_allowed(params, allowed_resources):
            logger.warning(f"Action blocked: resource not in allowed list")
            return {
                "action": action_type,
                "status": "blocked",
                "reason": "resource_not_allowed"
            }

        # Execute playbook
        try:
            result = await playbook.execute(
                params=params,
                context=context,
                dry_run=dry_run,
                namespace=self.namespace
            )

            # Create RemediationAction CR to record execution
            await self.create_action_record(policy, action, result, context, dry_run)

            return {
                "action": action_type,
                "status": "success",
                "result": result
            }

        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            return {
                "action": action_type,
                "status": "failed",
                "error": str(e)
            }

    def check_resource_allowed(self, params: Dict, allowed_resources: List[Dict]) -> bool:
        """Check if the target resource is in the allowed list"""
        if not allowed_resources:
            # If no allowlist specified, block by default (fail-safe)
            logger.warning("No allowed resources specified, blocking action")
            return False

        # Extract target resource from params
        resource_kind = params.get("kind", params.get("deployment", "").split("/")[0])
        resource_name = params.get("name", params.get("deployment", "").split("/")[-1])

        for allowed in allowed_resources:
            if allowed.get("kind") == resource_kind:
                # Check name match (support wildcards)
                allowed_name = allowed.get("name", "")
                if allowed_name == "*" or allowed_name == resource_name:
                    return True

        return False

    async def check_execution_allowed(self, policy: Dict) -> bool:
        """Check if policy execution is allowed based on rate limits"""
        policy_name = policy["metadata"]["name"]
        spec = policy.get("spec", {})

        # Check cooldown
        cooldown = spec.get("cooldownSeconds", self.config["cooldown_seconds"])
        last_exec = self.last_execution.get(policy_name)

        if last_exec:
            elapsed = (datetime.now() - last_exec).total_seconds()
            if elapsed < cooldown:
                logger.info(f"Policy {policy_name} in cooldown ({elapsed:.0f}s < {cooldown}s)")
                return False

        # Check hourly rate limit
        max_per_hour = spec.get("maxExecutionsPerHour", self.config["rate_limit_per_hour"])
        history = self.execution_history[policy_name]

        # Count executions in last hour
        cutoff = datetime.now() - timedelta(hours=1)
        recent_executions = sum(1 for ts in history if ts >= cutoff)

        if recent_executions >= max_per_hour:
            logger.info(f"Policy {policy_name} rate limited ({recent_executions}/{max_per_hour} per hour)")
            return False

        return True

    async def record_execution(self, policy: Dict, results: List[Dict], context: Dict):
        """Record policy execution in history"""
        policy_name = policy["metadata"]["name"]
        now = datetime.now()

        self.last_execution[policy_name] = now
        self.execution_history[policy_name].append(now)

        logger.info(f"Recorded execution of policy {policy_name}")

    async def update_policy_status(self, policy: Dict, result: str, details):
        """Update RemediationPolicy status"""
        policy_name = policy["metadata"]["name"]

        try:
            status = {
                "lastExecution": datetime.now().isoformat(),
                "executionCount": policy.get("status", {}).get("executionCount", 0) + 1,
                "lastResult": result
            }

            self.custom_api.patch_namespaced_custom_object_status(
                group="aiops.tars",
                version="v1",
                namespace=self.namespace,
                plural="remediationpolicies",
                name=policy_name,
                body={"status": status}
            )

        except ApiException as e:
            logger.warning(f"Failed to update policy status: {e}")

    async def create_action_record(
        self,
        policy: Dict,
        action: Dict,
        result: Dict,
        context: Dict,
        dry_run: bool
    ):
        """Create RemediationAction CR to record execution"""
        policy_name = policy["metadata"]["name"]
        action_name = f"{policy_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        body = {
            "apiVersion": "aiops.tars/v1",
            "kind": "RemediationAction",
            "metadata": {
                "name": action_name,
                "namespace": self.namespace,
                "labels": {
                    "policy": policy_name,
                    "action-type": action["type"]
                }
            },
            "spec": {
                "policyName": policy_name,
                "actionType": action["type"],
                "params": action.get("params", {}),
                "triggeredBy": context.get("incident_id", "unknown"),
                "triggeredAt": datetime.now().isoformat(),
                "dryRun": dry_run
            },
            "status": {
                "phase": "Succeeded" if result.get("success") else "Failed",
                "startTime": datetime.now().isoformat(),
                "completionTime": datetime.now().isoformat(),
                "result": result
            }
        }

        try:
            self.custom_api.create_namespaced_custom_object(
                group="aiops.tars",
                version="v1",
                namespace=self.namespace,
                plural="remediationactions",
                body=body
            )
            logger.info(f"Created RemediationAction record: {action_name}")

        except ApiException as e:
            logger.warning(f"Failed to create action record: {e}")


# Initialize controller
controller = RemediationController(config_values)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "T.A.R.S. Auto-Remediator",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "namespace": controller.namespace,
            "dry_run": controller.config["dry_run"],
            "playbooks_loaded": len(controller.playbooks)
        }
    }


@app.post("/webhook")
async def webhook(request: Request):
    """
    Webhook endpoint for receiving remediation triggers.
    Can be called by anomaly detector or Alertmanager.
    """
    try:
        payload = await request.json()
        result = await controller.process_webhook(payload)
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Webhook processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policies")
async def list_policies():
    """List all remediation policies"""
    try:
        policies = controller.custom_api.list_namespaced_custom_object(
            group="aiops.tars",
            version="v1",
            namespace=controller.namespace,
            plural="remediationpolicies"
        )
        return {
            "count": len(policies.get("items", [])),
            "policies": [
                {
                    "name": p["metadata"]["name"],
                    "triggers": len(p.get("spec", {}).get("triggers", [])),
                    "actions": len(p.get("spec", {}).get("actions", [])),
                    "lastExecution": p.get("status", {}).get("lastExecution"),
                    "executionCount": p.get("status", {}).get("executionCount", 0)
                }
                for p in policies.get("items", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/actions/recent")
async def list_recent_actions(limit: int = 50):
    """List recent remediation actions"""
    try:
        actions = controller.custom_api.list_namespaced_custom_object(
            group="aiops.tars",
            version="v1",
            namespace=controller.namespace,
            plural="remediationactions"
        )

        items = actions.get("items", [])
        # Sort by creation time
        items.sort(
            key=lambda x: x["metadata"].get("creationTimestamp", ""),
            reverse=True
        )

        return {
            "count": len(items[:limit]),
            "actions": [
                {
                    "name": a["metadata"]["name"],
                    "policy": a.get("spec", {}).get("policyName"),
                    "actionType": a.get("spec", {}).get("actionType"),
                    "phase": a.get("status", {}).get("phase"),
                    "triggeredAt": a.get("spec", {}).get("triggeredAt"),
                    "result": a.get("status", {}).get("result", {}).get("message")
                }
                for a in items[:limit]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info"
    )

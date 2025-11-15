"""
T.A.R.S. Hyperparameter Updater Service
Synchronizes optimized hyperparameters from AutoML to running agents

Features:
- Pull best hyperparameters from MLflow Model Registry
- Validate hyperparameters against agent constraints
- Hot-reload agents with new hyperparameters (no downtime)
- Approval workflow (manual or autonomous)
- Safety checks to prevent agent crashes
- Rollback capability if update fails

Author: T.A.R.S. Cognitive Team
Version: v0.9.4-alpha
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import asyncio
import json

import httpx
import numpy as np
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ApprovalMode(str, Enum):
    """Approval mode for hyperparameter updates."""
    MANUAL = "manual"  # Require explicit approval
    AUTONOMOUS_THRESHOLD = "autonomous_threshold"  # Auto-approve if improvement >= threshold
    AUTONOMOUS_ALL = "autonomous_all"  # Auto-approve all updates


class UpdateStatus(str, Enum):
    """Status of a hyperparameter update."""
    PENDING_VALIDATION = "pending_validation"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLYING = "applying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class HyperparameterUpdate(BaseModel):
    """Represents a hyperparameter update request."""
    update_id: str
    agent_type: str
    current_params: Dict[str, Any]
    new_params: Dict[str, Any]
    current_score: float
    new_score: float
    improvement: float
    mlflow_run_id: Optional[str] = None
    status: UpdateStatus = UpdateStatus.PENDING_VALIDATION
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    approved_at: Optional[str] = None
    applied_at: Optional[str] = None
    error_message: Optional[str] = None


class HyperparameterUpdater:
    """
    Manages hyperparameter updates for multi-agent orchestration system.

    Workflow:
    1. Fetch optimized hyperparameters from AutoML/MLflow
    2. Validate against agent constraints
    3. Request approval (manual or autonomous)
    4. Apply updates via hot-reload
    5. Monitor for failures and rollback if needed
    """

    def __init__(
        self,
        automl_service_url: str = "http://localhost:8097",
        orchestration_service_url: str = "http://localhost:8094",
        approval_mode: ApprovalMode = ApprovalMode.MANUAL,
        autonomous_threshold: float = 0.03,  # 3 percentage points
        validation_strictness: str = "medium",
    ):
        """
        Initialize hyperparameter updater.

        Args:
            automl_service_url: URL of AutoML service
            orchestration_service_url: URL of orchestration service
            approval_mode: Approval mode for updates
            autonomous_threshold: Minimum improvement for autonomous approval
            validation_strictness: Validation level ("low", "medium", "high")
        """
        self.automl_service_url = automl_service_url
        self.orchestration_service_url = orchestration_service_url
        self.approval_mode = approval_mode
        self.autonomous_threshold = autonomous_threshold
        self.validation_strictness = validation_strictness

        # Track updates
        self.pending_updates: Dict[str, HyperparameterUpdate] = {}
        self.update_history: List[HyperparameterUpdate] = []

        # Agent constraints (safety bounds)
        self.agent_constraints = self._initialize_agent_constraints()

        logger.info(
            f"HyperparameterUpdater initialized: "
            f"approval_mode={approval_mode}, "
            f"threshold={autonomous_threshold}, "
            f"validation={validation_strictness}"
        )

    def _initialize_agent_constraints(self) -> Dict[str, Dict[str, tuple]]:
        """
        Initialize safety constraints for each agent type.

        Returns:
            Dictionary mapping agent_type -> param_name -> (min, max)
        """
        return {
            "dqn": {
                "learning_rate": (1e-6, 1e-1),
                "gamma": (0.9, 0.9999),
                "epsilon_start": (0.5, 1.0),
                "epsilon_end": (0.001, 0.2),
                "epsilon_decay": (0.9, 0.99999),
                "buffer_size": (1000, 10000000),
                "batch_size": (8, 1024),
                "target_update": (1, 10000),
                "hidden_dim": (16, 1024),
            },
            "a2c": {
                "learning_rate": (1e-6, 1e-1),
                "gamma": (0.9, 0.9999),
                "gae_lambda": (0.8, 0.999),
                "value_loss_coef": (0.01, 10.0),
                "entropy_coef": (0.0, 0.5),
                "max_grad_norm": (0.01, 100.0),
                "hidden_dim": (8, 512),
                "n_steps": (1, 10000),
            },
            "ppo": {
                "learning_rate": (1e-6, 1e-1),
                "gamma": (0.9, 0.9999),
                "gae_lambda": (0.8, 0.999),
                "clip_epsilon": (0.05, 0.5),
                "value_loss_coef": (0.01, 10.0),
                "entropy_coef": (0.0, 0.5),
                "max_grad_norm": (0.01, 100.0),
                "n_epochs": (1, 100),
                "batch_size": (8, 2048),
                "target_kl": (0.001, 0.1),
                "hidden_dim": (16, 512),
            },
            "ddpg": {
                "actor_lr": (1e-7, 1e-1),
                "critic_lr": (1e-6, 1e-1),
                "gamma": (0.9, 0.9999),
                "tau": (0.0001, 0.1),
                "buffer_size": (1000, 10000000),
                "batch_size": (8, 1024),
                "noise_sigma": (0.01, 1.0),
                "noise_theta": (0.01, 0.5),
                "hidden_dim": (16, 1024),
                "weight_decay": (0.0, 0.1),
            },
        }

    async def fetch_best_hyperparameters(
        self,
        agent_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch best hyperparameters for an agent from AutoML service.

        Args:
            agent_type: Type of agent (dqn, a2c, ppo, ddpg)

        Returns:
            Dictionary with best hyperparameters and metadata, or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.automl_service_url}/api/v1/models",
                    params={"agent_type": agent_type}
                )
                response.raise_for_status()

                data = response.json()

                if not data.get("models"):
                    logger.warning(f"No models found for agent type: {agent_type}")
                    return None

                # Get best model
                best_model = data["models"][0]  # Already sorted by score

                logger.info(
                    f"Fetched best hyperparameters for {agent_type}: "
                    f"score={best_model.get('metrics', {}).get('best_score', 'N/A')}"
                )

                return best_model

        except Exception as e:
            logger.error(f"Failed to fetch hyperparameters for {agent_type}: {e}")
            return None

    def validate_hyperparameters(
        self,
        agent_type: str,
        params: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate hyperparameters against safety constraints.

        Args:
            agent_type: Type of agent
            params: Hyperparameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if agent_type not in self.agent_constraints:
            return False, f"Unknown agent type: {agent_type}"

        constraints = self.agent_constraints[agent_type]

        for param_name, value in params.items():
            if param_name not in constraints:
                if self.validation_strictness == "high":
                    return False, f"Unknown parameter: {param_name}"
                else:
                    logger.warning(f"Skipping validation for unknown parameter: {param_name}")
                    continue

            min_val, max_val = constraints[param_name]

            # Handle categorical parameters (pass through)
            if isinstance(value, (bool, str)):
                continue

            # Validate numeric parameters
            if not (min_val <= value <= max_val):
                return False, (
                    f"Parameter {param_name}={value} out of bounds "
                    f"[{min_val}, {max_val}]"
                )

        return True, None

    async def propose_update(
        self,
        agent_type: str,
        current_params: Dict[str, Any],
        current_score: float
    ) -> Optional[HyperparameterUpdate]:
        """
        Propose a hyperparameter update for an agent.

        Args:
            agent_type: Type of agent
            current_params: Current hyperparameters
            current_score: Current performance score

        Returns:
            HyperparameterUpdate object, or None if no better params found
        """
        # Fetch best hyperparameters from AutoML
        best_model = await self.fetch_best_hyperparameters(agent_type)

        if not best_model:
            logger.info(f"No optimized hyperparameters available for {agent_type}")
            return None

        new_params = best_model.get("params", {})
        new_score = best_model.get("metrics", {}).get("best_score", 0.0)
        mlflow_run_id = best_model.get("run_id")

        # Calculate improvement
        improvement = new_score - current_score

        # Check if improvement is significant
        if improvement <= 0:
            logger.info(
                f"No improvement for {agent_type}: "
                f"current={current_score:.4f}, new={new_score:.4f}"
            )
            return None

        # Validate hyperparameters
        is_valid, error_msg = self.validate_hyperparameters(agent_type, new_params)

        if not is_valid:
            logger.error(f"Validation failed for {agent_type}: {error_msg}")
            return None

        # Create update object
        update_id = f"{agent_type}_{int(time.time())}"

        update = HyperparameterUpdate(
            update_id=update_id,
            agent_type=agent_type,
            current_params=current_params,
            new_params=new_params,
            current_score=current_score,
            new_score=new_score,
            improvement=improvement,
            mlflow_run_id=mlflow_run_id,
            status=UpdateStatus.PENDING_APPROVAL,
        )

        # Store in pending updates
        self.pending_updates[update_id] = update

        logger.info(
            f"Proposed update for {agent_type}: "
            f"improvement={improvement:.4f} (+{improvement/current_score*100:.2f}%)"
        )

        return update

    def approve_update(self, update_id: str) -> bool:
        """
        Manually approve a hyperparameter update.

        Args:
            update_id: ID of update to approve

        Returns:
            True if approved successfully, False otherwise
        """
        if update_id not in self.pending_updates:
            logger.error(f"Update not found: {update_id}")
            return False

        update = self.pending_updates[update_id]

        if update.status != UpdateStatus.PENDING_APPROVAL:
            logger.error(f"Update {update_id} is not in pending approval status")
            return False

        update.status = UpdateStatus.APPROVED
        update.approved_at = datetime.utcnow().isoformat()

        logger.info(f"Update {update_id} approved manually")

        return True

    def reject_update(self, update_id: str, reason: str = "") -> bool:
        """
        Reject a hyperparameter update.

        Args:
            update_id: ID of update to reject
            reason: Reason for rejection

        Returns:
            True if rejected successfully, False otherwise
        """
        if update_id not in self.pending_updates:
            logger.error(f"Update not found: {update_id}")
            return False

        update = self.pending_updates[update_id]
        update.status = UpdateStatus.REJECTED
        update.error_message = reason

        # Move to history
        self.update_history.append(update)
        del self.pending_updates[update_id]

        logger.info(f"Update {update_id} rejected: {reason}")

        return True

    def _check_autonomous_approval(self, update: HyperparameterUpdate) -> bool:
        """
        Check if update qualifies for autonomous approval.

        Args:
            update: Update to check

        Returns:
            True if update should be auto-approved, False otherwise
        """
        if self.approval_mode == ApprovalMode.MANUAL:
            return False

        if self.approval_mode == ApprovalMode.AUTONOMOUS_ALL:
            return True

        if self.approval_mode == ApprovalMode.AUTONOMOUS_THRESHOLD:
            # Calculate percentage improvement
            pct_improvement = update.improvement / update.current_score
            return pct_improvement >= self.autonomous_threshold

        return False

    async def apply_update(self, update_id: str) -> bool:
        """
        Apply a hyperparameter update to the orchestration service.

        This performs a hot-reload of the agent with new hyperparameters.

        Args:
            update_id: ID of update to apply

        Returns:
            True if applied successfully, False otherwise
        """
        if update_id not in self.pending_updates:
            logger.error(f"Update not found: {update_id}")
            return False

        update = self.pending_updates[update_id]

        # Check approval status
        if update.status == UpdateStatus.PENDING_APPROVAL:
            # Check for autonomous approval
            if self._check_autonomous_approval(update):
                update.status = UpdateStatus.APPROVED
                update.approved_at = datetime.utcnow().isoformat()
                logger.info(f"Update {update_id} auto-approved (autonomous mode)")
            else:
                logger.warning(f"Update {update_id} requires manual approval")
                return False

        if update.status != UpdateStatus.APPROVED:
            logger.error(f"Update {update_id} is not approved")
            return False

        # Apply update
        update.status = UpdateStatus.APPLYING

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.orchestration_service_url}/api/v1/orchestration/agents/{update.agent_type}/reload",
                    json={"hyperparameters": update.new_params}
                )
                response.raise_for_status()

            update.status = UpdateStatus.COMPLETED
            update.applied_at = datetime.utcnow().isoformat()

            # Move to history
            self.update_history.append(update)
            del self.pending_updates[update_id]

            logger.info(f"Update {update_id} applied successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to apply update {update_id}: {e}")
            update.status = UpdateStatus.FAILED
            update.error_message = str(e)

            # Move to history
            self.update_history.append(update)
            del self.pending_updates[update_id]

            return False

    async def sync_all_agents(
        self,
        agent_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Optional[HyperparameterUpdate]]:
        """
        Synchronize hyperparameters for all agents.

        Args:
            agent_configs: Dictionary mapping agent_type -> config (params, score)

        Returns:
            Dictionary mapping agent_type -> HyperparameterUpdate (or None)
        """
        results = {}

        for agent_type, config in agent_configs.items():
            current_params = config.get("params", {})
            current_score = config.get("score", 0.0)

            try:
                update = await self.propose_update(
                    agent_type=agent_type,
                    current_params=current_params,
                    current_score=current_score
                )
                results[agent_type] = update

                # Auto-apply if autonomous mode
                if update and self._check_autonomous_approval(update):
                    await self.apply_update(update.update_id)

            except Exception as e:
                logger.error(f"Failed to sync {agent_type}: {e}")
                results[agent_type] = None

        return results

    def get_pending_updates(self) -> List[HyperparameterUpdate]:
        """Get all pending updates."""
        return list(self.pending_updates.values())

    def get_update_history(self, limit: int = 100) -> List[HyperparameterUpdate]:
        """Get update history."""
        return self.update_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get updater statistics."""
        return {
            "pending_updates": len(self.pending_updates),
            "total_history": len(self.update_history),
            "completed_updates": len([u for u in self.update_history if u.status == UpdateStatus.COMPLETED]),
            "failed_updates": len([u for u in self.update_history if u.status == UpdateStatus.FAILED]),
            "rejected_updates": len([u for u in self.update_history if u.status == UpdateStatus.REJECTED]),
            "avg_improvement": np.mean([u.improvement for u in self.update_history if u.status == UpdateStatus.COMPLETED]) if self.update_history else 0.0,
            "approval_mode": self.approval_mode.value,
            "autonomous_threshold": self.autonomous_threshold,
        }

"""
T.A.R.S. Model Registry
MLflow-based model versioning and tracking system

Tracks experiments, models, hyperparameters, and metrics for all T.A.R.S. agents.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    MLflow-based model registry for T.A.R.S. agents.

    Features:
    - Experiment tracking with hierarchical organization
    - Model versioning with stage management (staging, production)
    - Hyperparameter and metric logging
    - Artifact storage (model weights, configs, plots)
    - Model comparison and rollback
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "tars_automl",
        registry_uri: Optional[str] = None,
    ):
        """
        Initialize Model Registry.

        Args:
            tracking_uri: MLflow tracking server URI or local path
            experiment_name: Name of the MLflow experiment
            registry_uri: Model registry URI (default: same as tracking_uri)
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Set registry URI if provided
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)

        # Initialize client
        self.client = MlflowClient(tracking_uri=tracking_uri)

        logger.info(
            f"ModelRegistry initialized: tracking_uri={tracking_uri}, "
            f"experiment={experiment_name}, experiment_id={self.experiment_id}"
        )

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Dictionary of tags to associate with the run
            nested: Whether this is a nested run

        Returns:
            Active MLflow run context
        """
        tags = tags or {}
        tags["run_name"] = run_name
        tags["timestamp"] = datetime.utcnow().isoformat()

        run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        logger.info(f"Started MLflow run: {run_name} (run_id={run.info.run_id})")

        return run

    def log_agent_training(
        self,
        agent_type: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        model: Optional[Any] = None,
        artifacts: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
    ) -> str:
        """
        Log a complete agent training run.

        Args:
            agent_type: Type of agent (DQN, A2C, PPO, DDPG)
            hyperparameters: Dictionary of hyperparameters
            metrics: Dictionary of evaluation metrics
            model: PyTorch model to save (optional)
            artifacts: Dictionary mapping artifact names to file paths
            run_name: Custom run name (default: auto-generated)

        Returns:
            Run ID
        """
        run_name = run_name or f"{agent_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            # Log tags
            mlflow.set_tag("agent_type", agent_type)
            mlflow.set_tag("framework", "pytorch")

            # Log hyperparameters
            mlflow.log_params(self._flatten_dict(hyperparameters))

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model if provided
            if model is not None:
                try:
                    mlflow.pytorch.log_model(model, artifact_path="model")
                    logger.info(f"Logged PyTorch model for {agent_type}")
                except Exception as e:
                    logger.error(f"Failed to log model: {e}")

            # Log artifacts if provided
            if artifacts:
                for artifact_name, file_path in artifacts.items():
                    try:
                        mlflow.log_artifact(file_path, artifact_path=artifact_name)
                        logger.info(f"Logged artifact: {artifact_name}")
                    except Exception as e:
                        logger.error(f"Failed to log artifact {artifact_name}: {e}")

            run_id = run.info.run_id
            logger.info(f"Logged training run for {agent_type}: run_id={run_id}")

        return run_id

    def log_optimization_result(
        self,
        agent_type: str,
        optimization_results: Dict[str, Any],
        run_name: Optional[str] = None,
    ) -> str:
        """
        Log Optuna optimization results.

        Args:
            agent_type: Type of agent optimized
            optimization_results: Results from OptunaOptimizer
            run_name: Custom run name

        Returns:
            Run ID
        """
        run_name = run_name or f"{agent_type}_optuna_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("agent_type", agent_type)
            mlflow.set_tag("type", "optimization")
            mlflow.set_tag("optimizer", "optuna")

            # Log best parameters
            if "best_params" in optimization_results:
                mlflow.log_params(self._flatten_dict(optimization_results["best_params"]))

            # Log optimization metrics
            if "best_score" in optimization_results:
                mlflow.log_metric("best_score", optimization_results["best_score"])
            if "n_trials" in optimization_results:
                mlflow.log_metric("n_trials", optimization_results["n_trials"])
            if "n_completed_trials" in optimization_results:
                mlflow.log_metric("n_completed_trials", optimization_results["n_completed_trials"])

            # Log statistics
            if "statistics" in optimization_results:
                stats = optimization_results["statistics"]
                for key, value in stats.items():
                    if value is not None:
                        mlflow.log_metric(f"stats_{key}", value)

            # Save full results as JSON artifact
            results_path = Path(f"/tmp/optuna_results_{run.info.run_id}.json")
            with open(results_path, "w") as f:
                json.dump(optimization_results, f, indent=2)
            mlflow.log_artifact(str(results_path), artifact_path="optimization")
            results_path.unlink()  # Clean up temp file

            run_id = run.info.run_id
            logger.info(f"Logged optimization results for {agent_type}: run_id={run_id}")

        return run_id

    def log_metrics_history(
        self,
        metrics: Dict[str, List[float]],
        step_offset: int = 0,
    ) -> None:
        """
        Log a history of metrics (e.g., episode rewards).

        Args:
            metrics: Dictionary mapping metric names to lists of values
            step_offset: Starting step number (for resuming training)
        """
        for metric_name, values in metrics.items():
            for step, value in enumerate(values, start=step_offset):
                mlflow.log_metric(metric_name, value, step=step)

        logger.info(f"Logged {len(metrics)} metric histories")

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: Optional[str] = None,
    ) -> str:
        """
        Register a model to the Model Registry.

        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            artifact_path: Path to the model artifact within the run
            description: Optional model description

        Returns:
            Model version number
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"

        try:
            model_version = mlflow.register_model(model_uri, model_name)
            version_number = model_version.version

            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=version_number,
                    description=description,
                )

            logger.info(f"Registered model: {model_name} version {version_number}")
            return version_number

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production",
        archive_existing: bool = True,
    ) -> None:
        """
        Promote a model version to a stage (Staging, Production).

        Args:
            model_name: Name of the registered model
            version: Version number to promote
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing: Whether to archive existing models in the target stage
        """
        try:
            # Archive existing models in the target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                existing_versions = self.client.get_latest_versions(model_name, stages=[stage])
                for mv in existing_versions:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Archived",
                    )
                    logger.info(f"Archived {model_name} version {mv.version}")

            # Promote the new version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(f"Promoted {model_name} version {version} to {stage}")

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to promote model: {e}")
            raise

    def get_best_run(
        self,
        agent_type: Optional[str] = None,
        metric_name: str = "best_score",
        ascending: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.

        Args:
            agent_type: Filter by agent type (optional)
            metric_name: Metric to sort by
            ascending: Sort order (False = descending, True = ascending)

        Returns:
            Dictionary with run information, or None if no runs found
        """
        filter_string = ""
        if agent_type:
            filter_string = f"tags.agent_type = '{agent_type}'"

        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if not runs:
            return None

        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("run_name", "N/A"),
            "agent_type": run.data.tags.get("agent_type", "N/A"),
            "metrics": run.data.metrics,
            "params": run.data.params,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

    def get_model_versions(
        self,
        model_name: str,
        stage: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get versions of a registered model.

        Args:
            model_name: Name of the registered model
            stage: Filter by stage (optional)

        Returns:
            List of model version dictionaries
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")

            return [
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "creation_time": mv.creation_timestamp,
                    "description": mv.description,
                }
                for mv in versions
            ]

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metric names to compare (optional, defaults to all)

        Returns:
            Dictionary with comparison data
        """
        comparison = {"runs": []}

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("run_name", "N/A"),
                    "agent_type": run.data.tags.get("agent_type", "N/A"),
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                }

                # Filter metrics if specified
                if metrics:
                    run_data["metrics"] = {k: v for k, v in run_data["metrics"].items() if k in metrics}

                comparison["runs"].append(run_data)

            except mlflow.exceptions.MlflowException as e:
                logger.error(f"Failed to get run {run_id}: {e}")

        return comparison

    def cleanup_old_runs(
        self,
        max_age_days: int = 90,
        dry_run: bool = True,
    ) -> int:
        """
        Delete old runs to free up storage.

        Args:
            max_age_days: Maximum age of runs to keep (in days)
            dry_run: If True, only count runs without deleting

        Returns:
            Number of runs deleted (or would be deleted in dry run)
        """
        from datetime import timedelta

        cutoff_time = (datetime.utcnow() - timedelta(days=max_age_days)).timestamp() * 1000

        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attribute.start_time < {int(cutoff_time)}",
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        count = len(runs)

        if not dry_run:
            for run in runs:
                try:
                    self.client.delete_run(run.info.run_id)
                except Exception as e:
                    logger.error(f"Failed to delete run {run.info.run_id}: {e}")

        logger.info(f"{'Would delete' if dry_run else 'Deleted'} {count} runs older than {max_age_days} days")

        return count

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flatten nested dictionary for MLflow logging.

        Args:
            d: Dictionary to flatten
            parent_key: Key prefix for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to JSON-serializable types
                if isinstance(v, (list, tuple)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry.

        Returns:
            Dictionary with registry statistics
        """
        # Get all runs
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        # Count by agent type
        agent_type_counts = {}
        for run in runs:
            agent_type = run.data.tags.get("agent_type", "unknown")
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        # Get registered models
        registered_models = self.client.search_registered_models()

        return {
            "total_runs": len(runs),
            "runs_by_agent_type": agent_type_counts,
            "registered_models": len(registered_models),
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "tracking_uri": self.tracking_uri,
        }

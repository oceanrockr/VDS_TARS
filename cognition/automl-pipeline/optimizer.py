"""
T.A.R.S. AutoML Optimizer
Optuna-based hyperparameter optimization with TPE and CMA-ES samplers

Supports DQN, A2C, PPO, DDPG agents and causal inference models.
"""
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import numpy as np

import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna with multiple sampling strategies.

    Features:
    - TPE (Tree-structured Parzen Estimator) for efficient search
    - CMA-ES for continuous optimization
    - Median and Hyperband pruners for early stopping
    - Multi-objective optimization support
    - Distributed optimization via storage backends
    """

    def __init__(
        self,
        study_name: str = "tars_automl",
        sampler_type: str = "tpe",
        pruner_type: str = "median",
        storage: Optional[str] = None,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            study_name: Name of the optimization study
            sampler_type: Sampling strategy ("tpe", "cmaes", "random")
            pruner_type: Pruning strategy ("median", "hyperband", "none")
            storage: Database URL for distributed optimization (e.g., "sqlite:///optuna.db")
            n_startup_trials: Number of random trials before TPE
            n_ei_candidates: Number of candidates for expected improvement
        """
        self.study_name = study_name
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type
        self.storage = storage
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates

        # Initialize sampler
        self.sampler = self._create_sampler()

        # Initialize pruner
        self.pruner = self._create_pruner()

        logger.info(f"OptunaOptimizer initialized: sampler={sampler_type}, pruner={pruner_type}")

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        if self.sampler_type == "tpe":
            return TPESampler(
                n_startup_trials=self.n_startup_trials,
                n_ei_candidates=self.n_ei_candidates,
                multivariate=True,
                group=True,
            )
        elif self.sampler_type == "cmaes":
            return CmaEsSampler(
                n_startup_trials=self.n_startup_trials,
            )
        elif self.sampler_type == "random":
            return optuna.samplers.RandomSampler()
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner based on configuration."""
        if self.pruner_type == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5,
            )
        elif self.pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3,
            )
        elif self.pruner_type == "none":
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner type: {self.pruner_type}")

    def optimize_dqn(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize DQN agent hyperparameters.

        Search space:
        - learning_rate: [1e-5, 1e-2] (log scale)
        - gamma: [0.95, 0.999]
        - epsilon_start: [0.9, 1.0]
        - epsilon_end: [0.01, 0.1]
        - epsilon_decay: [0.99, 0.9999]
        - buffer_size: [10000, 1000000] (log scale)
        - batch_size: [16, 256] (power of 2)
        - target_update: [10, 1000]
        - hidden_dim: [32, 512] (power of 2)

        Args:
            objective_fn: Function that takes hyperparameters and returns reward
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds

        Returns:
            Dictionary with best parameters, best score, and optimization stats
        """
        logger.info(f"Starting DQN optimization: n_trials={n_trials}, timeout={timeout}")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "epsilon_start": trial.suggest_float("epsilon_start", 0.9, 1.0),
                "epsilon_end": trial.suggest_float("epsilon_end", 0.01, 0.1),
                "epsilon_decay": trial.suggest_float("epsilon_decay", 0.99, 0.9999),
                "buffer_size": trial.suggest_int("buffer_size", 10000, 1000000, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
                "target_update": trial.suggest_int("target_update", 10, 1000),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512]),
                "double_dqn": trial.suggest_categorical("double_dqn", [True, False]),
                "prioritized_replay": trial.suggest_categorical("prioritized_replay", [True, False]),
            }

            if params["prioritized_replay"]:
                params["priority_alpha"] = trial.suggest_float("priority_alpha", 0.4, 0.8)
                params["priority_beta_start"] = trial.suggest_float("priority_beta_start", 0.4, 0.6)

            return objective_fn(params)

        study = optuna.create_study(
            study_name=f"{self.study_name}_dqn",
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        return self._format_results(study, "DQN")

    def optimize_a2c(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize A2C agent hyperparameters.

        Search space:
        - learning_rate: [1e-5, 1e-2] (log scale)
        - gamma: [0.95, 0.999]
        - gae_lambda: [0.8, 0.99]
        - value_loss_coef: [0.1, 1.0]
        - entropy_coef: [0.0, 0.1]
        - max_grad_norm: [0.1, 10.0]
        - hidden_dim: [16, 256] (power of 2)
        - n_steps: [5, 2048]
        """
        logger.info(f"Starting A2C optimization: n_trials={n_trials}, timeout={timeout}")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
                "value_loss_coef": trial.suggest_float("value_loss_coef", 0.1, 1.0),
                "entropy_coef": trial.suggest_float("entropy_coef", 0.0, 0.1),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 10.0),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256]),
                "n_steps": trial.suggest_int("n_steps", 5, 2048),
                "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
            }
            return objective_fn(params)

        study = optuna.create_study(
            study_name=f"{self.study_name}_a2c",
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        return self._format_results(study, "A2C")

    def optimize_ppo(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize PPO agent hyperparameters.

        Search space:
        - learning_rate: [1e-5, 1e-2] (log scale)
        - gamma: [0.95, 0.999]
        - gae_lambda: [0.8, 0.99]
        - clip_epsilon: [0.1, 0.4]
        - value_loss_coef: [0.1, 1.0]
        - entropy_coef: [0.0, 0.1]
        - max_grad_norm: [0.1, 10.0]
        - n_epochs: [3, 30]
        - batch_size: [16, 512] (power of 2)
        - target_kl: [0.01, 0.05]
        - hidden_dim: [32, 256] (power of 2)
        """
        logger.info(f"Starting PPO optimization: n_trials={n_trials}, timeout={timeout}")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
                "clip_epsilon": trial.suggest_float("clip_epsilon", 0.1, 0.4),
                "value_loss_coef": trial.suggest_float("value_loss_coef", 0.1, 1.0),
                "entropy_coef": trial.suggest_float("entropy_coef", 0.0, 0.1),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 10.0),
                "n_epochs": trial.suggest_int("n_epochs", 3, 30),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512]),
                "target_kl": trial.suggest_float("target_kl", 0.01, 0.05),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
                "clip_value_loss": trial.suggest_categorical("clip_value_loss", [True, False]),
            }
            return objective_fn(params)

        study = optuna.create_study(
            study_name=f"{self.study_name}_ppo",
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        return self._format_results(study, "PPO")

    def optimize_ddpg(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize DDPG agent hyperparameters.

        Search space:
        - actor_lr: [1e-6, 1e-2] (log scale)
        - critic_lr: [1e-5, 1e-2] (log scale)
        - gamma: [0.95, 0.999]
        - tau: [0.001, 0.02]
        - buffer_size: [10000, 1000000] (log scale)
        - batch_size: [16, 256] (power of 2)
        - noise_type: ["ou", "gaussian", "adaptive_ou"]
        - noise_sigma: [0.05, 0.5]
        - noise_theta: [0.1, 0.3] (for OU)
        - hidden_dim: [64, 512] (power of 2)
        """
        logger.info(f"Starting DDPG optimization: n_trials={n_trials}, timeout={timeout}")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "actor_lr": trial.suggest_float("actor_lr", 1e-6, 1e-2, log=True),
                "critic_lr": trial.suggest_float("critic_lr", 1e-5, 1e-2, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "buffer_size": trial.suggest_int("buffer_size", 10000, 1000000, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
                "noise_type": trial.suggest_categorical("noise_type", ["ou", "gaussian", "adaptive_ou"]),
                "noise_sigma": trial.suggest_float("noise_sigma", 0.05, 0.5),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            }

            if params["noise_type"] in ["ou", "adaptive_ou"]:
                params["noise_theta"] = trial.suggest_float("noise_theta", 0.1, 0.3)

            return objective_fn(params)

        study = optuna.create_study(
            study_name=f"{self.study_name}_ddpg",
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        return self._format_results(study, "DDPG")

    def optimize_causal_engine(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize causal inference engine hyperparameters.

        Search space:
        - discovery_method: ["pc", "ges", "notears"]
        - alpha: [0.01, 0.2] (significance level)
        - max_iter: [100, 5000]
        - learning_rate: [1e-4, 1e-1] (for NOTEARS)
        - lambda_1: [0.0, 1.0] (L1 regularization)
        - lambda_2: [0.0, 1.0] (DAG regularization)
        """
        logger.info(f"Starting Causal Engine optimization: n_trials={n_trials}, timeout={timeout}")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "discovery_method": trial.suggest_categorical("discovery_method", ["pc", "ges", "notears"]),
                "alpha": trial.suggest_float("alpha", 0.01, 0.2),
                "max_iter": trial.suggest_int("max_iter", 100, 5000),
            }

            if params["discovery_method"] == "notears":
                params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
                params["lambda_1"] = trial.suggest_float("lambda_1", 0.0, 1.0)
                params["lambda_2"] = trial.suggest_float("lambda_2", 0.0, 1.0)

            return objective_fn(params)

        study = optuna.create_study(
            study_name=f"{self.study_name}_causal",
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        return self._format_results(study, "CausalEngine")

    def _format_results(self, study: optuna.Study, model_type: str) -> Dict[str, Any]:
        """Format optimization results into a standardized dictionary."""
        best_trial = study.best_trial

        # Calculate statistics
        all_values = [trial.value for trial in study.trials if trial.value is not None]

        result = {
            "model_type": model_type,
            "best_params": best_trial.params,
            "best_score": best_trial.value,
            "best_trial_number": best_trial.number,
            "n_trials": len(study.trials),
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "statistics": {
                "mean_score": float(np.mean(all_values)) if all_values else None,
                "std_score": float(np.std(all_values)) if all_values else None,
                "median_score": float(np.median(all_values)) if all_values else None,
                "min_score": float(np.min(all_values)) if all_values else None,
                "max_score": float(np.max(all_values)) if all_values else None,
            },
            "study_name": study.study_name,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Optimization complete for {model_type}: best_score={best_trial.value:.4f}")

        return result

    def get_optimization_history(self, study_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve optimization history for a study.

        Args:
            study_name: Name of the study to retrieve

        Returns:
            List of trial dictionaries with parameters and values
        """
        study = optuna.load_study(
            study_name=study_name,
            storage=self.storage,
        )

        history = []
        for trial in study.trials:
            history.append({
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                "duration": (trial.datetime_complete - trial.datetime_start).total_seconds()
                           if trial.datetime_complete and trial.datetime_start else None,
            })

        return history

"""
T.A.R.S. Feature Engineering
Automated feature generation using Featuretools with deep feature synthesis

Generates temporal, aggregation, and relational features for T.A.R.S. agents.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

import featuretools as ft
from featuretools import EntitySet
from featuretools.primitives import (
    Mean, Std, Min, Max, Count, Sum, Skew, Trend,
    TimeSincePrevious, Day, Hour, Weekend, Month
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Automated feature engineering using Featuretools.

    Features:
    - Deep Feature Synthesis (DFS) with configurable depth
    - Temporal aggregations (rolling windows, time-based)
    - Multi-table relationships for agent coordination
    - Domain-specific primitives for RL metrics
    - Feature importance tracking
    """

    def __init__(
        self,
        max_depth: int = 2,
        agg_primitives: Optional[List[str]] = None,
        trans_primitives: Optional[List[str]] = None,
        max_features: int = 100,
    ):
        """
        Initialize Feature Engineer.

        Args:
            max_depth: Maximum depth for deep feature synthesis
            agg_primitives: List of aggregation primitives (default: Mean, Std, Count, Sum)
            trans_primitives: List of transformation primitives (default: Day, Hour, Weekend)
            max_features: Maximum number of features to generate
        """
        self.max_depth = max_depth
        self.max_features = max_features

        # Default primitives if not specified
        self.agg_primitives = agg_primitives or ["mean", "std", "count", "sum", "min", "max"]
        self.trans_primitives = trans_primitives or ["day", "hour", "weekend", "month"]

        self.feature_history: List[Dict[str, Any]] = []

        logger.info(
            f"FeatureEngineer initialized: max_depth={max_depth}, "
            f"agg_primitives={self.agg_primitives}, trans_primitives={self.trans_primitives}"
        )

    def generate_agent_features(
        self,
        agent_states: pd.DataFrame,
        rewards: pd.DataFrame,
        actions: pd.DataFrame,
        episodes: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate features for agent performance analysis.

        Args:
            agent_states: DataFrame with columns [timestamp, agent_id, state_vector, episode_id]
            rewards: DataFrame with columns [timestamp, agent_id, reward, episode_id]
            actions: DataFrame with columns [timestamp, agent_id, action, episode_id]
            episodes: DataFrame with columns [episode_id, start_time, end_time, outcome]

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info("Generating agent features with Featuretools DFS")

        # Create EntitySet
        es = ft.EntitySet(id="agent_performance")

        # Add entities
        es = es.add_dataframe(
            dataframe_name="episodes",
            dataframe=episodes,
            index="episode_id",
            time_index="start_time" if "start_time" in episodes.columns else None,
        )

        es = es.add_dataframe(
            dataframe_name="agent_states",
            dataframe=agent_states,
            index="state_id" if "state_id" in agent_states.columns else None,
            make_index=True if "state_id" not in agent_states.columns else False,
            time_index="timestamp" if "timestamp" in agent_states.columns else None,
        )

        es = es.add_dataframe(
            dataframe_name="rewards",
            dataframe=rewards,
            index="reward_id" if "reward_id" in rewards.columns else None,
            make_index=True if "reward_id" not in rewards.columns else False,
            time_index="timestamp" if "timestamp" in rewards.columns else None,
        )

        es = es.add_dataframe(
            dataframe_name="actions",
            dataframe=actions,
            index="action_id" if "action_id" in actions.columns else None,
            make_index=True if "action_id" not in actions.columns else False,
            time_index="timestamp" if "timestamp" in actions.columns else None,
        )

        # Add relationships
        if "episode_id" in agent_states.columns:
            es = es.add_relationship("episodes", "episode_id", "agent_states", "episode_id")
        if "episode_id" in rewards.columns:
            es = es.add_relationship("episodes", "episode_id", "rewards", "episode_id")
        if "episode_id" in actions.columns:
            es = es.add_relationship("episodes", "episode_id", "actions", "episode_id")

        # Run Deep Feature Synthesis
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="episodes",
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            max_depth=self.max_depth,
            max_features=self.max_features,
            verbose=True,
        )

        feature_names = [str(f) for f in feature_defs]

        logger.info(f"Generated {len(feature_names)} features")

        # Store in history
        self.feature_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "n_features": len(feature_names),
            "feature_names": feature_names[:20],  # Store first 20
            "target": "episodes",
        })

        return feature_matrix, feature_names

    def generate_temporal_features(
        self,
        timeseries_data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        value_cols: Optional[List[str]] = None,
        window_sizes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generate temporal features with rolling windows.

        Args:
            timeseries_data: DataFrame with time-indexed data
            timestamp_col: Name of timestamp column
            value_cols: Columns to generate features for (default: all numeric)
            window_sizes: Rolling window sizes in timesteps (default: [5, 10, 20, 50])

        Returns:
            DataFrame with temporal features
        """
        logger.info("Generating temporal features with rolling windows")

        df = timeseries_data.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        df = df.sort_values(timestamp_col)

        # Default to all numeric columns
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if timestamp_col in value_cols:
                value_cols.remove(timestamp_col)

        # Default window sizes
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]

        generated_features = []

        for col in value_cols:
            for window in window_sizes:
                # Rolling mean
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                generated_features.append(f"{col}_rolling_mean_{window}")

                # Rolling std
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()
                generated_features.append(f"{col}_rolling_std_{window}")

                # Rolling min/max
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window, min_periods=1).min()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window, min_periods=1).max()
                generated_features.extend([f"{col}_rolling_min_{window}", f"{col}_rolling_max_{window}"])

            # Lag features
            for lag in [1, 2, 5, 10]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                generated_features.append(f"{col}_lag_{lag}")

            # Diff features
            df[f"{col}_diff_1"] = df[col].diff(1)
            df[f"{col}_diff_2"] = df[col].diff(2)
            generated_features.extend([f"{col}_diff_1", f"{col}_diff_2"])

        logger.info(f"Generated {len(generated_features)} temporal features")

        return df

    def generate_multiagent_features(
        self,
        agent_metrics: Dict[str, pd.DataFrame],
        coordination_events: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate features for multi-agent coordination analysis.

        Args:
            agent_metrics: Dictionary mapping agent_id to metrics DataFrame
                          Each DataFrame should have columns: [timestamp, reward, action, state_dim_0, ...]
            coordination_events: DataFrame with columns [timestamp, event_type, agents_involved]

        Returns:
            DataFrame with multi-agent coordination features
        """
        logger.info("Generating multi-agent coordination features")

        # Align all agent metrics on timestamp
        aligned_dfs = []
        for agent_id, metrics in agent_metrics.items():
            df = metrics.copy()
            df.columns = [f"{agent_id}_{col}" if col != "timestamp" else col for col in df.columns]
            aligned_dfs.append(df)

        # Merge on timestamp
        if len(aligned_dfs) > 0:
            merged = aligned_dfs[0]
            for df in aligned_dfs[1:]:
                merged = pd.merge(merged, df, on="timestamp", how="outer", suffixes=("", "_dup"))
            merged = merged.sort_values("timestamp")
        else:
            merged = pd.DataFrame()

        # Generate correlation features between agents
        feature_df = merged.copy()

        agent_ids = list(agent_metrics.keys())

        # Pairwise reward correlation (rolling)
        for i, agent1 in enumerate(agent_ids):
            for agent2 in agent_ids[i+1:]:
                reward_col1 = f"{agent1}_reward"
                reward_col2 = f"{agent2}_reward"

                if reward_col1 in feature_df.columns and reward_col2 in feature_df.columns:
                    # Rolling correlation
                    feature_df[f"reward_corr_{agent1}_{agent2}"] = (
                        feature_df[reward_col1]
                        .rolling(window=20, min_periods=5)
                        .corr(feature_df[reward_col2])
                    )

        # Coordination event features
        if not coordination_events.empty:
            # Count events in time windows
            feature_df = feature_df.sort_values("timestamp")
            feature_df["coord_events_last_10"] = 0
            feature_df["coord_events_last_50"] = 0

            # This is a simplified version - in production, use proper time-based windowing
            for idx in range(len(feature_df)):
                ts = feature_df.iloc[idx]["timestamp"]
                # Count recent events (simplified logic)
                recent_events = coordination_events[
                    coordination_events["timestamp"] <= ts
                ]
                if len(recent_events) > 0:
                    feature_df.loc[feature_df.index[idx], "coord_events_last_10"] = len(recent_events.tail(10))
                    feature_df.loc[feature_df.index[idx], "coord_events_last_50"] = len(recent_events.tail(50))

        logger.info(f"Generated multi-agent features: {feature_df.shape[1]} columns")

        return feature_df

    def generate_reward_features(
        self,
        rewards: pd.DataFrame,
        agent_col: str = "agent_id",
        reward_col: str = "reward",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Generate specialized features for reward analysis.

        Args:
            rewards: DataFrame with agent rewards over time
            agent_col: Name of agent identifier column
            reward_col: Name of reward column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with reward-specific features
        """
        logger.info("Generating reward-specific features")

        df = rewards.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        df = df.sort_values([agent_col, timestamp_col])

        # Per-agent reward statistics
        df["reward_cumsum"] = df.groupby(agent_col)[reward_col].cumsum()
        df["reward_cummean"] = df.groupby(agent_col)[reward_col].expanding().mean().reset_index(level=0, drop=True)
        df["reward_cumstd"] = df.groupby(agent_col)[reward_col].expanding().std().reset_index(level=0, drop=True)

        # Reward momentum
        df["reward_momentum_5"] = df.groupby(agent_col)[reward_col].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        df["reward_momentum_20"] = df.groupby(agent_col)[reward_col].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)

        # Reward volatility
        df["reward_volatility_10"] = df.groupby(agent_col)[reward_col].rolling(10, min_periods=1).std().reset_index(level=0, drop=True)

        # Reward improvement
        df["reward_diff"] = df.groupby(agent_col)[reward_col].diff()
        df["reward_improvement_rate"] = df["reward_diff"] / (df["reward_cummean"] + 1e-8)

        # Percentile rank within agent's history
        df["reward_percentile"] = df.groupby(agent_col)[reward_col].rank(pct=True)

        logger.info(f"Generated reward features: {df.shape[1] - rewards.shape[1]} new columns")

        return df

    def get_feature_importance(
        self,
        feature_matrix: pd.DataFrame,
        target: pd.Series,
        method: str = "mutual_info",
        top_k: int = 20,
    ) -> Dict[str, float]:
        """
        Calculate feature importance scores.

        Args:
            feature_matrix: DataFrame with features
            target: Target variable
            method: Method for importance calculation ("mutual_info", "correlation")
            top_k: Number of top features to return

        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.feature_selection import mutual_info_regression

        logger.info(f"Calculating feature importance using {method}")

        # Remove non-numeric columns
        numeric_features = feature_matrix.select_dtypes(include=[np.number])

        # Handle NaN values
        numeric_features = numeric_features.fillna(numeric_features.mean())
        target = target.fillna(target.mean())

        if method == "mutual_info":
            scores = mutual_info_regression(numeric_features, target, random_state=42)
            importance = dict(zip(numeric_features.columns, scores))

        elif method == "correlation":
            correlations = numeric_features.corrwith(target).abs()
            importance = correlations.to_dict()

        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort and return top_k
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]

        logger.info(f"Top features: {[f[0] for f in sorted_importance[:5]]}")

        return dict(sorted_importance)

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about feature generation history.

        Returns:
            Dictionary with feature generation statistics
        """
        if not self.feature_history:
            return {"total_runs": 0}

        return {
            "total_runs": len(self.feature_history),
            "total_features_generated": sum(h["n_features"] for h in self.feature_history),
            "average_features_per_run": np.mean([h["n_features"] for h in self.feature_history]),
            "last_run": self.feature_history[-1] if self.feature_history else None,
        }

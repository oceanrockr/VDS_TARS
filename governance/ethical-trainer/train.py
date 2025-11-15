"""
Federated Ethical Learning - Fairness Classifier Trainer
Trains fairness models from historical ethical policy audits
"""
import os
import logging
import pickle
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import asyncpg

logger = logging.getLogger(__name__)


class FairnessFeatureExtractor:
    """Extract features from ethical policy audit data"""

    def __init__(self):
        self.protected_groups = ["age", "gender", "race", "disability", "religion"]

    def extract_features(self, audit_record: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract feature vector from audit record"""

        try:
            metadata = audit_record.get("metadata", {})

            # Feature 1-5: Training data distribution for protected groups
            training_dist = metadata.get("training_data_distribution", {})
            dist_features = [
                training_dist.get(group, 0.0)
                for group in self.protected_groups
            ]

            # Feature 6: Overall fairness score
            fairness_score = metadata.get("fairness_score", 0.0)

            # Feature 7-11: Outcome distribution by group
            outcome_dist = metadata.get("outcome_by_group", {})
            outcome_features = [
                outcome_dist.get(group, 0.0)
                for group in self.protected_groups
            ]

            # Feature 12: Variance in outcomes (disparity measure)
            outcomes = list(outcome_dist.values()) if outcome_dist else [0.0]
            outcome_variance = np.var(outcomes) if len(outcomes) > 1 else 0.0

            # Feature 13: Min demographic representation
            min_representation = min(dist_features) if dist_features else 0.0

            # Feature 14: Max demographic representation
            max_representation = max(dist_features) if dist_features else 0.0

            # Feature 15: Sample size
            sample_size = metadata.get("sample_size", 0)
            log_sample_size = np.log1p(sample_size)  # Log transform

            # Combine all features
            features = np.array(
                dist_features +
                [fairness_score] +
                outcome_features +
                [outcome_variance, min_representation, max_representation, log_sample_size]
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        names = []

        # Training distribution features
        for group in self.protected_groups:
            names.append(f"training_dist_{group}")

        # Fairness score
        names.append("fairness_score")

        # Outcome distribution features
        for group in self.protected_groups:
            names.append(f"outcome_dist_{group}")

        # Aggregate features
        names.extend([
            "outcome_variance",
            "min_representation",
            "max_representation",
            "log_sample_size"
        ])

        return names


class EthicalFairnessTrainer:
    """Train fairness classifier from historical data"""

    def __init__(
        self,
        db_url: str,
        model_path: str = "/app/fairness_model.pkl",
        scaler_path: str = "/app/scaler.pkl"
    ):
        self.db_url = db_url
        self.model_path = model_path
        self.scaler_path = scaler_path

        self.feature_extractor = FairnessFeatureExtractor()
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)

                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

                logger.info("Loaded existing fairness model")
            else:
                logger.info("No existing model found")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def save_model(self):
        """Save trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            logger.info(f"Saved model to {self.model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    async def fetch_training_data(
        self,
        lookback_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch training data from ethical policy audits"""

        window_start = datetime.utcnow() - timedelta(days=lookback_days)

        query = """
        SELECT
            decision,
            policy_id,
            policy_type,
            reasons,
            metadata,
            timestamp
        FROM policy_audit
        WHERE policy_type = 'ethical'
          AND timestamp >= $1
        ORDER BY timestamp DESC
        """

        conn = await asyncpg.connect(self.db_url)

        try:
            rows = await conn.fetch(query, window_start)

            X = []  # Features
            y = []  # Labels (1 = allow, 0 = deny)

            for row in rows:
                features = self.feature_extractor.extract_features(dict(row))

                if features is not None:
                    X.append(features)
                    y.append(1 if row['decision'] == 'allow' else 0)

            if not X:
                logger.warning("No training data found")
                return np.array([]), np.array([])

            logger.info(f"Fetched {len(X)} training samples")

            return np.array(X), np.array(y)

        finally:
            await conn.close()

    async def train(
        self,
        lookback_days: int = 30,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train fairness classifier"""

        # Fetch data
        X, y = await self.fetch_training_data(lookback_days)

        if len(X) < 50:
            logger.warning(f"Insufficient training data: {len(X)} samples")
            return {
                "success": False,
                "reason": "insufficient_data",
                "samples": len(X)
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )

        logger.info("Training fairness classifier...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        accuracy = np.mean(y_pred == y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Feature importance
        feature_names = self.feature_extractor.get_feature_names()
        feature_importance = dict(zip(
            feature_names,
            self.model.feature_importances_
        ))

        # Sort by importance
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Save model
        self.save_model()

        logger.info(f"Training complete. Accuracy: {accuracy:.3f}")

        return {
            "success": True,
            "samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "top_features": top_features,
            "timestamp": datetime.utcnow().isoformat()
        }

    def predict_fairness(self, features: Dict[str, Any]) -> Tuple[int, float]:
        """Predict if decision will be fair"""

        if not self.model or not self.scaler:
            logger.warning("Model not trained")
            return 0, 0.0

        # Extract features
        feature_vector = self.feature_extractor.extract_features({"metadata": features})

        if feature_vector is None:
            return 0, 0.0

        # Scale and predict
        feature_scaled = self.scaler.transform([feature_vector])
        prediction = self.model.predict(feature_scaled)[0]
        probability = self.model.predict_proba(feature_scaled)[0][1]  # Probability of "allow"

        return int(prediction), float(probability)

    def suggest_fairness_threshold(
        self,
        current_threshold: float,
        target_fairness_rate: float = 0.85
    ) -> Dict[str, Any]:
        """Suggest new fairness threshold based on model analysis"""

        if not self.model:
            return {
                "success": False,
                "reason": "model_not_trained"
            }

        # Analyze feature importance
        feature_names = self.feature_extractor.get_feature_names()
        feature_importance = dict(zip(
            feature_names,
            self.model.feature_importances_
        ))

        # Find fairness_score importance
        fairness_importance = feature_importance.get("fairness_score", 0.0)

        # Heuristic adjustment based on importance
        if fairness_importance > 0.2:  # High importance
            # Current threshold is significant - adjust conservatively
            suggested_threshold = current_threshold * 0.95  # Reduce by 5%
            confidence = 0.85
        else:
            # Fairness score less important - other factors matter more
            suggested_threshold = current_threshold * 0.90  # Reduce by 10%
            confidence = 0.70

        # Clamp to reasonable range
        suggested_threshold = max(0.60, min(0.90, suggested_threshold))

        return {
            "success": True,
            "current_threshold": current_threshold,
            "suggested_threshold": suggested_threshold,
            "confidence": confidence,
            "reasoning": f"Fairness score importance: {fairness_importance:.3f}",
            "top_factors": sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""

        if not self.model:
            return {
                "trained": False
            }

        return {
            "trained": True,
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "n_features": self.model.n_features_in_,
            "model_path": self.model_path
        }

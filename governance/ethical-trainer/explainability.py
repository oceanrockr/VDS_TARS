"""
Explainability module for ethical fairness decisions
Provides LIME-style explanations
"""
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class SimpleLIMEExplainer:
    """Simplified LIME explainer for fairness decisions"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_decision(
        self,
        features: np.ndarray,
        prediction: int,
        probability: float,
        feature_importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate explanation for a fairness decision"""

        # Get feature values
        feature_values = dict(zip(self.feature_names, features))

        # Sort features by importance
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top positive contributors
        top_positive = []
        top_negative = []

        for feature_name, importance in sorted_importance[:5]:
            value = feature_values.get(feature_name, 0.0)

            if importance > 0.05:  # Significant importance
                contribution = {
                    "feature": feature_name,
                    "value": float(value),
                    "importance": float(importance),
                    "direction": "positive" if value > 0.5 else "negative"
                }

                if value > 0.5:
                    top_positive.append(contribution)
                else:
                    top_negative.append(contribution)

        # Generate natural language explanation
        decision_text = "ALLOW" if prediction == 1 else "DENY"
        confidence_text = f"{probability * 100:.1f}%"

        explanation_text = f"Decision: {decision_text} (confidence: {confidence_text})\n\n"

        if prediction == 1:
            explanation_text += "Key factors supporting ALLOW:\n"
            for contrib in top_positive[:3]:
                explanation_text += f"  • {contrib['feature']}: {contrib['value']:.3f} (importance: {contrib['importance']:.3f})\n"
        else:
            explanation_text += "Key factors supporting DENY:\n"
            for contrib in top_negative[:3]:
                explanation_text += f"  • {contrib['feature']}: {contrib['value']:.3f} (importance: {contrib['importance']:.3f})\n"

        explanation_text += "\nTo improve fairness:\n"

        # Suggest improvements
        if prediction == 0:  # DENY
            # Find lowest training_dist values
            training_dist_features = {
                k: v for k, v in feature_values.items()
                if k.startswith("training_dist_")
            }

            min_group = min(training_dist_features.items(), key=lambda x: x[1])
            explanation_text += f"  • Increase representation of {min_group[0].replace('training_dist_', '')} group (currently {min_group[1]:.1f}%)\n"

            # Check fairness score
            fairness_score = feature_values.get("fairness_score", 0.0)
            if fairness_score < 0.75:
                explanation_text += f"  • Improve overall fairness score (currently {fairness_score:.2f}, target: ≥0.75)\n"

        return {
            "decision": decision_text,
            "confidence": probability,
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "explanation_text": explanation_text,
            "feature_values": feature_values,
            "recommendations": self._generate_recommendations(
                feature_values,
                prediction,
                probability
            )
        }

    def _generate_recommendations(
        self,
        feature_values: Dict[str, float],
        prediction: int,
        probability: float
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""

        recommendations = []

        # Check demographic balance
        training_dist = {
            k.replace("training_dist_", ""): v
            for k, v in feature_values.items()
            if k.startswith("training_dist_")
        }

        min_group = min(training_dist.items(), key=lambda x: x[1])
        if min_group[1] < 5.0:
            recommendations.append({
                "type": "demographic_balance",
                "priority": "high",
                "action": f"Increase {min_group[0]} representation to at least 5.0%",
                "current_value": f"{min_group[1]:.1f}%",
                "target_value": "≥5.0%"
            })

        # Check fairness score
        fairness_score = feature_values.get("fairness_score", 0.0)
        if fairness_score < 0.75:
            recommendations.append({
                "type": "fairness_threshold",
                "priority": "critical" if fairness_score < 0.65 else "high",
                "action": "Improve fairness score through balanced training data",
                "current_value": f"{fairness_score:.2f}",
                "target_value": "≥0.75"
            })

        # Check outcome variance
        outcome_variance = feature_values.get("outcome_variance", 0.0)
        if outcome_variance > 0.05:
            recommendations.append({
                "type": "outcome_disparity",
                "priority": "medium",
                "action": "Reduce outcome disparity across demographic groups",
                "current_value": f"{outcome_variance:.3f}",
                "target_value": "≤0.05"
            })

        # Check sample size
        log_sample_size = feature_values.get("log_sample_size", 0.0)
        sample_size = int(np.expm1(log_sample_size))  # Inverse of log1p
        if sample_size < 1000:
            recommendations.append({
                "type": "sample_size",
                "priority": "low",
                "action": "Increase training sample size for better model confidence",
                "current_value": str(sample_size),
                "target_value": "≥1000"
            })

        return recommendations


def generate_explanation_summary(explanation: Dict[str, Any]) -> str:
    """Generate a concise summary of the explanation"""

    decision = explanation["decision"]
    confidence = explanation["confidence"]

    summary = f"**{decision}** (Confidence: {confidence * 100:.1f}%)\n\n"

    # Top factors
    if explanation["top_positive_factors"]:
        summary += "**Supporting Factors:**\n"
        for factor in explanation["top_positive_factors"][:2]:
            summary += f"- {factor['feature']}: {factor['value']:.2f}\n"

    if explanation["top_negative_factors"]:
        summary += "\n**Concerning Factors:**\n"
        for factor in explanation["top_negative_factors"][:2]:
            summary += f"- {factor['feature']}: {factor['value']:.2f}\n"

    # Recommendations
    if explanation["recommendations"]:
        summary += "\n**Recommendations:**\n"
        for rec in explanation["recommendations"][:3]:
            summary += f"- [{rec['priority'].upper()}] {rec['action']}\n"

    return summary

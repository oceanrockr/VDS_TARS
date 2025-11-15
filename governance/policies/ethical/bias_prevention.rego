# Ethical Policy: AI Bias Prevention
package tars.ethical.bias_prevention

default allow = false

# Allow AI decisions if bias checks pass
allow {
    input.action == "ai_decision"
    not has_demographic_bias
    not has_outcome_disparity
    fairness_score_acceptable
}

# Check for demographic bias in training data
has_demographic_bias {
    some group in data.protected_groups
    input.training_data_distribution[group] < data.limits.min_representation_pct
}

# Check for outcome disparity
has_outcome_disparity {
    some group in data.protected_groups
    input.outcome_rates[group] < (input.overall_outcome_rate * data.limits.min_parity_ratio)
}

# Check if fairness score is acceptable
fairness_score_acceptable {
    input.fairness_score >= data.limits.min_fairness_score
}

# Violations for bias in training data
violations[msg] {
    input.action == "ai_decision"
    has_demographic_bias
    underrepresented := {group |
        some group in data.protected_groups
        input.training_data_distribution[group] < data.limits.min_representation_pct
    }
    msg := sprintf("Training data has demographic bias - underrepresented groups: %s", [concat(", ", underrepresented)])
}

# Violations for outcome disparity
violations[msg] {
    input.action == "ai_decision"
    has_outcome_disparity
    disparate_groups := {group |
        some group in data.protected_groups
        input.outcome_rates[group] < (input.overall_outcome_rate * data.limits.min_parity_ratio)
    }
    msg := sprintf("AI decisions show outcome disparity for groups: %s", [concat(", ", disparate_groups)])
}

# Violations for low fairness score
violations[msg] {
    input.action == "ai_decision"
    not fairness_score_acceptable
    msg := sprintf("Fairness score %.2f below threshold %.2f", [input.fairness_score, data.limits.min_fairness_score])
}

# Require explainability for high-stakes decisions
violations[msg] {
    input.action == "ai_decision"
    input.decision_impact == "high"
    not input.explainability_available
    msg := "High-stakes AI decisions must have explainability"
}

# Warn on potential proxy discrimination
warn[msg] {
    input.action == "ai_decision"
    some feature in input.features_used
    feature in data.proxy_features
    msg := sprintf("Warning: feature '%s' may be proxy for protected characteristic", [feature])
}

# Protected groups and limits
data := {
    "protected_groups": [
        "race",
        "gender",
        "age",
        "disability",
        "religion",
        "national_origin"
    ],
    "proxy_features": [
        "zip_code",
        "education_level",
        "occupation"
    ],
    "limits": {
        "min_representation_pct": 5.0,
        "min_parity_ratio": 0.8,
        "min_fairness_score": 0.75
    }
}

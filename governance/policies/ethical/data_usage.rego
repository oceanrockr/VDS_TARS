# Ethical Policy: Data Usage and Privacy
package tars.ethical.data_usage

default allow = false

# Allow data usage if compliant with privacy rules
allow {
    input.action == "process_data"
    data_is_anonymized
    has_user_consent
    not contains_pii
}

# Check if data is anonymized
data_is_anonymized {
    input.data_anonymized == true
}

# Check if user consent obtained
has_user_consent {
    input.user_consent == true
}

# Check if data contains PII (Personally Identifiable Information)
contains_pii {
    some field in input.data_fields
    field in data.pii_fields
}

# Violations for missing anonymization
violations[msg] {
    input.action == "process_data"
    not data_is_anonymized
    msg := "Data must be anonymized before processing"
}

# Violations for missing consent
violations[msg] {
    input.action == "process_data"
    not has_user_consent
    msg := "User consent required for data processing"
}

# Violations for PII exposure
violations[msg] {
    input.action == "process_data"
    contains_pii
    msg := sprintf("Data contains PII fields: %s", [concat(", ", input.data_fields & data.pii_fields)])
}

# Deny training on user data without explicit permission
allow {
    input.action == "train_model"
    input.training_data_source == "public"
}

violations[msg] {
    input.action == "train_model"
    input.training_data_source == "user_data"
    not input.explicit_training_permission
    msg := "Cannot train model on user data without explicit permission"
}

# Warn on data retention beyond policy
warn[msg] {
    input.action == "store_data"
    input.retention_days > data.limits.max_retention_days
    msg := sprintf("Data retention exceeds policy: %d days (max: %d)", [input.retention_days, data.limits.max_retention_days])
}

# Default PII fields and limits
data := {
    "pii_fields": [
        "email",
        "phone",
        "ssn",
        "credit_card",
        "address",
        "name",
        "birthdate"
    ],
    "limits": {
        "max_retention_days": 365
    }
}

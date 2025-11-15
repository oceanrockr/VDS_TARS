"""
Rego Policy Patch Generator
Generates Rego policy patches based on cognitive insights
"""
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RegoPatch:
    """Represents a patch to a Rego policy"""
    policy_id: str
    parameter: str
    old_value: Any
    new_value: Any
    patch_type: str  # replace, insert, delete
    rego_snippet: str
    validation_result: Optional[Dict[str, Any]] = None


class RegoPatchGenerator:
    """Generate Rego policy patches from insights"""

    def __init__(self):
        self.supported_patches = {
            "cooldown_seconds": self._patch_numeric_value,
            "max_replicas": self._patch_numeric_value,
            "min_replicas": self._patch_numeric_value,
            "fairness_score": self._patch_numeric_value,
            "fairness_threshold": self._patch_numeric_value,
            "timeout_ms": self._patch_numeric_value,
            "rate_limit": self._patch_numeric_value,
        }

    def generate_patch(
        self,
        policy_id: str,
        recommendation: Dict[str, Any]
    ) -> Optional[RegoPatch]:
        """Generate a Rego patch from a recommendation"""

        parameter = recommendation.get("parameter")
        if not parameter:
            logger.warning(f"No parameter specified in recommendation for {policy_id}")
            return None

        if parameter not in self.supported_patches:
            logger.warning(f"Unsupported parameter: {parameter}")
            return None

        current_value = recommendation.get("current_value")
        suggested_value = recommendation.get("suggested_value")

        if current_value is None or suggested_value is None:
            logger.warning(f"Missing current or suggested value for {parameter}")
            return None

        # Generate patch using appropriate handler
        handler = self.supported_patches[parameter]
        rego_snippet = handler(parameter, current_value, suggested_value)

        patch = RegoPatch(
            policy_id=policy_id,
            parameter=parameter,
            old_value=current_value,
            new_value=suggested_value,
            patch_type="replace",
            rego_snippet=rego_snippet
        )

        logger.info(f"Generated patch for {policy_id}.{parameter}: {current_value} → {suggested_value}")
        return patch

    def _patch_numeric_value(
        self,
        parameter: str,
        old_value: float,
        new_value: float
    ) -> str:
        """Generate Rego snippet for numeric value replacement"""

        # Example Rego policy structure:
        # limits := {
        #     "cooldown_seconds": 60,
        #     "max_replicas": 10
        # }

        rego_snippet = f"""
# AUTO-GENERATED PATCH
# Parameter: {parameter}
# Change: {old_value} → {new_value}

{parameter} := {new_value}
"""
        return rego_snippet.strip()

    def generate_full_policy(
        self,
        base_policy: str,
        patches: List[RegoPatch]
    ) -> str:
        """Apply patches to a base policy to generate updated version"""

        updated_policy = base_policy

        for patch in patches:
            # Simple regex-based replacement
            # In production, use proper Rego AST manipulation
            pattern = rf'({patch.parameter}\s*:?=\s*)({patch.old_value})'
            replacement = rf'\1{patch.new_value}'

            updated_policy = re.sub(
                pattern,
                replacement,
                updated_policy,
                flags=re.MULTILINE
            )

            logger.info(f"Applied patch: {patch.parameter} = {patch.new_value}")

        return updated_policy

    def validate_patch_syntax(self, rego_code: str) -> Dict[str, Any]:
        """Validate Rego syntax (simplified - use OPA in production)"""

        # Basic syntax checks
        errors = []

        # Check balanced braces
        if rego_code.count('{') != rego_code.count('}'):
            errors.append("Unbalanced braces")

        # Check for common syntax errors
        if ':= :=' in rego_code:
            errors.append("Duplicate assignment operator")

        # Check for undefined variables (simplified)
        assignment_vars = set(re.findall(r'(\w+)\s*:=', rego_code))
        used_vars = set(re.findall(r'\b(\w+)\b', rego_code))

        undefined = used_vars - assignment_vars - {
            'input', 'data', 'true', 'false', 'null', 'count', 'sum', 'max', 'min',
            'sprintf', 'concat', 'allow', 'deny', 'violations', 'some', 'package'
        }

        # Filter out Rego keywords and built-ins
        rego_keywords = {
            'package', 'import', 'default', 'some', 'not', 'with', 'as', 'else',
            'if', 'in', 'every'
        }
        undefined = undefined - rego_keywords

        if undefined and len(undefined) > 3:  # Heuristic
            errors.append(f"Potentially undefined variables: {list(undefined)[:5]}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def generate_scaling_policy_patch(
        self,
        cooldown_seconds: int,
        max_replicas: int,
        min_replicas: int
    ) -> str:
        """Generate a complete scaling policy with updated parameters"""

        policy = f"""
package tars.operational.scaling

default allow = false

# AUTO-GENERATED POLICY
# Generated by Adaptive Policy Learner

limits := {{
    "max_replicas": {max_replicas},
    "min_replicas": {min_replicas},
    "cooldown_seconds": {cooldown_seconds}
}}

allow {{
    input.action == "scale_out"
    input.target_replicas <= limits.max_replicas
    input.target_replicas >= limits.min_replicas
    not within_cooldown_period
}}

allow {{
    input.action == "scale_in"
    input.target_replicas >= limits.min_replicas
    not within_cooldown_period
}}

within_cooldown_period {{
    current_time := time.now_ns() / 1000000000
    last_scale_time := data.last_scaling_event.timestamp
    elapsed := current_time - last_scale_time
    elapsed < limits.cooldown_seconds
}}

violations[msg] {{
    input.target_replicas > limits.max_replicas
    msg := sprintf("Exceeds max replicas: %d > %d",
        [input.target_replicas, limits.max_replicas])
}}

violations[msg] {{
    input.target_replicas < limits.min_replicas
    msg := sprintf("Below min replicas: %d < %d",
        [input.target_replicas, limits.min_replicas])
}}

violations[msg] {{
    within_cooldown_period
    msg := sprintf("Scaling cooldown active: %d seconds remaining",
        [limits.cooldown_seconds])
}}
"""
        return policy.strip()

    def generate_ethical_fairness_patch(
        self,
        fairness_threshold: float,
        min_demographic_balance: float = 5.0
    ) -> str:
        """Generate ethical fairness policy with updated threshold"""

        policy = f"""
package tars.ethical.bias_prevention

default allow = false

# AUTO-GENERATED POLICY
# Generated by Adaptive Policy Learner

fairness_config := {{
    "fairness_threshold": {fairness_threshold},
    "min_demographic_balance": {min_demographic_balance}
}}

allow {{
    input.action == "ai_decision"
    not has_demographic_bias
    not has_outcome_disparity
    input.fairness_score >= fairness_config.fairness_threshold
}}

has_demographic_bias {{
    some group in data.protected_groups
    input.training_data_distribution[group] < fairness_config.min_demographic_balance
}}

has_outcome_disparity {{
    # Check if any group has significantly different outcomes
    some group in data.protected_groups
    group_outcome := input.outcome_by_group[group]
    avg_outcome := sum([input.outcome_by_group[g] | g := data.protected_groups[_]]) / count(data.protected_groups)
    abs_diff := abs(group_outcome - avg_outcome)
    abs_diff > 0.15  # 15% disparity threshold
}}

abs(x) = x {{ x >= 0 }}
abs(x) = -x {{ x < 0 }}

violations[msg] {{
    input.fairness_score < fairness_config.fairness_threshold
    msg := sprintf("Fairness score %.2f below threshold %.2f",
        [input.fairness_score, fairness_config.fairness_threshold])
}}

violations[msg] {{
    has_demographic_bias
    underrepresented := {{group |
        some group in data.protected_groups
        input.training_data_distribution[group] < fairness_config.min_demographic_balance
    }}
    msg := sprintf("Demographic bias detected: %s underrepresented",
        [concat(", ", underrepresented)])
}}
"""
        return policy.strip()

    def diff_policies(self, old_policy: str, new_policy: str) -> str:
        """Generate a human-readable diff of policy changes"""

        import difflib

        old_lines = old_policy.splitlines(keepends=True)
        new_lines = new_policy.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='original.rego',
            tofile='updated.rego',
            lineterm=''
        )

        return ''.join(diff)

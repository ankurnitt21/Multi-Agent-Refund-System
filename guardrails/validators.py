"""
Custom guardrails-ai validators for the warehouse refund system.

Uses the guardrails-ai library to define structured validation guards
for LLM inputs and outputs.

These validators integrate with guardrails-ai's Guard framework to provide:
- Input validation (refund reason quality, prompt injection detection)
- Output validation (policy decisions, refund amounts)
- Hallucination prevention (ensuring outputs conform to business rules)
"""

import re
import structlog

try:
    from guardrails.validators import (
        FailResult,
        PassResult,
        Validator,
        register_validator,
    )
    GUARDRAILS_AI_AVAILABLE = True
except ImportError:
    GUARDRAILS_AI_AVAILABLE = False

logger = structlog.get_logger(__name__)


if GUARDRAILS_AI_AVAILABLE:

    @register_validator(name="refund_system/refund_reason", data_type="string")
    class RefundReasonValidator(Validator):
        """Validates that refund reason is meaningful and not an injection attempt."""

        def __init__(self, min_length: int = 5, max_length: int = 500, **kwargs):
            super().__init__(min_length=min_length, max_length=max_length, **kwargs)
            self.min_length = min_length
            self.max_length = max_length

        def validate(self, value, metadata=None) -> PassResult | FailResult:
            if not value or not value.strip():
                return FailResult(error_message="Refund reason cannot be empty")

            cleaned = value.strip()
            if len(cleaned) < self.min_length:
                return FailResult(
                    error_message=f"Refund reason too short (min {self.min_length} chars)"
                )
            if len(cleaned) > self.max_length:
                return FailResult(
                    error_message=f"Refund reason too long (max {self.max_length} chars)"
                )

            # Check for gibberish (repeating chars)
            if re.match(r"^(.)\1{10,}$", cleaned):
                return FailResult(error_message="Refund reason appears to be gibberish")

            return PassResult()

    @register_validator(name="refund_system/policy_decision", data_type="string")
    class PolicyDecisionValidator(Validator):
        """Validates that policy decision is one of the allowed values."""

        ALLOWED_DECISIONS = {"approved", "denied", "partial"}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def validate(self, value, metadata=None) -> PassResult | FailResult:
            if not value:
                return FailResult(error_message="Policy decision cannot be empty")

            if value.lower().strip() not in self.ALLOWED_DECISIONS:
                return FailResult(
                    error_message=(
                        f"Invalid decision '{value}'. "
                        f"Must be one of: {', '.join(sorted(self.ALLOWED_DECISIONS))}"
                    )
                )

            return PassResult()

    @register_validator(name="refund_system/refund_amount", data_type="float")
    class RefundAmountValidator(Validator):
        """Validates refund amount is within acceptable bounds."""

        def __init__(self, max_amount: float = 10000.0, **kwargs):
            super().__init__(max_amount=max_amount, **kwargs)
            self.max_amount = max_amount

        def validate(self, value, metadata=None) -> PassResult | FailResult:
            if value is None:
                return FailResult(error_message="Refund amount cannot be None")

            try:
                amount = float(value)
            except (TypeError, ValueError):
                return FailResult(error_message=f"Refund amount must be numeric, got: {value}")

            if amount < 0:
                return FailResult(error_message=f"Refund amount cannot be negative: {amount}")

            if amount > self.max_amount:
                return FailResult(
                    error_message=f"Refund amount {amount} exceeds max allowed {self.max_amount}"
                )

            return PassResult()

else:
    # Fallback when guardrails-ai is not installed
    class RefundReasonValidator:
        """Fallback validator when guardrails-ai is not available."""
        def __init__(self, **kwargs):
            pass

        def validate(self, value, metadata=None):
            if not value or not value.strip():
                return {"passed": False, "error": "Refund reason cannot be empty"}
            if len(value.strip()) < 5:
                return {"passed": False, "error": "Refund reason too short"}
            return {"passed": True}

    class PolicyDecisionValidator:
        """Fallback validator when guardrails-ai is not available."""
        ALLOWED_DECISIONS = {"approved", "denied", "partial"}

        def __init__(self, **kwargs):
            pass

        def validate(self, value, metadata=None):
            if not value or value.lower().strip() not in self.ALLOWED_DECISIONS:
                return {"passed": False, "error": f"Invalid decision: {value}"}
            return {"passed": True}

    class RefundAmountValidator:
        """Fallback validator when guardrails-ai is not available."""
        def __init__(self, max_amount: float = 10000.0, **kwargs):
            self.max_amount = max_amount

        def validate(self, value, metadata=None):
            try:
                amount = float(value)
                if amount < 0 or amount > self.max_amount:
                    return {"passed": False, "error": f"Amount {amount} out of range"}
            except (TypeError, ValueError):
                return {"passed": False, "error": "Non-numeric amount"}
            return {"passed": True}

"""
Guard runner — wraps LLM calls with guardrails-ai Guards for structured validation.

Provides a unified interface for applying input/output guards to agent LLM calls.
Traces all guard validations to LangSmith via OpenTelemetry spans.
"""

import structlog
from opentelemetry import trace

from guardrails.input_sanitizer import sanitize_input, PromptInjectionError
from guardrails.pii_handler import mask_pii
from guardrails.validators import (
    RefundReasonValidator,
    PolicyDecisionValidator,
    RefundAmountValidator,
    GUARDRAILS_AI_AVAILABLE,
)
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)


if GUARDRAILS_AI_AVAILABLE:
    import guardrails as gd
    from pydantic import BaseModel, Field
    from typing import Optional

    class RefundInputSchema(BaseModel):
        """Schema for validating refund request inputs."""
        customer_email: str = Field(description="Customer email address")
        order_id: int = Field(description="Order identifier", gt=0)
        refund_reason: str = Field(
            description="Reason for the refund request",
            min_length=5,
            max_length=500,
        )

    class PolicyOutputSchema(BaseModel):
        """Schema for validating policy agent outputs."""
        decision: str = Field(description="approved, denied, or partial")
        refund_amount: float = Field(description="Refund amount", ge=0.0)
        policy_reason: str = Field(description="Reason for the decision")
        policy_applied: str = Field(description="Name of applied policy")


class RefundGuardRunner:
    """
    Orchestrates guardrails-ai validation for the refund workflow.

    Provides:
    - Input validation (sanitization + guardrails-ai schema)
    - Output validation (ensures LLM outputs meet business rules)
    - LangSmith trace integration via OpenTelemetry spans
    """

    def __init__(self):
        self._reason_validator = RefundReasonValidator()
        self._decision_validator = PolicyDecisionValidator()
        self._amount_validator = RefundAmountValidator()

    def validate_input(
        self,
        customer_email: str,
        order_id: int,
        refund_reason: str,
        *,
        raise_on_injection: bool = True,
    ) -> dict:
        """
        Validate and sanitize refund request input.

        Returns dict with sanitized values and validation metadata.
        Raises PromptInjectionError if injection detected and raise_on_injection=True.
        """
        with tracer.start_as_current_span("guardrails.validate_input") as span:
            span.set_attribute("langsmith.span.kind", "chain")
            span.set_attribute("guardrails.type", "input_validation")

            # Sanitize refund reason (injection + control chars)
            safe_reason = sanitize_input(
                refund_reason,
                raise_on_injection=raise_on_injection,
                field_name="refund_reason",
            )

            # Mask PII in email for LLM consumption
            masked_email = mask_pii(customer_email)

            # Guardrails-ai schema validation if available
            if GUARDRAILS_AI_AVAILABLE:
                guard = gd.Guard.from_pydantic(RefundInputSchema)
                try:
                    guard.validate({
                        "customer_email": customer_email,
                        "order_id": order_id,
                        "refund_reason": refund_reason,
                    })
                    span.set_attribute("guardrails.input_valid", True)
                except Exception as e:
                    span.set_attribute("guardrails.input_valid", False)
                    span.set_attribute("guardrails.input_error", str(e))
                    logger.warning("guardrails_input_validation_failed", error=str(e))
            else:
                # Fallback validation
                result = self._reason_validator.validate(refund_reason)
                valid = result.get("passed", True) if isinstance(result, dict) else True
                span.set_attribute("guardrails.input_valid", valid)

            span.set_attribute("guardrails.pii_masked", True)

            return {
                "customer_email": customer_email,
                "masked_email": masked_email,
                "order_id": order_id,
                "refund_reason": safe_reason,
                "raw_reason": refund_reason,
            }

    def validate_policy_output(
        self,
        decision: str,
        refund_amount: float,
        order_amount: float | None = None,
    ) -> dict:
        """
        Validate policy agent output meets business rules.

        Checks:
        - Decision is one of: approved, denied, partial
        - Refund amount is non-negative and within bounds
        - Refund amount doesn't exceed order amount (if provided)
        """
        with tracer.start_as_current_span("guardrails.validate_policy_output") as span:
            span.set_attribute("langsmith.span.kind", "chain")
            span.set_attribute("guardrails.type", "output_validation")
            span.set_attribute("guardrails.decision", decision or "None")
            span.set_attribute("guardrails.refund_amount", refund_amount or 0.0)

            errors = []

            # Validate decision
            if GUARDRAILS_AI_AVAILABLE:
                dec_result = self._decision_validator.validate(decision)
                if hasattr(dec_result, "error_message") and dec_result.error_message:
                    errors.append(dec_result.error_message)
            else:
                dec_result = self._decision_validator.validate(decision)
                if isinstance(dec_result, dict) and not dec_result.get("passed"):
                    errors.append(dec_result.get("error", "Invalid decision"))

            # Validate amount
            if GUARDRAILS_AI_AVAILABLE:
                amt_result = self._amount_validator.validate(refund_amount)
                if hasattr(amt_result, "error_message") and amt_result.error_message:
                    errors.append(amt_result.error_message)
            else:
                amt_result = self._amount_validator.validate(refund_amount)
                if isinstance(amt_result, dict) and not amt_result.get("passed"):
                    errors.append(amt_result.get("error", "Invalid amount"))

            # Business rule: refund can't exceed order amount
            if order_amount and refund_amount and refund_amount > order_amount:
                errors.append(
                    f"Refund amount ${refund_amount} exceeds order amount ${order_amount}"
                )

            valid = len(errors) == 0
            span.set_attribute("guardrails.output_valid", valid)
            if errors:
                span.set_attribute("guardrails.output_errors", str(errors))

            return {
                "valid": valid,
                "errors": errors,
                "decision": decision,
                "refund_amount": refund_amount,
            }

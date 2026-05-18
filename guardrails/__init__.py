"""
Guardrails module — uses guardrails-ai for input/output validation.

Components:
- input_sanitizer: Prompt injection detection and neutralization
- pii_handler: PII detection and masking
- validators: Custom guardrails-ai validators for the refund system
- runner: Guard runner that wraps LLM calls with guardrails-ai guards
"""

from guardrails.input_sanitizer import sanitize_input, PromptInjectionError
from guardrails.pii_handler import mask_pii, PIIMasker, unmask_pii
from guardrails.validators import (
    RefundReasonValidator,
    PolicyDecisionValidator,
    RefundAmountValidator,
)


def get_guard_runner():
    """Lazy import to avoid circular / heavy import chains at module load."""
    from guardrails.runner import RefundGuardRunner
    return RefundGuardRunner()


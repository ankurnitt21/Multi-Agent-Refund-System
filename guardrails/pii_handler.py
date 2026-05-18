"""
PII (Personally Identifiable Information) Detection and Masking.

Detects and masks common PII patterns before data flows through LLM prompts.
Supports reversible masking (for internal use) and irreversible masking (for LLM).

Masked PII types:
- Email addresses    → j***@domain.com (partial mask)
- Phone numbers      → [PHONE_REDACTED]
- Credit card numbers → [CC_REDACTED]
- SSN / Tax IDs      → [SSN_REDACTED]
- IP addresses       → [IP_REDACTED]

Usage:
    from guardrails.pii_handler import mask_pii, PIIMasker

    # Simple one-shot masking (irreversible)
    masked_text = mask_pii("Contact john@example.com or 555-123-4567")

    # Reversible masking (maintains lookup table for later unmask)
    masker = PIIMasker()
    masked = masker.mask("Email is john@example.com")
    original = masker.unmask(masked)
"""

import re
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)


class PIIMatch(NamedTuple):
    pii_type: str
    original: str
    masked: str
    start: int
    end: int


# PII detection patterns
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]?\d{4}"
    r"(?!\d)"
)

_CC_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)

_SSN_PATTERN = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)

_IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)


class PIIMasker:
    """Reversible PII masker that maintains a mapping for later unmask."""

    def __init__(self):
        self._mappings: dict[str, str] = {}  # masked_token → original
        self._reverse: dict[str, str] = {}   # original → masked_token
        self._counters: dict[str, int] = {}

    def _get_token(self, pii_type: str) -> str:
        count = self._counters.get(pii_type, 0) + 1
        self._counters[pii_type] = count
        return f"[{pii_type}_{count}]"

    def _mask_match(self, pii_type: str, original: str) -> str:
        if original in self._reverse:
            return self._reverse[original]
        token = self._get_token(pii_type)
        self._mappings[token] = original
        self._reverse[original] = token
        return token

    def mask(self, text: str) -> str:
        """Mask all PII in text. Returns masked text with reversible tokens."""
        if not text:
            return text

        result = text

        # Order: longer patterns first to avoid partial matches
        for match in _CC_PATTERN.finditer(result):
            token = self._mask_match("CC", match.group())
            result = result.replace(match.group(), token, 1)

        for match in _SSN_PATTERN.finditer(result):
            if "[CC_" not in result[match.start():match.end() + 10]:
                token = self._mask_match("SSN", match.group())
                result = result.replace(match.group(), token, 1)

        for match in _EMAIL_PATTERN.finditer(result):
            token = self._mask_match("EMAIL", match.group())
            result = result.replace(match.group(), token, 1)

        for match in _PHONE_PATTERN.finditer(result):
            token = self._mask_match("PHONE", match.group())
            result = result.replace(match.group(), token, 1)

        for match in _IP_PATTERN.finditer(result):
            token = self._mask_match("IP", match.group())
            result = result.replace(match.group(), token, 1)

        if self._counters:
            logger.info(
                "pii_masked",
                counts={k: v for k, v in self._counters.items()},
            )

        return result

    def unmask(self, text: str) -> str:
        """Restore masked tokens back to original PII values."""
        result = text
        for token, original in self._mappings.items():
            result = result.replace(token, original)
        return result

    @property
    def detected_count(self) -> int:
        return sum(self._counters.values())


def mask_pii(text: str) -> str:
    """
    One-shot irreversible PII masking.
    Use this for data going to LLM where you never need the original back.
    """
    if not text:
        return text

    result = text
    result = _CC_PATTERN.sub("[CC_REDACTED]", result)
    result = _SSN_PATTERN.sub("[SSN_REDACTED]", result)
    result = _EMAIL_PATTERN.sub(lambda m: _mask_email(m.group()), result)
    result = _PHONE_PATTERN.sub("[PHONE_REDACTED]", result)
    result = _IP_PATTERN.sub("[IP_REDACTED]", result)
    return result


def unmask_pii(text: str, masker: PIIMasker) -> str:
    """Convenience wrapper — delegates to masker.unmask()."""
    return masker.unmask(text)


def _mask_email(email: str) -> str:
    """Partially mask email — preserves domain for debugging."""
    parts = email.split("@")
    if len(parts) == 2:
        local = parts[0]
        masked_local = local[0] + "***" if len(local) > 1 else "***"
        return f"{masked_local}@{parts[1]}"
    return "[EMAIL_REDACTED]"

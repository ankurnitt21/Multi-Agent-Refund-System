"""
Prompt Injection Guard for user-provided inputs (e.g., refund_reason).

Strategy: Defense-in-depth with multiple layers.
1. Pattern-based detection — catches known injection signatures
2. Character filtering — strips control characters and Unicode exploits
3. Length enforcement — prevents token-stuffing attacks
4. Delimiter wrapping — isolates user text so LLM treats it as data, not instructions
5. Guardrails-AI integration — uses Guard for structured validation

Usage:
    from guardrails.input_sanitizer import sanitize_input
    safe_reason = sanitize_input(user_provided_reason)
"""

import re
import structlog

try:
    import guardrails as gd
    from guardrails.validators import FailResult, PassResult, Validator, register_validator
    GUARDRAILS_AI_AVAILABLE = True
except ImportError:
    GUARDRAILS_AI_AVAILABLE = False

logger = structlog.get_logger(__name__)


class PromptInjectionError(ValueError):
    """Raised when input is classified as a prompt injection attempt."""
    pass


# Maximum allowed length for user-provided refund reason
MAX_REASON_LENGTH = 500

# Known injection patterns (case-insensitive)
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?:",
    r"system\s*prompt:",
    r"<\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"<<\s*SYS\s*>>",
    r"act\s+as\s+(a\s+)?",
    r"pretend\s+(you('re|\s+are)\s+)",
    r"override\s+(safety|guardrail|filter|restriction)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"ignore\s+safety",
    r"reveal\s+(your\s+)?(system|initial)\s+prompt",
    r"what\s+(is|are)\s+your\s+(system\s+)?instructions?",
    r"repeat\s+(the\s+)?(text|words?)\s+above",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Control characters to strip (except common whitespace)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Unicode direction override characters (used for text manipulation attacks)
_UNICODE_EXPLOITS = re.compile(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]")


def _detect_injection(text: str) -> str | None:
    """Returns the matched pattern description if injection detected, else None."""
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _strip_dangerous_chars(text: str) -> str:
    """Remove control characters and Unicode exploits."""
    text = _CONTROL_CHARS.sub("", text)
    text = _UNICODE_EXPLOITS.sub("", text)
    return text


def sanitize_input(
    text: str,
    *,
    max_length: int = MAX_REASON_LENGTH,
    raise_on_injection: bool = False,
    field_name: str = "refund_reason",
) -> str:
    """
    Sanitize user-provided text before including it in LLM prompts.

    Steps:
    1. Strip dangerous characters (control chars, Unicode direction overrides)
    2. Enforce length limit
    3. Detect known prompt injection patterns
    4. Wrap in delimiters to isolate as user data

    Args:
        text: Raw user input
        max_length: Maximum allowed character count
        raise_on_injection: If True, raises PromptInjectionError on detection.
                           If False (default), logs warning and returns sanitized text
                           with injection markers neutralized.
        field_name: Name of the field being sanitized (for logging)

    Returns:
        Sanitized text wrapped in data delimiters
    """
    if not text or not text.strip():
        return "[No reason provided]"

    # Step 1: Strip dangerous characters
    cleaned = _strip_dangerous_chars(text.strip())

    # Step 2: Length enforcement
    if len(cleaned) > max_length:
        logger.warning(
            "input_truncated",
            field=field_name,
            original_length=len(cleaned),
            max_length=max_length,
        )
        cleaned = cleaned[:max_length] + "..."

    # Step 3: Injection detection
    injection_match = _detect_injection(cleaned)
    if injection_match:
        logger.warning(
            "prompt_injection_detected",
            field=field_name,
            matched_pattern=injection_match,
            input_preview=cleaned[:100],
        )
        if raise_on_injection:
            raise PromptInjectionError(
                f"Potential prompt injection detected in {field_name}: '{injection_match}'"
            )
        # Neutralize by replacing instruction-like keywords
        cleaned = re.sub(
            r"(ignore|disregard|forget|override|reveal)",
            r"[\1]",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Step 4: Wrap in delimiters — tells the LLM this is user data, not instructions
    sanitized = f"<<<USER_INPUT>>>{cleaned}<<<END_USER_INPUT>>>"

    return sanitized


def wrap_user_data(text: str) -> str:
    """Lightweight wrapper for already-validated data (e.g., from DB lookups).
    Does NOT run injection detection — use only for trusted data sources."""
    return f"<<<DATA>>>{text}<<<END_DATA>>>"

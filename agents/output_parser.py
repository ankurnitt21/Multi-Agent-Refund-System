"""
Robust Output Parser for agent LLM responses.

Replaces fragile manual JSON extraction (```json``` fence stripping + json.loads)
with a multi-strategy parser:

1. Direct JSON parse (if response is pure JSON)
2. Fenced code block extraction (```json ... ```)
3. First valid JSON object extraction via brace matching
4. Pydantic model validation with clear error messages

Usage:
    from agents.output_parser import parse_agent_output
    from agents.output_models import ValidationOutput

    result = parse_agent_output(llm_response, ValidationOutput)
"""

import json
import re
from typing import TypeVar, Type

import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentOutputParseError(Exception):
    """Raised when all parsing strategies fail."""

    def __init__(self, message: str, raw_output: str, strategies_tried: list[str]):
        self.raw_output = raw_output
        self.strategies_tried = strategies_tried
        super().__init__(message)


def _try_direct_json(text: str) -> dict | None:
    """Strategy 1: Try parsing the entire text as JSON."""
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_fenced_json(text: str) -> dict | None:
    """Strategy 2: Extract JSON from ```json ... ``` or ``` ... ``` fences."""
    pattern = re.compile(r"```json\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = pattern.search(text)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    pattern = re.compile(r"```\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = pattern.search(text)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _try_brace_extraction(text: str) -> dict | None:
    """Strategy 3: Find first complete JSON object by brace matching."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass
                break

    return None


def parse_agent_output(
    raw_output: str,
    model: Type[T] | None = None,
    *,
    agent_name: str = "unknown",
) -> T | dict:
    """
    Parse and validate agent LLM output using multiple strategies.

    Args:
        raw_output: Raw text response from the LLM
        model: Optional Pydantic model class for validation.
               If None, returns the raw dict.
        agent_name: Agent name for logging context

    Returns:
        Validated Pydantic model instance (if model provided) or dict

    Raises:
        AgentOutputParseError: When all extraction strategies fail
    """
    if not raw_output or not raw_output.strip():
        raise AgentOutputParseError(
            "Empty output from agent",
            raw_output=raw_output or "",
            strategies_tried=[],
        )

    strategies_tried = []
    parsed_dict: dict | None = None

    # Strategy 1: Direct JSON
    strategies_tried.append("direct_json")
    parsed_dict = _try_direct_json(raw_output)

    # Strategy 2: Fenced JSON
    if parsed_dict is None:
        strategies_tried.append("fenced_json")
        parsed_dict = _try_fenced_json(raw_output)

    # Strategy 3: Brace extraction
    if parsed_dict is None:
        strategies_tried.append("brace_extraction")
        parsed_dict = _try_brace_extraction(raw_output)

    if parsed_dict is None:
        logger.error(
            "output_parse_failed",
            agent=agent_name,
            strategies_tried=strategies_tried,
            output_preview=raw_output[:200],
        )
        raise AgentOutputParseError(
            f"Failed to extract JSON from {agent_name} output after trying: {strategies_tried}",
            raw_output=raw_output,
            strategies_tried=strategies_tried,
        )

    # Validate against Pydantic model if provided
    if model is not None:
        try:
            return model.model_validate(parsed_dict)
        except ValidationError as e:
            logger.warning(
                "output_validation_failed",
                agent=agent_name,
                errors=e.error_count(),
                details=str(e),
            )
            raise AgentOutputParseError(
                f"Output from {agent_name} failed validation: {e}",
                raw_output=raw_output,
                strategies_tried=strategies_tried + ["pydantic_validation"],
            )

    logger.debug(
        "output_parsed",
        agent=agent_name,
        strategy=strategies_tried[-1] if strategies_tried else "unknown",
    )
    return parsed_dict

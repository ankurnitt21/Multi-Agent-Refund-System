"""Tests for output_parser.py — multi-strategy JSON extraction."""

import pytest
from agents.output_parser import (
    parse_agent_output,
    AgentOutputParseError,
)
from agents.output_models import ValidationOutput, PolicyOutput, CommunicationOutput


class TestParseAgentOutput:
    def test_direct_json(self):
        raw = '{"decision": "approved", "refund_amount": 50.0, "policy_reason": "eligible"}'
        result = parse_agent_output(raw)
        assert result["decision"] == "approved"

    def test_fenced_json(self):
        raw = """Here is my decision:
```json
{"decision": "denied", "refund_amount": 0, "policy_reason": "past window"}
```
"""
        result = parse_agent_output(raw)
        assert result["decision"] == "denied"

    def test_brace_extraction(self):
        raw = """After careful analysis, I conclude:
The refund should be {"decision": "partial", "refund_amount": 25.0, "policy_reason": "50% wear"}
That's my final answer."""
        result = parse_agent_output(raw)
        assert result["decision"] == "partial"

    def test_pydantic_validation(self):
        raw = '{"decision": "approved", "refund_amount": 99.99, "policy_reason": "valid claim"}'
        result = parse_agent_output(raw, PolicyOutput)
        assert isinstance(result, PolicyOutput)
        assert result.decision == "approved"
        assert result.refund_amount == 99.99

    def test_validation_output_model(self):
        raw = '{"validation_passed": true, "validation_reason": "All checks passed", "customer_id": 42}'
        result = parse_agent_output(raw, ValidationOutput)
        assert result.validation_passed is True
        assert result.customer_id == 42

    def test_communication_output_model(self):
        raw = '{"request_saved": true, "analytics_updated": true, "status_updated": true, "confirmation_message": "Done"}'
        result = parse_agent_output(raw, CommunicationOutput)
        assert result.request_saved is True

    def test_empty_output_raises(self):
        with pytest.raises(AgentOutputParseError):
            parse_agent_output("")

    def test_no_json_raises(self):
        with pytest.raises(AgentOutputParseError):
            parse_agent_output("This is just plain text with no JSON")

    def test_invalid_pydantic_raises(self):
        raw = '{"decision": "invalid_value", "refund_amount": -5}'
        with pytest.raises(AgentOutputParseError):
            parse_agent_output(raw, PolicyOutput)

    def test_nested_json(self):
        raw = '{"decision": "approved", "refund_amount": 10.0, "policy_reason": "ok", "metadata": {"source": "auto"}}'
        result = parse_agent_output(raw)
        assert result["metadata"]["source"] == "auto"

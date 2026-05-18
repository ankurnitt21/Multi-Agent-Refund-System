import structlog
from opentelemetry import trace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

from agents.react_loop import react_loop, ReactMaxIterationsError
from agents.state import RefundState
from agents.output_parser import parse_agent_output, AgentOutputParseError
from agents.output_models import ValidationOutput
from tools.db_tools import (
    lookup_customer, get_order_details,
    verify_order_ownership, get_customer_analytics,
)
from config import GROQ_API_KEY, GROQ_MODEL
from prompts.loader import load_prompt
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)

VALIDATION_TOOLS = [
    lookup_customer, get_order_details,
    get_customer_analytics, verify_order_ownership,
]
VALIDATION_TOOL_MAP = {t.name: t for t in VALIDATION_TOOLS}

# Dependency map — drives parallel wave execution inside react_loop.
#
# Wave 0 (parallel): lookup_customer + get_order_details
#   Both only need email / order_id from state.
#
# Wave 1 (parallel): get_customer_analytics + verify_order_ownership
#   Both need customer_id returned by lookup_customer in wave 0.
VALIDATION_DEPS: dict[str, list[str]] = {
    "lookup_customer":        [],
    "get_order_details":      [],
    "get_customer_analytics": ["lookup_customer"],
    "verify_order_ownership": ["lookup_customer"],
}

VALIDATION_SYSTEM = load_prompt("validation_agent")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
_llm_with_validation_tools = llm.bind_tools(VALIDATION_TOOLS)


async def validation_agent_node(state: RefundState) -> dict:
    """ReAct validation: LLM calls tools, reasons over results, returns structured decision."""
    with tracer.start_as_current_span("validation_agent") as span:
        span.set_attribute("langsmith.span.kind", "chain")
        span.set_attribute("agent.name", "validation_agent")

        customer_email = state.get("customer_email", "")
        order_id = state.get("order_id", 0)

        user_message = (
            f"Validate this refund request:\n"
            f"- Customer email: {customer_email}\n"
            f"- Order ID: {order_id}\n\n"
            f"Call all required tools, apply the business rules, then return the JSON result."
        )

        messages = [
            SystemMessage(content=VALIDATION_SYSTEM),
            HumanMessage(content=user_message),
        ]

        try:
            final_content = await react_loop(
                _llm_with_validation_tools, VALIDATION_TOOL_MAP, messages,
                VALIDATION_DEPS, span=span,
            )

            parsed = parse_agent_output(
                final_content, ValidationOutput, agent_name="validation"
            )
            # Convert Pydantic model to dict for state compatibility
            parsed = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed

            span.set_attribute("validation.passed", parsed.get("validation_passed", False))
            span.set_attribute("validation.reason", parsed.get("validation_reason", ""))

            if not parsed.get("validation_passed"):
                logger.warning(
                    "validation_failed",
                    order_id=order_id,
                    reason=parsed.get("validation_reason"),
                )
                return _fail(state, parsed.get("validation_reason", "Validation failed"), span)

            logger.info(
                "validation_passed",
                customer_id=parsed.get("customer_id"),
                order_id=parsed.get("order_id"),
                customer_tier=parsed.get("customer_tier"),
                risk_score=parsed.get("customer_risk_score"),
            )

            return {
                "customer_id":         parsed.get("customer_id"),
                "customer_name":       parsed.get("customer_name"),
                "customer_email":      parsed.get("customer_email"),
                "customer_tier":       parsed.get("customer_tier"),
                "customer_risk_score": parsed.get("customer_risk_score", 0.0),
                "order_id":            parsed.get("order_id"),
                "order_amount":        parsed.get("order_amount"),
                "order_date":          parsed.get("order_date"),
                "order_status":        parsed.get("order_status"),
                "product_name":        parsed.get("product_name"),
                "product_category":    parsed.get("product_category"),
                "payment_method":      parsed.get("payment_method"),
                "validation_passed":   True,
                "validation_reason":   parsed.get("validation_reason", "All validation checks passed"),
                "messages": [AIMessage(content=f"Validation passed for order #{order_id}")],
            }

        except ReactMaxIterationsError as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.warning("validation_hitl_max_iter", order_id=order_id, error=str(e))
            return {
                "hitl_required": True,
                "hitl_reason": "react_max_iter:validation",
                "order_id": order_id,
                "messages": [AIMessage(content=f"Validation HITL: ReAct max iterations — {e}")],
            }
        except AgentOutputParseError as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.error("validation_parse_error", order_id=order_id, error=str(e))
            return _fail(state, f"Output parse error: {e}", span)
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            return _fail(state, str(e), span)


def _fail(state: dict, reason: str, span) -> dict:
    span.set_attribute("validation.passed", False)
    span.set_attribute("validation.reason", reason)
    return {
        "validation_passed":  False,
        "validation_reason":  reason,
        "order_id":           state.get("order_id"),
        "messages": [AIMessage(content=f"Validation failed: {reason}")],
    }

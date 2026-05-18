import structlog
from opentelemetry import trace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

from agents.react_loop import react_loop, ReactMaxIterationsError
from agents.state import RefundState
from guardrails.input_sanitizer import sanitize_input
from guardrails.pii_handler import mask_pii
from agents.output_parser import parse_agent_output, AgentOutputParseError
from agents.output_models import PolicyOutput
from tools.policy_tools import (
    search_policies, check_eligibility,
    get_product_return_policy, calculate_refund,
)
from tools.db_tools import check_refund_history
from tools.analytics_tools import (
    get_customer_order_summary, get_refund_rate_by_category,
    get_similar_refund_decisions,
)
from config import GROQ_API_KEY, GROQ_MODEL
from prompts.loader import load_prompt
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)

POLICY_TOOLS = [
    search_policies, check_refund_history, check_eligibility,
    get_product_return_policy, calculate_refund,
    get_customer_order_summary, get_refund_rate_by_category,
    get_similar_refund_decisions,
]
POLICY_TOOL_MAP = {t.name: t for t in POLICY_TOOLS}

# Dependency map — calculate_refund must run after check_eligibility.
# All other tools are fully independent.
POLICY_DEPS: dict[str, list[str]] = {
    "search_policies":            [],
    "check_refund_history":        [],
    "check_eligibility":           [],
    "get_product_return_policy":   [],
    "get_customer_order_summary":  [],
    "get_refund_rate_by_category": [],
    "get_similar_refund_decisions": [],
    "calculate_refund":            ["check_eligibility"],
}

POLICY_SYSTEM = load_prompt("policy_agent")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
_llm_with_policy_tools = llm.bind_tools(POLICY_TOOLS)


async def policy_agent_node(state: RefundState) -> dict:
    with tracer.start_as_current_span("policy_agent") as span:
        span.set_attribute("langsmith.span.kind", "chain")
        span.set_attribute("agent.name", "policy_agent")

        # Sanitize user-provided refund reason (prompt injection + PII protection)
        raw_reason = state.get("refund_reason", "")
        safe_reason = sanitize_input(raw_reason, field_name="refund_reason")
        safe_reason = mask_pii(safe_reason)

        user_message = (
            f"Process this refund request:\n"
            f"- Customer: {state.get('customer_name')} (ID: {state.get('customer_id')}, "
            f"Tier: {state.get('customer_tier')}, Risk: {state.get('customer_risk_score', 0.0)})\n"
            f"- Order: #{state.get('order_id')}, Amount: ${state.get('order_amount')}, "
            f"Date: {state.get('order_date')}\n"
            f"- Product: {state.get('product_name')} (Category: {state.get('product_category')})\n"
            f"- Refund Reason: {safe_reason}\n\n"
            f"Use your tools step by step, then return the JSON decision."
        )

        messages = [
            SystemMessage(content=POLICY_SYSTEM),
            HumanMessage(content=user_message),
        ]

        try:
            final_content = await react_loop(
                _llm_with_policy_tools, POLICY_TOOL_MAP, messages, POLICY_DEPS, span=span
            )

            parsed = parse_agent_output(
                final_content, PolicyOutput, agent_name="policy"
            )
            parsed = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed

            span.set_attribute("policy.decision", parsed.get("decision", "unknown"))
            span.set_attribute("policy.refund_amount", str(parsed.get("refund_amount", 0.0)))

            logger.info(
                "policy_decision",
                decision=parsed.get("decision"),
                refund_amount=parsed.get("refund_amount"),
                policy_applied=parsed.get("policy_applied"),
                refund_count_90_days=parsed.get("refund_count_90_days"),
                order_id=state.get("order_id"),
            )

            return {
                "decision": parsed.get("decision"),
                "refund_amount": parsed.get("refund_amount", 0.0),
                "policy_reason": parsed.get("policy_reason", ""),
                "policy_applied": parsed.get("policy_applied", ""),
                "policy_context": parsed.get("policy_context", ""),
                "refund_count_90_days": parsed.get("refund_count_90_days", 0),
                "messages": [AIMessage(content=f"Policy → {parsed.get('decision')}, ${parsed.get('refund_amount', 0.0)}")],
            }
        except ReactMaxIterationsError as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.warning("policy_hitl_max_iter", order_id=state.get("order_id"), error=str(e))
            return {
                "hitl_required": True,
                "hitl_reason": "react_max_iter:policy",
                "messages": [AIMessage(content=f"Policy HITL: ReAct max iterations — {e}")],
            }
        except AgentOutputParseError as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.error("policy_parse_error", order_id=state.get("order_id"), error=str(e))
            return {
                "decision": "denied",
                "refund_amount": 0.0,
                "policy_reason": f"Output parse error: {e}",
                "policy_applied": "error",
                "messages": [AIMessage(content=f"Policy parse error: {e}")],
            }
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            return {
                "decision": "denied",
                "refund_amount": 0.0,
                "policy_reason": str(e),
                "policy_applied": "error",
                "messages": [AIMessage(content=f"Policy error: {e}")],
            }

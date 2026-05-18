import structlog
from opentelemetry import trace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from agents.react_loop import react_loop, ReactMaxIterationsError
from agents.state import RefundState
from agents.output_parser import parse_agent_output, AgentOutputParseError
from agents.output_models import CommunicationOutput
from tools.db_tools import (
    save_processed_request, update_customer_analytics,
    update_refund_request_status, log_audit_event,
    ensure_refund_request_id,
)
from config import OPENAI_API_KEY, OPENAI_FAST_MODEL
from prompts.loader import load_prompt
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)

COMMUNICATION_TOOLS = [
    save_processed_request, update_customer_analytics,
    update_refund_request_status, log_audit_event,
]
COMMUNICATION_TOOL_MAP = {t.name: t for t in COMMUNICATION_TOOLS}

# All four tools operate only on data already in state so they are fully independent.
COMMUNICATION_DEPS: dict[str, list[str]] = {
    "save_processed_request":      [],
    "update_customer_analytics":   [],
    "update_refund_request_status": [],
    "log_audit_event":             [],
}

COMMUNICATION_SYSTEM = load_prompt("communication_agent")

llm = ChatOpenAI(model=OPENAI_FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0)
_llm_with_comm_tools = llm.bind_tools(COMMUNICATION_TOOLS)


async def communication_agent_node(state: RefundState) -> dict:
    with tracer.start_as_current_span("communication_agent") as span:
        span.set_attribute("langsmith.span.kind", "chain")
        span.set_attribute("agent.name", "communication_agent")

        refund_request_id = state.get("refund_request_id")
        if not refund_request_id and state.get("order_id") and state.get("customer_id"):
            refund_request_id = await ensure_refund_request_id(
                order_id=state["order_id"],
                customer_id=state["customer_id"],
                reason=state.get("refund_reason", "Refund workflow"),
            )

        user_message = (
            f"Persist the following refund decision:\n"
            f"- Customer ID: {state.get('customer_id')}\n"
            f"- Customer Name: {state.get('customer_name')}\n"
            f"- Order ID: {state.get('order_id')}\n"
            f"- Refund Request ID: {refund_request_id}\n"
            f"- Decision: {state.get('decision')}\n"
            f"- Refund Amount: ${state.get('refund_amount', 0.0)}\n"
            f"- Policy Applied: {state.get('policy_applied')}\n"
            f"- Policy Reason: {state.get('policy_reason')}\n\n"
            f"Call all four tools step by step, then return the JSON confirmation.\n"
            f"IMPORTANT: When calling update_customer_analytics, pass "
            f"refund_request_id={state.get('refund_request_id')} for idempotency."
        )

        messages = [
            SystemMessage(content=COMMUNICATION_SYSTEM),
            HumanMessage(content=user_message),
        ]

        try:
            final_content = await react_loop(
                _llm_with_comm_tools, COMMUNICATION_TOOL_MAP, messages, COMMUNICATION_DEPS, span=span
            )

            parsed = parse_agent_output(
                final_content, CommunicationOutput, agent_name="communication"
            )
            parsed = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed

            return {
                "refund_request_id": refund_request_id,
                "request_saved": True,
                "analytics_updated": parsed.get("analytics_updated", True),
                "status_updated": parsed.get("status_updated", True),
                "messages": [AIMessage(content=parsed.get("confirmation_message", "All records updated."))],
            }
        except ReactMaxIterationsError as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.warning("communication_hitl_max_iter", order_id=state.get("order_id"), error=str(e))
            return {
                "hitl_required": True,
                "hitl_reason": "react_max_iter:communication",
                "messages": [AIMessage(content=f"Communication HITL: ReAct max iterations — {e}")],
            }
        except AgentOutputParseError as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.error("communication_parse_error", order_id=state.get("order_id"), error=str(e))
            return {
                "hitl_required": True,
                "hitl_reason": f"communication_parse_error:{e}",
                "messages": [AIMessage(content=f"Communication parse error: {e}")],
            }
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.error(
                "communication_error",
                error=str(e),
                order_id=state.get("order_id"),
                exc_info=True,
            )
            # Do NOT set request_saved=True — the decision was NOT persisted.
            # Route to HITL so a human can decide to retry or compensate.
            return {
                "hitl_required": True,
                "hitl_reason": f"communication_error:{e}",
                "messages": [AIMessage(content=f"Communication HITL: DB write failed — {e}")],
            }


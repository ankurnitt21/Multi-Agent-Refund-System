from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from opentelemetry import trace
import structlog

from agents.state import RefundState
from config import GROQ_API_KEY, GROQ_MODEL
from prompts.loader import load_prompt
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Resilience constants
# ---------------------------------------------------------------------------
MAX_CYCLES = 15              # Hard cap on supervisor passes before forced end
MAX_AGENT_RETRIES = 3        # Max times a single agent may be invoked


# ---------------------------------------------------------------------------
# Routing tools — pure signals, no logic.
# The LLM calls ONE of these to express which agent should run next.
# Tool descriptions are the only "rules" — LLM reasons from context.
# ---------------------------------------------------------------------------

@tool
def call_validation_agent() -> str:
    """Use when validation_passed is None — customer/order has not been validated yet."""
    return "validation"


@tool
def call_policy_agent() -> str:
    """Use when validation_passed is True AND decision is None — need a refund decision."""
    return "policy"


@tool
def call_communication_agent() -> str:
    """Use when decision is set (not None) AND request_saved is not True — need to save the decision."""
    return "communication"


@tool
def finish_workflow() -> str:
    """Use when validation_passed is False OR request_saved is True — workflow is done."""
    return "end"


ROUTING_TOOLS = [
    call_validation_agent,
    call_policy_agent,
    call_communication_agent,
    finish_workflow,
]

# Maps tool name → agent key used by route_supervisor in graph.py
TOOL_TO_AGENT: dict[str, str] = {
    "call_validation_agent":    "validation",
    "call_policy_agent":        "policy",
    "call_communication_agent": "communication",
    "finish_workflow":          "end",
}

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
# tool_choice="any" forces the model to always call exactly one routing tool
_supervisor_llm = llm.bind_tools(ROUTING_TOOLS, tool_choice="any")

SUPERVISOR_SYSTEM = load_prompt("supervisor")


async def supervisor_node(state: RefundState) -> dict:
    """Tool-calling supervisor — the LLM decides which agent to invoke next.

    The LLM receives the full workflow state and calls one of the routing tools
    to express its decision. Safety guards (cycle cap, per-agent retry cap) are
    enforced before the LLM is called so they cannot be bypassed.

    Resilience additions:
    - cycle_count: incremented every pass; forces END at MAX_CYCLES.
    - agent_retry_counts: per-agent invocation counter; triggers compensation
      when an agent exceeds MAX_AGENT_RETRIES.
    """
    with tracer.start_as_current_span("supervisor_node") as span:
        span.set_attribute("langsmith.span.kind", "chain")
        span.set_attribute("agent.name", "supervisor")

        # ── Cycle counter ────────────────────────────────────────────
        cycle_count = (state.get("cycle_count") or 0) + 1
        span.set_attribute("supervisor.cycle_count", cycle_count)

        # ── HITL pass-through: an agent already flagged human review needed ─
        if state.get("hitl_required"):
            return {
                "next_agent": "hitl",
                "cycle_count": cycle_count,
                "messages": [AIMessage(content=(
                    f"Supervisor → hitl | reason: {state.get('hitl_reason', 'unknown')}"
                ))],
            }

        if cycle_count > MAX_CYCLES:
            span.set_attribute("supervisor.forced_hitl", True)
            logger.warning(
                "supervisor_cycle_limit",
                cycle=cycle_count,
                limit=MAX_CYCLES,
            )
            return {
                "next_agent": "hitl",
                "hitl_required": True,
                "hitl_reason": f"Cycle limit ({MAX_CYCLES}) exceeded",
                "cycle_count": cycle_count,
                "messages": [AIMessage(content=f"Supervisor → hitl | cycle limit {MAX_CYCLES} exceeded")],
            }

        # ── Build state summary for the LLM ─────────────────────────
        validation_passed = state.get("validation_passed")
        decision          = state.get("decision")
        request_saved     = state.get("request_saved")

        state_summary = (
            f"Current workflow state:\n"
            f"- validation_passed: {validation_passed}\n"
            f"- decision: {decision}\n"
            f"- request_saved: {request_saved}\n"
            f"- error: {state.get('error')}\n\n"
            f"Call the routing tool that best matches the state above."
        )

        # ── LLM decides the route via tool-calling ───────────────────
        try:
            response = await _supervisor_llm.ainvoke([
                SystemMessage(content=SUPERVISOR_SYSTEM),
                HumanMessage(content=state_summary),
            ])
            if response.tool_calls:
                llm_tool  = response.tool_calls[0]["name"]
                next_agent = TOOL_TO_AGENT.get(llm_tool, "end")
                reasoning  = f"LLM→{llm_tool}"
            else:
                # LLM returned no tool call — safe fallback
                next_agent = "end"
                reasoning  = "[no tool call returned — defaulting to end]"
        except Exception as e:
            logger.error("supervisor_llm_error", error=str(e), cycle=cycle_count)
            span.set_status(trace.StatusCode.ERROR, str(e))
            return {
                "next_agent": "hitl",
                "hitl_required": True,
                "hitl_reason": f"supervisor_llm_error:{e}",
                "cycle_count": cycle_count,
                "agent_retry_counts": dict(state.get("agent_retry_counts") or {}),
                "messages": [AIMessage(content=f"Supervisor LLM error → hitl: {e}")],
            }

        # ── Per-agent retry tracking (applied after LLM decision) ────
        retry_counts = dict(state.get("agent_retry_counts") or {})
        if next_agent not in ("end", "hitl"):
            retry_counts[next_agent] = retry_counts.get(next_agent, 0) + 1
            span.set_attribute(f"supervisor.retry.{next_agent}", retry_counts[next_agent])

            if retry_counts[next_agent] > MAX_AGENT_RETRIES:
                span.set_attribute("supervisor.retry_exhausted", next_agent)
                logger.warning(
                    "supervisor_retry_exhausted",
                    agent=next_agent,
                    retries=retry_counts[next_agent],
                    limit=MAX_AGENT_RETRIES,
                )
                return {
                    "next_agent": "hitl",
                    "hitl_required": True,
                    "hitl_reason": f"retry_exhausted:{next_agent}",
                    "cycle_count": cycle_count,
                    "agent_retry_counts": retry_counts,
                    "messages": [AIMessage(
                        content=f"Supervisor → hitl | {next_agent} retries exhausted"
                    )],
                }

        span.set_attribute("supervisor.next_agent", next_agent)
        span.set_attribute("supervisor.reasoning", reasoning)

        logger.info(
            "supervisor_routing",
            next_agent=next_agent,
            cycle=cycle_count,
            retry_counts=retry_counts,
            reasoning=reasoning,
        )

        return {
            "next_agent": next_agent,
            "cycle_count": cycle_count,
            "agent_retry_counts": retry_counts,
            "messages": [AIMessage(content=f"Supervisor → {next_agent} | {reasoning}")],
        }

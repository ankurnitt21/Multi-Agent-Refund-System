import operator
from typing import Optional, Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class RefundState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

    # Set by validation agent
    customer_id: Optional[int]
    customer_name: Optional[str]
    customer_email: Optional[str]
    customer_tier: Optional[str]
    customer_risk_score: Optional[float]
    order_id: Optional[int]
    order_amount: Optional[float]
    order_date: Optional[str]
    order_status: Optional[str]
    product_name: Optional[str]
    product_category: Optional[str]
    payment_method: Optional[str]
    refund_request_id: Optional[int]
    validation_passed: Optional[bool]
    validation_reason: Optional[str]

    # Set by policy agent
    decision: Optional[str]
    refund_amount: Optional[float]
    policy_reason: Optional[str]
    policy_applied: Optional[str]
    policy_context: Optional[str]
    refund_count_90_days: Optional[int]

    # Set by communication agent
    request_saved: Optional[bool]
    analytics_updated: Optional[bool]
    status_updated: Optional[bool]

    # Router
    next_agent: Optional[str]
    error: Optional[str]
    refund_reason: Optional[str]

    # Resilience: loop/cycle counter — incremented by supervisor each pass
    cycle_count: Optional[int]

    # Resilience: per-agent retry tracking  {"validation": 2, "policy": 1, ...}
    agent_retry_counts: Optional[dict]

    # Resilience: set when retry exhaustion triggers compensation
    compensated: Optional[bool]
    compensation_reason: Optional[str]

    # Checkpoint: passed through so node wrappers can tag saves by thread
    thread_id: Optional[str]

    # HITL: set by supervisor or agents when human review is required
    hitl_required: Optional[bool]
    hitl_reason: Optional[str]

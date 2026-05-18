"""
Pydantic output models for each agent's expected LLM response.

These define the contract between agent LLM output and the system.
Used by output_parser.parse_agent_output() for validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ValidationOutput(BaseModel):
    """Expected output from the Validation Agent."""
    validation_passed: bool
    validation_reason: str
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_tier: Optional[str] = None
    customer_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    order_id: Optional[int] = None
    order_amount: Optional[float] = None
    order_date: Optional[str] = None
    order_status: Optional[str] = None
    product_name: Optional[str] = None
    product_category: Optional[str] = None
    payment_method: Optional[str] = None


class PolicyOutput(BaseModel):
    """Expected output from the Policy Agent."""
    decision: str = Field(description="approved | denied | partial")
    refund_amount: float = Field(default=0.0, ge=0.0)
    policy_reason: str = ""
    policy_applied: str = ""
    policy_context: str = ""
    refund_count_90_days: int = Field(default=0, ge=0)


class CommunicationOutput(BaseModel):
    """Expected output from the Communication Agent."""
    request_saved: bool = True
    analytics_updated: bool = True
    status_updated: bool = True
    confirmation_message: str = "All records updated."

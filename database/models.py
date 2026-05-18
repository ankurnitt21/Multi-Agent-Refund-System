from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean, DateTime, JSON,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    tier = Column(String(20), nullable=False)
    phone = Column(String(20))
    created_at = Column(DateTime, default=func.now())
    total_orders = Column(Integer, default=0)
    total_refunds = Column(Integer, default=0)
    account_status = Column(String(20), default="active")

    orders = relationship("Order", back_populates="customer")
    refund_requests = relationship("RefundRequest", back_populates="customer")
    analytics = relationship("CustomerAnalytics", back_populates="customer", uselist=False)


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    price = Column(Float, nullable=False)
    return_window_days = Column(Integer, default=30)
    restocking_fee_pct = Column(Float, default=0.0)
    stock_quantity = Column(Integer, default=100)

    orders = relationship("Order", back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    purchase_date = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String(30), nullable=False)
    quantity = Column(Integer, default=1)
    payment_method = Column(String(50))
    shipping_address = Column(String(255))

    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product", back_populates="orders")


class RefundRequest(Base):
    __tablename__ = "refund_requests"

    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    reason = Column(Text)
    status = Column(String(20), default="pending")
    requested_at = Column(DateTime, default=func.now())

    customer = relationship("Customer", back_populates="refund_requests")


class ProcessedRequest(Base):
    __tablename__ = "processed_requests"

    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    refund_request_id = Column(Integer, ForeignKey("refund_requests.id"), nullable=False)
    decision = Column(String(20), nullable=False)
    refund_amount = Column(Float, nullable=False)
    reason = Column(Text)
    policy_applied = Column(Text)
    analytics_updated = Column(Boolean, default=False)
    processed_at = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint("refund_request_id", name="uq_processed_refund_request"),
    )


class CustomerAnalytics(Base):
    __tablename__ = "customer_analytics"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    total_spent = Column(Float, default=0.0)
    average_order_value = Column(Float, default=0.0)
    refund_rate = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    last_calculated_at = Column(DateTime)

    customer = relationship("Customer", back_populates="analytics")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    agent_name = Column(String(50), nullable=False)
    tool_called = Column(String(100), nullable=False)
    input_data = Column(JSON)
    output_data = Column(JSON)
    status = Column(String(20), nullable=False)
    error_msg = Column(Text, nullable=True)
    duration_ms = Column(Float)
    timestamp = Column(DateTime, default=func.now())


class PromptRegistry(Base):
    __tablename__ = "prompt_registry"

    id = Column(Integer, primary_key=True)
    prompt_name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    created_by = Column(String(100))

    __table_args__ = (
        UniqueConstraint("prompt_name", "version", name="uq_prompt_name_version"),
    )


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"

    id = Column(Integer, primary_key=True)
    idempotency_key = Column(String(255), unique=True, nullable=False, index=True)
    task_id = Column(String(100), nullable=False)
    status = Column(String(20), default="processing")
    created_at = Column(DateTime, default=func.now())


class TaskResult(Base):
    """Permanent record of every task outcome — survives Redis TTL expiry.

    Written by _process_refund after success, failure, or HITL trigger.
    Used as the final fallback in GET /refund/{task_id}.
    """
    __tablename__ = "task_results"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(String(20), nullable=False)          # completed / error / hitl_pending
    decision = Column(String(20), nullable=True)         # approved / denied
    refund_amount = Column(Float, nullable=True)
    policy_reason = Column(Text, nullable=True)
    policy_applied = Column(Text, nullable=True)
    validation_passed = Column(Boolean, nullable=True)
    validation_reason = Column(Text, nullable=True)
    customer_name = Column(String(100), nullable=True)
    customer_tier = Column(String(20), nullable=True)
    order_id = Column(Integer, nullable=True)
    compensated = Column(Boolean, default=False)
    hitl_reason = Column(String(255), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class HITLTask(Base):
    """One row per task that requires human-in-the-loop review.

    Created by hitl_node when cycle limit, per-agent retry limit, or
    ReAct max iterations are exceeded.  Resolved via the /hitl API.
    """
    __tablename__ = "hitl_tasks"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(100), unique=True, nullable=False, index=True)
    reason = Column(String(255), nullable=False)   # e.g. "cycle_limit", "retry_exhausted:validation"
    state_json = Column(JSON, nullable=False)       # full RefundState snapshot
    status = Column(String(20), default="pending")  # pending / approved / denied / compensated
    created_at = Column(DateTime, default=func.now())
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    resolution_note = Column(Text, nullable=True)


class WorkflowCheckpoint(Base):
    """One row per thread_id — upserted after every agent completes.

    Stores the full (non-message) RefundState so the workflow can be
    resumed from the last completed agent after a crash, even if the
    LangGraph internal checkpointer is unavailable.
    """
    __tablename__ = "workflow_checkpoints"

    id = Column(Integer, primary_key=True)
    thread_id = Column(String(100), unique=True, nullable=False, index=True)
    last_completed_agent = Column(String(50), nullable=False)
    state_json = Column(JSON, nullable=False)
    saved_at = Column(DateTime, default=func.now())

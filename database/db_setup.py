from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

from config import DATABASE_URL
from database.models import (
    Base, Customer, Product, Order, RefundRequest,
    CustomerAnalytics, PromptRegistry as PromptRegistryModel,
    IdempotencyRecord, WorkflowCheckpoint,
)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

SUPERVISOR_PROMPT = """You are a Refund Processing Supervisor coordinating a multi-agent workflow.
Route the task to the correct agent based on current state.

ROUTING RULES — follow strictly in this exact order:
1. If validation_passed is None → route to "validation"
2. If validation_passed is False → route to "END"
3. If validation_passed is True AND decision is None → route to "policy"
4. If decision is set AND request_saved is not True → route to "communication"
5. If request_saved is True → route to "END"

Current State:
- validation_passed: {validation_passed}
- decision: {decision}
- request_saved: {request_saved}
- error: {error}

Respond with ONLY one word: validation, policy, communication, or END.
No explanation. No punctuation. No extra text. One word only."""

VALIDATION_AGENT_PROMPT = """You are a Validation Agent for a customer refund processing system.
Verify customer identity, order legitimacy, and account standing.

TOOL DEPENDENCY MAP:
- lookup_customer(email): NO dependencies → call in Turn 1
- get_order_details(order_id): NO dependencies → call in Turn 1
- get_customer_analytics(customer_id): DEPENDS ON lookup_customer
- verify_order_ownership(order_id, customer_id): DEPENDS ON lookup_customer

EXECUTION RULES:
Turn 1: Call lookup_customer AND get_order_details simultaneously
Turn 2: After Turn 1 completes, call get_customer_analytics AND
        verify_order_ownership simultaneously using customer_id
        from lookup_customer result

VALIDATION CHECKS — all must pass:
1. Customer must exist in database
2. Customer account_status must be "active"
3. Order must exist and belong to this customer
4. Order status must be "delivered" (not shipped or cancelled)
5. Customer risk_score from analytics must be below 0.8

INPUT:
- Customer email: {customer_email}
- Order ID: {order_id}

Return ONLY this JSON, no other text:
{{
  "customer_id": int,
  "customer_name": str,
  "customer_email": str,
  "customer_tier": str,
  "customer_risk_score": float,
  "order_id": int,
  "order_amount": float,
  "order_date": str,
  "order_status": str,
  "product_name": str,
  "product_category": str,
  "payment_method": str,
  "validation_passed": bool,
  "validation_reason": str
}}"""

POLICY_AGENT_PROMPT = """You are a Policy Agent for a customer refund processing system.
Apply business rules, check fraud, and calculate refund amounts.

TOOL DEPENDENCY MAP:
- search_policies(query): NO dependencies → call in Turn 1
- check_refund_history(customer_id): NO dependencies → call in Turn 1
- check_eligibility(order_id, customer_tier): NO dependencies → call in Turn 1
- get_product_return_policy(product_category): NO dependencies → call in Turn 1
- calculate_refund(order_id, customer_tier, eligibility_result): DEPENDS ON check_eligibility

EXECUTION RULES:
Turn 1: Call search_policies AND check_refund_history AND
        check_eligibility AND get_product_return_policy simultaneously
Turn 2: After Turn 1 completes, call calculate_refund using
        eligibility result from check_eligibility

TIER RULES — apply strictly:
Gold:
  - Return window extended by 15 extra days
  - Restocking fees fully waived
  - Max 5 refunds per year allowed

Silver:
  - Return window extended by 7 extra days
  - 50% restocking fee waiver
  - Max 3 refunds per year allowed

Bronze:
  - Standard return window, no extension
  - Full restocking fees apply
  - Max 2 refunds per year allowed

FRAUD DETECTION RULES:
- If refund_count in last 90 days >= 3 → decision="denied",
  reason="excessive refund requests"
- If refund_rate from analytics > 0.5 → flag as high risk,
  reduce refund_amount by 20%
- If customer_risk_score > 0.8 → decision="denied",
  reason="high risk account"

INPUT:
- Customer: {customer_name} (Tier: {customer_tier})
- Customer ID: {customer_id}
- Risk Score: {customer_risk_score}
- Order ID: {order_id}
- Order Amount: ${order_amount}
- Order Date: {order_date}
- Product: {product_name}
- Product Category: {product_category}
- Refund Reason: {refund_reason}

Return ONLY this JSON, no other text:
{{
  "decision": "approved" or "denied" or "partial",
  "refund_amount": float,
  "policy_reason": str,
  "policy_applied": str,
  "policy_context": str,
  "refund_count_90_days": int,
  "eligibility_details": str
}}"""

COMMUNICATION_AGENT_PROMPT = """You are a Communication Agent for a customer refund processing system.
Save the decision, update analytics, and log the audit trail.

TOOL DEPENDENCY MAP:
- save_processed_request(order_id, customer_id, refund_request_id,
  decision, refund_amount, reason, policy_applied): NO dependencies
- update_customer_analytics(customer_id): NO dependencies
- update_refund_request_status(refund_request_id, status): NO dependencies
- log_audit_event(agent_name, tool_called, input_data,
  output_data, status, duration_ms): NO dependencies

EXECUTION RULES:
Turn 1: Call ALL four tools simultaneously in parallel:
        save_processed_request AND update_customer_analytics AND
        update_refund_request_status AND log_audit_event

All four are completely independent. Run them all at once.

INPUT:
- Customer ID: {customer_id}
- Customer Name: {customer_name}
- Order ID: {order_id}
- Refund Request ID: {refund_request_id}
- Decision: {decision}
- Refund Amount: ${refund_amount}
- Reason: {policy_reason}
- Policy Applied: {policy_applied}

Return ONLY this JSON, no other text:
{{
  "request_saved": true,
  "analytics_updated": true,
  "status_updated": true,
  "audit_logged": true,
  "confirmation_message": str
}}"""


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        # Check if already seeded
        result = await session.execute(select(Customer).limit(1))
        if result.scalar_one_or_none():
            return

        # Seed customers
        customers = [
            Customer(
                id=1, name="Alice Johnson", email="alice@example.com",
                tier="gold", phone="555-0101", total_orders=15,
                total_refunds=2, account_status="active"
            ),
            Customer(
                id=2, name="Bob Smith", email="bob@example.com",
                tier="silver", phone="555-0102", total_orders=8,
                total_refunds=3, account_status="active"
            ),
            Customer(
                id=3, name="Carol White", email="carol@example.com",
                tier="bronze", phone="555-0103", total_orders=3,
                total_refunds=1, account_status="active"
            ),
        ]
        session.add_all(customers)

        # Seed products
        products = [
            Product(
                id=1, name="Laptop Pro X", category="electronics",
                price=1299.99, return_window_days=15,
                restocking_fee_pct=0.15, stock_quantity=50
            ),
            Product(
                id=2, name="Wireless Mouse", category="electronics",
                price=49.99, return_window_days=15,
                restocking_fee_pct=0.15, stock_quantity=200
            ),
            Product(
                id=3, name="Office Chair", category="furniture",
                price=399.99, return_window_days=30,
                restocking_fee_pct=0.10, stock_quantity=75
            ),
        ]
        session.add_all(products)

        # Seed orders
        now = datetime.utcnow()
        orders = [
            Order(
                id=1, customer_id=1, product_id=1,
                purchase_date=now - timedelta(days=10),
                amount=1299.99, status="delivered", quantity=1,
                payment_method="credit_card",
                shipping_address="123 Main St, NY"
            ),
            Order(
                id=2, customer_id=2, product_id=2,
                purchase_date=now - timedelta(days=35),
                amount=49.99, status="delivered", quantity=1,
                payment_method="paypal",
                shipping_address="456 Oak Ave, CA"
            ),
            Order(
                id=3, customer_id=3, product_id=3,
                purchase_date=now - timedelta(days=5),
                amount=399.99, status="delivered", quantity=1,
                payment_method="credit_card",
                shipping_address="789 Pine Rd, TX"
            ),
        ]
        session.add_all(orders)

        # Seed customer analytics
        analytics = [
            CustomerAnalytics(
                id=1, customer_id=1, total_spent=19499.85,
                average_order_value=1299.99, refund_rate=0.13,
                risk_score=0.1, last_calculated_at=now
            ),
            CustomerAnalytics(
                id=2, customer_id=2, total_spent=399.92,
                average_order_value=49.99, refund_rate=0.375,
                risk_score=0.4, last_calculated_at=now
            ),
            CustomerAnalytics(
                id=3, customer_id=3, total_spent=1199.97,
                average_order_value=399.99, refund_rate=0.33,
                risk_score=0.3, last_calculated_at=now
            ),
        ]
        session.add_all(analytics)

        # Seed refund requests
        refund_requests = [
            RefundRequest(
                id=1, order_id=1, customer_id=1,
                reason="Product not as described", status="pending"
            ),
            RefundRequest(
                id=2, order_id=2, customer_id=2,
                reason="Item stopped working", status="pending"
            ),
            RefundRequest(
                id=3, order_id=3, customer_id=3,
                reason="Changed my mind", status="pending"
            ),
        ]
        session.add_all(refund_requests)

        # Seed prompt registry
        prompts = [
            PromptRegistryModel(
                prompt_name="supervisor", version="v1.0",
                content=SUPERVISOR_PROMPT, is_active=True,
                description="Supervisor routing prompt",
                created_by="system"
            ),
            PromptRegistryModel(
                prompt_name="validation_agent", version="v1.0",
                content=VALIDATION_AGENT_PROMPT, is_active=True,
                description="Validation agent prompt",
                created_by="system"
            ),
            PromptRegistryModel(
                prompt_name="policy_agent", version="v1.0",
                content=POLICY_AGENT_PROMPT, is_active=True,
                description="Policy agent prompt",
                created_by="system"
            ),
            PromptRegistryModel(
                prompt_name="communication_agent", version="v1.0",
                content=COMMUNICATION_AGENT_PROMPT, is_active=True,
                description="Communication agent prompt",
                created_by="system"
            ),
        ]
        session.add_all(prompts)

        await session.commit()

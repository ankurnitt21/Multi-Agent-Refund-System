import json
from datetime import datetime, timedelta

from sqlalchemy import select, func as sqlfunc
from sqlalchemy.exc import IntegrityError
from langchain_core.tools import tool

from database.db_setup import async_session_factory
from database.models import (
    Customer, Order, Product, RefundRequest,
    ProcessedRequest, CustomerAnalytics, AuditLog
)


@tool
async def lookup_customer(email: str) -> str:
    """Look up a customer by email address."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Customer).where(Customer.email == email)
            )
            customer = result.scalar_one_or_none()
            if not customer:
                return json.dumps({"found": False, "error": "Customer not found"})
            return json.dumps({
                "found": True,
                "id": customer.id,
                "name": customer.name,
                "email": customer.email,
                "tier": customer.tier,
                "phone": customer.phone,
                "total_orders": customer.total_orders,
                "total_refunds": customer.total_refunds,
                "account_status": customer.account_status,
            })
    except Exception as e:
        return json.dumps({"found": False, "error": str(e)})


@tool
async def get_order_details(order_id: int) -> str:
    """Get order details including product and customer information."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Order, Product, Customer)
                .join(Product, Order.product_id == Product.id)
                .join(Customer, Order.customer_id == Customer.id)
                .where(Order.id == order_id)
            )
            row = result.one_or_none()
            if not row:
                return json.dumps({"found": False, "error": "Order not found"})
            order, product, customer = row
            return json.dumps({
                "found": True,
                "order_id": order.id,
                "amount": order.amount,
                "status": order.status,
                "purchase_date": str(order.purchase_date),
                "quantity": order.quantity,
                "payment_method": order.payment_method,
                "shipping_address": order.shipping_address,
                "product_name": product.name,
                "product_category": product.category,
                "return_window_days": product.return_window_days,
                "restocking_fee_pct": product.restocking_fee_pct,
                "customer_name": customer.name,
                "customer_tier": customer.tier,
            })
    except Exception as e:
        return json.dumps({"found": False, "error": str(e)})


@tool
async def verify_order_ownership(order_id: int, customer_id: int) -> str:
    """Verify that an order belongs to a specific customer."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Order).where(
                    Order.id == order_id,
                    Order.customer_id == customer_id
                )
            )
            order = result.scalar_one_or_none()
            if not order:
                return json.dumps({"valid": False, "error": "Order does not belong to customer"})
            return json.dumps({
                "valid": True,
                "order_id": order.id,
                "status": order.status,
                "amount": order.amount,
            })
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e)})


@tool
async def get_customer_analytics(customer_id: int) -> str:
    """Get customer analytics data including risk score."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(CustomerAnalytics).where(
                    CustomerAnalytics.customer_id == customer_id
                )
            )
            analytics = result.scalar_one_or_none()
            if not analytics:
                return json.dumps({"found": False, "error": "Analytics not found"})
            return json.dumps({
                "found": True,
                "total_spent": analytics.total_spent,
                "average_order_value": analytics.average_order_value,
                "refund_rate": analytics.refund_rate,
                "risk_score": analytics.risk_score,
                "last_calculated_at": str(analytics.last_calculated_at) if analytics.last_calculated_at else None,
            })
    except Exception as e:
        return json.dumps({"found": False, "error": str(e)})


@tool
async def check_refund_history(customer_id: int) -> str:
    """Check customer's refund history in the last 90 days and current year."""
    try:
        async with async_session_factory() as session:
            async with session.begin():  # single snapshot for both queries — prevents read skew
                now = datetime.utcnow()
                ninety_days_ago = now - timedelta(days=90)
                year_start = datetime(now.year, 1, 1)

                # Last 90 days
                result_90 = await session.execute(
                    select(ProcessedRequest).where(
                        ProcessedRequest.customer_id == customer_id,
                        ProcessedRequest.processed_at >= ninety_days_ago,
                    )
                )
                refunds_90 = result_90.scalars().all()

                # This year
                result_year = await session.execute(
                    select(ProcessedRequest).where(
                        ProcessedRequest.customer_id == customer_id,
                        ProcessedRequest.processed_at >= year_start,
                    )
                )
                refunds_year = result_year.scalars().all()

                last_refund_date = None
                refund_amounts = []
                for r in refunds_90:
                    refund_amounts.append(r.refund_amount)
                    if r.processed_at:
                        if not last_refund_date or r.processed_at > last_refund_date:
                            last_refund_date = r.processed_at

                return json.dumps({
                    "refund_count_90_days": len(refunds_90),
                    "refund_count_year": len(refunds_year),
                    "last_refund_date": str(last_refund_date) if last_refund_date else None,
                    "refund_amounts": refund_amounts,
                })
    except Exception as e:
        return json.dumps({"refund_count_90_days": 0, "error": str(e)})


@tool
async def save_processed_request(
    order_id: int, customer_id: int, refund_request_id: int,
    decision: str, refund_amount: float, reason: str, policy_applied: str
) -> str:
    """Save a processed refund request to the database (upsert — skips if already exists)."""
    try:
        async with async_session_factory() as session:
            async with session.begin():                   # outer transaction
                # ── Dedup check (read inside same tx) ────────────────
                existing = await session.execute(
                    select(ProcessedRequest).where(
                        ProcessedRequest.refund_request_id == refund_request_id
                    )
                )
                record = existing.scalar_one_or_none()
                if record is not None:
                    return json.dumps({
                        "saved": True,
                        "record_id": record.id,
                        "deduplicated": True,
                    })

                # ── Insert guarded by SAVEPOINT ───────────────────────
                # begin_nested() creates a Postgres SAVEPOINT so that an
                # IntegrityError on the INSERT rolls back only the savepoint,
                # leaving the outer transaction alive for the recovery SELECT.
                new_record = ProcessedRequest(
                    order_id=order_id,
                    customer_id=customer_id,
                    refund_request_id=refund_request_id,
                    decision=decision,
                    refund_amount=refund_amount,
                    reason=reason,
                    policy_applied=policy_applied,
                )
                session.add(new_record)
                try:
                    async with session.begin_nested():   # SAVEPOINT
                        await session.flush()            # send INSERT within savepoint
                    # SAVEPOINT released — INSERT will commit with outer tx
                except IntegrityError:
                    # Savepoint rolled back; outer tx still alive.
                    # Race condition: another request inserted this refund_request_id first.
                    winner = await session.execute(
                        select(ProcessedRequest).where(
                            ProcessedRequest.refund_request_id == refund_request_id
                        )
                    )
                    record = winner.scalar_one()
                    return json.dumps({
                        "saved": True,
                        "record_id": record.id,
                        "deduplicated": True,
                    })
            # outer tx auto-committed here
            return json.dumps({"saved": True, "record_id": new_record.id, "deduplicated": False})
    except Exception as e:
        return json.dumps({"saved": False, "error": str(e)})


@tool
async def update_customer_analytics(customer_id: int, refund_request_id: int = 0) -> str:
    """Recalculate and update customer analytics (idempotent per refund_request_id)."""
    try:
        async with async_session_factory() as session:
            # Everything — idempotency check, reads, writes, flag update — in ONE transaction.
            # This eliminates the previous double-commit: if the process crashes mid-way,
            # the whole operation rolls back and can safely re-run.
            async with session.begin():
                # ── Idempotency: skip if already updated for this request ──
                pr: ProcessedRequest | None = None
                if refund_request_id:
                    pr_result = await session.execute(
                        select(ProcessedRequest).where(
                            ProcessedRequest.refund_request_id == refund_request_id
                        )
                    )
                    pr = pr_result.scalar_one_or_none()
                    if pr and pr.analytics_updated:
                        return json.dumps({
                            "updated": True,
                            "skipped": True,
                            "reason": "Analytics already updated for this request",
                        })

                # ── Read customer — FOR UPDATE prevents concurrent lost-update
                # on total_refunds when two requests for the same customer run
                # simultaneously (both would read the same value and both write +1).
                cust_result = await session.execute(
                    select(Customer).where(Customer.id == customer_id).with_for_update()
                )
                customer = cust_result.scalar_one_or_none()
                if not customer:
                    return json.dumps({"updated": False, "error": "Customer not found"})

                # ── Calculate new values ─────────────────────────────────
                spent_result = await session.execute(
                    select(sqlfunc.sum(Order.amount)).where(Order.customer_id == customer_id)
                )
                total_spent = spent_result.scalar() or 0.0
                total_orders = customer.total_orders or 1
                total_refunds = customer.total_refunds + 1
                average_order_value = total_spent / total_orders if total_orders > 0 else 0.0
                refund_rate = total_refunds / total_orders if total_orders > 0 else 0.0
                risk_score = min(refund_rate * 1.5, 1.0)

                # ── Apply all writes (tracked objects, flushed on commit) ─
                analytics_result = await session.execute(
                    select(CustomerAnalytics).where(
                        CustomerAnalytics.customer_id == customer_id
                    )
                )
                analytics = analytics_result.scalar_one_or_none()
                if analytics:
                    analytics.total_spent = total_spent
                    analytics.average_order_value = average_order_value
                    analytics.refund_rate = refund_rate
                    analytics.risk_score = risk_score
                    analytics.last_calculated_at = datetime.utcnow()

                customer.total_refunds = total_refunds

                # ── Mark analytics updated — same commit, no second tx ───
                if pr is not None:
                    pr.analytics_updated = True

            # single auto-commit here — all writes land together
            return json.dumps({
                "updated": True,
                "new_risk_score": risk_score,
                "new_refund_rate": refund_rate,
            })
    except Exception as e:
        return json.dumps({"updated": False, "error": str(e)})


@tool
async def update_refund_request_status(refund_request_id: int, status: str) -> str:
    """Update the status of a refund request."""
    try:
        async with async_session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(RefundRequest).where(RefundRequest.id == refund_request_id)
                )
                request = result.scalar_one_or_none()
                if not request:
                    return json.dumps({"updated": False, "error": "Refund request not found"})
                request.status = status
            # auto-commit
            return json.dumps({
                "updated": True,
                "refund_request_id": refund_request_id,
                "new_status": status,
            })
    except Exception as e:
        return json.dumps({"updated": False, "error": str(e)})


@tool
async def log_audit_event(
    agent_name: str, tool_called: str, input_data: dict,
    output_data: dict, status: str, duration_ms: float
) -> str:
    """Log an audit event to the audit_logs table."""
    try:
        async with async_session_factory() as session:
            async with session.begin():
                log = AuditLog(
                    agent_name=agent_name,
                    tool_called=tool_called,
                    input_data=input_data,
                    output_data=output_data,
                    status=status,
                    duration_ms=duration_ms,
                )
                session.add(log)
                await session.flush()   # get log.id before commit
            # auto-commit
            return json.dumps({"logged": True, "log_id": log.id})
    except Exception as e:
        return json.dumps({"logged": False, "error": str(e)})

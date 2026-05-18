import json

from sqlalchemy import select, func as sqlfunc
from langchain_core.tools import tool

from database.db_setup import async_session_factory
from database.models import Order, Product, Customer, ProcessedRequest


@tool
async def get_customer_order_summary(customer_id: int) -> str:
    """Get a summary of all orders for a customer."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Order, Product)
                .join(Product, Order.product_id == Product.id)
                .where(Order.customer_id == customer_id)
            )
            rows = result.all()
            if not rows:
                return json.dumps({"total_orders": 0, "error": "No orders found"})

            total_spent = 0.0
            statuses = {}
            categories = set()
            for order, product in rows:
                total_spent += order.amount
                statuses[order.status] = statuses.get(order.status, 0) + 1
                categories.add(product.category)

            return json.dumps({
                "total_orders": len(rows),
                "total_spent": round(total_spent, 2),
                "order_statuses": statuses,
                "product_categories_bought": list(categories),
                "average_order_value": round(total_spent / len(rows), 2),
            })
    except Exception as e:
        return json.dumps({"total_orders": 0, "error": str(e)})


@tool
async def get_refund_rate_by_category(product_category: str) -> str:
    """Get refund statistics for a specific product category."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(ProcessedRequest, Order, Product)
                .join(Order, ProcessedRequest.order_id == Order.id)
                .join(Product, Order.product_id == Product.id)
                .where(Product.category == product_category)
            )
            rows = result.all()
            if not rows:
                return json.dumps({
                    "category": product_category,
                    "total_requests": 0,
                    "approved": 0,
                    "denied": 0,
                    "approval_rate": 0.0,
                    "average_refund_amount": 0.0,
                })

            approved = sum(1 for pr, _, _ in rows if pr.decision == "approved")
            denied = sum(1 for pr, _, _ in rows if pr.decision == "denied")
            total_refund = sum(pr.refund_amount for pr, _, _ in rows if pr.refund_amount)
            approval_rate = approved / len(rows) if rows else 0.0

            return json.dumps({
                "category": product_category,
                "total_requests": len(rows),
                "approved": approved,
                "denied": denied,
                "approval_rate": round(approval_rate, 2),
                "average_refund_amount": round(total_refund / len(rows), 2) if rows else 0.0,
            })
    except Exception as e:
        return json.dumps({"category": product_category, "error": str(e)})


@tool
async def get_similar_refund_decisions(product_category: str, customer_tier: str) -> str:
    """Get similar past refund decisions for a product category and customer tier."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(ProcessedRequest, Order, Product, Customer)
                .join(Order, ProcessedRequest.order_id == Order.id)
                .join(Product, Order.product_id == Product.id)
                .join(Customer, ProcessedRequest.customer_id == Customer.id)
                .where(Product.category == product_category, Customer.tier == customer_tier)
                .order_by(ProcessedRequest.processed_at.desc())
                .limit(10)
            )
            rows = result.all()
            if not rows:
                return json.dumps({
                    "similar_cases": 0,
                    "decisions_breakdown": {},
                    "average_approved_amount": 0.0,
                    "most_common_decision": "none",
                })

            decisions = {}
            approved_amounts = []
            for pr, order, product, customer in rows:
                decisions[pr.decision] = decisions.get(pr.decision, 0) + 1
                if pr.decision == "approved" and pr.refund_amount:
                    approved_amounts.append(pr.refund_amount)

            most_common = max(decisions, key=decisions.get) if decisions else "none"
            avg_approved = (
                round(sum(approved_amounts) / len(approved_amounts), 2)
                if approved_amounts else 0.0
            )

            return json.dumps({
                "similar_cases": len(rows),
                "decisions_breakdown": decisions,
                "average_approved_amount": avg_approved,
                "most_common_decision": most_common,
            })
    except Exception as e:
        return json.dumps({"similar_cases": 0, "error": str(e)})

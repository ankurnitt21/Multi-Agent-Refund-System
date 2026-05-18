import json
from datetime import datetime

from sqlalchemy import select
from langchain_core.tools import tool

from database.db_setup import async_session_factory
from database.models import Order, Product, Customer
from vectordb.pinecone_store import PineconeStore


_pinecone_store: PineconeStore | None = None


def get_pinecone_store() -> PineconeStore:
    global _pinecone_store
    if _pinecone_store is None:
        _pinecone_store = PineconeStore()
    return _pinecone_store


def set_pinecone_store(store: PineconeStore):
    global _pinecone_store
    _pinecone_store = store


@tool
async def search_policies(query: str) -> str:
    """Search refund policies using vector similarity search with semantic caching."""
    try:
        store = get_pinecone_store()
        results = await store.search(query, top_k=3)
        cache_hit = False
        cached = await store.cache.get(query)
        if cached:
            cache_hit = True
        return json.dumps({
            "policies_found": len(results),
            "policies": results,
            "cache_hit": cache_hit,
        })
    except Exception as e:
        return json.dumps({"policies_found": 0, "error": str(e)})


@tool
async def check_eligibility(order_id: int, customer_tier: str) -> str:
    """Check if an order is eligible for a refund based on return window and tier."""
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
                return json.dumps({"eligible": False, "reason": "Order not found"})
            order, product, customer = row

            days_since_purchase = (datetime.utcnow() - order.purchase_date).days
            base_window = product.return_window_days

            tier_extension = 0
            if customer_tier == "gold":
                tier_extension = 15
            elif customer_tier == "silver":
                tier_extension = 7

            return_window = base_window + tier_extension
            eligible = days_since_purchase <= return_window

            reason = "Within return window" if eligible else f"Past return window ({days_since_purchase} days > {return_window} days)"

            return json.dumps({
                "eligible": eligible,
                "days_since_purchase": days_since_purchase,
                "return_window": return_window,
                "tier_extension_days": tier_extension,
                "base_window": base_window,
                "reason": reason,
            })
    except Exception as e:
        return json.dumps({"eligible": False, "error": str(e)})


@tool
async def get_product_return_policy(product_category: str) -> str:
    """Get the return policy for a specific warehouse item category."""
    policies = {
        "electronics": {
            "window": 15,
            "restocking_fee": 0.15,
            "notes": "Original packaging required; must be unopened for full refund",
        },
        "industrial_equipment": {
            "window": 60,
            "restocking_fee": 0.10,
            "notes": "Equipment must be unused and in original crating",
        },
        "raw_materials": {
            "window": 30,
            "restocking_fee": 0.05,
            "notes": "Applicable for wrong SKU or quality defect claims only",
        },
        "chemicals": {
            "window": 7,
            "restocking_fee": 0.20,
            "notes": "Hazmat compliance required; safety data sheet must accompany return",
        },
        "food_perishables": {
            "window": 3,
            "restocking_fee": 0.0,
            "notes": "Perishables refunded only for damaged or wrong-item shipments",
        },
        "packaging": {
            "window": 30,
            "restocking_fee": 0.05,
            "notes": "Unused and unopened packaging materials only",
        },
        "furniture": {
            "window": 30,
            "restocking_fee": 0.10,
            "notes": "Assembly voids return policy",
        },
    }
    default_policy = {"window": 30, "restocking_fee": 0.0, "notes": "Standard warehouse policy"}
    policy = policies.get(product_category.lower(), default_policy)
    return json.dumps(policy)


@tool
async def calculate_refund(order_id: int, customer_tier: str, eligibility_result: str) -> str:
    """Calculate the refund amount based on eligibility and tier benefits."""
    try:
        eligibility = json.loads(eligibility_result)
        if not eligibility.get("eligible", False):
            return json.dumps({
                "refund_amount": 0,
                "reason": "not eligible",
                "base_amount": 0,
                "restocking_fee": 0,
                "restocking_fee_pct_applied": 0,
                "final_refund_amount": 0,
                "tier_benefit_applied": "none",
            })

        async with async_session_factory() as session:
            result = await session.execute(
                select(Order, Product)
                .join(Product, Order.product_id == Product.id)
                .where(Order.id == order_id)
            )
            row = result.one_or_none()
            if not row:
                return json.dumps({"refund_amount": 0, "reason": "Order not found"})
            order, product = row

            amount = order.amount
            restocking_fee_pct = product.restocking_fee_pct

            tier_benefit = "none"
            if customer_tier == "gold":
                restocking_fee_pct = 0.0
                tier_benefit = "gold - full restocking fee waiver"
            elif customer_tier == "silver":
                restocking_fee_pct *= 0.5
                tier_benefit = "silver - 50% restocking fee waiver"
            else:
                tier_benefit = "bronze - no waiver"

            restocking_fee = amount * restocking_fee_pct
            final_refund = round(amount - restocking_fee, 2)

            return json.dumps({
                "base_amount": amount,
                "restocking_fee": restocking_fee,
                "restocking_fee_pct_applied": restocking_fee_pct,
                "final_refund_amount": final_refund,
                "tier_benefit_applied": tier_benefit,
            })
    except Exception as e:
        return json.dumps({"refund_amount": 0, "error": str(e)})

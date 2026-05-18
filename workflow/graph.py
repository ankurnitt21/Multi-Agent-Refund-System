import asyncio
import sys

import structlog
from redis.asyncio import Redis

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage
from opentelemetry import trace

from agents.state import RefundState
from agents.supervisor import supervisor_node
from agents.validation_agent import validation_agent_node
from agents.policy_agent import policy_agent_node
from agents.communication_agent import communication_agent_node
from guardrails.input_sanitizer import sanitize_input
from guardrails.pii_handler import mask_pii
from checkpoint.store import save_checkpoint, load_checkpoint, delete_checkpoint
from database.db_setup import async_session_factory
from database.models import HITLTask
from config import DATABASE_URL, REDIS_URL
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)

# Module-level Redis client used by checkpoint node wrappers.
# Initialised in init_checkpointer(); wrappers skip saving when None.
_checkpoint_redis: Redis | None = None
_CHECKPOINT_DB_URL = DATABASE_URL.replace("+asyncpg", "")


# HITL node — triggered when cycle/retry limits or ReAct max iterations hit.
# Saves the full state to HITLTask DB table so a human can review via API.
async def hitl_node(state: RefundState) -> dict:
    """Persist the current state as a HITL task and end the workflow run.

    A human must call POST /hitl/tasks/{task_id}/resolve to continue:
      - approve     → reset counters, re-run workflow from checkpoint
      - deny        → write denied result, mark done
      - compensate  → write compensated result, mark done
    """
    with tracer.start_as_current_span("hitl_node") as span:
        span.set_attribute("langsmith.span.kind", "chain")
        span.set_attribute("agent.name", "hitl")

        task_id = state.get("thread_id")
        reason = state.get("hitl_reason", "unknown")
        span.set_attribute("hitl.reason", reason)
        span.set_attribute("hitl.task_id", str(task_id))

        logger.warning("hitl_triggered", task_id=task_id, reason=reason)

        # Persist to DB (best-effort — failure must not crash the workflow)
        try:
            from checkpoint.store import _serialize_state
            async with async_session_factory() as session:
                async with session.begin():
                    # Upsert: if task already in HITL (e.g. double-trigger), update it
                    from sqlalchemy.dialects.postgresql import insert as pg_insert
                    stmt = (
                        pg_insert(HITLTask)
                        .values(
                            task_id=task_id,
                            reason=reason,
                            state_json=_serialize_state(state),
                            status="pending",
                        )
                        .on_conflict_do_update(
                            index_elements=["task_id"],
                            set_={
                                "reason": reason,
                                "state_json": _serialize_state(state),
                                "status": "pending",
                            },
                        )
                    )
                    await session.execute(stmt)
            logger.info("hitl_task_saved", task_id=task_id)
        except Exception as exc:
            logger.error("hitl_task_save_failed", task_id=task_id, error=str(exc))

        return {
            "hitl_required": True,
            "request_saved": True,   # stop supervisor from looping
            "next_agent": "end",
            "messages": [AIMessage(
                content=f"HITL required for task {task_id}: {reason}. "
                        f"Queued for human review."
            )],
        }


# ---------------------------------------------------------------------------
# Checkpoint-aware node wrappers
# Each wrapper calls the real agent, then upserts state to Redis + Postgres.
# The save is best-effort: failures are logged but never raise.
# ---------------------------------------------------------------------------
async def _save(agent_name: str, state: RefundState, result: dict) -> None:
    """Merge result into state and persist a checkpoint (best-effort)."""
    thread_id = state.get("thread_id")
    if not thread_id or not _checkpoint_redis:
        return
    merged = {**state, **result}
    await save_checkpoint(_checkpoint_redis, thread_id, agent_name, merged)


async def _validation_node(state: RefundState) -> dict:
    result = await validation_agent_node(state)
    await _save("validation", state, result)
    return result


async def _policy_node(state: RefundState) -> dict:
    result = await policy_agent_node(state)
    await _save("policy", state, result)
    return result


async def _communication_node(state: RefundState) -> dict:
    result = await communication_agent_node(state)
    await _save("communication", state, result)
    return result


async def _hitl_node(state: RefundState) -> dict:
    # No checkpoint save here — checkpoint kept so approve action can resume.
    return await hitl_node(state)


def route_supervisor(state: RefundState) -> str:
    next_agent = state.get("next_agent", "end")
    if next_agent == "end":
        return END
    return next_agent


def _build_workflow() -> StateGraph:
    wf = StateGraph(RefundState)

    wf.add_node("supervisor", supervisor_node)
    wf.add_node("validation", _validation_node)
    wf.add_node("policy", _policy_node)
    wf.add_node("communication", _communication_node)
    wf.add_node("hitl", _hitl_node)

    wf.add_edge(START, "supervisor")

    wf.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "validation": "validation",
            "policy": "policy",
            "communication": "communication",
            "hitl": "hitl",
            END: END,
        },
    )

    wf.add_edge("validation", "supervisor")
    wf.add_edge("policy", "supervisor")
    wf.add_edge("communication", "supervisor")
    wf.add_edge("hitl", END)

    return wf


_workflow_builder = _build_workflow()

# Module-level compiled graph (without checkpointer, used as fallback)
refund_workflow = _workflow_builder.compile()

# Will be set during startup with the checkpointer-backed version
_checkpointed_workflow = None
_checkpoint_pool = None


async def init_checkpointer():
    """Set up PostgreSQL LangGraph checkpointer and Redis client. Called once at startup."""
    global _checkpointed_workflow, _checkpoint_redis, _checkpoint_pool

    try:
        _checkpoint_redis = Redis.from_url(REDIS_URL)
        logger.info("checkpoint_redis_ready")
    except Exception as e:
        logger.warning("checkpoint_redis_failed", error=str(e))
        _checkpoint_redis = None

    if sys.platform == "win32":
        logger.info("checkpointer_skipped", reason="Windows SelectorEventLoop + uvicorn compatibility")
        _checkpointed_workflow = None
        _checkpoint_pool = None
    else:
        try:
            from psycopg_pool import AsyncConnectionPool

            _checkpoint_pool = AsyncConnectionPool(
                conninfo=_CHECKPOINT_DB_URL,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            checkpointer = AsyncPostgresSaver(conn=_checkpoint_pool)
            await checkpointer.setup()
            _checkpointed_workflow = _build_workflow().compile(checkpointer=checkpointer)
            logger.info("checkpointer_ready", backend="postgresql")
        except Exception as e:
            logger.warning("checkpointer_failed_fallback", error=str(e), fallback="in-memory")
            _checkpointed_workflow = None
            _checkpoint_pool = None


MAX_RECURSION = 50  # Safety limit


async def run_workflow(
    customer_email: str,
    order_id: int,
    refund_reason: str,
    *,
    thread_id: str | None = None,
    recovered_state: dict | None = None,
) -> dict:
    with tracer.start_as_current_span("warehouse_refund_workflow") as root_span:
        root_span.set_attribute("langsmith.span.kind", "chain")
        root_span.set_attribute("workflow.name", "warehouse_refund_workflow")
        root_span.set_attribute("order.id", order_id)
        root_span.set_attribute("customer.email", f"***{customer_email.split('@')[-1]}")

        graph = _checkpointed_workflow or refund_workflow

        config: dict = {"recursion_limit": MAX_RECURSION}
        if thread_id and _checkpointed_workflow:
            config["configurable"] = {"thread_id": thread_id}

        # Sanitize user input at workflow entry point
        safe_reason = sanitize_input(refund_reason, field_name="refund_reason")
        masked_email = mask_pii(customer_email)

        base_state: dict = {
            "customer_email": customer_email,
            "order_id": order_id,
            "refund_reason": safe_reason,
            "cycle_count": 0,
            "agent_retry_counts": {},
            "thread_id": thread_id,
        }

        if recovered_state is not None:
            # HITL approve: delete stale LangGraph checkpoint so we start fresh.
            if _checkpointed_workflow and thread_id:
                try:
                    await _checkpointed_workflow.checkpointer.adelete(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    logger.info("langgraph_checkpoint_cleared_for_hitl_resume", thread_id=thread_id)
                except Exception as exc:
                    logger.warning(
                        "langgraph_checkpoint_clear_failed",
                        thread_id=thread_id,
                        error=str(exc),
                    )
            base_state = {**recovered_state, "thread_id": thread_id}
        elif not _checkpointed_workflow and thread_id and _checkpoint_redis:
            existing = await load_checkpoint(_checkpoint_redis, thread_id)
            if existing:
                recovered = existing.get("state", {})
                last_agent = existing.get("last_completed_agent", "—")
                logger.info(
                    "checkpoint_resume",
                    thread_id=thread_id,
                    last_completed_agent=last_agent,
                )
                root_span.set_attribute("workflow.resumed_from", last_agent)
                base_state = {**recovered, "thread_id": thread_id}

        initial_state = {
            **base_state,
            "messages": [HumanMessage(content=(
                f"Process warehouse refund: email={masked_email} "
                f"order={order_id} reason={safe_reason}"
            ))],
        }

        result = await graph.ainvoke(initial_state, config)

        root_span.set_attribute("workflow.decision", result.get("decision", "unknown"))
        root_span.set_attribute("workflow.compensated", result.get("compensated", False))

        # Clean up checkpoint on successful finish.
        # Keep checkpoint when HITL is pending so approve can resume.
        workflow_done = result.get("request_saved") or result.get("compensated")
        hitl_pending = result.get("hitl_required")
        if workflow_done and not hitl_pending and thread_id and _checkpoint_redis:
            await delete_checkpoint(_checkpoint_redis, thread_id)

        return result

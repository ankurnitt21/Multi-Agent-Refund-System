import uuid
import json
import hashlib
import asyncio
import sys
from datetime import datetime

# psycopg async pool requires SelectorEventLoop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import structlog
from fastapi import FastAPI, Header
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select, update

from config import REDIS_URL
from database.db_setup import init_db, async_session_factory
from database.models import CustomerAnalytics, AuditLog, IdempotencyRecord, HITLTask, TaskResult
from prompts.registry import PromptRegistry
from vectordb.pinecone_store import PineconeStore
from tools.policy_tools import set_pinecone_store
from workflow.graph import run_workflow, init_checkpointer
from task_queue_store.kafka_events import KafkaRefundEventBus
from logging_config import setup_logging
from guardrails.input_sanitizer import sanitize_input, PromptInjectionError
from guardrails.runner import RefundGuardRunner
from evaluation.ragas_evaluator import RefundRAGASEvaluator

setup_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(title="Warehouse Refund Processing System", version="1.0.0")
redis = Redis.from_url(REDIS_URL)
prompt_registry = PromptRegistry()
event_bus = KafkaRefundEventBus()
guard_runner = RefundGuardRunner()
ragas_evaluator = RefundRAGASEvaluator()

# ── Idempotency key TTL (24 hours) ──
IDEMPOTENCY_TTL = 86400


class RefundRequest(BaseModel):
    customer_email: str
    order_id: int
    refund_reason: str
    idempotency_key: str | None = None


class ActivateVersionRequest(BaseModel):
    version: str


class CreateVersionRequest(BaseModel):
    version: str
    content: str
    description: str
    created_by: str


class HITLResolveRequest(BaseModel):
    action: str          # "approve" | "deny" | "compensate"
    resolved_by: str | None = None
    note: str | None = None


@app.on_event("startup")
async def startup():
    await init_db()
    await prompt_registry.load_active_prompts()
    prompt_registry.start_auto_refresh()
    store = PineconeStore()
    await store.build_policy_index()
    set_pinecone_store(store)

    # Initialise LangGraph PostgreSQL checkpointer
    await init_checkpointer()

    await event_bus.start_producer()
    await event_bus.start_consumer(_handle_kafka_event)

    incomplete = await event_bus.get_incomplete_tasks()
    if incomplete:
        logger.info(
            "kafka_recovery_note",
            incomplete_meta=len(incomplete),
            message="Uncommitted Kafka offsets will be redelivered by the consumer group",
        )

    asyncio.create_task(_run_periodic_cleanup())

    removed = await event_bus.cleanup()
    if removed:
        logger.info("startup_task_meta_cleanup", removed=removed)

    logger.info("startup_complete")


@app.on_event("shutdown")
async def shutdown():
    await event_bus.stop()


async def _handle_kafka_event(event: dict) -> None:
    """Kafka consumer handler — runs refund workflow for each event."""
    task_id = event["task_id"]
    await _process_refund(
        task_id,
        event["customer_email"],
        event["order_id"],
        event["refund_reason"],
        recovered_state=event.get("recovered_state"),
    )


async def _run_periodic_cleanup() -> None:
    """Remove stale task metadata from Redis every 6 hours."""
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            removed = await event_bus.cleanup()
            logger.info("periodic_queue_cleanup", removed=removed)
        except Exception as e:
            logger.warning("periodic_queue_cleanup_failed", error=str(e))


async def _update_idempotency_status(task_id: str, status: str) -> None:
    """Update IdempotencyRecord.status after task completes/fails/hitl."""
    try:
        async with async_session_factory() as session:
            await session.execute(
                update(IdempotencyRecord)
                .where(IdempotencyRecord.task_id == task_id)
                .values(status=status)
            )
            await session.commit()
    except Exception as e:
        logger.warning("idempotency_status_update_failed", task_id=task_id, error=str(e))


async def _persist_task_result(task_id: str, payload: dict) -> None:
    """Upsert the task outcome to TaskResult for permanent DB storage."""
    try:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        async with async_session_factory() as session:
            async with session.begin():
                stmt = (
                    pg_insert(TaskResult)
                    .values(task_id=task_id, **payload)
                    .on_conflict_do_update(
                        index_elements=["task_id"],
                        set_={**payload, "updated_at": datetime.utcnow()},
                    )
                )
                await session.execute(stmt)
    except Exception as e:
        logger.warning("task_result_persist_failed", task_id=task_id, error=str(e))


async def _process_refund(
    task_id: str,
    customer_email: str,
    order_id: int,
    refund_reason: str,
    *,
    recovered_state: dict | None = None,
):
    # Bind context vars — flow into every log call in this async task
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        task_id=task_id,
        order_id=order_id,
    )
    try:
        existing = await redis.get(f"task:{task_id}")
        if existing:
            prior = json.loads(existing)
            if prior.get("status") in ("completed", "hitl_pending"):
                logger.info("task_already_finished", task_id=task_id, status=prior.get("status"))
                return

        await event_bus.mark_processing(task_id)
        logger.info("task_started", customer_email=f"***{customer_email.split('@')[-1]}")

        # Guardrails-AI input validation (traced to LangSmith)
        guard_result = guard_runner.validate_input(
            customer_email, order_id, refund_reason
        )

        result = await run_workflow(
            customer_email, order_id, refund_reason,
            thread_id=task_id,
            recovered_state=recovered_state,
        )

        # ── HITL: workflow ended because human review is needed ───────
        if result.get("hitl_required"):
            logger.warning(
                "task_hitl_required",
                hitl_reason=result.get("hitl_reason"),
            )
            hitl_output = {
                "status": "hitl_pending",
                "hitl_reason": result.get("hitl_reason"),
                "task_id": task_id,
                "message": "Task requires human review. Use POST /hitl/tasks/{task_id}/resolve.",
            }
            await redis.set(f"task:{task_id}", json.dumps(hitl_output), ex=86400)
            await event_bus.mark_done(task_id, status="hitl_pending")
            await _update_idempotency_status(task_id, "hitl_pending")
            await _persist_task_result(task_id, {
                "status": "hitl_pending",
                "hitl_reason": result.get("hitl_reason"),
                "order_id": result.get("order_id"),
            })
            return

        logger.info(
            "task_completed",
            decision=result.get("decision"),
            refund_amount=result.get("refund_amount"),
            compensated=result.get("compensated", False),
        )

        # Guardrails-AI output validation (traced to LangSmith)
        if result.get("decision"):
            guard_runner.validate_policy_output(
                decision=result.get("decision", ""),
                refund_amount=result.get("refund_amount", 0.0),
                order_amount=result.get("order_amount"),
            )

        # RAGAS evaluation — runs metrics in parallel (traced to LangSmith)
        asyncio.create_task(_run_ragas_evaluation(
            task_id=task_id,
            refund_reason=refund_reason,
            result=result,
        ))
        output = {
            "status": "completed",
            "decision": result.get("decision"),
            "refund_amount": result.get("refund_amount"),
            "policy_reason": result.get("policy_reason"),
            "policy_applied": result.get("policy_applied"),
            "validation_passed": result.get("validation_passed"),
            "validation_reason": result.get("validation_reason"),
            "customer_name": result.get("customer_name"),
            "customer_tier": result.get("customer_tier"),
            "order_id": result.get("order_id"),
            "compensated": result.get("compensated", False),
        }
        await redis.set(f"task:{task_id}", json.dumps(output), ex=3600)
        await event_bus.mark_done(task_id, status="completed")
        await _update_idempotency_status(task_id, "completed")
        await _persist_task_result(task_id, {
            "status": "completed",
            "decision": result.get("decision"),
            "refund_amount": result.get("refund_amount"),
            "policy_reason": result.get("policy_reason"),
            "policy_applied": result.get("policy_applied"),
            "validation_passed": result.get("validation_passed"),
            "validation_reason": result.get("validation_reason"),
            "customer_name": result.get("customer_name"),
            "customer_tier": result.get("customer_tier"),
            "order_id": result.get("order_id"),
            "compensated": result.get("compensated", False),
        })
    except Exception as e:
        import traceback
        logger.error(
            "task_failed",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        error_result = {"status": "error", "error": str(e)}
        await redis.set(f"task:{task_id}", json.dumps(error_result), ex=3600)
        await event_bus.mark_done(task_id, status="failed")
        await _update_idempotency_status(task_id, "failed")
        await _persist_task_result(task_id, {
            "status": "error",
            "error": str(e),
        })


async def _run_ragas_evaluation(task_id: str, refund_reason: str, result: dict) -> None:
    """Run RAGAS evaluation in the background (non-blocking). Traces to LangSmith."""
    try:
        question = f"Process refund for order #{result.get('order_id')}: {refund_reason}"
        answer = (
            f"Decision: {result.get('decision')}, "
            f"Amount: ${result.get('refund_amount', 0.0)}, "
            f"Reason: {result.get('policy_reason', '')}"
        )
        contexts = [
            result.get("policy_applied", ""),
            result.get("policy_reason", ""),
            result.get("validation_reason", ""),
        ]
        contexts = [c for c in contexts if c]

        await ragas_evaluator.evaluate_workflow_run(
            task_id=task_id,
            question=question,
            answer=answer,
            contexts=contexts if contexts else ["No context available"],
        )
    except Exception as e:
        logger.warning("ragas_evaluation_failed", task_id=task_id, error=str(e))


@app.post("/refund")
async def create_refund(request: RefundRequest):
    # ── Input validation: reject obvious prompt injection attempts ─────
    try:
        sanitize_input(request.refund_reason, raise_on_injection=True, field_name="refund_reason")
    except PromptInjectionError as e:
        return {"status": "rejected", "error": str(e)}

    # ── Compute idempotency key ──────────────────────────────────────
    # Client may supply an explicit key; otherwise derive from payload.
    idem_key = request.idempotency_key or hashlib.sha256(
        f"{request.customer_email}:{request.order_id}:{request.refund_reason}".encode()
    ).hexdigest()

    # ── Layer 1: Redis fast-path check (sub-millisecond) ────────────
    cached_tid = await redis.get(f"idempotency:{idem_key}")
    if cached_tid:
        tid = cached_tid.decode() if isinstance(cached_tid, bytes) else cached_tid
        return {"task_id": tid, "status": "duplicate", "idempotency_key": idem_key}

    # ── Layer 2: DB fallback (Redis may have been flushed/restarted) ─
    async with async_session_factory() as session:
        db_record = await session.execute(
            select(IdempotencyRecord).where(
                IdempotencyRecord.idempotency_key == idem_key
            )
        )
        existing = db_record.scalar_one_or_none()
        if existing:
            # Warm Redis cache back up so next call is fast again
            await redis.set(
                f"idempotency:{idem_key}", existing.task_id, ex=IDEMPOTENCY_TTL
            )
            return {
                "task_id": existing.task_id,
                "status": "duplicate",
                "idempotency_key": idem_key,
                "recovered_from": "db",
            }

    task_id = str(uuid.uuid4())

    # ── Write to DB first (durable) ──────────────────────────────────
    async with async_session_factory() as session:
        record = IdempotencyRecord(
            idempotency_key=idem_key,
            task_id=task_id,
            status="processing",
        )
        session.add(record)
        try:
            await session.commit()
        except IntegrityError:
            # Race condition: another request inserted the same key first
            await session.rollback()
            dup = await session.execute(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.idempotency_key == idem_key
                )
            )
            existing = dup.scalar_one()
            await redis.set(
                f"idempotency:{idem_key}", existing.task_id, ex=IDEMPOTENCY_TTL
            )
            return {
                "task_id": existing.task_id,
                "status": "duplicate",
                "idempotency_key": idem_key,
            }

    # ── Write to Redis cache (fast subsequent lookups) ───────────────
    await redis.set(f"idempotency:{idem_key}", task_id, ex=IDEMPOTENCY_TTL)

    await event_bus.publish_refund_requested(task_id, {
        "customer_email": request.customer_email,
        "order_id": request.order_id,
        "refund_reason": request.refund_reason,
    })
    return {
        "task_id": task_id,
        "status": "processing",
        "idempotency_key": idem_key,
        "transport": "kafka",
        "topic": "refund.requests",
    }


@app.get("/refund/{task_id}")
async def get_refund_status(task_id: str):
    # Fast path: Redis (full result with decision, amount etc.)
    result = await redis.get(f"task:{task_id}")
    if result:
        return json.loads(result)

    # Fallback 1: TaskResult table — permanent, survives Redis TTL
    async with async_session_factory() as session:
        tr = await session.execute(
            select(TaskResult).where(TaskResult.task_id == task_id)
        )
        task_result = tr.scalar_one_or_none()
        if task_result:
            return {
                "status": task_result.status,
                "task_id": task_id,
                "decision": task_result.decision,
                "refund_amount": task_result.refund_amount,
                "policy_reason": task_result.policy_reason,
                "policy_applied": task_result.policy_applied,
                "validation_passed": task_result.validation_passed,
                "validation_reason": task_result.validation_reason,
                "customer_name": task_result.customer_name,
                "customer_tier": task_result.customer_tier,
                "order_id": task_result.order_id,
                "compensated": task_result.compensated,
                "hitl_reason": task_result.hitl_reason,
                "error": task_result.error,
                "source": "db",
            }

    # Fallback 2: IdempotencyRecord — status only, no detail
    async with async_session_factory() as session:
        db_record = await session.execute(
            select(IdempotencyRecord).where(IdempotencyRecord.task_id == task_id)
        )
        record = db_record.scalar_one_or_none()
        if record:
            return {"status": record.status, "task_id": task_id, "source": "idempotency_db"}

    return {"status": "not_found", "task_id": task_id}


@app.get("/hitl/tasks")
async def list_hitl_tasks(status: str = "pending"):
    """List HITL tasks awaiting human review."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(HITLTask)
            .where(HITLTask.status == status)
            .order_by(HITLTask.created_at.desc())
        )
        tasks = result.scalars().all()
        return [
            {
                "task_id": t.task_id,
                "reason": t.reason,
                "status": t.status,
                "created_at": str(t.created_at) if t.created_at else None,
                "resolved_at": str(t.resolved_at) if t.resolved_at else None,
                "resolved_by": t.resolved_by,
                "resolution_note": t.resolution_note,
                "state_snapshot": t.state_json,
            }
            for t in tasks
        ]


@app.post("/hitl/tasks/{task_id}/resolve")
async def resolve_hitl_task(
    task_id: str,
    request: HITLResolveRequest,
):
    """Resolve a HITL task.

    Actions:
    - deny        — mark denied, write final result
    - compensate  — mark compensated, write compensated result (refund_amount=0)
    - approve     — reset cycle/retry counters, re-run workflow from last checkpoint
    """
    if request.action not in ("approve", "deny", "compensate"):
        return {"error": "action must be one of: approve, deny, compensate"}
    async with async_session_factory() as session:
        result = await session.execute(
            select(HITLTask).where(HITLTask.task_id == task_id)
        )
        hitl_task = result.scalar_one_or_none()

    if not hitl_task:
        return {"error": f"HITL task {task_id} not found"}
    if hitl_task.status != "pending":
        return {"error": f"Task {task_id} already resolved (status: {hitl_task.status})"}

    now = datetime.utcnow()

    if request.action == "deny":
        output = {
            "status": "completed",
            "decision": "denied",
            "refund_amount": 0.0,
            "policy_reason": f"Denied by human reviewer: {request.note or 'no reason given'}",
            "hitl_resolved": True,
            "resolved_by": request.resolved_by,
        }
        await redis.set(f"task:{task_id}", json.dumps(output), ex=3600)
        await _update_idempotency_status(task_id, "completed")
        await _persist_task_result(task_id, {
            "status": "completed",
            "decision": "denied",
            "refund_amount": 0.0,
            "policy_reason": output["policy_reason"],
            "order_id": hitl_task.state_json.get("state", {}).get("order_id") if isinstance(hitl_task.state_json, dict) else None,
        })

    elif request.action == "compensate":
        output = {
            "status": "completed",
            "decision": "denied",
            "refund_amount": 0.0,
            "compensated": True,
            "policy_reason": f"Compensated by human reviewer: {request.note or 'no reason given'}",
            "hitl_resolved": True,
            "resolved_by": request.resolved_by,
        }
        await redis.set(f"task:{task_id}", json.dumps(output), ex=3600)
        await _update_idempotency_status(task_id, "completed")
        await _persist_task_result(task_id, {
            "status": "completed",
            "decision": "denied",
            "refund_amount": 0.0,
            "compensated": True,
            "policy_reason": output["policy_reason"],
            "order_id": hitl_task.state_json.get("state", {}).get("order_id") if isinstance(hitl_task.state_json, dict) else None,
        })

    elif request.action == "approve":
        # Load saved state, reset problematic counters, re-run workflow
        saved = hitl_task.state_json
        if isinstance(saved, dict) and "state" in saved:
            recovered = saved["state"]
        else:
            recovered = saved or {}

        # Reset counters so workflow doesn't immediately re-trigger HITL
        recovered = {
            **recovered,
            "cycle_count": 0,
            "agent_retry_counts": {},
            "hitl_required": None,
            "hitl_reason": None,
            "request_saved": None,  # allow communication agent to re-run if needed
        }

        customer_email = recovered.get("customer_email", "")
        order_id = recovered.get("order_id", 0)
        refund_reason = recovered.get("refund_reason", "")

        await _update_idempotency_status(task_id, "processing")
        await event_bus.publish_refund_requested(
            task_id,
            {
                "customer_email": customer_email,
                "order_id": order_id,
                "refund_reason": refund_reason,
            },
            recovered_state=recovered,
        )

        # Update HITL record before returning
        async with async_session_factory() as session:
            await session.execute(
                update(HITLTask)
                .where(HITLTask.task_id == task_id)
                .values(
                    status="approved",
                    resolved_at=now,
                    resolved_by=request.resolved_by,
                    resolution_note=request.note,
                )
            )
            await session.commit()

        return {
            "task_id": task_id,
            "action": "approve",
            "status": "reprocessing",
            "message": "Workflow re-submitted. Poll GET /refund/{task_id} for result.",
        }

    # For deny and compensate, update HITL record
    status_map = {"deny": "denied", "compensate": "compensated"}
    async with async_session_factory() as session:
        await session.execute(
            update(HITLTask)
            .where(HITLTask.task_id == task_id)
            .values(
                status=status_map[request.action],
                resolved_at=now,
                resolved_by=request.resolved_by,
                resolution_note=request.note,
            )
        )
        await session.commit()

    return {
        "task_id": task_id,
        "action": request.action,
        "status": "resolved",
    }



@app.get("/prompts/{prompt_name}")
async def get_prompt_versions(prompt_name: str):
    versions = await prompt_registry.list_versions(prompt_name)
    return {"prompt_name": prompt_name, "versions": versions}


@app.post("/prompts/{prompt_name}/activate")
async def activate_prompt_version(prompt_name: str, request: ActivateVersionRequest):
    await prompt_registry.activate_version(prompt_name, request.version)
    return {"activated": True, "prompt_name": prompt_name, "version": request.version}


@app.post("/prompts/{prompt_name}/version")
async def create_prompt_version(prompt_name: str, request: CreateVersionRequest):
    new_id = await prompt_registry.create_version(
        prompt_name, request.version, request.content,
        request.description, request.created_by
    )
    return {"created": True, "id": new_id}


@app.get("/analytics/customer/{customer_id}")
async def get_customer_analytics(customer_id: int):
    async with async_session_factory() as session:
        result = await session.execute(
            select(CustomerAnalytics).where(CustomerAnalytics.customer_id == customer_id)
        )
        analytics = result.scalar_one_or_none()
        if not analytics:
            return {"error": "Analytics not found"}
        return {
            "customer_id": customer_id,
            "total_spent": analytics.total_spent,
            "average_order_value": analytics.average_order_value,
            "refund_rate": analytics.refund_rate,
            "risk_score": analytics.risk_score,
            "last_calculated_at": str(analytics.last_calculated_at) if analytics.last_calculated_at else None,
        }


@app.get("/audit-logs")
async def get_audit_logs():
    async with async_session_factory() as session:
        result = await session.execute(
            select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50)
        )
        logs = result.scalars().all()
        return [
            {
                "id": log.id,
                "timestamp": str(log.timestamp) if log.timestamp else None,
                "agent_name": log.agent_name,
                "tool_called": log.tool_called,
                "input_data": log.input_data,
                "output_data": log.output_data,
                "status": log.status,
                "error_msg": log.error_msg,
                "duration_ms": log.duration_ms,
            }
            for log in logs
        ]


class EvalRequest(BaseModel):
    task_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@app.post("/evaluate")
async def run_evaluation(request: EvalRequest):
    """Run RAGAS evaluation on a completed task. Results traced to LangSmith."""
    report = await ragas_evaluator.evaluate_workflow_run(
        task_id=request.task_id,
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        ground_truth=request.ground_truth,
    )
    return report.to_dict()


# ── Resilience Endpoints ─────────────────────────────────────────────────────

from executor.resilience import get_fallback_chain


@app.get("/resilience/health")
async def resilience_health():
    """Get circuit breaker states and retry budget metrics for all models."""
    chain = get_fallback_chain()
    return chain.get_metrics()


# ── A/B Testing Endpoints ────────────────────────────────────────────────────

from evaluation.ab_testing import get_ab_manager, ExperimentConfig, Variant


class CreateExperimentRequest(BaseModel):
    experiment_id: str
    prompt_name: str
    variants: list[dict]  # [{name, content, traffic_percentage, is_control?}]
    min_samples_per_variant: int = 30


@app.post("/experiments")
async def create_experiment(request: CreateExperimentRequest):
    """Create a new A/B testing experiment for a prompt."""
    manager = get_ab_manager()
    variants = [
        Variant(
            name=v["name"],
            content=v["content"],
            traffic_percentage=v["traffic_percentage"],
            is_control=v.get("is_control", False),
        )
        for v in request.variants
    ]
    config = ExperimentConfig(
        experiment_id=request.experiment_id,
        prompt_name=request.prompt_name,
        variants=variants,
        min_samples_per_variant=request.min_samples_per_variant,
    )
    exp_id = manager.create_experiment(config)
    return {"experiment_id": exp_id, "status": "created"}


@app.get("/experiments")
async def list_experiments():
    """List all A/B testing experiments."""
    manager = get_ab_manager()
    return {"experiments": manager.list_experiments()}


@app.get("/experiments/{experiment_id}/report")
async def get_experiment_report(experiment_id: str):
    """Get statistical report for an A/B experiment."""
    manager = get_ab_manager()
    try:
        report = manager.get_report(experiment_id)
        return report.to_dict()
    except ValueError as e:
        return {"error": str(e)}


@app.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop a running experiment."""
    manager = get_ab_manager()
    manager.stop_experiment(experiment_id)
    return {"experiment_id": experiment_id, "status": "stopped"}


if __name__ == "__main__":
    import asyncio
    import sys

    import uvicorn

    # psycopg async pool requires SelectorEventLoop on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

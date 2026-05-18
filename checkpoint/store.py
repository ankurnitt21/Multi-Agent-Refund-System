"""
Explicit agent-level checkpoint store.

After every agent node completes, the workflow state (minus non-serialisable
message objects) is upserted to both Redis and PostgreSQL under the task's
thread_id.

On crash recovery:
  1. Redis is checked first (fast path, 24-hour TTL).
  2. If Redis is cold, PostgreSQL is used as the durable fallback.

The recovered state dict is injected as the initial state when re-invoking
the workflow, so the LangGraph supervisor sees the already-completed fields
(validation_passed, decision, etc.) and routes directly to the next pending
agent without re-running finished work.
"""
import json
from datetime import datetime

import structlog
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql import func

from database.db_setup import async_session_factory
from database.models import WorkflowCheckpoint

logger = structlog.get_logger(__name__)

CHECKPOINT_REDIS_PREFIX = "wf_checkpoint:"
CHECKPOINT_REDIS_TTL = 86_400          # 24 hours
_EXCLUDE_KEYS = frozenset({"messages"})  # BaseMessage objects are not JSON-safe


def _serialize_state(state: dict) -> dict:
    """Return a JSON-safe copy of state, dropping message history."""
    return {k: v for k, v in state.items() if k not in _EXCLUDE_KEYS}


async def save_checkpoint(
    redis: Redis,
    thread_id: str,
    agent_name: str,
    state: dict,
) -> None:
    """Upsert a checkpoint to Redis and PostgreSQL after an agent completes.

    Both writes are best-effort — a failure logs a warning but does not
    propagate so it never disrupts the workflow itself.
    """
    payload = {
        "thread_id": thread_id,
        "last_completed_agent": agent_name,
        "state": _serialize_state(state),
        "saved_at": datetime.utcnow().isoformat(),
    }

    # ── Redis (fast path) ──────────────────────────────────────────────
    try:
        await redis.set(
            f"{CHECKPOINT_REDIS_PREFIX}{thread_id}",
            json.dumps(payload),
            ex=CHECKPOINT_REDIS_TTL,
        )
    except Exception as exc:
        logger.warning("checkpoint_redis_write_failed", thread_id=thread_id, error=str(exc))

    # ── PostgreSQL (durable path — upsert on thread_id) ───────────────
    try:
        async with async_session_factory() as session:
            async with session.begin():
                stmt = (
                    pg_insert(WorkflowCheckpoint)
                    .values(
                        thread_id=thread_id,
                        last_completed_agent=agent_name,
                        state_json=payload,
                    )
                    .on_conflict_do_update(
                        index_elements=["thread_id"],
                        set_={
                            "last_completed_agent": agent_name,
                            "state_json": payload,
                            "saved_at": func.now(),
                        },
                    )
                )
                await session.execute(stmt)
        logger.info("checkpoint_saved", thread_id=thread_id, agent=agent_name)
    except Exception as exc:
        logger.warning("checkpoint_db_write_failed", thread_id=thread_id, error=str(exc))


async def load_checkpoint(redis: Redis, thread_id: str) -> dict | None:
    """Load the last saved checkpoint for a thread_id.

    Tries Redis first; falls back to PostgreSQL for cold-start recovery.
    Returns the full payload dict (keys: thread_id, last_completed_agent,
    state, saved_at) or None if no checkpoint exists.
    """
    # ── Redis (fast path) ──────────────────────────────────────────────
    try:
        raw = await redis.get(f"{CHECKPOINT_REDIS_PREFIX}{thread_id}")
        if raw:
            logger.info("checkpoint_loaded_from_redis", thread_id=thread_id)
            return json.loads(raw)
    except Exception as exc:
        logger.warning("checkpoint_redis_read_failed", thread_id=thread_id, error=str(exc))

    # ── PostgreSQL (cold-start fallback) ──────────────────────────────
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(WorkflowCheckpoint).where(WorkflowCheckpoint.thread_id == thread_id)
            )
            record = result.scalar_one_or_none()
            if record:
                logger.info("checkpoint_loaded_from_db", thread_id=thread_id)
                return record.state_json
    except Exception as exc:
        logger.warning("checkpoint_db_read_failed", thread_id=thread_id, error=str(exc))

    return None


async def delete_checkpoint(redis: Redis, thread_id: str) -> None:
    """Remove the checkpoint once the workflow finishes successfully.

    Prevents a stale checkpoint from being replayed on the next fresh request
    that happens to reuse the same thread_id (shouldn't happen with UUIDs,
    but is a safe guard).
    """
    try:
        await redis.delete(f"{CHECKPOINT_REDIS_PREFIX}{thread_id}")
    except Exception as exc:
        logger.warning("checkpoint_redis_delete_failed", thread_id=thread_id, error=str(exc))

    try:
        async with async_session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(WorkflowCheckpoint).where(WorkflowCheckpoint.thread_id == thread_id)
                )
                record = result.scalar_one_or_none()
                if record:
                    await session.delete(record)
        logger.info("checkpoint_deleted", thread_id=thread_id)
    except Exception as exc:
        logger.warning("checkpoint_db_delete_failed", thread_id=thread_id, error=str(exc))

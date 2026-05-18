"""
Redis-backed persistent task queue for crash recovery.

Every refund task is persisted in a Redis hash *before* processing begins.
On application startup, any tasks still marked "pending" or "processing"
are re-enqueued automatically so work is never silently lost.
"""
import json
from datetime import datetime

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)


class PersistentTaskQueue:
    TASKS_KEY = "refund:tasks"            # Hash:  task_id → JSON payload
    PENDING_SET = "refund:tasks:pending"  # Set of task_ids not yet completed

    def __init__(self, redis: Redis):
        self.redis = redis

    # ── Enqueue ──────────────────────────────────────────────────────
    async def enqueue(self, task_id: str, payload: dict) -> None:
        """Persist the task payload and mark it as pending."""
        payload_with_meta = {
            **payload,
            "task_id": task_id,
            "status": "pending",
            "enqueued_at": datetime.utcnow().isoformat(),
        }
        pipe = self.redis.pipeline()
        pipe.hset(self.TASKS_KEY, task_id, json.dumps(payload_with_meta))
        pipe.sadd(self.PENDING_SET, task_id)
        await pipe.execute()
        logger.info("task_enqueued", task_id=task_id)

    # ── Mark in-progress ─────────────────────────────────────────────
    async def mark_processing(self, task_id: str) -> None:
        raw = await self.redis.hget(self.TASKS_KEY, task_id)
        if raw:
            data = json.loads(raw)
            data["status"] = "processing"
            await self.redis.hset(self.TASKS_KEY, task_id, json.dumps(data))

    # ── Mark completed / failed ──────────────────────────────────────
    async def mark_done(self, task_id: str, status: str = "completed") -> None:
        """Remove from pending set and update status."""
        raw = await self.redis.hget(self.TASKS_KEY, task_id)
        if raw:
            data = json.loads(raw)
            data["status"] = status
            data["finished_at"] = datetime.utcnow().isoformat()
            await self.redis.hset(self.TASKS_KEY, task_id, json.dumps(data))
        await self.redis.srem(self.PENDING_SET, task_id)
        logger.info("task_status_changed", task_id=task_id, status=status)

    # ── Recovery: get all incomplete tasks ────────────────────────────
    async def get_incomplete_tasks(self) -> list[dict]:
        """Return payloads for all tasks that never completed."""
        task_ids = await self.redis.smembers(self.PENDING_SET)
        tasks = []
        for tid_bytes in task_ids:
            tid = tid_bytes.decode() if isinstance(tid_bytes, bytes) else tid_bytes
            raw = await self.redis.hget(self.TASKS_KEY, tid)
            if raw:
                tasks.append(json.loads(raw))
        return tasks

    # ── Cleanup old completed tasks (call periodically) ──────────────
    async def cleanup(self, keep_seconds: int = 86400) -> int:
        """Remove completed tasks older than keep_seconds."""
        all_tasks = await self.redis.hgetall(self.TASKS_KEY)
        cutoff = datetime.utcnow().timestamp() - keep_seconds
        removed = 0
        for tid_bytes, raw in all_tasks.items():
            tid = tid_bytes.decode() if isinstance(tid_bytes, bytes) else tid_bytes
            data = json.loads(raw)
            if data.get("status") in ("completed", "failed"):
                finished = data.get("finished_at")
                if finished:
                    finished_ts = datetime.fromisoformat(finished).timestamp()
                    if finished_ts < cutoff:
                        await self.redis.hdel(self.TASKS_KEY, tid)
                        removed += 1
        return removed

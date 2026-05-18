"""
Kafka-backed event bus for refund task orchestration.

Architecture
------------
- API publishes `RefundRequested` events to `refund.requests`.
- An in-process consumer group (`refund-workers`) processes events and runs the workflow.
- Offsets commit only after successful handling → crash recovery via Kafka redelivery.
- Redis `refund:task_meta` hash tracks task status for observability (not the work queue).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Awaitable, Callable

import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
from redis.asyncio import Redis

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    KAFKA_TOPIC_REFUND_REQUESTS,
    REDIS_URL,
)

logger = structlog.get_logger(__name__)

ProcessHandler = Callable[[dict], Awaitable[None]]

TASK_META_KEY = "refund:task_meta"


class KafkaRefundEventBus:
    """Publish refund work to Kafka; consume with at-least-once delivery."""

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None
        self._consumer: AIOKafkaConsumer | None = None
        self._redis = Redis.from_url(REDIS_URL)
        self._consumer_task: asyncio.Task | None = None
        self._handler: ProcessHandler | None = None
        self._running = False

    async def start_producer(self) -> None:
        if self._producer is not None:
            return
        self._producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        await self._producer.start()
        logger.info("kafka_producer_ready", brokers=KAFKA_BOOTSTRAP_SERVERS)

    async def start_consumer(self, handler: ProcessHandler) -> None:
        """Start background consumer loop."""
        self._handler = handler
        if self._consumer_task and not self._consumer_task.done():
            return
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info(
            "kafka_consumer_started",
            topic=KAFKA_TOPIC_REFUND_REQUESTS,
            group=KAFKA_CONSUMER_GROUP,
        )

    async def stop(self) -> None:
        self._running = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
        if self._producer:
            await self._producer.stop()
            self._producer = None
        await self._redis.aclose()
        logger.info("kafka_event_bus_stopped")

    async def publish_refund_requested(
        self,
        task_id: str,
        payload: dict,
        *,
        recovered_state: dict | None = None,
    ) -> None:
        """Publish a refund processing event (replaces Redis queue enqueue)."""
        await self.start_producer()
        assert self._producer is not None

        event = {
            "event_type": "RefundRequested",
            "task_id": task_id,
            "customer_email": payload["customer_email"],
            "order_id": payload["order_id"],
            "refund_reason": payload["refund_reason"],
            "status": "pending",
            "enqueued_at": datetime.utcnow().isoformat(),
            "recovered_state": recovered_state,
        }
        await self._set_task_meta(task_id, event)

        await self._producer.send_and_wait(
            KAFKA_TOPIC_REFUND_REQUESTS,
            value=event,
            key=task_id,
        )
        logger.info("kafka_event_published", task_id=task_id, topic=KAFKA_TOPIC_REFUND_REQUESTS)

    async def mark_processing(self, task_id: str) -> None:
        meta = await self._get_task_meta(task_id) or {"task_id": task_id}
        meta["status"] = "processing"
        meta["processing_at"] = datetime.utcnow().isoformat()
        await self._set_task_meta(task_id, meta)

    async def mark_done(self, task_id: str, status: str = "completed") -> None:
        meta = await self._get_task_meta(task_id) or {"task_id": task_id}
        meta["status"] = status
        meta["finished_at"] = datetime.utcnow().isoformat()
        await self._set_task_meta(task_id, meta)
        logger.info("task_status_changed", task_id=task_id, status=status)

    async def get_incomplete_tasks(self) -> list[dict]:
        """Kafka redelivers uncommitted messages; return pending meta for logging only."""
        all_meta = await self._redis.hgetall(TASK_META_KEY)
        incomplete = []
        for raw in all_meta.values():
            data = json.loads(raw)
            if data.get("status") in ("pending", "processing"):
                incomplete.append(data)
        return incomplete

    async def cleanup(self, keep_seconds: int = 86400) -> int:
        """Remove old completed task metadata from Redis."""
        all_meta = await self._redis.hgetall(TASK_META_KEY)
        cutoff = datetime.utcnow().timestamp() - keep_seconds
        removed = 0
        for tid_bytes, raw in all_meta.items():
            tid = tid_bytes.decode() if isinstance(tid_bytes, bytes) else tid_bytes
            data = json.loads(raw)
            if data.get("status") not in ("completed", "failed", "hitl_pending"):
                continue
            finished = data.get("finished_at")
            if finished and datetime.fromisoformat(finished).timestamp() < cutoff:
                await self._redis.hdel(TASK_META_KEY, tid)
                removed += 1
        return removed

    async def _set_task_meta(self, task_id: str, data: dict) -> None:
        await self._redis.hset(TASK_META_KEY, task_id, json.dumps(data))

    async def _get_task_meta(self, task_id: str) -> dict | None:
        raw = await self._redis.hget(TASK_META_KEY, task_id)
        return json.loads(raw) if raw else None

    async def _ensure_consumer(self) -> AIOKafkaConsumer:
        if self._consumer is None:
            self._consumer = AIOKafkaConsumer(
                KAFKA_TOPIC_REFUND_REQUESTS,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=KAFKA_CONSUMER_GROUP,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
            await self._consumer.start()
        return self._consumer

    async def _consume_loop(self) -> None:
        retries = 0
        while self._running:
            try:
                consumer = await self._ensure_consumer()
                retries = 0
                async for msg in consumer:
                    if not self._running:
                        break
                    event = msg.value
                    task_id = event.get("task_id") or msg.key
                    if not task_id:
                        logger.warning("kafka_skip_message", reason="missing task_id")
                        await consumer.commit()
                        continue

                    if event.get("status") == "hitl_pending":
                        logger.info("kafka_skip_hitl_pending", task_id=task_id)
                        await consumer.commit()
                        continue

                    if self._handler is None:
                        await consumer.commit()
                        continue

                    try:
                        await self._handler(event)
                        await consumer.commit()
                    except Exception as exc:
                        logger.error(
                            "kafka_handler_failed",
                            task_id=task_id,
                            error=str(exc),
                        )
                        # Do not commit — message will be redelivered for crash recovery
                        await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except KafkaConnectionError as exc:
                retries += 1
                wait = min(30, 2 ** retries)
                logger.warning("kafka_connection_retry", error=str(exc), wait_s=wait)
                if self._consumer:
                    try:
                        await self._consumer.stop()
                    except Exception:
                        pass
                    self._consumer = None
                await asyncio.sleep(wait)
            except Exception as exc:
                logger.error("kafka_consumer_error", error=str(exc))
                await asyncio.sleep(5)

# Section 9: Real-time Systems

> **Project:** `task_queue_store/kafka_events.py`, async FastAPI

---

## Q31. Kafka: consumer group rebalancing and failure modes

### Rebalancing

Consumers join/leave → partitions reassigned → **stop-the-world** pause (classic) or incremental (`CooperativeStickyAssignor`).

### Failure modes

| Mode | Fix |
|------|-----|
| Rebalance storm | Increase `max.poll.interval.ms` or smaller batches |
| Duplicate processing | Idempotent consumer (`task_id`, `order_id`) |
| Offset commit after revoke | `on_partitions_revoked` → commit |
| Slow consumer kicked | Tune heartbeat + processing time |

### Our consumer

- Commits after successful `_process_refund`
- Crash before commit → redelivery (at-least-once)
- `IdempotencyRecord` makes redelivery safe

---

## Q32. Message ordering in Kafka with multiple partitions

**Rule:** Order guaranteed **within a partition** only.

```python
producer.send(topic, key=task_id, value=event)  # same refund → same partition
```

**Refund system:** Partition by `task_id` (or `order_id` if one active refund per order).

**Producer reordering:** `enable.idempotence=true` for retry safety.

---

## Q33. Design real-time agent event streaming at 100k events/sec

```
Agents → Kafka (key=session_id) → stream processor → Redis hot state
                                 → ClickHouse analytics
                                 → WebSocket/SSE gateway
```

**Scale knobs:** 30–50 partitions, lz4 compression, batch producers, horizontal consumers (≤ partitions).

**Our current scale:** In-process Kafka for refund **requests** (lower volume); audit events could fan out to same bus for observability at higher scale.

---

## Q34. WebSocket vs SSE vs long polling for agent streaming

| | WebSocket | SSE | Long polling |
|--|-----------|-----|--------------|
| Direction | Bidirectional | Server → client | Server → client |
| Agent tokens | Chat UIs | **LLM token stream** | Fallback |

**Our API today:** Async `POST /refund` + poll task status (job model). **SSE upgrade path** for streaming supervisor reasoning or policy explanations to UI (`sse_starlette`).

**Choose WebSocket** when user can interrupt agent mid-run; **SSE** for one-way token stream through corporate proxies.

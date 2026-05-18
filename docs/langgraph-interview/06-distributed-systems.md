# Section 6: Distributed Systems & Backend

> **Project:** `main.py`, `task_queue_store/kafka_events.py`, `database/models.py`, `cache/redis_cache.py`

---

## Q18. Design a distributed task queue with exactly-once delivery — how do you handle idempotency?

### Core insight

**Exactly-once = at-least-once delivery + idempotent consumers.** Networks always duplicate; eliminate duplicate *effects*.

### In our refund pipeline

```
POST /refund → idempotency check (Redis → Postgres) → Kafka publish → consumer → run_workflow
```

| Layer | Mechanism |
|-------|-----------|
| API | Client `Idempotency-Key` header → `IdempotencyRecord` + Redis TTL 24h |
| Broker | Kafka consumer commits offset **after** successful handler |
| Workflow | `request_saved`, `ProcessedRequest.order_id` UNIQUE |
| Policy cache | Semantic dedup (cosine ~0.92) — same rationale → same cached policy result |

```python
# Consumer pattern (conceptual)
async def process(event):
    if await already_processed(event["task_id"]):
        return
    await run_workflow(...)
    await mark_processed(event["task_id"])
```

### Industry patterns

- **Transactional outbox** — business row + outbox in one DB tx; relay publishes to Kafka
- **Kafka idempotent producer** — `enable.idempotence=true`
- **DB dedup table** — `processed_tasks(task_id PRIMARY KEY)`

---

## Q19. How would you handle split-brain scenarios in a distributed system?

### Definition

Network partition → two groups each think they're primary → divergent writes.

### Mitigations

| Strategy | Use |
|----------|-----|
| Quorum (Raft/etcd) | Leader election needs majority |
| Fencing tokens | Stale leader writes rejected |
| Lease-based leadership | Leader stops writes if lease can't renew |

### For agent platforms

Prefer **consistency over availability** for refund decisions — better to queue HITL than approve twice from split workers.

**Our angle:** Single-writer per `thread_id` via idempotency + DB constraints; Kafka partition key = `task_id` for ordered processing per refund.

---

## Q20. Explain backpressure in async systems — Go/Rust implementation

### Concept

Producer faster than consumer → unbounded buffers → OOM. Backpressure signals "slow down."

### Python/asyncio in our stack

- Kafka consumer processes one event at a time per partition (natural bound)
- `asyncio` task queue for background refunds — limit concurrent `run_workflow` with semaphore
- Redis connection pool sizes cap DB fan-out

### Go pattern

```go
taskCh := make(chan Task, 100) // full channel blocks producer
```

### Rust (tokio)

```rust
let (tx, mut rx) = mpsc::channel::<Task>(100);
tx.send(task).await?; // blocks when full
```

**Agent platforms:** Cap concurrent LLM calls — surge of `/refund` requests must not spawn unbounded OpenAI calls; queue or 429.

---

## Q21. Partial failures in microservices — circuit breakers, bulkheads, retries

### Applied in `executor/resilience.py`

- **Retry** with budget for transient LLM/API errors
- **Circuit breaker** — stop calling failing dependency
- **Model fallback** — secondary model when primary trips

### Per dependency

| Dependency | Isolation |
|------------|-----------|
| OpenAI | Breaker + fallback model |
| PostgreSQL | Pool limits; communication failure → HITL |
| Pinecone | Cache fallback; degrade to rule-only policy |
| Redis | Checkpoint falls back to Postgres |

### Retry rules

- Exponential backoff + jitter
- Only retry **idempotent** steps (validation read, policy search)
- Never blind-retry communication writes without `order_id` dedup

---

## Q22. Compare event sourcing vs CQRS — when would you use each?

### Event sourcing

Immutable log of changes; state = replay. **Our `AuditLog`** is a lightweight audit trail — not full ES, but same compliance idea.

### CQRS

Separate write model (agent state updates) from read model (dashboards). **Our split:** write path = LangGraph + Postgres; read path = task status in Redis meta + `TaskResult` table.

### For AI agent platforms

- **Event sourcing** — every tool call, routing decision, LLM invocation as event (Kafka `refund.requests` + audit)
- **CQRS** — analytics on ClickHouse/Postgres views without loading graph state

**Combined:** Kafka events build read-side projections for ops dashboards.

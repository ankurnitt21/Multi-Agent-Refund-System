# Section 12: System Design — Scalable AI Agent Platform

> **Map our refund system to the full platform question**

---

## Prompt

*"Design a scalable AI agent platform where multiple agents collaborate, share memory, and call external tools — with full observability."*

---

## Architecture (our system + extensions)

```
Client → API Gateway (auth, rate limit, idempotency)
      → Kafka (refund.requests)
      → LangGraph workers (supervisor + agents)
      → Tool layer (db, policy/RAG, analytics)
      → Memory (Redis session + Pinecone semantic + Postgres episodic)
      → Observability (OTel, LangSmith, structlog, RAGAS)
      → Storage (Postgres, Redis, Pinecone)
```

---

## Component mapping

| Component | Our implementation | Scale-out |
|-----------|-------------------|-----------|
| **Orchestrator** | LangGraph `warehouse_refund_workflow` | Stateless workers, shared Postgres checkpointer |
| **Multi-agent** | Supervisor + validation/policy/communication | Add agents as nodes + routing tools |
| **Memory** | `RefundState` + checkpoints + Pinecone | Redis hot, Qdrant/Pinecone cold, pg episodic |
| **Tool gateway** | LangChain tools (→ MCP + sandbox) | Per-tool circuit breaker |
| **Queue** | `KafkaRefundEventBus` | Partition by `task_id`, consumer group scale |
| **HITL** | `HITLTask` + resolve API | Review queue UI |
| **Guardrails** | injection + PII + output parser | Tenant policies |
| **Observability** | spans + audit log + RAGAS | Prometheus + LangSmith |

---

## Data flow (refund)

1. `POST /refund` → idempotency → publish Kafka event
2. Consumer → `run_workflow(thread_id=task_id)`
3. Supervisor loop until `request_saved` or HITL
4. Checkpoints after each agent
5. Result in `TaskResult`; audit in `AuditLog`

---

## Failure recovery

| Failure | Response |
|---------|----------|
| Worker crash | Kafka redelivery + checkpoint resume |
| LLM outage | Circuit breaker + model fallback (`resilience.py`) |
| Pinecone down | Degraded policy rules + HITL |
| Memory down | Continue without semantic cache |
| Poison message | DLQ + alert |

---

## Scaling strategy

- **Workers:** HPA on consumer lag
- **Kafka:** partitions ≥ max consumers
- **Postgres:** connection pooling, read replicas for status API
- **LLM:** global semaphore + provider circuit breakers
- **Tenants:** Pinecone namespace per tenant; RLS in Postgres

---

## What we'd add for Tryangle42-scale

1. Temporal for cross-day workflows and SLA timers
2. MCP tool servers with sandbox gateway
3. SSE streaming of agent progress to UI
4. Event sourcing bus for all `agent.*` events → ClickHouse
5. RBAC + tenant isolation on every path

**Closing:** *"We shipped the core loop — async ingress, durable multi-agent graph, RAG policy, HITL, idempotency, and observability — as a template for a broader agent platform."*

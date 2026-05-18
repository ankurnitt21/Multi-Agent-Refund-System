# Section 7: Agent Orchestration (Temporal / Dagster)

> **Project comparison:** LangGraph + Kafka vs Temporal/Dagster

---

## Q23. How does Temporal handle workflow state durability vs a simple job queue like Redis?

| | Redis queue (BullMQ-style) | Our Kafka + LangGraph | Temporal |
|--|---------------------------|----------------------|----------|
| Mid-run state | Worker memory | `RefundState` + checkpoints | Event history replay |
| Crash recovery | Restart job from scratch | Resume from last agent checkpoint | Resume from last activity |
| Multi-step | Manual | Native graph | Native workflow code |
| Human wait | Custom HITL table | `HITLTask` + API | Signals |

**Our stack:** Kafka gives **at-least-once work distribution**; LangGraph + `checkpoint/store.py` gives **step-level resume**; `HITLTask` gives **human wait** without Temporal.

**When we'd add Temporal:** Long-running refunds with timers (SLA escalation), cross-service sagas, or strict workflow versioning across dozens of microservices.

---

## Q24. How would you handle a stuck or zombie workflow in Temporal?

### Temporal-native (interview answer)

- Activity `StartToCloseTimeout`, `HeartbeatTimeout`
- `activity.RecordHeartbeat` in long loops
- Signal to cancel; `TerminateWorkflow`; **Reset** to prior event
- `WorkflowExecutionTimeout` as hard cap

### Our equivalent

| Temporal concept | Our implementation |
|------------------|-------------------|
| Activity timeout | ReAct max iterations → HITL |
| Workflow timeout | `MAX_CYCLES`, `MAX_RECURSION` |
| Stuck waiting | HITL `pending` tasks — ops resolves via API |
| Zombie worker | Kafka redelivers uncommitted offset; idempotency prevents double effect |

**Ops playbook:** List `HITLTask` where `status=pending`; approve/deny/compensate; monitor Kafka consumer lag.

---

## Q25. When would you choose Dagster over Temporal?

| Choose **Dagster** | Choose **Temporal** | Choose **our stack** |
|--------------------|---------------------|----------------------|
| ETL / asset pipelines | Business workflows, API orchestration | Agent graphs with LLM routing |
| dbt/Spark lineage | Signals, timers, human steps | FastAPI + LangGraph team skills |
| Batch embedding jobs | Cross-service sagas | Pinecone index build on startup |

**For Tryangle42-style agent platform:**

- **Temporal** — orchestrate long-running agent *business processes*
- **Dagster** — ingest documents, chunk, embed, refresh vector index
- **LangGraph** — *decision logic inside* each refund (what we built)

Not mutually exclusive: Dagster refreshes Pinecone; LangGraph consumes at request time.

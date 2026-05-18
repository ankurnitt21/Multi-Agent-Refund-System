# Quick Reference Cheat Sheet

> Warehouse Refund System + LangGraph interview essentials

---

| Topic | Key point | Our project |
|-------|-----------|-------------|
| StateGraph vs MessageGraph | Custom state vs messages-only | `StateGraph(RefundState)` |
| Checkpointing | Save after nodes; `thread_id` | Redis + Postgres + optional `AsyncPostgresSaver` |
| Reducers | Required for contested parallel keys | `operator.add` on messages; hub-and-spoke avoids graph races |
| Conditional edges | Pure router reads state | `route_supervisor` ← `next_agent` |
| Node errors | Catch in node; HITL/error state | `ReactMaxIterationsError`, supervisor `except` |
| State bloat | Lean checkpoints | Exclude `messages` from JSON checkpoints |
| `add_messages` | ID-based merge/update | Know concept; we use `operator.add` |
| Parallel branches | Reducers / Send / in-node parallel | `react_loop` + `VALIDATION_DEPS` waves |
| Supervisor | Hub-and-spoke, routing tools | `agents/supervisor.py`, `MAX_CYCLES=15` |
| Infinite loops | `recursion_limit` + counters | 50 recursion, 15 cycles, 3 retries/agent |
| HITL | interrupt_before OR custom node | `hitl` node + `HITLTask` table |
| Long context | Trim/summary; don't checkpoint blobs | State summary to supervisor |
| Observability | Traces + structlog + eval | OTel spans, RAGAS |
| Schema versioning | Optional fields + migration | `RefundState` optional keys |
| Exactly-once | Idempotent consumer | Kafka + idempotency key + UNIQUE order |
| RAG | Retrieve + generate + eval | Pinecone + semantic cache + RAGAS |
| Kafka | Ordering per partition key | `task_id` as key |
| Security | Injection + PII + audit | `guardrails/`, `AuditLog` |
| LCEL | `Runnable` + `\|` pipe | Used inside nodes; graph is LangGraph |
| LangChain Memory | Buffer/summary classes | We use `RefundState` + checkpointer instead |
| Custom ReAct | Parallel tool waves | `agents/react_loop.py` vs `create_react_agent` |
| LangSmith | `LANGCHAIN_TRACING_V2` | + OTel spans in `telemetry/setup.py` |

---

## Constants to remember

```python
MAX_CYCLES = 15
MAX_AGENT_RETRIES = 3
MAX_RECURSION = 50
CHECKPOINT_REDIS_TTL = 86400  # 24h
```

---

## Graph topology (one line)

`START → supervisor ⇄ [validation | policy | communication] → hitl → END`

---

## Files to cite on whiteboard

| Concept | File |
|---------|------|
| State | `agents/state.py` |
| Graph | `workflow/graph.py` |
| Supervisor | `agents/supervisor.py` |
| Checkpoints | `checkpoint/store.py` |
| Kafka | `task_queue_store/kafka_events.py` |
| RAG | `vectordb/pinecone_store.py` |
| Guards | `guardrails/input_sanitizer.py` |
| ReAct / LCEL | `agents/react_loop.py` |
| LangChain §14 | `docs/langgraph-interview/14-langchain-fundamentals.md` |

---

*Prepared for Senior Backend Engineer (Agents & Infrastructure) — LangGraph + LangChain depth + Multi-Agent Refund System.*

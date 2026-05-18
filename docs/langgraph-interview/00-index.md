# LangGraph Interview Preparation Guide

**Role:** Senior Backend Engineer — Agents & Infrastructure  
**Project context:** [Multi-Agent Refund System](https://github.com/) — Warehouse Refund Processing System (FastAPI, LangGraph, ReAct agents, Redis, PostgreSQL, Pinecone, Kafka, Guardrails-AI, LangSmith)

---

## How to use these notes

Each file is one section, in interview order. Answers combine **LangGraph fundamentals** with **what we actually built** in this repo so you can cite real nodes, state fields, and failure modes.

| # | File | Topics |
|---|------|--------|
| 1 | [01-langgraph-core-concepts.md](./01-langgraph-core-concepts.md) | StateGraph vs MessageGraph, checkpointing, reducers, conditional edges, node errors |
| 2 | [02-state-management.md](./02-state-management.md) | RefundState design, bloat, `add_messages`, parallel branches |
| 3 | [03-multi-agent-patterns.md](./03-multi-agent-patterns.md) | Handoffs, supervisor, infinite loops, HITL |
| 4 | [04-production-scale.md](./04-production-scale.md) | Long context, observability, failure modes, versioning |
| 5 | [05-complex-workflow-walkthrough.md](./05-complex-workflow-walkthrough.md) | **Opening question** — full refund workflow in 3 minutes |
| 6 | [06-distributed-systems.md](./06-distributed-systems.md) | Exactly-once, split-brain, backpressure, resilience patterns |
| 7 | [07-agent-orchestration.md](./07-agent-orchestration.md) | Temporal vs Redis queue vs our Kafka + LangGraph stack |
| 8 | [08-rag-systems.md](./08-rag-systems.md) | Pinecone policy RAG, hybrid search, evaluation |
| 9 | [09-real-time-systems.md](./09-real-time-systems.md) | Kafka consumer groups, ordering, streaming, SSE |
| 10 | [10-security-compliance.md](./10-security-compliance.md) | RBAC, PII, audit, prompt injection |
| 11 | [11-mcp-tool-integration.md](./11-mcp-tool-integration.md) | MCP vs function calling, sandboxing, OAuth |
| 12 | [12-system-design-platform.md](./12-system-design-platform.md) | Scalable multi-agent platform design |
| 13 | [13-behavioral-questions.md](./13-behavioral-questions.md) | 0→1, speed vs correctness, incidents |
| 14 | [14-langchain-fundamentals.md](./14-langchain-fundamentals.md) | LCEL, tools, memory vs state, ReAct, streaming, LangSmith |
| — | [99-quick-reference.md](./99-quick-reference.md) | Cheat sheet |

---

## Our graph at a glance

```
START → supervisor → [validation | policy | communication | hitl] → supervisor → … → END
```

- **State:** `RefundState` in `agents/state.py`
- **Graph:** `workflow/graph.py` — `warehouse_refund_workflow`
- **Agents:** validation → policy → communication (ReAct + tools), orchestrated by LLM tool-calling supervisor
- **Durability:** dual-layer checkpoints (Redis + Postgres) + optional `AsyncPostgresSaver` for LangGraph-native checkpoints

*Prepared for Tryangle42-style Senior Backend (Agents & Infrastructure) interviews.*

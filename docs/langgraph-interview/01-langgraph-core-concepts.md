# Section 1: LangGraph Core Concepts

> **Project:** Warehouse Refund Processing System — `workflow/graph.py`, `agents/state.py`

---

## Q1. What is the difference between a StateGraph and a MessageGraph? When would you use each?

### StateGraph

- General-purpose graph with a **custom state schema** (TypedDict or Pydantic).
- Nodes receive full state and return **partial updates** (only changed keys).
- You control everything beyond chat: business fields, routing flags, retry counters, HITL signals.

**Use when:** Multi-agent workflows, tool pipelines, or any flow where state is more than messages.

### MessageGraph

- Specialized graph where state is **only** `list[BaseMessage]`.
- Built for conversational chains; each node reads/writes messages.

**Use when:** Simple chatbots where message history is the entire state.

### In our project

We use **`StateGraph(RefundState)`** — not MessageGraph — because refund processing needs structured fields the supervisor routes on:

| Field | Purpose |
|-------|---------|
| `validation_passed` | Skip validation on resume |
| `decision` | Route to communication |
| `request_saved` | End workflow |
| `next_agent` | Supervisor → conditional edge |
| `cycle_count`, `agent_retry_counts` | Loop safety |
| `hitl_required` | Human escalation |

`messages` still exists (with `operator.add` reducer) for audit and LLM context, but **routing uses typed fields**, not message parsing alone.

```9:9:agents/state.py
    messages: Annotated[list[BaseMessage], operator.add]
```

**Interview line:** *"MessageGraph for chat-only prototypes; StateGraph for production multi-agent systems like our refund pipeline where the supervisor reads sentinel fields, not just the last AIMessage."*

---

## Q2. How does LangGraph's checkpointing work internally? What persistence backends have you used?

### How it works

1. A **checkpointer** (`BaseCheckpointSaver`) serializes state after each node (or on configured steps).
2. Identified by **`thread_id`** (conversation/run) and **`checkpoint_id`** (per step).
3. On resume/interrupt: load latest checkpoint, continue from that point.
4. State is serialized (JSON-compatible dict; messages need special handling).

### Flow

```
START → node A → checkpoint₁ → node B → checkpoint₂ → [interrupt/crash]
→ reload checkpoint₂ → continue from B
```

### Backends (general + our project)

| Backend | Use case | Our usage |
|---------|----------|-----------|
| `MemorySaver` | Dev/tests | Fallback when Postgres checkpointer unavailable |
| `SqliteSaver` | Local single-process | Not used |
| `PostgresSaver` / `AsyncPostgresSaver` | Production multi-worker | **`init_checkpointer()`** in `workflow/graph.py` on Linux; skipped on Windows dev |

### Our dual-layer approach

We run **two** persistence layers:

1. **LangGraph native** — `AsyncPostgresSaver` when `compile(checkpointer=...)` succeeds.
2. **Explicit agent checkpoints** — `checkpoint/store.py` saves after each agent wrapper (`_validation_node`, `_policy_node`, `_communication_node`):
   - Redis first (24h TTL, fast resume)
   - PostgreSQL `WorkflowCheckpoint` table (durable fallback)
   - Messages **excluded** from JSON (`_EXCLUDE_KEYS = {"messages"}`) — business fields only

On crash, `run_workflow()` loads Redis/Postgres checkpoint and merges into `base_state`. Supervisor sees `validation_passed=True` and routes to **policy**, not validation again.

```267:278:workflow/graph.py
        elif not _checkpointed_workflow and thread_id and _checkpoint_redis:
            existing = await load_checkpoint(_checkpoint_redis, thread_id)
            if existing:
                recovered = existing.get("state", {})
                last_agent = existing.get("last_completed_agent", "—")
                ...
                base_state = {**recovered, "thread_id": thread_id}
```

**Production note:** Shared Postgres (or Redis) so **any worker** can resume any `thread_id` — critical with Kafka consumers (`task_queue_store/kafka_events.py`).

---

## Q3. Explain the reducer function in state management — how do you handle concurrent state updates?

### What reducers do

When multiple nodes update the same key (especially in parallel), a **reducer** defines merge logic. Without one: **last-write-wins** (non-deterministic under parallelism).

### Defining reducers

```python
from typing import Annotated
import operator
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # ID-aware message merge
    tool_results: Annotated[list, operator.add]  # append lists
    final_answer: str  # no reducer → last write wins
```

### `add_messages` (built-in)

- Merges new messages into existing list.
- **Same message ID → replace** (update), not duplicate.
- Critical for tool call / `ToolMessage` pairing.

### Our project

We use **`operator.add`** for `messages` (append). For refund routing we rely on **scalar fields** with single-writer semantics per phase (validation writes `validation_passed`, policy writes `decision`). Parallelism lives **inside** agents via `react_loop` + `VALIDATION_DEPS` wave map — tool results merge in the ReAct loop, not contested graph keys.

**Rule:** Any key written by **parallel graph branches** needs an explicit reducer. Our graph is hub-and-spoke (supervisor → one agent → supervisor), so contention is rare at the graph level; parallel tool waves are isolated inside the node.

---

## Q4. How do you define conditional edges? Walk me through a real example you built.

### Concept

A **routing function** reads state and returns the next node name (or `END`). `add_conditional_edges(source, router, path_map)` wires it.

### Our refund workflow

```134:167:workflow/graph.py
def route_supervisor(state: RefundState) -> str:
    next_agent = state.get("next_agent", "end")
    if next_agent == "end":
        return END
    return next_agent

    wf.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "validation": "validation",
            "policy": "policy",
            "communication": "communication",
            "hitl": "hitl",
            END: END,
        },
    )
    ...
    wf.add_edge("validation", "supervisor")
    wf.add_edge("policy", "supervisor")
    wf.add_edge("communication", "supervisor")
    wf.add_edge("hitl", END)
```

**Who sets `next_agent`?** The **supervisor node** (`agents/supervisor.py`):

1. **HITL pass-through** — if `hitl_required`, route `hitl` immediately (before LLM).
2. **Cycle cap** — `cycle_count > MAX_CYCLES` (15) → `hitl`.
3. **LLM tool-calling** — `call_validation_agent`, `call_policy_agent`, `call_communication_agent`, `finish_workflow`.
4. **Retry cap** — per-agent `agent_retry_counts > MAX_AGENT_RETRIES` (3) → `hitl`.
5. **LLM failure** — exception → `hitl` with reason `supervisor_llm_error`.

Dynamic loop: `supervisor → validation → supervisor → policy → supervisor → communication → supervisor → end`.

**Lesson:** Keep `route_supervisor` **pure** (read-only). All side effects and LLM calls stay in `supervisor_node`.

---

## Q5. What happens when a node raises an exception — how does LangGraph handle it vs how should you handle it?

### Default LangGraph behavior

- Unhandled exception → graph **halts**, exception propagates to caller.
- Last successful checkpoint remains (if checkpointer enabled).
- **No automatic retry** at the framework level.

### What we do

| Layer | Pattern | Example |
|-------|---------|---------|
| Inside agent | try/except → structured state or HITL | `ReactMaxIterationsError` → `hitl_required=True` |
| Supervisor | LLM errors → HITl, not crash | `except Exception` → `next_agent: hitl` |
| Checkpoint save | Best-effort, never raise | `_save()` logs warning on failure |
| HITL node | DB save best-effort | `hitl_task_save_failed` logged, workflow still ends |
| API / queue | Kafka redelivery | at-least-once + idempotency keys in `main.py` |

**Validation agent pattern:**

```python
try:
    final_content = await react_loop(...)
    parsed = parse_agent_output(...)
except ReactMaxIterationsError:
    return {"hitl_required": True, "hitl_reason": "react_max_iterations", ...}
except AgentOutputParseError:
    return {"hitl_required": True, "hitl_reason": "parse_error", ...}
```

**Production:** Wrap `graph.ainvoke()` with `recursion_limit` (we use `MAX_RECURSION = 50`), idempotent DB writes (`ProcessedRequest` unique on `order_id`), and never let raw OpenAI errors bubble uncaught from the supervisor.

**Resume after failure:** Reload checkpoint by `thread_id`, re-invoke with recovered state; supervisor skips completed phases via sentinel fields.

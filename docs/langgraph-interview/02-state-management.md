# Section 2: State Management

> **Project:** `agents/state.py`, `checkpoint/store.py`, `workflow/graph.py`

---

## Q6. How did you design your state schema? What went in it vs what you kept external?

### What went INTO `RefundState`

**Conversation & orchestration**

- `messages` — HumanMessage + supervisor AIMessages (audit trail)
- `next_agent` — supervisor output for conditional routing
- `thread_id` — ties to task id, checkpoints, HITL

**Per-agent business slices**

| Agent | Key fields |
|-------|------------|
| Validation | `validation_passed`, `customer_id`, `order_*`, `customer_risk_score`, … |
| Policy | `decision`, `refund_amount`, `policy_reason`, `policy_applied`, `refund_count_90_days` |
| Communication | `request_saved`, `analytics_updated`, `status_updated` |

**Resilience & HITL**

- `cycle_count`, `agent_retry_counts`
- `error`, `hitl_required`, `hitl_reason`
- `compensated`, `compensation_reason`

### What we kept EXTERNAL

| Data | Storage | Referenced in state as |
|------|---------|------------------------|
| Full policy corpus | Pinecone index `warehouse-refund-policies` | Policy agent calls `search_policies` tool; only top-k snippets in LLM context |
| Customer/order rows | PostgreSQL | IDs + denormalized fields after tool fetch |
| Semantic cache embeddings | Redis (`cache/redis_cache.py`) | Not in state — cache key from query hash/similarity |
| Full message history for compliance | `AuditLog` table | Summaries/hashes in audit, not full replay in state |
| HITL queue payload | `HITLTask.state_json` | Full serialized state minus messages |

**Why:** Checkpoints serialize on every agent completion. Large state → slow Redis/Postgres writes and expensive LangGraph `PostgresSaver` rows.

```36:38:checkpoint/store.py
def _serialize_state(state: dict) -> dict:
    """Return a JSON-safe copy of state, dropping message history."""
    return {k: v for k, v in state.items() if k not in _EXCLUDE_KEYS}
```

**Interview line:** *"State holds routing sentinels and denormalized facts agents need for the next hop; vector DB and OLTP stay external."*

---

## Q7. How do you avoid state bloat over long-running conversations or workflows?

### Strategies in our system

1. **Exclude messages from checkpoints** — only business dict persisted (see above).

2. **Short supervisor context** — supervisor gets a **state summary** string (`validation_passed`, `decision`, `request_saved`), not full ReAct tool transcripts:

```123:130:agents/supervisor.py
        state_summary = (
            f"Current workflow state:\n"
            f"- validation_passed: {validation_passed}\n"
            f"- decision: {decision}\n"
            f"- request_saved: {request_saved}\n"
            ...
        )
```

3. **ReAct loops are node-local** — tool call history lives inside `react_loop` for that invocation; only structured `ValidationOutput` / policy output lands in `RefundState`.

4. **Single refund = bounded graph** — typical path is ~4–8 supervisor cycles, not hundreds of turns. Hard caps: `MAX_CYCLES=15`, `MAX_RECURSION=50`.

5. **Delete checkpoint on success** — `delete_checkpoint()` after `request_saved` and no pending HITL.

6. **Entry-point sanitization** — `sanitize_input(refund_reason)` caps length (500 chars) before it enters state/messages.

### General patterns (also apply)

- `trim_messages` before LLM calls for long chats
- Summarization node when token count exceeds threshold
- Archive old turns to DB by `thread_id`; keep last N in state
- Clear intermediate keys after phase completes (e.g. raw retrieval lists)

**Rule:** If only one node needs data for one invocation, fetch inside the node — don't checkpoint it.

---

## Q8. Have you used Annotated types with reducers? Explain how `add_messages` works under the hood.

### Yes — with a project nuance

```9:9:agents/state.py
    messages: Annotated[list[BaseMessage], operator.add]
```

We use **`operator.add`** (list concatenation), not `add_messages`. For our supervisor-summary pattern, duplicate-ID tool threading is less critical at the graph level than in a heavy tool-calling chat loop.

### How `add_messages` works (know for interviews)

```python
messages: Annotated[list, add_messages]
```

1. Node returns `{"messages": [new_msg]}`.
2. LangGraph calls `add_messages(existing, new)`.
3. For each new message: if **same `.id`** exists → **replace**; else **append**.
4. Enables correct `AIMessage` + `ToolMessage` pairing without duplicates.

### When we'd switch

If we moved full ReAct transcripts into graph state (for replay/debug), we'd change to `add_messages` to handle tool message updates safely.

### Custom reducer example (policy retrieval)

```python
def dedupe_by_doc_id(existing: list, new: list) -> list:
    seen = {d["id"] for d in existing}
    return existing + [d for d in new if d["id"] not in seen]
```

Useful if multiple parallel retrieval nodes wrote to `retrieved_docs` — analogous to deduping Pinecone chunks in policy agent.

---

## Q9. How do you handle shared state across parallel branches?

### Problem

Fan-out nodes writing the same key without reducers → races and last-write-wins.

### Our architecture

**Graph level:** Sequential hub-and-spoke — no parallel LangGraph branches writing the same keys.

**Inside-node level:** Parallel **tool waves** via `react_loop` + dependency map:

```27:39:agents/validation_agent.py
# Wave 0 (parallel): lookup_customer + get_order_details
# Wave 1 (parallel): get_customer_analytics + verify_order_ownership
VALIDATION_DEPS: dict[str, list[str]] = {
    "lookup_customer":        [],
    "get_order_details":      [],
    "get_customer_analytics": ["lookup_customer"],
    "verify_order_ownership": ["lookup_customer"],
}
```

`executor/parallel_executor.py` runs independent tools concurrently; results feed the LLM inside one node return — **one partial state update** when the agent finishes.

### Patterns for true graph-level parallelism

| Pattern | When |
|---------|------|
| `Annotated[list, operator.add]` | Both branches append to same list |
| Separate keys + join node | `branch_a_results`, `branch_b_results` → merge node |
| `Send` API | Independent sub-invocations per item |
| Barrier edge | LangGraph waits for all incoming edges |

**Interview line:** *"We parallelize tool execution inside ReAct nodes; the StateGraph stays a supervisor loop to avoid reducer contention on RefundState."*

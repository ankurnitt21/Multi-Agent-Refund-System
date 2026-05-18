# Section 3: Multi-Agent Patterns

> **Project:** `agents/supervisor.py`, `workflow/graph.py`, `agents/*_agent.py`

---

## Q10. How did you implement agent handoffs between nodes?

### Pattern: explicit handoff via shared state + supervisor hub

Agents **do not** route to each other directly. Every agent returns to **supervisor**, which sets `next_agent`.

```python
# Each agent node returns business fields only — no routing autonomy
async def validation_agent_node(state: RefundState) -> dict:
    ...
    return {"validation_passed": True, "customer_id": 123, ...}

# Supervisor decides next hop
return {"next_agent": "policy", "cycle_count": cycle_count, ...}
```

**Handoff signals:**

| From | State signal | Supervisor routes to |
|------|--------------|----------------------|
| Validation | `validation_passed is None` | validation (via `call_validation_agent`) |
| Validation | `validation_passed is False` | end (`finish_workflow`) |
| Validation | `validation_passed is True`, `decision is None` | policy |
| Policy | `decision` set, `request_saved` not True | communication |
| Communication | `request_saved is True` | end |

LLM routing tools encode these rules in tool descriptions; safety guards (cycle/retry/HITL) run **before** trusting the LLM.

**Messages:** Supervisor appends `AIMessage` with routing reasoning — `name` field optional; we use content like `"Supervisor → policy | LLM→call_policy_agent"`.

---

## Q11. Have you built a supervisor agent pattern? How did you route between sub-agents?

### Yes — tool-calling supervisor (hub-and-spoke)

```
        ┌─────────────┐
        │  supervisor │
        └──────┬──────┘
   ┌──────┼──────┬──────────┐
   ▼      ▼      ▼          ▼
validation policy communication hitl
   │      │      │          │
   └──────┴──────┴──────────┘
              (all return to supervisor)
```

**Implementation details:**

1. **Routing tools** — pure signals, no business logic in tool bodies:

```27:48:agents/supervisor.py
@tool
def call_validation_agent() -> str:
    """Use when validation_passed is None ..."""
    return "validation"
...
@tool
def finish_workflow() -> str:
    """Use when validation_passed is False OR request_saved is True ..."""
    return "end"
```

2. **`tool_choice="any"`** — forces exactly one routing tool per supervisor pass.

3. **Sub-agents are ReAct nodes** — validation/policy run `react_loop` with parallel tool waves; communication persists to DB.

4. **Termination** — `finish_workflow` → `next_agent: end` → `route_supervisor` returns `END`.

5. **Production guards:**
   - `MAX_CYCLES = 15`
   - `MAX_AGENT_RETRIES = 3` per agent key in `agent_retry_counts`
   - HITL on exhaustion or LLM failure

**Why hub-and-spoke:** Avoids spaghetti edges (validation → policy directly). One place for caps, logging, and LangSmith spans.

---

## Q12. How do you prevent infinite loops in cyclic graphs?

Our graph is intentionally cyclic (`agent → supervisor → agent`). Protections:

### 1. Cycle counter in state

```89:116:agents/supervisor.py
        cycle_count = (state.get("cycle_count") or 0) + 1
        ...
        if cycle_count > MAX_CYCLES:
            return {
                "next_agent": "hitl",
                "hitl_required": True,
                "hitl_reason": f"Cycle limit ({MAX_CYCLES}) exceeded",
                ...
            }
```

### 2. LangGraph `recursion_limit`

```235:235:workflow/graph.py
        config: dict = {"recursion_limit": MAX_RECURSION}  # 50
```

Raises `GraphRecursionError` if total node executions explode.

### 3. Per-agent retry cap

Same supervisor block — `agent_retry_counts[next_agent] > MAX_AGENT_RETRIES` → HITL.

### 4. Sentinel convergence

`request_saved=True` or `validation_passed=False` → `finish_workflow` should fire; LLM is nudged via state summary.

### 5. ReAct iteration cap

`ReactMaxIterationsError` inside agents → HITL, not infinite tool loop.

### 6. Operational timeout

Kafka/API layer can bound total wall-clock (async task cancellation in production deployments).

**Optional:** Detect identical last N supervisor messages — we log `supervisor_routing` with reasoning instead.

---

## Q13. How would you implement human-in-the-loop interrupts — what does `interrupt_before` actually do?

### LangGraph native: `interrupt_before` / `interrupt_after`

When compiled with `interrupt_before=["human_review"]`:

1. Runs all nodes **up to** that node.
2. **Saves checkpoint**.
3. Pauses — graph waits for human input.
4. Resume: `graph.update_state(config, {...})` then `graph.invoke(None, config)`.

- **`interrupt_before`** — review *inputs* to a risky step.
- **`interrupt_after`** — review *outputs* before downstream use.

### Our project: application-level HITL (same semantics, custom UX)

We route to a dedicated **`hitl` node** instead of LangGraph interrupts:

```93:101:agents/supervisor.py
        if state.get("hitl_required"):
            return {"next_agent": "hitl", ...}
```

`hitl_node` in `workflow/graph.py`:

- Persists full state to `HITLTask` (Postgres upsert on `task_id`)
- Sets `request_saved=True` to stop supervisor looping
- Edge `hitl → END`

**Human resolves via API** (`POST /hitl/tasks/{task_id}/resolve`):

| Action | Behavior |
|--------|----------|
| `approve` | Clear LangGraph checkpoint, re-run `run_workflow` with `recovered_state` |
| `deny` | Write denied outcome, mark task done |
| `compensate` | Saga-style compensation fields in state |

**Triggers for HITL:**

- Cycle/retry limits
- Supervisor LLM error
- `ReactMaxIterationsError`
- Parse/validation failures
- DB write failures (communication agent)

**Why both patterns:** LangGraph interrupts are ideal for in-graph pause/resume; our HITL table + REST API fits ops dashboards and Tryangle42-style audit requirements.

**Checkpoint on approve:** We delete stale LangGraph checkpoint before resume so state doesn't fork:

```252:259:workflow/graph.py
            if _checkpointed_workflow and thread_id:
                await _checkpointed_workflow.checkpointer.adelete(...)
```

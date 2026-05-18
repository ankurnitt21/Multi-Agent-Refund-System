# Section 4: Production & Scale

> **Project:** `telemetry/setup.py`, `executor/resilience.py`, `evaluation/`, `main.py`

---

## Q14. How did you handle long-running workflows that exceed LLM context limits?

### Our refund workflow is bounded by design

Typical run: 1 user message + ~4–8 supervisor AIMessages + **no** full ReAct traces in graph state. Heavy context stays inside a single agent invocation.

### Techniques we use

1. **Supervisor state summary** — not full message history (see Section 2).

2. **Structured agent outputs** — Pydantic `ValidationOutput`, policy models via `output_parser.py` — only fields land in state, not raw tool JSON dumps.

3. **Policy RAG top-k** — Pinecone `top_k=3` policy chunks, not full policy corpus (`vectordb/pinecone_store.py`).

4. **Semantic cache** — Redis cache for similar policy queries avoids repeat embedding + LLM context.

5. **Input length cap** — `MAX_REASON_LENGTH = 500` in `guardrails/input_sanitizer.py`.

6. **Fast vs main model** — validation uses `OPENAI_FAST_MODEL`; supervisor/policy use configured main model — cost/latency tradeoff.

### If context grew (e.g. multi-turn HITL revisions)

- `trim_messages` before each LLM call
- Summarization node when token count > threshold
- Compress tool outputs before adding to messages
- External transcript store keyed by `thread_id`

---

## Q15. What observability did you add — tracing, logging, LangSmith?

### LangSmith / OpenTelemetry

- Spans on workflow root: `warehouse_refund_workflow`
- Per-node: `validation_agent`, `policy_agent`, `supervisor_node`, `hitl_node`
- Attributes: `langsmith.span.kind`, `agent.name`, `supervisor.next_agent`, `validation.passed`, `workflow.decision`

```227:231:workflow/graph.py
    with tracer.start_as_current_span("warehouse_refund_workflow") as root_span:
        root_span.set_attribute("langsmith.span.kind", "chain")
        root_span.set_attribute("workflow.name", "warehouse_refund_workflow")
```

### Structured logging (structlog)

- `supervisor_routing`, `checkpoint_saved`, `hitl_triggered`, `kafka_producer_ready`
- Correlate with `thread_id` / `task_id` on every line

### Metrics to mention in interviews

| Metric | Why |
|--------|-----|
| Node latency per agent | SLA tuning |
| Token usage per phase | Cost attribution |
| HITL rate by reason | Product/ops signal |
| Checkpoint resume count | Reliability |
| Kafka consumer lag | Queue health |
| RAGAS scores (`evaluation/ragas_evaluator.py`) | Quality regression |

### RAGAS + A/B testing

- `RefundRAGASEvaluator` for faithfulness/relevancy on policy answers
- `evaluation/ab_testing.py` + `PromptRegistry` for prompt version experiments

---

## Q16. What were the failure modes you hit in production and how did you fix them?

### 1. Malformed LLM structured output

**Symptom:** Policy/validation JSON doesn't match Pydantic schema.  
**Fix:** `parse_agent_output` + `AgentOutputParseError` → HITL with reason; optional retry inside ReAct before escalate.

### 2. Supervisor infinite re-routing

**Symptom:** LLM keeps calling `call_validation_agent` after validation done.  
**Fix:** `MAX_CYCLES`, per-agent retry caps, state summary with explicit field values; deterministic tool descriptions.

### 3. Checkpoint / schema drift

**Symptom:** New `RefundState` fields break old JSON checkpoints.  
**Fix:** `Optional` fields with defaults; exclude non-serializable `messages`; migration via `schema_version` if needed.

### 4. Tool / DB timeouts

**Symptom:** Communication agent partial write.  
**Fix:** `ProcessedRequest` UNIQUE on `order_id` — retry treats duplicate as success; `request_saved` sentinel; circuit breaker in `executor/resilience.py`.

### 5. Windows dev vs Linux prod checkpointer

**Symptom:** `AsyncPostgresSaver` incompatible with Windows selector loop.  
**Fix:** Explicit Redis+Postgres checkpoints always; LangGraph checkpointer only on Linux production.

### 6. Stale RAG policy answers

**Symptom:** Pinecone index out of date after policy doc change.  
**Fix:** `build_policy_index()` on startup; content-hash invalidation pattern for doc updates (operational runbook).

### 7. Kafka duplicate delivery

**Symptom:** Same refund processed twice after consumer crash.  
**Fix:** Idempotency keys (`IdempotencyRecord`), Redis fast path + Postgres durable; `order_id` uniqueness.

---

## Q17. How did you version your graphs when the schema or logic changed?

### State schema

- Add fields as `Optional[...]` with defaults in nodes
- Never remove/rename without migration
- `checkpoint/store.py` serialization drops unknown keys safely on read if you add a migration layer

### Graph logic

- `PromptRegistry` — versioned prompts (`prompts/registry.py`), activate without redeploying graph structure
- Feature flag pattern: route new traffic to `refund_workflow_v2` compile name
- In-flight threads finish on old graph until checkpoints age out

### Migration sketch

```python
def migrate_checkpoint(state: dict) -> dict:
    if state.get("schema_version", 1) == 1:
        state.setdefault("agent_retry_counts", {})
        state["schema_version"] = 2
    return state
```

### Checkpointer compatibility

On breaking changes: bump `schema_version`, run batch migration on `WorkflowCheckpoint.state_json`, or invalidate old checkpoints with TTL.

**Lesson:** Treat `RefundState` like a DB schema — the supervisor's routing depends on sentinel fields staying stable.

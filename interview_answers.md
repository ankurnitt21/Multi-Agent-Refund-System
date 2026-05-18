# Senior Backend Engineer (Agents & Infrastructure) — Interview Q&A

> Answers grounded in the `refund_system` project (LangGraph + LLM multi-agent pipeline) and industry best practices.

---

## Topic 1 — Durable Workflow Engine & State Persistence

### Q2. Describe a durable workflow engine that persists the step state of each node execution and supports restart/recovery without re-executing completed steps.

#### In Our Project

The `StateGraph` in `workflow/graph.py` is the workflow engine. `RefundState` in `agents/state.py` — a flat TypedDict — is the checkpoint state. Each agent node owns a dedicated slice of that state:

| Node                  | Fields it writes                                                   |
| --------------------- | ------------------------------------------------------------------ |
| `validation_agent`    | `validation_passed`, `customer_id`, `order_id`, `customer_tier`, … |
| `policy_agent`        | `decision`, `refund_amount`, `policy_reason`, `policy_applied`     |
| `communication_agent` | `request_saved`, `analytics_updated`, `status_updated`             |

The supervisor's `_deterministic_route` function in `supervisor.py` acts as the step-skip guard. It inspects the three sentinel fields (`validation_passed`, `decision`, `request_saved`) before routing. If the process crashes after the validation node completes but before policy runs, restarting with the persisted state immediately routes to `policy`, skipping validation entirely.

For true durability, LangGraph supports `PostgresSaver` as a checkpointer. Every node execution is written to a `checkpoints` table keyed by `thread_id` and `checkpoint_id`. Our project currently uses in-memory state, but the `StateGraph.compile(checkpointer=...)` API makes it a one-line swap to persistent storage.

#### Industry Standard

- **Temporal.io** persists workflow state (history events) after every activity. Activities are retried independently; completed ones are never re-run.
- **AWS Step Functions** writes the full execution history to a managed store. Each state transition is durable; resumption reads the last persisted state.
- **Apache Airflow** tracks task instance states (`queued`, `running`, `success`, `failed`) in a PostgreSQL metadata database. Re-triggering a DAG with `mark_success` on completed tasks skips them.
- **Prefect** uses a task-run result store; completed task results are cached and reused on retry.

**Core principle shared by all**: before executing a step, check a persistent store for an existing result keyed by `(workflow_run_id, step_id)`. Only execute if no result exists.

---

## Topic 2 — Exactly-Once Execution & Idempotency

### Q3. How would you guarantee exactly-once execution of workflows in the presence of retries and crashes?

#### In Our Project

True exactly-once is achieved through layered guards:

1. **DB-level deduplication** — `ProcessedRequest` in `database/models.py` has a `UniqueConstraint` on `order_id`. If the communication agent crashes after writing to the DB but before updating `request_saved`, a retry attempt hits a unique-constraint violation, which the tool catches and treats as "already done" rather than an error.

2. **Workflow-level flag** — The `request_saved` field in `RefundState` acts as a soft idempotency guard. The supervisor checks it before ever routing to the communication agent again.

3. **Audit trail** — `AuditLog` in `database/models.py` records every significant action. A replay can compare the log against expected steps to detect partial execution.

4. **Semantic cache** — `cache/redis_cache.py` deduplicates policy lookups by cosine similarity. The same refund rationale never triggers a redundant LLM call.

#### Industry Standard

True exactly-once processing is impossible at the network level; the industry achieves it through **idempotent consumers**:

- **Outbox Pattern**: Write the business record and an outbox event to the same DB transaction. A separate relay reads the outbox and publishes to the message broker. Consumers process with deduplication.
- **Kafka Transactions**: `enable.idempotence=true` + transactional producers give exactly-once delivery to Kafka. Consumers use `isolation.level=read_committed`.
- **Two-Phase Commit (2PC)**: A coordinator locks all participants before committing. Rarely used in microservices due to latency and coordinator single-point-of-failure.
- **Saga Pattern**: Each step has a compensating transaction. If step N fails, steps N-1 through 1 are rolled back via their compensations.

---

### Q4. What strategies would you use for idempotency keys, deduplication, and transactional output patterns?

#### In Our Project

| Strategy                          | Where in project                                                                                                                                                                 |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Natural idempotency key**       | `order_id` scopes every operation — all DB writes, cache entries, and audit logs are keyed to one order                                                                          |
| **Unique constraint guard**       | `ProcessedRequest.order_id` UNIQUE — prevents double inserts at the DB layer                                                                                                     |
| **Embedding-based deduplication** | `SemanticCache` in `cache/redis_cache.py` uses cosine similarity (threshold 0.92) — semantically equivalent policy queries return the same cached result without hitting the LLM |
| **State field as lock**           | `request_saved`, `analytics_updated`, `status_updated` in `RefundState` — each communication tool sets its flag; the agent exits cleanly on retry if flags are already `True`    |

#### Industry Standard

- **Stripe-style idempotency keys**: Client sends a UUID in the `Idempotency-Key` header. Server stores `(key → response)` in Redis with TTL. If the same key arrives again, the stored response is returned without re-execution.
- **Redis SET NX (set-if-not-exists)**: Distributed lock. A worker acquires the lock for `order:{id}:processing` before starting. If the lock exists, another worker is already handling it.
- **Transactional Outbox**: Business write + outbox row in one local DB transaction. Eliminates the two-write problem (DB write succeeded, message publish failed).
- **Content-addressable deduplication**: Hash the input payload; use the hash as the idempotency key. Same input always maps to same key.
- **Exactly-once sinks**: Kafka Connect JDBC Sink uses `pk.mode=record_value` — row key prevents duplicate inserts.

---

## Topic 3 — Dynamic Replanning & Runtime Graph Modification

### Q5. How does dynamic replanning differ from fixed workflow execution? What internal components of an agent system are responsible for modifying the execution graph at runtime?

#### In Our Project

**Fixed execution** in our project: `workflow/graph.py` defines a static graph at compile time — `START → supervisor → [validation | policy | communication] → supervisor → END`. The edges and nodes are immutable after `workflow.compile()`.

**Dynamic routing** in our project: The supervisor does runtime path selection through two mechanisms:

- `_deterministic_route` in `supervisor.py` — reads live state fields and decides the next node
- LLM tool-calling — the LLM reasons over the full state and calls one of `call_validation_agent`, `call_policy_agent`, `call_communication_agent`, or `finish_workflow`. The LLM's choice influences but does not override the deterministic route

**Components responsible for runtime modification:**

| Component                           | Role                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------ |
| `supervisor_node` (`supervisor.py`) | Primary planner — reads full state, computes next step                                     |
| `route_supervisor` (`graph.py`)     | Translates `next_agent` state field into actual graph edges                                |
| `react_loop` (`react_loop.py`)      | Within-agent dynamic tool selection — LLM generates tool calls based on prior observations |
| `next_agent` field in `RefundState` | Message bus between supervisor and graph router                                            |
| `error` field in `RefundState`      | Signals early termination; supervisor routes to `END` on any error                         |

#### Industry Standard

| Approach                                | Characteristics                                                                   |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| **Fixed DAG** (Airflow, Step Functions) | Nodes and edges defined at design time; reliable but inflexible                   |
| **LangGraph conditional edges**         | Edges exist statically but activation is decided at runtime by a routing function |
| **ReAct loop** (our `react_loop.py`)    | Agent generates the next action from observations; no predefined edge list        |
| **AutoGPT / BabyAGI**                   | Planner generates and reprioritises an entire task list after each step           |
| **CrewAI dynamic delegation**           | Agents can delegate tasks to other agents on the fly, creating edges at runtime   |

The key internal components in any system: **planner/supervisor**, **state store**, **edge resolver**, and **execution queue**.

---

### Q6. When an agent dynamically modifies the execution graph at runtime, how would you ensure state consistency across already-executed, in-progress, or newly added nodes?

#### In Our Project

1. **Append-only message history**: `messages` in `RefundState` uses `Annotated[list[BaseMessage], operator.add]`. LangGraph only ever appends to this list — no node can overwrite a prior message. This gives a full, immutable audit trail.

2. **Strict field ownership**: Each node owns specific fields and never writes fields owned by another node. `validation_agent` never touches `decision`; `policy_agent` never touches `request_saved`. This eliminates write conflicts.

3. **Supervisor reads full state snapshot**: Before every routing decision, `supervisor_node` receives the complete, merged `RefundState`. Any newly written field by a completed node is immediately visible.

4. **LangGraph merge semantics**: Node outputs are merged into state via reducers. Non-annotated fields use last-write-wins. This means a retry of a node safely overwrites its own previous output without corrupting other fields.

5. **RecursionLimit guard**: `MAX_RECURSION = 50` in `graph.py` caps total supervisor iterations, preventing a dynamic routing bug from causing infinite cycles.

#### Industry Standard

- **Event Sourcing**: State is derived by replaying an append-only event log. New nodes can be added; state is always reconstructible from the log. Current state = reduce(all past events).
- **Snapshot + delta**: Store a periodic snapshot plus incremental deltas. On recovery, load latest snapshot and apply deltas.
- **Optimistic Concurrency Control**: Attach a version number to state. A node that reads version N must write to version N+1. Concurrent writes are rejected and retried.
- **CRDT (Conflict-free Replicated Data Types)**: State fields designed so concurrent writes always merge deterministically (e.g., counters, sets).
- **LangGraph PostgresSaver**: Stores a full checkpoint after every node. Each checkpoint has a `thread_id` + `checkpoint_id`. Rolling back means loading a prior checkpoint by ID.

---

## Topic 4 — Cycle Detection & Failure Recovery

### Q7. Write pseudo code for cycle detection in a directed graph used for agent task planning, and explain how failures are surfaced back to the planner.

#### In Our Project

Cycle detection exists at two levels:

**Compile-time (graph level)**: LangGraph's `workflow.compile()` validates the graph topology. An invalid edge configuration (e.g., adding a direct edge from `validation` back to itself) would raise at compile time, never at runtime.

**Runtime (tool dependency level)**: `_execution_waves` in `react_loop.py` runs a topological wave sort over the tool calls generated by the LLM. The dead-giveaway for a cycle is when the `ready` list becomes empty while tools remain — all remaining tools are waiting on each other. The code handles this by treating all remaining tools as one wave (breaking the deadlock) rather than hanging forever.

**Runtime (workflow level)**: `MAX_RECURSION = 50` in `graph.py`. LangGraph counts supervisor iterations. If the supervisor loops without terminating (e.g., due to a bug where `request_saved` never gets set to `True`), LangGraph raises a `GraphRecursionError`.

**Failure surfacing path:**

1. Tool-level error → `_call_tool` catches the exception and returns `{"error": "..."}` as a JSON string — error becomes a `ToolMessage` in the conversation
2. Agent-level error → agent node catches it, writes to the `error` field in `RefundState`
3. Supervisor reads `error` field → `_deterministic_route` treats any error state as a terminal condition and routes to `END`
4. `run_workflow` in `graph.py` propagates the exception to the caller (the UI)

#### Industry Standard (DFS Cycle Detection Concept)

The standard algorithm uses three node colours — `WHITE` (unvisited), `GRAY` (in current DFS path), `BLACK` (fully processed). A back-edge from a node to a `GRAY` ancestor signals a cycle.

After detection:

- Surface the cycle path (list of node IDs forming the cycle) in an error message
- Log it as a span event in the tracer (OTEL `span.add_event`)
- Set a `cycle_detected` flag in the workflow state
- Planner receives the flag and can either abort, reroute, or request human intervention

Used by: Apache Airflow (DAG cycle check at import), LangGraph `compile()`, Kubernetes resource dependency validator.

---

### Q8. If a cycle is detected after some nodes have already been executed, how do you decide whether to roll back, compensate, or continue?

#### In Our Project

The decision maps directly onto which nodes have already committed side effects:

| Scenario                                        | What has executed                                                                                 | Strategy                                                                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Cycle detected before communication agent       | Only in-memory state mutations                                                                    | **Abort** — set `error` in `RefundState`, supervisor routes to `END`. No DB writes, nothing to undo                                            |
| Cycle detected mid-communication agent          | Some DB writes committed (`save_processed_request` succeeded, `update_customer_analytics` failed) | **Compensate** — `AuditLog` records what succeeded; a compensating transaction reverses the partial write                                      |
| Cycle in tool dependency graph within one agent | LLM-generated tool calls form a loop                                                              | **Continue** — `react_loop.py` breaks the deadlock by running all remaining tools in one wave; error JSON is passed back to LLM as observation |

The `request_saved`, `analytics_updated`, `status_updated` boolean fields in `RefundState` act as compensability indicators — the supervisor can read them to know exactly which sub-steps committed and which need reversal.

#### Industry Standard (Decision Framework)

**Roll back**:

- Use when no side effects have left the system boundary (no external API calls, no published events, no committed DB rows)
- Cheapest option — simply discard in-progress state

**Compensate (Saga pattern)**:

- Use when one or more steps have committed irreversible side effects (payment captured, email sent, inventory reserved)
- Each step has a paired compensating action (refund payment, send cancellation email, release inventory)
- AWS Step Functions supports explicit compensating state machines
- Temporal.io saga workflows model compensation as regular activities

**Continue**:

- Use when the cycle is in an optional or non-critical subgraph that can be safely skipped
- Mark the cyclic path as `SKIPPED` in state and proceed with the rest of the plan
- Appropriate when the cyclic nodes are enrichment steps (analytics, caching) rather than core processing

**General rule**: the further downstream the cycle is detected, the more compensation is needed and the more expensive recovery becomes. Catch cycles at compile time when possible.

---

## Topic 5 — PII Redaction & Security

### Q9. Write a high-level pipeline that ensures PII (phone numbers, emails) is redacted before logging to an agent execution trace.

#### In Our Project — Current State

The project applies multi-layer PII redaction via the `guardrails/` module:

- `guardrails/pii_handler.py` provides both reversible (`PIIMasker`) and irreversible (`mask_pii()`) PII masking for emails, phones, CC numbers, SSNs, and IP addresses
- `guardrails/input_sanitizer.py` strips control characters and Unicode exploits, then wraps user data in delimiters
- `guardrails/runner.py` runs `RefundGuardRunner.validate_input()` which masks PII before workflow entry — traced to LangSmith
- `workflow/graph.py` masks email in the OTEL span: the `customer.email` attribute is set as `***@domain.com`
- `agents/policy_agent.py` runs `mask_pii()` on the refund reason before constructing the LLM prompt

**Gaps that need to be addressed:**

1. The `messages` list in `RefundState` contains the raw `HumanMessage` with the full email address. This entire list is exported to LangSmith via the OTLP exporter in `telemetry/setup.py` — the raw email is visible in the trace.
2. `lookup_customer` tool response in `db_tools.py` returns `email` and `phone` as plain JSON. These flow into `ToolMessages` which are also exported.
3. `COMMUNICATION_SYSTEM` prompt content and `VALIDATION_SYSTEM` prompt content reference customer data without scrubbing.

#### Full Pipeline Design (No Code)

**Layer 1 — Pre-LLM prompt scrubbing (real-time)**

Before constructing the `SystemMessage` or `HumanMessage` that goes to the LLM, run a PII detector over the string. Replace matches with placeholder tokens like `[EMAIL_1]` and store a reverse mapping locally for the duration of the request. The LLM sees only tokens; results containing tokens are de-tokenised before returning to the user but remain tokenised in logs.

**Layer 2 — Tool response scrubbing (real-time)**

The `_call_tool` function in `react_loop.py` is the single choke point for all tool outputs. After receiving the raw JSON string from a tool, pass it through a field-level redactor that replaces known PII fields (`email`, `phone`, `name`) with masked values before the string is appended as a `ToolMessage`. This keeps PII out of the message history that gets exported.

**Layer 3 — OTEL span attribute filter (real-time)**

Add a custom `SpanProcessor` in `telemetry/setup.py` that intercepts every span before export. The processor scans all span attributes and span events for regex patterns matching email addresses and phone numbers, replaces them with `***REDACTED***`, then forwards the cleaned span to the OTLP exporter.

**Layer 4 — LangSmith / log store post-processing**

LangSmith supports masking rules on a project level. Configure regex-based masking rules for `[\w.-]+@[\w.-]+\.\w+` (email) and `\+?[0-9]{10,13}` (phone). These run on stored traces server-side and are a safety net for anything that slipped through layers 1–3.

**Layer 5 — Database field-level encryption**

`Customer.email` and `Customer.phone` in `models.py` should use application-level encryption (e.g., SQLAlchemy `TypeDecorator` with AES-256) or database-level column encryption. The encryption key is stored in a secrets manager (AWS KMS, HashiCorp Vault), never in `config.py`.

#### Industry Standard Tools

| Tool                             | Purpose                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| **Microsoft Presidio**           | NLP + regex-based PII detector; supports 50+ entity types; runs as a microservice        |
| **AWS Comprehend**               | Managed PII detection and redaction via API                                              |
| **Google Cloud DLP**             | Field-level de-identification with tokenisation and format-preserving encryption         |
| **Langfuse masking**             | Server-side trace masking rules in the observability platform                            |
| **Vault Transit Secrets Engine** | Encryption-as-a-service; application calls Vault to encrypt/decrypt, never holds the key |

---

## Topic 6 — MCP Tool Schema Evolution & Backward Compatibility

### Q10. How do you handle backward compatibility when MCP tool schemas evolve while existing agents still depend on the older schema version?

#### In Our Project

Tools are defined via the `@tool` decorator in `tools/db_tools.py`, `tools/policy_tools.py`, and `tools/analytics_tools.py`. Each tool's docstring and parameter signature constitute its schema. This schema is serialised and sent to the LLM inside `llm.bind_tools(...)` calls in each agent.

**Where breakage occurs:**

- `validation_agent.py` calls `llm.bind_tools(VALIDATION_TOOLS)` — if `lookup_customer` gains a new required parameter `tenant_id`, the LLM starts generating calls with the new parameter. But the old production deployment still has the old function signature, so calls fail.
- The VALIDATION_SYSTEM prompt text explicitly names tool call patterns — a renamed tool breaks the LLM's learned call pattern.
- `VALIDATION_DEPS` in `validation_agent.py` and `TOOL_DEPENDENCY_MAP` in `executor/parallel_executor.py` both hard-code tool names. A renamed tool breaks the dependency graph.

#### Backward Compatibility Strategy

**1. Additive changes only (non-breaking)**

New parameters must have default values. Existing callers pass nothing for the new parameter and continue working. The tool internally uses the default. This is how `lookup_customer` should add `tenant_id=None` rather than `tenant_id: str`.

**2. Tool versioning**

Register both `lookup_customer_v1` and `lookup_customer_v2` in the tool map. Agents that have not been updated use `v1`. New agents use `v2`. Both live in production simultaneously. Deprecate `v1` only after all agents have migrated.

**3. Adapter / shim layer**

The adapter is a thin wrapper tool registered under the old name. Internally, it translates the old call shape into the new shape, calls the real `v2` implementation, and transforms the `v2` response back into the `v1` response shape before returning. The calling agent sees no change. This is the same pattern as an API Gateway request/response transformer.

**4. Tool registry with schema versioning**

Maintain a central registry (could be a table in PostgreSQL or a config file) that maps `(tool_name, version) → implementation`. When an agent binds tools, it requests a specific version. The registry resolves the binding. Schema changes bump the version; agents pin to a version explicitly.

**5. Contract testing**

Before deploying a schema change, run consumer-driven contract tests (Pact framework). Each agent's expected call shape is a "contract." The tool must satisfy all active contracts before deployment.

**6. Deprecation via docstring**

Since the LLM reads the tool docstring to understand usage, add `DEPRECATED: use lookup_customer_v2 instead` at the top of the old tool's docstring. The LLM will naturally migrate if given both tools.

#### Industry Standard Patterns

| Pattern                             | Use Case                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| **Semantic versioning**             | `tool@1.2.3` — breaking changes bump major version                                          |
| **Confluent Schema Registry**       | Central store for Avro/JSON schemas with compatibility checks (BACKWARD, FORWARD, FULL)     |
| **API Gateway versioning**          | `/v1/` and `/v2/` routes live simultaneously; v1 is proxied to the adapter                  |
| **Strangler Fig**                   | Incrementally route traffic from old tool to new tool; remove old once traffic is zero      |
| **Feature flags**                   | Toggle which tool version an agent uses per environment without redeployment                |
| **gRPC with proto evolution rules** | Field IDs never reused; old fields deprecated, not removed; wire-compatible across versions |

---

## Topic 7 — Guardrails-AI Integration & Input/Output Validation

### Q11. How do you implement guardrails for LLM input/output validation in a multi-agent system?

#### In Our Project

The `guardrails/` directory contains a layered defense system using the **guardrails-ai** library:

| Module                          | Purpose                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `guardrails/input_sanitizer.py` | Prompt injection detection, control char stripping, delimiter wrapping                                        |
| `guardrails/pii_handler.py`     | PII detection (email, phone, CC, SSN, IP) with reversible and irreversible masking                            |
| `guardrails/validators.py`      | Custom guardrails-ai validators (`RefundReasonValidator`, `PolicyDecisionValidator`, `RefundAmountValidator`) |
| `guardrails/runner.py`          | `RefundGuardRunner` — unified guard orchestrator with LangSmith tracing                                       |

**How it flows:**

1. **API entry** (`main.py` `/refund` endpoint): `sanitize_input()` rejects prompt injections early (HTTP 200 with `status: rejected`)
2. **Pre-workflow** (`_process_refund`): `RefundGuardRunner.validate_input()` runs guardrails-ai schema validation, PII masking, and traces to LangSmith
3. **Agent-level** (`policy_agent.py`): `sanitize_input()` + `mask_pii()` before constructing the HumanMessage
4. **Post-workflow** (`_process_refund`): `RefundGuardRunner.validate_policy_output()` validates business rules on LLM output
5. **All guards** are wrapped in OpenTelemetry spans → visible in LangSmith dashboard

**guardrails-ai validators** are registered with `@register_validator` and enforce:

- Refund reason quality (min/max length, gibberish detection)
- Decision enum enforcement (approved/denied/partial only)
- Amount bounds checking (non-negative, within max allowed)

#### Industry Standard

- **Guardrails AI**: Python library for structured LLM output validation with Pydantic schemas
- **NeMo Guardrails (NVIDIA)**: Conversation-level rails for topic control and safety
- **LlamaGuard (Meta)**: Fine-tuned safety classifier for input/output moderation
- **Rebuff**: Prompt injection detection service (canary tokens + heuristics)
- **LangChain OutputParser + retry**: Validates and retries malformed LLM outputs

---

## Topic 8 — RAGAS Evaluation & Quality Metrics

### Q12. How do you evaluate the quality of a multi-agent RAG system in production?

#### In Our Project

The `evaluation/` directory uses the **ragas** Python library to evaluate agent outputs:

| Component                            | Purpose                                                    |
| ------------------------------------ | ---------------------------------------------------------- |
| `evaluation/ragas_evaluator.py`      | `RefundRAGASEvaluator` class + `run_parallel_evaluation()` |
| `main.py` `/evaluate` endpoint       | On-demand evaluation API                                   |
| Background task in `_process_refund` | Automatic post-workflow evaluation                         |

**Parallel execution architecture:**

```python
# All RAGAS metrics run concurrently via asyncio.gather()
tasks = [
    _evaluate_metric(faithfulness, "faithfulness", sample),
    _evaluate_metric(answer_relevancy, "answer_relevancy", sample),
    _evaluate_metric(context_precision, "context_precision", sample),
    _evaluate_metric(context_recall, "context_recall", sample),  # if ground_truth provided
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Metrics evaluated:**

- **Faithfulness**: Does the agent output stay faithful to retrieved policy context?
- **Answer Relevancy**: Is the refund decision relevant to the customer's request?
- **Context Precision**: Were the retrieved policies precisely relevant?
- **Context Recall**: Did we retrieve all the necessary policies? (requires ground truth)

**LangSmith visibility:** Each metric is traced as an OTEL span with attributes:

- `ragas.overall_score`, `ragas.faithfulness`, `ragas.answer_relevancy`, etc.
- Duration per metric, errors, and task correlation

#### Industry Standard

| Tool                     | Purpose                                                       |
| ------------------------ | ------------------------------------------------------------- |
| **RAGAS**                | Framework-agnostic RAG evaluation with reference-free metrics |
| **DeepEval**             | LLM evaluation with hallucination, toxicity, and bias metrics |
| **LangSmith Evaluators** | Built-in LLM-as-judge evaluators tied to LangSmith datasets   |
| **Arize Phoenix**        | Open-source traces + evaluation for LLM apps                  |
| **TruLens**              | Feedback functions for groundedness, relevance, and context   |
| **Braintrust**           | Evaluation + scoring platform with prompt experiment tracking |

---

## Topic 9 — Resilience: Circuit Breaker, Retry Budget & Model Fallback

### Q13. How do you prevent cascading failures when the LLM provider has degraded performance or is down?

#### In Our Project

The `executor/resilience.py` module implements a three-layer resilience pattern:

**1. Circuit Breaker** (`CircuitBreaker` class)

| State     | Behaviour                                                       |
| --------- | --------------------------------------------------------------- |
| CLOSED    | All requests pass through; consecutive failures are counted     |
| OPEN      | All requests fail-fast immediately (no LLM call made)           |
| HALF_OPEN | One test request allowed; success → CLOSED, failure → re-OPEN   |

Configuration: `failure_threshold=5`, `recovery_timeout=30s`, `success_threshold=2`. Each model in the fallback chain has its own circuit breaker — a failure in the primary doesn't block the fallback.

**2. Retry Budget** (`RetryBudget` class)

Token-bucket style limiter that prevents thundering herd during outages:

- Max 20 retries per 60-second rolling window
- Retry ratio cap: retries cannot exceed 10% of total requests
- Exponential backoff with jitter: `base_delay * 2^attempt * random(0.5, 1.5)`
- When budget is exhausted, raises `RetryBudgetExhausted` — no more retry attempts

**3. Model Fallback Chain** (`ModelFallbackChain` class)

Automatic failover between LLM providers/models:

```
Primary (llama-3.1-8b-instant)
  ↓ circuit open or error
Fallback Fast (llama-3.1-8b-instant on different endpoint)
  ↓ circuit open or error
Fallback Large (llama-3.3-70b-versatile)
```

Each model has its own circuit breaker. The chain respects the retry budget before trying each subsequent model. Used by `react_loop.py` — if the primary LLM call fails, the fallback chain is invoked automatically.

**Integration with LangSmith:** All fallback attempts are traced with OTEL span attributes (`fallback.model_used`, `fallback.attempt`, `fallback.error.*`). The `/resilience/health` endpoint exposes real-time circuit breaker states.

#### Industry Standard

| Pattern | Tool / Library |
| ------- | -------------- |
| **Circuit Breaker** | pybreaker, resilience4j (Java), Polly (.NET), Istio (mesh-level) |
| **Retry with Budget** | tenacity + custom budget, AWS SDK built-in retry budgets, gRPC retry policy |
| **Model Fallback** | LiteLLM Router, Portkey AI Gateway, Azure OpenAI deployment groups |
| **Load Shedding** | Token bucket / leaky bucket at API gateway (Kong, Envoy) |
| **Bulkhead Isolation** | Separate thread/connection pools per downstream service |

---

## Topic 10 — A/B Testing for Prompts

### Q14. How do you run controlled experiments on prompt variations to optimize agent performance?

#### In Our Project

The `evaluation/ab_testing.py` module provides experiment-driven prompt optimization:

**Experiment workflow:**

1. **Create experiment** — Define variants with traffic splits (e.g., 50/50 control vs treatment)
2. **Assign variant** — Deterministic hashing on `request_id` ensures same request always gets same variant (safe for retries)
3. **Record results** — Each workflow completion records the evaluation score per variant
4. **Analyze significance** — Welch's t-test compares variant means with configurable confidence level

**Key features:**

| Feature | Implementation |
| ------- | -------------- |
| Deterministic assignment | SHA-256 hash of `experiment_id:request_id` → bucket |
| Traffic splitting | Configurable per-variant percentage (must sum to 1.0) |
| Statistical significance | Welch's t-test with p-value < 0.05 threshold |
| Confidence intervals | 95% CI per variant for mean score |
| Minimum sample size | Configurable `min_samples_per_variant` before declaring winner |
| LangSmith visibility | Variant info in OTEL spans (`ab_test.variant`, `ab_test.experiment_id`) |

**API endpoints:**

- `POST /experiments` — Create a new experiment
- `GET /experiments` — List all experiments
- `GET /experiments/{id}/report` — Get statistical report with winner
- `POST /experiments/{id}/stop` — End an experiment

**Integration with RAGAS:** Evaluation scores from `RefundRAGASEvaluator` feed into A/B results, enabling comparison of prompt variants on faithfulness, relevancy, precision, and recall.

#### Industry Standard

| Tool | Purpose |
| ---- | ------- |
| **LangSmith Experiments** | Built-in A/B comparison with dataset-driven evaluation |
| **Braintrust** | Prompt scoring with automatic statistical comparison |
| **Weights & Biases Prompts** | Version tracking with experiment comparison |
| **Statsig** | Feature flagging with statistical rigor (Bayesian + frequentist) |
| **LaunchDarkly** | Feature flags with targeting rules and experiment analysis |
| **Optimizely** | Multi-armed bandit experiments for real-time optimization |

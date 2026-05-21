# Dry Run: Complete Request Flow

## Scenario
- **Customer**: alice@example.com
- **Order ID**: 42 (Electronics, $500, delivered 10 days ago)
- **Reason**: "Product screen is cracked, requesting full refund"
- **Customer Tier**: Silver | **Risk Score**: 0.3 | **Refunds in 90 days**: 0

---

## Step 1: User Submits Request (Streamlit UI)

**Component**: `ui/app.py`

```
User fills form:
  ├─ customer_email: "alice@example.com"
  ├─ order_id: 42
  └─ refund_reason: "Product screen is cracked, requesting full refund"

→ HTTP POST /refund → FastAPI (main.py)
```

**UI starts polling**: `GET /refund/{task_id}` every 2 seconds (max 30 polls = 60s timeout)

---

## Step 2: API Receives Request (FastAPI)

**Component**: `main.py` → `POST /refund`

### 2a. Input Sanitization (Pre-check)

```python
# Quick injection check BEFORE any DB/cache work
sanitize_input(request.refund_reason, raise_on_injection=True, field_name="refund_reason")
# "Product screen is cracked, requesting full refund"
# → No injection patterns detected ✓ (proceed)
# → If injection found → return {"status": "rejected", "error": "..."} immediately
```

### 2b. Compute Idempotency Key

```python
# Client may supply an explicit key; otherwise derive from payload hash
idem_key = request.idempotency_key or hashlib.sha256(
    f"alice@example.com:42:Product screen is cracked, requesting full refund".encode()
).hexdigest()

# Result: idem_key = "7f3a9c2b1d..."  (SHA-256 deterministic hash)
```

### 2c. Layer 1 — Redis Fast-Path Check (sub-millisecond)

```python
# Check if this exact request was already submitted
cached_tid = await redis.get(f"idempotency:7f3a9c2b1d...")

# CASE A: Cache HIT (duplicate request)
→ cached_tid = "a1b2c3d4-..."
→ Return: {"task_id": "a1b2c3d4-...", "status": "duplicate", "idempotency_key": "7f3a9c2b1d..."}
→ STOP (no further processing)

# CASE B: Cache MISS (first time OR Redis was flushed)
→ cached_tid = None
→ Proceed to Layer 2
```

### 2d. Layer 2 — PostgreSQL Fallback Check (Redis may have been flushed/restarted)

```sql
-- Query DB for existing idempotency record
SELECT * FROM idempotency_records WHERE idempotency_key = '7f3a9c2b1d...'

-- CASE A: Record EXISTS (Redis was flushed but DB has it)
→ existing.task_id = "a1b2c3d4-..."
→ Re-warm Redis: SET idempotency:7f3a9c2b1d... "a1b2c3d4-..." EX 86400
→ Return: {"task_id": "a1b2c3d4-...", "status": "duplicate", "recovered_from": "db"}
→ STOP

-- CASE B: Record NOT FOUND (genuinely new request)
→ Proceed to generate task_id
```

### 2e. Generate Task ID (New Request)

```python
task_id = str(uuid.uuid4())  # "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

### 2f. Write Idempotency Record to DB First (Durable)

```sql
INSERT INTO idempotency_records (idempotency_key, task_id, status)
VALUES ('7f3a9c2b1d...', 'a1b2c3d4-...', 'processing')

-- CASE A: SUCCESS → proceed
-- CASE B: IntegrityError (race condition — another request inserted same key concurrently)
```

```python
# Race condition handling:
try:
    await session.commit()
except IntegrityError:
    # Another concurrent request won the race
    await session.rollback()
    
    # Fetch the winner's task_id
    SELECT * FROM idempotency_records WHERE idempotency_key = '7f3a9c2b1d...'
    → existing.task_id = "xyz-other-task-id"
    
    # Warm Redis so subsequent calls are fast
    SET idempotency:7f3a9c2b1d... "xyz-other-task-id" EX 86400
    
    → Return: {"task_id": "xyz-other-task-id", "status": "duplicate"}
    → STOP
```

### 2g. Write to Redis Cache (Fast Subsequent Lookups)

```python
# Now that DB has the durable record, cache in Redis for fast-path dedup
await redis.set(f"idempotency:7f3a9c2b1d...", task_id, ex=IDEMPOTENCY_TTL)
# SET idempotency:7f3a9c2b1d... "a1b2c3d4-..." EX 86400  (24-hour TTL)
```

### 2h. Publish Event to Kafka

```python
await event_bus.publish_refund_requested(task_id, payload)

# Kafka topic: "refund.requests"
# Message:
{
  "event_type": "RefundRequested",
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "customer_email": "alice@example.com",
  "order_id": 42,
  "refund_reason": "Product screen is cracked, requesting full refund",
  "status": "pending",
  "enqueued_at": "2026-05-21T10:30:00Z",
  "recovered_state": null
}

# Also stores task metadata in Redis hash (for observability)
HSET refund:task_meta a1b2c3d4-... '{"status":"pending","enqueued_at":"..."}'
```

### 2i. Return Immediately (Fire-and-Forget)

```python
→ Response 200:
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "processing",
  "idempotency_key": "7f3a9c2b1d...",
  "transport": "kafka",
  "topic": "refund.requests"
}
```

### Idempotency Decision Tree (Summary)

```
POST /refund
  │
  ├─ Sanitize input → reject if injection
  │
  ├─ Compute idempotency_key (SHA-256 of email:order:reason)
  │
  ├─ Redis GET idempotency:{key}
  │   ├─ HIT → return {status: "duplicate", task_id} ← FAST (~0.1ms)
  │   └─ MISS ↓
  │
  ├─ PostgreSQL SELECT WHERE idempotency_key = ?
  │   ├─ FOUND → re-warm Redis + return {status: "duplicate"} ← SLOW (~5ms)
  │   └─ NOT FOUND ↓
  │
  ├─ Generate task_id (UUID)
  │
  ├─ PostgreSQL INSERT idempotency_record
  │   ├─ IntegrityError (race) → return {status: "duplicate"}
  │   └─ SUCCESS ↓
  │
  ├─ Redis SET idempotency:{key} = task_id (TTL 24h)
  │
  ├─ Kafka publish RefundRequested event
  │
  └─ Return {task_id, status: "processing"}
```

**State at this point**:
| Store | Key | Value |
|-------|-----|-------|
| PostgreSQL | `idempotency_records` | key="7f3a9c2b1d...", task_id="a1b2c3d4-...", status="processing" |
| Redis | `idempotency:7f3a9c2b1d...` | "a1b2c3d4-..." (TTL 24h) |
| Kafka | `refund.requests` topic | RefundRequested event (uncommitted offset) |
| Redis | `refund:task_meta` | status="pending" |

---

## Step 3: Kafka Consumer Picks Up Event

**Component**: `task_queue_store/kafka_events.py` → background consumer

```python
# Consumer group: "refund-workers"
# Polls Kafka topic "refund.requests"

event = consumer.poll()  # Receives RefundRequested

# Calls handler
await _handle_kafka_event(event)
  → await _process_refund(
      task_id="a1b2c3d4-...",
      customer_email="alice@example.com",
      order_id=42,
      refund_reason="Product screen is cracked, requesting full refund",
      recovered_state=None
    )
```

**Note**: Kafka offset NOT committed yet. If crash here → redelivery on restart.

---

## Step 3.5: _process_refund Entry — Dedup & Context Binding

**Component**: `main.py` → `_process_refund()`

```python
# 3.5a. BIND STRUCTURED LOG CONTEXT (all subsequent logs include these)
structlog.contextvars.clear_contextvars()
structlog.contextvars.bind_contextvars(task_id="a1b2c3d4-...", order_id=42)

# 3.5b. REDIS TASK DEDUP CHECK (guards against Kafka redelivery of already-finished tasks)
existing = await redis.get(f"task:a1b2c3d4-...")

# CASE A: Task already completed (e.g. Kafka redelivered after offset commit failed)
→ existing = '{"status": "completed", "decision": "approved", ...}'
→ prior["status"] in ("completed", "hitl_pending") → True
→ logger.info("task_already_finished") → RETURN immediately (skip all work)

# CASE B: Task not yet done (first attempt)
→ existing = None
→ Proceed to processing

# 3.5c. MARK TASK AS PROCESSING (observability)
await event_bus.mark_processing(task_id)
# Updates Redis hash: HSET refund:task_meta a1b2c3d4-... '{"status":"processing",...}'

logger.info("task_started", customer_email="***example.com")
```

**State at this point**:
| Store | Key | Value |
|-------|-----|-------|
| Redis | `refund:task_meta` | status="processing" (updated from "pending") |

---

## Step 4: Guardrails — Input Validation (Guard Runner)

**Component**: `guardrails/runner.py` → `RefundGuardRunner.validate_input()`

```python
# Called inside _process_refund(), BEFORE workflow invocation:
guard_result = guard_runner.validate_input(customer_email, order_id, refund_reason)
```

```python
# 4a. INPUT SANITIZATION (input_sanitizer.py)
reason = "Product screen is cracked, requesting full refund"

├─ Strip control characters: (none found) ✓
├─ Check length: 52 chars < 500 max ✓
├─ Detect injection patterns:
│   regex check: "ignore previous|you are now|jailbreak|system prompt"
│   → No matches found ✓
└─ Wrap in delimiters:
   sanitized = "<<<USER_INPUT>>>Product screen is cracked, requesting full refund<<<END_USER_INPUT>>>"

# 4b. PII MASKING (pii_handler.py)
email = "alice@example.com"

├─ Email pattern detected: alice@example.com
├─ Mask: "a***@example.com"
└─ Store reversible mapping: {"a***@example.com" → "alice@example.com"}

# 4c. SCHEMA VALIDATION (guardrails-ai if available, else fallback)
# If guardrails-ai installed:
Guard.from_pydantic(RefundInputSchema).validate({
    "customer_email": "alice@example.com",
    "order_id": 42,
    "refund_reason": "Product screen is cracked, requesting full refund"
})
├─ email: valid string ✓
├─ order_id: > 0 ✓
├─ refund_reason: 5 ≤ 52 ≤ 500 chars ✓

# If guardrails-ai NOT available (fallback):
RefundReasonValidator.validate(refund_reason)
├─ Length ≥ 5 chars: 52 ✓
├─ Length ≤ 500 chars ✓
└─ Not gibberish (has real words) ✓

# 4d. TELEMETRY TRACE
span: "guardrails.validate_input"
├─ attributes: {guardrails.type: "input_validation", guardrails.pii_masked: true,
│               guardrails.input_valid: true}
└─ status: OK

# 4e. RETURN (stays in _process_refund scope — not passed to workflow)
→ {"customer_email": "alice@example.com", "masked_email": "a***@example.com",
   "order_id": 42, "refund_reason": "<<<USER_INPUT>>>...<<<END_USER_INPUT>>>",
   "raw_reason": "Product screen is cracked, requesting full refund"}
```

**Result**: Input passes all guardrails. Proceed to workflow.

---

## Step 5: Workflow Entry — Sanitize + Checkpoint Recovery

**Component**: `workflow/graph.py` → `run_workflow()`

```python
# 5a. OPEN TELEMETRY ROOT SPAN
span: "warehouse_refund_workflow"
├─ attributes: {workflow.name: "warehouse_refund_workflow",
│               order.id: 42, customer.email: "***example.com"}

# 5b. SELECT GRAPH (checkpointed vs fallback)
graph = _checkpointed_workflow or refund_workflow
# On Linux: AsyncPostgresSaver-backed graph (LangGraph native checkpointing)
# On Windows: fallback (no native LangGraph checkpointer due to uvicorn/SelectorEventLoop)

config = {"recursion_limit": 50}  # MAX_RECURSION safety cap
if thread_id and _checkpointed_workflow:
    config["configurable"] = {"thread_id": "a1b2c3d4-..."}

# 5c. SECOND SANITIZATION PASS (defense-in-depth at workflow boundary)
safe_reason = sanitize_input(refund_reason, field_name="refund_reason")
# → "<<<USER_INPUT>>>Product screen is cracked, requesting full refund<<<END_USER_INPUT>>>"
masked_email = mask_pii(customer_email)
# → "a***@example.com"

# 5d. BUILD BASE STATE
base_state = {
    "customer_email": "alice@example.com",  # Raw email (tools need it for DB lookups)
    "order_id": 42,
    "refund_reason": "<<<USER_INPUT>>>Product screen is cracked...<<<END_USER_INPUT>>>",  # Sanitized
    "cycle_count": 0,
    "agent_retry_counts": {},
    "thread_id": "a1b2c3d4-...",
}

# 5e. CHECKPOINT RECOVERY CHECK (only when LangGraph native checkpointer NOT available)
# If _checkpointed_workflow is None AND _checkpoint_redis is available:
existing = await load_checkpoint(_checkpoint_redis, "a1b2c3d4-...")

# load_checkpoint() internally:
#   1. Redis fast path: GET wf_checkpoint:a1b2c3d4-... → None
#   2. PostgreSQL fallback: SELECT * FROM workflow_checkpoints WHERE thread_id = '...' → None
#   Result: None (fresh execution)

# CASE A: Checkpoint FOUND (crash recovery scenario):
#   recovered = existing["state"]  # e.g. {"validation_passed": True, "decision": "approved", ...}
#   last_agent = existing["last_completed_agent"]  # e.g. "policy"
#   base_state = {**recovered, "thread_id": "a1b2c3d4-..."}
#   span.set_attribute("workflow.resumed_from", "policy")
#   → Supervisor will see completed fields, skip re-running finished agents

# CASE B: No checkpoint (this run — first attempt):
#   base_state stays as-is
```

---

## Step 6: Build Initial State & Invoke LangGraph

**Component**: `workflow/graph.py` → `run_workflow()` (continued)

```python
# 6a. BUILD INITIAL STATE (add HumanMessage with masked email)
initial_state = {
    **base_state,  # customer_email, order_id, refund_reason, cycle_count, agent_retry_counts, thread_id
    "messages": [
        HumanMessage(content="Process warehouse refund: email=a***@example.com order=42 reason=<<<USER_INPUT>>>Product screen is cracked, requesting full refund<<<END_USER_INPUT>>>")
    ],
}

# Note: All other RefundState fields (customer_id, validation_passed, decision, etc.)
# are Optional[...] in the TypedDict — they default to None when not provided.
# The supervisor checks these None values to decide routing.

# 6b. INVOKE LANGGRAPH STATE MACHINE
result = await graph.ainvoke(initial_state, config)

# Graph execution begins: START → supervisor (first node)
```

**Graph structure**:
```
START ──→ supervisor ──conditional_edges──→ validation ──→ supervisor (loop)
                      │                  → policy ──────→ supervisor (loop)
                      │                  → communication → supervisor (loop)
                      │                  → hitl ─────────→ END
                      └──────────────────→ END
```

---

## Step 7: Supervisor Node — Cycle 1

**Component**: `agents/supervisor.py`

```python
# 7a. INCREMENT CYCLE COUNTER
cycle_count = (state.get("cycle_count") or 0) + 1  # 0 + 1 = 1
span.set_attribute("supervisor.cycle_count", 1)

# 7b. HITL PASS-THROUGH CHECK
# If a previous agent already flagged hitl_required=True, skip LLM entirely:
if state.get("hitl_required"):  # → False (no agent ran yet)
    return {"next_agent": "hitl", ...}
# → Not triggered, proceed

# 7c. CYCLE LIMIT CHECK
if cycle_count > MAX_CYCLES:  # 1 > 15 → False
    return {"next_agent": "hitl", "hitl_required": True, "hitl_reason": "Cycle limit exceeded"}
# → Not triggered, proceed

# 7d. BUILD STATE SUMMARY FOR LLM
state_summary = """
Current workflow state:
- validation_passed: None
- decision: None
- request_saved: None
- error: None

Call the routing tool that best matches the state above.
"""

# 7e. LLM CALL (gpt-4o-mini with tool_choice="any")
# tool_choice="any" → FORCES the model to always call exactly one routing tool
# (no free-text responses allowed — deterministic routing)
response = await _supervisor_llm.ainvoke([
    SystemMessage(content=SUPERVISOR_SYSTEM),  # from prompts/supervisor.md
    HumanMessage(content=state_summary)
])

# Tools available (bound with .bind_tools()):
# - call_validation_agent()  → "Use when validation_passed is None"
# - call_policy_agent()      → "Use when validation_passed=True AND decision=None"
# - call_communication_agent() → "Use when decision is set AND request_saved is not True"
# - finish_workflow()        → "Use when validation_passed=False OR request_saved=True"

# LLM sees validation_passed=None → calls call_validation_agent()
response.tool_calls = [{"name": "call_validation_agent", "args": {}, "id": "..."}]
next_agent = TOOL_TO_AGENT["call_validation_agent"]  # → "validation"

# 7f. PER-AGENT RETRY TRACKING
retry_counts = dict(state.get("agent_retry_counts") or {})  # {}
retry_counts["validation"] = retry_counts.get("validation", 0) + 1  # → 1
# 1 ≤ MAX_AGENT_RETRIES(3) ✓ → proceed (no HITL)

# 7g. TELEMETRY
span: "supervisor_node"
├─ attributes: {supervisor.cycle_count: 1, supervisor.next_agent: "validation",
│               supervisor.reasoning: "LLM→call_validation_agent",
│               supervisor.retry.validation: 1}
└─ status: OK

# 7h. LOG
logger.info("supervisor_routing", next_agent="validation", cycle=1,
            retry_counts={"validation": 1}, reasoning="LLM→call_validation_agent")
```

**State update** (returned by supervisor_node):
```python
{
    "next_agent": "validation",
    "cycle_count": 1,
    "agent_retry_counts": {"validation": 1},
    "messages": [AIMessage(content="Supervisor → validation | LLM→call_validation_agent")]
}
```

**Routing**: `route_supervisor(state)` reads `next_agent="validation"` → conditional edge → `validation` node

---

## Step 8: Validation Agent Node — ReAct Loop

**Component**: `agents/validation_agent.py` + `agents/react_loop.py`  
**LLM Model**: `gpt-4.1-nano` (OPENAI_FAST_MODEL — fast/cheap for structured validation)  
**Max Iterations**: 8 (default)

### Turn 1: LLM Plans Tool Calls

```python
# 8a. BUILD AGENT PROMPT (no re-sanitization — validation only uses email/order_id)
user_message = f"""
Validate this refund request:
- Customer email: alice@example.com
- Order ID: 42

Call all required tools, apply the business rules, then return the JSON result.
"""

messages = [
    SystemMessage(content=VALIDATION_SYSTEM),  # from prompts/validation_agent.md
    HumanMessage(content=user_message),
]

# 8b. REACT LOOP STARTS (react_loop.py)
# iteration=1: LLM call with tools bound
response = await _llm_with_validation_tools.ainvoke(messages)
# (uses get_fallback_chain() on failure → retries with fallback model)

# LLM returns tool_calls:
tool_calls = [
    {"name": "lookup_customer", "args": {"email": "alice@example.com"}, "id": "tc_001"},
    {"name": "get_order_details", "args": {"order_id": 42}, "id": "tc_002"}
]
```

### Turn 1: Wave Scheduling + Execution

```python
# 8c. TOPOLOGICAL WAVE SORT (react_loop.py → _execution_waves())
# DEPENDENCY GRAPH defined in validation_agent.py:
VALIDATION_DEPS = {
    "lookup_customer":        [],                    # Wave 0
    "get_order_details":      [],                    # Wave 0
    "get_customer_analytics": ["lookup_customer"],   # Wave 1 (needs customer_id)
    "verify_order_ownership": ["lookup_customer"],   # Wave 1 (needs customer_id)
}

# Tool calls this turn: [lookup_customer, get_order_details]
# Both have deps=[] → all go into Wave 0
waves = [[lookup_customer, get_order_details]]  # Single wave, both ready

# 8d. IMPORTANT: Only FIRST wave executed per LLM iteration
# If LLM had also called get_customer_analytics (dependent tool), 
# react_loop would TRIM the AIMessage to wave-0 only, preventing hallucinated args.

# 8e. WAVE 0 PARALLEL EXECUTION
t0 = time.perf_counter()
results = await asyncio.gather(
    _call_tool(tool_map, {"name": "lookup_customer", "args": {"email": "alice@example.com"}, ...}, messages),
    _call_tool(tool_map, {"name": "get_order_details", "args": {"order_id": 42}, ...}, messages)
)
# duration: ~90ms (parallel — wall time = max of both)
```

#### Tool: `lookup_customer("alice@example.com")`
```sql
-- DB Query (tools/db_tools.py)
SELECT id, email, name, tier, account_status, total_orders, total_refunds
FROM customers
WHERE email = 'alice@example.com'

-- Result:
→ {"customer_id": 123, "name": "Alice Johnson", "tier": "silver",
   "account_status": "active", "total_orders": 12, "total_refunds": 1}
```

#### Tool: `get_order_details(42)`
```sql
-- DB Query (tools/db_tools.py)
SELECT o.id, o.customer_id, o.amount, o.status, o.purchase_date,
       p.name as product_name, p.category, p.return_window_days,
       c.name as customer_name
FROM orders o
JOIN products p ON o.product_id = p.id
JOIN customers c ON o.customer_id = c.id
WHERE o.id = 42

-- Result:
→ {"order_id": 42, "customer_id": 123, "amount": 500.00, "status": "delivered",
   "purchase_date": "2026-05-11", "product_name": "UltraScreen Monitor",
   "category": "electronics", "return_window_days": 30}
```

```python
# Append tool results to messages
messages.append(ToolMessage(content='{"customer_id": 123, "tier": "silver", ...}', tool_call_id="tc_001"))
messages.append(ToolMessage(content='{"order_id": 42, "amount": 500.00, ...}', tool_call_id="tc_002"))
```

### Turn 2: LLM Plans Next Tools

```python
response = await llm_with_tools.ainvoke(messages)

# LLM sees customer_id=123 from Wave 0, now calls dependent tools:
tool_calls = [
    {"name": "get_customer_analytics", "args": {"customer_id": 123}, "id": "tc_003"},
    {"name": "verify_order_ownership", "args": {"order_id": 42, "customer_id": 123}, "id": "tc_004"}
]
```

### Turn 2: Wave 1 Execution (Parallel)

```python
results = await asyncio.gather(
    get_customer_analytics(123),
    verify_order_ownership(42, 123)
)
```

#### Tool: `get_customer_analytics(123)`
```sql
SELECT * FROM customer_analytics WHERE customer_id = 123

-- Result:
→ {"customer_id": 123, "total_spent": 6000.00, "refund_rate": 0.08,
   "risk_score": 0.3, "last_calculated_at": "2026-05-15T..."}
```

#### Tool: `verify_order_ownership(42, 123)`
```sql
SELECT COUNT(*) FROM orders WHERE id = 42 AND customer_id = 123

-- Result:
→ {"verified": true, "order_belongs_to_customer": true}
```

### Turn 3: LLM Applies Validation Logic

```python
response = await llm_with_tools.ainvoke(messages)
# No tool_calls → LLM returns final text answer

# LLM reasoning:
# ✓ Customer exists (id=123) and account_status="active"
# ✓ Order exists (id=42) and belongs to customer
# ✓ Order status = "delivered"
# ✓ Risk score 0.3 < 0.8 threshold
# ✓ All validation checks pass

# LLM output (structured):
"""
{
  "validation_passed": true,
  "customer_id": 123,
  "customer_name": "Alice Johnson",
  "customer_tier": "silver",
  "customer_risk_score": 0.3,
  "order_amount": 500.00,
  "product_name": "UltraScreen Monitor",
  "product_category": "electronics",
  "validation_reason": "All checks passed: active customer, valid order ownership, delivered status, acceptable risk score"
}
"""
```

### Output Parsing & Post-Processing

```python
# agents/output_parser.py → Parse LLM text into ValidationOutput Pydantic model
parsed = parse_agent_output(final_content, ValidationOutput, agent_name="validation")

validated_output = ValidationOutput(
    validation_passed=True,
    customer_id=123,
    customer_name="Alice Johnson",
    customer_tier="silver",
    customer_risk_score=0.3,
    order_amount=500.00,
    product_name="UltraScreen Monitor",
    product_category="electronics",
    validation_reason="All checks passed..."
)

# 8f. ENSURE REFUND REQUEST RECORD EXISTS (DB)
# Called AFTER validation passes — creates the RefundRequest row for downstream agents
refund_request_id = await ensure_refund_request_id(
    order_id=42,
    customer_id=123,
    reason="Product screen is cracked, requesting full refund"
)
```
```sql
-- tools/db_tools.py → ensure_refund_request_id()
-- First check if it already exists:
SELECT id FROM refund_requests WHERE order_id = 42 AND customer_id = 123

-- Not found → INSERT:
INSERT INTO refund_requests (order_id, customer_id, reason, status)
VALUES (42, 123, 'Product screen is cracked, requesting full refund', 'pending')
RETURNING id;

→ refund_request_id = 99
```

### Checkpoint Save

```python
# checkpoint/store.py → save_checkpoint()

# Redis (hot path):
SET wf_checkpoint:a1b2c3d4-... '{
  "last_completed_agent": "validation",
  "state": {"customer_id": 123, "validation_passed": true, "customer_tier": "silver", ...}
}' EX 86400  # TTL: 24 hours

# PostgreSQL (cold path):
INSERT INTO workflow_checkpoints (thread_id, last_completed_agent, state_json)
VALUES ('a1b2c3d4-...', 'validation', '{"customer_id": 123, ...}')
ON CONFLICT (thread_id) DO UPDATE SET ...
```

**State after Step 8** (validation_agent_node returns):
```python
# Returned dict merged into RefundState:
{
    "refund_request_id": 99,         # ← NEW: created by ensure_refund_request_id
    "customer_id": 123,
    "customer_name": "Alice Johnson",
    "customer_email": "alice@example.com",
    "customer_tier": "silver",
    "customer_risk_score": 0.3,
    "order_amount": 500.00,
    "order_date": "2026-05-11",
    "order_status": "delivered",
    "product_name": "UltraScreen Monitor",
    "product_category": "electronics",
    "payment_method": "credit_card",
    "validation_passed": True,
    "validation_reason": "All checks passed...",
    "decision": None,              # ← still empty
    "cycle_count": 1,
    "agent_retry_counts": {"validation": 1},
    "messages": [..., AIMessage(content="Validation passed for order #42")]
}
```

---

## Step 9: Supervisor Node — Cycle 2

**Component**: `agents/supervisor.py`

```python
# 9a. INCREMENT + SAFETY CHECKS
cycle_count = 1 + 1 = 2  # 2 < MAX_CYCLES(15) ✓
# hitl_required = False ✓ (validation passed normally)

# 9b. STATE SUMMARY FOR LLM
state_summary = """
- validation_passed: True
- decision: None
- request_saved: None
- error: None
"""
# LLM sees validation_passed=True AND decision=None
# → Tool description match: call_policy_agent() 
# → "Use when validation_passed is True AND decision is None"

# 9c. LLM CALL (tool_choice="any")
response.tool_calls = [{"name": "call_policy_agent", ...}]
next_agent = "policy"

# 9d. PER-AGENT RETRY TRACKING
retry_counts = {"validation": 1}  # from prior state
retry_counts["policy"] = 0 + 1 = 1  # First invocation of policy
# 1 ≤ MAX_AGENT_RETRIES(3) ✓

# 9e. RETURN
→ {"next_agent": "policy", "cycle_count": 2, 
   "agent_retry_counts": {"validation": 1, "policy": 1},
   "messages": [AIMessage(content="Supervisor → policy | LLM→call_policy_agent")]}
```

**Routing**: → `policy` node

---

## Step 10: Policy Agent Node — ReAct Loop

**Component**: `agents/policy_agent.py` + `agents/react_loop.py`  
**LLM Model**: `gpt-4o-mini` (OPENAI_MAIN_MODEL — stronger reasoning for policy decisions)  
**Max Iterations**: 12 (higher than default 8 — policy needs more tool calls)

### Pre-processing: Re-sanitize User Input (Defense-in-Depth)

```python
# Policy agent re-sanitizes because it passes refund_reason to LLM prompt
raw_reason = state.get("refund_reason", "")  # Already sanitized at workflow entry
safe_reason = sanitize_input(raw_reason, field_name="refund_reason")
safe_reason = mask_pii(safe_reason)  # Mask any PII in the reason text

# Build rich prompt with validated state data:
user_message = f"""
Process this refund request:
- Customer: Alice Johnson (ID: 123, Tier: silver, Risk: 0.3)
- Order: #42, Amount: $500.0, Date: 2026-05-11
- Product: UltraScreen Monitor (Category: electronics)
- Refund Reason: <<<USER_INPUT>>>Product screen is cracked, requesting full refund<<<END_USER_INPUT>>>

Use your tools step by step, then return the JSON decision.
"""
```

### Turn 1: Wave 0 Execution (7 tools in parallel)

```python
# LLM plans all initial tools at once:
tool_calls = [
    {"name": "search_policies", "args": {"query": "cracked screen electronics refund"}, "id": "tc_010"},
    {"name": "check_refund_history", "args": {"customer_id": 123}, "id": "tc_011"},
    {"name": "check_eligibility", "args": {"order_id": 42, "customer_tier": "silver"}, "id": "tc_012"},
    {"name": "get_product_return_policy", "args": {"product_category": "electronics"}, "id": "tc_013"},
    {"name": "get_customer_order_summary", "args": {"customer_id": 123}, "id": "tc_014"},
    {"name": "get_refund_rate_by_category", "args": {"product_category": "electronics"}, "id": "tc_015"},
    {"name": "get_similar_refund_decisions", "args": {"product_category": "electronics", "customer_tier": "silver"}, "id": "tc_016"}
]
```

#### Tool: `search_policies("cracked screen electronics refund")`
```python
# cache/redis_cache.py → Semantic Cache Check

# 1. Embed query
embedding = openai.embed("cracked screen electronics refund")  # → [0.12, -0.45, ...]

# 2. Check Redis semantic cache
KEYS policy_cache:*
# For each cached key: compute cosine_similarity(query_embedding, cached_embedding)
# Result: No cache hit (similarity < 0.92 threshold)

# 3. Cache MISS → Query Pinecone (vectordb/pinecone_store.py)
results = pinecone_index.query(
    vector=embedding,
    top_k=5,
    namespace="refund-policies",
    include_metadata=True
)

# Pinecone results:
→ [
    {"score": 0.89, "text": "Electronics items with physical damage eligible for full refund within 30 days..."},
    {"score": 0.85, "text": "Defective product policy: manufacturer defect vs user damage distinction..."},
    {"score": 0.82, "text": "Silver tier customers receive +7 day extension on return windows..."}
  ]

# 4. Cache result in Redis (for future similar queries)
SET policy_cache:cracked_screen_elec_abc123 '{
  "embedding": [0.12, -0.45, ...],
  "result": [{"text": "Electronics items...", ...}],
  "created_at": "2026-05-21T10:30:05Z"
}' EX 3600  # TTL: 1 hour
```

#### Tool: `check_refund_history(123)`
```sql
-- Count refunds in last 90 days
SELECT COUNT(*) FROM processed_requests pr
JOIN refund_requests rr ON pr.refund_request_id = rr.id
WHERE rr.customer_id = 123
  AND pr.created_at > NOW() - INTERVAL '90 days'
  AND pr.decision IN ('approved', 'partial')

-- Result:
→ {"refund_count_90_days": 0, "refund_count_this_year": 1}
```

#### Tool: `check_eligibility(42, "silver")`
```python
# Business logic:
order_date = "2026-05-11"
today = "2026-05-21"
days_since_purchase = 10

base_return_window = 30  # electronics
tier_extension = 7       # silver tier
effective_window = 37    # 30 + 7

# 10 days < 37 days → ELIGIBLE
→ {"eligible": true, "days_remaining": 27, "effective_window_days": 37,
   "base_window": 30, "tier_extension": 7}
```

#### Tool: `get_product_return_policy("electronics")`
```sql
SELECT return_window_days, restocking_fee_pct
FROM products WHERE category = 'electronics' LIMIT 1

→ {"return_window_days": 30, "restocking_fee_pct": 10.0,
   "policy_notes": "Physical damage claims require photo evidence for orders > $200"}
```

#### Tool: `get_customer_order_summary(123)`
```sql
SELECT COUNT(*) as total_orders, SUM(amount) as total_spent,
       AVG(amount) as avg_order_value
FROM orders WHERE customer_id = 123

→ {"total_orders": 12, "total_spent": 6000.00, "avg_order_value": 500.00}
```

#### Tool: `get_refund_rate_by_category("electronics")`
```sql
SELECT COUNT(CASE WHEN pr.decision='approved' THEN 1 END)::float / COUNT(*) as approval_rate
FROM processed_requests pr
JOIN refund_requests rr ON pr.refund_request_id = rr.id
JOIN orders o ON rr.order_id = o.id
JOIN products p ON o.product_id = p.id
WHERE p.category = 'electronics'

→ {"category": "electronics", "approval_rate": 0.75, "total_requests": 48}
```

#### Tool: `get_similar_refund_decisions("electronics", "silver")`
```sql
SELECT pr.decision, pr.refund_amount, pr.policy_applied, rr.reason
FROM processed_requests pr
JOIN refund_requests rr ON pr.refund_request_id = rr.id
JOIN orders o ON rr.order_id = o.id
JOIN products p ON o.product_id = p.id
JOIN customers c ON rr.customer_id = c.id
WHERE p.category = 'electronics' AND c.tier = 'silver'
ORDER BY pr.created_at DESC LIMIT 5

→ [
    {"decision": "approved", "amount": 450, "policy": "silver_tier_electronics"},
    {"decision": "approved", "amount": 280, "policy": "silver_tier_electronics"},
    {"decision": "partial", "amount": 150, "policy": "silver_tier_electronics_late"},
    ...
  ]
```

### Turn 2: Wave 1 (Dependent Tool)

```python
# LLM sees eligibility result, now calls calculate_refund
tool_calls = [
    {"name": "calculate_refund", "args": {
        "order_id": 42,
        "customer_tier": "silver",
        "eligibility_result": {"eligible": true, "effective_window_days": 37, ...}
    }, "id": "tc_017"}
]
```

#### Tool: `calculate_refund(42, "silver", eligibility)`
```python
# Business logic (tools/policy_tools.py):
order_amount = 500.00
restocking_fee_pct = 10.0  # electronics base: 10%

# Silver tier: 50% restocking fee waiver
effective_restocking_fee = 10.0 * 0.5 = 5.0%  # 5% after waiver
restocking_fee = 500.00 * 0.05 = $25.00

# Final refund calculation
refund_amount = 500.00 - 25.00 = $475.00

→ {"refund_amount": 475.00, "restocking_fee": 25.00,
   "restocking_fee_pct_applied": 5.0, "tier_discount": "50% waiver",
   "calculation_breakdown": "order=$500 - restocking_fee=$25 (5% after silver waiver) = $475"}
```

### Turn 3: LLM Applies Policy Decision

```python
# LLM reasoning with ALL tool results:
#
# ✓ Eligible (within 37-day window, only 10 days since purchase)
# ✓ Refund history clean (0 refunds in 90 days < 3 threshold)
# ✓ Refund rate OK (0.08 < 0.5 threshold)
# ✓ Risk score OK (0.3 < 0.8 threshold)
# ✓ Category approval rate high (75%)
# ✓ Similar cases approved
# ✓ Calculated amount: $475
#
# Decision: APPROVED

# LLM output:
"""
{
  "decision": "approved",
  "refund_amount": 475.00,
  "policy_applied": "silver_tier_electronics_standard",
  "policy_reason": "Order within return window (10/37 days). Silver tier 50% restocking fee waiver applied. No fraud indicators. Refund amount: $475 ($500 - $25 restocking fee).",
  "refund_count_90_days": 0
}
"""
```

### Output Parsing (Pydantic Model)

```python
# agents/output_parser.py → parse_agent_output()
parsed = parse_agent_output(final_content, PolicyOutput, agent_name="policy")

# PolicyOutput Pydantic model validates structure:
PolicyOutput(
    decision="approved",        # must be str
    refund_amount=475.00,       # must be float
    policy_applied="silver_tier_electronics_standard",
    policy_reason="Order within return window...",
    refund_count_90_days=0
)

# If parse fails → AgentOutputParseError → policy returns decision="denied" with error reason
# (does NOT trigger HITL — just denies safely)

# Telemetry attributes:
span.set_attribute("policy.decision", "approved")
span.set_attribute("policy.refund_amount", "475.0")
```

**Note**: Full guardrails output validation (PolicyDecisionValidator, RefundAmountValidator) 
happens LATER in `_process_refund()` after workflow returns — see Step 14.

### Checkpoint Save

```python
# Redis:
SET wf_checkpoint:a1b2c3d4-... '{
  "last_completed_agent": "policy",
  "state": {"validation_passed": true, "decision": "approved", "refund_amount": 475, ...}
}' EX 86400

# PostgreSQL:
UPDATE workflow_checkpoints SET last_completed_agent='policy', state_json='...'
WHERE thread_id = 'a1b2c3d4-...'
```

**State after Step 10**:
```python
state = {
    ...
    "validation_passed": True,
    "decision": "approved",
    "refund_amount": 475.00,
    "policy_applied": "silver_tier_electronics_standard",
    "policy_reason": "Order within return window...",
    "refund_count_90_days": 0,
    "request_saved": None,  # ← still empty
    "cycle_count": 2,
    ...
}
```

---

## Step 11: Supervisor Node — Cycle 3

**Component**: `agents/supervisor.py`

```python
# 11a. SAFETY CHECKS
cycle_count = 2 + 1 = 3  # 3 < MAX_CYCLES(15) ✓
# hitl_required = False ✓

# 11b. STATE SUMMARY
state_summary = """
- validation_passed: True
- decision: approved
- request_saved: None
- error: None
"""
# LLM sees decision="approved" (not None) AND request_saved=None (not True)
# → Tool match: call_communication_agent()
# → "Use when decision is set AND request_saved is not True"

# 11c. PER-AGENT RETRY TRACKING
retry_counts = {"validation": 1, "policy": 1}
retry_counts["communication"] = 0 + 1 = 1  # First invocation
# 1 ≤ MAX_AGENT_RETRIES(3) ✓

# 11d. RETURN
→ {"next_agent": "communication", "cycle_count": 3,
   "agent_retry_counts": {"validation": 1, "policy": 1, "communication": 1},
   "messages": [AIMessage(content="Supervisor → communication | LLM→call_communication_agent")]}
```

**Routing**: → `communication` node

---

## Step 12: Communication Agent Node — ReAct Loop

**Component**: `agents/communication_agent.py` + `agents/react_loop.py`  
**LLM Model**: `gpt-4.1-nano` (OPENAI_FAST_MODEL — simple tool orchestration)  
**Max Iterations**: 8 (default)

### Pre-requisite: Ensure RefundRequest Exists

```python
# Communication agent double-checks refund_request_id exists in state
refund_request_id = state.get("refund_request_id")  # → 99 (set by validation agent in Step 8)

# If missing (edge case): create it now
if not refund_request_id and state.get("order_id") and state.get("customer_id"):
    refund_request_id = await ensure_refund_request_id(order_id=42, customer_id=123, reason="...")

# In our flow: already 99 from Step 8f ✓
```

### Build Prompt

```python
user_message = f"""
Persist the following refund decision:
- Customer ID: 123
- Customer Name: Alice Johnson
- Order ID: 42
- Refund Request ID: 99
- Decision: approved
- Refund Amount: $475.0
- Policy Applied: silver_tier_electronics_standard
- Policy Reason: Order within return window (10/37 days)...

Call all four tools step by step, then return the JSON confirmation.
IMPORTANT: When calling update_customer_analytics, pass refund_request_id=99 for idempotency.
"""
```

### Turn 1: Wave 0 (All 4 tools in parallel — no dependencies)

```python
tool_calls = [
    {"name": "save_processed_request", "args": {...}, "id": "tc_020"},
    {"name": "update_customer_analytics", "args": {...}, "id": "tc_021"},
    {"name": "update_refund_request_status", "args": {...}, "id": "tc_022"},
    {"name": "log_audit_event", "args": {...}, "id": "tc_023"}
]

results = await asyncio.gather(
    save_processed_request(...),
    update_customer_analytics(...),
    update_refund_request_status(...),
    log_audit_event(...)
)
```

#### Tool: `save_processed_request(42, 123, 99, "approved", 475.00, ...)`
```sql
-- Idempotent insert using SAVEPOINT (nested transaction)
BEGIN;
SAVEPOINT sp_save_request;

INSERT INTO processed_requests (
    refund_request_id, order_id, customer_id, decision, refund_amount,
    policy_applied, policy_reason, analytics_updated
) VALUES (99, 42, 123, 'approved', 475.00,
          'silver_tier_electronics_standard', 'Order within return window...', false)
RETURNING id;

-- If UniqueViolation on refund_request_id:
--   ROLLBACK TO sp_save_request;
--   SELECT * FROM processed_requests WHERE refund_request_id = 99;
--   return {deduplicated: true, ...}

RELEASE SAVEPOINT sp_save_request;
COMMIT;

→ {"id": 550, "deduplicated": false, "saved": true}
```

#### Tool: `update_customer_analytics(123, 99)`
```sql
-- Idempotency check: skip if already processed
SELECT analytics_updated FROM processed_requests WHERE refund_request_id = 99
→ analytics_updated = false  (proceed)

-- Atomic read-calculate-write with row lock
BEGIN;

SELECT total_orders, total_refunds FROM customers WHERE id = 123 FOR UPDATE;
→ total_orders=12, total_refunds=1

-- Increment refund count
UPDATE customers SET total_refunds = 2 WHERE id = 123;

-- Recalculate analytics
new_refund_rate = 2 / 12 = 0.167
new_risk_score = MIN(0.167 * 1.5, 1.0) = 0.25

INSERT INTO customer_analytics (customer_id, total_spent, refund_rate, risk_score, last_calculated_at)
VALUES (123, 6000.00, 0.167, 0.25, NOW())
ON CONFLICT (customer_id) DO UPDATE SET
    refund_rate = 0.167,
    risk_score = 0.25,
    last_calculated_at = NOW();

-- Mark as processed (idempotency flag)
UPDATE processed_requests SET analytics_updated = true WHERE refund_request_id = 99;

COMMIT;

→ {"updated": true, "new_refund_rate": 0.167, "new_risk_score": 0.25}
```

#### Tool: `update_refund_request_status(99, "completed")`
```sql
UPDATE refund_requests SET status = 'completed', updated_at = NOW()
WHERE id = 99

→ {"updated": true}
```

#### Tool: `log_audit_event("communication", "save_processed_request", ...)`
```sql
INSERT INTO audit_logs (agent_name, tool_called, input_data, output_data, status, duration_ms, created_at)
VALUES ('communication', 'save_processed_request',
        '{"order_id":42,"decision":"approved","amount":475}',
        '{"id":550,"saved":true}',
        'success', 45, NOW())

→ {"logged": true, "audit_id": 1234}
```

### Turn 2: LLM Confirms Success

```python
# LLM sees all 4 tools succeeded
# No more tool_calls → returns final output:
"""
{
  "request_saved": true,
  "analytics_updated": true,
  "status_updated": true,
  "confirmation_message": "Refund of $475.00 approved for order #42. Customer Alice Johnson (Silver tier) will receive refund minus $25 restocking fee."
}
"""
```

### Checkpoint Save

```python
# Redis:
SET wf_checkpoint:a1b2c3d4-... '{
  "last_completed_agent": "communication",
  "state": {"request_saved": true, "analytics_updated": true, ...}
}' EX 86400

# PostgreSQL:
UPDATE workflow_checkpoints SET last_completed_agent='communication', state_json='...'
WHERE thread_id = 'a1b2c3d4-...'
```

**State after Step 12**:
```python
state = {
    ...
    "validation_passed": True,
    "decision": "approved",
    "refund_amount": 475.00,
    "request_saved": True,
    "analytics_updated": True,
    "status_updated": True,
    "cycle_count": 3,
    ...
}
```

---

## Step 13: Supervisor Node — Cycle 4 (Final)

**Component**: `agents/supervisor.py`

```python
# 13a. SAFETY CHECKS
cycle_count = 3 + 1 = 4  # 4 < MAX_CYCLES(15) ✓
# hitl_required = False ✓

# 13b. STATE SUMMARY
state_summary = """
- validation_passed: True
- decision: approved
- request_saved: True
- error: None
"""
# LLM sees request_saved=True
# → Tool match: finish_workflow()
# → "Use when validation_passed=False OR request_saved=True"

# 13c. finish_workflow() returns "end" — no retry tracking for "end"
next_agent = "end"

# 13d. RETURN
→ {"next_agent": "end", "cycle_count": 4,
   "agent_retry_counts": {"validation": 1, "policy": 1, "communication": 1},
   "messages": [AIMessage(content="Supervisor → end | LLM→finish_workflow")]}
```

**Routing**: `route_supervisor(state)` reads `next_agent="end"` → returns `END` → graph terminates

---

## Step 14: Workflow Completion — Checkpoint Cleanup (Inside run_workflow)

**Component**: `workflow/graph.py` → `run_workflow()` (post-graph)

```python
# 14a. GRAPH RETURNS FINAL STATE
result = await graph.ainvoke(initial_state, config)
# result = full RefundState dict with all fields populated

# 14b. SET TELEMETRY ATTRIBUTES ON ROOT SPAN
root_span.set_attribute("workflow.decision", "approved")
root_span.set_attribute("workflow.compensated", False)

# 14c. DETERMINE IF WORKFLOW IS TRULY DONE
workflow_done = result.get("request_saved") or result.get("compensated")
# → True (request_saved=True)
hitl_pending = result.get("hitl_required")
# → False

# 14d. DELETE CHECKPOINT (only if done AND not HITL)
# Purpose: prevent stale checkpoint from being replayed on next fresh request
if workflow_done and not hitl_pending and thread_id and _checkpoint_redis:
    await delete_checkpoint(_checkpoint_redis, "a1b2c3d4-...")

# delete_checkpoint() internally:
#   Redis: DEL wf_checkpoint:a1b2c3d4-...
#   PostgreSQL: DELETE FROM workflow_checkpoints WHERE thread_id = 'a1b2c3d4-...'

# 14e. RETURN RESULT TO _process_refund()
return result
# → {"decision": "approved", "refund_amount": 475.00, "validation_passed": True,
#    "request_saved": True, "hitl_required": False, "compensated": False, ...}
```

---

## Step 15: Output Guardrails Validation

**Component**: `main.py` → `_process_refund()` (back from run_workflow)

```python
# 15a. CHECK HITL
if result.get("hitl_required"):  # → False
    # Would persist HITL state and return early — skipped here

# 15b. LOG COMPLETION
logger.info("task_completed", decision="approved", refund_amount=475.00, compensated=False)

# 15c. GUARDRAILS-AI OUTPUT VALIDATION (traced to LangSmith)
if result.get("decision"):  # → "approved" (truthy)
    guard_runner.validate_policy_output(
        decision="approved",
        refund_amount=475.00,
        order_amount=500.00  # from result
    )

# validate_policy_output() internally:
# span: "guardrails.validate_policy_output"
#
# PolicyDecisionValidator:
#   decision ∈ {"approved", "denied", "partial"} → "approved" ✓
#
# RefundAmountValidator:
#   amount ≥ 0 → 475.00 ✓
#   amount ≤ 10000 → 475.00 ✓
#
# Business rule:
#   refund_amount ≤ order_amount → 475.00 ≤ 500.00 ✓
#
# → {"valid": True, "errors": []}
```

---

## Step 16: Background RAGAS Evaluation (Non-blocking)

**Component**: `evaluation/ragas_evaluator.py` (launched from `_process_refund`)

```python
# Launched as fire-and-forget async task — does NOT block result delivery
asyncio.create_task(_run_ragas_evaluation(
    task_id="a1b2c3d4-...",
    refund_reason="Product screen is cracked, requesting full refund",
    result=result
))

# _run_ragas_evaluation() builds:
question = "Process refund for order #42: Product screen is cracked, requesting full refund"
answer = "Decision: approved, Amount: $475.0, Reason: Order within return window..."
contexts = [
    "silver_tier_electronics_standard",
    "Order within return window (10/37 days)...",
    "All checks passed..."
]

# Calls RAGAS evaluator:
await ragas_evaluator.evaluate_workflow_run(
    task_id="a1b2c3d4-...",
    question=question,
    answer=answer,
    contexts=contexts
)

# Computes metrics (traced to LangSmith):
# - faithfulness: 0.92 (answer grounded in contexts)
# - answer_relevancy: 0.88 (answer addresses the question)
# - context_precision: 0.85 (retrieved contexts are relevant)
```

---

## Step 17: Persist Final Result & Mark Task Done

**Component**: `main.py` → `_process_refund()` (continued)

```python
# 17a. BUILD OUTPUT DICT
output = {
    "status": "completed",
    "decision": "approved",
    "refund_amount": 475.00,
    "policy_reason": "Order within return window...",
    "policy_applied": "silver_tier_electronics_standard",
    "validation_passed": True,
    "validation_reason": "All checks passed...",
    "customer_name": "Alice Johnson",
    "customer_tier": "silver",
    "order_id": 42,
    "compensated": False,
}

# 17b. STORE RESULT IN REDIS (fast lookup for polling)
SET task:a1b2c3d4-... '{...output as JSON...}' EX 3600  # TTL: 1 hour

# 17c. PERMANENT DB RECORD (survives Redis TTL)
INSERT INTO task_results (task_id, status, decision, refund_amount, validation_passed, compensated, error)
VALUES ('a1b2c3d4-...', 'completed', 'approved', 475.00, true, false, null)
ON CONFLICT (task_id) DO UPDATE SET status='completed', ...

# 17d. UPDATE IDEMPOTENCY RECORD
UPDATE idempotency_records SET status = 'completed'
WHERE task_id = 'a1b2c3d4-...'

# 17e. MARK TASK DONE IN KAFKA EVENT BUS
HSET refund:task_meta a1b2c3d4-... '{"status":"completed","completed_at":"2026-05-21T10:30:12Z"}'

# 17f. KAFKA OFFSET COMMITTED (by consumer loop after handler returns successfully)
# ← Critical! Only now is the message acknowledged.
# If crash before this line → Kafka redelivers → Step 3.5b dedup catches it.
```

**Final state across all stores**:
| Store | Key | Status |
|-------|-----|--------|
| Redis | `task:a1b2c3d4-...` | completed (TTL 1h) |
| Redis | `idempotency:7f3a9c2b1d...` | "a1b2c3d4-..." (TTL 24h) |
| Redis | `refund:task_meta` | completed |
| Redis | `wf_checkpoint:a1b2c3d4-...` | DELETED |
| Redis | `policy_cache:cracked_screen_*` | cached (TTL 1h) |
| PostgreSQL | `idempotency_records` | status="completed" |
| PostgreSQL | `task_results` | completed, decision="approved", amount=475 |
| PostgreSQL | `processed_requests` | approved, $475, analytics_updated=true |
| PostgreSQL | `refund_requests` | status="completed" |
| PostgreSQL | `customers` | total_refunds=2 |
| PostgreSQL | `customer_analytics` | refund_rate=0.167, risk_score=0.25 |
| PostgreSQL | `audit_logs` | new entry for communication agent |
| PostgreSQL | `workflow_checkpoints` | DELETED |
| Kafka | `refund.requests` | offset committed (message acknowledged) |

---

## Step 18: User Receives Response (Polling)

**Component**: `main.py` → `GET /refund/{task_id}` + `ui/app.py`

```python
# Streamlit poll #4 (at ~8 seconds):
GET /refund/a1b2c3d4-...

# FastAPI handler — 3-layer lookup cascade:

# Layer 1: Redis (fast path — full result with all fields)
result = await redis.get("task:a1b2c3d4-...")
→ '{"status": "completed", "decision": "approved", "refund_amount": 475.00, ...}'
→ Return json.loads(result) ← DONE (most common path)

# Layer 2 (fallback): TaskResult table — permanent, survives Redis TTL expiry
# (Only reached if Redis key expired or was flushed)
SELECT * FROM task_results WHERE task_id = 'a1b2c3d4-...'
→ If found: return full result with source="db"

# Layer 3 (fallback): IdempotencyRecord — minimal status only
# (Only reached if task not yet finished or DB write partially failed)
SELECT * FROM idempotency_records WHERE task_id = 'a1b2c3d4-...'
→ If found: return {"status": "processing", "task_id": "...", "source": "idempotency_db"}

# Layer 4: Not found at all
→ Return {"status": "not_found", "task_id": "..."}
```

```python
# In our scenario — Layer 1 hits:
→ Response 200:
{
  "status": "completed",
  "decision": "approved",
  "refund_amount": 475.00,
  "policy_applied": "silver_tier_electronics_standard",
  "policy_reason": "Order within return window (10/37 days). Silver tier 50% restocking fee waiver applied.",
  "validation_passed": true,
  "validation_reason": "All checks passed...",
  "customer_name": "Alice Johnson",
  "customer_tier": "silver",
  "order_id": 42,
  "compensated": false
}

# UI displays:
# ┌──────────────────────────────────────┐
# │  ✅ REFUND APPROVED                  │
# │  Amount: $475.00                     │
# │  Policy: Silver Tier Electronics     │
# │  Reason: Within return window...     │
# └──────────────────────────────────────┘
```

---

## Telemetry Trace (Full Span Tree)

```
warehouse_refund_workflow [12.3s]                              ← Root span (Step 5a)
│
├─ [inside _process_refund, before run_workflow]
│  └─ guardrails.validate_input [15ms]                        ← Step 4
│      ├─ sanitize_input [2ms]
│      ├─ mask_pii [3ms]
│      └─ schema_validation [10ms]
│
├─ [inside run_workflow]
│  ├─ supervisor_node (cycle=1) [800ms]                       ← Step 7
│  │   └─ llm.invoke (gpt-4o-mini, tool_choice=any) [750ms]
│  │
│  ├─ validation_agent [3.2s]                                 ← Step 8
│  │   ├─ react.iteration.1 [1.8s]
│  │   │   ├─ llm.invoke (gpt-4.1-nano) [600ms]
│  │   │   ├─ tool.lookup_customer [85ms]          ─┐ Wave 0
│  │   │   └─ tool.get_order_details [92ms]        ─┘ (parallel)
│  │   ├─ react.iteration.2 [1.0s]
│  │   │   ├─ llm.invoke [550ms]
│  │   │   ├─ tool.get_customer_analytics [78ms]   ─┐ Wave 1
│  │   │   └─ tool.verify_order_ownership [65ms]   ─┘ (parallel)
│  │   └─ react.iteration.3 [400ms]
│  │       └─ llm.invoke [400ms] (final reasoning, no tools)
│  │
│  ├─ checkpoint.save (validation) [25ms]                     ← After Step 8
│  │
│  ├─ supervisor_node (cycle=2) [780ms]                       ← Step 9
│  │   └─ llm.invoke [730ms]
│  │
│  ├─ policy_agent [4.5s]                                     ← Step 10
│  │   ├─ react.iteration.1 [2.8s]
│  │   │   ├─ llm.invoke (gpt-4o-mini) [700ms]
│  │   │   ├─ tool.search_policies [350ms]         ─┐
│  │   │   ├─ tool.check_refund_history [70ms]     │
│  │   │   ├─ tool.check_eligibility [45ms]        │ Wave 0
│  │   │   ├─ tool.get_product_return_policy [55ms]│ (all 7
│  │   │   ├─ tool.get_customer_order_summary [60ms]│ parallel)
│  │   │   ├─ tool.get_refund_rate_by_category [65ms]│
│  │   │   └─ tool.get_similar_refund_decisions [80ms]─┘
│  │   ├─ react.iteration.2 [1.2s]
│  │   │   ├─ llm.invoke [650ms]
│  │   │   └─ tool.calculate_refund [30ms]           Wave 1
│  │   └─ react.iteration.3 [500ms]
│  │       └─ llm.invoke [500ms] (final reasoning)
│  │
│  ├─ checkpoint.save (policy) [22ms]                         ← After Step 10
│  │
│  ├─ supervisor_node (cycle=3) [760ms]                       ← Step 11
│  │   └─ llm.invoke [710ms]
│  │
│  ├─ communication_agent [1.8s]                              ← Step 12
│  │   ├─ react.iteration.1 [1.4s]
│  │   │   ├─ llm.invoke (gpt-4.1-nano) [600ms]
│  │   │   ├─ tool.save_processed_request [120ms]    ─┐
│  │   │   ├─ tool.update_customer_analytics [150ms]  │ Wave 0
│  │   │   ├─ tool.update_refund_request_status [80ms]│ (all 4
│  │   │   └─ tool.log_audit_event [65ms]            ─┘ parallel)
│  │   └─ react.iteration.2 [400ms]
│  │       └─ llm.invoke [400ms] (confirm success)
│  │
│  ├─ checkpoint.save (communication) [20ms]                  ← After Step 12
│  │
│  ├─ supervisor_node (cycle=4) [700ms]                       ← Step 13
│  │   └─ llm.invoke [650ms]
│  │
│  └─ checkpoint.delete [15ms]                                ← Step 14
│
├─ [back in _process_refund, after run_workflow returns]
│  ├─ guardrails.validate_policy_output [8ms]                 ← Step 15
│  └─ persist_result [45ms]                                   ← Step 17
│
└─ [async background task]
   └─ ragas.evaluate [2.5s]                                   ← Step 16
       ├─ faithfulness [800ms]
       ├─ answer_relevancy [850ms]
       └─ context_precision [850ms]
```

---

## Error/Recovery Scenario: Crash After Policy Agent

What happens if the system crashes after Step 10 (policy complete) but before Step 12 (communication)?

```
1. Kafka offset was NOT committed (crash before Step 17f)
2. On restart: Kafka consumer redelivers the RefundRequested event
3. Consumer calls _process_refund() again with same task_id

4. Step 3.5b — Redis task dedup:
   GET task:a1b2c3d4-...
   → None (never wrote "completed" before crash)
   → Proceed (not a duplicate)

5. Step 5e — Checkpoint recovery:
   GET wf_checkpoint:a1b2c3d4-...
   → {"last_completed_agent": "policy", "state": {"decision": "approved", ...}}

6. base_state = recovered state (validation & policy fields already filled)

7. Graph re-invoked with recovered initial_state

8. Supervisor Cycle 1 (of recovered run) sees:
   - validation_passed: True ✓ (from checkpoint)
   - decision: "approved" ✓ (from checkpoint)
   - request_saved: None (not yet done)
   → Routes to communication agent (skips validation + policy!)

9. Communication agent runs (idempotent tools):
   - save_processed_request: SAVEPOINT catches duplicate if partial write happened
   - update_customer_analytics: checks analytics_updated flag before recalculating
   - Result: exactly-once semantics achieved

10. Workflow completes normally from Step 12 onward
```

---

## HITL Escalation Scenario

What if the customer has risk_score=0.85 (above 0.8 threshold)?

```
Step 8 (Validation): LLM checks get_customer_analytics() → risk_score=0.85
  → LLM applies business rule: risk_score 0.85 > 0.8 threshold
  → validation_passed = False
  → validation_reason = "High risk score (0.85) exceeds 0.8 safety threshold"

Step 9 (Supervisor Cycle 2):
  → Sees validation_passed = False, decision = None, request_saved = None
  → Tool match: finish_workflow() — "Use when validation_passed is False OR request_saved is True"
  → next_agent = "end"
  → Graph terminates (no policy/communication runs)

Back in _process_refund():
  → result["decision"] = None (policy never ran)
  → No output guardrails validation (decision is falsy)
  → Output: {"status": "completed", "validation_passed": false, "decision": null, ...}
```

### When HITL IS triggered (max iterations / retries):

```
Scenario A: Validation agent ReAct loop exceeds 8 iterations
  Step 8: react_loop raises ReactMaxIterationsError
  → validation_agent_node catches it
  → Returns: {"hitl_required": True, "hitl_reason": "react_max_iter:validation"}

  Step 9 (Supervisor): sees state.get("hitl_required") == True at the TOP
  → SKIPS LLM call entirely (pass-through)
  → Returns: {"next_agent": "hitl"}
  → Routes to hitl node

Scenario B: Supervisor cycle limit exceeded (>15)
  Step N (Supervisor): cycle_count > MAX_CYCLES(15)
  → Returns: {"next_agent": "hitl", "hitl_required": True, 
              "hitl_reason": "Cycle limit (15) exceeded"}

Scenario C: Per-agent retry exhausted (>3 invocations)
  Step N (Supervisor): retry_counts["policy"] > MAX_AGENT_RETRIES(3)
  → Returns: {"next_agent": "hitl", "hitl_required": True, 
              "hitl_reason": "retry_exhausted:policy"}

Scenario D: Supervisor LLM call fails (API error)
  → try/except catches exception
  → Returns: {"next_agent": "hitl", "hitl_required": True, 
              "hitl_reason": "supervisor_llm_error:<error msg>"}
```

### HITL Node Execution:
```python
# workflow/graph.py → hitl_node()
# Persists full state to DB for human review:

INSERT INTO hitl_tasks (task_id, reason, state_json, status)
VALUES ('a1b2c3d4-...', 'react_max_iter:validation', '{"full serialized state..."}', 'pending')
ON CONFLICT (task_id) DO UPDATE SET reason=..., state_json=..., status='pending'

# Returns:
→ {"hitl_required": True, "request_saved": True, "next_agent": "end"}
# (request_saved=True prevents supervisor from looping after hitl)

# hitl → END edge in graph (terminates)
```

### Back in _process_refund():
```python
if result.get("hitl_required"):  # → True
    # Store HITL status in Redis (24h TTL for polling)
    SET task:a1b2c3d4-... '{"status":"hitl_pending","hitl_reason":"...","message":"Use POST /hitl/tasks/{id}/resolve"}' EX 86400
    
    # Mark in event bus
    event_bus.mark_done(task_id, status="hitl_pending")
    
    # Update idempotency
    UPDATE idempotency_records SET status='hitl_pending' WHERE task_id=...
    
    # Persist task result
    INSERT INTO task_results (...) VALUES (..., status='hitl_pending', hitl_reason='...')
    
    return  # STOP — await human action

# Human resolves via API:
# POST /hitl/tasks/a1b2c3d4-.../resolve
# body: {"action": "approve"} | {"action": "deny"} | {"action": "compensate"}
#
# "approve": loads state_json from hitl_tasks → re-runs workflow with recovered_state
#            (deletes LangGraph checkpoint first, then fresh invoke)
# "deny": writes denied TaskResult, marks done
# "compensate": writes compensated TaskResult with compensated=True
```

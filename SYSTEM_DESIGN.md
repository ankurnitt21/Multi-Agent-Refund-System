# Multi-Agent AI Refund System — Complete Production Design

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack & Libraries](#2-tech-stack--libraries)
3. [Agent Design (Supervisor + ReAct)](#3-agent-design-supervisor--react)
4. [Crash Recovery & Transaction Handling](#4-crash-recovery--transaction-handling)
5. [Development Guide](#5-development-guide)
6. [Testing Strategy](#6-testing-strategy)
7. [Decision Framework](#7-decision-framework)
8. [Deployment Pipeline](#8-deployment-pipeline)
9. [Production Operations](#9-production-operations)
10. [Step-by-Step Implementation Roadmap](#10-step-by-step-implementation-roadmap)

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│  Streamlit UI │ REST API (FastAPI) │ Webhook Consumers                  │
└────────────────────────────┬───────────────────────────────────────────┘
                             │ HTTP/WebSocket
┌────────────────────────────▼───────────────────────────────────────────┐
│                        API GATEWAY                                      │
│  Rate Limiting │ Auth (JWT/API Key) │ Request Validation                │
│  Idempotency Guard (Redis → PostgreSQL fallback)                       │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────────┐
│                    GUARDRAILS LAYER                                     │
│  Input Sanitizer │ PII Masker │ Prompt Injection Detector               │
│  Output Validators │ Policy Compliance Check                           │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────────┐
│                 LANGGRAPH WORKFLOW ENGINE                               │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              SUPERVISOR (Router LLM)                         │       │
│  │  • Decides next agent via tool-calling                      │       │
│  │  • Enforces MAX_CYCLES=15, MAX_RETRIES=3                   │       │
│  │  • Escalates to HITL on limits/uncertainty                  │       │
│  └────┬──────────────┬──────────────┬──────────────┬───────────┘       │
│       │              │              │              │                    │
│  ┌────▼────┐   ┌─────▼─────┐  ┌────▼─────┐  ┌────▼────┐              │
│  │VALIDATION│   │  POLICY   │  │  COMMS   │  │  HITL   │              │
│  │  AGENT  │   │  AGENT    │  │  AGENT   │  │  NODE   │              │
│  │(ReAct)  │   │ (ReAct)   │  │ (ReAct)  │  │         │              │
│  └────┬────┘   └─────┬─────┘  └────┬─────┘  └─────────┘              │
│       │              │              │                                   │
│  ┌────▼──────────────▼──────────────▼─────────────────────┐           │
│  │            TOOL EXECUTION LAYER                         │           │
│  │  Wave-based parallel execution │ Circuit Breaker        │           │
│  │  Retry with exponential backoff │ Model Fallback        │           │
│  └────────────────────────────────────────────────────────┘           │
│                                                                        │
│  CHECKPOINTING: After every agent → Redis (hot) + PostgreSQL (durable)│
└────────────────────────────┬───────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────────┐
│                     DATA & SERVICES LAYER                              │
│                                                                        │
│  PostgreSQL        │ Redis              │ Pinecone (Vector DB)         │
│  • 12 tables       │ • Semantic cache   │ • Policy embeddings          │
│  • Audit logs      │ • Kafka events     │ • Similar decisions          │
│  • Checkpoints     │ • Checkpoints(hot) │                              │
│  • Idempotency     │ • Idempotency      │                              │
│                    │ • Circuit state    │                              │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────────┐
│                    OBSERVABILITY LAYER                                  │
│  OpenTelemetry → OTLP Exporter │ LangSmith Tracing                    │
│  Structlog (JSON) │ RAGAS Evaluation │ A/B Testing                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack & Libraries

### Core Framework

| Component | Library | Version | Why |
|-----------|---------|---------|-----|
| Agent orchestration | `langgraph` | >=0.2 | StateGraph with conditional routing, native checkpointing, HITL support |
| LLM abstraction | `langchain` | >=0.3 | Unified tool-calling, prompt templates, model wrappers |
| LLM provider | `langchain-openai` | latest | gpt-4o-mini (main) and gpt-4.1-nano (fast agents) |
| Embeddings | `langchain-openai` | latest | `text-embedding-3-small` for Pinecone and semantic cache |
| Checkpointing | `langgraph-checkpoint-postgres` | >=2.0 | Durable state persistence for crash recovery |

### Infrastructure

| Component | Library/Service | Why |
|-----------|----------------|-----|
| API server | `fastapi` + `uvicorn` | Async, fast, OpenAPI docs auto-generated |
| Database | PostgreSQL 15+ | ACID transactions, JSON columns, reliable |
| ORM | `sqlalchemy` >=2.0 (async) | Type-safe queries, async session support |
| Migrations | `alembic` | Versioned schema changes, rollback support |
| Cache/Queue | Redis 7+ | Sub-ms latency, pub/sub, sorted sets for queue |
| Vector DB | Pinecone | Managed, scalable policy retrieval |
| Task queue | Kafka (`refund.requests`) + aiokafka | Event-driven, consumer groups, offset-based crash recovery |

### Resilience & Safety

| Component | Library | Why |
|-----------|---------|-----|
| Retries | `tenacity` >=9.0 | Exponential backoff, retry budgets, custom predicates |
| Circuit breaker | Custom (Redis-backed) | Shared state across workers, configurable thresholds |
| Input guard | `guardrails-ai` >=0.5 | Declarative output validation, prompt injection detection |
| PII handling | Custom regex + masking | GDPR compliance, reversible for internal use |

### Observability

| Component | Library | Why |
|-----------|---------|-----|
| Tracing | `opentelemetry-sdk` + OTLP exporter | Distributed tracing, vendor-agnostic |
| LLM tracing | `langsmith` | LangChain-native, replay debugging |
| Logging | `structlog` | Structured JSON logs, correlation IDs |
| Evaluation | `ragas` >=0.2 | Faithfulness, answer relevancy, context precision |
| A/B testing | Custom (Welch's t-test) | Statistical significance for prompt/model variants |

### Testing

| Component | Library | Why |
|-----------|---------|-----|
| Unit/Integration | `pytest` + `pytest-asyncio` | Async test support, fixtures |
| Coverage | `pytest-cov` | Branch coverage reporting |
| Load testing | `locust` | Python-native, distributed load gen |
| Contract testing | `schemathesis` | Auto-generated API tests from OpenAPI spec |
| Mocking | `pytest-mock` + `respx` | HTTP mock for LLM calls, Redis mock |

### Deployment

| Component | Tool | Why |
|-----------|------|-----|
| Containerization | Docker + docker-compose | Reproducible environments |
| Orchestration | Kubernetes (EKS/GKE) | Auto-scaling, rolling deployments |
| CI/CD | GitHub Actions | Integrated with repo, matrix testing |
| Secrets | AWS Secrets Manager / Vault | Rotation, audit trail |
| Monitoring | Grafana + Prometheus | Dashboards, alerting |

---

## 3. Agent Design (Supervisor + ReAct)

### 3.1 Supervisor Pattern

```python
# The supervisor is a tool-calling LLM that routes to agents
# It does NOT process refund logic — only orchestration

ROUTING TOOLS:
├── call_validation_agent()   → Routes to validation
├── call_policy_agent()       → Routes to policy
├── call_communication_agent()→ Routes to communication
└── finish_workflow()         → Terminates graph
```

**Why Supervisor over sequential?**
- Dynamic routing based on state (skip validation if pre-verified)
- Retry individual agents without restarting entire flow
- Natural escalation point for HITL
- LLM-powered decision allows handling edge cases

### 3.2 ReAct Loop Per Agent

Each agent runs an independent ReAct (Reason + Act) loop:

```
┌──────────────────────────────────────────────┐
│ AGENT ReAct LOOP (max 8 iterations)          │
│                                              │
│  1. THINK: Analyze current state + history   │
│  2. ACT: Select tool(s) to call             │
│  3. OBSERVE: Process tool results           │
│  4. DECIDE: Need more info? → Loop          │
│             Have answer? → Return            │
│                                              │
│  OPTIMIZATION: Dependency-aware wave exec    │
│  Wave 0: [independent tools in parallel]     │
│  Wave 1: [tools depending on wave 0]         │
└──────────────────────────────────────────────┘
```

### 3.3 State Schema (Shared Across Agents)

```python
class RefundState(TypedDict):
    # Input
    customer_id: str
    order_id: str
    reason: str
    
    # Validation results
    validation_passed: Optional[bool]
    validation_details: Optional[dict]
    
    # Policy decision
    decision: Optional[str]  # "approved" | "denied" | "partial" | "escalate"
    refund_amount: Optional[float]
    policy_explanation: Optional[str]
    
    # Communication
    request_saved: Optional[bool]
    
    # Control
    messages: list[BaseMessage]
    current_agent: Optional[str]
    cycle_count: int
    agent_retry_counts: dict
    errors: list[str]
```

### 3.4 Agent Responsibilities

| Agent | Input State | Output State | Tools |
|-------|-------------|--------------|-------|
| **Validation** | customer_id, order_id | validation_passed, validation_details | lookup_customer, get_order_details, verify_ownership, get_analytics |
| **Policy** | validation_details, reason | decision, refund_amount, policy_explanation | search_policies, check_eligibility, calculate_refund, check_history |
| **Communication** | decision, refund_amount | request_saved | save_request, update_analytics, update_status, log_audit |

---

## 4. Crash Recovery & Transaction Handling

### 4.1 Checkpoint Strategy (Dual-Layer)

```
Every state transition checkpointed:

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SUPERVISOR │────▶│  AGENT RUN  │────▶│  SUPERVISOR │
│  (decide)   │     │  (execute)  │     │  (decide)   │
└──────┬──────┘     └──────┬──────┘     └─────────────┘
       │                   │
       ▼                   ▼
  ┌─────────┐        ┌─────────┐
  │  Redis  │        │  Redis  │     ← Hot (fast writes, TTL 24h)
  │  (hot)  │        │  (hot)  │
  └────┬────┘        └────┬────┘
       │                   │
       ▼                   ▼
  ┌─────────┐        ┌─────────┐
  │PostgreSQL│       │PostgreSQL│     ← Cold (durable, no TTL)
  │  (cold) │        │  (cold) │
  └─────────┘        └─────────┘
```

**Recovery Flow:**
1. On startup → Kafka consumer group joins `refund-workers` and processes uncommitted `refund.requests` events
2. For each pending task → load checkpoint from Redis (fast) or PostgreSQL (fallback)
3. Resume LangGraph from last completed agent
4. If no checkpoint → restart from beginning (idempotent tools)

### 4.2 Transaction Safety

```python
# CRITICAL: All DB operations use explicit transactions

# Pattern 1: Tool-level atomicity
async def save_processed_request(data):
    async with async_session() as session:
        async with session.begin():  # Auto-rollback on exception
            # Insert processed_request
            # Update refund_request status
            # Both succeed or both fail

# Pattern 2: Idempotency key prevents double-processing
async def check_idempotency(key):
    # 1. Check Redis (fast path)
    # 2. If miss → Check PostgreSQL (durable path)
    # 3. If new → Set "processing" in both stores
    # 4. On completion → Set "completed" with result

# Pattern 3: Saga pattern for multi-step operations
async def process_refund():
    try:
        await step1_validate()          # Checkpoint ✓
        await step2_decide_policy()     # Checkpoint ✓
        await step3_save_decision()     # Checkpoint ✓
        await step4_update_analytics()  # Checkpoint ✓
    except Exception:
        # Each step is idempotent — safe to replay from checkpoint
        await escalate_to_hitl(reason="unrecoverable_error")
```

### 4.3 Failure Modes & Handling

| Failure | Detection | Recovery |
|---------|-----------|----------|
| LLM timeout | `tenacity` retry after 30s | Retry 3x → fallback model → HITL |
| LLM bad output | Output parser fails | Re-prompt with error context (3x) → HITL |
| DB connection lost | SQLAlchemy disconnect | Connection pool auto-reconnect, retry operation |
| Redis down | `redis.ConnectionError` | Degrade to PostgreSQL-only (slower but works) |
| Worker crash (OOM/kill) | Task stays "processing" | Startup recovery scans + replays from checkpoint |
| Pinecone unavailable | Circuit breaker trips | Use cached policies, degrade to rule-based |
| Infinite loop | cycle_count > MAX_CYCLES | Force terminate → HITL escalation |
| Partial completion | Agent fails mid-execution | Resume from last checkpoint (agent-level) |

### 4.4 Circuit Breaker (Redis-Backed)

```
States: CLOSED → OPEN → HALF_OPEN → CLOSED

CLOSED:    Normal operation, count failures
           failure_count >= 5 in 60s → OPEN

OPEN:      All calls rejected immediately (fast-fail)
           After 30s cooldown → HALF_OPEN

HALF_OPEN: Allow 1 probe request
           Success → CLOSED (reset counters)
           Failure → OPEN (restart cooldown)
```

### 4.5 Model Fallback Chain

```
Primary:    openai/gpt-4o-mini           (supervisor, policy)
Fallback 1: openai/gpt-4.1-nano         (fast agents)
Fallback 2: openai/gpt-4o               (better reasoning)
```

---

## 5. Development Guide

### 5.1 Local Setup

```bash
# 1. Clone and setup
git clone <repo>
cd refund_system
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start infrastructure
docker-compose up -d  # PostgreSQL (5433) + Redis (6379)

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=...
#   PINECONE_API_KEY=...
#   LANGSMITH_API_KEY=... (optional)

# 5. Initialize database
python -c "from database.db_setup import init_db; import asyncio; asyncio.run(init_db())"

# 6. Seed test data
python scripts/seed_data.py

# 7. Run migrations
alembic upgrade head

# 8. Start API server
uvicorn main:app --reload --port 8000

# 9. Start UI (separate terminal)
streamlit run ui/app.py --server.port 8501
```

### 5.2 Development Workflow

```
Feature Branch → Write Code → Unit Tests → Integration Tests
    → Pre-commit hooks (ruff lint + format) → PR → CI passes → Merge
```

### 5.3 Project Structure (Target)

```
refund_system/
├── agents/                    # Agent definitions + ReAct loop
├── alembic/                   # DB migrations
│   ├── versions/
│   └── env.py
├── cache/                     # Redis cache layer
├── checkpoint/                # Dual-layer checkpointing
├── config.py                  # Centralized config (env vars)
├── database/                  # Models + DB setup
├── docker/                    # Dockerfiles per service
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   └── Dockerfile.ui
├── docker-compose.yml         # Local dev infrastructure
├── docker-compose.prod.yml    # Production overrides
├── evaluation/                # RAGAS + A/B testing
├── executor/                  # Parallel execution + resilience
├── guardrails/                # Input/output validation
├── k8s/                       # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── configmap.yaml
├── logging_config.py          # Structured logging
├── main.py                    # FastAPI application
├── prompts/                   # Agent prompts (markdown + registry)
├── scripts/                   # Operational scripts
│   ├── seed_data.py
│   ├── run_evaluation.py
│   └── migrate.py
├── task_queue_store/          # Kafka event bus (refund.requests)
├── telemetry/                 # OpenTelemetry + LangSmith
├── tests/                     # All tests
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── load/
├── tools/                     # Agent tools (DB, policy, analytics)
├── ui/                        # Streamlit dashboard
├── vectordb/                  # Pinecone operations
└── workflow/                   # LangGraph definition
```

### 5.4 Key Environment Variables

```env
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MAIN_MODEL=gpt-4o-mini
OPENAI_FAST_MODEL=gpt-4.1-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5433/refund_db
DATABASE_URL_SYNC=postgresql://user:pass@localhost:5433/refund_db

# Redis
REDIS_URL=redis://localhost:6379/0
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_REFUND_REQUESTS=refund.requests
KAFKA_CONSUMER_GROUP=refund-workers

# Vector DB
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=refund-policies

# Observability
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=refund-system
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Feature Flags
ENABLE_SEMANTIC_CACHE=true
ENABLE_RAGAS_EVALUATION=true
ENABLE_AB_TESTING=false

# Resilience
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=30
MAX_RETRIES=3
RETRY_BUDGET_WINDOW=60
```

---

## 6. Testing Strategy

### 6.1 Testing Pyramid

```
           ┌───────────┐
           │   E2E     │  5%  — Full workflow through API
           │  Tests    │       (real LLM, real DB)
          ┌┴───────────┴┐
          │ Integration  │ 25% — Agent + tools + DB
          │   Tests      │       (mocked LLM, real DB)
         ┌┴─────────────┴─┐
         │   Unit Tests    │ 70% — Individual functions
         │                 │       (everything mocked)
         └─────────────────┘
```

### 6.2 Unit Tests

```python
# tests/unit/test_policy_tools.py
@pytest.fixture
def mock_db_session():
    """In-memory SQLite for fast unit tests."""
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    # ... setup tables, seed data
    yield session

async def test_check_eligibility_within_window(mock_db_session):
    """Order within return window → eligible."""
    result = await check_eligibility(
        order_id="ORD-001",
        reason="defective",
        _session=mock_db_session
    )
    assert result["eligible"] == True
    assert "within return window" in result["explanation"]

async def test_check_eligibility_expired(mock_db_session):
    """Order past return window → not eligible."""
    result = await check_eligibility(
        order_id="ORD-EXPIRED",
        reason="changed_mind",
        _session=mock_db_session
    )
    assert result["eligible"] == False
```

### 6.3 Integration Tests

```python
# tests/integration/test_workflow.py
@pytest.fixture
def mock_llm():
    """Deterministic LLM responses for integration tests."""
    return FakeListChatModel(responses=[
        # Supervisor → validation
        AIMessage(content="", tool_calls=[{"name": "call_validation_agent", ...}]),
        # Validation agent reasoning
        AIMessage(content="Customer verified, order valid"),
        # Supervisor → policy
        AIMessage(content="", tool_calls=[{"name": "call_policy_agent", ...}]),
        # ... etc
    ])

async def test_full_approved_refund_flow(mock_llm, test_db):
    """Happy path: valid customer + eligible order → approved refund."""
    state = RefundState(customer_id="C001", order_id="O001", reason="defective")
    result = await run_workflow(state, llm=mock_llm)
    
    assert result["decision"] == "approved"
    assert result["refund_amount"] > 0
    assert result["request_saved"] == True

async def test_crash_recovery_resumes_from_checkpoint(mock_llm, test_db, redis):
    """Simulate crash after validation → resumes at policy agent."""
    # Pre-set checkpoint at "validation completed"
    checkpoint = {"validation_passed": True, "validation_details": {...}}
    await save_checkpoint("thread-123", "validation", checkpoint)
    
    result = await recover_task("thread-123")
    assert result["decision"] is not None  # Continued past validation
```

### 6.4 E2E Tests

```python
# tests/e2e/test_api_flow.py
@pytest.mark.e2e
async def test_refund_api_full_flow(api_client):
    """Full API test with real LLM (use sparingly, costs money)."""
    response = await api_client.post("/refund", json={
        "customer_id": "CUST-001",
        "order_id": "ORD-001",
        "reason": "Product arrived damaged",
        "idempotency_key": str(uuid4())
    })
    assert response.status_code == 202
    
    task_id = response.json()["task_id"]
    # Poll for completion
    result = await poll_until_done(api_client, task_id, timeout=60)
    assert result["status"] in ["completed", "escalated"]
```

### 6.5 Evaluation Tests (LLM Quality)

```python
# tests/evaluation/test_ragas.py
async def test_policy_agent_faithfulness():
    """Policy decisions must be grounded in retrieved policies."""
    test_cases = load_evaluation_dataset("policy_decisions.json")
    
    results = await ragas_evaluate(
        test_cases,
        metrics=["faithfulness", "answer_relevancy", "context_precision"]
    )
    
    assert results["faithfulness"] >= 0.85
    assert results["answer_relevancy"] >= 0.80

# tests/evaluation/test_decision_accuracy.py
async def test_refund_decision_accuracy():
    """Decisions match expected outcomes for known scenarios."""
    scenarios = [
        {"input": {...}, "expected_decision": "approved", "expected_amount": 99.99},
        {"input": {...}, "expected_decision": "denied", "expected_amount": 0},
        {"input": {...}, "expected_decision": "partial", "expected_amount": 50.00},
    ]
    
    accuracy = await evaluate_decisions(scenarios)
    assert accuracy >= 0.90  # 90%+ correct decisions
```

### 6.6 Load Tests

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class RefundUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(10)
    def submit_refund(self):
        self.client.post("/refund", json={
            "customer_id": f"CUST-{random.randint(1,1000)}",
            "order_id": f"ORD-{random.randint(1,5000)}",
            "reason": random.choice(REASONS),
            "idempotency_key": str(uuid4())
        })
    
    @task(3)
    def check_status(self):
        self.client.get(f"/refund/{random.choice(KNOWN_TASK_IDS)}/status")
    
    @task(1)
    def resolve_hitl(self):
        self.client.post(f"/hitl/{random.choice(HITL_IDS)}/resolve", json={
            "action": "approve", "resolver": "admin@company.com"
        })

# Target: 100 concurrent users, <2s p95 latency, 0% error rate
```

### 6.7 Test Commands

```bash
# Unit tests (fast, no infra needed)
pytest tests/unit/ -v --cov=. --cov-report=html

# Integration tests (needs Docker services)
docker-compose up -d
pytest tests/integration/ -v --timeout=30

# E2E tests (needs API keys, costs money)
pytest tests/e2e/ -v -m e2e --timeout=120

# Evaluation suite
python scripts/run_evaluation.py --dataset=golden_set.json

# Load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

---

## 7. Decision Framework

### 7.1 Refund Decision Tree

```
START
│
├─ Is customer account active?
│  └─ NO → DENY ("Account suspended")
│
├─ Does customer own this order?
│  └─ NO → DENY ("Order ownership mismatch")
│
├─ Is order status "delivered"?
│  └─ NO → DENY ("Order not yet delivered")
│
├─ Is customer risk score > 0.8?
│  └─ YES → ESCALATE to human review
│
├─ Is order within return window?
│  │  (base: product.return_window_days)
│  │  (+ tier bonus: gold=+15, silver=+7)
│  ├─ NO → Check reason
│  │  ├─ "defective" → APPROVE (warranty override)
│  │  └─ other → DENY ("Return window expired")
│  └─ YES → Continue
│
├─ Check refund history (fraud signals)
│  ├─ refund_rate > 40% → ESCALATE
│  ├─ 3+ refunds in 30 days → ESCALATE
│  └─ OK → Continue
│
├─ Calculate refund amount
│  ├─ Full refund: defective, wrong_item, not_received
│  ├─ Partial refund: changed_mind (minus restocking fee)
│  │   └─ Restocking fee waived for gold/silver tier
│  └─ Custom: damage_in_transit (carrier claim amount)
│
└─ APPROVE with calculated amount
```

### 7.2 When to Escalate to Human (HITL)

| Trigger | Condition |
|---------|-----------|
| High-value refund | amount > $500 |
| Suspicious pattern | 3+ refunds in 30 days |
| High risk score | risk_score > 0.8 |
| Agent uncertainty | LLM confidence < 0.6 |
| System limits | cycle_count > 15 or retries > 3 |
| Policy ambiguity | Multiple conflicting policies match |
| VIP customer | Customer tier = "enterprise" |

### 7.3 A/B Testing Decisions

```python
# What to A/B test:
EXPERIMENTS = {
    "prompt_version": {
        "control": "policy_agent_v1.md",
        "variant": "policy_agent_v2.md",
        "metric": "decision_accuracy",
        "min_samples": 100,
        "significance": 0.05  # p-value threshold
    },
    "model_choice": {
        "control": "gpt-4o-mini",
        "variant": "gpt-4o",
        "metric": "latency_vs_accuracy_tradeoff"
    },
    "max_iterations": {
        "control": 8,
        "variant": 5,
        "metric": "completion_rate"
    }
}
```

---

## 8. Deployment Pipeline

### 8.1 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ --cov --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: refund_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/integration/ --timeout=60
        env:
          DATABASE_URL: postgresql+asyncpg://test:test@localhost:5432/refund_test
          REDIS_URL: redis://localhost:6379/0

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install bandit safety
      - run: bandit -r . -x ./tests
      - run: safety check -r requirements.txt

  build-and-push:
    needs: [lint, unit-tests, integration-tests, security-scan]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}/api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - run: |
          kubectl set image deployment/refund-api \
            api=ghcr.io/${{ github.repository }}/api:${{ github.sha }} \
            --namespace=staging

  e2e-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/e2e/ --timeout=120
        env:
          API_BASE_URL: https://staging.refund-api.internal

  deploy-production:
    needs: e2e-tests
    runs-on: ubuntu-latest
    environment: production  # Requires manual approval
    steps:
      - run: |
          kubectl set image deployment/refund-api \
            api=ghcr.io/${{ github.repository }}/api:${{ github.sha }} \
            --namespace=production
```

### 8.2 Docker Setup

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 8.3 Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: refund-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
        - name: api
          image: ghcr.io/org/refund-system/api:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: refund-secrets
                  key: database-url
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: refund-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: refund-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

### 8.4 Environment Promotion

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐
│   Dev    │───▶│ Staging  │───▶│  Canary (5%) │───▶│ Production │
│ (local)  │    │ (auto)   │    │  (auto)      │    │ (manual)   │
└──────────┘    └──────────┘    └──────────────┘    └────────────┘
     │               │                │                    │
  Feature         Full CI          10 min soak        Approval gate
  branches        + E2E            + error rate        + rollback plan
                  tests            monitoring
```

---

## 9. Production Operations

### 9.1 Health Checks

```python
# Already in main.py — enhance with detailed checks
@app.get("/health")
async def health():
    checks = {
        "api": "ok",
        "database": await check_db_connection(),
        "redis": await check_redis_connection(),
        "llm": await check_llm_availability(),
    }
    status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
```

### 9.2 Monitoring & Alerting

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| API p95 latency | > 5s | > 15s | Scale up, check LLM |
| Error rate | > 1% | > 5% | Page on-call |
| HITL queue depth | > 50 | > 200 | Alert ops team |
| Circuit breaker OPEN | Any | 2+ providers | Failover alert |
| Checkpoint age | > 1h | > 4h | Check stuck tasks |
| Decision accuracy (eval) | < 90% | < 80% | Rollback prompt/model |
| Redis memory | > 70% | > 90% | Eviction policy review |
| DB connections | > 80% pool | > 95% pool | Scale DB / tune pool |

### 9.3 Grafana Dashboard Panels

```
Row 1: Traffic & Latency
├── Requests/sec (by endpoint)
├── p50/p95/p99 latency
└── Error rate %

Row 2: Agent Performance  
├── Decisions/min (approved/denied/escalated)
├── Avg cycles per refund
└── Agent retry rate

Row 3: LLM Metrics
├── Token usage (in/out per agent)
├── LLM latency per model
└── Circuit breaker state

Row 4: Infrastructure
├── DB connection pool utilization
├── Redis memory / hit rate
└── Pod CPU/Memory
```

### 9.4 Runbooks

**Runbook: High Error Rate**
```
1. Check /health endpoint → identify failing component
2. Check Grafana LLM panel → is it LLM timeout?
   YES → Check OpenAI status page, circuit breaker should auto-failover
   NO  → Continue
3. Check recent deployments → rollback if < 30 min old
4. Check DB connections → restart pods if pool exhausted
5. Check Redis → if down, system degrades but continues
6. Escalate if unresolved in 15 min
```

**Runbook: Stuck Tasks**
```
1. Query: SELECT * FROM persistent_task_queue WHERE status='processing' 
   AND updated_at < NOW() - INTERVAL '1 hour'
2. For each stuck task:
   a. Check checkpoint table for last_completed_agent
   b. If checkpoint exists → trigger recovery endpoint
   c. If no checkpoint → mark as failed, notify customer
3. Investigate root cause (OOM? Deadlock? LLM stuck?)
```

### 9.5 Backup & Disaster Recovery

| Component | RPO | RTO | Strategy |
|-----------|-----|-----|----------|
| PostgreSQL | 5 min | 30 min | Streaming replication + WAL archival |
| Redis | 1 hour | 5 min | AOF persistence + Redis Sentinel |
| Pinecone | N/A | N/A | Managed, multi-AZ by default |
| Prompts | 0 | 5 min | Git-versioned, DB is cache |
| Secrets | 0 | 10 min | AWS Secrets Manager (versioned) |

### 9.6 Security Checklist (Production)

- [ ] JWT/API Key authentication on all endpoints
- [ ] Rate limiting (100 req/min per client)
- [ ] PII masking in all logs (already implemented)
- [ ] Prompt injection detection (already implemented)
- [ ] SQL injection prevention (SQLAlchemy parameterized queries)
- [ ] HTTPS everywhere (TLS 1.3)
- [ ] Network policies (pod-to-pod isolation in K8s)
- [ ] Secrets rotation every 90 days
- [ ] Audit log retention (7 years for financial)
- [ ] RBAC for HITL resolution endpoint
- [ ] Input size limits (prevent DoS)
- [ ] Container image scanning (Trivy)
- [ ] Dependency vulnerability scanning (safety/snyk)

---

## 10. Step-by-Step Implementation Roadmap

### Phase 1: Foundation (Week 1-2) ✅ DONE
- [x] Project structure
- [x] LangGraph workflow with supervisor routing
- [x] 3 ReAct agents with tools
- [x] PostgreSQL schema (12 tables)
- [x] Redis caching layer
- [x] Docker Compose for local dev
- [x] Basic FastAPI endpoints
- [x] Dual-layer checkpointing
- [x] Idempotency handling
- [x] Crash recovery on startup

### Phase 2: Safety & Quality (Week 3-4) ✅ DONE
- [x] Input sanitization (prompt injection)
- [x] PII detection and masking
- [x] Output parser (3-strategy)
- [x] Circuit breaker + retry + fallback
- [x] Semantic cache
- [x] HITL escalation flow
- [x] Kafka event bus (`refund.requests`)
- [x] Structured logging
- [x] OpenTelemetry + LangSmith tracing

### Phase 3: Testing & Evaluation (Week 5-6) — IN PROGRESS
- [x] Unit tests (guardrails, parser, resilience, A/B)
- [ ] **Add integration tests with mocked LLM**
- [ ] **Add E2E tests with real API**
- [ ] **Set up RAGAS evaluation golden dataset**
- [ ] **Add conftest.py fixtures (DB, Redis, LLM mocks)**
- [ ] **Load testing with Locust**
- [ ] **Contract tests with Schemathesis**

### Phase 4: Production Hardening (Week 7-8)
- [ ] **Add authentication (JWT + API keys)**
- [ ] **Add rate limiting (slowapi or custom Redis)**
- [ ] **Alembic migrations setup**
- [ ] **Seed data scripts**
- [ ] **.env.example file**
- [ ] **Health check enhancements**
- [ ] **Graceful shutdown handling**
- [ ] **Request timeout middleware**

### Phase 5: Deployment (Week 9-10)
- [ ] **Multi-stage Dockerfile**
- [ ] **GitHub Actions CI/CD pipeline**
- [ ] **Kubernetes manifests (deployment, service, HPA, configmap)**
- [ ] **Staging environment**
- [ ] **Canary deployment strategy**
- [ ] **Grafana + Prometheus monitoring**
- [ ] **PagerDuty/Opsgenie alerting**

### Phase 6: Optimization (Week 11-12)
- [ ] **A/B testing framework activation**
- [ ] **Prompt optimization based on RAGAS scores**
- [ ] **Model upgrade evaluation (larger models)**
- [ ] **Cost optimization (batch processing, caching tuning)**
- [ ] **Customer notification integration (email/webhook)**
- [ ] **Analytics dashboard for business metrics**

---

## Appendix A: Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Supervisor vs. sequential | Supervisor (LLM router) | Handles edge cases, dynamic retries, natural escalation |
| ReAct vs. plan-and-execute | ReAct per agent | Better for tool-heavy tasks, self-correcting |
| LLM provider | OpenAI | Unified API for main, fast, and fallback models; strong tool-calling |
| Redis + PostgreSQL | Dual-layer everything | Redis = speed, PostgreSQL = durability |
| Background tasks vs. sync | Background (202 + polling) | Refund processing takes 5-30s, don't block HTTP |
| Kafka vs. Redis queue | Kafka (`refund.requests`) | Durable log, consumer groups, offset-based crash recovery, event-driven scale-out |
| Pinecone vs. pgvector | Pinecone | Managed, no ops overhead, ANN at scale |
| Single repo vs. microservices | Monolith (single FastAPI) | Simpler at current scale, split later if needed |

## Appendix B: Cost Estimation (per 10K refunds/month)

| Component | Cost | Notes |
|-----------|------|-------|
| OpenAI API (gpt-4o-mini + gpt-4.1-nano) | ~$8 | ~3 calls/refund × 1K tokens avg |
| Pinecone (Starter) | $0 | Free tier covers policy retrieval |
| PostgreSQL (RDS db.t3.small) | ~$30 | 2 vCPU, 2GB RAM |
| Redis (ElastiCache t3.micro) | ~$15 | Single node sufficient |
| Kubernetes (3 nodes t3.small) | ~$50 | Min viable cluster |
| LangSmith (Developer) | $0 | Free tier: 5K traces/month |
| **Total** | **~$100/month** | Scales linearly with volume |

## Appendix C: SLA Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Availability | 99.9% | Uptime over 30 days |
| Refund processing time | < 30s (p95) | Request to decision |
| Decision accuracy | > 92% | Against golden dataset |
| HITL response time | < 4 hours | Escalation to resolution |
| Data retention | 7 years | Financial compliance |
| Recovery time (crash) | < 60s | Task pickup after restart |

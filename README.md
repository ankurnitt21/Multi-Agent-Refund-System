# Warehouse Refund Processing System

Multi-agent refund orchestration system built with FastAPI, LangGraph, ReAct agents, Redis, PostgreSQL, Pinecone, Guardrails-AI, and LangSmith telemetry.

This repository implements a durable, stateful workflow for warehouse refund decisions with:

- Supervisor-driven multi-agent routing
- Dual-layer idempotency (Redis + PostgreSQL)
- Dual-layer checkpointing and crash recovery
- HITL (Human-In-The-Loop) escalation and resolution
- Vector policy retrieval with semantic cache
- Guardrails for prompt-injection and PII protection
- RAGAS evaluation and A/B testing endpoints

## System Overview

The workflow is orchestrated by a LangGraph supervisor pattern:

1. Validation agent verifies customer/order eligibility and risk.
2. Policy agent applies refund rules and computes decision/amount.
3. Communication agent persists output and audit records.
4. Supervisor decides next step based on shared state and retry/cycle limits.

The API is asynchronous: `POST /refund` immediately returns a `task_id`, and processing continues in background tasks.

## Core Architecture

- API layer: FastAPI app in `main.py`
- Workflow engine: LangGraph in `workflow/graph.py`
- Agent nodes: `agents/` (supervisor, validation, policy, communication)
- Tools: `tools/` for DB, policy, analytics operations
- Persistence:
  - Redis for queueing, fast idempotency, short-term task results
  - PostgreSQL for durable idempotency, checkpoints, HITL tasks, and permanent task results
- Vector retrieval: Pinecone store in `vectordb/pinecone_store.py`
- Guardrails: input sanitizer, PII masking, output validation/parser in `guardrails/`
- Resilience: circuit breaker + retry budget + model fallback in `executor/resilience.py`
- Evaluation: RAGAS + prompt A/B testing in `evaluation/`

## End-to-End Flow

1. Client sends `POST /refund` with email, order ID, reason (and optional idempotency key).
2. API checks idempotency in Redis (fast path), then PostgreSQL (durable fallback).
3. New requests are persisted in queue and processed in background.
4. Guardrails sanitize and validate the request.
5. Supervisor routes across validation -> policy -> communication agents.
6. State is checkpointed for resumability and crash recovery.
7. Result is written to Redis + PostgreSQL and exposed by `GET /refund/{task_id}`.
8. If human review is required, request is parked in HITL queue and resolved via HITL endpoints.

## Key Features

- ReAct parallel wave execution for tool-calling agents
- Recovery-safe processing with pending-task replay on startup
- Durable `TaskResult` fallback when Redis TTL expires
- Prompt version management and activation endpoints
- Built-in telemetry with OpenTelemetry + LangSmith

## Project Structure

```
agents/              Multi-agent logic, supervisor and ReAct loop
cache/               Redis semantic cache
checkpoint/          Explicit checkpoint persistence
database/            SQLAlchemy models and DB setup
evaluation/          RAGAS evaluator and A/B testing
executor/            Parallel execution and resilience controls
guardrails/          Input/output safety and PII handling
prompts/             Prompt files, registry, versioning support
task_queue_store/    Redis-backed durable task queue
telemetry/           Tracing and telemetry setup
tools/               Domain tools used by agents
ui/                  Streamlit frontend
vectordb/            Pinecone vector index integration
workflow/            LangGraph definition and routing
main.py              FastAPI entrypoint
```

## Prerequisites

- Python 3.10+
- Redis
- PostgreSQL
- Pinecone account/API key
- OpenAI API key
- (Optional) LangSmith API key for tracing and RAGAS observability

## Configuration

Environment variables used by `config.py`:

- `OPENAI_API_KEY`
- `OPENAI_MAIN_MODEL` (default: `gpt-4o-mini`) — supervisor and policy agent
- `OPENAI_FAST_MODEL` (default: `gpt-4.1-nano`) — validation and communication agents
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `EMBEDDING_DIMENSION` (default: `1536`)
- `DATABASE_URL` (default: `postgresql+asyncpg://user:pass@localhost:5432/refund_db`)
- `REDIS_URL` (default: `redis://localhost:6379`)
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default: `warehouse-refund-policies`)
- `PINECONE_ENVIRONMENT` (default: `us-east-1`)
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT` / `LANGCHAIN_PROJECT`
- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_SERVICE_NAME`

## Local Development

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start infrastructure (Redis + PostgreSQL):

```bash
docker compose up -d
```

4. Run the API:

```bash
python main.py
```

5. (Optional) Run Streamlit UI:

```bash
streamlit run ui/app.py
```

## Docker Services

The included `docker-compose.yml` starts:

- `postgres` on host port `5433` (container `5432`)
- `redis` on host port `6379`

If you use compose defaults, set:

- `DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5433/refund_db`

## API Quick Reference

### Refund processing

- `POST /refund`
- `GET /refund/{task_id}`

### HITL operations

- `GET /hitl/tasks`
- `POST /hitl/tasks/{task_id}/resolve` with `approve | deny | compensate`

### Prompt registry

- `GET /prompts/{prompt_name}`
- `POST /prompts/{prompt_name}/activate`
- `POST /prompts/{prompt_name}/version`

### Analytics & observability

- `GET /analytics/customer/{customer_id}`
- `GET /audit-logs`
- `GET /resilience/health`

### Evaluation & experiments

- `POST /evaluate`
- `POST /experiments`
- `GET /experiments`
- `GET /experiments/{experiment_id}/report`
- `POST /experiments/{experiment_id}/stop`

## Testing

Run all tests:

```bash
pytest
```

## Documentation Sources

Detailed architecture and interview-oriented explanations were consolidated from:

- `interview_theory.html`
- `refund_flow_diagram.html`
- `interview_answers.md`

Open those files for expanded design rationale, visual flow diagram, and deeper Q&A.

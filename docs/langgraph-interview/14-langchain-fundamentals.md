# Section 14: LangChain Fundamentals

> **Project:** `langchain_core` + `langchain_openai` across `agents/`, `tools/`, `workflow/`  
> **Pair with:** [01-langgraph-core-concepts.md](./01-langgraph-core-concepts.md) — LangGraph orchestrates; LangChain composes LLM calls and tools.

---

## How LangChain and LangGraph divide labor

| Layer | Library | Our project |
|-------|---------|-------------|
| **Orchestration** | LangGraph | `StateGraph`, supervisor loop, checkpoints |
| **LLM + tools + messages** | LangChain | `ChatOpenAI`, `@tool`, `bind_tools`, ReAct loop |
| **Observability** | LangChain + OTel | `LANGCHAIN_TRACING_V2`, custom spans in `telemetry/setup.py` |

**Interview line:** *"LangChain is the runtime for model I/O and tools; LangGraph is the state machine on top."*

---

## Q1. What is LCEL (LangChain Expression Language)?

### Core idea

Every major building block implements the **`Runnable`** interface:

- `.invoke(input)` / `.ainvoke(input)` — single call
- `.batch(inputs)` — parallel batch
- `.stream(input)` / `.astream(input)` — token/chunk stream

Runnables compose with the **pipe operator** `|`:

```python
chain = prompt | llm | output_parser
result = await chain.ainvoke({"email": "user@example.com"})
```

Under the hood: each step's output becomes the next step's input (with automatic dict key mapping for prompt templates).

### Runnable types to know

| Runnable | Role |
|----------|------|
| `ChatPromptTemplate` | Build message list from variables |
| `ChatOpenAI` (model) | Call provider API |
| `RunnableLambda` | Wrap plain Python function |
| `RunnableParallel` | Run branches concurrently, merge dict |
| `StrOutputParser` | `AIMessage` → string content |

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
)
# {"summary": "...", "keywords": [...]}
```

### In our project

We **don't** wire the whole refund flow as one LCEL chain. We use **LangGraph nodes** instead. LCEL still appears **inside** nodes:

```python
# Conceptually what each agent does:
messages = [SystemMessage(...), HumanMessage(...)]
response = await llm_with_tools.ainvoke(messages)  # Runnable (bound model)
```

Supervisor is: `system_prompt | llm.bind_tools(ROUTING_TOOLS)` invoked manually in `supervisor_node`.

**Why graphs over one big chain:** Branching (validation vs policy), cycles (supervisor loop), checkpoints, and HITL need a graph — a linear LCEL pipe cannot express that cleanly.

---

## Q2. `langchain` vs `langchain_core` vs `langchain_openai`?

| Package | Contains |
|---------|----------|
| **`langchain_core`** | Messages, tools, runnables, base abstractions — **use this in production** |
| **`langchain_openai`** | `ChatOpenAI`, OpenAI embeddings |
| **`langchain`** (umbrella) | Legacy chains, community integrations — shrinking over time |
| **`langgraph`** | StateGraph, checkpointing — separate install |

Our imports:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
```

**Interview line:** *"We pin integrations to `langchain_core` + provider package; avoid deprecated `langchain.chains` in new code."*

---

## Q3. How do LangChain tools work with LLMs?

### Defining tools

```python
from langchain_core.tools import tool

@tool
async def lookup_customer(email: str) -> str:
    """Look up a customer by email address."""
    ...
```

- **Name** = function name (or override)
- **Description** = docstring → sent to model for tool selection
- **Schema** = inferred from type hints (Pydantic args)

### Binding tools to the model

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(VALIDATION_TOOLS, tool_choice="auto")
# Supervisor forces a call:
_supervisor_llm = llm.bind_tools(ROUTING_TOOLS, tool_choice="any")
```

Model returns `AIMessage` with optional `tool_calls: [{name, args, id}, ...]`.

### Executing and returning results

```python
for tc in response.tool_calls:
    result = await tool_map[tc["name"]].ainvoke(tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
```

**Critical:** `tool_call_id` on `ToolMessage` must match the `AIMessage` tool call id — same reason LangGraph provides `add_messages` reducer.

### Our tool layout

| Module | Tools |
|--------|-------|
| `tools/db_tools.py` | `lookup_customer`, `get_order_details`, … |
| `tools/policy_tools.py` | `search_policies`, `check_eligibility`, … |
| `tools/analytics_tools.py` | aggregates, refund history |

Routing tools in `supervisor.py` are **signals only** — no DB side effects.

---

## Q4. Message types — what goes in the conversation?

| Type | Role | Our usage |
|------|------|-----------|
| `SystemMessage` | Instructions | Agent prompts from `prompts/loader.py` |
| `HumanMessage` | User / state summary | Refund request; supervisor state summary |
| `AIMessage` | Model output | Supervisor routing narration; ReAct turns |
| `ToolMessage` | Tool result | ReAct loop after each tool batch |

Messages are **not** the only state — `RefundState` holds business fields. Messages are audit + in-node ReAct context.

---

## Q5. LangChain Memory vs LangGraph state — when to use which?

### Legacy LangChain memory (know for interviews)

| Class | Behavior |
|-------|----------|
| `ConversationBufferMemory` | Full history |
| `ConversationBufferWindowMemory` | Last K turns |
| `ConversationSummaryMemory` | LLM summarizes old turns |
| `VectorStoreRetrieverMemory` | Retrieve relevant past turns |

Typically attached to a **Chain** via `memory` key — loads/saves each `invoke`.

### Why we use graph state instead

| Concern | Memory class | Our approach |
|---------|--------------|--------------|
| Multi-agent routing | Awkward | `validation_passed`, `decision`, `next_agent` in `RefundState` |
| Durability | Separate store | Checkpointer + `checkpoint/store.py` |
| Partial updates | Opaque | TypedDict partial returns per node |
| Trimming | Built-in helpers | Exclude messages from JSON checkpoints; supervisor summary |

**Interview answer:**

> *"For multi-agent workflows I put durable facts in LangGraph state and checkpointers, not `ConversationBufferMemory`. Memory classes are fine for single-chain chatbots; graphs need typed, mergeable state."*

If asked **"how would you add long chat history?"**:

1. `trim_messages` before `ainvoke`
2. Summary node writing one `SystemMessage` with compressed context
3. External store (Postgres) keyed by `thread_id`; load last N turns into state on start

---

## Q6. `create_react_agent` vs custom ReAct — what do we use and why?

### LangGraph prebuilt

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)
# Single agent loop: LLM ↔ tools until no tool_calls
```

Good for **quick prototypes** and standard tool loops.

### Our custom `react_loop` (`agents/react_loop.py`)

| Feature | Prebuilt | Our loop |
|---------|----------|----------|
| Parallel tool execution | Limited / version-dependent | **Topological waves** via `VALIDATION_DEPS` |
| Max iterations → HITL | Manual | `ReactMaxIterationsError` |
| Circuit breaker / fallback | Manual | `executor/resilience.py` |
| Structured final output | Manual | `output_parser.py` + Pydantic models |

```python
await react_loop(
    _llm_with_validation_tools,
    VALIDATION_TOOL_MAP,
    messages,
    VALIDATION_DEPS,
    span=span,
)
```

**Interview line:** *"We use LangGraph for multi-agent orchestration and a custom ReAct engine inside nodes for parallel dependent tools and production error handling."*

---

## Q7. Structured output — how do you get reliable JSON?

### Options

| Approach | API |
|----------|-----|
| Tool calling | Force a "submit" tool with schema |
| Native structured output | `llm.with_structured_output(PydanticModel)` |
| Parser on text | `PydanticOutputParser` + prompt instructions |

### Our project

Agents produce JSON text after ReAct; we parse with **`parse_agent_output`** into `ValidationOutput`, policy models, etc. On `AgentOutputParseError` → **HITL**.

Supervisor uses **routing tools** instead of structured JSON — the tool call *is* the structured decision.

```python
# Alternative pattern (know for interviews):
router = llm.with_structured_output(RouterDecision)
decision = await router.ainvoke(messages)
```

We chose tool-calling for supervisor because descriptions encode routing rules and `tool_choice="any"` forces a decision.

---

## Q8. Callbacks, tracing, and LangSmith

### LangChain auto-tracing

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=warehouse-refund
```

Every `ChatOpenAI.ainvoke` inside LangChain runnables gets a trace (inputs, outputs, latency, tokens).

### Our dual tracing

1. **LangChain native** — LLM calls (via env in `telemetry/setup.py`)
2. **OpenTelemetry spans** — workflow, nodes, guardrails, RAGAS

```python
with tracer.start_as_current_span("validation_agent") as span:
    span.set_attribute("langsmith.span.kind", "chain")
    ...
```

Both can land in **LangSmith** when OTEL exporter points there.

### Callbacks (if asked)

```python
from langchain_core.callbacks import AsyncCallbackHandler

class TokenCounter(AsyncCallbackHandler):
    async def on_llm_end(self, response, **kwargs):
        ...
```

Use for custom metrics; LangSmith replaces much manual callback wiring.

---

## Q9. Streaming — `stream`, `astream`, `astream_events`

| API | Use |
|-----|-----|
| `chain.astream()` | Token chunks from LLM step |
| `graph.astream()` | **State updates per node** after each node completes |
| `graph.astream_events(version="v2")` | Fine-grained: `on_chat_model_stream`, `on_tool_start`, … |

### Our API today

`POST /refund` is **async job** (Kafka + task_id), not token streaming.

**Production upgrade path:**

```python
async for event in graph.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content  # SSE to UI
```

**Interview line:** *"LangGraph streaming is node- or event-level; LangChain streaming is token-level inside a node. For refund UX we'd SSE from `astream_events` inside the policy/validation node."*

---

## Q10. Prompts — templates vs file-based

### LangChain style

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a validation agent..."),
    ("human", "Validate order {order_id} for {email}"),
])
chain = prompt | llm
```

### Our style

- Markdown prompts in `prompts/*.md`
- `prompts/loader.py` loads content
- `PromptRegistry` for versioned A/B prompts (`evaluation/ab_testing.py`)

```python
VALIDATION_SYSTEM = load_prompt("validation_agent")
messages = [SystemMessage(content=VALIDATION_SYSTEM), HumanMessage(content=user_message)]
```

**Tradeoff:** File prompts are git-friendly and ops can activate versions without code deploy; templates are better for highly dynamic per-request slots.

---

## Q11. Embeddings and retrievers (RAG glue)

```python
from langchain_openai import OpenAIEmbeddings
# vectorstore.add_documents(docs)  # Pinecone via vectordb/pinecone_store.py
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

Our policy agent calls **`search_policies`** tool — wraps Pinecone + semantic Redis cache, not a LangChain `RetrievalQA` chain.

**Know the pattern:**

```python
rag_chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    | prompt
    | llm
)
```

We split retrieval (tool) and reasoning (ReAct LLM) for observability and HITL on tool failure.

---

## Q12. Common interview comparisons (quick answers)

| Question | Answer |
|----------|--------|
| Chain vs Agent vs Graph? | Chain = linear LCEL; Agent = LLM+tool loop; Graph = state machine with branches/cycles |
| Why LangGraph on top of LangChain? | Checkpointing, multi-agent, conditional edges, HITL |
| Sync vs async? | Production APIs use `ainvoke` / `async` tools throughout |
| How to test? | Mock tools; invoke nodes with fixture `RefundState`; eval with RAGAS |
| Biggest LangChain footgun? | Stuffing unbounded history into prompts — trim or summarize |

---

## Map: LangChain concept → project file

| Concept | File |
|---------|------|
| Tools | `tools/db_tools.py`, `tools/policy_tools.py` |
| ReAct loop | `agents/react_loop.py` |
| bind_tools / supervisor LLM | `agents/supervisor.py` |
| Messages in graph | `agents/state.py`, `workflow/graph.py` |
| Structured parse | `agents/output_parser.py`, `agents/output_models.py` |
| Resilience on LLM | `executor/resilience.py` |
| LangSmith env | `telemetry/setup.py`, `config.py` |
| RAG / embeddings | `vectordb/pinecone_store.py` |

---

## 5-minute drill before a LangChain-heavy round

1. Draw **Runnable pipe**: `prompt | llm | parser` and explain `.ainvoke`.
2. Explain **tool loop**: `AIMessage.tool_calls` → execute → `ToolMessage` → `ainvoke` again.
3. Contrast **Memory** vs **`RefundState` + checkpointer**.
4. Explain why **`create_react_agent`** wasn't enough (parallel waves, HITL, breaker).
5. Name two tracing paths: **LANGCHAIN_TRACING_V2** + **OTel spans**.

*Next: [05-complex-workflow-walkthrough.md](./05-complex-workflow-walkthrough.md) for the full system story.*

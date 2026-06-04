# Interview Q&A Deep-Dive — Multi-Agent Refund System

> Each answer follows the same structure:
> **🏗️ In This Project** → what this codebase actually does, with file references
> **💡 Why It Matters** → the reasoning behind the design choice
> **🌐 Industry Best Practices** → broader theory, alternatives, and trade-offs

---

## Table of Contents

| #   | Question                                                                                           | Category     |
| --- | -------------------------------------------------------------------------------------------------- | ------------ |
| 1   | [How to handle hallucination](#1-how-to-handle-hallucination)                                      | LLM Quality  |
| 2   | [Sudden accuracy drop in production](#2-sudden-accuracy-drop-in-production)                        | LLM Quality  |
| 3   | [RAGAS metrics explained](#3-ragas--faithfulness-relevancy-context-precision-context-recall)       | Evaluation   |
| 4   | [Golden dataset, synthetic dataset, A/B testing](#4-golden-dataset-synthetic-dataset-ab-testing)   | Evaluation   |
| 5   | [Prompt injection & PII handling](#5-prompt-injection-handling-pii-handling)                       | Security     |
| 6   | [MRR and HNSW](#6-mrr-and-hnsw)                                                                    | Retrieval    |
| 7   | [Why a reranker after cosine similarity](#7-why-a-reranker-is-needed-even-after-cosine-similarity) | Retrieval    |
| 8   | [BM25 and hybrid search](#8-bm25-and-why-hybrid-search-is-needed)                                  | Retrieval    |
| 9   | [Updating a document in a vector DB](#9-how-to-update-a-document-already-stored-in-a-vector-db)    | Retrieval    |
| 10  | [Pinecone vs pgvector vs Milvus vs FAISS](#10-choosing-between-pinecone-pgvector-milvus-faiss)     | Retrieval    |
| 11  | [Fallback methods and circuit breakers](#11-fallback-methods-and-circuit-breakers)                 | Resilience   |
| 12  | [Precision, Recall, F1, P@1](#12-precision-recall-f1-p1-and-similar-metrics)                       | Metrics      |
| 13  | [Top-K, Top-P, Temperature](#13-top-k-top-p-temperature)                                           | LLM Controls |
| 14  | [Chunking strategies in RAG](#14-chunking-strategies-in-rag)                                       | Advanced     |
| 15  | [Embedding model selection](#15-embedding-model-selection)                                         | Advanced     |
| 16  | [RAG vs Fine-tuning](#16-rag-vs-fine-tuning--when-to-choose-which)                                 | Advanced     |
| 17  | [Multi-tenancy in vector DBs](#17-multi-tenancy-in-vector-databases)                               | Advanced     |
| 18  | [Agentic loop termination](#18-agentic-loop-termination-and-infinite-loop-prevention)              | Advanced     |
| 19  | [LangGraph vs CrewAI vs AutoGen](#19-langgraph-vs-crewai-vs-autogen--trade-offs)                   | Advanced     |
| 20  | [ColBERT and late interaction](#20-colbert-and-late-interaction-models)                            | Advanced     |
| 21  | [Semantic cache threshold calibration](#21-semantic-cache-design-and-threshold-calibration)        | Advanced     |
| 22  | [Cost optimization in LLM systems](#22-cost-optimization-in-llm-systems)                           | Advanced     |
| 23  | [Debugging multi-agent systems](#23-debugging-multi-agent-systems)                                 | Advanced     |
| 24  | [Knowledge freshness in RAG](#24-knowledge-freshness-in-rag)                                       | Advanced     |
| 25  | [Wave-based parallel tool execution](#25-wave-based-parallel-tool-execution-in-react)              | Advanced     |
| 26  | [ANN and HNSW — how they work](#26-ann-and-hnsw--how-approximate-nearest-neighbor-search-works)    | Retrieval    |
| 27  | [P50, P95, P99 latency percentiles](#27-p50-p95-p99-latency-percentiles)                           | Metrics      |
| 28  | [Citations and source attribution in RAG](#28-citations-and-source-attribution-in-rag)             | Retrieval    |
| 29  | [Deciding chunking strategy and token count](#29-deciding-chunking-strategy-and-token-count)       | Advanced     |
| 30  | [How a reranker works internally](#30-how-a-reranker-works-internally)                             | Retrieval    |

---

## 1. How to Handle Hallucination

> ### 🏗️ In This Project
>
> The system defends against hallucination at **four distinct layers**:
>
> **Layer 1 — Tool-grounded reasoning** (`agents/react_loop.py`)
>
> - Every agent runs inside a ReAct loop
> - Agents can **only** assert facts backed by actual tool return values
> - Tools like `lookup_customer`, `search_policies`, `calculate_refund` return structured JSON from real DBs
> - The LLM reasons over tool results — it never invents data
>
> **Layer 2 — Structured output + multi-strategy parsing** (`agents/output_parser.py`, `agents/output_models.py`)
>
> - Agents must return a specific Pydantic schema
> - Parser tries 4 strategies in order: direct JSON → fenced code block → brace matching → Pydantic validation
> - If **all 4 fail** → `AgentOutputParseError` raised → workflow escalates to HITL
> - Garbled or hallucinated outputs are rejected, not accepted
>
> **Layer 3 — RAGAS faithfulness scoring** (`evaluation/ragas_evaluator.py`)
>
> - Runs after every completed workflow
> - Decomposes agent's answer into atomic claims
> - Verifies each claim against retrieved context via NLI pass
> - Score below threshold = signal that agent asserted something ungrounded
>
> **Layer 4 — Input framing** (`guardrails/input_sanitizer.py`)
>
> - Every user string wrapped in `<<<USER_INPUT>>> ... <<<END_USER_INPUT>>>`
> - Tells the LLM: _this is data to reason about, not instructions to follow_

> ### 💡 Why It Matters
>
> A hallucinated approval or denial has **real financial and legal consequences**.
> Each layer catches a different failure mode:
>
> - Tool-grounding → prevents invented data
> - Structured parsing → rejects malformed claims
> - RAGAS → catches quality drift over time
> - HITL → human reviews anything the system is unsure about

### 🌐 Industry Best Practices

| Technique                             | How It Works                                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **RAG grounding + citations**         | Force model to cite exact chunks; verify cited chunk exists in context at inference time                    |
| **Self-consistency sampling**         | Run the same prompt N=5 times, majority-vote on structured fields; escalate if high variance                |
| **Constitutional AI / self-critique** | After initial answer, prompt: _"Does any claim go beyond what the policies say?"_                           |
| **NLI entailment checks**             | Use `cross-encoder/nli-deberta-v3-small` to verify each sentence against retrieved chunks; flag score < 0.7 |
| **Faithfulness alerting**             | Alert when faithfulness < 0.80 over rolling 100-request window — leading indicator of prompt regression     |

---

## 2. Sudden Accuracy Drop in Production

> ### 🏗️ In This Project
>
> Four independent signal sources help narrow down the root cause fast:
>
> | Signal Source              | File                                        | What it shows                                                                                                               |
> | -------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
> | RAGAS scores as OTEL spans | `telemetry/setup.py` + `ragas_evaluator.py` | Faithfulness / relevancy time-series; sudden cliff is immediately visible in LangSmith                                      |
> | A/B variant score history  | `evaluation/ab_testing.py`                  | Timestamped `ExperimentResult` list per variant — pinpoints exactly when a metric moved                                     |
> | Prompt version registry    | `prompts/registry.py`                       | Full audit trail of prompt activations; rollback = single `activate_version()` call, takes effect in < 5 min (auto-refresh) |
> | Structlog JSON logs        | All agents                                  | `task_id` + `thread_id` on every line; trace a single request end-to-end                                                    |

> ### 💡 Why It Matters
>
> Accuracy drops have **many orthogonal causes** — you need multiple independent signals to separate them. With only one signal source, you're guessing.

### 🌐 Industry Best Practices

**Root cause checklist** (work through in order):

1. **Model version change** — did the provider silently update? Always pin: `gpt-4o-mini-2024-07-18` not `gpt-4o-mini`
2. **Prompt regression** — was a prompt version deployed without eval? Run golden dataset on every PR
3. **Input distribution drift** — are users submitting different kinds of requests? Check embedding centroid shift
4. **Tool / schema change** — did a database column or API response format change silently?
5. **Embedding model update** — invalidated the semantic cache or Pinecone index?

**Proven detection strategies:**

- **Golden dataset regression in CI** — Run full agent pipeline on every pull request; block merge if mean RAGAS drops > 3%
- **Shadow evaluation pipeline** — Route 5% of prod traffic to a shadow instance running the candidate change; compare metric distributions before promoting
- **Canary deployment** — Shift traffic 1% → 5% → 25% → 100%; auto-rollback at each gate if metric threshold breached
- **Embedding drift detection** — Compare MMD (Maximum Mean Discrepancy) between recent request embeddings vs golden set embeddings; shift = distribution drift

---

## 3. RAGAS — Faithfulness, Relevancy, Context Precision, Context Recall

> ### 🏗️ In This Project
>
> **File:** `evaluation/ragas_evaluator.py` — class `RefundRAGASEvaluator`
>
> - All 4 metrics run **in parallel** via `asyncio.gather()` inside `run_parallel_evaluation()`
> - Each metric returns an `EvaluationResult` with score (0–1) and duration_ms
> - `overall_score = mean(valid_scores)` in `EvaluationReport`
> - Every score is logged as an **OTEL span attribute** (`ragas.faithfulness`, `ragas.answer_relevancy`, etc.) visible in LangSmith
> - `evaluate_agent_output()` uses **faithfulness only** for quick per-agent spot checks

> ### 💡 Why It Matters
>
> Running all 4 in parallel keeps eval latency bounded.
> Logging scores as span attributes means eval data lives **alongside** execution traces — no separate pipeline needed.

### 🌐 Industry Deep Dive

#### Faithfulness (Groundedness)

> _"Did the model only say things the context supports?"_

- **How it works:** Decompose answer into atomic claims → LLM checks each claim against context via NLI
- **Formula:** `faithfulness = supported_claims / total_claims`
- **No ground truth required** — fully reference-free
- **Failure = hallucination:** any claim not entailed by context is a hallucination

#### Answer Relevancy

> _"Did the answer actually address the question?"_

- **How it works:** Embed the generated answer → compute cosine similarity vs original question
- **Formula:** Generate N reverse-questions from the answer, average their similarity to the original
- Does **not** penalize factual errors — use faithfulness for that

#### Context Precision

> _"Were the retrieved chunks actually useful?"_

- **How it works:** LLM grades each retrieved chunk: "Is this relevant given the ideal answer?"
- Precision@K averages across all K chunks, weighted by rank
- High precision = retrieval returns signal, not noise

#### Context Recall

> _"Did retrieval miss anything important?"_

- **How it works:** Compare ground-truth answer facts against what appeared in retrieved context
- **Formula:** `recall = facts_covered_by_context / total_facts_in_ground_truth`
- **Requires ground truth** — needs reference answers

#### Additional RAGAS Metrics Worth Knowing

| Metric                  | Use Case                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `answer_correctness`    | Weighted combo of semantic similarity + factual overlap vs ground truth                                |
| `context_entity_recall` | Did retrieved context contain the named entities (policy names, product categories) from ground truth? |

> **For this project:** Faithfulness + context recall are most critical. You cannot approve a refund based on a hallucinated policy or one that was never retrieved.

---

## 4. Golden Dataset, Synthetic Dataset, A/B Testing

> ### 🏗️ In This Project
>
> **File:** `evaluation/ab_testing.py` — class `ABTestManager`
>
> **Experiment configuration** (`ExperimentConfig`):
>
> - Define variants with traffic split percentages (must sum to 1.0)
> - `min_samples_per_variant = 30` before results are valid
> - `confidence_level = 0.95` (95% CI)
>
> **Deterministic variant assignment** (`get_variant()`):
>
> - SHA-256 hash over `experiment_id:request_id` → bucket (0.0 to 1.0)
> - **Same request_id always gets the same variant**, even on retries
> - Prevents assignment drift corrupting both variant sample sets
>
> **Statistical analysis** (`get_report()`):
>
> - Runs **Welch's t-test** (handles unequal variances between groups)
> - Returns: `winner`, `is_significant`, `p_value`, per-variant CI
>
> **Closed loop with RAGAS:**
>
> - `EvaluationReport.overall_score` from `ragas_evaluator.py` feeds directly into `record_result()`
> - Prompt versions stored in `prompts/registry.py` serve as the experiment variants

> ### 💡 Why It Matters
>
> Welch's t-test is chosen over Student's t-test because variant groups often have **different sample sizes and variances** in practice. Consistent hashing is critical — without it, retries could flip variants and corrupt both sample sets.

### 🌐 Industry Best Practices

#### Golden Dataset

- Manually curated set of `(question, expected_answer, expected_contexts)` tuples
- Labeled by **subject matter experts**, immutable once created
- Covers all edge cases in your domain
- Run on every code or prompt change — target **≥ 200 rows** for meaningful coverage
- **Use:** regression gate before any deployment

#### Synthetic Dataset

Three generation methods:

| Method               | How It Works                                                           | Best For                     |
| -------------------- | ---------------------------------------------------------------------- | ---------------------------- |
| **Self-instruct**    | Ask LLM to generate question variations from seed docs                 | Expanding coverage quickly   |
| **Evol-instruct**    | Iteratively evolve questions to be harder (add constraints, multi-hop) | Testing edge cases           |
| **Back-translation** | Paraphrase existing questions for lexical diversity                    | Augmenting existing datasets |

> Always quality-filter synthetics using `answer_correctness` vs a reference model before adding to your eval set.

#### A/B Testing — Statistical Traps to Avoid

- **Multiple comparisons:** Running 5 variants simultaneously? Use **Bonferroni correction** (divide α by 5)
- **Early stopping:** Peeking at p-values before `min_samples` inflates false-positive rates — use sequential testing (mSPRT) if you must stop early
- **Effect size:** p < 0.05 with n=10,000 can represent a meaningless difference — always report **Cohen's d** alongside p-value

#### Online vs Offline Evaluation

```
Offline (golden dataset, shadow) → controlled, fast, gates the deploy
Online (A/B, logging)           → real traffic, validates impact at scale
```

**Tooling:** Arize Phoenix · Weights & Biases · LangFuse

---

## 5. Prompt Injection Handling, PII Handling

> ### 🏗️ In This Project
>
> #### Prompt Injection — `guardrails/input_sanitizer.py`
>
> Four layers of defense, applied in order:
>
> 1. **Pattern matching** — 16 compiled regex patterns catch known signatures:
>    - "ignore previous instructions", "act as", "pretend you are"
>    - "override safety", "reveal system prompt", "jailbreak", "DAN"
>    - On match: replace with `[keyword]` token (soft mode) or raise `PromptInjectionError` (strict mode)
> 2. **Character filtering** — strips characters used in stealth attacks:
>    - Control chars: `0x00–0x08`, `0x0B–0x1F`, `0x7F–0x9F`
>    - Unicode direction overrides: `U+200B–200F`, `U+202A–202E`, `U+2066–206F`
> 3. **Length enforcement** — hard cap at **500 characters** (token-stuffing prevention)
> 4. **Delimiter wrapping** — output wrapped in `<<<USER_INPUT>>> ... <<<END_USER_INPUT>>>`
>    - Signals to LLM: _this is data, not a system instruction_
>
> #### PII Handling — `guardrails/pii_handler.py`
>
> Two distinct modes for different purposes:
>
> | Mode         | Function          | Use case                                                 | Reversible? |
> | ------------ | ----------------- | -------------------------------------------------------- | ----------- |
> | Irreversible | `mask_pii()`      | LLM prompts — PII never leaves the system                | ❌ No       |
> | Reversible   | `PIIMasker` class | Internal pipelines — original stored in `_mappings` dict | ✅ Yes      |
>
> **PII patterns detected:** email (partial: `j***@domain.com`) · phone → `[PHONE_REDACTED]` · credit card → `[CC_REDACTED]` · SSN → `[SSN_REDACTED]` · IP → `[IP_REDACTED]`
>
> **Orchestration** (`guardrails/runner.py`): `validate_input()` calls sanitize + mask before any agent sees data; each step wrapped in OTEL span for compliance audit trail in LangSmith.

> ### 💡 Why It Matters
>
> - **Prompt injection** = OWASP LLM Top 10 item **LLM01** — most common LLM-specific vulnerability. A successful injection could approve fraudulent refunds, leak customer data, or reveal the system prompt.
> - **PII masking** = GDPR + SOC 2 requirement. The LLM provider (OpenAI) should **never** receive raw PII in prompts.

### 🌐 Industry Best Practices

#### Defense-in-Depth for Injection

| Technique                       | Why Blocklist Alone Isn't Enough                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Allowlist validation**        | Regex patterns are a blocklist — attackers find novel phrasing. Accept only input consistent with legitimate refund language.        |
| **LLM-as-judge detector**       | Secondary cheap model (gpt-4.1-nano): _"Does this text attempt to override instructions?"_ Catches semantic injections regex misses. |
| **Tokenizer anomaly detection** | High token-to-character ratio or rare token sequences = likely Unicode homoglyph attack                                              |
| **Strict role separation**      | Never interpolate untrusted user text into the `system` message — use `user` role exclusively                                        |

#### Defense-in-Depth for PII

- **Data minimization** — include PII in prompts only if strictly required; refund reason rarely needs the email
- **Scan LLM outputs** — run PII patterns on model responses before logging; models occasionally echo PII from context
- **Differential privacy for embeddings** — add calibrated Gaussian noise to embedding vectors stored in vector DB; prevents re-identification attacks
- **Tokenization-safe placeholders** — ensure `[PHONE_REDACTED]` is a **single token** in the target tokenizer; multi-token placeholders cause attention artifacts

---

## 6. MRR and HNSW

> ### 🏗️ In This Project
>
> **File:** `vectordb/pinecone_store.py`
>
> - Index: cosine similarity, **1536 dimensions** (`text-embedding-3-small`), `top_k=3`
> - Pinecone implements **HNSW internally** — not directly configurable; managed service auto-tunes `ef_construction` and `M`
> - MRR is **not explicitly computed** in the codebase, but RAGAS context precision is its conceptual sibling

> ### 💡 Why It Matters
>
> Pinecone trades direct HNSW parameter control for **zero operational burden** — no index builds, no shard management, no replication config.

### 🌐 Industry Deep Dive

#### MRR — Mean Reciprocal Rank

> _"How high up does the first correct result appear?"_

```
MRR = (1/|Q|) × Σ (1 / rank_i)
```

| First correct result at rank... | MRR contribution |
| ------------------------------- | ---------------- |
| #1                              | 1.0              |
| #2                              | 0.5              |
| #3                              | 0.33             |
| #10                             | 0.1              |

- **MRR@K** — only considers top-K; a result outside top-K contributes 0
- **When to use:** exactly one correct answer per query (policy lookup, entity resolution)
- **When NOT to use:** multiple relevant docs per query → use MAP or NDCG instead
- **Project relevance:** If the policy agent always gets the right policy at rank-3 instead of rank-1, it reads more noise first → increases hallucination risk and token cost

#### HNSW — Hierarchical Navigable Small World

**Structure:**

- Multi-layer graph; each node = a vector
- **Bottom layer:** dense proximity graph (every vector connected to K nearest neighbors)
- **Upper layers:** increasingly sparse, long-range "highway" links
- Controlled by `M` parameter (edges per node)

**Search:** Start at top layer, greedily traverse toward query → drop one layer → repeat → converge to ANN in **O(log N)**

**Key tuning parameters:**

| Parameter            | Effect                                                    | Typical value |
| -------------------- | --------------------------------------------------------- | ------------- |
| `M` (edges per node) | ↑ M = better recall, more memory, slower build            | 16–64         |
| `ef_construction`    | Beam width during build — ↑ = better recall, slower build | 100–400       |
| `ef_search`          | Beam width at query time — tunable without rebuild        | 50–200        |

**Comparison with alternatives:**

| Algorithm  | Recall | Query Speed | Memory     | Build Speed | Updates |
| ---------- | ------ | ----------- | ---------- | ----------- | ------- |
| **HNSW**   | High   | Fast        | High       | Slow        | Slow    |
| IVF-Flat   | Medium | Medium      | Low        | Fast        | Fast    |
| **IVF-PQ** | Medium | Fast        | Very Low   | Fast        | Fast    |
| ScaNN      | High   | Very Fast   | Medium     | Slow        | Hard    |
| DiskANN    | High   | Fast        | Low (disk) | Slow        | Slow    |

> **IVF-PQ** is the go-to for memory-constrained systems: vectors compressed into subspace codes, reducing memory 8–32× at small recall cost. At 10M × 1536-dim vectors: ~60 GB uncompressed → under 2 GB with IVF-PQ.

---

## 7. Why a Reranker Is Needed Even After Cosine Similarity

> ### 🏗️ In This Project
>
> **File:** `vectordb/pinecone_store.py`
>
> - Current setup: cosine similarity → top 3 policy chunks → policy agent
> - **No reranker stage** — this is a known gap
> - With only 8 policy documents it works fine, but would degrade at scale

> ### 💡 Why It Matters
>
> With hundreds of policies, cosine alone can return 3 "close enough" chunks that individually miss the precise answer. A reranker catches this.

### 🌐 The Two-Stage Retrieval Pipeline

#### Stage 1 — Bi-encoder (Cosine Similarity)

- Each document embedded **independently** at index time → single vector per doc
- Query embedded at runtime → cosine against all pre-computed vectors → ANN search
- **Speed:** milliseconds over millions of docs (precomputed, ANN-approximate)
- **Weakness:** query and document are **never seen together** — no cross-attention, no query-conditioned signal

#### Stage 2 — Cross-encoder (Reranker)

- Input: `[CLS] query [SEP] document [SEP]` — fed **jointly** to transformer
- Full attention between every query token and every document token
- **Accuracy:** much higher than bi-encoder — captures fine-grained token interactions
- **Speed:** cannot precompute → O(K) full forward passes at query time → only practical on small candidate set

#### The Pattern: Use Both Together

```
Cosine (recall)        →  retrieve top-50   →  fast, approximate
Reranker (precision)   →  rerank to top-5   →  slow, accurate, on small set
```

#### Concrete Example

Query: _"I'm a Gold member and the product was damaged in transit"_

| Rank | Cosine retrieval                 | Cross-encoder result              |
| ---- | -------------------------------- | --------------------------------- |
| 1    | General damage policy (0.87)     | **Gold tier benefits** ← moved up |
| 2    | Gold tier benefits (0.84)        | General damage policy             |
| 3    | Electronics return policy (0.82) | Electronics return policy         |

The cross-encoder sees the full query + full policy text simultaneously and recognizes the Gold-tier document is more directly relevant. Cosine would have placed it at rank 2, potentially causing the policy agent to miss the tier benefit.

#### Common Reranker Models

| Model                                  | Type        | Notes                    |
| -------------------------------------- | ----------- | ------------------------ |
| `Cohere Rerank`                        | Managed API | Strong multilingual      |
| `BAAI/bge-reranker-v2-m3`              | Open source | Strong on technical text |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Open source | Lightweight, English     |
| `mixedbread-ai/mxbai-rerank-large-v1`  | Open source | Top MTEB reranking score |

---

## 8. BM25 and Why Hybrid Search Is Needed

> ### 🏗️ In This Project
>
> **File:** `vectordb/pinecone_store.py`
>
> - Current setup: **pure dense search** — cosine similarity on `text-embedding-3-small` embeddings
> - BM25 is **not implemented** — this is a gap
> - Would matter if users quote order IDs (`ORD-98765`), policy codes, or exact product SKUs

> ### 💡 Why It Matters
>
> Dense embeddings handle _"my item broke"_ perfectly but fail on exact tokens like order numbers. BM25 fails on paraphrases but nails exact matches. Hybrid gives you both.

### 🌐 Industry Deep Dive

#### BM25 Formula

```
score(d, t) = IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d| / avgdl))
```

| Parameter | What it controls                                       | Typical value |
| --------- | ------------------------------------------------------ | ------------- |
| `IDF(t)`  | Rare terms get higher weight                           | —             |
| `k1`      | TF saturation (diminishing returns for repeated terms) | 1.2–2.0       |
| `b`       | Length normalization strength                          | 0.75          |

BM25 produces a **sparse** score vector — only terms present in the document have non-zero weights.

#### When Each Approach Wins

| Scenario                                                        | BM25 wins | Dense wins |
| --------------------------------------------------------------- | --------- | ---------- |
| Exact product codes / order IDs                                 | ✅        | ❌         |
| Out-of-vocabulary terms (new product names, typos)              | ✅        | ❌         |
| Paraphrases ("item broke" → "product malfunction")              | ❌        | ✅         |
| Semantic intent ("unhappy with purchase" → satisfaction policy) | ❌        | ✅         |
| Multilingual queries                                            | ❌        | ✅         |

#### Combining Them — Reciprocal Rank Fusion (RRF)

```
RRF_score(d) = 1 / (k + rank_sparse(d))  +  1 / (k + rank_dense(d))
```

- `k = 60` (smoothing constant)
- **Rank-based not score-based** — sidesteps incompatible scales (BM25: 10–50, cosine: 0–1)
- Alternative: weighted linear combination `α × norm(bm25) + (1-α) × cosine` — α tuned on validation set

#### Ecosystem Support

| Platform                   | Hybrid support                                          |
| -------------------------- | ------------------------------------------------------- |
| Pinecone                   | Native sparse-dense via `sparse_values` field in upsert |
| Elasticsearch / OpenSearch | BM25 + kNN in single query                              |
| Weaviate                   | Built-in RRF                                            |
| Qdrant                     | Sparse + dense vectors in same collection               |

---

## 9. How to Update a Document Already Stored in a Vector DB

> ### 🏗️ In This Project
>
> **Files:** `vectordb/pinecone_store.py` · `cache/redis_cache.py`
>
> - 8 policy documents with stable deterministic IDs (e.g., `"policy_general"`, `"policy_electronics"`)
> - `initialize_policies()` calls `index.upsert(vectors=vectors)` — if ID exists, it overwrites
> - **Critical:** after any update → must call `cache.invalidate_all()` to flush Redis semantic cache
>   - Without this: stale cached response (similarity ≥ 0.92) served for up to 1 hour

> ### 💡 Why It Matters
>
> The semantic cache is a second copy of the retrieval results. Updating Pinecone without flushing the cache means users get old answers for similar queries until TTL expires.

### 🌐 Update Strategies

| Strategy                          | How                                                                      | Downtime?    | When to use                            |
| --------------------------------- | ------------------------------------------------------------------------ | ------------ | -------------------------------------- |
| **Upsert by ID** _(this project)_ | Pinecone overwrites vector + metadata atomically if ID exists            | ❌ None      | Stable IDs, same embedding dimension   |
| **Delete + reinsert**             | `index.delete(ids=[...])` then `index.upsert(...)`                       | Brief window | When ID scheme changes                 |
| **Metadata versioning**           | Insert new doc with `version=v2`, filter queries on `version=latest`     | ❌ None      | Need instant rollback capability       |
| **Namespace swap**                | Write to `policies_v2` namespace, switch pointer, delete `policies_v1`   | ❌ None      | Full corpus replacement, zero-downtime |
| **Re-embed on model change**      | Rebuild entire index — delete, create new with new dimension, upsert all | Full rebuild | Embedding model version change         |

#### Always Invalidate the Cache

Regardless of which strategy you use:

```
update doc in Pinecone
→ call cache.invalidate_all()     # flushes all policy_cache:* Redis keys
→ next query goes to Pinecone fresh
```

#### Embedding Model Change = Full Rebuild

> If you change `text-embedding-3-small` → `text-embedding-3-large`, **you cannot upsert** — dimensions differ (1536 vs 3072). You must delete the index, create a new one with the correct dimension, re-embed all documents, and upsert everything. This is why pinning the embedding model version is as critical as pinning the LLM version.

---

## 10. Choosing Between Pinecone, pgvector, Milvus, FAISS

> ### 🏗️ In This Project
>
> **File:** `vectordb/pinecone_store.py`
>
> - **Chose Pinecone** for zero operational burden
> - Project already uses PostgreSQL — pgvector was the viable alternative that would eliminate a separate managed service
> - Trade-off accepted: Pinecone per-query cost in exchange for no index tuning, no sharding, no replication

### 🌐 Comparison Matrix

| Dimension          | FAISS            | pgvector             | **Pinecone**           | Milvus/Zilliz       | Qdrant              |
| ------------------ | ---------------- | -------------------- | ---------------------- | ------------------- | ------------------- |
| Scale              | Millions (RAM)   | 1–10M                | Billions               | Billions            | ~100M               |
| Deployment         | In-process lib   | PG extension         | Managed SaaS           | Self-hosted / cloud | Self-hosted / cloud |
| Ops burden         | None (no server) | Low (existing PG)    | **None**               | High                | Medium              |
| ACID / SQL joins   | ❌               | ✅ Full SQL          | ❌                     | ❌                  | ❌                  |
| Metadata filtering | Basic            | Full WHERE clause    | Filter expressions     | Rich scalar filters | Payload filters     |
| Hybrid search      | ❌               | Partial (+ tsvector) | ✅ Native sparse-dense | ✅                  | ✅                  |
| GPU support        | ✅               | ❌                   | ❌                     | ✅                  | ❌                  |
| Real-time upsert   | Rebuild needed   | ✅                   | ✅                     | ✅                  | ✅                  |
| Cost at scale      | Free             | PG hosting           | Per-query + storage    | Infra cost          | Infra cost          |

### When to Use Each

**FAISS** — Research, offline batch, prototyping only. No network API, no persistence, no replication.

**pgvector** — Already on PostgreSQL + need relational joins + under 5M vectors. One-line extension install. Index with:

```sql
CREATE INDEX ON policies USING hnsw(embedding vector_cosine_ops) WITH (m=16, ef_construction=200);
```

**Pinecone** — Zero ops, managed scaling, sparse-dense hybrid needed, team can't afford to operate infra. _(This project's choice.)_

**Milvus / Zilliz** — Billions of vectors, GPU-accelerated index building, willing to operate a distributed system. Complex: separate coordinator, query node, data node, index node processes.

**Qdrant** — Rust-based, fast, strong payload filters, lower ops complexity than Milvus. Named vectors (multiple embeddings per record) + quantization built-in.

**Weaviate** — Needs multi-modal embeddings or tight GraphQL integration; modular architecture (swap embedding models via modules).

### Decision Heuristic

```
Existing PG + small scale (<5M)  →  pgvector
Zero ops / managed needed        →  Pinecone
Billion-scale + self-hosted      →  Milvus / Weaviate
Research / offline only          →  FAISS
```

---

## 11. Fallback Methods and Circuit Breakers

> ### 🏗️ In This Project
>
> **File:** `executor/resilience.py`
>
> Three interlocking patterns, plus HITL as the ultimate human safety valve:
>
> #### CircuitBreaker — State Machine
>
> ```
> CLOSED ──(5 consecutive failures)──► OPEN
>   ▲                                    │
>   │                                    │ (30 seconds)
>   │                                    ▼
>   └──(2 successes in HALF_OPEN)── HALF_OPEN
> ```
>
> | State         | Behavior                                                   |
> | ------------- | ---------------------------------------------------------- |
> | **CLOSED**    | All requests pass through normally                         |
> | **OPEN**      | Raises `CircuitBreakerOpen` immediately — no I/O wait      |
> | **HALF_OPEN** | One test request allowed; success → CLOSED, failure → OPEN |
>
> - Implemented as `async with circuit_breaker:` context manager
> - `__aexit__` auto-calls `_on_success()` or `_on_failure()`
>
> #### RetryBudget — Token-Bucket
>
> Two limits enforced simultaneously (both must pass):
>
> - **Absolute cap:** max 20 retries per 60-second rolling window
> - **Ratio cap:** retries ≤ 10% of total requests in same window
>
> **Backoff formula:**
>
> ```
> delay = min(base_delay × 2^attempt, max_delay) × uniform(0.5, 1.5)
> ```
>
> Jitter (±50%) prevents thundering herd — all workers retrying simultaneously after outage clears.
>
> #### ModelFallbackChain
>
> ```
> gpt-4o-mini  ──(circuit open or budget exhausted)──►  gpt-4.1-nano
> ```
>
> - Transparent to agents — they receive a response, don't know which model produced it
> - Caught in `agents/react_loop.py` when `use_fallback=True`
>
> #### HITL — Ultimate Fallback
>
> If all automated fallbacks fail → `hitl_required=True` in state → supervisor routes to HITL node → task parked in `HITLTask` DB table for human review. No request is ever silently dropped.

> ### 💡 Why It Matters
>
> Without circuit breaker: 20 workers × 30s timeout on a dead LLM = 600 seconds of blocked threads.
> With circuit breaker: after 5 failures, all subsequent requests fail in **microseconds**, freeing worker capacity.
>
> Without retry budget: a 10-minute outage causes every worker to retry indefinitely, burning API quota. The 10% ratio cap bounds retry traffic even during sustained outages.

### 🌐 Industry Best Practices

| Pattern                            | What it solves                                                                                                          |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Bulkhead**                       | Isolate resource pools per service — Kafka outage can't exhaust DB connection pool; use `asyncio.Semaphore` per service |
| **Timeout + deadline propagation** | Set total request deadline at edge (e.g., 30s); each downstream call gets `(original_deadline - elapsed)`               |
| **Graceful degradation**           | LLM unavailable → serve cached response or return deterministic rule-based fallback                                     |
| **Multi-cloud routing**            | `ChatOpenAI` and `ChatAnthropic` share the same LangChain interface; LiteLLM proxies 100+ models with one API           |

---

## 12. Precision, Recall, F1, P@1 and Similar Metrics

> ### 🏗️ In This Project
>
> The policy agent produces a **3-class output**: `approved` · `denied` · `partial`
> HITL escalations = cases the system was not confident enough → treat as a 4th "escalate" class
>
> RAGAS computes context precision and context recall on the retrieval path (see Q3).

### 🌐 Classification Metrics

#### The Confusion Matrix (Binary: approved vs not-approved)

```
                    Predicted Approved   Predicted Not Approved
Actual Approved          TP                      FN
Actual Not Approved      FP                      TN
```

| Metric        | Formula                    | What it means in this system                                                                       |
| ------------- | -------------------------- | -------------------------------------------------------------------------------------------------- |
| **Precision** | TP / (TP + FP)             | Of approved decisions, what % were genuinely eligible? High precision = fewer fraudulent approvals |
| **Recall**    | TP / (TP + FN)             | Of eligible requests, what % were approved? High recall = fewer wrongful denials                   |
| **F1**        | 2 × P × R / (P + R)        | Harmonic mean — use when both errors cost equally                                                  |
| **F-beta**    | (1+β²) × P × R / (β²P + R) | β < 1 weights precision more (fraud-sensitive); β > 1 weights recall more                          |

> **For a refund system with fraud risk:** β < 1 is typically right — false approvals (fraud) cost more than false denials (bad UX).

#### Multi-class F1 Variants

| Variant         | How averaged                           | Best for                                                                     |
| --------------- | -------------------------------------- | ---------------------------------------------------------------------------- |
| **Macro F1**    | Mean of per-class F1, equal weight     | All 3 classes matter equally                                                 |
| **Micro F1**    | Aggregate TP/FP/FN across all classes  | Dominated by majority class — use for imbalanced data where majority matters |
| **Weighted F1** | Per-class F1 weighted by class support | Imbalanced dataset where class frequency matters                             |

### 🌐 Retrieval Metrics

| Metric     | Formula                            | What it asks                                        |
| ---------- | ---------------------------------- | --------------------------------------------------- |
| **P@K**    | relevant in top-K / K              | Of top-K chunks, what fraction were useful?         |
| **P@1**    | Is the top result relevant?        | Strictest — critical for single-answer systems      |
| **R@K**    | relevant in top-K / total relevant | Did retrieval miss anything?                        |
| **NDCG@K** | DCG / IDCG (rank-weighted)         | Is the ordering within top-K good?                  |
| **MAP**    | Mean AP across queries             | Multiple relevant docs per query, all must be found |

**NDCG formula:**

```
DCG@K  = Σ (2^rel_i - 1) / log2(i + 1)   for i = 1 to K
NDCG@K = DCG@K / IDCG@K
```

> **For this project:** Run 50 policy queries with known relevant chunks. Measure P@1, NDCG@3, and RAGAS context recall together — these three metrics fully characterize retrieval pipeline fitness.

### 🌐 Each Metric Explained in Plain Words

#### Precision — "When I say yes, how often am I right?"

> Of all the refunds the agent **approved**, what fraction were genuinely eligible?

- **Formula:** `Precision = TP / (TP + FP)`
- **Punishes false positives** — approving a refund that should have been denied (fraud, abuse)
- **High precision** = the agent is cautious; almost everything it approves is legitimate
- **Low precision** = it approves too eagerly, letting fraudulent refunds through
- **When you care most:** the cost of a wrong "yes" is high. For refunds, a false approval = real money lost.

#### Recall (Sensitivity) — "Of all the real yeses, how many did I catch?"

> Of all the refunds that **should** have been approved, what fraction did the agent actually approve?

- **Formula:** `Recall = TP / (TP + FN)`
- **Punishes false negatives** — denying a refund that was actually eligible
- **High recall** = the agent rarely wrongly denies a deserving customer
- **Low recall** = many eligible customers get wrongly rejected (bad customer experience)
- **When you care most:** the cost of a missed "yes" is high. In medical screening, a missed disease (FN) is far worse than a false alarm.

#### The Precision–Recall Trade-off

> You almost always trade one for the other. Approve more freely → recall ↑ but precision ↓. Approve more strictly → precision ↑ but recall ↓.

The "knob" you turn is the **decision threshold** (e.g. only auto-approve when confidence ≥ 0.8). Raising the threshold makes the system stricter → higher precision, lower recall.

#### F1 Score — "One number that balances both"

> The **harmonic mean** of precision and recall. It stays low unless _both_ are reasonably high.

- **Formula:** `F1 = 2 × (P × R) / (P + R)`
- **Why harmonic (not simple average)?** If precision = 1.0 and recall = 0.0, a plain average gives 0.5 (looks okay), but F1 gives 0.0 (correctly flags it as useless). The harmonic mean refuses to reward a model that is good at one metric and terrible at the other.
- **Use when:** false positives and false negatives cost roughly the same, and you want a single comparison number.

#### F-beta — "When one error matters more than the other"

> A weighted version of F1. The `β` knob decides whether precision or recall gets more weight.

- **Formula:** `F-beta = (1 + β²) × (P × R) / (β² × P + R)`
- `β < 1` (e.g. **F0.5**) → **weights precision more** → use for fraud-sensitive systems where a false approval is expensive
- `β > 1` (e.g. **F2**) → **weights recall more** → use for screening where a missed positive is expensive
- `β = 1` → exactly F1 (equal weight)

#### Accuracy — "How many did I get right overall?"

> The fraction of _all_ predictions that were correct.

- **Formula:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **The trap — class imbalance:** if 95% of refund requests are legitimately approved, a lazy model that approves _everything_ scores 95% accuracy while catching **zero** fraud. This is why accuracy alone is misleading; precision/recall/F1 expose the failure.

### 🧮 A Concrete Example (No Code)

Evaluate the policy agent on **10 refund requests** where the correct decision is already known. Treat `approved` as the positive class:

| Request | Should be | Agent said | Outcome                           |
| ------- | --------- | ---------- | --------------------------------- |
| 1       | approved  | approved   | TP ✅                             |
| 2       | approved  | approved   | TP ✅                             |
| 3       | approved  | denied     | FN ❌ (missed an eligible refund) |
| 4       | approved  | approved   | TP ✅                             |
| 5       | denied    | approved   | FP ❌ (approved a bad one)        |
| 6       | denied    | denied     | TN ✅                             |
| 7       | denied    | denied     | TN ✅                             |
| 8       | approved  | approved   | TP ✅                             |
| 9       | denied    | approved   | FP ❌ (approved a bad one)        |
| 10      | denied    | denied     | TN ✅                             |

**Step 1 — count outcomes:** TP = 4, FP = 2, FN = 1, TN = 3

**Step 2 — plug into the formulas:**

```
Precision = TP / (TP + FP) = 4 / (4 + 2) = 0.667   → 67% of approvals were correct
Recall    = TP / (TP + FN) = 4 / (4 + 1) = 0.800   → caught 80% of eligible refunds
F1        = 2 × (0.667 × 0.800) / (0.667 + 0.800) = 0.727
Accuracy  = (TP + TN) / 10 = (4 + 3) / 10          = 0.700
```

**Step 3 — interpret:** Recall (0.80) > Precision (0.667). The agent rarely _denies_ a deserving customer, but it lets **2 bad refunds through** (the false positives). In a fraud-sensitive refund system that is the dangerous direction — you'd tune the threshold to **raise precision**, accepting slightly lower recall, and track it with **F0.5** instead of F1.

#### Multi-class Averaging (approved / denied / partial)

The policy agent has 3 classes, so precision/recall/F1 are computed **per class**, then combined:

| Averaging method | How it combines                                                      | Use when                                                          |
| ---------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Macro F1**     | Plain average of each class's F1 (every class weighted equally)      | All classes equally important, even rare ones                     |
| **Micro F1**     | Pool all TP/FP/FN across classes, then compute once                  | You care about overall correctness; dominated by the common class |
| **Weighted F1**  | Average of per-class F1, weighted by how many samples each class has | Imbalanced data where frequency should count                      |

---

## 13. Top-K, Top-P, Temperature

> ### 🏗️ In This Project
>
> | Agent                     | Model          | Rationale                                         |
> | ------------------------- | -------------- | ------------------------------------------------- |
> | Supervisor, Policy        | `gpt-4o-mini`  | Higher capability needed for multi-step reasoning |
> | Validation, Communication | `gpt-4.1-nano` | Simpler tasks, cost-optimized                     |
>
> - All agents use **structured output** (Pydantic models via `agents/output_models.py`) → `temperature=0` is implicitly the right choice (deterministic classification/parsing)
> - **Pinecone `top_k=3`** → this is retrieval top-K, completely different from LLM sampling Top-K
> - The system never explicitly tunes temperature/top-p — OpenAI defaults are used

> ### 💡 Why It Matters
>
> Confusing Pinecone's `top_k=3` (number of retrieved documents) with LLM sampling's `top_k` (token sampling width) is a common interview mistake. They're the same parameter name in completely different systems.
>
> For structured output generation, temperature near 0 is correct — you want deterministic JSON, not creative variation.

### 🌐 How These Controls Work

#### Temperature

Divides every token's logit by T before softmax:

$$P(token_i) = \frac{e^{logit_i / T}}{\sum_j e^{logit_j / T}}$$

| T value   | Effect                                                             |
| --------- | ------------------------------------------------------------------ |
| `0`       | Argmax — always pick the highest probability token (deterministic) |
| `0.1–0.3` | Near-deterministic, small variation                                |
| `0.7–1.0` | Creative, diverse outputs — chatbots, brainstorming                |
| `> 1.5`   | Random, incoherent                                                 |

> **Rule:** Use `temperature=0` for classification, JSON extraction, any structured output. Use `0.7–1.0` for open-ended generation.

#### Top-K Sampling

Truncates the distribution to K highest-probability tokens, then samples within them.

- Prevents any catastrophically low-probability token from being chosen
- Small K (10–50) → conservative; Large K (100–500) → more diverse
- Problem: the cutoff is arbitrary — "top 50" at the start of a sentence vs. mid-sentence behaves very differently

#### Top-P (Nucleus Sampling)

Selects the smallest set of tokens whose cumulative probability ≥ P.

- `top_p=0.9` → include enough tokens to cover 90% of the probability mass
- Self-adapting: at high-confidence positions few tokens qualify; at uncertain positions many do
- Preferred over Top-K in production — typically `top_p=0.95`

> **Do NOT set both Top-K and Top-P simultaneously.** They interact unpredictably. Pick one.

#### Retrieval Top-K (Different System)

| Context                 | What "top_k=3" means                                                  |
| ----------------------- | --------------------------------------------------------------------- |
| Pinecone `top_k=3`      | Return the 3 most similar vectors from the index                      |
| LLM sampling `top_k=40` | During token generation, consider only the 40 most likely next tokens |

These are **completely unrelated** despite sharing the name.

---

## 14. Chunking Strategies in RAG

> ### 🏗️ In This Project
>
> **File:** `vectordb/pinecone_store.py`
>
> - 8 policy documents stored as **complete single chunks** — simplest strategy
> - Works fine for 8 documents; as corpus grows, this degrades:
>   - Retrieving a 2,000-token policy doc to answer a 1-sentence question wastes context window
>   - Long irrelevant context increases hallucination surface area
> - **Gap:** No chunk strategy is implemented — ideal next step is document-aware chunking at markdown header boundaries

> ### 💡 Why It Matters
>
> The chunk size is the single biggest lever for retrieval quality. Too large → poor embedding specificity; too small → missing context for the LLM.

### 🌐 Chunking Strategies Compared

| Strategy                        | How                                                            | Best for                           |
| ------------------------------- | -------------------------------------------------------------- | ---------------------------------- |
| **Fixed-size + overlap**        | N tokens per chunk, O tokens overlap between chunks            | Simple corpora, baseline           |
| **Sentence-level**              | Split on sentence boundaries, group to token limit             | Fact-retrieval tasks               |
| **Semantic**                    | Embed each sentence; split at cosine similarity drops          | Topic-diverse documents            |
| **Parent-child** (hierarchical) | Index small child chunks; return large parent to LLM           | Precision retrieval + full context |
| **Document-aware**              | Split at structural boundaries (markdown headers, code blocks) | Legal/policy/structured docs       |

### 🌐 Each Chunking Strategy Explained in Plain Words

#### 1. Fixed-Size Chunking

> Cut the text every N tokens (or characters), regardless of where sentences or paragraphs end.

- **How:** "every 256 tokens → new chunk." A blunt ruler laid over the text.
- **Pro:** dead simple, predictable chunk count, fast.
- **Con:** blindly slices through the middle of a sentence or even a word. A fact split across the cut is now half in one chunk, half in another → both embeddings are weakened.
- **Best for:** quick baselines, uniform text with no structure.

#### 2. Overlapping (Sliding Window) Chunking

> Same as fixed-size, but each chunk **repeats the last few tokens** of the previous one.

- **How:** chunk size 256 with **overlap 32** → chunk 2 starts 32 tokens _before_ chunk 1 ended.
- **Why:** the overlap is a safety net. If an important sentence lands on a boundary, the overlap guarantees it appears _whole_ in at least one chunk.
- **Trade-off:** overlap = duplicated text = more vectors, more storage, slightly higher cost. Typical overlap is 10–20% of chunk size.
- **Best for:** almost every fixed-size setup — overlap is the cheap insurance that makes fixed-size usable.

#### 3. Sentence-Level Chunking

> Split on sentence boundaries, then pack whole sentences together until you hit a token budget.

- **How:** detect sentence ends (`.`, `?`, `!`), group consecutive sentences up to ~256 tokens, never cutting mid-sentence.
- **Pro:** every chunk is grammatically self-contained → clean embeddings.
- **Con:** sentence length varies, so chunk sizes are uneven; two unrelated topics in adjacent sentences can still land in the same chunk.
- **Best for:** fact-retrieval over prose, FAQs, Q&A-style content.

#### 4. Paragraph-Level Chunking

> One paragraph (or a few small ones grouped) = one chunk. Split on blank lines / `\n\n`.

- **How:** authors already grouped related ideas into paragraphs — respect that natural boundary.
- **Pro:** preserves a complete unit of thought; very little semantic bleed between chunks.
- **Con:** paragraphs vary wildly in length — a 1,000-token paragraph may still need a secondary split; a one-line paragraph may be too small to be useful alone.
- **Best for:** articles, documentation, and policies written in well-structured prose.

#### 5. Recursive Character/Text Splitting

> Try to split on the **biggest separator first**; if a piece is still too large, recurse down to the next-smaller separator.

- **How:** a priority list of separators, e.g. `["\n\n", "\n", ". ", " ", ""]`.
  1. First split on paragraphs (`\n\n`).
  2. Any paragraph still over the limit → split on lines (`\n`).
  3. Still too big → split on sentences (`. `).
  4. Still too big → split on words, then characters as the last resort.
- **Why it's the popular default** (LangChain's `RecursiveCharacterTextSplitter`): it keeps the **largest meaningful unit that fits**, only breaking finer when forced. You get paragraph-sized chunks when possible and never overflow the limit.
- **Best for:** general-purpose mixed text — the safe, sensible default when you're not sure.

#### 6. Document-Aware (Structural) Chunking

> Split at the document's own structural markers — markdown headers, HTML tags, code function boundaries.

- **How:** for markdown policies, split at `##` / `###` headers so each chunk = exactly one section, keeping the heading as a title.
- **Pro:** chunks align with the author's logical sections; the heading itself adds useful context to the embedding.
- **Con:** only works when the document actually has structure (headers, tags).
- **Best for:** legal/policy docs, technical docs, code. **This is the recommended fit for this project's markdown policy corpus.**

#### 7. Semantic Chunking

> Let _meaning_ decide the boundaries, not punctuation or length.

- **How:**
  1. Split into sentences and embed each one.
  2. Walk through sentence by sentence, measuring the **cosine similarity** between consecutive sentences.
  3. When similarity **drops sharply** (the topic shifted), start a new chunk there.
- **Pro:** boundaries land exactly where the subject changes → each chunk is a single coherent topic. Best retrieval quality.
- **Con:** most expensive — you must embed every sentence _before_ you can even chunk; more compute at indexing time.
- **Best for:** topic-diverse documents where one file covers many unrelated subjects.

#### 8. Parent-Child (Hierarchical) Chunking

> Index **small** chunks for precise matching, but return the **large** parent to the LLM for full context.

- **How:** embed small child chunks (e.g. single sentences). On a hit, don't hand the tiny chunk to the LLM — hand it the **parent** (the whole section the child came from).
- **Why:** solves the core tension — small chunks retrieve precisely, big chunks give the LLM enough context. You get both.
- **Best for:** precision retrieval where the LLM still needs surrounding context to answer well. Often layered **on top of** document-aware chunking.

### 📊 Quick Comparison

| Strategy           | Splits on                 | Respects meaning? | Cost   | Best for              |
| ------------------ | ------------------------- | ----------------- | ------ | --------------------- |
| Fixed-size         | Token count               | ❌                | Lowest | Baselines             |
| Overlapping        | Token count + repeat      | ❌ (but safer)    | Low    | Default safety net    |
| Sentence           | Sentence ends             | Partly            | Low    | Prose, FAQs           |
| Paragraph          | Blank lines               | Partly            | Low    | Articles, docs        |
| Recursive splitter | Priority of separators    | Partly            | Low    | General default       |
| Document-aware     | Headers / structure       | ✅ structure      | Low    | **Policy/legal/code** |
| Semantic           | Embedding similarity drop | ✅ meaning        | High   | Topic-diverse docs    |
| Parent-child       | Two levels (small + big)  | ✅                | Medium | Precision + context   |

#### For This Refund Policy Corpus

Policy documents have clear markdown section structure, so the natural fit is:

- **Document-aware** split at `##` headers → each section becomes a chunk
- Cap at ~512 tokens per chunk; add ~50-token **overlap** at boundaries as insurance
- Layer **parent-child** on top: retrieve the section chunk, but feed the full policy to the LLM

```
## 3. Electronics Return Policy   ← section boundary  (document-aware)
### 3.1 Standard Returns          ← child chunk boundary (parent-child)
```

This **document-aware + parent-child** hybrid is the recommended upgrade path.

---

## 15. Embedding Model Selection

> ### 🏗️ In This Project
>
> **Files:** `config.py` · `vectordb/pinecone_store.py` · `cache/redis_cache.py`
>
> - Model: `text-embedding-3-small`, dimension: **1536**
> - Used in **two** places: Pinecone indexing/querying AND Redis semantic cache key computation
> - Changing the model requires: rebuild Pinecone index + invalidate Redis cache + update `EMBEDDING_DIMENSION` in config

> ### 💡 Why It Matters
>
> Embedding model is a foundational choice. All stored vectors are in that model's space — you can never mix vectors from different models in the same index. Consistency is the #1 constraint.

### 🌐 Selection Guide

#### Benchmark: MTEB (Massive Text Embedding Benchmark)

56 tasks, 7 categories. Always look at the **Retrieval** subcategory specifically — overall rank ≠ retrieval rank.

| Model                                     | Dims | Notes                                                      |
| ----------------------------------------- | ---- | ---------------------------------------------------------- |
| `text-embedding-3-small` _(this project)_ | 1536 | Good baseline, cost-effective                              |
| `text-embedding-3-large`                  | 3072 | ~5× cost, higher recall on complex queries                 |
| `BAAI/bge-large-en-v1.5`                  | 768  | Self-hostable, zero per-query cost                         |
| `nomic-embed-text-v1.5`                   | 768  | 8192-token context, strong on long docs                    |
| `intfloat/multilingual-e5-large`          | 1024 | Multilingual — use if policies exist in multiple languages |

#### Matryoshka Embeddings (MRL)

`text-embedding-3-small` and `text-embedding-3-large` support truncation:

```
1536 dims  →  full quality, full cost
 768 dims  →  ~95% quality, half storage
 256 dims  →  ~85% quality, 1/6 storage
```

Useful when Pinecone storage/query cost is a concern at scale.

#### Domain Adaptation

For specialized corpora (legal, financial, medical), fine-tune with contrastive loss:

```python
# sentence-transformers MultipleNegativesRankingLoss
# triplet: (query, positive_doc, negative_doc)
# Typical improvement: +10–20% on domain retrieval
```

> **Golden rule:** Once you choose an embedding model, pin the version and never change it without a full index rebuild.

---

## 16. RAG vs Fine-Tuning — When to Choose Which

> ### 🏗️ In This Project
>
> **Why RAG was chosen:**
>
> 1. **Policies change frequently** — new product categories, tier rule updates, seasonal promotions. RAG = simple document upsert; fine-tuning = expensive retraining cycle.
> 2. **Auditability required** — "approved per Electronics Tier-Gold policy v2.3". RAG shows the retrieved chunk; fine-tuning bakes knowledge into weights opaquely.
> 3. **Base models already reason well** — `gpt-4o-mini` / `gpt-4.1-nano` understand structured reasoning; fine-tuning would add cost without proportional gain.

> ### 💡 Why It Matters
>
> RAG and fine-tuning solve different problems. RAG injects external knowledge at inference time; fine-tuning changes the model's weights. They're not alternatives for the same problem — they're complements.

### 🌐 Decision Framework

| Dimension                  | RAG                             | Fine-Tuning                |
| -------------------------- | ------------------------------- | -------------------------- |
| Knowledge change frequency | Frequent (monthly/weekly) ✅    | Stable (annual) ✅         |
| Auditability               | ✅ — source chunk is visible    | ❌ — opaque weights        |
| Output format control      | Limited                         | ✅ Strong                  |
| Latency                    | +50–200ms retrieval overhead    | Fast — no retrieval        |
| Cost                       | Per-query embedding + retrieval | Upfront training cost      |
| Failure mode               | Bad retrieval → bad answer      | Hallucination + forgetting |
| Knowledge freshness        | ✅ Real-time upsert             | ❌ Requires retraining     |

#### RAFT (Best of Both)

Fine-tune the model on **(question, noisy_retrieved_context, answer)** triples — include deliberately irrelevant "distractor" chunks in the context during training.

The model learns:

- Which retrieved chunks are actually relevant (noise filtering)
- How to produce the correct structured output from real RAG inputs

At inference: still uses RAG (dynamic knowledge). After fine-tuning: more robust to retrieval noise.

#### Quick Decision Rule

```
Knowledge changes often?         → RAG
Need source attribution?         → RAG
Custom output style/format?      → Fine-tune
Stable jargon model gets wrong?  → Fine-tune
Both problems at once?           → RAFT
```

---

## 17. Multi-Tenancy in Vector Databases

> ### 🏗️ In This Project
>
> - Currently **single-tenant**: all 8 policy docs share one Pinecone namespace
> - All queries have no tenant scoping — works for one warehouse
> - Scaling to multi-warehouse (multiple tenants with different policies) requires picking an isolation strategy

> ### 💡 Why It Matters
>
> Multi-tenancy in vector search has a hidden security risk: metadata filters can be bypassed or misconfigured. The isolation strategy directly determines what happens when a bug lets tenant A see tenant B's data.

### 🌐 Isolation Strategies

| Strategy                      | Isolation Level                             | Cost                            | Max Tenants |
| ----------------------------- | ------------------------------------------- | ------------------------------- | ----------- |
| **Separate index per tenant** | Maximum — total isolation                   | 💰💰💰 High (per-index billing) | ~20–50      |
| **Namespace per tenant**      | Strong — no cross-namespace leakage         | 💰 Low                          | Hundreds    |
| **Metadata filter**           | Weak — filter misconfiguration = data leak  | 💰 Lowest                       | Thousands   |
| **Hybrid tier namespaces**    | Medium — group by tier (gold/silver/bronze) | 💰 Low                          | Hundreds    |

#### Recommended for This System: Namespace Per Tenant

```python
# Query scoped to tenant
index.query(
    vector=query_embedding,
    top_k=3,
    namespace=f"warehouse_{tenant_id}",   # server-side isolation
    include_metadata=True
)
```

#### Security: Metadata Filter Injection

**Never** build the filter from user input:

```python
# ❌ DANGEROUS — user controls filter expression
filter = user_provided_filter

# ✅ SAFE — tenant_id comes from authenticated session, not user input
filter = {"tenant_id": {"$eq": session.tenant_id}}
```

An attacker who can set `filter={"tenant_id": {"$exists": true}}` sees all tenants' documents.

Namespace isolation is safer because the namespace is set server-side and never reaches the user-facing API surface.

---

## 18. Agentic Loop Termination and Infinite Loop Prevention

> ### 🏗️ In This Project
>
> **Four interlocking limits** — each catches a different failure pattern:
>
> | Limit               | Where                  | Value  | What it catches                          |
> | ------------------- | ---------------------- | ------ | ---------------------------------------- |
> | `MAX_CYCLES`        | `agents/supervisor.py` | 15     | Supervisor routing loop                  |
> | `MAX_AGENT_RETRIES` | `agents/supervisor.py` | 3      | Stuck agent being retried repeatedly     |
> | `max_iterations`    | `agents/react_loop.py` | 8 / 12 | Infinite reasoning loop within one agent |
> | `recursion_limit`   | `workflow/graph.py`    | 50     | Hard LangGraph framework cap             |
>
> - Any limit exceeded → `hitl_required=True` → supervisor routes to HITL node → parked in `HITLTask` DB table
> - `ReactMaxIterationsError` raised by ReAct loop → caught by agent node → converted to HITL escalation

> ### 💡 Why It Matters
>
> A single limit can be fooled. If only `MAX_CYCLES` existed, a supervisor rapidly bouncing between two agents could consume all 15 cycles without the per-agent retry counter catching it. Each limit targets a distinct failure mode — they're defense in depth, not redundancy.

### 🌐 Industry Best Practices

| Pattern                   | How to implement                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Goal-state detection**  | Check explicit terminal condition first (`validation_passed=False` → skip policy entirely). Don't burn cycles on already-decided cases.                |
| **Convergence detection** | If tool calls at iteration N == tool calls at N-1, agent is stuck in a loop — terminate immediately.                                                   |
| **Wall-clock deadline**   | Set absolute time budget (e.g., 30s per workflow). Use `asyncio.wait_for()` or an `asyncio.Event` timeout to cancel and escalate when budget exceeded. |
| **Idempotent re-entry**   | When HITL resumes a parked task, the graph reloads from the last checkpoint — partial work is never re-executed.                                       |

**Idempotent termination.** Ensure that HITL escalation itself is idempotent — if the workflow crashes after setting `hitl_required=True` but before checkpointing, the next recovery should detect the HITL signal in the last checkpoint and route correctly.

---

## 19. LangGraph vs CrewAI vs AutoGen — Trade-offs

> ### 🏗️ In This Project
>
> **File:** `workflow/graph.py`
>
> **Chose LangGraph** for three non-negotiable requirements:
>
> - **Durable checkpointing** — dual-layer Redis + PostgreSQL via `AsyncPostgresSaver`
> - **HITL mid-workflow** — `interrupt_before`/`interrupt_after` compile-time hooks
> - **Strict routing logic** — `conditional_edges` with arbitrary Python (not conversational heuristics)

> ### 💡 Why It Matters
>
> All three frameworks can build multi-agent systems. The differences emerge at production requirements: crash recovery, audit trails, and controlled routing. For a financial workflow (refund decisions), non-deterministic routing is unacceptable.

### 🌐 Framework Comparison

| Requirement            | **LangGraph**          | CrewAI               | AutoGen                          |
| ---------------------- | ---------------------- | -------------------- | -------------------------------- |
| Durable checkpointing  | ✅ Native (PG / Redis) | ❌                   | ❌                               |
| HITL mid-workflow      | ✅ Native              | Limited              | Via human proxy                  |
| Strict routing logic   | ✅ Full Python control | Limited              | Limited                          |
| Crash recovery         | ✅ Native              | Manual               | Manual                           |
| Structured JSON output | ✅ Pydantic            | Via tools            | Via tools                        |
| Boilerplate            | High                   | Low                  | Medium                           |
| Token cost growth      | Controlled             | Controlled           | Unbounded (conversation history) |
| Best for               | Production workflows   | Role-based pipelines | Collaborative dialogue / code    |

#### When NOT to Use LangGraph

- **Simple sequential pipelines** — CrewAI's low boilerplate wins
- **Code review, debate, critique** — AutoGen's conversation model is more natural
- **RAG-only Q&A** — LlamaIndex Workflows or a plain ReAct loop is simpler

---

## 20. ColBERT and Late Interaction Models

> ### 🏗️ In This Project
>
> - Uses **bi-encoder** (dense embedding) via Pinecone — standard approach
> - ColBERT is **not implemented** — but knowing when it would help is the interview answer
> - **Trigger for adding ColBERT:** RAGAS context recall < 0.75 on complex multi-part policy queries

> ### 💡 Why It Matters
>
> Bi-encoder and cross-encoder aren't the only options. ColBERT sits in between — token-level interaction without the O(N) inference cost. This is the answer to "how do you improve retrieval beyond simple cosine similarity without using a full reranker?"

### 🌐 The Retrieval Model Spectrum

| Model                           | Representation                         | Computation                            | Storage | Accuracy |
| ------------------------------- | -------------------------------------- | -------------------------------------- | ------- | -------- |
| **Bi-encoder** _(this project)_ | 1 vector per doc (pre-computed)        | ANN search                             | 1×      | Good     |
| **ColBERT**                     | M token vectors per doc (pre-computed) | MaxSim scan                            | 20×     | Better   |
| **Cross-encoder**               | No pre-computation                     | Full transformer per (query, doc) pair | 0×      | Best     |

#### How ColBERT Works

```
Query:    [q₁, q₂, ..., qM]  ← M token vectors (from query encoder)
Document: [d₁, d₂, ..., dL]  ← L token vectors (pre-computed at index time)

Score = Σ  max  (qᵢ · dⱼ)
        i   j
```

Each query token independently finds its best match in the document. "Gold tier" as a query has two tokens — each attends to the most relevant document token independently. A document discussing "gold membership" in one paragraph and "tier benefits" in another will score high even though a bi-encoder might not capture this split.

#### Trade-offs

|                      | Bi-encoder    | ColBERT v2        | Cross-encoder         |
| -------------------- | ------------- | ----------------- | --------------------- |
| Storage              | 1×            | 2–3× (compressed) | 0×                    |
| Query latency        | Fastest (ANN) | Fast (MaxSim)     | Slowest (O(N) passes) |
| Recall vs bi-encoder | baseline      | +3–7% (BEIR)      | +10–15%               |

**Implementation:** `RAGatouille` library — drop-in ColBERT integration with minimal changes.

---

## 21. Semantic Cache Design and Threshold Calibration

> ### 🏗️ In This Project
>
> **File:** `cache/redis_cache.py`
>
> | Property             | Value                                                                    |
> | -------------------- | ------------------------------------------------------------------------ |
> | Storage              | Redis hash: `policy_cache:{uuid}` → `{embedding: float[], result: dict}` |
> | Similarity threshold | **0.92** cosine (if ≥ 0.92 → cache hit)                                  |
> | TTL                  | 3600 seconds (1 hour)                                                    |
> | Lookup algorithm     | Linear scan over all `policy_cache:*` keys (O(N))                        |
> | Invalidation         | `invalidate_all()` — flushes all `policy_cache:*` keys                   |
>
> - Called immediately after any policy doc update to Pinecone
> - Linear O(N) scan is acceptable for 8 policy documents; needs upgrade at ~1000+ entries

> ### 💡 Why It Matters
>
> The 0.92 threshold is the most consequential design parameter. Too low → semantically different queries incorrectly share a cached result (wrong answers). Too high → valid cache hits are missed (wasted Pinecone calls + latency).

### 🌐 Threshold Calibration

#### Method: P95 of True-Positive Pairs

1. Collect query pairs `(query_A, query_B)` where both are known to have the same answer
2. Compute pairwise cosine similarities for all pairs
3. Plot similarity distributions: true-positive pairs vs true-negative pairs
4. **Set threshold = P95 of true-positive similarities**
   - 95% of same-answer query pairs will be within the threshold of each other
   - Most different-answer pairs fall below it
5. Validate on held-out test set: measure cache precision (correct hits / total hits) and recall

#### Scaling the Cache Beyond Linear Scan

| Scale                       | Solution                                                     |
| --------------------------- | ------------------------------------------------------------ |
| < 1,000 entries _(current)_ | Linear scan — fine                                           |
| 1,000–100,000               | FAISS flat index in-process, or **RedisVL** (HNSW in Redis)  |
| 100,000+                    | LSH (locality-sensitive hashing) for O(1) approximate lookup |

---

## 22. Cost Optimization in LLM Systems

> ### 🏗️ In This Project
>
> **Four cost levers implemented:**
>
> | Lever                     | Where                  | Impact                                                                                                |
> | ------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
> | Model tiering             | `config.py`            | Supervisor/Policy → `gpt-4o-mini`; Validation/Communication → `gpt-4.1-nano` (~10× cheaper per token) |
> | Semantic caching          | `cache/redis_cache.py` | Repeated policy queries hit cache → 0 Pinecone calls, 0 LLM input tokens for context                  |
> | Wave-based parallel tools | `agents/react_loop.py` | ~50% fewer ReAct iterations → ~50% fewer LLM calls                                                    |
> | Async I/O throughout      | FastAPI + asyncio      | No blocked threads — more concurrency per instance → lower horizontal scale cost                      |

> ### 💡 Why It Matters
>
> LLM cost is dominated by two things: total tokens and number of requests. Every technique above targets one or both.

### 🌐 Industry Cost Techniques

| Technique                          | How                                                                                       | Savings                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **LLMLingua / LLMLingua-2**        | Compress prompt by removing non-attended tokens                                           | 4–6× compression, <2% quality loss                                    |
| **KV cache prefix ordering**       | Put static system prompt + policy context first; dynamic content last                     | 50% discount on cached prompt tokens (OpenAI pricing)                 |
| **Batch inference API**            | Non-real-time tasks (RAGAS eval, golden dataset scoring)                                  | 50% cost reduction vs sync API                                        |
| **`max_tokens` enforcement**       | Cap every LLM call — prevents runaway responses                                           | Eliminates accidentally expensive calls                               |
| **Response streaming**             | `StreamingResponse` in FastAPI                                                            | No cost saving, but reduces timeout retries (which would double cost) |
| **Model fallback as cost control** | `ModelFallbackChain` can be triggered proactively on budget thresholds, not just failures | Direct substitution of cheaper model for simple tasks                 |

---

## 23. Debugging Multi-Agent Systems

> ### 🏗️ In This Project
>
> **Four observability layers, each filling a different gap:**
>
> | Layer                   | Where                 | What it gives you                                                            |
> | ----------------------- | --------------------- | ---------------------------------------------------------------------------- |
> | **LangSmith via OTEL**  | `telemetry/setup.py`  | Hierarchical trace tree — full execution timeline for one request            |
> | **Structlog JSON logs** | All modules           | `task_id`-keyed structured logs — aggregate across services by task          |
> | **Checkpoint replay**   | `checkpoint/store.py` | Re-run failed workflow from last good state locally — no Kafka needed        |
> | **AuditLog DB table**   | `database/models.py`  | Immutable record of every tool call: agent, tool, input, output, duration_ms |
>
> Span attributes logged: `task_id`, `agent_name`, `tool_called`, `ragas.faithfulness`, `ab_test.variant`, `guardrails.pii_masked`

> ### 💡 Why It Matters
>
> Multi-agent bugs are typically not in a single function — they're in the routing logic, state transitions, or the combination of multiple agents' outputs. You need full-trace observability, not just per-function logs.

### 🌐 Industry Best Practices

| Practice                         | How                                                                                                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **W3C TraceContext propagation** | Propagate `traceparent` header across all HTTP calls (FastAPI → Kafka → LangGraph → tool HTTP calls). Creates one trace tree across all services.                                 |
| **Span waterfall critical path** | In LangSmith / Jaeger waterfall view, identify the longest sequential span chain — that's the minimum achievable latency. Optimizing a non-critical-path span gives zero benefit. |
| **State diffing**                | Serialize `RefundState` to JSON before/after each node. Store diffs as span attributes. Shows exactly which fields each agent mutated.                                            |
| **Reproduction scripts**         | For every bug, create a minimal script: fixed seed state, mocked tool responses, no Kafka dependency. Enables fast local iteration.                                               |
| **LangFuse / Arize Phoenix**     | Open-source LangSmith alternatives. LangFuse adds prompt versioning + dataset management. Arize Phoenix specializes in embedding drift detection.                                 |

---

## 24. Knowledge Freshness in RAG

> ### 🏗️ In This Project
>
> | Component                   | Freshness behavior                                                    |
> | --------------------------- | --------------------------------------------------------------------- |
> | Semantic cache (Redis)      | TTL = 1h — stale policy answer served for up to 1 hour after update   |
> | Policy documents (Pinecone) | Hardcoded strings, initialized at startup — **no auto-reindex** (gap) |
> | Prompt registry             | Auto-refresh every 5 minutes from DB (`prompts/registry.py`)          |
>
> - Urgent fix: call `cache.invalidate_all()` immediately after any Pinecone upsert
> - Long-term gap: no CDC pipeline to re-index docs when the policy management DB changes

> ### 💡 Why It Matters
>
> A RAG system can have perfectly fresh vectors but still serve stale answers if the semantic cache isn't flushed. Both layers must be updated atomically.

### 🌐 Freshness Strategies

#### Timestamp Metadata + Recency Filter

Add `updated_at` to every document's Pinecone metadata. Hard filter at query time:

```python
filter={"updated_at": {"$gte": cutoff_timestamp}}
```

Old documents become invisible. Use when stale data is actively harmful (news, regulatory changes).

#### Recency Re-Ranking (Soft Signal)

$$\text{final\_score} = \text{cosine\_score} \times (1 - \lambda \times \text{age\_days})$$

Blends semantic relevance with freshness. $\lambda$ controls decay rate. Use when older docs may still be authoritative.

#### CDC Auto-Reindex Pipeline (Recommended)

```
Policy DB update
    → Kafka event (CDC)
    → Consumer re-embeds document
    → Pinecone upsert
    → cache.invalidate_all()
```

End-to-end latency: < 60 seconds. The correct production solution for this project.

#### Embedding Model Version Pinning

Pin the model version explicitly: `text-embedding-3-small-2024-09-01`. If OpenAI silently updates the model weights, all stored vectors become inconsistent with new query vectors — even though the text hasn't changed. Pinning prevents invisible drift.

---

## 25. Wave-Based Parallel Tool Execution in ReAct

> ### 🏗️ In This Project
>
> **File:** `agents/react_loop.py` — `_execution_waves()` function
>
> **The problem:** Standard ReAct calls tools one at a time. `lookup_customer` then `get_order_details` = 200ms sequential. But they're independent — neither depends on the other's output.
>
> **The solution:** Topological sort → execute independent tools in parallel per wave, then return to LLM.
>
> #### Validation Agent Example
>
> ```
> Wave 0 (parallel via asyncio.gather):
>   lookup_customer(email)        # no dependencies
>   get_order_details(order_id)   # no dependencies
>
> ← LLM turn: sees both results, fills in customer_id ←
>
> Wave 1 (parallel via asyncio.gather):
>   get_customer_analytics(customer_id)         # depends on: lookup_customer
>   verify_order_ownership(customer_id, order_id) # depends on: both
> ```
>
> - **Latency:** 4 sequential × ~100ms = 400ms → 2 waves × ~100ms = **200ms (50% reduction)**
> - **Critical design:** execute ONLY Wave 0 per LLM turn — even if LLM tries to call Wave 0 + Wave 1 together, the loop trims to Wave 0 only
> - This prevents hallucinated arguments: LLM cannot know `customer_id` until `lookup_customer` actually runs

> ### 💡 Why It Matters
>
> Without the trim-to-Wave-0 rule, a LLM that confidently hallucinates `customer_id=12345` for a Wave 1 tool would silently produce wrong results. The wave boundary forces the LLM to wait for real values before calling dependent tools.

### 🌐 Industry Analogs

| System                       | Same concept                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| **Apache Airflow / Prefect** | DAG task groups — tasks with no upstream dependencies run in parallel                             |
| **Apache Spark**             | Stage-based execution — stages separated by shuffle boundaries, tasks within a stage are parallel |
| **Plan-and-execute agents**  | Planning phase (LLM decides tool set) → parallel execution phase → results fed back               |
| **Temporal Workflows**       | Durable DAG execution with fault tolerance and retry — production-grade version of wave execution |

The difference between wave-based ReAct and standard plan-and-execute: wave-based does planning and execution interleaved per wave, one LLM turn per wave. Plan-and-execute does one planning phase then executes the full plan. Wave-based is more adaptive — the LLM can adjust the plan based on intermediate results.

---

## 26. ANN and HNSW — How Approximate Nearest Neighbor Search Works

> ### 🏗️ In This Project
>
> **File:** `vectordb/pinecone_store.py`
>
> - Pinecone uses HNSW internally for its vector index
> - This project stores 8 policy documents at 1536 dims with cosine similarity
> - `top_k=3` triggers an ANN search — not an exact exhaustive scan
> - At 8 documents the difference is invisible; at 10M documents ANN is the only viable option

> ### 💡 Why It Matters
>
> Exact nearest-neighbor search is O(N × D) — scanning every vector in the index. At 10M documents × 1536 dims, that's 15.36 billion multiply-adds per query. HNSW reduces this to O(log N) with ≥95% recall. The trade-off is that you might miss the single closest vector; in practice for RAG this rarely matters.

### 🌐 How ANN Works

#### The Core Problem

Given a query vector $q$ and a database of $N$ vectors, find the $k$ closest vectors.

- **Exact (brute-force):** Check all N. O(N·D). Accurate, slow at scale.
- **Approximate:** Accept ~1–5% recall loss in exchange for 100–1000× speedup.

#### HNSW (Hierarchical Navigable Small World)

HNSW builds a **multi-layer proximity graph** at index time. Think of it like a highway system:

```
Layer 2 (express, sparse):   A ─────────────── F
Layer 1 (arterial):          A ── C ── D ───── F
Layer 0 (local, dense):      A─B─C─D─E─F─G─H─I
```

**Index construction (per vector inserted):**

1. Assign the vector a maximum layer level (drawn from exponential distribution — most vectors are Layer 0 only)
2. For each layer from top down, greedily navigate the graph to find the `ef_construction` nearest neighbors already in the layer
3. Create bidirectional edges to the `M` closest neighbors found (M = connectivity parameter, typically 16–32)

**Query (search):**

1. Enter at the top layer, greedily navigate to the closest node to the query
2. Use that node as the entry point to the layer below
3. Repeat until Layer 0 — do a beam search with beam width `ef_search`
4. Return top-K results from the Layer 0 candidates

#### Key Parameters

| Parameter             | Default  | What it controls                                                               |
| --------------------- | -------- | ------------------------------------------------------------------------------ |
| `M`                   | 16       | Edges per node per layer — higher = better recall, more memory, slower build   |
| `ef_construction`     | 200      | Candidate pool size during build — higher = better index quality, slower build |
| `ef_search`           | 50       | Candidate pool during query — higher = better recall, slower query             |
| `ef_search` ≥ `top_k` | required | Cannot retrieve K results with a beam smaller than K                           |

#### Recall vs Latency Trade-off

```
ef_search=10   → P99 latency ~1ms,  recall ~0.90
ef_search=50   → P99 latency ~3ms,  recall ~0.97   ← typical default
ef_search=200  → P99 latency ~10ms, recall ~0.995
```

Pinecone manages these parameters internally and auto-tunes based on index size.

#### Why Cosine Similarity Works with HNSW

HNSW works with any distance metric. Pinecone normalizes all vectors to unit length at upsert time, so **cosine similarity = dot product** (since |a||b| = 1). This makes the MaxSim scan in HNSW identical to a dot product scan — highly optimized with SIMD instructions.

#### Other ANN Algorithms

| Algorithm          | Index type                           | Trade-off                           |
| ------------------ | ------------------------------------ | ----------------------------------- |
| **HNSW**           | Graph                                | Best recall/latency, high memory    |
| **IVF-PQ** (FAISS) | Inverted file + product quantization | Lower memory, slightly lower recall |
| **DiskANN**        | Graph on disk                        | Billion-scale on SSDs, low RAM      |
| **ScaNN** (Google) | Learned quantization                 | Excellent on Google hardware        |

---

## 27. P50, P95, P99 Latency Percentiles

> ### 🏗️ In This Project
>
> **File:** `telemetry/setup.py`
>
> - Every span emitted via OTEL has a `duration_ms` attribute
> - LangSmith aggregates spans → computes latency distributions per agent, per tool, per full workflow
> - `AuditLog` table in `database/models.py` stores `duration_ms` per tool call — enables custom percentile queries
>
> A typical end-to-end refund request spans:
>
> | Step                                     | Typical duration |
> | ---------------------------------------- | ---------------- |
> | Guardrails check                         | ~5ms             |
> | Validation agent (Wave 0 tools parallel) | ~100ms           |
> | Policy agent (Pinecone + LLM)            | ~300ms           |
> | Communication agent                      | ~150ms           |
> | Total P50                                | ~600ms           |

> ### 💡 Why It Matters
>
> **Averages lie.** If P50 = 500ms but P99 = 8s, the average is ~550ms — looks fine. But 1% of users wait 8 seconds. For a refund system, that 1% is real customers experiencing timeouts. SLAs must be expressed in percentiles, not averages.

### 🌐 Percentile Metrics Explained

#### What Each Percentile Means

Given 1000 requests sorted by latency:

| Metric           | Position    | What it tells you                                                 |
| ---------------- | ----------- | ----------------------------------------------------------------- |
| **P50** (median) | 500th value | "Half of requests are faster than this" — typical user experience |
| **P75**          | 750th value | Upper-normal experience                                           |
| **P95**          | 950th value | What the slowest 5% experience — tail latency                     |
| **P99**          | 990th value | What the slowest 1% experience — worst-case for most users        |
| **P99.9**        | 999th value | What 1-in-1000 requests experience — extreme outliers             |

#### The Bathtub Distribution of LLM Latency

```
Requests
   ▲
   │████
   │█████
   │██████
   │████████
   │██████████████
   │████████████████████████        ██
   └──────────────────────────────────────► latency
     fast         P50      P95  P99
```

LLM latency has a **heavy right tail** due to:

- Variable token generation length
- LLM API cold starts / throttling bursts
- Pinecone HNSW `ef_search` variability

#### How to Use Percentiles for SLAs

```
SLA: "95% of refund decisions complete within 2 seconds"
→ This is a P95 SLA of 2s

Alert: "Page on-call if P99 > 5s over any 5-minute window"
→ Use a histogram metric in Prometheus / Datadog
```

#### Measuring Percentiles Correctly

**Histograms over summaries.** Use metric histograms (Prometheus `histogram` type), not summaries. Histograms can be aggregated across instances; summaries cannot.

```python
# Prometheus histogram — bucket boundaries chosen around expected SLA
latency_histogram = Histogram(
    "refund_workflow_duration_seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)
```

**HDR Histogram** — use `hdrhistogram` library for in-process high-dynamic-range recording when sub-millisecond accuracy matters.

#### P95 in the Context of This System

- **Circuit breaker threshold:** If P95 tool latency exceeds 500ms, consider pre-emptive timeout + retry
- **Retry budget ratio:** A P99 latency spike causes more retries, which can exceed the 10% ratio cap in `RetryBudget` — the two systems interact
- **HITL escalation time:** Track P95 of time-parked-in-HITL to ensure human reviewers are meeting SLA

---

## 28. Citations and Source Attribution in RAG

> ### 🏗️ In This Project
>
> **Files:** `tools/policy_tools.py` · `agents/policy_agent.py`
>
> - `search_policies` returns `top_k=3` Pinecone results, each with metadata: `{"policy_type": "electronics", "source": "Electronics Return Policy v2.1"}`
> - The policy agent prompt instructs the LLM to reference the retrieved policy by name in its decision
> - The policy agent's Pydantic output (`PolicyDecision`) includes a `policy_references` list of source IDs
> - **Gap:** No inline citation markers (e.g., `[1]`, `[2]`) are attached to specific sentences in the decision rationale

> ### 💡 Why It Matters
>
> Without citations, a RAG answer is just as opaque as a fine-tuned model. For a refund system with legal/audit requirements, every decision must be traceable to a specific policy version. "Denied per Electronics Return Policy v2.1, Section 3.2" is auditable; "Denied because of our policy" is not.

### 🌐 Citation Implementation Patterns

#### Pattern 1 — Metadata Tags (This Project's Approach)

Return chunk metadata alongside results. Include `source_id` in the structured output schema:

```python
class PolicyDecision(BaseModel):
    decision: Literal["approved", "denied", "partial"]
    rationale: str
    policy_references: list[str]  # e.g., ["electronics_v2", "tier_gold_v1"]
    confidence: float
```

Simple, auditable, works well when chunks map 1:1 to documents.

#### Pattern 2 — Inline Citation Markers

Prompt the LLM to insert `[1]`, `[2]` markers mid-sentence referencing the retrieved chunks:

```
Prompt addition:
"When making claims, cite the source chunk using [N] notation
where N is the chunk index (1-indexed) from the retrieved context.
Example: 'Electronics returns are allowed within 30 days [1],
except for Gold tier which extends to 60 days [2].'"
```

Then parse the response to extract citation positions and map `[N]` → `chunk[N-1].metadata.source`.

#### Pattern 3 — Post-hoc Attribution (NLI-based)

After generating the response, run an NLI (Natural Language Inference) model over each sentence:

- For each sentence in the response, score `entailment(sentence, chunk_i)` for all retrieved chunks
- Attach the highest-scoring chunk as the citation for that sentence
- **Advantage:** Works even when the LLM doesn't insert markers
- **Tools:** `cross-encoder/nli-deberta-v3-small` from Hugging Face

#### Pattern 4 — Chunk ID Prefix in Context

Prepend a unique ID to each chunk in the context window:

```
[CHUNK-1] Electronics standard return: 30 days from purchase date...
[CHUNK-2] Gold tier extended return: 60 days with receipt...
[CHUNK-3] Fraud prevention: items over $500 require manager approval...
```

Prompt: "Always cite the CHUNK-ID when using information from that chunk."

Easiest for the LLM to follow consistently.

#### What to Store for Audit

| Field             | Why                                                                       |
| ----------------- | ------------------------------------------------------------------------- |
| `chunk_id`        | Exact vector DB record that was retrieved                                 |
| `document_source` | Human-readable policy name + version                                      |
| `chunk_text`      | Snapshot of the text at retrieval time (policies change)                  |
| `retrieval_score` | Cosine similarity — shows how confident the retrieval was                 |
| `timestamp`       | When the retrieval happened — cross-reference with policy version history |

---

## 29. Deciding Chunking Strategy and Token Count

> ### 🏗️ In This Project
>
> **File:** `vectordb/pinecone_store.py`
>
> - Currently: whole documents as single chunks (simplest strategy)
> - Embedding model: `text-embedding-3-small`, **max input: 8,192 tokens**
> - Policy documents are short enough that they fit in one chunk today
> - As corpus grows, this breaks down — need a decision framework

> ### 💡 Why It Matters
>
> Chunk size is the single biggest lever you have on retrieval quality after choosing the embedding model. Too large → embeddings are diluted (one vector tries to represent too many topics). Too small → chunks lack enough context for the LLM to use them. The right answer depends on 4 factors: document structure, query type, embedding model limits, and LLM context window.

### 🌐 The Decision Framework

#### Step 1 — What is Your Document Type?

| Document type                            | Recommended strategy                                                |
| ---------------------------------------- | ------------------------------------------------------------------- |
| Structured (markdown, HTML with headers) | **Document-aware** — split at heading boundaries                    |
| Legal / policy / contracts               | **Document-aware** + parent-child                                   |
| Narrative prose (articles, books)        | **Semantic chunking** — split at topic shifts                       |
| Code                                     | **AST-aware** — split at function/class level, never mid-function   |
| Tables / CSVs                            | **Row-level** — one row or group of rows per chunk                  |
| Mixed                                    | **Hierarchical** — detect type per section, apply per-type strategy |

#### Step 2 — What is Your Query Type?

| Query type                                           | Chunk preference                                      |
| ---------------------------------------------------- | ----------------------------------------------------- |
| Factual lookup ("what is the return window?")        | **Small chunks** (128–256 tokens) — precise embedding |
| Multi-fact synthesis ("compare Gold vs Silver tier") | **Large chunks** (512–1024 tokens) — context needed   |
| Both (most RAG systems)                              | **Parent-child** — retrieve small, return large       |

#### Step 3 — Token Size Boundaries

```
Embedding model max:  text-embedding-3-small = 8,192 tokens
                      (truncates silently beyond this — never exceed)

Practical sweet spot:
  Small child chunks:   128–256 tokens   (high retrieval precision)
  Standard chunks:      512 tokens       (good balance)
  Large parent chunks:  1,024 tokens     (full context for LLM)

LLM context window check:
  top_k=3 chunks × 512 tokens = 1,536 tokens for context
  gpt-4o-mini limit = 128K tokens → fine
  But: too much context → "lost in the middle" phenomenon
  Keep retrieved context under 2,000–3,000 tokens total
```

#### Step 4 — Overlap

```
overlap = 10–15% of chunk size

chunk_size=512, overlap=50  ← standard
chunk_size=256, overlap=30  ← small chunks

Purpose: preserve continuity at chunk boundaries
         A sentence split across two chunks can be
         retrieved from either
```

#### Step 5 — Validate with Retrieval Metrics

After choosing a strategy, measure on your golden dataset:

| Metric                    | Target |
| ------------------------- | ------ |
| P@1                       | ≥ 0.85 |
| Context Recall (RAGAS)    | ≥ 0.80 |
| Context Precision (RAGAS) | ≥ 0.75 |

If P@1 is low → chunks are too large (embedding dilution). If Context Recall is low → chunks are too small (relevant info split across chunks, only one retrieved).

#### Recommended Config for This Project

```python
# Policy corpus — document-aware + parent-child
CHILD_CHUNK_SIZE   = 256    # tokens — indexed in Pinecone, precise retrieval
CHILD_OVERLAP      = 30     # tokens
PARENT_CHUNK_SIZE  = 1024   # tokens — returned to LLM as context
SPLIT_AT           = "##"   # markdown H2 header boundary
```

---

## 30. How a Reranker Works Internally

> ### 🏗️ In This Project
>
> **File:** `tools/policy_tools.py`
>
> - `search_policies` retrieves `top_k=3` via Pinecone cosine similarity — **no reranker**
> - The bi-encoder does not compare query and document jointly — it embeds them independently
> - This is a known gap: a cross-encoder reranker on the top-10 Pinecone results would improve precision
> - Trigger: if RAGAS context precision drops below 0.75 in production, add a reranker

> ### 💡 Why It Matters
>
> Cosine similarity ranks by geometric proximity in embedding space. Two vectors can be close because they share general domain vocabulary — not because the document actually answers the query. A reranker performs deep query-document interaction and catches these false positives.

### 🌐 Bi-encoder vs Cross-encoder Architecture

#### Bi-encoder (How Pinecone Works)

```
Query ──► [Encoder] ──► q_vec (1536 dims)
                                            ──► cosine_sim(q_vec, d_vec)  → score
Doc   ──► [Encoder] ──► d_vec (1536 dims)
```

- Query and document encoded **independently** — no interaction between their tokens
- Document vectors **pre-computed** at index time → fast ANN search
- Weakness: no word-level alignment — "gold tier return" ↔ "tier benefits for gold members" may rank lower than expected

#### Cross-encoder (How a Reranker Works)

```
[CLS] query tokens [SEP] document tokens [SEP]
           │
      [Transformer]
      (full self-attention across ALL tokens)
           │
      [Linear head]
           │
        relevance score (scalar)
```

- Query and document tokens attend to **each other** at every transformer layer
- "gold" in the query can directly attend to "gold" in the document
- Cannot pre-compute — must run one full forward pass per `(query, doc)` pair
- 10–50× more accurate than bi-encoder on hard negatives

#### The Two-Stage Pipeline

```
Stage 1 — Recall (bi-encoder + ANN):
  Pinecone top_k=50  →  50 candidate chunks  (~5ms)

Stage 2 — Precision (cross-encoder reranker):
  cross_encoder.predict([(query, chunk_i) for chunk_i in candidates])
  → re-sort by cross-encoder score
  → return top_k=3                           (~50–150ms)
```

The key insight: **Stage 1 is cheap and handles recall (don't miss relevant docs). Stage 2 is expensive but only runs on 50 candidates — it handles precision (put the right doc at rank 1).**

#### Reranker Models

| Model                                  | Size   | Latency        | Notes                               |
| -------------------------------------- | ------ | -------------- | ----------------------------------- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 66M    | ~20ms/50 docs  | Fast, good general-purpose reranker |
| `cross-encoder/ms-marco-electra-base`  | 110M   | ~40ms/50 docs  | Slightly higher accuracy            |
| `BAAI/bge-reranker-v2-m3`              | 570M   | ~100ms/50 docs | Strong multilingual reranker        |
| Cohere Rerank API                      | hosted | ~100ms         | Managed, no self-hosting            |
| Jina Reranker v2                       | 137M   | ~50ms/50 docs  | Open-source, long context (8K)      |

#### What "Score" the Cross-encoder Produces

The `[CLS]` token's representation after all transformer layers is passed through a linear layer to produce a single scalar. This scalar is trained with:

```
loss = binary_cross_entropy(score, is_relevant_label)
```

on MS-MARCO or similar (query, document, relevance) triples. Higher score = model believes the document answers the query.

#### Temperature Scaling on Reranker Scores

Cross-encoder scores are not calibrated probabilities — they're logits. Before using them as weights in a hybrid system, apply temperature scaling:

```python
calibrated = [score / temperature for score in raw_scores]
probs = softmax(calibrated)
```

This is important if you weight reranker scores against BM25 scores in a fusion step.

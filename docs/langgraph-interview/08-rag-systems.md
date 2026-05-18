# Section 8: RAG Systems

> **Project:** `vectordb/pinecone_store.py`, `tools/policy_tools.py`, `cache/redis_cache.py`, `evaluation/ragas_evaluator.py`

---

## Q26. Walk me through building a production RAG pipeline

### Our policy RAG path

1. **Ingest** — warehouse refund policies indexed in Pinecone (`build_policy_index()` on startup)
2. **Embed** — `text-embedding-3-small` (1536-dim), cosine similarity
3. **Retrieve** — `search_policies` tool, `top_k=3`
4. **Cache** — Redis semantic cache (similar queries skip re-embed + retrieval)
5. **Generate** — policy ReAct agent reasons over chunks + eligibility tools
6. **Evaluate** — `RefundRAGASEvaluator` (RAGAS metrics)

### Chunking (general + our docs)

```python
RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "],
)
```

Policy docs are relatively small — 8 indexed documents in demo; production scales with metadata (`category`, `effective_date`).

### Retrieval tuning

- Same embedding model for index and query
- Metadata filters for tenant/category when multi-tenant
- Re-rank top 20 → 5 if quality insufficient

---

## Q27. How do you handle stale embeddings when source documents update?

1. **Delete-by-doc_id** — remove all Pinecone vectors with `doc_id` metadata
2. **Re-chunk + re-embed** — upsert new vectors with new `version_hash`
3. **Change detection** — content hash in registry; webhook on policy update
4. **TTL** — optional `expires_at` on chunks; filter at query time

**Incident pattern:** Answers coherent but wrong policy version → fix hash-based detection (not file mtime alone).

---

## Q28. Compare Pinecone vs Qdrant vs FAISS

| | Pinecone | Qdrant | FAISS |
|--|----------|--------|-------|
| Ops | Managed SaaS | Self-host / cloud | In-process library |
| Filters | Metadata | Rich payload | Manual |
| Our project | **Yes** — `PineconeStore` | Alternative for GDPR self-host | Prototype only |

**Interview:** We use Pinecone for speed of delivery; Qdrant if data residency requires self-host; FAISS for offline eval benchmarks.

---

## Q29. How do you evaluate RAG quality?

### Metrics

| Axis | Metric |
|------|--------|
| Retrieval | Recall@K, MRR |
| Generation | RAGAS faithfulness, answer relevancy, context recall |
| E2E | Golden set correctness, P95 latency |
| Production | Sampled human eval, thumbs down on wrong policy citations |

```python
# evaluation/ragas_evaluator.py
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
```

**Our hook:** Evaluation endpoints + periodic runs on policy change PRs.

---

## Q30. How would you implement hybrid search?

**Dense (Pinecone)** + **sparse (BM25)** → **RRF fusion**

```python
def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**When it helps refund domain:** Exact policy IDs, SKU codes, order numbers — BM25; conceptual "damaged in transit" — vectors.

**Qdrant:** native hybrid with dense + sparse vectors in one collection.

**Next step for our project:** Add BM25 over policy text in Postgres + ensemble with Pinecone scores for production hardening.

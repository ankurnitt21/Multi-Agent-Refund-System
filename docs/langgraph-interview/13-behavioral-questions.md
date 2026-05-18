# Section 13: Behavioral Questions

> Frame answers with **Warehouse Refund System** examples where possible.

---

## Q42. Tell me about a time you built something 0→1 under ambiguity

### Framework: Situation → Ambiguity → Resolution → Outcome

**Situation:** Build an automated warehouse refund pipeline — stakeholders unsure how much to trust LLM vs hard rules.

**Ambiguity:** Routing model (pure `if` vs LLM supervisor), checkpoint strategy, when to escalate to humans.

**Resolution:**
- Shipped **week 1:** deterministic sentinels (`validation_passed`, `decision`, `request_saved`) + simple graph
- **Week 2:** LLM supervisor with routing tools, but **hard caps** (cycles, retries) LLM cannot override
- **Week 3:** dual checkpoints, Kafka async path, HITL API
- Made chunking, prompts, and policy index **config-driven** (`PromptRegistry`, Pinecone rebuild)

**Outcome:** Working async refund API with observability and human escalation — iterated routing and RAG without rewriting the graph topology.

**Lesson:** Optimize for **changeability** — hub-and-spoke supervisor, external policy store, optional fields in state.

---

## Q43. How do you balance speed vs correctness in early-stage infra?

### Non-negotiable (correctness)

- **Money paths:** idempotent DB writes, `order_id` uniqueness
- **Security:** prompt injection scan, PII masking in logs
- **Audit:** `AuditLog` for decisions

### Accept speed debt

- In-memory graph fallback on Windows dev
- Semantic cache tuning later
- Full RBAC deferred with clear extension points

### Techniques

- Feature flags for new prompt versions (`PromptRegistry`)
- Test **contracts** (API + state sentinels), not every tool implementation detail
- **Loud failures** → HITL, not silent wrong approvals
- Idempotent consumers so we can ship Kafka before exactly-once perfection

**Honest line:** *"We shipped HITL early so we could move fast on LLM routing without betting the business on 100% automation."*

---

## Q44. Describe a production incident you owned

### Template applied to RAG/policy staleness

**What broke:** Policy agent cited outdated return window — answers plausible, wrong vs current policy PDF.

**Detection:** User complaint + manual review; RAGAS faithfulness dip on weekly eval (if wired).

**Root cause:** Policy index rebuilt only on deploy; source docs updated without redeploy trigger; compared wrong version metadata.

**Immediate fix:** Force `build_policy_index()` + invalidate Redis semantic cache keys.

**Long-term fix:**
- Content-hash change detection on ingest
- Alert if index `updated_at` lag > 24h
- Golden-set regression in CI on policy PRs

**Process change:** "RAG freshness" on weekly ops dashboard; ingestion failure pages on-call.

### Alternative: Kafka duplicate processing

**Symptom:** Duplicate `ProcessedRequest` attempts after consumer crash.  
**Fix:** Strengthened idempotency key path + treat UNIQUE violation as success in communication tools.  
**Long-term:** Document at-least-once semantics in README; metric for idempotent skips.

---

## Tips for the interview

- Lead with **Section 5** three-minute workflow story
- Cite **real files** (`workflow/graph.py`, `agents/supervisor.py`)
- Connect every LangGraph concept to **RefundState** or **supervisor**
- Admit tradeoffs (Windows checkpointer skip, `operator.add` vs `add_messages`)

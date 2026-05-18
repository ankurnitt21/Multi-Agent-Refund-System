# Section 11: MCP & Tool Integration

> **Project:** LangChain tools in `tools/` — function calling today; MCP as evolution path

---

## Q39. What is MCP and how does it differ from function calling?

| | Function calling | MCP |
|--|------------------|-----|
| Definition | Schema in LLM API request | Open standard (JSON-RPC) |
| Execution | In-process in your app | MCP server process |
| Discovery | Static in prompt | Dynamic tool list |
| Reuse | Per application | Cross-client |

### Our project today

**Function calling** via LangChain `@tool` decorators:

- `tools/db_tools.py` — customer, order, ownership
- `tools/policy_tools.py` — Pinecone search, eligibility
- `tools/analytics_tools.py` — aggregates

Bound to agents with `llm.bind_tools(VALIDATION_TOOLS)`.

### MCP migration path

Expose `db_tools` as MCP server over stdio/HTTP → same refund agent, tools maintained once for Claude Desktop + our FastAPI service.

**Interview line:** *"We use function calling in-process for latency; MCP when tools must be shared across teams or sandboxed separately."*

---

## Q40. Design a secure, sandboxed tool execution environment

```
Agent → Tool Gateway (policy + rate limit) → Sandbox → sanitized result → Agent
```

### Controls

| Control | Implementation |
|---------|----------------|
| Allowlist | Only registered tools per agent |
| Timeout | 30s on DB/API calls |
| Network | Egress allowlist for external APIs |
| Credentials | Scoped tokens from vault — agent never sees raw keys |
| Resources | cgroup/容器 CPU memory limits |
| Audit | Log tool name, redacted params, result hash |

**Our step toward this:** Tools are Python functions with typed schemas; production hardening = subprocess/container per tool class + gateway service.

---

## Q41. Auth (OAuth, API keys) for 3rd-party SaaS tools

### Credential vault pattern

```
User OAuth → encrypt in KMS → store per (user_id, integration)
Agent requests action → Gateway fetches token → calls API → returns result only
```

### Principles

1. **Per-user OAuth** for Slack/Drive — not one global key
2. **Scope minimization** — read-only Gmail for lookup tools
3. **Token never in LLM context**
4. **Auto-refresh** before expiry
5. **Revocation** deletes vault row immediately

**Refund system:** DB credentials in env/config — no per-user SaaS yet; same vault pattern applies when adding shipping-carrier APIs.

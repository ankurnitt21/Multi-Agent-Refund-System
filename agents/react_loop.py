"""
Parallel ReAct loop engine with dependency-aware tool scheduling.

How it works
------------
Each LLM response may contain N tool calls.  Instead of executing them one at
a time, we run a *topological wave sort* over those calls using a pre-defined
dependency map (dict[tool_name → list[tool_name_it_depends_on]]).

Wave 0 — tools with no unmet dependencies  → run in parallel via asyncio.gather
Wave 1 — tools whose deps were in wave 0   → run in parallel
...and so on, until all tool calls in this LLM turn are executed.

This gives maximum parallelism while preserving correctness for tools that need
an earlier tool's output passed as a parameter (e.g. calculate_refund requires
the output of check_eligibility as its `eligibility_result` argument — the LLM
is responsible for reading the ToolMessage and forwarding the value).
"""
import asyncio
import json
import time

import structlog
from langchain_core.messages import AIMessage, ToolMessage

from executor.resilience import get_fallback_chain, CircuitBreakerOpen, RetryBudgetExhausted

logger = structlog.get_logger(__name__)


class ReactMaxIterationsError(Exception):
    """Raised when the ReAct loop exhausts its iteration budget.

    Caught by agent nodes to trigger HITL instead of silently returning
    a potentially incomplete answer.
    """


def _execution_waves(
    tool_calls: list[dict],
    deps: dict[str, list[str]],
) -> list[list[dict]]:
    """
    Topological wave sort.

    Returns a list of *waves*; tools in the same wave are independent and can
    be executed in parallel.  Tools in wave N+1 have at least one dependency
    that appears in wave N or earlier.

    Only inter-wave ordering is enforced.  Tools whose dependencies are NOT in
    the current call batch are treated as having no pending dependency.
    """
    called_names = {tc["name"] for tc in tool_calls}
    remaining = list(tool_calls)
    completed: set[str] = set()
    waves: list[list[dict]] = []

    while remaining:
        # A tool is "ready" when every dependency that is also being called
        # this turn has already been placed in an earlier wave.
        ready = [
            tc for tc in remaining
            if all(
                dep not in called_names or dep in completed
                for dep in deps.get(tc["name"], [])
            )
        ]

        if not ready:
            # Circular or unresolvable deps — run everything remaining together
            # to avoid an infinite loop.
            ready = list(remaining)

        waves.append(ready)
        completed.update(tc["name"] for tc in ready)
        ready_ids = {id(tc) for tc in ready}
        remaining = [tc for tc in remaining if id(tc) not in ready_ids]

    return waves


async def _call_tool(tool_map: dict, tc: dict) -> str:
    """Execute one tool call; always returns a JSON string."""
    tool = tool_map.get(tc["name"])
    try:
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tc['name']}"})
        raw = await tool.ainvoke(tc["args"])
        return json.dumps(raw) if not isinstance(raw, str) else raw
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def react_loop(
    llm_with_tools,
    tool_map: dict[str, object],
    messages: list,
    deps: dict[str, list[str]],
    *,
    span=None,
    max_iterations: int = 8,
    use_fallback: bool = True,
) -> str:
    """
    Run a ReAct loop with parallel tool execution.

    Parameters
    ----------
    llm_with_tools : LLM already bound to tools via .bind_tools(...)
    tool_map       : {tool_name: tool} lookup
    messages       : mutable list — SystemMessage + HumanMessage prepopulated;
                     ToolMessages and AIMessages are appended in-place.
    deps           : dependency map — {tool_name: [tools_it_depends_on]}
    span           : optional OpenTelemetry span for attribute logging
    max_iterations : safety cap on LLM turns
    use_fallback   : if True, uses ModelFallbackChain on LLM errors

    Returns
    -------
    str — content of the final AIMessage (no-tool-call response)
    """
    fallback_chain = get_fallback_chain() if use_fallback else None
    tools_list = list(tool_map.values()) if tool_map else []

    for iteration in range(max_iterations):
        if span is not None:
            span.set_attribute("react.iteration", iteration + 1)

        try:
            response = await llm_with_tools.ainvoke(messages)
        except (CircuitBreakerOpen, RetryBudgetExhausted):
            raise
        except Exception as e:
            if fallback_chain and tools_list:
                logger.warning("react_primary_failed_using_fallback", error=str(e), iteration=iteration + 1)
                response = await fallback_chain.invoke(messages, tools=tools_list)
            else:
                raise

        messages.append(response)

        # No tool calls → LLM is done reasoning, return final answer
        if not response.tool_calls:
            return response.content

        # -----------------------------------------------------------------
        # Execute tool calls in parallel waves respecting the dep map.
        #
        # IMPORTANT: Only execute the FIRST wave per LLM iteration.
        # If the LLM batched multiple waves in a single response (e.g.
        # called both lookup_customer and verify_order_ownership before
        # lookup results were available), the dependent-tool args would
        # be hallucinated.  We instead:
        #   1. Trim the appended AIMessage to only wave-0 calls.
        #   2. Execute wave-0 in parallel.
        #   3. Let the next LLM iteration generate fresh, correct args
        #      for the remaining (dependent) waves.
        # -----------------------------------------------------------------
        waves = _execution_waves(response.tool_calls, deps)
        first_wave = waves[0]
        wave_log = [[tc["name"] for tc in w] for w in waves]
        if span is not None:
            span.set_attribute(f"react.iteration.{iteration+1}.waves", str(wave_log))

        # If the LLM smuggled dependent-tool calls into this response,
        # replace the last message with a trimmed version that only
        # contains wave-0 calls (prevents dangling tool_call_id errors).
        if len(first_wave) < len(response.tool_calls):
            messages[-1] = AIMessage(
                content=response.content,
                tool_calls=first_wave,
            )

        # Execute wave-0 in parallel
        wave_tools = [tc["name"] for tc in first_wave]
        t0 = time.perf_counter()
        logger.debug("tool_wave_start", iteration=iteration + 1, tools=wave_tools)

        results = await asyncio.gather(
            *[_call_tool(tool_map, tc) for tc in first_wave]
        )

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.debug("tool_wave_complete", iteration=iteration + 1, tools=wave_tools, duration_ms=duration_ms)
        for tc, result in zip(first_wave, results):
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

    # Safety: max iterations hit — raise so the caller can route to HITL
    raise ReactMaxIterationsError(
        f"ReAct loop exhausted {max_iterations} iterations without a final answer"
    )

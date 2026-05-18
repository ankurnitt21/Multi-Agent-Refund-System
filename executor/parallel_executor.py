import asyncio
import json
import time

from opentelemetry import trace, context
from langchain_core.tools import BaseTool

from telemetry.setup import tracer

TOOL_DEPENDENCY_MAP = {
    "lookup_customer": [],
    "get_order_details": [],
    "get_customer_analytics": ["lookup_customer"],
    "verify_order_ownership": ["lookup_customer"],
    "search_policies": [],
    "check_refund_history": [],
    "check_eligibility": [],
    "get_product_return_policy": [],
    "calculate_refund": ["check_eligibility"],
    "save_processed_request": [],
    "update_customer_analytics": [],
    "update_refund_request_status": [],
    "log_audit_event": [],
}


class DependencyAwareExecutor:
    def __init__(self):
        self.dependency_map = TOOL_DEPENDENCY_MAP

    async def execute(
        self,
        tool_calls: list[dict],
        tool_map: dict[str, BaseTool],
        parent_ctx=None,
    ) -> dict:
        if parent_ctx is None:
            parent_ctx = context.get_current()

        pending = {tc["name"]: tc for tc in tool_calls}
        results = {}

        while pending:
            ready = []
            for name, tc in pending.items():
                deps = self.dependency_map.get(name, [])
                if all(d in results for d in deps):
                    ready.append((name, tc))

            if not ready:
                # Prevent infinite loop if dependencies can't be resolved
                break

            tasks = [
                self._run_single_tool(tc, tool_map, parent_ctx, results)
                for _, tc in ready
            ]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for (name, _), result in zip(ready, task_results):
                if isinstance(result, Exception):
                    results[name] = json.dumps({"error": str(result)})
                else:
                    results[name] = result
                pending.pop(name, None)

        return results

    async def _run_single_tool(
        self,
        tool_call: dict,
        tool_map: dict[str, BaseTool],
        parent_ctx,
        existing_results: dict,
    ) -> str:
        token = context.attach(parent_ctx)
        try:
            with tracer.start_as_current_span(
                f"tool.{tool_call['name']}",
                context=parent_ctx,
            ) as span:
                span.set_attribute("langsmith.span.kind", "tool")
                span.set_attribute("tool.name", tool_call["name"])
                span.set_attribute("tool.input", json.dumps(tool_call.get("args", {})))

                start = time.perf_counter()
                tool = tool_map.get(tool_call["name"])
                if not tool:
                    result = json.dumps({"error": f"Tool {tool_call['name']} not found"})
                else:
                    args = tool_call.get("args", {})
                    result = await tool.ainvoke(args)

                duration_ms = (time.perf_counter() - start) * 1000
                span.set_attribute("tool.output", str(result)[:4096])
                span.set_attribute("tool.duration_ms", duration_ms)
                return result
        finally:
            context.detach(token)

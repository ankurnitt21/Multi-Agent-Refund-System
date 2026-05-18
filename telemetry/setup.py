import os
from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from config import (
    OTEL_EXPORTER_OTLP_ENDPOINT,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGCHAIN_PROJECT,
)


def setup_telemetry():
    """Initialize OpenTelemetry with LangSmith OTEL endpoint.

    This sends custom spans (guardrails, RAGAS eval, tool execution) to LangSmith.
    LangChain's native tracing (LANGCHAIN_TRACING_V2=true) handles LLM call tracing
    automatically — both streams appear in the same LangSmith project.
    """
    # Ensure LangChain env vars are set for native tracing
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)
    if LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)

    provider = TracerProvider()

    exporter = OTLPSpanExporter(
        endpoint=f"{OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces",
        headers={
            "x-api-key": LANGSMITH_API_KEY or "",
            "langsmith-project": LANGSMITH_PROJECT,
        },
    )

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    return trace.get_tracer("refund_system")


tracer = setup_telemetry()

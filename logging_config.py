"""
Structured logging setup using structlog.

Industry standard for Python services — used by Stripe, HashiCorp, Sentry.

Two output modes controlled by the LOG_FORMAT env var:
  LOG_FORMAT=text  (default)  →  colored, human-readable console
  LOG_FORMAT=json             →  JSON lines, compatible with Datadog / Splunk / ELK / CloudWatch

Context binding (flows automatically into ALL log calls in the same async task):
    import structlog
    structlog.contextvars.bind_contextvars(task_id="f47ac10b", order_id=7)
    # Every subsequent log.info(...) in this task includes task_id and order_id
    structlog.contextvars.clear_contextvars()   # call before each new task
"""
import logging
import os
import sys

import structlog


def setup_logging() -> None:
    json_logs = os.getenv("LOG_FORMAT", "text").lower() == "json"

    # ── Processors applied to EVERY log record ───────────────────────
    # These run on both structlog-native calls AND stdlib logging calls
    # (captured via ProcessorFormatter's foreign_pre_chain).
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,      # inject task_id, order_id, etc.
        structlog.stdlib.add_log_level,               # level="info"
        structlog.stdlib.add_logger_name,             # logger="agents.supervisor"
        structlog.processors.TimeStamper(fmt="iso"),  # timestamp="2026-05-15T10:23:41Z"
        structlog.processors.StackInfoRenderer(),     # stack_info if present
        structlog.processors.format_exc_info,         # exc_info → traceback string
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True, sort_keys=False)
    )

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ── Attach to stdlib root logger so third-party libs are captured ─
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # ── Silence noisy third-party loggers ────────────────────────────
    for name in (
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "httpx",
        "httpcore",
        "opentelemetry",
        "urllib3",
        "asyncio",
        "langchain",
        "langsmith",
        "sentence_transformers",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

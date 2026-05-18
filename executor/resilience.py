"""
Resilience layer: Circuit Breaker + Retry Budget + Model Fallback.

Industry-standard patterns for production LLM systems:
- Circuit Breaker: Prevents cascading failures by short-circuiting after N consecutive errors
- Retry Budget: Limits total retries per time window to avoid thundering herd
- Model Fallback: Automatically routes to backup models when primary is unavailable

All state is tracked with OTEL spans for LangSmith visibility.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import structlog
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from config import GROQ_API_KEY, GROQ_MODEL
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation — requests flow through
    OPEN = "open"            # Tripped — all requests rejected immediately
    HALF_OPEN = "half_open"  # Testing — one request allowed through


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""
    failure_threshold: int = 5       # Consecutive failures before OPEN
    recovery_timeout: float = 30.0   # Seconds before OPEN → HALF_OPEN
    success_threshold: int = 2       # Successes in HALF_OPEN before CLOSED
    excluded_exceptions: tuple = ()  # Exceptions that don't trip the breaker


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is OPEN and request is rejected."""

    def __init__(self, name: str, time_until_retry: float):
        self.name = name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """
    Thread-safe circuit breaker for LLM provider calls.

    States:
    - CLOSED: All calls pass through normally
    - OPEN: All calls rejected immediately (fail-fast)
    - HALF_OPEN: One test call allowed; success → CLOSED, failure → OPEN
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    async def __aenter__(self):
        await self._before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self._on_success()
        elif exc_type and not issubclass(exc_type, self.config.excluded_exceptions):
            await self._on_failure(exc_val)
        return False  # Don't suppress the exception

    async def _before_call(self):
        async with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info("circuit_breaker_half_open", name=self.name)
                else:
                    self._total_rejections += 1
                    raise CircuitBreakerOpen(
                        self.name,
                        self.config.recovery_timeout - elapsed,
                    )

    async def _on_success(self):
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("circuit_breaker_closed", name=self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self, error: Exception):
        async with self._lock:
            self._total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("circuit_breaker_reopened", name=self.name, error=str(error))
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "circuit_breaker_opened",
                        name=self.name,
                        failures=self._failure_count,
                        error=str(error),
                    )

    def get_metrics(self) -> dict:
        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
            "failure_count": self._failure_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Retry Budget
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetryBudgetConfig:
    """Configuration for the retry budget."""
    max_retries_per_window: int = 20   # Max retries in the time window
    window_seconds: float = 60.0       # Time window for budget tracking
    retry_ratio: float = 0.1           # Max ratio of retries to total requests
    base_delay: float = 1.0            # Base delay for exponential backoff
    max_delay: float = 30.0            # Max delay between retries
    jitter: bool = True                # Add randomized jitter to delays


class RetryBudgetExhausted(Exception):
    """Raised when retry budget is exhausted."""
    pass


class RetryBudget:
    """
    Token-bucket style retry budget.

    Limits total retry attempts within a rolling time window to prevent
    thundering herd problems during outages.
    """

    def __init__(self, config: RetryBudgetConfig | None = None):
        self.config = config or RetryBudgetConfig()
        self._requests: list[float] = []  # timestamps of all requests
        self._retries: list[float] = []   # timestamps of retries only
        self._lock = asyncio.Lock()

    def _prune(self, now: float):
        """Remove entries outside the rolling window."""
        cutoff = now - self.config.window_seconds
        self._requests = [t for t in self._requests if t > cutoff]
        self._retries = [t for t in self._retries if t > cutoff]

    async def acquire(self) -> bool:
        """Check if a retry is allowed. Returns True if budget permits."""
        async with self._lock:
            now = time.monotonic()
            self._prune(now)

            # Check absolute limit
            if len(self._retries) >= self.config.max_retries_per_window:
                return False

            # Check ratio limit
            if self._requests and len(self._retries) / len(self._requests) >= self.config.retry_ratio:
                return False

            self._retries.append(now)
            return True

    async def record_request(self):
        """Record a request (successful or not) for ratio calculation."""
        async with self._lock:
            self._requests.append(time.monotonic())

    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        import random
        delay = min(
            self.config.base_delay * (2 ** attempt),
            self.config.max_delay,
        )
        if self.config.jitter:
            delay *= (0.5 + random.random())
        return delay

    def get_metrics(self) -> dict:
        now = time.monotonic()
        cutoff = now - self.config.window_seconds
        active_retries = sum(1 for t in self._retries if t > cutoff)
        active_requests = sum(1 for t in self._requests if t > cutoff)
        return {
            "retries_in_window": active_retries,
            "requests_in_window": active_requests,
            "max_retries": self.config.max_retries_per_window,
            "budget_remaining": self.config.max_retries_per_window - active_retries,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Model Fallback Chain
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelConfig:
    """Configuration for a single model in the fallback chain."""
    name: str
    model_id: str
    api_key: str
    provider: str = "groq"  # groq | openai | anthropic
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 30.0
    priority: int = 0       # Lower = higher priority (tried first)


class ModelFallbackChain:
    """
    Model fallback with automatic failover.

    Tries models in priority order. Each model has its own circuit breaker.
    If all models fail, raises the last exception.

    LangSmith traces show which model was used via span attributes.
    """

    def __init__(self, models: list[ModelConfig] | None = None):
        if models is None:
            models = self._default_models()

        self._models = sorted(models, key=lambda m: m.priority)
        self._breakers: dict[str, CircuitBreaker] = {
            m.name: CircuitBreaker(
                f"model:{m.name}",
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0),
            )
            for m in self._models
        }
        self._retry_budget = RetryBudget()
        self._llm_cache: dict[str, BaseChatModel] = {}

    @staticmethod
    def _default_models() -> list[ModelConfig]:
        """Default fallback chain: primary → fast fallback → large fallback."""
        return [
            ModelConfig(
                name="primary",
                model_id=GROQ_MODEL,
                api_key=GROQ_API_KEY or "",
                provider="groq",
                priority=0,
            ),
            ModelConfig(
                name="fallback_fast",
                model_id="llama-3.1-8b-instant",
                api_key=GROQ_API_KEY or "",
                provider="groq",
                priority=1,
            ),
            ModelConfig(
                name="fallback_large",
                model_id="llama-3.3-70b-versatile",
                api_key=GROQ_API_KEY or "",
                provider="groq",
                priority=2,
            ),
        ]

    def _get_llm(self, config: ModelConfig) -> BaseChatModel:
        """Get or create LLM instance for a model config."""
        if config.name not in self._llm_cache:
            if config.provider == "groq":
                self._llm_cache[config.name] = ChatGroq(
                    model=config.model_id,
                    api_key=config.api_key,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
            else:
                # Extensible for other providers
                self._llm_cache[config.name] = ChatGroq(
                    model=config.model_id,
                    api_key=config.api_key,
                    temperature=config.temperature,
                )
        return self._llm_cache[config.name]

    async def invoke(
        self,
        messages: list,
        tools: list | None = None,
        **kwargs,
    ) -> Any:
        """
        Invoke LLM with automatic fallback.

        Tries each model in priority order. Uses circuit breakers to skip
        known-bad models. Respects retry budget.
        """
        with tracer.start_as_current_span("model_fallback.invoke") as span:
            span.set_attribute("langsmith.span.kind", "llm")
            span.set_attribute("fallback.models_available", len(self._models))

            await self._retry_budget.record_request()
            last_error: Exception | None = None

            for i, model_config in enumerate(self._models):
                breaker = self._breakers[model_config.name]

                # Skip if circuit is open
                if breaker.state == CircuitState.OPEN:
                    elapsed = time.monotonic() - breaker._last_failure_time
                    if elapsed < breaker.config.recovery_timeout:
                        logger.debug(
                            "model_skipped_circuit_open",
                            model=model_config.name,
                        )
                        continue

                try:
                    async with breaker:
                        llm = self._get_llm(model_config)
                        if tools:
                            llm = llm.bind_tools(tools)

                        response = await llm.ainvoke(messages, **kwargs)

                        span.set_attribute("fallback.model_used", model_config.name)
                        span.set_attribute("fallback.model_id", model_config.model_id)
                        span.set_attribute("fallback.attempt", i + 1)

                        if i > 0:
                            logger.info(
                                "model_fallback_used",
                                model=model_config.name,
                                attempt=i + 1,
                            )

                        return response

                except CircuitBreakerOpen:
                    continue
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "model_call_failed",
                        model=model_config.name,
                        error=str(e),
                        attempt=i + 1,
                    )
                    span.set_attribute(f"fallback.error.{model_config.name}", str(e))

                    # Check retry budget before trying next model
                    if i < len(self._models) - 1:
                        can_retry = await self._retry_budget.acquire()
                        if not can_retry:
                            span.set_attribute("fallback.budget_exhausted", True)
                            raise RetryBudgetExhausted(
                                "Retry budget exhausted — cannot try fallback models"
                            ) from e
                    continue

            # All models failed
            span.set_attribute("fallback.all_failed", True)
            raise last_error or RuntimeError("All models in fallback chain failed")

    async def invoke_with_retry(
        self,
        messages: list,
        tools: list | None = None,
        max_attempts: int = 3,
        **kwargs,
    ) -> Any:
        """Invoke with retry + fallback combined."""
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                return await self.invoke(messages, tools, **kwargs)
            except (RetryBudgetExhausted, CircuitBreakerOpen):
                raise
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    can_retry = await self._retry_budget.acquire()
                    if not can_retry:
                        raise RetryBudgetExhausted("Retry budget exhausted") from e
                    delay = self._retry_budget.get_delay(attempt)
                    logger.info(
                        "retry_after_delay",
                        attempt=attempt + 1,
                        delay_s=round(delay, 2),
                    )
                    await asyncio.sleep(delay)

        raise last_error or RuntimeError("All retry attempts failed")

    def get_metrics(self) -> dict:
        """Get health metrics for all models and the retry budget."""
        return {
            "models": {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            },
            "retry_budget": self._retry_budget.get_metrics(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Global instance (singleton for the application)
# ═══════════════════════════════════════════════════════════════════════════════

_fallback_chain: ModelFallbackChain | None = None


def get_fallback_chain() -> ModelFallbackChain:
    """Get or create the global model fallback chain."""
    global _fallback_chain
    if _fallback_chain is None:
        _fallback_chain = ModelFallbackChain()
    return _fallback_chain

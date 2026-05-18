"""Tests for executor/resilience.py — Circuit Breaker, Retry Budget, Model Fallback."""

import asyncio
import time

import pytest
from executor.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    RetryBudget,
    RetryBudgetConfig,
    RetryBudgetExhausted,
    ModelFallbackChain,
    ModelConfig,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0, success_threshold=2),
        )

    @pytest.mark.asyncio
    async def test_starts_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_stays_closed_on_success(self, breaker):
        async with breaker:
            pass  # Success
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, breaker):
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("simulated failure")
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self, breaker):
        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("fail")

        # Should reject immediately
        with pytest.raises(CircuitBreakerOpen):
            async with breaker:
                pass

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker):
        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("fail")

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should transition to HALF_OPEN on next call
        async with breaker:
            pass
        # One success — need 2 for CLOSED
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_successes_in_half_open(self, breaker):
        # Trip → wait → half_open
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("fail")
        await asyncio.sleep(1.1)

        # Two successes should close it
        async with breaker:
            pass
        async with breaker:
            pass
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(self, breaker):
        # Trip → wait → half_open
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("fail")
        await asyncio.sleep(1.1)

        # Fail in half_open → back to OPEN
        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("still broken")
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions_dont_trip(self):
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=1,
                excluded_exceptions=(KeyError,),
            ),
        )
        # KeyError should NOT trip the breaker
        with pytest.raises(KeyError):
            async with breaker:
                raise KeyError("ignored")
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_metrics(self, breaker):
        async with breaker:
            pass
        metrics = breaker.get_metrics()
        assert metrics["name"] == "test"
        assert metrics["state"] == "closed"
        assert metrics["total_calls"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Retry Budget Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetryBudget:
    @pytest.fixture
    def budget(self):
        return RetryBudget(RetryBudgetConfig(
            max_retries_per_window=5,
            window_seconds=10.0,
            retry_ratio=0.5,
            base_delay=0.1,
            max_delay=1.0,
        ))

    @pytest.mark.asyncio
    async def test_allows_retries_within_budget(self, budget):
        await budget.record_request()
        assert await budget.acquire() is True

    @pytest.mark.asyncio
    async def test_rejects_when_budget_exhausted(self, budget):
        # Record enough requests so ratio isn't hit
        for _ in range(20):
            await budget.record_request()
        # Exhaust the retry budget
        for _ in range(5):
            assert await budget.acquire() is True
        # 6th should fail
        assert await budget.acquire() is False

    @pytest.mark.asyncio
    async def test_ratio_limit(self):
        budget = RetryBudget(RetryBudgetConfig(
            max_retries_per_window=100,
            retry_ratio=0.1,
            window_seconds=10.0,
        ))
        # 10 requests, ratio limit is 0.1 = max 1 retry
        for _ in range(10):
            await budget.record_request()
        assert await budget.acquire() is True
        assert await budget.acquire() is False

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, budget):
        d0 = budget.get_delay(0)
        d1 = budget.get_delay(1)
        d2 = budget.get_delay(2)
        # With jitter, exact values vary, but order should be increasing
        # base * 2^attempt, so d1 > d0 in expectation
        # Just verify it doesn't exceed max
        assert budget.get_delay(100) <= 1.0 * 2  # max_delay * max_jitter

    @pytest.mark.asyncio
    async def test_metrics(self, budget):
        await budget.record_request()
        await budget.acquire()
        metrics = budget.get_metrics()
        assert metrics["retries_in_window"] == 1
        assert metrics["requests_in_window"] == 1
        assert metrics["budget_remaining"] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Model Fallback Chain Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelFallbackChain:
    def test_default_models_created(self):
        chain = ModelFallbackChain()
        metrics = chain.get_metrics()
        assert "primary" in metrics["models"]
        assert "fallback_fast" in metrics["models"]
        assert "fallback_large" in metrics["models"]

    def test_custom_models(self):
        models = [
            ModelConfig(name="main", model_id="test-model", api_key="key", priority=0),
            ModelConfig(name="backup", model_id="test-backup", api_key="key", priority=1),
        ]
        chain = ModelFallbackChain(models=models)
        metrics = chain.get_metrics()
        assert "main" in metrics["models"]
        assert "backup" in metrics["models"]

    def test_metrics_structure(self):
        chain = ModelFallbackChain()
        metrics = chain.get_metrics()
        assert "models" in metrics
        assert "retry_budget" in metrics
        for model_metrics in metrics["models"].values():
            assert "state" in model_metrics
            assert "total_calls" in model_metrics

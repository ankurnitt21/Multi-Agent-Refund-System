"""
RAGAS-based evaluator for the warehouse refund system.

Uses the ragas Python library to evaluate agent outputs across multiple
dimensions. All evaluation metrics run in parallel using asyncio.gather()
for maximum throughput.

Evaluation dimensions:
- Faithfulness: Agent stays true to tool outputs / context
- Answer Relevancy: Response addresses the refund request properly
- Context Precision: Retrieved policies are precisely relevant
- Context Recall: All necessary context was retrieved

Results are exported to LangSmith for dashboard visibility.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from telemetry.setup import tracer

logger = structlog.get_logger(__name__)

try:
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # v0.4+ moved metrics to ragas.metrics.collections
    try:
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("ragas_not_installed", message="Install ragas: pip install ragas")


@dataclass
class EvaluationResult:
    """Result of a single evaluation metric."""
    metric_name: str
    score: float
    duration_ms: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Aggregated evaluation report for a refund workflow run."""
    task_id: str
    results: list[EvaluationResult] = field(default_factory=list)
    overall_score: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "overall_score": self.overall_score,
            "duration_ms": self.duration_ms,
            "metrics": {r.metric_name: r.score for r in self.results},
            "errors": {r.metric_name: r.error for r in self.results if r.error},
        }


class RefundRAGASEvaluator:
    """
    Evaluates refund workflow outputs using RAGAS metrics in parallel.

    Integrates with LangSmith by:
    1. Creating OpenTelemetry spans for each evaluation
    2. Logging metric scores as span attributes
    3. Using LangSmith-compatible LLM wrappers for evaluation
    """

    def __init__(self, llm=None, embeddings=None):
        """
        Args:
            llm: LangChain LLM for evaluation (uses ragas default if None)
            embeddings: LangChain embeddings for similarity metrics
        """
        self._llm = llm
        self._embeddings = embeddings

    async def evaluate_workflow_run(
        self,
        task_id: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Run all RAGAS evaluation metrics in parallel for a workflow run.

        Args:
            task_id: Unique identifier for the refund task
            question: The original refund request/question
            answer: The agent's final response/decision
            contexts: Retrieved context (policies, customer data, etc.)
            ground_truth: Expected answer for recall computation (optional)

        Returns:
            EvaluationReport with all metric scores
        """
        with tracer.start_as_current_span("ragas.evaluate_workflow") as span:
            span.set_attribute("langsmith.span.kind", "chain")
            span.set_attribute("ragas.task_id", task_id)
            span.set_attribute("ragas.question_length", len(question))
            span.set_attribute("ragas.answer_length", len(answer))
            span.set_attribute("ragas.num_contexts", len(contexts))

            t0 = time.perf_counter()
            report = EvaluationReport(task_id=task_id)

            if not RAGAS_AVAILABLE:
                logger.warning("ragas_evaluation_skipped", reason="ragas not installed")
                span.set_attribute("ragas.skipped", True)
                return report

            # Run all metrics in parallel
            results = await run_parallel_evaluation(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            report.results = results
            report.duration_ms = (time.perf_counter() - t0) * 1000

            # Compute overall score (mean of valid results)
            valid_scores = [r.score for r in results if r.error is None]
            if valid_scores:
                report.overall_score = sum(valid_scores) / len(valid_scores)

            # Record to LangSmith span
            span.set_attribute("ragas.overall_score", report.overall_score)
            span.set_attribute("ragas.duration_ms", report.duration_ms)
            for r in results:
                span.set_attribute(f"ragas.{r.metric_name}", r.score)
                if r.error:
                    span.set_attribute(f"ragas.{r.metric_name}.error", r.error)

            logger.info(
                "ragas_evaluation_complete",
                task_id=task_id,
                overall_score=report.overall_score,
                metrics={r.metric_name: r.score for r in results},
                duration_ms=report.duration_ms,
            )

            return report

    async def evaluate_agent_output(
        self,
        agent_name: str,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        """
        Quick single-metric evaluation for an individual agent step.
        Uses faithfulness as the primary metric for agent outputs.
        """
        with tracer.start_as_current_span(f"ragas.evaluate_{agent_name}") as span:
            span.set_attribute("langsmith.span.kind", "chain")
            span.set_attribute("ragas.agent_name", agent_name)

            if not RAGAS_AVAILABLE:
                return EvaluationResult(
                    metric_name="faithfulness",
                    score=0.0,
                    duration_ms=0.0,
                    error="ragas not installed",
                )

            t0 = time.perf_counter()
            try:
                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                )
                dataset = EvaluationDataset(samples=[sample])

                eval_kwargs = {"dataset": dataset, "metrics": [faithfulness]}
                if self._llm:
                    eval_kwargs["llm"] = LangchainLLMWrapper(self._llm)
                if self._embeddings:
                    eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(self._embeddings)

                result = evaluate(**eval_kwargs)
                score = result["faithfulness"] if "faithfulness" in result else 0.0

                duration_ms = (time.perf_counter() - t0) * 1000
                span.set_attribute("ragas.faithfulness", score)
                span.set_attribute("ragas.duration_ms", duration_ms)

                return EvaluationResult(
                    metric_name="faithfulness",
                    score=float(score),
                    duration_ms=duration_ms,
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                logger.error("ragas_agent_eval_failed", agent=agent_name, error=str(e))
                return EvaluationResult(
                    metric_name="faithfulness",
                    score=0.0,
                    duration_ms=duration_ms,
                    error=str(e),
                )


async def _evaluate_metric(
    metric,
    metric_name: str,
    sample: "SingleTurnSample",
    llm=None,
    embeddings=None,
) -> EvaluationResult:
    """Evaluate a single RAGAS metric — designed to run in parallel."""
    t0 = time.perf_counter()
    try:
        dataset = EvaluationDataset(samples=[sample])

        eval_kwargs = {"dataset": dataset, "metrics": [metric]}
        if llm:
            eval_kwargs["llm"] = LangchainLLMWrapper(llm)
        if embeddings:
            eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(embeddings)

        result = evaluate(**eval_kwargs)
        score = result[metric_name] if metric_name in result else 0.0

        duration_ms = (time.perf_counter() - t0) * 1000
        return EvaluationResult(
            metric_name=metric_name,
            score=float(score),
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.error("ragas_metric_failed", metric=metric_name, error=str(e))
        return EvaluationResult(
            metric_name=metric_name,
            score=0.0,
            duration_ms=duration_ms,
            error=str(e),
        )


async def run_parallel_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    llm=None,
    embeddings=None,
) -> list[EvaluationResult]:
    """
    Run all RAGAS evaluation metrics in parallel using asyncio.gather().

    This is the core parallel execution engine. Each metric is evaluated
    independently and concurrently for maximum throughput.

    Metrics evaluated:
    - faithfulness: Does the answer stay true to the contexts?
    - answer_relevancy: Is the answer relevant to the question?
    - context_precision: Are the contexts precisely relevant?
    - context_recall: Do contexts contain all needed information? (requires ground_truth)
    """
    if not RAGAS_AVAILABLE:
        return []

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth or answer,
    )

    # Define metrics to run in parallel
    metrics_to_run = [
        (faithfulness, "faithfulness"),
        (answer_relevancy, "answer_relevancy"),
        (context_precision, "context_precision"),
    ]

    # context_recall requires ground_truth
    if ground_truth:
        metrics_to_run.append((context_recall, "context_recall"))

    # Execute ALL metrics in parallel
    tasks = [
        _evaluate_metric(metric, name, sample, llm, embeddings)
        for metric, name in metrics_to_run
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions from gather
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            metric_name = metrics_to_run[i][1]
            final_results.append(EvaluationResult(
                metric_name=metric_name,
                score=0.0,
                duration_ms=0.0,
                error=str(result),
            ))
        else:
            final_results.append(result)

    return final_results

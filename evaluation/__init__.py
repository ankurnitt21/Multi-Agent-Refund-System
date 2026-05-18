"""
RAGAS Evaluation Module — runs evaluation metrics in parallel using the ragas library.

Uses ragas (Retrieval Augmented Generation Assessment) for evaluating:
- Faithfulness: Does the agent output stay faithful to the context?
- Answer Relevancy: Is the response relevant to the question?
- Context Precision: Is the retrieved context precise?
- Context Recall: Does the context contain all needed info?

Also includes A/B testing for prompt optimization experiments.

Results are traced to LangSmith via OpenTelemetry spans for observability.
"""

from evaluation.ragas_evaluator import (
    RefundRAGASEvaluator,
    run_parallel_evaluation,
    EvaluationResult,
)
from evaluation.ab_testing import (
    ABTestManager,
    get_ab_manager,
    ExperimentConfig,
    Variant,
)

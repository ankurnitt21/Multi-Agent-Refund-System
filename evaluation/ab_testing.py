"""
A/B Testing for Prompts — experiment-driven prompt optimization.

Enables controlled experiments where different prompt versions are served
to different requests based on configurable traffic splits. Results are
tracked and compared using statistical significance testing.

Integration:
- LangSmith: Each experiment run is tagged with variant info in OTEL spans
- RAGAS: Evaluation metrics are collected per variant for comparison
- Prompt Registry: Variants are stored as prompt versions in the DB

Usage:
    from evaluation.ab_testing import ABTestManager, get_ab_manager

    manager = get_ab_manager()
    variant = await manager.get_variant("policy_agent", request_id="req-123")
    # Use variant.content as the prompt
    await manager.record_result("policy_agent", variant.name, score=0.85, request_id="req-123")
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog
from telemetry.setup import tracer

logger = structlog.get_logger(__name__)


@dataclass
class Variant:
    """A single variant in an A/B test."""
    name: str
    content: str
    traffic_percentage: float  # 0.0 to 1.0
    description: str = ""
    is_control: bool = False


@dataclass
class ExperimentResult:
    """Result of a single evaluation for one variant."""
    variant_name: str
    score: float
    request_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for a prompt A/B experiment."""
    experiment_id: str
    prompt_name: str
    variants: list[Variant]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None          # None = runs indefinitely
    min_samples_per_variant: int = 30          # For statistical significance
    confidence_level: float = 0.95             # 95% confidence for winner
    is_active: bool = True

    def __post_init__(self):
        total = sum(v.traffic_percentage for v in self.variants)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Variant traffic must sum to 1.0, got {total}")


@dataclass
class ExperimentReport:
    """Statistical summary of an A/B experiment."""
    experiment_id: str
    prompt_name: str
    variants: dict[str, dict]  # variant_name → {mean, std, count, ci_lower, ci_upper}
    winner: Optional[str] = None
    is_significant: bool = False
    p_value: Optional[float] = None
    duration_hours: float = 0.0

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "prompt_name": self.prompt_name,
            "variants": self.variants,
            "winner": self.winner,
            "is_significant": self.is_significant,
            "p_value": self.p_value,
            "duration_hours": self.duration_hours,
        }


class ABTestManager:
    """
    Manages A/B testing experiments for prompt variants.

    Features:
    - Deterministic assignment: same request_id always gets same variant
    - Traffic splitting: configurable percentage per variant
    - Statistical analysis: t-test for significance
    - LangSmith integration: variant info in OTEL spans
    """

    def __init__(self):
        self._experiments: dict[str, ExperimentConfig] = {}
        self._results: dict[str, list[ExperimentResult]] = {}
        self._lock = asyncio.Lock()

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Register a new A/B experiment."""
        self._experiments[config.experiment_id] = config
        self._results[config.experiment_id] = []
        logger.info(
            "ab_experiment_created",
            experiment_id=config.experiment_id,
            prompt=config.prompt_name,
            variants=[v.name for v in config.variants],
        )
        return config.experiment_id

    async def get_variant(
        self,
        prompt_name: str,
        request_id: str,
    ) -> Optional[Variant]:
        """
        Get the assigned variant for a request.

        Uses consistent hashing on request_id for deterministic assignment
        (same request always gets same variant regardless of retries).
        """
        with tracer.start_as_current_span("ab_test.get_variant") as span:
            span.set_attribute("langsmith.span.kind", "chain")
            span.set_attribute("ab_test.prompt_name", prompt_name)

            # Find active experiment for this prompt
            experiment = self._get_active_experiment(prompt_name)
            if experiment is None:
                return None

            # Deterministic assignment via consistent hashing
            hash_input = f"{experiment.experiment_id}:{request_id}"
            hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
            bucket = (hash_val % 10000) / 10000.0  # 0.0 to 1.0

            cumulative = 0.0
            selected = experiment.variants[-1]  # Default to last
            for variant in experiment.variants:
                cumulative += variant.traffic_percentage
                if bucket < cumulative:
                    selected = variant
                    break

            span.set_attribute("ab_test.experiment_id", experiment.experiment_id)
            span.set_attribute("ab_test.variant", selected.name)
            span.set_attribute("ab_test.is_control", selected.is_control)

            logger.debug(
                "ab_variant_assigned",
                experiment=experiment.experiment_id,
                variant=selected.name,
                request_id=request_id,
            )
            return selected

    async def record_result(
        self,
        prompt_name: str,
        variant_name: str,
        score: float,
        request_id: str,
        metadata: dict | None = None,
    ):
        """Record an evaluation result for a variant."""
        experiment = self._get_active_experiment(prompt_name)
        if experiment is None:
            return

        result = ExperimentResult(
            variant_name=variant_name,
            score=score,
            request_id=request_id,
            metadata=metadata or {},
        )

        async with self._lock:
            self._results.setdefault(experiment.experiment_id, []).append(result)

        with tracer.start_as_current_span("ab_test.record_result") as span:
            span.set_attribute("ab_test.experiment_id", experiment.experiment_id)
            span.set_attribute("ab_test.variant", variant_name)
            span.set_attribute("ab_test.score", score)

    def get_report(self, experiment_id: str) -> ExperimentReport:
        """Generate statistical report for an experiment."""
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        results = self._results.get(experiment_id, [])
        variant_data: dict[str, list[float]] = {}
        for r in results:
            variant_data.setdefault(r.variant_name, []).append(r.score)

        # Compute stats per variant
        variant_stats = {}
        for name, scores in variant_data.items():
            n = len(scores)
            mean = sum(scores) / n if n > 0 else 0.0
            variance = sum((s - mean) ** 2 for s in scores) / n if n > 1 else 0.0
            std = variance ** 0.5
            # 95% CI approximation (z=1.96 for large samples)
            ci_margin = 1.96 * std / (n ** 0.5) if n > 0 else 0.0
            variant_stats[name] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "count": n,
                "ci_lower": round(mean - ci_margin, 4),
                "ci_upper": round(mean + ci_margin, 4),
            }

        # Determine winner via two-sample t-test
        winner = None
        is_significant = False
        p_value = None

        variant_names = list(variant_data.keys())
        if len(variant_names) == 2:
            a_scores = variant_data[variant_names[0]]
            b_scores = variant_data[variant_names[1]]
            if len(a_scores) >= experiment.min_samples_per_variant and \
               len(b_scores) >= experiment.min_samples_per_variant:
                p_value = self._welch_ttest(a_scores, b_scores)
                is_significant = p_value < (1 - experiment.confidence_level)
                if is_significant:
                    a_mean = sum(a_scores) / len(a_scores)
                    b_mean = sum(b_scores) / len(b_scores)
                    winner = variant_names[0] if a_mean > b_mean else variant_names[1]

        duration = (time.time() - experiment.start_time) / 3600.0

        return ExperimentReport(
            experiment_id=experiment_id,
            prompt_name=experiment.prompt_name,
            variants=variant_stats,
            winner=winner,
            is_significant=is_significant,
            p_value=p_value,
            duration_hours=round(duration, 2),
        )

    def list_experiments(self) -> list[dict]:
        """List all experiments with basic status."""
        return [
            {
                "experiment_id": exp.experiment_id,
                "prompt_name": exp.prompt_name,
                "is_active": exp.is_active,
                "variants": [v.name for v in exp.variants],
                "total_results": len(self._results.get(exp.experiment_id, [])),
            }
            for exp in self._experiments.values()
        ]

    def stop_experiment(self, experiment_id: str):
        """Stop an experiment and mark it inactive."""
        if experiment_id in self._experiments:
            self._experiments[experiment_id].is_active = False
            self._experiments[experiment_id].end_time = time.time()
            logger.info("ab_experiment_stopped", experiment_id=experiment_id)

    def _get_active_experiment(self, prompt_name: str) -> Optional[ExperimentConfig]:
        """Find the active experiment for a prompt."""
        for exp in self._experiments.values():
            if exp.prompt_name == prompt_name and exp.is_active:
                if exp.end_time and time.time() > exp.end_time:
                    exp.is_active = False
                    continue
                return exp
        return None

    @staticmethod
    def _welch_ttest(a: list[float], b: list[float]) -> float:
        """Welch's t-test (unequal variance) — returns p-value approximation."""
        import math

        n_a, n_b = len(a), len(b)
        mean_a = sum(a) / n_a
        mean_b = sum(b) / n_b
        var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

        se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-10
        t_stat = abs(mean_a - mean_b) / se

        # Degrees of freedom (Welch-Satterthwaite)
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else 1

        # Approximate p-value using normal distribution for large df
        # For production, use scipy.stats.t.sf — this is a good approximation
        p_value = math.exp(-0.717 * t_stat - 0.416 * t_stat ** 2)
        return min(p_value, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Global instance
# ═══════════════════════════════════════════════════════════════════════════════

_ab_manager: ABTestManager | None = None


def get_ab_manager() -> ABTestManager:
    """Get or create the global A/B test manager."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
    return _ab_manager

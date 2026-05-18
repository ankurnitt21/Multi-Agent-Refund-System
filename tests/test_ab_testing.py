"""Tests for evaluation/ab_testing.py — A/B testing for prompts."""

import pytest
from evaluation.ab_testing import (
    ABTestManager,
    ExperimentConfig,
    Variant,
    ExperimentReport,
)


@pytest.fixture
def manager():
    return ABTestManager()


@pytest.fixture
def sample_experiment():
    return ExperimentConfig(
        experiment_id="exp-001",
        prompt_name="policy_agent",
        variants=[
            Variant(
                name="control",
                content="You are a policy agent. Evaluate refunds strictly.",
                traffic_percentage=0.5,
                is_control=True,
            ),
            Variant(
                name="variant_a",
                content="You are a policy agent. Evaluate refunds with customer satisfaction focus.",
                traffic_percentage=0.5,
            ),
        ],
        min_samples_per_variant=5,  # Lower for testing
    )


class TestABTestManager:
    def test_create_experiment(self, manager, sample_experiment):
        exp_id = manager.create_experiment(sample_experiment)
        assert exp_id == "exp-001"

    def test_list_experiments(self, manager, sample_experiment):
        manager.create_experiment(sample_experiment)
        experiments = manager.list_experiments()
        assert len(experiments) == 1
        assert experiments[0]["experiment_id"] == "exp-001"
        assert experiments[0]["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_variant_deterministic(self, manager, sample_experiment):
        manager.create_experiment(sample_experiment)

        # Same request_id should always get same variant
        v1 = await manager.get_variant("policy_agent", request_id="req-123")
        v2 = await manager.get_variant("policy_agent", request_id="req-123")
        assert v1.name == v2.name

    @pytest.mark.asyncio
    async def test_get_variant_distributes_traffic(self, manager, sample_experiment):
        manager.create_experiment(sample_experiment)

        # Generate many assignments — should see both variants
        variants_seen = set()
        for i in range(100):
            v = await manager.get_variant("policy_agent", request_id=f"req-{i}")
            variants_seen.add(v.name)

        assert "control" in variants_seen
        assert "variant_a" in variants_seen

    @pytest.mark.asyncio
    async def test_get_variant_no_experiment(self, manager):
        result = await manager.get_variant("nonexistent_prompt", request_id="req-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_record_result(self, manager, sample_experiment):
        manager.create_experiment(sample_experiment)
        await manager.record_result(
            "policy_agent", "control", score=0.85, request_id="req-1"
        )
        report = manager.get_report("exp-001")
        assert report.variants["control"]["count"] == 1
        assert report.variants["control"]["mean"] == 0.85

    @pytest.mark.asyncio
    async def test_report_statistics(self, manager, sample_experiment):
        manager.create_experiment(sample_experiment)

        # Record enough results for significance
        for i in range(10):
            await manager.record_result(
                "policy_agent", "control", score=0.7 + (i * 0.01), request_id=f"c-{i}"
            )
            await manager.record_result(
                "policy_agent", "variant_a", score=0.9 + (i * 0.01), request_id=f"v-{i}"
            )

        report = manager.get_report("exp-001")
        assert report.variants["control"]["count"] == 10
        assert report.variants["variant_a"]["count"] == 10
        assert report.variants["variant_a"]["mean"] > report.variants["control"]["mean"]

    def test_stop_experiment(self, manager, sample_experiment):
        manager.create_experiment(sample_experiment)
        manager.stop_experiment("exp-001")
        experiments = manager.list_experiments()
        assert experiments[0]["is_active"] is False

    @pytest.mark.asyncio
    async def test_stopped_experiment_returns_none(self, manager, sample_experiment):
        manager.create_experiment(sample_experiment)
        manager.stop_experiment("exp-001")
        result = await manager.get_variant("policy_agent", request_id="req-1")
        assert result is None

    def test_invalid_traffic_sum(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ExperimentConfig(
                experiment_id="bad",
                prompt_name="test",
                variants=[
                    Variant(name="a", content="x", traffic_percentage=0.3),
                    Variant(name="b", content="y", traffic_percentage=0.3),
                ],
            )

    def test_report_not_found(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.get_report("nonexistent")

    @pytest.mark.asyncio
    async def test_significance_detection(self, manager):
        """Test that clearly different distributions are detected as significant."""
        config = ExperimentConfig(
            experiment_id="exp-sig",
            prompt_name="test_prompt",
            variants=[
                Variant(name="bad", content="x", traffic_percentage=0.5, is_control=True),
                Variant(name="good", content="y", traffic_percentage=0.5),
            ],
            min_samples_per_variant=5,
        )
        manager.create_experiment(config)

        # Bad variant: scores around 0.3
        for i in range(10):
            await manager.record_result("test_prompt", "bad", score=0.3, request_id=f"b-{i}")
        # Good variant: scores around 0.9
        for i in range(10):
            await manager.record_result("test_prompt", "good", score=0.9, request_id=f"g-{i}")

        report = manager.get_report("exp-sig")
        assert report.is_significant is True
        assert report.winner == "good"

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for core metrics functionality."""

import time
from unittest.mock import MagicMock, patch

import pytest

from forge.observability.metrics import (
    ConsoleBackend,
    get_logger_backend_class,
    MaxAccumulator,
    MeanAccumulator,
    Metric,
    MetricCollector,
    MinAccumulator,
    record_metric,
    Reduce,
    reduce_metrics_states,
    StdAccumulator,
    SumAccumulator,
    WandbBackend,
)


class TestMetricCreation:
    """Test Metric object creation and record_metric function."""

    def test_metric_creation_automatic_timestamp(self, mock_rank):
        """Test Metric object creation with automatic timestamp."""
        before_time = time.time()
        metric = Metric("test_key", 42.0, Reduce.MEAN)
        after_time = time.time()

        assert metric.key == "test_key"
        assert metric.value == 42.0
        assert metric.reduction == Reduce.MEAN
        assert metric.timestamp is not None
        assert before_time <= metric.timestamp <= after_time

    def test_metric_creation_custom_timestamp(self, mock_rank):
        """Test Metric object creation with custom timestamp."""
        custom_time = 1234567890.0
        metric = Metric("test_key2", 24.0, Reduce.SUM, timestamp=custom_time)
        assert metric.timestamp == custom_time

    def test_record_metric(self, mock_rank):
        """Test record_metric creates correct Metric and calls collector."""
        # Mock the MetricCollector constructor to return a mock instance
        mock_collector = MagicMock()

        with patch(
            "forge.observability.metrics.MetricCollector", return_value=mock_collector
        ):
            record_metric("loss", 1.5, Reduce.MEAN)

            # Verify push was called on the mock collector
            mock_collector.push.assert_called_once()

            # Verify the metric passed to push
            pushed_metric = mock_collector.push.call_args[0][0]
            assert pushed_metric.key == "loss"
            assert pushed_metric.value == 1.5
            assert pushed_metric.reduction == Reduce.MEAN

    @patch.dict("os.environ", {"FORGE_DISABLE_METRICS": "true"})
    @patch("forge.observability.metrics.MetricCollector")
    def test_record_metric_disabled(self, mock_collector_class):
        """Test record_metric is no-op when FORGE_DISABLE_METRICS=true."""
        record_metric("loss", 1.5, Reduce.MEAN)
        mock_collector_class.assert_not_called()

    @patch.dict("os.environ", {"FORGE_DISABLE_METRICS": "false"})
    @patch("forge.observability.metrics.MetricCollector")
    def test_record_metric_enabled_explicit(self, mock_collector_class, mock_rank):
        """Test record_metric works when FORGE_DISABLE_METRICS=false."""
        mock_collector = MagicMock()
        mock_collector_class.return_value = mock_collector

        record_metric("loss", 1.5, Reduce.MEAN)
        mock_collector_class.assert_called_once()
        mock_collector.push.assert_called_once()


class TestAccumulators:
    """Test all accumulator classes and their operations."""

    def test_mean_accumulator(self):
        """Test MeanAccumulator operations."""
        acc = MeanAccumulator(Reduce.MEAN)

        # Test initial state
        assert acc.get_value() == 0.0
        state = acc.get_state()
        assert state["sum"] == 0.0
        assert state["count"] == 0

        # Test append and get_value
        acc.append(10.0)
        acc.append(20.0)
        assert acc.get_value() == 15.0

        # Test state
        state = acc.get_state()
        assert state["sum"] == 30.0
        assert state["count"] == 2
        assert state["reduction_type"] == "mean"

        # Test reset
        acc.reset()
        assert acc.get_value() == 0.0
        assert acc.get_state()["sum"] == 0.0
        assert acc.get_state()["count"] == 0

    def test_sum_accumulator(self):
        """Test SumAccumulator operations."""
        acc = SumAccumulator(Reduce.SUM)

        acc.append(5.0)
        acc.append(3.0)
        assert acc.get_value() == 8.0

        state = acc.get_state()
        assert state["total"] == 8.0
        assert state["reduction_type"] == "sum"

        acc.reset()
        assert acc.get_value() == 0.0

    def test_max_accumulator(self):
        """Test MaxAccumulator operations."""
        acc = MaxAccumulator(Reduce.MAX)

        acc.append(5.0)
        acc.append(10.0)
        acc.append(3.0)
        assert acc.get_value() == 10.0

        state = acc.get_state()
        assert state["max_val"] == 10.0
        assert state["reduction_type"] == "max"

    def test_min_accumulator(self):
        """Test MinAccumulator operations."""
        acc = MinAccumulator(Reduce.MIN)

        acc.append(5.0)
        acc.append(10.0)
        acc.append(3.0)
        assert acc.get_value() == 3.0

        state = acc.get_state()
        assert state["min_val"] == 3.0
        assert state["reduction_type"] == "min"

    def test_std_accumulator(self):
        """Test StdAccumulator operations."""
        acc = StdAccumulator(Reduce.STD)

        # Test with zero/one values
        assert acc.get_value() == 0.0
        acc.append(5.0)
        assert acc.get_value() == 0.0  # std of single value is 0

        # Test with multiple values
        acc.append(7.0)  # values: 5, 7, mean=6, std=1
        assert abs(acc.get_value() - 1.0) < 0.001

        state = acc.get_state()
        assert state["sum"] == 12.0
        assert state["sum_sq"] == 74.0  # 5^2 + 7^2 = 25 + 49 = 74
        assert state["count"] == 2

    @pytest.mark.parametrize(
        "accumulator_class,states,expected",
        [
            (
                MeanAccumulator,
                [
                    {"reduction_type": "mean", "sum": 10.0, "count": 2},
                    {"reduction_type": "mean", "sum": 20.0, "count": 3},
                ],
                6.0,  # (10+20) / (2+3)
            ),
            (
                SumAccumulator,
                [
                    {"reduction_type": "sum", "total": 10.0},
                    {"reduction_type": "sum", "total": 15.0},
                ],
                25.0,
            ),
        ],
    )
    def test_accumulator_state_reduction(self, accumulator_class, states, expected):
        """Test cross-accumulator state reduction."""
        result = accumulator_class.get_reduced_value_from_states(states)
        assert result == expected

    def test_reduce_enum_accumulator_mapping(self):
        """Test that Reduce enum correctly maps to accumulator classes."""
        assert Reduce.MEAN.accumulator_class == MeanAccumulator
        assert Reduce.SUM.accumulator_class == SumAccumulator
        assert Reduce.MAX.accumulator_class == MaxAccumulator
        assert Reduce.MIN.accumulator_class == MinAccumulator
        assert Reduce.STD.accumulator_class == StdAccumulator


class TestMetricCollector:
    """Test MetricCollector singleton behavior and operations."""

    def test_singleton_per_rank(self, mock_rank):
        """Test MetricCollector singleton behavior per rank."""
        mock_rank.return_value.rank = 0
        collector1 = MetricCollector()
        collector2 = MetricCollector()
        assert collector1 is collector2

        # Different rank should get different instance
        mock_rank.return_value.rank = 1
        collector3 = MetricCollector()
        assert collector1 is not collector3

    def test_uninitialized_push_raises_error(self, mock_rank):
        """Test MetricCollector.push() raises error when uninitialized."""
        collector = MetricCollector()
        metric = Metric("test", 1.0, Reduce.MEAN)

        with pytest.raises(ValueError, match="MetricCollector was not initialized"):
            collector.push(metric)

    def test_invalid_metric_type_raises_error(self, mock_rank):
        """Test MetricCollector.push() raises error for invalid metric type."""
        collector = MetricCollector()
        collector._is_initialized = True
        collector.per_rank_no_reduce_backends = []
        collector.per_rank_reduce_backends = []

        with pytest.raises(TypeError, match="Expected .* object, got"):
            # Type ignore because we're intentionally testing invalid input
            collector.push("invalid_metric")  # type: ignore

    @patch("forge.observability.metrics.get_actor_name_with_rank")
    @pytest.mark.asyncio
    async def test_push_and_flush(self, mock_actor_name, initialized_collector):
        """Test MetricCollector push and flush with mock backends."""
        mock_actor_name.return_value = "TestActor_abcd_r0"

        collector = initialized_collector["collector"]
        no_reduce_backend = initialized_collector["no_reduce_backend"]
        reduce_backend = initialized_collector["reduce_backend"]

        # Test push
        metric = Metric("loss", 1.5, Reduce.MEAN)
        collector.push(metric)

        # Should log immediately to no_reduce backend
        assert len(no_reduce_backend.immediate_metrics) == 1
        assert no_reduce_backend.immediate_metrics[0][0].key == "loss"
        assert no_reduce_backend.immediate_metrics[0][1] == 0  # step

        # Should not log to reduce backend yet
        assert len(reduce_backend.logged_metrics) == 0

        # Test flush
        result = await collector.flush(step=1, return_state=True)

        # Should have returned state
        assert "loss" in result
        assert result["loss"]["reduction_type"] == "mean"
        assert result["loss"]["sum"] == 1.5
        assert result["loss"]["count"] == 1

        # Should have logged to reduce backend
        assert len(reduce_backend.logged_metrics) == 1
        logged_metric, step = reduce_backend.logged_metrics[0]
        assert logged_metric.key == "loss"
        assert logged_metric.value == 1.5
        assert step == 1

    @pytest.mark.asyncio
    async def test_flush_uninitialized_returns_empty(self, mock_rank):
        """Test MetricCollector.flush() returns empty dict when uninitialized."""
        collector = MetricCollector()
        result = await collector.flush(step=1, return_state=True)
        assert result == {}

    @pytest.mark.asyncio
    async def test_flush_no_metrics_returns_empty(self, mock_rank):
        """Test MetricCollector.flush() returns empty dict when no metrics."""
        collector = MetricCollector()
        collector._is_initialized = True
        collector.per_rank_no_reduce_backends = []
        collector.per_rank_reduce_backends = []

        result = await collector.flush(step=1, return_state=True)
        assert result == {}


class TestReduceOperations:
    """Test reduce_metrics_states function."""

    def test_empty_states(self):
        """Test reduce_metrics_states with empty input."""
        result = reduce_metrics_states([])
        assert result == []

    def test_single_state(self):
        """Test reduce_metrics_states with single state."""
        states = [{"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2}}]
        result = reduce_metrics_states(states)
        assert len(result) == 1
        assert result[0].key == "loss"
        assert result[0].value == 5.0
        assert result[0].reduction == Reduce.MEAN

    def test_multiple_states(self):
        """Test reduce_metrics_states with multiple states."""
        states = [
            {"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2}},
            {"loss": {"reduction_type": "mean", "sum": 20.0, "count": 3}},
            {"accuracy": {"reduction_type": "sum", "total": 15.0}},
        ]
        result = reduce_metrics_states(states)

        # Convert to dict for easier testing
        result_dict = {metric.key: metric.value for metric in result}
        assert result_dict["loss"] == 30.0 / 5.0  # 6.0
        assert result_dict["accuracy"] == 15.0

        # Also check reduction types
        for metric in result:
            if metric.key == "loss":
                assert metric.reduction == Reduce.MEAN
            elif metric.key == "accuracy":
                assert metric.reduction == Reduce.SUM

    def test_mismatched_reduction_types_raises_error(self):
        """Test reduce_metrics_states raises error for mismatched reduction types."""
        states = [
            {"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2}},
            {"loss": {"reduction_type": "sum", "total": 20.0}},
        ]
        with pytest.raises(ValueError, match="Mismatched reduction types"):
            reduce_metrics_states(states)

    def test_partial_key_overlap(self):
        """Test reduce_metrics_states with partial key overlap."""
        states = [
            {
                "loss": {"reduction_type": "mean", "sum": 10.0, "count": 2},
                "accuracy": {"reduction_type": "sum", "total": 5.0},
            },
            {"loss": {"reduction_type": "mean", "sum": 20.0, "count": 3}},
            {"throughput": {"reduction_type": "max", "max_val": 100.0}},
        ]
        result = reduce_metrics_states(states)

        # Convert to dict for easier testing
        result_dict = {metric.key: metric.value for metric in result}
        assert result_dict["loss"] == 30.0 / 5.0  # 6.0
        assert result_dict["accuracy"] == 5.0
        assert result_dict["throughput"] == 100.0


class TestBackends:
    """Test backend classes and factory function."""

    def test_backend_factory(self):
        """Test get_logger_backend_class factory function."""
        assert get_logger_backend_class("console") == ConsoleBackend
        assert get_logger_backend_class("wandb") == WandbBackend

        with pytest.raises(ValueError, match="Unknown logger backend type"):
            get_logger_backend_class("invalid_backend")

    @patch("forge.observability.metrics.get_actor_name_with_rank")
    @pytest.mark.asyncio
    async def test_console_backend(self, mock_actor_name):
        """Test ConsoleBackend basic operations."""
        mock_actor_name.return_value = "TestActor_abcd_r0"

        backend = ConsoleBackend({})

        await backend.init(role="local")

        # Test log_immediately
        metric = Metric("test", 1.0, Reduce.MEAN)
        backend.log_immediately(metric, step=1)  # Should not raise

        # Test log
        await backend.log([metric], step=1)  # Should not raise

        await backend.finish()  # Should not raise

    @patch("forge.observability.metrics.get_actor_name_with_rank")
    @pytest.mark.asyncio
    async def test_wandb_backend_creation(self, mock_actor_name):
        """Test WandbBackend creation and basic setup."""
        mock_actor_name.return_value = "TestActor_abcd_r0"

        config = {
            "project": "test_project",
            "group": "test_group",
            "logging_mode": "per_rank_reduce",
        }
        backend = WandbBackend(config)

        assert backend.project == "test_project"
        assert backend.group == "test_group"
        assert backend.per_rank_share_run is False  # default

        # Test metadata method
        metadata = backend.get_metadata_for_secondary_ranks()
        assert metadata == {}  # Should be empty when no run

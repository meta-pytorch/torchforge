# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for core metrics functionality focusing on critical fixes in Diff 1."""

from unittest.mock import MagicMock, patch

import pytest

from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import (
    ConsoleBackend,
    get_logger_backend_class,
    MeanAccumulator,
    MetricCollector,
    record_metric,
    Reduce,
    WandbBackend,
)


class TestCriticalFixes:
    """Test critical production fixes from Diff 1."""

    def test_uninitialized_push_logs_warning(self, mock_rank, caplog):
        """Test MetricCollector.push() logs warning when uninitialized."""
        collector = MetricCollector()

        # Should not raise error, just log warning and return
        collector.push("test", 1.0, Reduce.MEAN)
        assert any(
            "Metric logging backends" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_uninitialized_flush_logs_warning(self, mock_rank, caplog):
        """Test MetricCollector.flush() logs warning when uninitialized."""
        collector = MetricCollector()

        # Should not raise error, just log warning and return empty dict
        result = await collector.flush(step=1, return_state=True)
        assert result == {}
        assert any(
            "Cannot flush collected metrics" in record.message
            for record in caplog.records
        )

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

    @patch("forge.observability.metrics.get_actor_name_with_rank")
    def test_wandb_backend_creation(self, mock_actor_name):
        """Test WandbBackend creation and basic setup without WandB dependency."""
        mock_actor_name.return_value = "TestActor_abcd_r0"

        config = {
            "project": "test_project",
            "group": "test_group",
            "reduce_across_ranks": True,
        }
        backend = WandbBackend(config)

        assert backend.project == "test_project"
        assert backend.group == "test_group"
        assert backend.reduce_across_ranks is True
        assert backend.share_run_id is False  # default

        # Test metadata method
        metadata = backend.get_metadata_for_secondary_ranks()
        assert metadata == {}  # Should be empty when no run

    @patch("forge.observability.metrics.get_actor_name_with_rank")
    @pytest.mark.asyncio
    async def test_console_backend(self, mock_actor_name):
        """Test ConsoleBackend basic operations."""
        mock_actor_name.return_value = "TestActor_abcd_r0"

        backend = ConsoleBackend({})

        await backend.init(role="local")

        # Test log - should not raise
        await backend.log({"test": 1.0}, step=1)

        await backend.finish()  # Should not raise


class TestBasicAccumulators:
    """Test basic accumulator functionality."""

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

    def test_reduce_enum_accumulator_mapping(self):
        """Test that Reduce enum correctly maps to accumulator classes."""
        assert Reduce.MEAN.accumulator_class == MeanAccumulator


class TestBackendFactory:
    """Test backend factory function."""

    def test_backend_factory(self):
        """Test get_logger_backend_class factory function."""
        assert get_logger_backend_class("console") == ConsoleBackend
        assert get_logger_backend_class("wandb") == WandbBackend

        with pytest.raises(ValueError, match="Unknown logger backend type"):
            get_logger_backend_class("invalid_backend")


class TestMetricCollector:
    """Test MetricCollector singleton behavior."""

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


class TestMetricActorDisabling:
    """Test environment flag to disable metric actors."""

    async def _test_fetcher_registration(self, env_var_value, should_register_fetchers):
        """Check if FORGE_DISABLE_METRICS=[True, False, None] correctly disables fetcher registration.

        Args:
            env_var_value: Value to set for FORGE_DISABLE_METRICS (None means unset)
            should_register_fetchers: Whether fetchers should be registered (True) or not (False)
        """
        import os

        import forge.observability.metric_actors
        from forge.env_constants import FORGE_DISABLE_METRICS
        from monarch.actor import this_host

        # set fresh env
        # Note: Environment variable setup is handled by clean_metrics_environment fixture
        forge.observability.metric_actors._global_logger = None

        if env_var_value is not None:
            os.environ[FORGE_DISABLE_METRICS] = env_var_value

        procs = this_host().spawn_procs(per_host={"cpus": 1})

        if hasattr(procs, "_local_fetcher"):
            delattr(procs, "_local_fetcher")

        # Test functionality
        global_logger = await get_or_create_metric_logger(proc_mesh=procs)

        # Get results to check
        proc_has_fetcher = hasattr(procs, "_local_fetcher")
        global_has_fetcher = await global_logger.has_fetcher.call_one(procs)

        # Assert based on expected behavior
        if should_register_fetchers:
            assert (
                proc_has_fetcher
            ), f"Expected process to have _local_fetcher when {env_var_value=}"
            assert (
                global_has_fetcher
            ), f"Expected global logger to have fetcher registered when {env_var_value=}"
        else:
            assert (
                not proc_has_fetcher
            ), f"Expected process to NOT have _local_fetcher when {env_var_value=}"
            assert (
                not global_has_fetcher
            ), f"Expected global logger to NOT have fetcher registered when {env_var_value=}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "env_value,should_register",
        [
            ("false", True),
            ("true", False),
            (None, True),
        ],
    )
    async def test_fetcher_registration_with_env_flag(self, env_value, should_register):
        """Test fetcher registration behavior with different environment flag values."""
        await self._test_fetcher_registration(env_value, should_register)

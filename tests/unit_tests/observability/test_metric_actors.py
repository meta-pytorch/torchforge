# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimized unit tests for metric actors functionality."""

import pytest

from forge.observability.metric_actors import (
    get_or_create_metric_logger,
    GlobalLoggingActor,
    LocalFetcherActor,
)
from forge.observability.metrics import LoggingMode
from monarch.actor import this_host


@pytest.fixture
def global_logger():
    """Create a GlobalLoggingActor for testing."""
    p = this_host().spawn_procs(per_host={"cpus": 1})
    return p.spawn("TestGlobalLogger", GlobalLoggingActor)


@pytest.fixture
def local_fetcher(global_logger):
    """Create a LocalFetcherActor linked to global logger."""
    p = this_host().spawn_procs(per_host={"cpus": 1})
    return p.spawn("TestLocalFetcher", LocalFetcherActor, global_logger)


class TestBasicOperations:
    """Test basic operations for actors."""

    @pytest.mark.asyncio
    async def test_local_fetcher_flush(self, local_fetcher):
        """Test LocalFetcherActor flush operations."""
        result_with_state = await local_fetcher.flush.call_one(
            step=1, return_state=True
        )
        assert result_with_state == {}

        result_without_state = await local_fetcher.flush.call_one(
            step=1, return_state=False
        )
        assert result_without_state == {}

    @pytest.mark.asyncio
    async def test_global_logger_basic_ops(self, global_logger):
        """Test GlobalLoggingActor basic operations."""
        count = await global_logger.get_fetcher_count.call_one()
        assert count >= 0

        has_fetcher = await global_logger.has_fetcher.call_one("nonexistent")
        assert has_fetcher is False

        # Global logger flush (should not raise error)
        await global_logger.flush.call_one(step=1)

    @pytest.mark.asyncio
    async def test_backend_init(self, local_fetcher):
        """Test backend initialization and shutdown."""
        metadata = {"wandb": {"shared_run_id": "test123"}}
        config = {"console": {"logging_mode": "per_rank_reduce"}}

        await local_fetcher.init_backends.call_one(metadata, config, step=5)
        await local_fetcher.shutdown.call_one()


class TestRegistrationLifecycle:
    """Test registration lifecycle."""

    @pytest.mark.timeout(3)
    @pytest.mark.asyncio
    async def test_registration_lifecycle(self, global_logger, local_fetcher):
        """Test complete registration/deregistration lifecycle."""
        proc_name = "lifecycle_test_proc"

        # Initial state
        initial_count = await global_logger.get_fetcher_count.call_one()
        assert await global_logger.has_fetcher.call_one(proc_name) is False

        # Register
        await global_logger.register_fetcher.call_one(local_fetcher, proc_name)

        # Verify registered
        new_count = await global_logger.get_fetcher_count.call_one()
        assert new_count == initial_count + 1
        assert await global_logger.has_fetcher.call_one(proc_name) is True

        # Deregister
        await global_logger.deregister_fetcher.call_one(proc_name)

        # Verify deregistered
        final_count = await global_logger.get_fetcher_count.call_one()
        assert final_count == initial_count
        assert await global_logger.has_fetcher.call_one(proc_name) is False


class TestBackendConfiguration:
    """Test backend configuration validation."""

    @pytest.mark.timeout(3)
    @pytest.mark.asyncio
    async def test_valid_backend_configs(self, global_logger):
        """Test valid backend configurations."""
        # Empty config
        await global_logger.init_backends.call_one({})

        # Valid configs for all logging modes
        for mode in ["per_rank_reduce", "per_rank_no_reduce", "global_reduce"]:
            config = {"console": {"logging_mode": mode}}
            await global_logger.init_backends.call_one(config)

    @pytest.mark.timeout(3)
    @pytest.mark.asyncio
    async def test_invalid_backend_configs(self, global_logger):
        """Test invalid backend configurations raise errors."""
        from monarch.actor import ActorError

        # Missing logging_mode should work (has fallback to global_reduce)
        await global_logger.init_backends.call_one({"console": {}})

        # Invalid logging_mode should raise error (wrapped in ActorError since it's in an actor call)
        with pytest.raises(ActorError):
            await global_logger.init_backends.call_one(
                {"console": {"logging_mode": "invalid_mode"}}
            )


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.timeout(3)
    @pytest.mark.asyncio
    async def test_deregister_nonexistent_fetcher(self, global_logger):
        """Test deregistering non-existent fetcher doesn't crash."""
        await global_logger.deregister_fetcher.call_one("nonexistent_proc")

    @pytest.mark.timeout(3)
    @pytest.mark.asyncio
    async def test_shutdown(self, global_logger):
        """Test shutdown without issues."""
        await global_logger.shutdown.call_one()


class TestGetOrCreateMetricLogger:
    """Test the integration function."""

    @pytest.mark.timeout(3)
    @pytest.mark.asyncio
    async def test_get_or_create_functionality(self):
        """Test get_or_create_metric_logger basic functionality."""
        result = await get_or_create_metric_logger()

        # Should return a GlobalLoggingActor mesh
        assert result is not None

        # Should be able to call basic methods
        count = await result.get_fetcher_count.call_one()
        assert count >= 0


class TestSynchronousLogic:
    """Test synchronous logic without actor system (fastest tests)."""

    def test_all_validation_logic(self):
        """COMBINED: Test all synchronous validation logic."""
        actor = GlobalLoggingActor()

        # Test 1: Valid config validation
        config = {"logging_mode": "per_rank_reduce", "project": "test_project"}
        result = actor._validate_backend_config("test_backend", config)
        assert result["logging_mode"] == LoggingMode.PER_RANK_REDUCE
        assert result["project"] == "test_project"

        # Test 2: Missing logging_mode (should work with default)
        result2 = actor._validate_backend_config(
            "test_backend", {"project": "test_project"}
        )
        assert (
            result2["logging_mode"] == LoggingMode.GLOBAL_REDUCE
        )  # Should default to global_reduce
        assert result2["project"] == "test_project"

        # Test 3: Invalid logging_mode error (enum will raise ValueError)
        with pytest.raises(ValueError, match="is not a valid LoggingMode"):
            actor._validate_backend_config(
                "test_backend", {"logging_mode": "invalid_mode"}
            )

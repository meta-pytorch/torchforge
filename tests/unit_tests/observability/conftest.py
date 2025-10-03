# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures and mocks for observability unit tests."""

from unittest.mock import MagicMock, patch

import pytest
from forge.observability.metrics import LoggerBackend, MetricCollector


class MockBackend(LoggerBackend):
    """Mock backend for testing metrics logging without external dependencies."""

    def __init__(self, logger_backend_config=None):
        super().__init__(logger_backend_config or {})
        self.logged_metrics = []
        self.immediate_metrics = []
        self.init_called = False
        self.finish_called = False
        self.metadata = {}

    async def init(self, role="local", primary_logger_metadata=None, actor_name=None):
        self.init_called = True
        self.role = role
        self.primary_logger_metadata = primary_logger_metadata or {}
        self.actor_name = actor_name

    def log_immediate(self, metric, step, *args, **kwargs):
        self.immediate_metrics.append((metric, step))

    async def log(self, metrics, step, *args, **kwargs):
        for metric in metrics:
            self.logged_metrics.append((metric, step))

    async def finish(self):
        self.finish_called = True

    def get_metadata_for_secondary_ranks(self):
        return self.metadata


@pytest.fixture(autouse=True)
def clear_metric_collector_singletons():
    """Clear MetricCollector singletons before each test to avoid state leakage."""
    MetricCollector._instances.clear()
    yield
    MetricCollector._instances.clear()


@pytest.fixture(autouse=True)
def clean_metrics_environment():
    """Ensure clean environment state for metrics tests."""
    import os

    # Save original environment state
    original_env = os.environ.get("FORGE_DISABLE_METRICS")

    # Set default state for tests (metrics enabled)
    if "FORGE_DISABLE_METRICS" in os.environ:
        del os.environ["FORGE_DISABLE_METRICS"]

    yield

    # Restore original environment state
    if original_env is not None:
        os.environ["FORGE_DISABLE_METRICS"] = original_env
    elif "FORGE_DISABLE_METRICS" in os.environ:
        del os.environ["FORGE_DISABLE_METRICS"]


@pytest.fixture
def mock_rank():
    """Mock current_rank function with configurable rank."""
    with patch("forge.observability.metrics.current_rank") as mock:
        rank_obj = MagicMock()
        rank_obj.rank = 0
        mock.return_value = rank_obj
        yield mock


@pytest.fixture
def mock_actor_context():
    """Mock Monarch actor context for testing actor name generation."""
    with patch("forge.observability.metrics.context") as mock_context, patch(
        "forge.observability.metrics.current_rank"
    ) as mock_rank:

        # Setup mock context
        ctx = MagicMock()
        actor_instance = MagicMock()
        actor_instance.actor_id = "_1rjutFUXQrEJ[0].TestActorConfigured[0]"
        ctx.actor_instance = actor_instance
        mock_context.return_value = ctx

        # Setup mock rank
        rank_obj = MagicMock()
        rank_obj.rank = 0
        mock_rank.return_value = rank_obj

        yield {
            "context": mock_context,
            "rank": mock_rank,
            "expected_name": "TestActor_0XQr_r0",
        }


@pytest.fixture
def initialized_collector():
    """Create an initialized MetricCollector with mock backends for testing."""
    with patch("forge.observability.metrics.current_rank") as mock_rank:
        mock_rank.return_value = MagicMock(rank=0)

        MetricCollector._instances.clear()
        collector = MetricCollector()

        # Setup mock backends
        no_reduce_backend = MockBackend()
        reduce_backend = MockBackend()

        collector._is_initialized = True
        collector.per_rank_no_reduce_backends = [no_reduce_backend]
        collector.per_rank_reduce_backends = [reduce_backend]
        collector.step = 0

        yield {
            "collector": collector,
            "no_reduce_backend": no_reduce_backend,
            "reduce_backend": reduce_backend,
        }

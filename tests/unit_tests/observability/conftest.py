# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures and mocks for observability unit tests."""

from unittest.mock import MagicMock, patch

import pytest
from forge.observability.metrics import MetricCollector


@pytest.fixture(autouse=True)
def clear_metric_collector_singletons():
    """Clear MetricCollector singletons before each test to avoid state leakage."""
    MetricCollector._instances.clear()
    yield
    MetricCollector._instances.clear()


@pytest.fixture(autouse=True)
def clean_metrics_environment():
    """Override the global mock_metrics_globally fixture to allow real metrics testing."""
    import os

    from forge.env import FORGE_DISABLE_METRICS

    # Set default state for tests (metrics enabled)
    if FORGE_DISABLE_METRICS.name in os.environ:
        del os.environ[FORGE_DISABLE_METRICS.name]

    yield


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
    with (
        patch("forge.observability.metrics.context") as mock_context,
        patch("forge.observability.metrics.current_rank") as mock_rank,
    ):
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

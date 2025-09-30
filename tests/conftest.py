# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Global test configuration for all tests.

This file contains pytest fixtures that are automatically applied to all tests
in the forge test suite.
"""

from unittest.mock import Mock

import pytest

from forge.env_constants import FORGE_DISABLE_METRICS


@pytest.fixture(autouse=True)
def mock_metrics_globally(monkeypatch):
    """
    Automatically disable `forge.observability.metrics.record_metrics` during tests,
    which could otherwise introduce flakiness if not properly configured.

    To disable this mock in a specific test, override the fixture:

        @pytest.fixture
        def mock_metrics_globally():
            # Return None to disable the mock for this test
            return None

        def test_real_metrics(mock_metrics_globally):
            # This test will use the real metrics system
            pass
    """

    monkeypatch.setenv(FORGE_DISABLE_METRICS, "true")
    return Mock()

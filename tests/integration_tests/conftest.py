# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--config",
        action="store",
        default=None,
        help="Path to YAML config file for sanity check tests"
    )


@pytest.fixture
def config_path(request):
    """Fixture to provide the config path from command line."""
    return request.config.getoption("--config")

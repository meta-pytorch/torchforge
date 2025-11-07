# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests configuration.

IMPORTANT: Due to Monarch's cleanup issues, integration tests should be run
separately for proper isolation. Run individual test files like:

    pytest tests/integration_tests/test_grpo_e2e.py -vv
    pytest tests/integration_tests/test_policy_update.py -vv

Or run all tests (each file in separate process):

    for f in tests/integration_tests/test_*.py; do pytest "$f" -vv; done
"""

import argparse

import pytest


def str_to_bool(value):
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--config",
        action="store",
        default=None,
        help="Path to YAML config file for sanity check tests",
    )

    parser.addoption(
        "--use_dcp",
        action="store",
        type=str_to_bool,
        default=None,
        help="Overrides the YAML config `trainer.use_dcp` field.",
    )


@pytest.fixture
def config_path(request):
    """Fixture to provide the config path from command line."""
    return request.config.getoption("--config")

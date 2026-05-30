# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Centralized constants for environment variable names used in the project."""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvVar:
    """Configuration for an environment variable."""

    name: str
    default: Any
    description: str

    def get_value(self) -> Any:
        """Get the value of this environment variable with fallback to default.

        Returns:
            The environment variable value, auto-converted to the appropriate type
            based on the default value, or the default value if not set.

        Example:
            >>> DISABLE_PERF_METRICS.get_value()
            False
            >>> os.environ["DISABLE_PERF_METRICS"] = "true"
            >>> DISABLE_PERF_METRICS.get_value()
            True
        """
        value = os.environ.get(self.name)

        if value is None:
            return self.default

        # Auto-convert based on the default type
        if isinstance(self.default, bool):
            return value.lower() in ("true", "1", "yes")
        elif isinstance(self.default, int):
            return int(value)
        elif isinstance(self.default, float):
            return float(value)
        else:
            # Return as string for other types
            return value


# Environment variable definitions
DISABLE_PERF_METRICS = EnvVar(
    name="DISABLE_PERF_METRICS",
    default=False,
    description="Performance metrics in forge.observability.perf_tracker.py becomes no-op",
)

METRIC_TIMER_USES_GPU = EnvVar(
    name="METRIC_TIMER_USES_GPU",
    default=None,
    description=(
        "Force all timing methods in forge.observability.perf_tracker.py "
        "to use CPU timer if False or GPU timer if True. If unset (None), defaults to the timer parameter."
    ),
)

FORGE_DISABLE_METRICS = EnvVar(
    name="FORGE_DISABLE_METRICS",
    default=False,
    description=(
        "Makes forge.observability.metrics.record_metric a no-op and disables spawning LocalFetcherActor"
        " in get_or_create_metric_logger"
    ),
)

MONARCH_STDERR_LEVEL = EnvVar(
    name="MONARCH_STDERR_LOG",
    default="warning",
    description="Sets Monarch's stderr log level, i.e. set to 'info' or 'debug'",
)

RUST_BACKTRACE = EnvVar(
    name="RUST_BACKTRACE",
    default="full",
    description="Sets the level for Rust-level failures. I.e. set to full for full stack traces.",
)

MONARCH_MESSAGE_DELIVERY_TIMEOUT = EnvVar(
    name="HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS",
    default=600,
    description="Sets the timeout limit for Monarch's actor message delivery in seconds.",
)

MONARCH_MAX_FRAME_LENGTH = EnvVar(
    name="HYPERACTOR_CODE_MAX_FRAME_LENGTH",
    default=1073741824,
    description="Sets the maximum frame length for Monarch's actor message delivery in bytes.",
)

TORCHSTORE_USE_RDMA = EnvVar(
    name="TORCHSTORE_RDMA_ENABLED",
    default=1,
    description="Whether or not to use RDMA in TorchStore.",
)

MONARCH_OLD_ASYNC_WORKAROUND = EnvVar(
    name="MONARCH_OLD_ASYNC_WORKAROUND",
    default=1,
    description=(
        "If enabled, monarch messages will be sent immediately even it's not"
        " awaited. This is needed for parallel fetching of weights, as using"
        " create_task creates race condition. This is a temporary workaround"
        " and will be removed once we have a better solution."
    ),
)


def all_env_vars() -> list[EnvVar]:
    """Retrieves all registered environment variable names."""
    env_vars = []
    for _, value in globals().items():
        if isinstance(value, EnvVar):
            env_vars.append(value)
    return env_vars

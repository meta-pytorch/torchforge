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
        "to use CPU timer if False or GPU timer if True. If unset, defaults to the assigned value to the function."
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

TORCHSTORE_USE_RDMA = EnvVar(
    name="TORCHSTORE_RDMA_ENABLED",
    default=False,
    description="Whether or not to use RDMA in TorchStore.",
)

RUST_BACKTRACE = EnvVar(
    name="RUST_BACKTRACE",
    default="0",
    description="Sets the level for Rust-level failures. I.e. set to full for full stack traces.",
)


def _build_registry() -> dict[str, EnvVar]:
    """Automatically builds registry from all EnvVar instances in module globals."""
    registry = {}
    for name, value in globals().items():
        if isinstance(value, EnvVar):
            registry[value.name] = value
    return registry


# Registry of all environment variables (automatically populated)
_ENV_VAR_REGISTRY: dict[str, EnvVar] = _build_registry()


def get_value(env_var_name: str) -> Any:
    """Get the value of an environment variable with fallback to default.

    Args:
        env_var_name: The name of the environment variable.

    Returns:
        The environment variable value, auto-converted to the appropriate type
        based on the default value, or the default value if not set.

    Raises:
        KeyError: If the env_var_name is not registered.
    """
    if env_var_name not in _ENV_VAR_REGISTRY:
        raise KeyError(
            f"Environment variable '{env_var_name}' is not registered. "
            f"Available variables: {list(_ENV_VAR_REGISTRY.keys())}"
        )

    env_var = _ENV_VAR_REGISTRY[env_var_name]
    value = os.environ.get(env_var_name)

    if value is None:
        return env_var.default

    # Auto-convert based on the default type
    if isinstance(env_var.default, bool):
        return value.lower() in ("true", "1", "yes")
    elif isinstance(env_var.default, int):
        return int(value)
    elif isinstance(env_var.default, float):
        return float(value)
    else:
        # Return as string for other types
        return value


def all_env_vars() -> dict[str, EnvVar]:
    """Returns all registered environment variable configurations."""
    return _ENV_VAR_REGISTRY.copy()


def all_constants() -> list[str]:
    """Retrieves all registered environment variable names."""
    return list(_ENV_VAR_REGISTRY.keys())

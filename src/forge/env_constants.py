# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Centralized constants for environment variable names used in the project."""

# Performance metrics in forge.observability.perf_tracker.py becomes no-op
DISABLE_PERF_METRICS = "DISABLE_PERF_METRICS"

# Force all timing methods in forge.observability.perf_tracker.py to use
# CPU timer if False or GPU timer if True. If unset, defaults to the assigned value to the function.
METRIC_TIMER_USES_GPU = "METRIC_TIMER_USES_GPU"

# Makes forge.observability.metrics.record_metric a no-op
# and disables spawning LocalFetcherActor in get_or_create_metric_logger
FORGE_DISABLE_METRICS = "FORGE_DISABLE_METRICS"

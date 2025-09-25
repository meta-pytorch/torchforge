# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextvars
import inspect
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import wraps

import torch

from forge.observability.metrics import record_metric, ReductionType

# Per-task flags for nesting detection
_MEMORY_TRACKING_ACTIVE = contextvars.ContextVar("memory_active", default=False)
_MEMORY_NESTING_WARNED = False


@asynccontextmanager
async def _perf_metrics_cm(
    prefix: str, track_time: bool, track_memory: bool, sync_cuda_event: bool
):
    """
    Shared async context manager for recording performance metrics (time, memory).

    Args:
        prefix (str): Required prefix for metric keys (e.g., 'trainer_step').
        track_time (bool): If True, record timing metrics (avg/max duration).
        track_memory (bool): If True, record memory metrics (delta_end_start/peak, GiB).
        sync_cuda_event (bool): If True, use CUDA events for precise timing synchronization.

    Notes:
        - Uses CUDA events for synchronization when sync_cuda_event=True and CUDA is available.
        - Only the outermost scope tracks memory to avoid torch.cuda.max_memory_allocated() conflicts.
             i.e. foo(bar()) will only track memory for foo().
        - Inner scopes skip memory metrics with a one-time warning.
        - DISABLE_PERF_METRICS='true' skips all metrics.
        - METRICS_USE_CUDA_SYNC='true' forces CUDA event sync (overrides sync_cuda_event arg).
        - Uses contextvars for per-task nesting detection.
    """
    if os.getenv("DISABLE_PERF_METRICS", "false").lower() == "true":
        yield
        return

    do_sync = (
        os.getenv("METRICS_USE_CUDA_SYNC", "false").lower() == "true" or sync_cuda_event
    )
    global _MEMORY_NESTING_WARNED

    # Check nesting; disable memory tracking for inner scopes
    token = None
    is_outer_scope = not _MEMORY_TRACKING_ACTIVE.get()
    local_memory = track_memory and is_outer_scope
    if track_memory and not is_outer_scope and not _MEMORY_NESTING_WARNED:
        # TODO: add log_once in forge utilities and remove _MEMORY_NESTING_WARNED
        logging.warning(
            f"Nested memory tracking detected in {prefix}. Skipping memory metrics to prioritize outer scope."
        )
        _MEMORY_NESTING_WARNED = True

    if local_memory:
        token = _MEMORY_TRACKING_ACTIVE.set(True)
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

    # Initialize CUDA events for timing if needed
    start_event = None
    end_event = None
    if do_sync and torch.cuda.is_available() and track_time:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # Record start in default stream

    start_time = time.perf_counter() if track_time else 0
    start_mem = (
        torch.cuda.memory_allocated() / 1024**3
        if local_memory and torch.cuda.is_available()
        else 0
    )

    try:
        yield
    finally:
        if token:
            _MEMORY_TRACKING_ACTIVE.reset(token)

        duration = 0
        if track_time:
            if do_sync and torch.cuda.is_available() and start_event:
                end_event.record()  # Record end in default stream
                end_event.synchronize()  # Wait for end event to complete
                duration = (
                    start_event.elapsed_time(end_event) / 1000.0
                )  # Convert ms to seconds
            else:
                duration = time.perf_counter() - start_time

            await record_metric(
                f"{prefix}/duration_avg_s", duration, ReductionType.MEAN
            )
            await record_metric(f"{prefix}/duration_max_s", duration, ReductionType.MAX)

        if local_memory:
            end_mem = (
                torch.cuda.memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0
            )
            delta = end_mem - start_mem
            peak_mem = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0
            )
            await record_metric(
                f"{prefix}/memory_delta_avg_gb", delta, ReductionType.MEAN
            )
            await record_metric(
                f"{prefix}/memory_peak_max_gb", peak_mem, ReductionType.MAX
            )


def record_perf_metrics(
    prefix: str,
    track_time: bool = True,
    track_memory: bool = False,
    sync_cuda_event: bool = False,
):
    """Decorator for recording performance metrics (time, memory).
    Works with both sync and async functions by converting sync functions to async.
    For details, see _perf_metrics_cm.

    Example:
        @record_perf_metrics("trainer_step")
        async def async_step(...):
            ...

        @record_perf_metrics("sync_step")
        def sync_step(...):  # Will become async!
            ...
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            # Already async function
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with _perf_metrics_cm(
                    prefix, track_time, track_memory, sync_cuda_event
                ):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            # Sync function - convert to async
            @wraps(func)
            async def sync_to_async_wrapper(*args, **kwargs):
                async with _perf_metrics_cm(
                    prefix, track_time, track_memory, sync_cuda_event
                ):
                    return func(*args, **kwargs)  # Call sync function in async context

            return sync_to_async_wrapper

    return decorator


@asynccontextmanager
async def record_perf_metrics_ctx(
    prefix: str,
    track_time: bool = True,
    track_memory: bool = False,
    sync_cuda_event: bool = False,
):
    """Async context manager for recording performance metrics (time, memory).
    For details, see _perf_metrics_cm.

    Example:
        async with record_perf_metrics_ctx("trainer_step"):
            ...
    """
    async with _perf_metrics_cm(prefix, track_time, track_memory, sync_cuda_event):
        yield


class StepTimer:
    """Multi-step timer for loops with prefix-based metrics and sync cuda event support.

    Args:
        prefix (str): Prefix for metric keys (e.g., 'trainer_step/')
        sync_cuda_event (bool): If True, use CUDA events for precise timing synchronization
        aggregations (list[ReductionType]): List of aggregation types to compute.
            Default is [ReductionType.MEAN]. Can be overridden per step.

    Example:
        timer = StepTimer("trainer_perf/step", aggregations=[ReductionType.MEAN])
        await timer.start()
        foo()
        await timer.step("foo")
        bar()
        await timer.step("bar")
        await timer.end()

        Metrics logged:
            trainer_perf/step/foo/duration_avg (s)
            trainer_perf/step/bar/duration_avg (s)
            trainer_perf/step/total_duration_avg (s)
    """

    def __init__(
        self,
        prefix: str,
        sync_cuda_event: bool = False,
        aggregations: list[ReductionType] = None,
    ):
        self.prefix = prefix
        self.sync_cuda_event = (
            os.getenv("METRICS_USE_CUDA_SYNC", "false").lower() == "true"
            or sync_cuda_event
        )
        self.aggregations = aggregations or [ReductionType.MEAN]
        self.start_time = None
        self.last_time = None
        self.start_event = None
        self.last_event = None

    async def start(self):
        """Start the timer."""
        if self.sync_cuda_event and torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()  # Record start in default stream
            self.last_event = self.start_event
        self.start_time = time.perf_counter()
        self.last_time = self.start_time

    async def step(self, step_name: str, aggregations: list[ReductionType] = None):
        """
        Record timing for a step within the timer's scope.

        Args:
            step_name (str): Name of the step for metric suffix
            aggregations (list[ReductionType]): Override default aggregations for this step
        """
        duration = 0
        if self.sync_cuda_event and torch.cuda.is_available():
            current_event = torch.cuda.Event(enable_timing=True)
            current_event.record()  # Record step end in default stream
            if self.last_event:
                current_event.synchronize()  # Wait for current event
                duration = (
                    self.last_event.elapsed_time(current_event) / 1000.0
                )  # ms to seconds
            self.last_event = current_event
        else:
            now = time.perf_counter()
            if self.last_time is not None:
                duration = now - self.last_time
            self.last_time = now

        # Use provided aggregations or fall back to instance default
        aggs = aggregations or self.aggregations

        if self.last_time is not None or self.last_event:
            for reduction_type in aggs:
                suffix = self._get_suffix_for_reduction_type(reduction_type)
                await record_metric(
                    f"{self.prefix}/{suffix}_{step_name}_duration_s",
                    duration,
                    reduction_type,
                )

    def _get_suffix_for_reduction_type(self, reduction_type: ReductionType) -> str:
        """Map ReductionType enum to metric suffix using the enum name."""
        # Special case: MEAN -> avg for readability
        if reduction_type == ReductionType.MEAN:
            return "avg"
        return reduction_type.name.lower()

    async def end(self, aggregations: list[ReductionType] = None):
        """End the timer and record total duration.

        Args:
            aggregations (list[ReductionType]): Override default aggregations for total duration
        """
        duration = 0
        if self.sync_cuda_event and torch.cuda.is_available() and self.start_event:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            duration = self.start_event.elapsed_time(end_event) / 1000.0
        else:
            if self.start_time is not None:
                duration = time.perf_counter() - self.start_time

        # Use provided aggregations or fall back to instance default
        aggs = aggregations or self.aggregations

        for reduction_type in aggs:
            suffix = self._get_suffix_for_reduction_type(reduction_type)
            await record_metric(
                f"{self.prefix}/total_duration_{suffix}_s",
                duration,
                reduction_type,
            )

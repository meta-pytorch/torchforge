# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import inspect
import logging
import os
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import List, Optional, Protocol, Tuple

import torch

from forge.env_constants import DISABLE_PERF_METRICS, METRIC_TIMER_USES_CUDA
from forge.observability.metrics import record_metric, ReductionType

# Thread-local memory tracking state
_local = threading.local()


def _is_memory_active() -> bool:
    """Check if memory tracking is active in current thread.
    Used to detect nested memory tracking and skip inner tracking."""
    return getattr(_local, "memory_active", False)


def _set_memory_active(value: bool) -> None:
    """Set memory tracking state for current thread.
    Used to detect nested memory tracking and skip inner tracking."""
    _local.memory_active = value


@lru_cache(maxsize=1000)
def _warn_nested_memory_tracking(prefix: str) -> None:
    """Log nesting warning once per prefix using lru_cache for deduplication. Avoids spamming logs."""
    logging.warning(
        f"Nested memory tracking detected in {prefix}. Skipping inner tracking."
    )


"""
==========
Step Timer
==========
"""


class Timer:
    """
    Multi-step timer that records average and max timing metrics.

    Supports non-blocking CUDA timing via CUDA events and background polling threads;

    Aggregation is handled externally by the metrics system via record_metric.

    User must call start() and end() explicitly.
    Supports reuse: after calling end(), you may call start() again to begin a new timing session.

    Local env flag DISABLE_PERF_METRICS can be used to skip all timing operations.
    Local env flag METRIC_TIMER_USES_CUDA can be used to set CUDA timing.

    Args:
        prefix (str): Prefix for metric names, e.g. "my_prefix" -> "{my_prefix}/{step_name}/duration_avg_s".
        use_cuda (bool): Sets CUDA timing if True and CUDA is available.

    Example:
        timer = Timer("my_prefix")
        timer.start()
        time.sleep(0.1)  # Work for step "a"
        timer.step("my_step_a")
        for i in range(1, 4):  # 3 iterations
            time.sleep(i/10)  # 0.1, 0.2, 0.3 seconds
            timer.step("my_step_b")
        timer.end()  # Records metrics, including total time end-start.

        >>> Records:
        >>> my_prefix/my_step_a/duration_avg_s: 0.1
        >>> my_prefix/my_step_a/duration_max_s: 0.1
        >>> my_prefix/my_step_b/duration_avg_s: 0.2 # Average of 0.1, 0.2, 0.3
        >>> my_prefix/my_step_b/duration_max_s: 0.3 # Max of 0.1, 0.2, 0.3
        >>> my_prefix/total_duration_avg_s: 0.7 # Total: 0.1 + 0.1 + 0.2 + 0.3
        >>> my_prefix/total_duration_max_s: 0.7

        # Can reuse the same timer after end()
        timer.start()  # Begin new session
        timer.step("step1")
        timer.end()
    """

    def __init__(self, prefix: str, use_cuda: bool = False):
        self.prefix = prefix
        use_cuda_events = (
            os.getenv(METRIC_TIMER_USES_CUDA, str(use_cuda)).lower() == "true"
        ) and torch.cuda.is_available()
        use_cpu = not use_cuda_events
        self.timer: _TimerProtocol = _TimerCPU() if use_cpu else _TimerCUDA()
        self._active = False
        self._disable = os.getenv(DISABLE_PERF_METRICS, "false") == "true"

    def start(self) -> None:
        if self._disable:
            return
        if self._active:
            raise ValueError("Timer has already been started")
        self._active = True
        self.timer.start()

    def step(self, step_name: str) -> None:
        if self._disable:
            return
        if not self._active:
            raise ValueError("Timer must be started before calling step")
        self.timer.step(step_name)

    def end(self) -> None:
        if self._disable:
            return
        if not self._active:
            raise ValueError("Timer must be started before calling end")
        self.timer.step("end")  # dropped from steps, included in total
        self._record_metrics()
        self._active = False  # Allow reuse by calling start() again

    def _record_metrics(self) -> None:
        durations = self.timer.get_all_durations()

        # Total: sum all recorded durations (full timeline including end)
        total_ms = sum(d_ms for name, d_ms in durations)
        total_s = total_ms / 1000.0
        record_metric(
            f"{self.prefix}/total_duration_avg_s", total_s, ReductionType.MEAN
        )
        record_metric(f"{self.prefix}/total_duration_max_s", total_s, ReductionType.MAX)

        # Steps: record each individually (drop last "end")
        for name, d_ms in durations[:-1]:
            d_s = d_ms / 1000.0
            record_metric(
                f"{self.prefix}/{name}/duration_avg_s", d_s, ReductionType.MEAN
            )
            record_metric(
                f"{self.prefix}/{name}/duration_max_s", d_s, ReductionType.MAX
            )


class _TimerProtocol(Protocol):
    def start(self) -> None:
        ...

    def step(self, name: str) -> None:
        ...

    def get_all_durations(self) -> List[Tuple[str, float]]:
        ...


class _TimerCPU(_TimerProtocol):
    """
    CPU timing backend using perf_counter.
    """

    def __init__(self) -> None:
        self._durations: List[Tuple[str, float]] = []
        self._chain_start: Optional[float] = None

    def start(self) -> None:
        # Reset state for reuse
        self._durations = []
        self._chain_start = time.perf_counter()

    def step(self, name: str) -> None:
        if self._chain_start is None:
            raise ValueError("Timer must be started before calling step")
        now = time.perf_counter()
        delta_ms = (now - self._chain_start) * 1000
        self._durations.append((name, delta_ms))
        self._chain_start = now

    def get_all_durations(self) -> List[Tuple[str, float]]:
        return self._durations[:]


class _TimerCUDA(_TimerProtocol):
    """CUDA timing backend with non-blocking events and futures.
    Uses a thread pool to poll CUDA events asynchronously without blocking the main thread.
    """

    def __init__(self, max_workers: int = 2) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for timing")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: List[
            Tuple[str, Future[float], int]
        ] = []  # (name, future, submission_index)
        self._durations: List[Tuple[str, float]] = []
        self._chain_start: Optional[torch.cuda.Event] = None

    def start(self) -> None:
        """Call before any steps. Clear state for reuse; record initial event on current stream."""
        self._futures.clear()
        self._durations.clear()
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        self._chain_start = start_event

    def step(self, name: str) -> None:
        """Mark the end of a GPU workload segment and start the next, submitting async polling.
        Records a CUDA end event on the current stream; a background thread polls completion.

        Args:
            name: Label for this segment's duration
        """
        # Submit polling future; chain to next event.
        if self._chain_start is None:
            raise ValueError("Timer must be started before calling step")

        stream = torch.cuda.current_stream()
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(stream)

        def _compute_elapsed(start_event, end_event):
            # Poll with backoff: starts fast (1ms), grows to cap (50ms) for mixed workloads.
            sleep_time = 0.001  # Start at 1ms
            while not end_event.query():
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 0.05)  # Backoff, cap at 50ms
            return start_event.elapsed_time(end_event)

        future = self._executor.submit(_compute_elapsed, self._chain_start, end_event)
        index = len(self._futures)
        self._futures.append((name, future, index))

        if len(self._futures) >= 20:  # clean up every 20 to avoid memory leak
            self._collect_completed_futures()

        self._chain_start = end_event

    def _collect_completed_futures(self) -> None:
        """Drain done futures to avoid memory leak; update durations in submission order."""
        completed = []
        still_pending = []
        for name, future, idx in self._futures:
            if future.done():
                try:
                    dur = future.result()
                    completed.append((idx, name, dur))
                except Exception as e:
                    raise RuntimeError(f"Timing failed for {name}: {e}") from e
            else:
                still_pending.append((name, future, idx))

        # Sort completed by submission index to preserve order
        completed.sort(key=lambda x: x[0])
        for _, name, dur in completed:
            self._durations.append((name, dur))

        self._futures = still_pending

    def get_all_durations(self) -> List[Tuple[str, float]]:
        """Retrieve list of (name, duration) tuples in submission order after waiting for background polls to finish."""
        # Wait and collect if pendings; return durations.
        self._collect_completed_futures()
        completed = []
        for name, future, idx in self._futures:
            try:
                dur = future.result()
                completed.append((idx, name, dur))
            except Exception as e:
                raise RuntimeError(f"Timing failed for {name}: {e}") from e

        # Sort by submission index to preserve order
        completed.sort(key=lambda x: x[0])
        for _, name, dur in completed:
            self._durations.append((name, dur))

        self._futures.clear()
        return self._durations[:]

    def __del__(self) -> None:
        # Fallback cleanup in finalizer; ignores errors to avoid shutdown noise.
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            return


"""
=======================================
Memory+timer as decorator / ctx manager
=======================================
"""


def record_perf_metrics(
    prefix: str,
    track_time: bool = True,
    track_memory: bool = False,
    use_cuda: bool = False,
):
    """
    Decorator for functions with performance tracking, supporting both sync and async functions by detecting coroutine status.
    Tracks time and/or memory if enabled, recording metrics via record_metric. Skips if DISABLE_PERF_METRICS env flag is true.

    Args:
        prefix (str): Prefix for metric names
        track_time (bool): Whether to track execution time. Defaults to True.
        track_memory (bool): Whether to track CUDA memory usage. Defaults to False.
        use_cuda (bool): Whether to use CUDA events for timing (overridden by METRIC_TIMER_USES_CUDA env var).
            Defaults to False.

    Example:
        @record_perf_metrics("my_prefix", track_time=True, track_memory=True)
        async def my_async_func():
            pass

        @record_perf_metrics("my_prefix", track_time=True, track_memory=True)
        def my_sync_func():
            pass
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if os.getenv(DISABLE_PERF_METRICS, "false").lower() == "true":
                    return await func(*args, **kwargs)
                with _memory_tracking_cm(prefix, track_memory), _timer_cm(
                    prefix, track_time, use_cuda
                ):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if os.getenv(DISABLE_PERF_METRICS, "false").lower() == "true":
                    return func(*args, **kwargs)
                with _memory_tracking_cm(prefix, track_memory), _timer_cm(
                    prefix, track_time, use_cuda
                ):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


@contextmanager
def record_perf_metrics_ctx(
    prefix: str,
    track_time: bool = True,
    track_memory: bool = False,
    use_cuda: bool = False,
):
    """
    Context manager for performance tracking in a code block, combining time and memory metrics.
    Safe for use in async code; skips if DISABLE_PERF_METRICS env flag is true.

    Args:
        prefix (str): Prefix for metric names
        track_time (bool): Whether to track execution time. Defaults to True.
        track_memory (bool): Whether to track CUDA memory usage. Defaults to False.
        use_cuda (bool): Whether to use CUDA events for timing (overridden by METRIC_TIMER_USES_CUDA env var). Defaults to False.

    Example:
       with record_perf_metrics_ctx("my_block", track_time=True, track_memory=True):
            await some_task()
            some_other_task()
    """

    if os.getenv(DISABLE_PERF_METRICS, "false").lower() == "true":
        yield
        return

    with _memory_tracking_cm(prefix, track_memory), _timer_cm(
        prefix, track_time, use_cuda
    ):
        yield


@contextmanager
def _memory_tracking_cm(prefix: str, track_memory: bool):
    """
    Context manager for tracking CUDA's memory delta and peak during execution of a function
    in a process. Metrics are logged and aggregated using `record_metrics`

    One challenge is if we have nested functions using this utility. When it is called,
    it executes `torch.cuda.reset_max_memory_allocated()`. The inner function would wipe
    the memory stats of the outer function. To avoid this, we use a thread safe variable
    to mark that memory is currently being tracked. If this hapens, then the inner function
    does **NOT** track memory and logs a warning once.
    """
    is_outer_scope = not _is_memory_active()
    should_track_memory = track_memory and is_outer_scope and torch.cuda.is_available()

    if track_memory and not is_outer_scope:
        _warn_nested_memory_tracking(prefix)

    if should_track_memory:
        _set_memory_active(True)
        torch.cuda.reset_max_memory_allocated()
        start_mem = torch.cuda.memory_allocated()
    else:
        start_mem = 0.0

    try:
        yield
    finally:
        if should_track_memory:
            end_mem = torch.cuda.memory_allocated()
            delta = (end_mem - start_mem) / 1024**3
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            record_metric(
                f"{prefix}/memory_delta_end_start_avg_gb", delta, ReductionType.MEAN
            )
            record_metric(f"{prefix}/memory_peak_max_gb", peak_mem, ReductionType.MAX)
            _set_memory_active(False)
            torch.cuda.reset_max_memory_allocated()


@contextmanager
def _timer_cm(prefix: str, track_time: bool, use_cuda: bool):
    """
    Sync context manager for measuring execution time.
    Used internally by decorators and async context managers for timing measurements.
    """

    use_cuda_events = (
        os.getenv(METRIC_TIMER_USES_CUDA, str(use_cuda)).lower() == "true"
    ) and torch.cuda.is_available()
    use_cpu = not use_cuda_events
    timer: _TimerProtocol = _TimerCPU() if use_cpu else _TimerCUDA()

    if track_time:
        timer.start()

    try:
        yield
    finally:
        if track_time:
            timer.step("duration")  # Capture full time from start to here
            durations = timer.get_all_durations()
            # Single block: one duration, record directly
            if durations:
                _, d_ms = durations[0]
                d_s = d_ms / 1000.0
                record_metric(f"{prefix}/duration_avg_s", d_s, ReductionType.MEAN)
                record_metric(f"{prefix}/duration_max_s", d_s, ReductionType.MAX)

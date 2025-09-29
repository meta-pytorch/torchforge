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
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import List, Optional, Protocol, Tuple

import torch

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
    Local env flag METRIC_TIMER_USES_GPU can be used to set CUDA timing.

    Args:
        prefix (str): Prefix for metric names, e.g. "my_step" -> "my_step/{step_name}/duration_avg_s".
        use_gpu (bool): Sets CUDA timing if True and CUDA is available.

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

    def __init__(self, prefix: str, use_gpu: bool = False):
        self.prefix = prefix
        use_gpu_events = (
            os.getenv("METRIC_TIMER_USES_GPU", str(use_gpu)).lower() == "true"
        ) and torch.cuda.is_available()
        use_cpu = not use_gpu_events
        self.timer: _TimerProtocol = _TimerCPU() if use_cpu else _TimerCUDA()
        self._active = False

    def start(self) -> None:
        if self._active:
            raise ValueError("Timer has already been started")
        self._active = True
        self.timer.start()

    def step(self, step_name: str) -> None:
        if not self._active:
            raise ValueError("Timer must be started before calling step")
        self.timer.step(step_name)

    def end(self) -> None:
        if not self._active:
            raise ValueError("Timer must be started before calling end")
        self.timer.step("end")  # dropped from steps, included in total
        self.timer.wait_for_all()  # only needed for CUDA, no-op for CPU
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

    def wait_for_all(self) -> None:
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

    def wait_for_all(self) -> None:
        """No threads for CPU timing - nothing to wait for."""
        pass

    def get_all_durations(self) -> List[Tuple[str, float]]:
        return self._durations[:]


class _TimerCUDA(_TimerProtocol):
    """
    CUDA timing backend with non-blocking events and polling threads.
    Uses background threads to poll CUDA events without blocking main thread,
    necessary because CUDA event queries can be expensive and we want to avoid
    torch.cuda.synchronize() calls during timing operations.

    Reference: https://github.com/meta-pytorch/monarch/blob/main/python/monarch/timer/execution_timer.py
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for timing")
        self._durations: List[Tuple[str, float]] = []
        self._threads: List[threading.Thread] = []
        self._lock = (
            threading.Lock()
        )  # Protects _durations and _threads from concurrent access
        self._chain_start: Optional[torch.cuda.Event] = None

    def start(self) -> None:
        # Reset state for reuse
        self._durations = []
        with self._lock:
            self._threads = []
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        self._chain_start = start_event

    def step(self, name: str) -> None:
        if self._chain_start is None:
            raise ValueError("Timer must be started before calling step")

        # Pre-allocate slot with placeholder to preserve ordering when pooling
        with self._lock:
            idx = len(self._durations)
            self._durations.append((name, -1.0))  # Placeholder, -1 indicates pending

        stream = torch.cuda.current_stream()
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(stream)

        thread = threading.Thread(
            daemon=True,  # Use daemon threads: auto-terminate on process exit to prevent leaks/hangs.
            # Polls are short-lived and non-critical; explicit end() joins fully anyway.
            target=self._poll_and_store,
            args=(idx, name, self._chain_start, end_event),
        )
        thread.start()

        with self._lock:
            self._threads.append(thread)

        # Clean up completed threads to prevent accumulation
        self._join_completed_threads()

        self._chain_start = end_event

    def _poll_and_store(
        self,
        idx: int,
        name: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
    ) -> None:
        """
        Background thread function that polls CUDA events without blocking.
        Necessary to avoid torch.cuda.synchronize() calls which would stall GPU work.
        Polls every 0.01 seconds until event completes, then stores the timing result.
        """
        while not end_event.query():
            time.sleep(0.01)

        elapsed_ms = start_event.elapsed_time(end_event)
        with self._lock:
            self._durations[idx] = (name, elapsed_ms)

    def _join_completed_threads(self) -> None:
        """Clean up completed threads to prevent accumulation."""
        with self._lock:
            still_active = []
            for t in self._threads:
                if not t.is_alive():
                    t.join()  # Safe since thread is already done
                else:
                    still_active.append(t)
            self._threads = still_active

    def wait_for_all(self) -> None:
        """
        Join all background polling threads to ensure timing measurements are complete.
        This is CPU-bound waiting (joining threads) and doesn't block GPU operations.
        Called before reading final timing results to ensure all measurements are available.
        """
        with self._lock:
            threads = self._threads[:]

        for t in threads:
            t.join()

        with self._lock:
            self._threads.clear()

    def get_all_durations(self) -> List[Tuple[str, float]]:
        with self._lock:
            # Check for unfinished timing (defensive programming)
            for name, duration in self._durations:
                if duration < 0:
                    raise RuntimeError(f"Unfinished timing for {name}")
            return self._durations[:]


"""
=======================================
Memory+timer as decorator / ctx manager
=======================================
"""


def record_perf_metrics(
    prefix: str,
    track_time: bool = True,
    track_memory: bool = False,
    use_gpu: bool = False,
):
    """
    Decorator for functions with performance tracking, supporting both sync and async functions by detecting coroutine status.
    Tracks time and/or memory if enabled, recording metrics via record_metric. Skips if DISABLE_PERF_METRICS env flag is true.

    Args:
        prefix (str): Prefix for metric names
        track_time (bool): Whether to track execution time. Defaults to True.
        track_memory (bool): Whether to track CUDA memory usage. Defaults to False.
        use_gpu (bool): Whether to use CUDA events for timing (overridden by METRIC_TIMER_USES_GPU env var).
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
                if os.getenv("DISABLE_PERF_METRICS", "false").lower() == "true":
                    return await func(*args, **kwargs)
                with _memory_tracking_cm(prefix, track_memory), _timer_cm(
                    prefix, track_time, use_gpu
                ):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if os.getenv("DISABLE_PERF_METRICS", "false").lower() == "true":
                    return func(*args, **kwargs)
                with _memory_tracking_cm(prefix, track_memory), _timer_cm(
                    prefix, track_time, use_gpu
                ):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


@contextmanager
def record_perf_metrics_ctx(
    prefix: str,
    track_time: bool = True,
    track_memory: bool = False,
    use_gpu: bool = False,
):
    """
    Context manager for performance tracking in a code block, combining time and memory metrics.
    Safe for use in async code; skips if DISABLE_PERF_METRICS env flag is true.

    Args:
        prefix (str): Prefix for metric names
        track_time (bool): Whether to track execution time. Defaults to True.
        track_memory (bool): Whether to track CUDA memory usage. Defaults to False.
        use_gpu (bool): Whether to use CUDA events for timing (overridden by METRIC_TIMER_USES_GPU env var). Defaults to False.

    Example:
       with record_perf_metrics_ctx("my_block", track_time=True, track_memory=True):
            await some_task()
            some_other_task()
    """

    if os.getenv("DISABLE_PERF_METRICS", "false").lower() == "true":
        yield
        return

    with _memory_tracking_cm(prefix, track_memory), _timer_cm(
        prefix, track_time, use_gpu
    ):
        yield


@contextmanager
def _memory_tracking_cm(prefix: str, track_memory: bool):
    """
    Context manager for tracking CUDA's memory delta (peak - start) and peak during execution of a function
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
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            delta = (peak_mem * 1024**3 - start_mem) / 1024**3
            record_metric(
                f"{prefix}/memory_delta_peak_start_avg_gb", delta, ReductionType.MEAN
            )
            record_metric(f"{prefix}/memory_peak_max_gb", peak_mem, ReductionType.MAX)
            _set_memory_active(False)
            torch.cuda.reset_max_memory_allocated()


@contextmanager
def _timer_cm(prefix: str, track_time: bool, use_gpu: bool):
    """
    Sync context manager for measuring execution time.
    Used internally by decorators and async context managers for timing measurements.
    """

    use_gpu_events = (
        os.getenv("METRIC_TIMER_USES_GPU", str(use_gpu)).lower() == "true"
    ) and torch.cuda.is_available()
    use_cpu = not use_gpu_events
    timer: _TimerProtocol = _TimerCUDA() if use_gpu_events else _TimerCPU()

    if track_time:
        timer.start()

    try:
        yield
    finally:
        if track_time:
            timer.step("duration")  # Capture full time from start to here
            timer.wait_for_all()
            durations = timer.get_all_durations()
            # Single block: one duration, record directly
            if durations:
                _, d_ms = durations[0]
                d_s = d_ms / 1000.0
                record_metric(f"{prefix}/duration_avg_s", d_s, ReductionType.MEAN)
                record_metric(f"{prefix}/duration_max_s", d_s, ReductionType.MAX)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Protocol

import torch

from forge.env import DISABLE_PERF_METRICS, FORGE_DISABLE_METRICS, METRIC_TIMER_USES_GPU
from forge.observability.metrics import record_metric, Reduce

logger = logging.getLogger(__name__)

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

class Tracer:
==========
"""


class Tracer:
    """
    Tracer with multi-step timing and optional memory tracking at start/stop boundaries.
    Steps only affect timing; memory is tracked from start() to stop().

    Supports non-blocking CUDA timing via CUDA events and background polling threads.
    Aggregation is handled externally by the metrics system via record_metric.

    User must call start() and stop() explicitly.
    Supports reuse: after calling stop(), you may call start() again to begin a new timing session.

    Local env flag DISABLE_PERF_METRICS can be used to skip all timing operations.
    Local env flag METRIC_TIMER_USES_GPU can be used to set CUDA timing.

    Args:
        prefix (str): Prefix for metric names, e.g. "my_prefix" -> "{my_prefix}/{step_name}/duration_avg_s".
        track_memory (bool): Whether to track CUDA memory usage. Defaults to False.
        timer (str): Timing backend; "cpu" (default) or "gpu".

    Example:
        tracer = Tracer("my_prefix", track_memory=True, timer="gpu")
        tracer.start()  # Memory tracking starts here
        time.sleep(0.1)  # Work for step "a"
        tracer.step("my_step_a")  # Only affects timing
        for i in range(1, 4):  # 3 iterations
            time.sleep(i/10)  # 0.1, 0.2, 0.3 seconds
            tracer.step("my_step_b")  # Only affects timing
        tracer.stop()  # Memory tracking ends here, records metrics

        >>> Records:
        >>> my_prefix/my_step_a/duration_avg_s: 0.1
        >>> my_prefix/my_step_a/duration_max_s: 0.1
        >>> my_prefix/my_step_b/duration_avg_s: 0.2 # Average of 0.1, 0.2, 0.3
        >>> my_prefix/my_step_b/duration_max_s: 0.3 # Max of 0.1, 0.2, 0.3
        >>> my_prefix/total_duration_avg_s: 0.7 # Total: 0.1 + 0.1 + 0.2 + 0.3
        >>> my_prefix/total_duration_max_s: 0.7
        >>> my_prefix/memory_delta_end_start_avg_gb: 0.5 # Memory delta
        >>> my_prefix/memory_peak_max_gb: 1.2 # Memory peak

        # Can reuse the same tracer after stop()
        tracer.start()  # Begin new session
        tracer.step("step1")
        tracer.stop()
    """

    def __init__(
        self,
        prefix: str,
        track_memory: bool = False,
        timer: str = "cpu",  # "cpu" or "gpu"
    ):
        if timer not in ("cpu", "gpu"):
            raise ValueError('timer must be "cpu" or "gpu"')

        self.prefix = prefix
        self.track_memory = track_memory
        self.time_with_gpu = timer == "gpu"
        self._disable = (
            DISABLE_PERF_METRICS.get_value() or FORGE_DISABLE_METRICS.get_value()
        )
        self._active = False

        # Timing state
        self._timer: _TimerProtocol | None = None

        # Memory tracking state
        self._memory_started = False
        self._start_mem = 0.0

    def start(self) -> None:
        if self._disable:
            return
        if self._active:
            raise ValueError("Tracer has already been started")

        # Start timing (always enabled)

        # TODO - follow up on if this env var behavior makes sense.
        # METRIC_TIMER_USES_GPU env var overrides the timer parameter
        metric_timer_uses_gpu = METRIC_TIMER_USES_GPU.get_value()
        if metric_timer_uses_gpu is not None:
            # Env var is set - convert string to bool if needed
            if isinstance(metric_timer_uses_gpu, str):
                use_gpu = metric_timer_uses_gpu.lower() in ("true", "1", "yes")
            else:
                use_gpu = bool(metric_timer_uses_gpu)
        else:
            # Env var not set - use the timer parameter
            use_gpu = self.time_with_gpu
        time_with_gpu_events = use_gpu and torch.cuda.is_available()
        self._timer = _TimerCUDA() if time_with_gpu_events else _TimerCPU()
        self._timer.start()

        self._active = True

        # Start memory tracking
        if self.track_memory:
            self._start_memory_tracking()

    def step(self, step_name: str) -> None:
        """Record a timing step. Does not affect memory tracking."""
        if self._disable:
            return
        if not self._active:
            raise ValueError("Tracer must be started before calling step")
        self._timer.step(step_name)  # pyre-ignore

    def stop(self) -> None:
        if self._disable:
            return
        if not self._active:
            raise ValueError("Tracer must be started before calling stop")

        # Stop timing
        durations, stop_step_ms = self._timer.get_all_durations()  # pyre-ignore
        self._record_timing_metrics(durations, stop_step_ms)
        self._timer = None

        # Stop memory tracking
        if self._memory_started:
            self._stop_memory_tracking()

        self._active = False

    def _start_memory_tracking(self) -> None:
        is_outer_scope = not _is_memory_active()
        should_track = (
            self.track_memory and is_outer_scope and torch.cuda.is_available()
        )

        if self.track_memory and not is_outer_scope:
            _warn_nested_memory_tracking(self.prefix)
            return

        if should_track:
            _set_memory_active(True)
            torch.cuda.reset_peak_memory_stats()
            self._start_mem = torch.cuda.memory_allocated()
            self._memory_started = True

    def _stop_memory_tracking(self) -> None:
        if not self._memory_started:
            return

        end_mem = torch.cuda.memory_allocated()
        delta = (end_mem - self._start_mem) / 1024**3
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        record_metric(
            f"{self.prefix}/memory_delta_end_start_avg_gb", delta, Reduce.MEAN
        )
        record_metric(f"{self.prefix}/memory_peak_max_gb", peak_mem, Reduce.MAX)
        _set_memory_active(False)
        torch.cuda.reset_peak_memory_stats()
        self._memory_started = False

    def _record_timing_metrics(
        self, durations: list[tuple[str, float]], stop_step_ms: float
    ) -> None:
        total_ms = sum(d_ms for _, d_ms in durations) + stop_step_ms
        total_s = total_ms / 1000.0
        record_metric(f"{self.prefix}/total_duration_avg_s", total_s, Reduce.MEAN)
        record_metric(f"{self.prefix}/total_duration_max_s", total_s, Reduce.MAX)

        for name, d_ms in durations:
            d_s = d_ms / 1000.0
            record_metric(f"{self.prefix}/{name}/duration_avg_s", d_s, Reduce.MEAN)
            record_metric(f"{self.prefix}/{name}/duration_max_s", d_s, Reduce.MAX)


class _TimerProtocol(Protocol):
    def start(self) -> None: ...

    def step(self, name: str) -> None: ...

    def get_all_durations(self) -> tuple[list[tuple[str, float]], float]: ...


class _TimerCPU(_TimerProtocol):
    """
    CPU timing backend using perf_counter.
    """

    def __init__(self) -> None:
        self._durations: list[tuple[str, float]] = []
        self._chain_start: float | None = None

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

    def get_all_durations(self) -> tuple[list[tuple[str, float]], float]:
        """Retrieve list of (step_name, duration) tuples and last step duration
        between tracer.stop and the last step (or start if none)."""
        stop_step_ms = 0.0
        if self._chain_start is not None:
            now = time.perf_counter()
            stop_step_ms = (now - self._chain_start) * 1000
        return self._durations[:], stop_step_ms


class _TimerCUDA(_TimerProtocol):
    """CUDA timing backend with non-blocking events and futures.
    Uses a thread pool to poll CUDA events asynchronously without blocking the main thread.

    Example:
        timer = _TimerCUDA()
        timer.start()
        # torch.mm(a, b)  # ~100ms GPU
        timer.step("matmul")
        # torch.mm(c, d)  # ~200ms
        durs_steps, stop_step_ms = timer.get_all_durations()  # ([( "matmul", 100 )], 200)
    """

    def __init__(self, max_workers: int = 2) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for timing")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[tuple[str, Future[float], int]] = (
            []
        )  # (name, future, submission_index)
        self._durations: list[tuple[str, float]] = []
        self._chain_start: torch.cuda.Event | None = None

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
        if self._chain_start is None:
            raise ValueError("Timer must be started before calling step")

        stream = torch.cuda.current_stream()
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(stream)

        future = self._executor.submit(self._poll_elapsed, self._chain_start, end_event)
        index = len(self._futures)
        self._futures.append((name, future, index))
        if len(self._futures) >= 5:  # clean up every 5
            self._collect_completed_futures()

        self._chain_start = end_event

    def _poll_elapsed(
        self, start_event: torch.cuda.Event, end_event: torch.cuda.Event
    ) -> float:
        """Compute elapsed time after polling with backoff."""
        # Poll until ready
        sleep_time = 0.001  # Start at 1ms
        while not end_event.query():
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 1.5, 0.05)  # Backoff, cap at 50ms
        return start_event.elapsed_time(end_event)

    def _collect_completed_futures(self, wait_till_done: bool = False) -> None:
        """Drain done futures to avoid memory leak; update durations in submission order."""
        still_pending = []
        for name, future, idx in self._futures:
            if future.done() or wait_till_done:
                dur = future.result()
                self._durations.append((name, dur))
            else:
                still_pending.append((name, future, idx))

        self._futures = still_pending

    def get_all_durations(self) -> tuple[list[tuple[str, float]], float]:
        """Retrieve list of (step_name, duration) tuples and last step duration
        between tracer.stop and the last step (or start if none). Order of tuples is random.
        """
        # Final timing since last step (or start) until this function is called
        stop_step = f"_stop_step_{id(self)}"
        self.step(stop_step)

        # Wait on remaining futures
        self._collect_completed_futures(wait_till_done=True)
        self._futures.clear()

        # Extract stop_step_ms
        stop_step_ms = 0.0
        durations = [
            (name, duration) for name, duration in self._durations if name != stop_step
        ]
        for name, duration in self._durations:
            if name == stop_step:
                stop_step_ms = duration
                break

        return durations, stop_step_ms

    def __del__(self) -> None:
        # Fallback cleanup in finalizer
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            return


"""
=======================================
Memory+timer as decorator / ctx manager
=======================================
"""


def trace(
    prefix: str,
    track_memory: bool = False,
    timer: str = "cpu",  # "cpu" or "gpu"
):
    """
    Dual-purpose: Decorator or context manager for performance tracking.
    Uses Tracer internally for both modes to ensure consistency.

    Decorator mode: Simple single-block tracking (no steps available)
    Context manager mode: Multi-step tracking via returned Tracer object

    Args:
        prefix (str): Prefix for metric names
        track_memory (bool): Whether to track CUDA memory usage. Defaults to False.
        timer (str): Timing backend; "cpu" (default) or "gpu" (requires CUDA).

    Decorator Examples:
        @trace("my_prefix", track_memory=True, timer="gpu")
        async def my_async_func():
            pass

        @trace("my_prefix", track_memory=True, timer="gpu")
        def my_sync_func():
            pass

    Context Manager Example:
        with trace("my_block", track_memory=True, timer="gpu") as tracer:
            tracer.step("fwd")  # Optional: mark steps
            await some_task()
            tracer.step("bwd")  # Optional
            some_other_task()
            # tracer.stop() called automatically on exit
    """
    if timer not in ("cpu", "gpu"):
        raise ValueError('timer must be "cpu" or "gpu"')

    class _Dual:
        def __call__(self, func):
            """Decorator mode: Use Tracer internally (single-block, no steps exposed)"""
            if inspect.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    tracer = Tracer(prefix, track_memory=track_memory, timer=timer)
                    tracer.start()
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        tracer.stop()  # Single block - no steps called

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    tracer = Tracer(prefix, track_memory=track_memory, timer=timer)
                    tracer.start()
                    try:
                        return func(*args, **kwargs)
                    finally:
                        tracer.stop()  # Single block - no steps called

                return sync_wrapper

        def __enter__(self):
            """Context manager mode: Return Tracer for steps"""
            self._tracer = Tracer(prefix, track_memory=track_memory, timer=timer)
            self._tracer.start()
            return self._tracer

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Context manager cleanup: stop the tracer"""
            if hasattr(self, "_tracer"):
                self._tracer.stop()

    return _Dual()

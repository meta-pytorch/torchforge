# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import time
from contextlib import contextmanager
from typing import List, Tuple
from unittest.mock import Mock, patch

import pytest
import torch
from forge.observability.metrics import ReductionType

from forge.observability.perf_tracker import (
    _TimerCPU,
    _TimerCUDA,
    record_perf_metrics,
    record_perf_metrics_ctx,
    Timer,
)


@pytest.fixture
def mock_record_metric_calls(monkeypatch):
    """Mock record_metric that tracks all calls."""
    calls: List[Tuple[str, float, ReductionType]] = []

    def mock_record_metric(name, val, red):
        calls.append((name, val, red))

    monkeypatch.setattr(
        "forge.observability.perf_tracker.record_metric",
        Mock(side_effect=mock_record_metric),
    )
    return calls


@contextmanager
def mock_cuda_memory():
    """Mock CUDA memory with 1GB start, 2GB end, 3GB peak."""
    gb_bytes = 1024**3
    with patch.multiple(
        "torch.cuda",
        is_available=Mock(return_value=True),
        memory_allocated=Mock(side_effect=[gb_bytes, 2 * gb_bytes]),
        max_memory_allocated=Mock(return_value=3 * gb_bytes),
        reset_max_memory_allocated=Mock(),
    ):
        yield


def assert_metrics_recorded(calls, expected_metrics):
    """Assert metrics by pattern/value, with reduction checks and count for multiples."""
    recorded = {name: (val, red) for name, val, red in calls}
    for pattern, exp_val in expected_metrics.items():
        # Handle special count checks
        if pattern == "repeated_count":
            actual_count = len([c for c in calls if "b/duration" in c[0]])
            assert (
                actual_count == exp_val
            ), f"Expected {exp_val} repeated calls, got {actual_count}"
            continue

        matching = [n for n in recorded if pattern in n]
        assert len(matching) >= 1, f"No/insufficient matches for: {pattern}"
        actual_val, actual_red = recorded[matching[0]]
        assert actual_val == pytest.approx(
            exp_val, abs=0.02
        ), f"Expected {pattern}≈{exp_val}, got {actual_val}"
        # Reduction check
        if "_avg_" in pattern:
            assert actual_red == ReductionType.MEAN
        elif "_max_" in pattern:
            assert actual_red == ReductionType.MAX


class TestTimer:
    def setup_method(self, method):
        """Very first cuda call adds ~0.4s to test times, so warmup here."""
        if torch.cuda.is_available():
            # Mock record_metric for warmup since fixtures aren't available in setup_method
            with patch("forge.observability.perf_tracker.record_metric"):
                warmup_timer = Timer("cuda_warmup", use_cuda=True)
                warmup_timer.start()
                warmup_timer.step("init")  # Need a step before end()
                warmup_timer.end()

    @pytest.mark.parametrize("use_cuda", [False, True])
    def test_timer_comprehensive_workflow(
        self, use_cuda, mock_record_metric_calls, monkeypatch
    ):
        """Test Timer + async concurrency: a=~0.1s, b=[0.1,0.2,0.3], total=~0.8s"""
        if use_cuda and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monkeypatch.setenv("METRIC_TIMER_USES_CUDA", str(use_cuda))

        # Test concurrency: run two instances in parallel
        async def async_func_test(task_id: int):

            timer = Timer(f"example_{task_id}")
            timer.start()
            await asyncio.sleep(0.1)
            timer.step("a")

            for i in range(1, 4):  # i = 1, 2, 3
                await asyncio.sleep(i / 10.0)  # 0.1s, 0.2s, 0.3s
                timer.step("b")

            await asyncio.sleep(0.1)
            timer.end()
            return task_id

        async def run_workflow():
            start_time = time.perf_counter()
            results = await asyncio.gather(async_func_test(1), async_func_test(2))
            total_time = time.perf_counter() - start_time
            return results, total_time

        results, total_time = asyncio.run(run_workflow())

        # Test concurrency: should be ~1x (0.8s) not 2x (1.6s) if truly concurrent
        assert results == [1, 2]
        assert (
            total_time < 1.0  # Should be 0.8, avoid flaky tests
        ), f"Expected ~0.8s concurrent execution, got {total_time:.3f}s"

        # Verify metrics for both tasks
        metrics = {name: val for name, val, _ in mock_record_metric_calls}

        # Both tasks should have same timing pattern
        for task_id in [1, 2]:
            prefix = f"example_{task_id}"

            # Step "a" duration ~0.1s for each task
            assert metrics[f"{prefix}/a/duration_avg_s"] == pytest.approx(0.1, abs=0.02)
            assert metrics[f"{prefix}/a/duration_max_s"] == pytest.approx(0.1, abs=0.02)

            # Total duration ~0.8s for each task
            assert metrics[f"{prefix}/total_duration_avg_s"] == pytest.approx(
                0.8, abs=0.02
            )
            assert metrics[f"{prefix}/total_duration_max_s"] == pytest.approx(
                0.8, abs=0.02
            )

        # Step "b" should have 3 individual recordings per task: ~0.1s, ~0.2s, ~0.3s
        b_avg_values = [
            val
            for name, val, _ in mock_record_metric_calls
            if "/b/duration_avg_s" in name
        ]
        b_max_values = [
            val
            for name, val, _ in mock_record_metric_calls
            if "/b/duration_max_s" in name
        ]

        # Each task contributes [0.1, 0.2, 0.3], so we should see these patterns twice
        expected_b_pattern = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        assert sorted(b_avg_values) == pytest.approx(
            sorted(expected_b_pattern), abs=0.005
        )
        assert sorted(b_max_values) == pytest.approx(
            sorted(expected_b_pattern), abs=0.005
        )

    @pytest.mark.parametrize(
        "error_case",
        [
            ("step_before_start", lambda t: t.step("x")),
            ("end_before_start", lambda t: t.end()),
            ("double_start", lambda t: (t.start(), t.start())),
        ],
    )
    def test_error_conditions(self, error_case):
        """User errors should raise ValueError, e.g. step before start."""
        timer = Timer("test")
        _, action = error_case
        with pytest.raises(ValueError):
            action(timer)


class TestMemoryTracking:
    """Memory tracking functionality tests."""

    def test_memory_tracking_and_context_manager(self, mock_record_metric_calls):
        """Test delta/peak via ctx."""
        with mock_cuda_memory():
            with record_perf_metrics_ctx(
                "mem_test", track_time=False, track_memory=True
            ):
                pass

        assert_metrics_recorded(
            mock_record_metric_calls,
            {
                "memory_delta_end_start_avg_gb": 1.0,  # (2GB - 1GB)
                "memory_peak_max_gb": 3.0,  # 3GB peak
            },
        )

    def test_nested_memory_tracking_warning(self, caplog, mock_record_metric_calls):
        """Test nesting skips with warning, called only once per prefix due to @lru_cache."""

        def test_prefix_warns_once(outer_prefix: str, inner_prefix: str):
            """Helper: nest inner_prefix multiple times, should warn only once."""
            with mock_cuda_memory():
                with record_perf_metrics_ctx(outer_prefix, track_memory=True):
                    for _ in range(3):  # Nest 3 times with same prefix
                        with record_perf_metrics_ctx(inner_prefix, track_memory=True):
                            pass

        with caplog.at_level("WARNING"):
            test_prefix_warns_once("outer1", "prefix1")
            test_prefix_warns_once("outer2", "prefix2")

        # Should have exactly 2 warnings: one for each unique prefix
        warning_lines = [
            line
            for line in caplog.text.split("\n")
            if "Nested memory tracking detected" in line
        ]
        assert (
            len(warning_lines) == 2
        ), f"Expected exactly 2 warnings, got {len(warning_lines)}: {warning_lines}"
        assert "Nested memory tracking detected in prefix1" in caplog.text
        assert "Nested memory tracking detected in prefix2" in caplog.text


class TestDecoratorAPI:
    """Decorator and context manager API tests."""

    @pytest.mark.parametrize("track_time,track_memory", [(True, False), (False, True)])
    def test_decorator_parameter_combinations(
        self, track_time, track_memory, mock_record_metric_calls
    ):
        """Test track_time/track_memory combos."""
        with mock_cuda_memory():

            @record_perf_metrics(
                "param_test", track_time=track_time, track_memory=track_memory
            )
            def test_func():
                import time

                time.sleep(0.01)
                return "success"

            result = test_func()
            assert result == "success"

            if track_time:
                assert_metrics_recorded(
                    mock_record_metric_calls, {"duration_avg_s": 0.01}
                )
            if track_memory:
                assert_metrics_recorded(
                    mock_record_metric_calls,
                    {
                        "memory_delta_end_start_avg_gb": 1.0,
                        "memory_peak_max_gb": 3.0,
                    },
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_gather", [False, True])
    async def test_async_decorator(self, use_gather, mock_record_metric_calls):
        """Test async, including concurrency via gather."""

        @record_perf_metrics("async_test")
        async def async_func(task_id=1):
            await asyncio.sleep(0.02)
            return task_id

        if use_gather:
            start = time.perf_counter()
            results = await asyncio.gather(async_func(1), async_func(2))
            total = time.perf_counter() - start
            assert total == pytest.approx(0.02, abs=0.01)  # Concurrent timing
            assert results == [1, 2]
            # Should have 2 tasks x 2 metrics each
            assert len(mock_record_metric_calls) == 4
        else:
            result = await async_func()
            assert result == 1
            assert_metrics_recorded(mock_record_metric_calls, {"duration_avg_s": 0.02})

    def test_context_manager(self, mock_record_metric_calls):
        """Test ctx."""
        import time

        with record_perf_metrics_ctx("ctx_test"):
            time.sleep(0.01)
        assert_metrics_recorded(mock_record_metric_calls, {"duration_avg_s": 0.01})


class TestEnvironmentConfiguration:
    """Env config tests."""

    def test_disable_all_metrics(self, monkeypatch, mock_record_metric_calls):
        """Test skip for dec and ctx."""
        monkeypatch.setenv("DISABLE_PERF_METRICS", "true")

        @record_perf_metrics("disabled_test")
        def func():
            return "result"

        assert func() == "result"
        with record_perf_metrics_ctx("disabled_ctx"):
            pass
        assert not mock_record_metric_calls

    def test_env_override_backend_selection(self, monkeypatch):
        """Test METRIC_TIMER_USES_CUDA overrides use_cuda parameter."""
        with patch("torch.cuda.is_available", return_value=True):
            # Default: should use CPU when use_cuda=False
            timer1 = Timer("test", use_cuda=False)
            assert isinstance(timer1.timer, _TimerCPU)

            # Env override: should use CUDA despite use_cuda=False
            monkeypatch.setenv("METRIC_TIMER_USES_CUDA", "true")
            timer2 = Timer("test", use_cuda=False)
            assert isinstance(timer2.timer, _TimerCUDA)

            # Env override: should use CPU despite use_cuda=True
            monkeypatch.setenv("METRIC_TIMER_USES_CUDA", "false")
            timer3 = Timer("test", use_cuda=True)
            assert isinstance(timer3.timer, _TimerCPU)

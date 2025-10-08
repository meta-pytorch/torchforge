# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import time
from contextlib import contextmanager
from typing import List, Literal, Tuple, Union
from unittest.mock import Mock, patch

import pytest
import torch
from forge.env_constants import DISABLE_PERF_METRICS, METRIC_TIMER_USES_GPU
from forge.observability.metrics import Reduce

from forge.observability.perf_tracker import _TimerCPU, _TimerCUDA, trace, Tracer


@pytest.fixture
def mock_record_metric_calls(monkeypatch):
    """Mock record_metric that tracks all calls."""
    calls: List[Tuple[str, float, Reduce]] = []

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


def assert_metrics_dict_matches(calls, expected_metrics):
    """Assert recorded metrics match expected dict."""
    actual_metrics = {name: val for name, val, _ in calls}

    for metric_name, expected_val in expected_metrics.items():
        assert metric_name in actual_metrics, f"Missing metric: {metric_name}"
        actual_val = actual_metrics[metric_name]
        assert actual_val == pytest.approx(
            expected_val, rel=0.1  # 10% relative tolerance for timing tests
        ), f"Expected {metric_name}={expected_val}, got {actual_val}"


class TracingModes:
    """Utility to execute the same workflow in different tracing modes."""

    @staticmethod
    async def run_workflow(
        mode: str, prefix: str, track_memory=False, timer="cpu"
    ) -> Union[
        Literal["direct_done"], Literal["decorator_done"], Literal["context_done"]
    ]:
        """Run the comprehensive test workflow: a=~0.05s, b=[0.05,0.1,0.15], total=~0.4s"""

        if mode == "direct":
            # Direct Tracer usage
            tracer = Tracer(prefix, track_memory=track_memory, timer=timer)
            tracer.start()
            await asyncio.sleep(0.05)
            tracer.step("a")

            for i in range(1, 4):  # i = 1, 2, 3
                await asyncio.sleep(i * 0.05)  # 0.05s, 0.1s, 0.15s
                tracer.step("b")

            await asyncio.sleep(0.05)
            tracer.stop()
            return "direct_done"

        elif mode == "decorator":
            # Decorator usage (no steps available)
            @trace(prefix, track_memory=track_memory, timer=timer)
            async def decorated_workflow():
                await asyncio.sleep(0.05)  # step "a" equivalent
                for i in range(1, 4):  # step "b" iterations
                    await asyncio.sleep(i * 0.05)
                await asyncio.sleep(0.05)  # final timing
                return "decorator_done"

            return await decorated_workflow()

        elif mode == "context":
            # Context manager usage with steps
            with trace(prefix, track_memory=track_memory, timer=timer) as tracer:
                await asyncio.sleep(0.05)
                tracer.step("a")

                for i in range(1, 4):
                    await asyncio.sleep(i * 0.05)
                    tracer.step("b")

                await asyncio.sleep(0.05)
                return "context_done"

        else:
            raise ValueError(f"Invalid mode: {mode}")


class TestTracingModes:
    """Test all tracing modes with comprehensive workflows."""

    def setup_method(self, method):
        """CUDA warmup to avoid ~0.4s first-call delay in tests."""
        if torch.cuda.is_available():
            with patch("forge.observability.perf_tracker.record_metric"):
                warmup_tracer = Tracer("cuda_warmup", timer="gpu")
                warmup_tracer.start()
                warmup_tracer.step("init")
                warmup_tracer.stop()

    @pytest.mark.parametrize("mode", ["direct", "decorator", "context"])
    @pytest.mark.parametrize("timer", ["cpu", "gpu"])
    def test_comprehensive_workflow(
        self, mode, timer, mock_record_metric_calls, monkeypatch
    ):
        """Test comprehensive workflow: timing + concurrency across all modes."""
        if timer == "gpu" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monkeypatch.setenv(METRIC_TIMER_USES_GPU, str(timer == "gpu"))

        async def run_concurrent_tasks():
            start_time = time.perf_counter()
            results = await asyncio.gather(
                TracingModes.run_workflow(mode, f"task1_{mode}", timer=timer),
                TracingModes.run_workflow(mode, f"task2_{mode}", timer=timer),
            )
            total_time = time.perf_counter() - start_time
            return results, total_time

        results, total_time = asyncio.run(run_concurrent_tasks())

        # Test concurrency: should be ~1x (0.4s) not 2x (0.8s) if truly concurrent
        assert results[0] == f"{mode}_done"
        assert results[1] == f"{mode}_done"
        assert (
            total_time < 0.5
        ), f"Expected ~0.4s concurrent execution, got {total_time:.3f}s"

        # Verify backend selection
        if mode == "direct":
            tracer = Tracer("backend_test", timer=timer)
            tracer.start()
            if timer == "gpu" and torch.cuda.is_available():
                assert isinstance(tracer._timer, _TimerCUDA), "Expected CUDA timer"
            else:
                assert isinstance(tracer._timer, _TimerCPU), "Expected CPU timer"
            tracer.step("backend_check")
            tracer.stop()

        # Verify expected metrics based on mode
        if mode in ["direct", "context"]:
            # These modes support steps - should have step and total metrics
            expected_metrics = {
                f"task1_{mode}/a/duration_avg_s": 0.05,
                f"task1_{mode}/a/duration_max_s": 0.05,
                f"task2_{mode}/a/duration_avg_s": 0.05,
                f"task2_{mode}/a/duration_max_s": 0.05,
                f"task1_{mode}/total_duration_avg_s": 0.4,
                f"task2_{mode}/total_duration_avg_s": 0.4,
            }
            # Should also have 3 "b" steps per task (6 total avg + 6 max = 12 "b" metrics)
            b_metrics = [
                name for name, _, _ in mock_record_metric_calls if "/b/duration" in name
            ]
            assert (
                len(b_metrics) == 12
            ), f"Expected 12 'b' metrics, got {len(b_metrics)}"
        else:  # decorator mode
            # Decorator mode only has total duration (no steps)
            expected_metrics = {
                f"task1_{mode}/total_duration_avg_s": 0.4,
                f"task1_{mode}/total_duration_max_s": 0.4,
                f"task2_{mode}/total_duration_avg_s": 0.4,
                f"task2_{mode}/total_duration_max_s": 0.4,
            }

        assert_metrics_dict_matches(mock_record_metric_calls, expected_metrics)

    @pytest.mark.parametrize("mode", ["direct", "context"])
    def test_memory_tracking(self, mode, mock_record_metric_calls):
        """Test memory tracking across modes that support it."""

        async def memory_workflow():
            with mock_cuda_memory():
                return await TracingModes.run_workflow(
                    mode, f"mem_{mode}", track_memory=True
                )

        result = asyncio.run(memory_workflow())
        assert result == f"{mode}_done"

        # Should record both timing and memory metrics
        expected_metrics = {
            f"mem_{mode}/memory_delta_end_start_avg_gb": 1.0,  # 2GB - 1GB
            f"mem_{mode}/memory_peak_max_gb": 3.0,
        }
        assert_metrics_dict_matches(mock_record_metric_calls, expected_metrics)

    def test_nested_memory_tracking_warning(self, caplog, mock_record_metric_calls):
        """Test nested memory tracking logs warning once per prefix."""

        async def nested_workflow():
            with mock_cuda_memory():
                outer_tracer = Tracer("outer", track_memory=True)
                outer_tracer.start()

                # Inner tracer should warn
                inner_result = await TracingModes.run_workflow(
                    "direct", "inner", track_memory=True
                )

                outer_tracer.step("outer_step")
                outer_tracer.stop()
                return inner_result

        with caplog.at_level("WARNING"):
            result = asyncio.run(nested_workflow())

        assert result == "direct_done"
        assert "Nested memory tracking detected in inner" in caplog.text

        # Only outer tracer should record memory metrics
        memory_metrics = [
            name for name, _, _ in mock_record_metric_calls if "memory_" in name
        ]
        assert all(
            "outer/" in m for m in memory_metrics
        ), "Only outer should track memory"


class TestErrorConditionsAndCompatibility:
    """Test error conditions, reuse, and timer backend compatibility."""

    @pytest.mark.parametrize(
        "error_case,action",
        [
            ("step_before_start", lambda t: t.step("x")),
            ("stop_before_start", lambda t: t.stop()),
            ("double_start", lambda t: (t.start(), t.start())),
        ],
    )
    def test_tracer_error_conditions(self, error_case, action):
        """Test tracer error conditions raise appropriate errors."""
        tracer = Tracer("test_error")

        if error_case == "double_start":
            tracer.start()  # First start is valid

        with pytest.raises(ValueError):
            action(tracer)

    def test_timer_parameter_validation(self):
        """Test that invalid timer parameter values raise ValueError."""
        with pytest.raises(ValueError, match='timer must be "cpu" or "gpu"'):
            Tracer("test", timer="invalid")

        with pytest.raises(ValueError, match='timer must be "cpu" or "gpu"'):
            trace("test", timer="invalid")

        # Valid values should work
        tracer_cpu = Tracer("test", timer="cpu")
        tracer_cuda = Tracer("test", timer="gpu")
        assert tracer_cpu is not None
        assert tracer_cuda is not None

    def test_tracer_and_timer_reuse(self, mock_record_metric_calls):
        """Test both tracer and timer backends can be reused."""
        # Test Tracer reuse
        tracer = Tracer("test_reuse")

        # First session
        tracer.start()
        time.sleep(0.005)
        tracer.step("session1")
        tracer.stop()

        # Second session (should work)
        tracer.start()
        time.sleep(0.01)
        tracer.step("session2")
        tracer.stop()

        # Verify both sessions recorded metrics
        metrics = [name for name, _, _ in mock_record_metric_calls]
        assert any("session1" in m for m in metrics)
        assert any("session2" in m for m in metrics)

        # Test CPU timer reuse
        cpu_timer = _TimerCPU()
        cpu_timer.start()
        time.sleep(0.005)
        cpu_timer.step("cpu_step1")
        durations1 = cpu_timer.get_all_durations()

        cpu_timer.start()
        time.sleep(0.005)
        cpu_timer.step("cpu_step2")
        durations2 = cpu_timer.get_all_durations()

        assert len(durations1) == 1 and durations1[0][0] == "cpu_step1"
        assert len(durations2) == 1 and durations2[0][0] == "cpu_step2"

        # Test CUDA timer reuse (if available)
        if torch.cuda.is_available():
            cuda_timer = _TimerCUDA()
            cuda_timer.start()
            cuda_timer.step("cuda_step1")
            cuda_durations1 = cuda_timer.get_all_durations()

            cuda_timer.start()
            cuda_timer.step("cuda_step2")
            cuda_durations2 = cuda_timer.get_all_durations()

            assert len(cuda_durations1) == 1 and cuda_durations1[0][0] == "cuda_step1"
            assert len(cuda_durations2) == 1 and cuda_durations2[0][0] == "cuda_step2"

    def test_exception_handling_context_manager(self, mock_record_metric_calls):
        """Test context manager properly cleans up on exception."""
        with pytest.raises(ValueError, match="test exception"):
            with trace("ctx_exception") as tracer:
                time.sleep(0.01)
                tracer.step("before_error")
                raise ValueError("test exception")

        # Should still record metrics despite exception
        metrics = [name for name, _, _ in mock_record_metric_calls]
        assert any("before_error" in m for m in metrics)


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    @pytest.mark.parametrize("mode", ["direct", "decorator", "context"])
    def test_disable_perf_metrics_all_modes(
        self, mode, monkeypatch, mock_record_metric_calls
    ):
        """Test DISABLE_PERF_METRICS disables all modes."""
        monkeypatch.setenv(DISABLE_PERF_METRICS, "true")

        async def disabled_workflow():
            return await TracingModes.run_workflow(mode, f"disabled_{mode}")

        result = asyncio.run(disabled_workflow())
        assert result == f"{mode}_done"
        assert not mock_record_metric_calls, "Expected no metrics when disabled"

    @pytest.mark.parametrize(
        "env_value,expected_backend",
        [
            ("true", _TimerCUDA),
            ("false", _TimerCPU),
        ],
    )
    def test_metric_timer_uses_gpu_override(
        self, env_value, expected_backend, monkeypatch
    ):
        """Test METRIC_TIMER_USES_GPU env var overrides timer parameter."""
        if env_value == "true" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with patch("torch.cuda.is_available", return_value=True), patch(
            "forge.observability.perf_tracker.record_metric"
        ):
            monkeypatch.setenv(METRIC_TIMER_USES_GPU, env_value)

            # Test with timer="cpu" (should be overridden by env)
            tracer = Tracer("env_test", timer="cpu")
            tracer.start()

            assert isinstance(tracer._timer, expected_backend)

            tracer.step("env_step")
            tracer.stop()

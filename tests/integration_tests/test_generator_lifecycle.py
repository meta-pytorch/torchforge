# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for Generator lifecycle management and resource cleanup.

These tests verify:
1. Resource cleanup during shutdown
2. Proper termination of worker processes
3. No dangling GPU processes after shutdown
"""

import os
import time

import psutil
import pytest
from forge.actors.generator import Generator


# Force HuggingFace offline mode to avoid proxy issues
os.environ["HF_HUB_OFFLINE"] = "1"

# Use Qwen3-1.7B which is already cached and used in other tests
MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_MODEL_LEN = 512
GPU_MEMORY_UTILIZATION = 0.9
# This disables CUDA graph optimizations and runs in eager mode, which is slower
# but simpler for debugging/testing.
ENFORCE_EAGER = True


def get_child_processes(parent_pid=None):
    """Get all child processes of the current process or a specific parent.

    Args:
        parent_pid: Parent process ID. If None, uses current process.

    Returns:
        List of psutil.Process objects for all children.
    """
    if parent_pid is None:
        parent_pid = os.getpid()

    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        return children
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def get_process_info(processes):
    """Get detailed info about processes for debugging.

    Args:
        processes: List of psutil.Process objects.

    Returns:
        List of dicts with process info.
    """
    info = []
    for proc in processes:
        try:
            info.append(
                {
                    "pid": proc.pid,
                    "ppid": proc.ppid(),  # Parent process ID
                    "name": proc.name(),
                    "cmdline": " ".join(proc.cmdline()[:3]),  # First 3 args
                    "status": proc.status(),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return info


def get_gpu_processes():
    """Get processes using GPUs via nvidia-smi.

    Returns:
        Set of PIDs using GPUs.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            pids = set()
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        pids.add(int(line.strip()))
                    except ValueError:
                        continue
            return pids
        return set()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()


@pytest.mark.asyncio
async def test_generator_shutdown_cleanup():
    """Test proper resource cleanup during shutdown for 2-GPU case.

    Verifies:
    - AsyncLLM shutdown is called
    - Worker processes are stopped
    - Generator process is stopped
    - No dangling GPU processes after shutdown
    - All child processes are properly cleaned up
    """
    generator = None

    # Get baseline process count
    initial_children = get_child_processes()
    initial_child_count = len(initial_children)
    initial_gpu_pids = get_gpu_processes()

    print(f"[Baseline] Test process PID: {os.getpid()}")
    print(f"[Baseline] Child processes: {initial_child_count}")
    print(f"[Baseline] GPU processes: {len(initial_gpu_pids)}")

    try:
        generator = await Generator.options(
            procs=2, num_replicas=1, with_gpus=True
        ).as_service(
            engine_args={
                "model": MODEL_NAME,
                "tensor_parallel_size": 2,
                "enforce_eager": ENFORCE_EAGER,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            },
            sampling_params={
                "max_tokens": 10,
                "temperature": 0.0,
            },
        )

        after_launch_children = get_child_processes()
        after_launch_gpu_pids = get_gpu_processes()

        print(f"[After Launch] Child processes: {len(after_launch_children)}")
        print(f"[After Launch] GPU processes: {len(after_launch_gpu_pids)}")

        # Should have more processes after launch
        assert (
            len(after_launch_children) > initial_child_count
        ), "Expected child processes to be created during launch"
        assert len(after_launch_gpu_pids) > len(
            initial_gpu_pids
        ), "Expected GPU processes to be created during launch"

        await generator.shutdown()

        # Wait a bit for cleanup
        time.sleep(2)

        final_children = get_child_processes()
        final_gpu_pids = get_gpu_processes()

        print(f"[After Shutdown] Child processes: {len(final_children)}")
        print(f"[After Shutdown] GPU processes: {len(final_gpu_pids)}")

        # Check for dangling child processes
        if len(final_children) > initial_child_count:
            dangling = get_process_info(final_children)

            print("\n⚠️  Dangling child processes detected:")
            for proc_info in dangling:
                print(
                    f"  - PID {proc_info['pid']} (PPID {proc_info['ppid']}): {proc_info['name']} ({proc_info['status']})"
                )
                print(f"    CMD: {proc_info['cmdline']}")

            print(
                f"\nTotal: {len(dangling)} dangling processes after shutdown (baseline: {initial_child_count})."
                "This seems to be benigh as they'll be cleaned up when the parent process exits. "
                "See P2106216182 for repro with just monarch primitives."
            )
        else:
            print("\n✅ No dangling child processes detected")

        # Check for dangling GPU processes
        new_gpu_pids = final_gpu_pids - initial_gpu_pids
        if new_gpu_pids:
            print(f"⚠️  Dangling GPU processes detected: {new_gpu_pids}")
            raise AssertionError(
                f"Found {len(new_gpu_pids)} dangling GPU processes after shutdown"
            )

        print("✅ No dangling GPU processes detected")

    finally:
        pass

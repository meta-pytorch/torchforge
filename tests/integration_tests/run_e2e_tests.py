# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end integration test runner for Forge applications.

This test runner validates that training can run without crashes or exceptions.
Similar to TorchTitan's integration test approach, we focus on functional
correctness (no crashes) rather than numerical validation.

Usage:
    python tests/integration_tests/run_e2e_tests.py
"""

import subprocess
import sys
import time
from pathlib import Path


def run_grpo_test(
    config_path: str,
    max_steps: int = 5,
    timeout: int = 1800,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run GRPO training and verify it completes without crashes.

    Args:
        config_path: Path to YAML config file
        max_steps: Number of training steps to run
        timeout: Maximum time in seconds to wait
        extra_args: Additional CLI arguments to pass

    Returns:
        CompletedProcess object with stdout/stderr

    Raises:
        Exception: If training fails with non-zero exit code
    """
    cmd = [
        sys.executable,
        "-m",
        "apps.grpo.main",
        "--config",
        config_path,
        "--trainer.training.steps",
        str(max_steps),
        # Disable WandB for CI to avoid auth issues - only use console logging
        "--metric_logging",
        '{"console": {"reduce_across_ranks": true}}',
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running e2e test: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Max steps: {max_steps}, Timeout: {timeout}s")
    print("-" * 80)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        raise Exception(
            f"Training timed out after {elapsed:.1f}s (timeout={timeout}s)"
        )

    elapsed = time.time() - start_time

    # Print output for debugging
    if result.stdout:
        print("STDOUT:")
        print(result.stdout[-2000:])  # Print last 2000 chars to avoid overwhelming logs

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr[-2000:])

    print("-" * 80)

    # Check for success
    if result.returncode != 0:
        raise Exception(
            f"Training failed with return code {result.returncode} after {elapsed:.1f}s"
        )

    print(f"✓ Training completed successfully in {elapsed:.1f}s")
    return result


def main():
    """Run all e2e tests."""
    print("=" * 80)
    print("Forge E2E Integration Tests")
    print("=" * 80)

    # Test 1: GRPO with smallest model
    test_config = "apps/grpo/qwen3_1_7b.yaml"

    if not Path(test_config).exists():
        raise FileNotFoundError(f"Config file not found: {test_config}")

    try:
        run_grpo_test(test_config, max_steps=5, timeout=1800)
        print("\n" + "=" * 80)
        print("✓ All e2e tests passed!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ E2E test failed: {e}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

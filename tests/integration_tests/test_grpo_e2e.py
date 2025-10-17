# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end integration test for GRPO training.

This test validates that GRPO training can run without crashes or exceptions.
Similar to TorchTitan's integration test approach, we focus on functional
correctness (no crashes) rather than numerical validation.

Usage:
    python tests/integration_tests/test_grpo_e2e.py
"""

import subprocess
import sys
import time
from pathlib import Path


def run_grpo_training(
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
        f"trainer.training.steps={str(max_steps)}",
        # Disable WandB for CI to avoid auth issues - only use console logging
        "~metric_logging.wandb",
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running GRPO e2e test: {config_path}")
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
            f"GRPO training timed out after {elapsed:.1f}s (timeout={timeout}s)"
        ) from None

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
            f"GRPO training failed with return code {result.returncode} after {elapsed:.1f}s"
        )

    print(f"✓ GRPO training completed successfully in {elapsed:.1f}s")
    return result


def main():
    """Run GRPO e2e test."""
    print("=" * 80)
    print("GRPO E2E Integration Test")
    print("=" * 80)

    # Test GRPO with smallest model
    test_config = "apps/grpo/qwen3_1_7b.yaml"

    if not Path(test_config).exists():
        raise FileNotFoundError(f"Config file not found: {test_config}")

    try:
        run_grpo_training(test_config, max_steps=5, timeout=1800)
        print("\n" + "=" * 80)
        print("✓ GRPO e2e test passed!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ GRPO e2e test failed: {e}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

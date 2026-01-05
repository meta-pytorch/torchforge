# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metrics collection and reporting for generator throughput benchmarks.

Based on vLLM's throughput benchmark metrics patterns.
Reference: vllm/benchmarks/throughput.py (lines 762-809)
"""

import json
from dataclasses import asdict, dataclass

from forge.data_models.completion import Completion


@dataclass
class ThroughputMetrics:
    """Throughput benchmark metrics for offline inference.
    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/throughput.py

    Attributes:
        elapsed_time: Total wall-clock time in seconds
        num_requests: Total number of requests processed
        num_completions: Total number of completions (requests * n samples)
        total_prompt_tokens: Sum of all prompt tokens
        total_output_tokens: Sum of all generated output tokens
        total_tokens: Sum of prompt and output tokens
        requests_per_second: Request throughput (requests/sec)
        completions_per_second: Completion throughput (completions/sec)
        tokens_per_second: Total token throughput (tokens/sec)
        output_tokens_per_second: Output token throughput (output tokens/sec)
        model: Optional model name for reporting
        config: Optional benchmark configuration dict
    """

    elapsed_time: float
    num_requests: int
    num_completions: int
    total_prompt_tokens: int
    total_output_tokens: int
    total_tokens: int
    requests_per_second: float
    completions_per_second: float
    tokens_per_second: float
    output_tokens_per_second: float
    model: str | None = None
    config: dict | None = None


def extract_token_counts(completions: list[list[Completion]]) -> tuple[int, int]:
    """Extract token counts from generator completions.

    Args:
        completions: List of completion lists from Generator.generate() calls.
                     Each Generator.generate() call returns a list of Completion objects.

    Returns:
        Tuple of (total_prompt_tokens, total_output_tokens)
    """
    total_prompt_tokens = 0
    total_output_tokens = 0

    for completion_list in completions:
        for completion in completion_list:
            # Completion has prompt_ids and token_ids as torch.Tensor
            # Shape: (seq_len,)
            total_prompt_tokens += completion.prompt_ids.shape[0]
            total_output_tokens += completion.token_ids.shape[0]

    return total_prompt_tokens, total_output_tokens


def calculate_metrics(
    completions: list[list[Completion]],
    elapsed_time: float,
    model: str | None = None,
    config: dict | None = None,
) -> ThroughputMetrics:
    """Calculate throughput metrics from completions and timing.

    Args:
        completions: List of completion lists from Generator.generate() calls
        elapsed_time: Total time elapsed in seconds
        model: Optional model name
        config: Optional benchmark configuration

    Returns:
        ThroughputMetrics object with calculated metrics
    """
    num_requests = len(completions)
    num_completions = sum(len(completion_list) for completion_list in completions)
    total_prompt_tokens, total_output_tokens = extract_token_counts(completions)
    total_tokens = total_prompt_tokens + total_output_tokens

    return ThroughputMetrics(
        elapsed_time=elapsed_time,
        num_requests=num_requests,
        num_completions=num_completions,
        total_prompt_tokens=total_prompt_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        requests_per_second=num_requests / elapsed_time if elapsed_time > 0 else 0.0,
        completions_per_second=(
            num_completions / elapsed_time if elapsed_time > 0 else 0.0
        ),
        tokens_per_second=total_tokens / elapsed_time if elapsed_time > 0 else 0.0,
        output_tokens_per_second=(
            total_output_tokens / elapsed_time if elapsed_time > 0 else 0.0
        ),
        model=model,
        config=config,
    )


def print_metrics(metrics: ThroughputMetrics) -> None:
    """Print metrics to console in a formatted table.

    Args:
        metrics: ThroughputMetrics to print
    """
    print("=" * 55)
    print("Throughput Benchmark Results".center(55))
    print("=" * 55)

    if metrics.model:
        print(f"Model: {metrics.model}")

    # Calculate samples per request
    samples_per_request = (
        metrics.num_completions / metrics.num_requests
        if metrics.num_requests > 0
        else 0
    )

    print(f"Requests: {metrics.num_requests}")
    print(
        f"Completions: {metrics.num_completions} ({samples_per_request:.1f} per request)"
    )
    print(f"Elapsed Time: {metrics.elapsed_time:.2f} seconds")
    print("-" * 55)
    print(f"Total Prompt Tokens: {metrics.total_prompt_tokens}")
    print(f"Total Output Tokens: {metrics.total_output_tokens}")
    print(f"Total Tokens: {metrics.total_tokens}")
    print("-" * 55)
    print("Throughput:")
    print(f"  Requests/sec: {metrics.requests_per_second:.2f}")
    print(f"  Completions/sec: {metrics.completions_per_second:.2f}")
    print(f"  Total Tokens/sec: {metrics.tokens_per_second:.2f}")
    print(f"  Output Tokens/sec: {metrics.output_tokens_per_second:.2f}")
    print("=" * 55)


def save_metrics_json(metrics: ThroughputMetrics, output_path: str) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: ThroughputMetrics to save
        output_path: Path to output JSON file
    """
    metrics_dict = asdict(metrics)

    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")

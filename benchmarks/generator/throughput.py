# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline throughput benchmark for torchforge generators.

Based on vLLM's offline throughput pattern but adapted for torchforge's
generator infrastructure. Measures maximum generation throughput without
simulating request arrival patterns.

Reference: https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/throughput.py

Example usage:
    # Using config file with defaults (randomly generated prompt)
    python -m benchmarks.generator.throughput --config apps/grpo/qwen3_1_7b.yaml

    # With benchmark parameter overrides
    python -m benchmarks.generator.throughput \\
        --config apps/grpo/qwen3_1_7b.yaml \\
        benchmark.num_requests=100 \\
        benchmark.input_len=1024 \\
        benchmark.output_len=2048 \\
        benchmark.dataset=random \\
        benchmark.output_json=results.json

    # For sanity check, you could use a fixed prompt and sample the responses.
    python -m benchmarks.generator.throughput \\
        --config apps/grpo/qwen3_1_7b.yaml \\
        benchmark.num_requests=10 \\
        benchmark.dataset=fixed \\
        benchmark.fixed_prompt="Tell me a joke" \\
        benchmark.num_samples=5

If you are interested in validating the benchmark by introducing regression, you can
try reducing `max_num_seqs` to limit the number of parallel request sequences the system
tries to fit into a single batch for processing

    policy:
      engine_args:
      model: ${model}
      max_num_seqs: 32  # ← Add this line (default is 256)

"""

import argparse
import asyncio
import os
import random
import time

from benchmarks.generator.datasets import BenchmarkRequest, FixedDataset, RandomDataset
from benchmarks.generator.metrics import (
    calculate_metrics,
    print_metrics,
    save_metrics_json,
)
from forge.actors.generator import Generator
from forge.controller.provisioner import init_provisioner, shutdown
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from omegaconf import DictConfig

from vllm import __version__ as vllm_version

if vllm_version >= "0.13.0":
    from vllm.tokenizers import get_tokenizer
else:
    from vllm.transformers_utils.tokenizer import get_tokenizer

os.environ.setdefault("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS", "600")
os.environ.setdefault("HYPERACTOR_CODE_MAX_FRAME_LENGTH", "1073741824")


async def run_throughput_benchmark(
    cfg: DictConfig,
    num_requests: int,
    input_len: int,
    output_len: int,
    dataset_type: str = "random",
    range_ratio: float = 0.0,
    fixed_prompt: str | None = None,
    output_json: str | None = None,
    num_samples: int = 5,
) -> None:
    """Run offline throughput benchmark.

    Based on vLLM's run_vllm() pattern (throughput.py:42-122).

    Args:
        cfg: TorchForge config from YAML
        num_requests: Number of requests to benchmark
        input_len: Input prompt length in tokens
        output_len: Output generation length in tokens
        dataset_type: Dataset type ("random" or "fixed")
        range_ratio: Variance ratio for input/output lengths (0.0-1.0)
        fixed_prompt: Prompt to use for "fixed" dataset type
        output_json: Optional path to save JSON metrics
        num_samples: Number of sample responses to print (0 to print all)
    """
    if cfg.get("provisioner", None) is not None:
        await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )

    print("Spawning Generator service...")
    generator = await Generator.options(**cfg.services.generator).as_service(
        **cfg.generator
    )

    print(f"Generating {num_requests} benchmark requests...")
    model_name = cfg.generator.engine_args.get("model", "unknown")
    tokenizer = get_tokenizer(
        model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    if dataset_type == "random":
        dataset = RandomDataset(
            tokenizer=tokenizer,
            num_requests=num_requests,
            input_len=input_len,
            output_len=output_len,
            range_ratio=range_ratio,
        )
    elif dataset_type == "fixed":
        if fixed_prompt is None:
            fixed_prompt = "Tell me a joke"
        dataset = FixedDataset(
            tokenizer=tokenizer,
            prompt=fixed_prompt,
            num_requests=num_requests,
            output_len=output_len,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    requests: list[BenchmarkRequest] = dataset.generate()

    print(f"\nRunning throughput benchmark ({num_requests} requests)...")
    prompts = [req.prompt for req in requests]
    request_ids = [req.request_id for req in requests]

    start = time.perf_counter()
    # TODO: here we're measuring two things together: compute (vllm) and io (monarch).
    # We shall consider finer grained metrics collection to distinguish the two.
    completions = await asyncio.gather(
        *[generator.generate.route(prompt) for prompt in prompts]
    )
    end = time.perf_counter()

    elapsed_time = end - start

    print("\nBenchmark completed!")

    config_dict = {
        "num_requests": num_requests,
        "input_len": input_len,
        "output_len": output_len,
        "dataset_type": dataset_type,
        "range_ratio": range_ratio,
    }

    metrics = calculate_metrics(
        completions=completions,
        elapsed_time=elapsed_time,
        model=model_name,
        config=config_dict,
    )

    print_metrics(metrics)

    if output_json:
        save_metrics_json(metrics, output_json)

    if dataset_type == "fixed" and completions:
        print("\n" + "=" * 80)
        print("SAMPLED RESPONSES (Fixed Prompt)")
        print("=" * 80)
        print(f"\nPrompt: {fixed_prompt or 'Tell me a joke'}")

        # Flatten completions: each request returns multiple completions (n samples)
        # completions is list[list[Completion]] where outer list = requests, inner list = n samples
        all_completions = []
        all_request_ids = []
        for req_idx, completion_group in enumerate(completions):
            for sample_idx, completion in enumerate(completion_group):
                all_completions.append(completion)
                all_request_ids.append(f"{request_ids[req_idx]}-sample{sample_idx}")

        samples_to_show = (
            len(all_completions)
            if num_samples == 0
            else min(num_samples, len(all_completions))
        )
        print(
            f"\nShowing {samples_to_show} of {len(all_completions)} total completions"
        )
        print(
            f"({len(completions)} requests × {len(completions[0])} samples per request)\n"
        )

        # Randomly sample from all completions
        if samples_to_show < len(all_completions):
            sample_indices = random.sample(range(len(all_completions)), samples_to_show)
            sample_indices.sort()
        else:
            sample_indices = range(len(all_completions))

        for i, idx in enumerate(sample_indices, 1):
            completion = all_completions[idx]
            request_id = all_request_ids[idx]
            print(f"\n--- Response {i} (Request ID: {request_id}) ---")
            print(completion.text)
            print("-" * 40)

        print("\n" + "=" * 80)

    print("\nShutting down...")
    await shutdown()


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for throughput benchmark.

    Based on vLLM's add_cli_args() pattern (throughput.py:544-696).
    """
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to torchforge YAML config file",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to benchmark (default: 100)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Input prompt length in tokens (default: 128)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output generation length in tokens (default: 128)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["random", "fixed"],
        default="random",
        help="Dataset type: random or fixed (default: random)",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.0,
        help="Variance ratio for input/output lengths (default: 0.0 for fixed lengths)",
    )
    parser.add_argument(
        "--fixed-prompt",
        type=str,
        default=None,
        help="Prompt to use for 'fixed' dataset type (default: 'Tell me a joke')",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON metrics (optional)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample responses to print for fixed prompts (0 to print all, default: 5)",
    )


@parse
def recipe_main(cfg: DictConfig) -> None:
    """Main entry point for throughput benchmark.

    Args:
        cfg: Config loaded from YAML file via @parse decorator.
             Benchmark parameters can be specified in the config or via key=value overrides like
             python -m benchmarks.generator.throughput --config apps/grpo/qwen3_1_7b.yaml benchmark.num_requests=200
    """
    benchmark_cfg = cfg.get("benchmark", {})
    num_requests = benchmark_cfg.get("num_requests", 100)
    input_len = benchmark_cfg.get("input_len", 1024)
    output_len = benchmark_cfg.get("output_len", 2048)
    dataset_type = benchmark_cfg.get("dataset", "random")
    range_ratio = benchmark_cfg.get("range_ratio", 0.0)
    fixed_prompt = benchmark_cfg.get("fixed_prompt", None)
    output_json = benchmark_cfg.get("output_json", None)
    num_samples = benchmark_cfg.get("num_samples", 5)

    asyncio.run(
        run_throughput_benchmark(
            cfg=cfg,
            num_requests=num_requests,
            input_len=input_len,
            output_len=output_len,
            dataset_type=dataset_type,
            range_ratio=range_ratio,
            fixed_prompt=fixed_prompt,
            output_json=output_json,
            num_samples=num_samples,
        )
    )


if __name__ == "__main__":
    recipe_main()

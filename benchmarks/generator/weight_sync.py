# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight sync benchmark for torchforge generators.

Measures the time for weight synchronization between trainer and generator,
with and without shared memory prefetching enabled.

Example usage:
    # Basic benchmark (no prefetch)
    python -m benchmarks.generator.weight_sync --config apps/grpo/qwen3_8b.yaml

    # With prefetch enabled
    python -m benchmarks.generator.weight_sync \
        --config apps/grpo/qwen3_8b.yaml \
        benchmark.prefetch_enabled=true \
        benchmark.n_fetcher_procs=4 \
        benchmark.iterations=5
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torchstore as ts
from forge.actors.generator import Generator
from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner, shutdown
from forge.controller.service.service import uuid
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse, resolve_hf_hub_paths
from monarch.actor import endpoint
from omegaconf import DictConfig

os.environ.setdefault("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS", "600")
os.environ.setdefault("HYPERACTOR_CODE_MAX_FRAME_LENGTH", "1073741824")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BenchmarkTitanTrainer(TitanTrainer):
    """TitanTrainer with weight modification capabilities for benchmarking."""

    @endpoint
    async def modify_weights(self):
        """Scale all model weights by a factor (simulates training step)."""
        scale: float = 1.001
        for model_part in self.engine.model_parts:
            sd = model_part.state_dict()
            for k in sd.keys():
                if torch.is_floating_point(sd[k]):
                    sd[k] *= scale

    @endpoint
    async def get_model_size_bytes(self) -> int:
        """Get total model size in bytes across all model parts."""
        total_bytes = 0
        for model_part in self.engine.model_parts:
            for param in model_part.parameters():
                total_bytes += param.numel() * param.element_size()
        return total_bytes


@dataclass
class WeightSyncMetrics:
    """Metrics from a single weight sync operation."""

    version: int
    total_time_s: float
    push_time_s: float
    update_time_s: float
    prefetch_enabled: bool


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    model: str
    iterations: int
    prefetch_enabled: bool
    n_fetcher_procs: int
    model_size_bytes: int = 0
    metrics: list[WeightSyncMetrics] = field(default_factory=list)

    @property
    def model_size_gb(self) -> float:
        return self.model_size_bytes / (1024**3)

    @property
    def avg_total_time_s(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.total_time_s for m in self.metrics) / len(self.metrics)

    @property
    def avg_push_time_s(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.push_time_s for m in self.metrics) / len(self.metrics)

    @property
    def avg_update_time_s(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.update_time_s for m in self.metrics) / len(self.metrics)

    @property
    def push_throughput_gb_s(self) -> float:
        if self.avg_push_time_s <= 0 or self.model_size_bytes <= 0:
            return 0.0
        return self.model_size_gb / self.avg_push_time_s

    @property
    def update_throughput_gb_s(self) -> float:
        if self.avg_update_time_s <= 0 or self.model_size_bytes <= 0:
            return 0.0
        return self.model_size_gb / self.avg_update_time_s


def print_results(results: BenchmarkResults):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("WEIGHT SYNC BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: {results.model}")
    print(f"Model size: {results.model_size_gb:.2f} GB")
    print(f"Iterations: {results.iterations}")
    print(f"Prefetch enabled: {results.prefetch_enabled}")
    if results.prefetch_enabled:
        print(f"Fetcher procs: {results.n_fetcher_procs}")
    print("-" * 80)
    print(f"{'Metric':<30} {'Time (s)':<15} {'Throughput (GB/s)':<20}")
    print("-" * 80)
    print(
        f"{'Avg push_weights':<30} {results.avg_push_time_s:>12.3f} s "
        f"{results.push_throughput_gb_s:>12.2f} GB/s"
    )
    print(
        f"{'Avg update_weights':<30} {results.avg_update_time_s:>12.3f} s "
        f"{results.update_throughput_gb_s:>12.2f} GB/s"
    )
    print(f"{'Avg total (push + update)':<30} {results.avg_total_time_s:>12.3f} s")
    print("=" * 80 + "\n")


async def run_weight_sync_benchmark(
    cfg: DictConfig,
    iterations: int,
    prefetch_enabled: bool,
    n_fetcher_procs: int,
    warmup_iterations: int,
) -> BenchmarkResults:
    """Run weight sync benchmark with knobs to enable prefetch, fetcher procs, etc.

    Args:
        cfg: TorchForge config from YAML
        iterations: Number of weight sync iterations to benchmark
        prefetch_enabled: Whether to enable shared memory prefetching
        n_fetcher_procs: Number of fetcher processes (when prefetch_enabled=True)
        warmup_iterations: Number of warmup iterations before timing

    Returns:
        BenchmarkResults with timing metrics
    """
    model_name = cfg.generator.engine_args.get("model", "unknown")

    generator_cfg = cfg.generator.copy()
    if prefetch_enabled:
        generator_cfg.prefetch_weights_to_shm = True
        generator_cfg.n_fetcher_procs = n_fetcher_procs
    else:
        generator_cfg.prefetch_weights_to_shm = False
        generator_cfg.n_fetcher_procs = 0

    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    services_generator_cfg = cfg.services.generator.copy()
    services_generator_cfg.num_replicas = 1

    logger.info("Spawning Generator and Trainer...")
    generator, trainer = await asyncio.gather(
        Generator.options(**services_generator_cfg).as_service(**generator_cfg),
        BenchmarkTitanTrainer.options(**cfg.actors.trainer).as_actor(**cfg.trainer),
    )
    logger.info("Generator and Trainer spawned.")

    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = await provisioner.get_host_mesh(trainer_host_mesh_name)
    # same as the main grpo app.
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    logger.info("Torchstore initialized with LocalRankStrategy")

    if warmup_iterations > 0:
        logger.info(f"Running {warmup_iterations} warmup iteration(s)...")
        for i in range(warmup_iterations):
            v = uuid.uuid4().int
            await trainer.push_weights.call(policy_version=v)
            await generator.update_weights.fanout(version=v)
            await trainer.modify_weights.call()
        logger.info("Warmup complete.")

    # Get model size for throughput calculation
    # With DTensor/TP, each rank's param.numel() returns global size, not shard size
    # So just take one rank's value
    model_size_result = await trainer.get_model_size_bytes.call()
    _, model_size_bytes = next(iter(model_size_result.items()))
    model_size_gb = model_size_bytes / (1024**3)
    logger.info(f"Model size: {model_size_gb:.2f} GB")

    logger.info(f"Running {iterations} timed iteration(s)...")
    metrics: list[WeightSyncMetrics] = []

    # Generate a test prompt for in-flight requests
    test_prompt = "What is the capital of France? Please explain in detail."

    for i in range(iterations):
        v = uuid.uuid4().int

        # Modify weights to simulate training
        await trainer.modify_weights.call()

        # Time push_weights
        push_start = time.perf_counter()
        await trainer.push_weights.call(policy_version=v)
        push_end = time.perf_counter()
        push_time_s = push_end - push_start

        # Simulate in-flight requests that pause_generation must wait for
        num_inflight = 4
        generation_tasks = [
            asyncio.create_task(generator.generate.route(test_prompt))
            for _ in range(num_inflight)
        ]
        # Give generation a moment to start
        await asyncio.sleep(0.1)

        # Time update_weights (includes pause_generation waiting for in-flight)
        update_start = time.perf_counter()
        await generator.update_weights.fanout(version=v)
        update_end = time.perf_counter()
        update_time_s = update_end - update_start

        # Wait for generation to complete (after weight update)
        await asyncio.gather(*generation_tasks)

        total_time_s = push_time_s + update_time_s

        metrics.append(
            WeightSyncMetrics(
                version=v,
                total_time_s=total_time_s,
                push_time_s=push_time_s,
                update_time_s=update_time_s,
                prefetch_enabled=prefetch_enabled,
            )
        )

        logger.info(
            f"Iteration {i + 1}/{iterations}: push={push_time_s:.3f}s, "
            f"update={update_time_s:.3f}s, total={total_time_s:.3f}s"
        )

    logger.info("Cleaning up...")
    await trainer.cleanup.call()
    await generator.shutdown()
    await BenchmarkTitanTrainer.shutdown(trainer)
    await ts.shutdown()

    return BenchmarkResults(
        model=model_name,
        iterations=iterations,
        prefetch_enabled=prefetch_enabled,
        n_fetcher_procs=n_fetcher_procs if prefetch_enabled else 0,
        model_size_bytes=model_size_bytes,
        metrics=metrics,
    )


@parse
def recipe_main(cfg: DictConfig = None) -> None:  # type: ignore[assignment]
    """Main entry point for weight sync benchmark.

    Args:
        cfg: Config loaded from YAML file via @parse decorator.
             Benchmark parameters can be specified via key=value overrides:
             benchmark.iterations=5
             benchmark.prefetch_enabled=true
             benchmark.n_fetcher_procs=4
             benchmark.warmup_iterations=1
    """
    cfg = resolve_hf_hub_paths(cfg)

    benchmark_cfg = cfg.get("benchmark", {})
    iterations = benchmark_cfg.get("iterations", 3)
    prefetch_enabled = benchmark_cfg.get("prefetch_enabled", False)
    n_fetcher_procs = benchmark_cfg.get("n_fetcher_procs", 8)
    warmup_iterations = benchmark_cfg.get("warmup_iterations", 1)

    results = asyncio.run(
        run_weight_sync_benchmark(
            cfg=cfg,
            iterations=iterations,
            prefetch_enabled=prefetch_enabled,
            n_fetcher_procs=n_fetcher_procs,
            warmup_iterations=warmup_iterations,
        )
    )
    print_results(results)

    asyncio.run(shutdown())


if __name__ == "__main__":
    recipe_main()

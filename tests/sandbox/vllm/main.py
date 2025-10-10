# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""To run:
export HF_HUB_DISABLE_XET=1
python -m tests.sandbox.vllm.main --config tests/sandbox/vllm/llama3_8b.yaml
"""

import asyncio

import os

from forge.actors.policy import Policy
from forge.cli.config import parse
from forge.controller.provisioner import shutdown
from dataclasses import dataclass
from datasets import load_dataset

from forge.controller.actor import ForgeActor
from monarch.actor import endpoint
from vllm.transformers_utils.tokenizer import get_tokenizer
from forge.controller.launcher import JOB_NAME_KEY, LAUNCHER_KEY
from forge.controller.provisioner import init_provisioner, shutdown

from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.types import LauncherConfig, ProvisionerConfig
from omegaconf import DictConfig

from forge.observability.perf_tracker import Tracer
from forge.types import (
    Launcher,
    LauncherConfig,
    ProcessConfig,
    ProvisionerConfig,
    ServiceConfig,
)

import time
from collections import deque

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"


import time
import statistics
from collections import deque

class ThroughputTracker:
    def __init__(self, window_size=60):  # 60 second window
        self.window_size = window_size
        self.request_times = deque()
        self.token_counts = deque()
        self.latencies = deque()  # Store latency for each request
        self.last_print = time.time()
        self.print_interval = 10  # Print every 10 seconds

    def start_request(self):
        """Call this when starting a request. Returns the start time."""
        return time.time()

    def end_request(self, start_time, num_tokens):
        """Call this when a request completes."""
        end_time = time.time()
        latency = end_time - start_time

        self.request_times.append(end_time)
        self.token_counts.append(num_tokens)
        self.latencies.append(latency)

        # Remove old entries outside the window
        cutoff_time = end_time - self.window_size
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
            self.token_counts.popleft()
            self.latencies.popleft()

        # Print throughput info periodically
        if end_time - self.last_print >= self.print_interval:
            self.print_metrics()
            self.last_print = end_time

    def print_metrics(self):
        if not self.request_times:
            return

        time_window = time.time() - self.request_times[0] if len(self.request_times) > 1 else self.print_interval
        requests_per_sec = len(self.request_times) / max(time_window, 1)
        tokens_per_sec = sum(self.token_counts) / max(time_window, 1)

        # Calculate latency statistics
        if self.latencies:
            avg_latency = statistics.mean(self.latencies)
            sorted_latencies = sorted(self.latencies)
            p50_latency = statistics.median(sorted_latencies)
            p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))] if len(sorted_latencies) > 0 else 0
            p99_latency = sorted_latencies[int(0.99 * len(sorted_latencies))] if len(sorted_latencies) > 0 else 0

            print(f"📊 Throughput: {requests_per_sec:.2f} req/sec | {tokens_per_sec:.2f} tok/sec")
            print(f"⏱️ Latency: avg={avg_latency:.1f}s | p50={p50_latency:.1f}s | p95={p95_latency:.1f}s | p99={p99_latency:.1f}s")


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str = "openai/gsm8k"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = True
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)

        def gsm8k_transform(sample):
            system_prompt = """
            Put all your scratchpad work between <think> and </think> tags.
            Your final answer should be between <answer> and </answer> tags otherwise it will not be scored.
            """
            request: str = sample["question"]
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            target: str = sample["answer"]
            formatted_target = target.split("#### ")[1]
            return {"request": formatted_request, "target": formatted_target}

        ds = load_dataset(
            self.path, self.revision, split=self.data_split, streaming=self.streaming
        )
        ds = ds.map(gsm8k_transform)
        ds = ds.shuffle()
        self._iterator = iter(ds)

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)
            return sample
        except StopIteration:
            return None


async def run(cfg: DictConfig):
    if cfg.get("provisioner", None) is not None:
        await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    print("Spawning service...")
    (dataloader, policy) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
    )

    max_res_tokens = cfg.get("max_res_tokens", None)
    assert max_res_tokens is not None, "max_res_tokens must be specified in config"
    group_size = cfg.get("group_size", None)
    assert group_size is not None, "group_size must be specified in config"
    token_per_request = max_res_tokens * group_size
    num_rollout_threads = cfg.get("rollout_threads", 1)

    throughput_tracker = ThroughputTracker()

    async def continuous_rollouts():
        print("Starting continuous rollouts")
        print(f"  {max_res_tokens=}")
        print(f"  {group_size=}")
        print(f"  {num_rollout_threads=}")
        while True:
            t = Tracer("main_perf/continuous_rollouts")
            t.start()
            sample = await dataloader.sample.call_one()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            t.step("data_loading")
            prompt, target = sample["request"], sample["target"]

            request_start_time = throughput_tracker.start_request()
            responses = await policy.generate.route(prompt)
            throughput_tracker.end_request(request_start_time, token_per_request)
            t.step("policy_generation")

            # print(f"#------ request  ------#")
            # print(prompt)
            # print("#------ target   ------#")
            # print(target)
            print(f"#------ responses ------#")
            # print(responses[0].text)
            # print()
            assert len(responses) == group_size


    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]

    await asyncio.gather(*rollout_tasks)
    print("\nShutting down...")
    await policy.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()

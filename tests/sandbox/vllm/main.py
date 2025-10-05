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

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"



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

    async def continuous_rollouts():
        print("Starting continuous rollouts")
        while True:
            t = Tracer("main_perf/continuous_rollouts")
            t.start()
            sample = await dataloader.sample.call_one()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            t.step("data_loading")
            prompt, target = sample["request"], sample["target"]
            responses = await policy.generatlee.route(prompt)
            version = await policy.get_version.route()

            t.step("policy_generation")

            print("request: ", prompt)
            print("response: ", responses[0].text)


    rollout_task = asyncio.create_task(continuous_rollouts())
    await asyncio.gather(rollout_task)
    print("\nShutting down...")
    await policy.shutdown()
    await shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    recipe_main()

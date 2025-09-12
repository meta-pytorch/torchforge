# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A working example showcasing a practical example of forge with RL.

Run this with:
    python -m apps.rl.main --config apps/rl/llama3_8b.yaml

"""

import asyncio
import logging
import sys
from dataclasses import dataclass

import torch
from forge.actors import ReplayBuffer, RLTrainer
from forge.cli.config import parse
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from forge.data import Episode
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def run(cfg: DictConfig):
    trainer, replay_buffer = await asyncio.gather(
        spawn_service(
            ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=4),
            RLTrainer,
            **cfg.trainer,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            ReplayBuffer,
            **cfg.replay_buffer,
        ),
    )
    print("Services initialized...")

    print("Collecting Data...")
    g = torch.manual_seed(0)
    for i in range(cfg.replay_buffer.batch_size):
        req_len, res_len = torch.randint(64,256, (2,), generator=g).tolist()
        e = Episode(
            episode_id=i,
            request="",
            policy_version=0,
            pad_id=0,
            request_len=256,
            response_len=256,
            request_tokens=torch.randint(64_000, (req_len,), generator=g),
            response_tokens=torch.randint(64_000, (res_len,), generator=g),
            ref_logprobs=torch.randn((512, 64_000), generator=g),
            advantage=torch.randn((1,), generator=g)
        )
        replay_buffer.add.choose(e)

    print("Train step...")
    batch = await replay_buffer.sample.choose(curr_policy_version=0)
    output = await trainer.train_step.choose(batch)
    print(output) # philip
    #print("Loss: ", output["loss"])


    print("Shutting down...")
    await shutdown_service(trainer)
    await shutdown_service(replay_buffer)


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())

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

from forge.actors import ReplayBuffer, RLTrainer
from forge.cli.config import parse
from forge.controller.orchestrator import ServiceConfig, shutdown_service, spawn_service
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
    print("Services initialized....")

    print("shutting down...")
    await shutdown_service(trainer)
    await shutdown_service(replay_buffer)


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())

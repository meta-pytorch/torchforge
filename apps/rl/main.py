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
from forge.controller import spawn_actors

from forge.controller.service import ServiceConfig, spawn_service
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def run(cfg: DictConfig):

    trainer = await spawn_service(
        ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=1),
        RLTrainer,
        **cfg.trainer,
    )
    buffer = (
        await spawn_actors(
            name="replay_buffer",
            actor_cls=ReplayBuffer,
            cfg=cfg.replay_buffer,
            processes=cfg.replay_buffer.pop("processes"),
        ),
    )
    print("Actors spawned")

    # Initialize everything
    await trainer.setup.call()
    print("Setup done")

    print("shutting down...")
    await asyncio.gather(*[a.mesh.stop() for a in [trainer]])


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())

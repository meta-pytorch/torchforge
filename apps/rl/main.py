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

from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def run(cfg: DictConfig):
    trainer = await RLTrainer.options(
        procs_per_replica=1, with_gpus=True, num_replicas=4
    ).as_service(**cfg.trainer)
    replay_buffer = await ReplayBuffer.options(
        procs_per_replica=1, num_replicas=1
    ).as_service(**cfg.replay_buffer)

    print("Services initialized....")

    print("shutting down...")
    await trainer.shutdown()
    await replay_buffer.shutdown()


@parse
def recipe_main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    sys.exit(recipe_main())

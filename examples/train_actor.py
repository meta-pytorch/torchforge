# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single host train actor example.

Make sure you do

HF_HUB_DISABLE_XET=1  python scripts/download_tokenizer.py --repo_id meta-llama/Ll
ama-3.1-8B --hf_token=...

"""

import asyncio

import os

import time
from typing import Optional

from forge.train import ConfigManager, init_logger, Trainer
from monarch.actor import Actor, current_rank, current_size, endpoint, proc_mesh


class TrainerActor(Actor):
    """TorchTitan trainer actor."""

    def __init__(self):
        self.rank = current_rank().rank
        self.world_size = current_size()["gpus"]

    def _init_dist(self):
        """Initialize torch distributed.

        torchrun would normally do this. We'll need to do something
        similar for all Torch-based actors - probably enough
        reason to introduce a `TorchActor` abstraction, or something
        similar.

        """
        env = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.rank),
            "LOCAL_WORLD_SIZE": str(self.world_size),
            "GROUP_RANK": str(self.world_size),
            "GROUP_WORLD_SIZE": str(self.world_size),
            "ROLE_RANK": str(self.rank),
            "ROLE_WORLD_SIZE": str(self.world_size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self.world_size),
        }
        os.environ.update(env)
        self.rlog("env: {}".format(env))
        self.clog("full env: {}".format(os.environ))

    @endpoint
    async def initialize(self):
        """Initialize the trainer actor.

        This is convenient to keep separate as a way to surface any
        possible errors in complex initializations.
        """
        self.rlog("Initializing")
        self._init_dist()
        config_manager = ConfigManager()
        config = config_manager.parse_args()
        config.model.tokenizer_path = os.path.join(
            "assets", "tokenizer", "Llama-3.1-8B", "original", "tokenizer.model"
        )
        self.clog("Config: {}".format(config))
        trainer: Optional[Trainer] = None

        try:
            trainer = Trainer(config)
            if config.checkpoint.create_seed_checkpoint:
                assert (
                    int(os.environ["WORLD_SIZE"]) == 1
                ), "Must create seed checkpoint using a single device, to disable sharding."
                assert (
                    config.checkpoint.enable_checkpoint
                ), "Must enable checkpointing when creating a seed checkpoint."
                trainer.checkpointer.save(curr_step=0, force=True)
                self.rlog("Created seed checkpoint")
        except Exception as e:
            self.rlog("Caught error {}".format(e))
            if trainer:
                trainer.close()
        self.trainer = trainer
        self.config = config
        self.rlog("Done initializing")

    @endpoint
    async def train(self):
        """Run the training loop."""
        assert self.trainer, "Initialization is expected to have been run successfully."
        self.rlog("Starting to train.")
        init_logger()
        self.trainer.train()

    @endpoint
    async def shutdown(self):
        if self.trainer:
            self.trainer.close()

    def __repr__(self) -> str:
        return "Trainer"

    def rlog(self, msg: str):
        """Log for all replicas."""
        timestamp = time.strftime("%m-%d %H:%M:%S")
        print(
            "{} [{}-{}/{}] {}".format(timestamp, self, self.rank, self.world_size, msg)
        )

    def clog(self, msg: str):
        """Log only for worker 0."""
        timestamp = time.strftime("%m-%d %H:%M:%S")
        print("{} [Trainer] {}".format(timestamp, msg))


async def main():
    print("Running TorchTitan train example.")

    print("Creating a proc_mesh")

    # Do we want a "torch proc mesh" that honors GPUs?
    procs = await proc_mesh(gpus=4)
    trainer = await procs.spawn("trainer", TrainerActor)
    await trainer.initialize.call()
    print("Starting training.")
    await trainer.train.call()


if __name__ == "__main__":
    asyncio.run(main())

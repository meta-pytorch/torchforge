# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single host train actor example."""

import asyncio

# from forge.train import ConfigManager, Trainer

from monarch.actor_mesh import Actor, endpoint
from monarch.proc_mesh import proc_mesh


class TrainerActor(Actor):
    """TorchTitan trainer actor."""

    def __init__(self):
        pass

    def initialize(self):
        """Initialize the trainer actor.

        This is convenient to keep separate as a way to surface any
        possible errors in complex initializations.
        """
        pass

    @endpoint
    async def train(self):
        pass


async def main():

    print("Running TorchTitan train example.")

    print("Creating a proc_mesh")

    # Do we want a "torch proc mesh" that honors GPUs?
    procs = await proc_mesh(gpus=4)
    trainer = await procs.spawn("trainer", TrainerActor)


if __name__ == "__main__":
    asyncio.run(main())

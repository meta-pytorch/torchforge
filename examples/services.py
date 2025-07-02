# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A working example showcasing how a 'services-based' approach works.

Run this with:
    python -m examples.services
"""

import asyncio

# from forge.monarch_utils.stack import stack
from monarch.actor_mesh import Actor, endpoint
from monarch.proc_mesh import proc_mesh


# ==== Assume the following are components that live within forge ====
# Showing this here because they don't exist yet.


# A set of atomic entities and their interfaces:
class Trainer(Actor):
    @endpoint
    async def step(self):
        pass


class Policy(Actor):
    @endpoint
    async def generate(self):
        pass


class Rewarder(Actor):
    @endpoint
    async def reward(self):
        pass


class Environment(Actor):
    """"""

    @endpoint
    async def step(self):
        pass


class RolloutActor(Actor):
    @endpoint
    def run_episode(self):
        pass


async def main():
    m = await proc_mesh(gpus=1)


asyncio.run(main())

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A working example showcasing a practical example of forge with RL.

Run this with:
    python -m examples.rl_main
"""

import asyncio
from dataclasses import dataclass

# from functools import partial

from forge.monarch_utils.stack import stack
from forge.rl.configs import GeneratorConfig
from forge.rl.entities import Policy, ReplayBuffer
from forge.rl.generator import Generator

# from monarch.actor_mesh import Actor, endpoint
from monarch.proc_mesh import proc_mesh


@dataclass
class Config:
    pass


async def main():
    # Process allocation
    policy_procs = await proc_mesh(gpus=1)
    replay_procs = await proc_mesh(gpus=1)

    deep_research_procs = await proc_mesh(gpus=1)
    math_procs = await proc_mesh(gpus=1)
    writer_procs = await proc_mesh(gpus=1)

    # Actor instantiation
    replay_buffer = await replay_procs.spawn("replay_buffer", ReplayBuffer)
    policy = await policy_procs.spawn("policy", Policy)

    deep_research_mesh = await deep_research_procs.spawn(
        "deep_research",
        Generator,
    )

    generator_config = GeneratorConfig(
        max_generator_steps=1,
    )
    math_mesh = await math_procs.spawn(
        "math",
        Generator,
        config=generator_config,
        policy=policy,
        # environment_creator=partial(),
    )
    writer_mesh = await writer_procs.spawn("writer", Generator)
    generators = stack(deep_research_mesh, math_mesh, writer_mesh, interface=Generator)


asyncio.run(main())

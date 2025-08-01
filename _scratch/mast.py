# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A basic example of running something on local and on MAST.

To run this:
python -m _scratch.mast.py
"""

import asyncio
import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from forge.controller import ForgeActor, get_proc_mesh
from monarch.actor import endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SchedulerConfig:
    """A sample scheduler config.

    For demonstration purposes: this would be
    handled by our config system.
    """

    scheduler: str = "mast"  # "local" or "mast"
    num_hosts: int = 1
    num_gpus: int = 8
    # The following should probably not be changed.
    oncall: str = "torchtune"
    identity: str = "pytorch_distributed"
    image: str = "forge_workspace:latest"


class TestActor(ForgeActor):
    """Silly actor that computes the world size by all-reducing rank-hot tensors"""

    @endpoint
    async def compute_world_size(self, master_addr: str, master_port: int) -> int:
        backend = "gloo"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        logger.info(
            f"""Initializing process group `{backend}`:
  MASTER_ADDR = {master_addr}
  MASTER_PORT = {master_port}
  RANK        = {self._rank}
  WORLD_SIZE  = {self._size}"""
        )

        dist.init_process_group(backend, rank=self._rank, world_size=self._size)

        try:
            t = F.one_hot(torch.tensor(self._rank), num_classes=dist.get_world_size())
            dist.all_reduce(t)
            return int(torch.sum(t).item())
        finally:
            dist.destroy_process_group()


async def main():
    logging.info("Creating proc mesh...")
    p = await get_proc_mesh(SchedulerConfig())
    logging.info("Proc mesh created: %s", p)
    t = await p.spawn("test", TestActor)
    logging.info("Actor: %s", t)
    c = await t.compute_world_size.call("localhost", 12345)
    c = c._values
    print("result: ", c)


if __name__ == "__main__":
    asyncio.run(main())

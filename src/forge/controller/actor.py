# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
import sys

from forge.controller.proc_mesh import get_proc_mesh, stop_proc_mesh
from forge.types import ProcessConfig

from monarch.actor import Actor, current_rank, current_size, endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ForgeActor(Actor):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_rank"):
            self._rank = current_rank().rank
        if not hasattr(self, "_size"):
            self._size = math.prod(current_size().values())

        # Custom formatter that includes rank/size info with blue prefix
        BLUE = "\033[34m"
        RESET = "\033[0m"
        formatter = logging.Formatter(
            f"{BLUE}[{self.__class__.__name__}-{self._rank}/{self._size}] %(asctime)s %(levelname)s{RESET} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)

        self._proc_mesh = None
        self.logger.root.setLevel(logging.INFO)
        self.logger.root.addHandler(stdout_handler)
        super().__init__(*args, **kwargs)

    # @endpoint
    async def setup(self):
        """Sets up the actor.

        We assume a specific setup function for all actors. The
        best practice for actor deployment is to:
        1. Pass all data to the actor via the constructor.
        2. Call setup() to for heavy weight initializations.

        This is to ensure that any failures during initialization
        can be propagated back to the caller.

        """
        pass

    @classmethod
    async def launch(cls, *, process_config: ProcessConfig, **kwargs) -> "ForgeActor":
        """Provisions and deploys a new actor.

        This method is used by `Service` to provision a new replica.

        We implement it this way because special actors like inference servers
        may be composed of multiple actors spawned across multiple processes.
        This allows you to specify how your actor gets launched together.

        This implementation is basic, assuming that we're spawning
        a homogeneous set of actors on a single proc mesh.

        """
        proc_mesh = await get_proc_mesh(process_config=process_config)

        # TODO - expand support so name can stick within kwargs
        actor_name = kwargs.pop("name", cls.__name__)
        actor = await proc_mesh.spawn(actor_name, cls, **kwargs)
        actor._proc_mesh = proc_mesh

        await actor.setup.call()
        return actor

    @classmethod
    async def shutdown(cls, actor: "ForgeActor"):
        """Shuts down an actor.

        This method is used by `Service` to teardown a replica.
        """
        if actor._proc_mesh is None:
            raise AssertionError("Called shutdown on a replica with no proc_mesh.")
        await stop_proc_mesh(actor._proc_mesh)

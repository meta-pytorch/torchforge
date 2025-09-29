# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from .actor import ForgeActor
from .proc_mesh import get_proc_mesh, stop_proc_mesh


# TODO - remove this once everything has moved to
# service
async def spawn_actors(
    name: str,
    actor_cls: ForgeActor,
    cfg,
    processes,
    set_address: bool = False,
):
    """Setup process Mesh and spawn Actors."""
    mesh = await get_proc_mesh(processes)
    actors = await mesh.spawn(name, actor_cls, **cfg)
    actors.mesh = mesh
    return actors


__all__ = ["spawn_actors", "stop_proc_mesh", "get_proc_mesh", "ForgeActor"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Spawning utils for actors and proc_meshes."""
import logging

from monarch.actor import ProcMesh, HostMesh

from forge.controller.provisioner import (
    Provisioner,
    get_proc_mesh as _get_proc_mesh,
    stop_proc_mesh as _stop_proc_mesh,
)
from forge.types import ProcessConfig

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def get_proc_mesh(
    process_config: ProcessConfig,
    host_mesh: HostMesh | None = None,
    env_vars: dict[str, str] = {}) -> ProcMesh:
    """Returns a proc mesh with the given process config."""
    # TODO - remove this
    return await _get_proc_mesh(process_config, host_mesh=host_mesh, env_vars=env_vars)


async def stop_proc_mesh(mesh: ProcMesh) -> None:
    """Stops the given proc mesh."""
    # TODO - remove this
    return await _stop_proc_mesh(mesh)


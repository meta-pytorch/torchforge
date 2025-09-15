# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Spawning utils for actors and proc_meshes."""
import logging

from monarch.actor import ProcMesh

from forge.controller.provisioner import (
    get_proc_mesh as _get_proc_mesh,
    stop_proc_mesh as _stop_proc_mesh,
)
from forge.types import ProcessConfig

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def get_proc_mesh(process_config: ProcessConfig) -> ProcMesh:
    """Returns a proc mesh with the given process config."""
    # TODO - remove this
    return await _get_proc_mesh(process_config)


async def stop_proc_mesh(mesh: ProcMesh) -> None:
    """Stops the given proc mesh."""
    # TODO - remove this
    return await _stop_proc_mesh(mesh)

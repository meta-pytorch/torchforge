# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements an actor responsible for tracking and assigning GPU devices on HostMesh."""

import logging

from monarch.actor import endpoint, get_or_spawn_controller

from forge.controller import ForgeActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GpuManager(ForgeActor):
    """An actor that tracks and assigns GPU devices on given HostMeshes."""

    def __init__(self):
        # TODO - extend this to support multiple HostMeshes too
        self.available_gpus = set(range(0, 8))

    @endpoint
    def get_gpus(self, num_gpus: int) -> list[str]:
        """Assigns GPU devices."""
        if num_gpus > len(self.available_gpus):
            raise RuntimeError("Not enough GPUs available")
        gpus = list(self.available_gpus)[:num_gpus]
        self.available_gpus -= set(gpus)
        return [str(gpu) for gpu in gpus]

    @endpoint
    def release_gpus(self, gpu_ids: list[str]) -> None:
        """Releases the given GPU devices."""
        for gpu_id in gpu_ids:
            self.available_gpus.add(int(gpu_id))

    def __repr__(self) -> str:
        return "GpuManager"


async def get_gpu_manager() -> GpuManager:
    return await get_or_spawn_controller("gpu_manager", GpuManager)


async def get_gpu_ids(num_gpus: int) -> list[str]:
    gpu_manager = await get_or_spawn_controller("gpu_manager", GpuManager)
    return await gpu_manager.get_gpus.call_one(num_gpus)


async def release_gpus(gpu_ids: list[str]) -> None:
    gpu_manager = await get_or_spawn_controller("gpu_manager", GpuManager)
    await gpu_manager.release_gpus.call_one(gpu_ids)

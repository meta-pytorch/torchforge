# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Remote resource allocation and provisioning."""
import asyncio
import functools
import logging

import os
import socket
import uuid
from typing import Optional

from monarch._src.actor.shape import NDSlice, Shape
from monarch.actor import HostMesh, ProcMesh, this_host
from monarch.tools import commands

from forge.controller.launcher import BaseLauncher, get_launcher

from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.utils import detect_actor_name_from_call_stack

from forge.types import ProcessConfig, ProvisionerConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GpuManager:
    """Tracks and assigns GPU devices on a host.

    This currently mimics the `gpu_manager` in system_controllers - we will
    consolidate as part of the "proper HostMesh integration" work.

    """

    def __init__(self, available_devices: set[int] | None = None):
        if available_devices is None:
            available_devices = set(range(0, 8))
        assert all(isinstance(x, int) for x in available_devices)
        assert all(x >= 0 and x < 8 for x in available_devices)
        self.available_gpus = available_devices

    def get_available_gpus(self) -> list[str]:
        """Returns a list of available GPU devices."""
        return [str(gpu) for gpu in self.available_gpus]

    def get_gpus(self, num_gpus: int) -> list[str]:
        """Assigns GPU devices."""
        if num_gpus > len(self.available_gpus):
            raise RuntimeError("Not enough GPUs available")
        gpus = list(self.available_gpus)[:num_gpus]
        self.available_gpus -= set(gpus)
        return [str(gpu) for gpu in gpus]

    def release_gpus(self, gpu_ids: list[str]) -> None:
        """Releases the given GPU devices."""
        for gpu_id in gpu_ids:
            self.available_gpus.add(int(gpu_id))


class Provisioner:
    """A global resource provisioner."""

    def __init__(self, cfg: ProvisionerConfig | None = None):
        self._server_names = []
        self._proc_server_map = {}
        self._lock = asyncio.Lock()

        # HostMeshes are currently not hashable, so
        # we generate a hash per HostMesh. We'll
        # remove this once this is supported in Monarch.
        self._this_host_id = uuid.uuid1()

        # For the local host, we may want to set CUDA_VISIBLE_DEVICES
        # for small scale testing. We inherit the environment's
        # CUDA_VISIBLE_DEVICES **only for the local host** and not
        # for remote hosts.
        available_local_devices = None
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is not None and cuda_visible_devices.strip():
            try:
                available_local_devices = set(
                    int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid CUDA_VISIBLE_DEVICES format: '{cuda_visible_devices}'. "
                    f"Expected comma-separated integers (e.g., '0,1,2'). Error: {e}"
                ) from e
        self._host_gpu_map = {
            self._this_host_id: GpuManager(available_local_devices),
        }
        self.launcher: BaseLauncher | None = get_launcher(
            cfg.launcher_config if cfg is not None else None
        )
        if not self.launcher:
            logger.warning("Launcher not provided, remote allocations will not work.")

    async def initialize(self):
        """Call this after creating the instance"""
        if self.launcher is not None:
            await self.launcher.initialize()

    async def create_host_mesh(self, name: str, num_hosts: int) -> HostMesh:
        """Creates a remote server and a HostMesh on it."""
        # no need to lock here because this is already locked behind `get_proc_mesh`
        if not self.launcher:
            raise RuntimeError(
                "You tried to create a remote allocation by specifying the number of hosts on an actor or service, "
                "but no launcher was specified."
            )
        logger.debug(f"Creating remote server for alloc {name}")
        alloc, alloc_constraints, server_name = await self.launcher.get_allocator(
            name, num_hosts
        )
        return (
            HostMesh(
                Shape(["hosts"], NDSlice.new_row_major([num_hosts])),
                allocator=alloc,
                alloc_constraints=alloc_constraints,
            ),
            server_name,
        )

    async def get_proc_mesh(
        self,
        num_procs: int,
        with_gpus: bool = False,
        num_hosts: int | None = None,
        mesh_name: Optional[str] = None,
    ):
        """Gets a proc mesh.

        num_hosts = None implies that you want a local allocation, this may change.

        """
        async with self._lock:
            server_name = None
            if num_hosts is not None and num_hosts > 0:
                created_hosts = len(self._server_names)
                host_mesh, server_name = await self.create_host_mesh(
                    name=mesh_name,
                    num_hosts=num_hosts,
                )
                host_id = uuid.uuid1()
                gpu_manager = GpuManager()
                self._host_gpu_map[host_id] = gpu_manager
                host_mesh._host_id = host_id
            else:
                host_mesh = this_host()
                gpu_manager = self._host_gpu_map[self._this_host_id]
                host_mesh._host_id = self._this_host_id

            if with_gpus:
                # The ideal path here:
                # - Create a host mesh
                # - Grab a host from host mesh, from proc 0 spawn an actor that
                # gets addr/port
                # - Spawn procs on the HostMesh with addr/port, setting the
                # addr/port in bootstrap.
                # We can't currently do this because HostMesh only supports single
                # proc_mesh creation at the moment. This will be possible once
                # we have "proper HostMesh support".
                def bootstrap(gpu_ids: list[str]):
                    # This works for single host, needed for vLLM currently.
                    import os

                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
                    os.environ["MASTER_ADDR"] = socket.gethostname()
                    # Multiple actors trying to call _get_port doesn't work
                    # os.environ["MASTER_PORT"] = _get_port()

                    # Setting the last digit to the first GPU id allows us to i.e.
                    # create multiple vLLM instances on the same local host.
                    os.environ["MASTER_PORT"] = f"1234{gpu_ids[0]}"
                    os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
                    os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"

                gpu_ids = gpu_manager.get_gpus(num_procs)
                procs = host_mesh.spawn_procs(
                    per_host={"gpus": num_procs},
                    bootstrap=functools.partial(bootstrap, gpu_ids=gpu_ids),
                )
                # Pick a random host/port, we'll feed this in afterwards
                # Once we have true HostMesh support, we can do this on proc 0 of each host
                # then spin up the proc meshes with the environment afterwards.
                hostname, port = await self.launcher.remote_setup(procs)
                procs._hostname = hostname
                procs._port = port
                procs._gpu_ids = gpu_ids
            else:
                procs = host_mesh.spawn_procs(per_host={"gpus": num_procs})

            procs._host = host_mesh

            # If we created a server, track so we can tear it down later.
            if server_name:
                self._server_names.append(server_name)
                self._proc_server_map[procs] = server_name

        # Detect actor name and spawn local logging actor on each process
        actor_name = detect_actor_name_from_call_stack()
        _ = await get_or_create_metric_logger(procs, actor_name=actor_name)

        return procs

    async def stop_proc_mesh(self, proc_mesh: ProcMesh):
        """Stops a proc mesh."""
        async with self._lock:
            # Deregister local logger from global logger
            if hasattr(proc_mesh, "_local_fetcher"):
                global_logger = await get_or_create_metric_logger(proc_mesh)
                await global_logger.deregister_fetcher.call_one(proc_mesh)

            if hasattr(proc_mesh, "_gpu_ids"):
                gpu_manager = self._host_gpu_map[proc_mesh._host._host_id]
                gpu_manager.release_gpus(proc_mesh._gpu_ids)
            await proc_mesh.stop()
            if proc_mesh in self._proc_server_map:
                server_name = self._proc_server_map[proc_mesh]
                commands.kill(server_name)

    async def shutdown(self):
        """Tears down all remaining remote allocations."""
        async with self._lock:
            for server_name in self._server_names:
                commands.kill(server_name)


_provisioner: Provisioner | None = None


async def init_provisioner(cfg: ProvisionerConfig | None = None):
    global _provisioner
    if not _provisioner:
        _provisioner = Provisioner(cfg)
        await _provisioner.initialize()
    return _provisioner


async def _get_provisioner():
    if not _provisioner:
        await init_provisioner()
    return _provisioner


async def get_proc_mesh(config: ProcessConfig) -> ProcMesh:
    provisioner = await _get_provisioner()
    return await provisioner.get_proc_mesh(
        num_procs=config.procs,
        with_gpus=config.with_gpus,
        num_hosts=config.hosts,
        mesh_name=config.mesh_name,
    )


async def stop_proc_mesh(proc_mesh: ProcMesh):
    provisioner = await _get_provisioner()
    return await provisioner.stop_proc_mesh(proc_mesh=proc_mesh)


async def shutdown():
    logger.info("Shutting down provisioner..")
    provisioner = await _get_provisioner()
    return await provisioner.shutdown()

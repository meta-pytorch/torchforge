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
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import monarch

from forge.observability.metric_actors import setup_metric_logger

from forge.types import ProcessConfig, Scheduler
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch._src.actor.shape import NDSlice, Shape
from monarch.actor import Actor, endpoint, HostMesh, ProcMesh, this_host
from monarch.tools import commands
from monarch.tools.components import hyperactor
from monarch.tools.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        addr = s.getsockname()
        port = addr[1]
        return str(port)


class _SetupActor(Actor):
    @endpoint
    def get_info(self) -> [str, str]:
        return socket.gethostname(), _get_port()


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


class BaseProvisioner(ABC):
    """Abstract base class for resource provisioners."""

    @abstractmethod
    async def create_host_mesh(self, name: str, num_hosts: int) -> HostMesh:
        """Creates a remote server and a HostMesh on it.

        Args:
            name: Name identifier for the host mesh
            num_hosts: Number of hosts to create

        Returns:
            HostMesh: The created host mesh
        """
        pass

    @abstractmethod
    async def get_proc_mesh(
        self,
        num_procs: int,
        with_gpus: bool = False,
        num_hosts: Optional[int] = None,
        mesh_name: Optional[str] = None,
    ) -> ProcMesh:
        """Gets a proc mesh.
        Args:
            num_procs: Number of processes needed
            with_gpus: Whether GPU support is required
            num_hosts: Number of hosts (None implies local allocation)
            mesh_name: Name identifier for the proc mesh

        Returns:
            ProcMesh: The allocated process mesh
        """
        pass

    @abstractmethod
    async def stop_proc_mesh(self, proc_mesh: ProcMesh) -> None:
        """Stops a proc mesh.

        Args:
            proc_mesh: The process mesh to stop
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Tears down all remaining remote allocations."""
        pass


class Provisioner(BaseProvisioner):
    """A global resource provisioner."""

    def __init__(self):
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

    async def create_host_mesh(self, name: str, num_hosts: int) -> HostMesh:
        """Creates a remote server and a HostMesh on it."""
        # no need to lock here because this is already locked behind `get_proc_mesh`
        logger.debug(f"Creating remote server for alloc {name}")
        appdef = hyperactor.host_mesh(
            image="test", meshes=[f"{name}:{num_hosts}:gpu.small"]
        )
        for role in appdef.roles:
            # Note - this is hardcoded to SLURM
            # We got this with sinfo
            role.resource.memMB = 2062607
            role.resource.cpu = 128
            role.resource.gpu = 8

        # TODO - multi scheduler support
        server_config = Config(
            scheduler="slurm",
            appdef=appdef,
            workspace=monarch.tools.config.workspace.Workspace(dirs=[""]),
        )
        server_info = await commands.get_or_create(
            "forge_job",
            server_config,
            force_restart=False,
        )
        alloc = RemoteAllocator(
            world_id=name,
            initializer=TorchXRemoteAllocInitializer(server_info.server_handle),
        )
        server_name = f"slurm:///{server_info.name}"
        return (
            HostMesh(Shape(["hosts"], NDSlice.new_row_major([num_hosts])), alloc),
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
                    name=f"alloc-{created_hosts}",
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
                setup = await procs.spawn(f"setup-{uuid.uuid1()}", _SetupActor)
                # Pick a random host/port, we'll feed this in afterwards
                # Once we have true HostMesh support, we can do this on proc 0 of each host
                # then spin up the proc meshes with the environment afterwards.
                hostname, port = await setup.get_info.choose()
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

        # Spawn local logging actor on each process and register with global logger
        _ = await setup_metric_logger(procs)

        return procs

    async def stop_proc_mesh(self, proc_mesh: ProcMesh):
        """Stops a proc mesh."""
        async with self._lock:
            # Deregister local logger from global logger
            if hasattr(proc_mesh, "_local_fetcher"):
                global_logger = await setup_metric_logger(proc_mesh)
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


_provisioner: BaseProvisioner | None = None


def _get_provisioner(scheduler: Scheduler = Scheduler.LOCAL):
    global _provisioner
    if not _provisioner:
        if scheduler == Scheduler.MAST.value:
            from forge.controller.launcher.mast import MastProvisioner

            _provisioner = MastProvisioner("rithesh-forge-c8072a")
        else:
            _provisioner = Provisioner()
    return _provisioner


async def get_proc_mesh(config: ProcessConfig) -> ProcMesh:
    return await _get_provisioner(config.scheduler).get_proc_mesh(
        num_procs=config.procs,
        with_gpus=config.with_gpus,
        num_hosts=config.hosts,
        mesh_name=config.mesh_name,
    )


async def stop_proc_mesh(proc_mesh: ProcMesh):
    return await _get_provisioner().stop_proc_mesh(proc_mesh=proc_mesh)


async def shutdown():
    logger.info("Shutting down provisioner..")
    await _get_provisioner().shutdown()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Remote resource allocation and provisioning."""
from monarch.actor import this_host, HostMesh, Actor, endpoint, ProcMesh
import monarch
from monarch.tools.components import hyperactor
from monarch.tools import commands
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch.tools.config import Config
from monarch._src.actor.shape import Shape, NDSlice
import functools
import socket
import uuid
from forge.types import ProcessConfig


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
    """Tracks and assigns GPU devices on a host mesh."""

    def __init__(self):
        # TODO - extend this to support multiple HostMeshes too
        self.available_gpus = set(range(0, 8))

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
    def __init__(self):
        self._server_names = []
        self._proc_server_map = {}

        # TODO - HostMeshes are not hashabel
        self._this_host_id = uuid.uuid1()
        self._host_gpu_map = {
            self._this_host_id: GpuManager(),
        }

    async def create_host_mesh(self, name: str, num_hosts: int) -> RemoteAllocator:
        print(f"for {name}, creating {num_hosts}")
        appdef = hyperactor.host_mesh(
            image="test",
            meshes=[f"{name}:{num_hosts}:gpu.small"]
        )
        for role in appdef.roles:
            # Note - this is hardcoded to SLURM 
            # We got this with sinfo
            role.resource.memMB = 2062607
            role.resource.cpu = 128
            role.resource.gpu = 8

        # TODO - multi scheduler
        server_config = Config(
            scheduler="slurm",
            appdef=appdef,
            workspace=monarch.tools.config.workspace.Workspace(dirs=[""]),
        )
        print("Creating server")
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
        return HostMesh(Shape(["hosts"], NDSlice.new_row_major([num_hosts])), alloc), server_name

    async def get_proc_mesh(self, num_procs: int, with_gpus: bool = False, num_hosts: int | None = None):
        """TODO - None hosts means don't use remote alloc"""
        # TODO - issues/144
        server_name = None
        if num_hosts is not None and num_hosts > 0:
            print("using remote mesh")
            host_mesh, server_name = await self.create_host_mesh(name=f"alloc-{num_hosts}", num_hosts=num_hosts)

            host_id = uuid.uuid()
            gpu_manager = GpuManager()
            self._host_gpu_map[host_id] = gpu_manager
            host_mesh._host_id = host_id
        else:
            print("using local host mesh")
            host_mesh = this_host()
            gpu_manager = self._host_gpu_map[self._this_host_id]

        if with_gpus:
            # Known issues:
            # - We don't do any intelligent resource management here, just assuming every
            # resource asked for is placed on its own host.
            def bootstrap(gpu_ids: int):
                # These env variables will work for single host, but on multi-host
                # we will need to override env variables.
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
                os.environ["MASTER_ADDR"] = socket.gethostname()
                # Multiple actors trying to call _get_port doesn't work
                # os.environ["MASTER_PORT"] = _get_port()
                os.environ["MASTER_PORT"] = "12345"

            gpu_ids = gpu_manager.get_gpus(num_procs)
            procs = host_mesh.spawn_procs(
                per_host={"gpus": num_procs},
                bootstrap=functools.partial(
                    bootstrap, gpu_ids=gpu_ids),
            )
            setup = await procs.spawn(f"setup-{uuid.uuid1()}", _SetupActor)
            # Pick a random host/port, we'll feed this in afterwards
            # Once we have true HostMesh support, we can do this on proc 0 of each host
            # then spin up the proc meshes with the environment afterwards.
            hostname, port = await setup.get_info.choose()
            procs._host = hostname
            procs._addr = port
            procs._gpu_ids = gpu_ids
        else:
            procs = host_mesh.spawn_procs(
                per_host={"gpus": num_procs}
            )

        procs._host = host_mesh
        if server_name:
            self._server_names.append(server_name)
            self._proc_server_name[procs] = server_name

        return procs

    async def stop_proc_mesh(self, procs: ProcMesh):
        if procs._gpu_ids:
            gpu_manager = self._host_gpu_map[procs._host._host_id]
            gpu_manager.release_gpus(procs._gpu_ids)
        await procs.stop()
        if procs in self._proc_server_map:
            server_name = self._proc_server_map[procs]
            print("Shutting down ", server_name)
            commands.kill(server_name)

    async def shutdown(self):
        for server_name in self._server_names:
            print(f"shutting down {server_name}")
            commands.kill(server_name)


_provisioner: Provisioner | None = None


def _get_provisioner():
    global _provisioner
    if not _provisioner:
        _provisioner = Provisioner()
    return _provisioner


async def get_proc_mesh(config: ProcessConfig) -> ProcMesh:
    # note - num_hosts = None indicates use local
    return await _get_provisioner().get_proc_mesh(
        num_procs=config.num_procs,
        with_gpus=config.with_gpus,
        num_hosts=config.num_hosts)


async def stop_proc_mesh(proc_mesh: ProcMesh):
    return await _get_provisioner().stop_proc_mesh(
        proc_mesh=proc_mesh
    )


async def shutdown():
    await _get_provisioner().shutdown()

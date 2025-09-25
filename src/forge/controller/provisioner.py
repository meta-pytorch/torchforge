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
import subprocess
import uuid

import monarch

from forge.types import ProcessConfig
from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch._src.actor.meta.allocator import MastAllocator, MastAllocatorConfig
from monarch._src.actor.shape import NDSlice, Shape
from monarch.actor import Actor, endpoint, HostMesh, ProcMesh, this_host
from monarch._src.actor.actor_mesh import Actor, current_rank
from monarch.tools import commands
from monarch.tools.components import hyperactor
from monarch.tools.config import Config


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import getpass

USER = getpass.getuser()
WORK_DIR = f"/data/users/{USER}"  # on DEVGPU
WOKR_DIR_MAST = f"/home/{USER}"  # on MAST
EDITABLE_WORKSPACES = ["forge"]

EDITABLE_WORKSPACE_PATHS = [
    f"{WORK_DIR}/{workspace}" for workspace in EDITABLE_WORKSPACES
]

JOB_NAME = "rithesh-forge-grpo-eccb6f"


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

    @endpoint
    def mount(self, mount_dst: str, procs_per_host: int):
        assert procs_per_host > 0
        if current_rank().rank % procs_per_host != 0:
            # Only use one rank per host to mount the directory
            return
        self.mount_mnt_directory(mount_dst)

    @endpoint
    def check_path(self, path) -> bool:
        if os.path.exists(path):
            return True
        return False

    @endpoint
    def list_files(self, path) -> list[str]:
        if os.path.exists(path):
            return os.listdir(path)
        return ["NO FILES EXIST"]

    def mount_mnt_directory(self, mount_dst: str) -> None:
        # Sanity check of the mounted directory
        sanity_path = os.path.join(mount_dst, "huggingface_models/")
        if os.path.exists(sanity_path):
            print(f"Found directory {sanity_path}; skip mounting.")
            return

        # Otherwise, mount the directory
        if not os.path.exists(mount_dst):
            os.makedirs(mount_dst, exist_ok=True)

        # Store original LD_LIBRARY_PATH to restore after mounting
        original_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

        try:
            # Create an environment without LD_LIBRARY_PATH
            # libc.so.6 version within LD_LIBRARY_PATH is too old causing error like
            # fbcode/platform010/lib/libc.so.6: version `GLIBC_ABI_DT_RELR' not found (required by /usr/bin/env)
            # /usr/bin/env: /usr/local/fbcode/platform010/lib/libc.so.6: version `GLIBC_2.38' not found (required by /usr/bin/env)

            clean_env = os.environ.copy()
            if "LD_LIBRARY_PATH" in clean_env:
                del clean_env["LD_LIBRARY_PATH"]

            subprocess.run(
                [
                    "/packages/oil.oilfs/oilfs-wrapper",
                    "ws://ws.ai.pci0ai/genai_fair_llm",
                    mount_dst,
                ],
                capture_output=True,
                text=True,
                check=True,
                env=clean_env,
            )
            print("Done mounting")
        except subprocess.CalledProcessError as e:
            print(
                f"Get error during mounting {e}, Stderr: {e.stderr}, Stdout: {e.stdout}"
            )
        finally:
            # Restore original LD_LIBRARY_PATH
            if original_ld_library_path:
                os.environ["LD_LIBRARY_PATH"] = original_ld_library_path
            elif "LD_LIBRARY_PATH" in os.environ:
                del os.environ["LD_LIBRARY_PATH"]

        assert os.path.exists(
            sanity_path
        ), f"Did not find directory {sanity_path}; something wrong with mounting."


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

    async def get_mast_allocator(
        self,
        task_group: str = "mesh0",
        monarch_port: int = 26600,
        num_hosts: int = 1,
        num_gpus: int = 8,
    ):
        job_name = JOB_NAME
        allocator = MastAllocator(
            MastAllocatorConfig(
                job_name=job_name,
                remote_allocator_port=26600,  # This is the default monarch port
            ),
        )
        spec = AllocSpec(
            AllocConstraints({MastAllocator.ALLOC_LABEL_TASK_GROUP: task_group}),
            hosts=num_hosts,
            gpus=num_gpus,
        )
        # allocation = await allocator.allocate(spec)
        # return allocation
        alloc_constraints = AllocConstraints(
            {MastAllocator.ALLOC_LABEL_TASK_GROUP: task_group}
        )

        return allocator, alloc_constraints

    async def create_host_mesh(self, name: str, num_hosts: int):
        """Creates a remote server and a HostMesh on it."""
        # no need to lock here because this is already locked behind `get_proc_mesh`
        logger.debug(f"Creating remote server for alloc {name}")
        # appdef = hyperactor.host_mesh(
        #     image="test", meshes=[f"{name}:{num_hosts}:gpu.small"]
        # )
        # for role in appdef.roles:
        #     # Note - this is hardcoded to SLURM
        #     # We got this with sinfo
        #     role.resource.memMB = 2062607
        #     role.resource.cpu = 128
        #     role.resource.gpu = 8

        # # TODO - multi scheduler support
        # server_config = Config(
        #     scheduler="slurm",
        #     appdef=appdef,
        #     workspace=monarch.tools.config.workspace.Workspace(dirs=[""]),
        # )
        # server_info = await commands.get_or_create(
        #     "forge_job",
        #     server_config,
        #     force_restart=False,
        # )
        # alloc = RemoteAllocator(
        #     world_id=name,
        #     initializer=TorchXRemoteAllocInitializer(server_info.server_handle),
        # )
        # server_name = f"slurm:///{server_info.name}"
        # return (
        #     HostMesh(Shape(["hosts"], NDSlice.new_row_major([num_hosts])), alloc),
        #     server_name,
        # )
        server_name = f"mast_conda:///{JOB_NAME}"

        # allocation = await self.get_mast_allocator(task_group=name)
        # return allocation, server_name

        alloc, alloc_constraints = await self.get_mast_allocator(task_group=name)
        return (
            HostMesh(
                shape=Shape(["hosts"], NDSlice.new_row_major([num_hosts])),
                allocator=alloc,
                alloc_constraints=alloc_constraints,
            ),
            server_name,
        )

    async def get_proc_mesh(
        self, num_procs: int, with_gpus: bool = False, num_hosts: int | None = None
    ):
        """Gets a proc mesh.

        num_hosts = None implies that you want a local allocation, this may change.

        """
        async with self._lock:
            server_name = None
            if num_hosts is not None and num_hosts > 0:
                created_hosts = len(self._server_names)
                _name = f"policy"
                host_mesh, server_name = await self.create_host_mesh(
                    name=_name,
                    num_hosts=num_hosts,
                )
                # allocation, server_name = await self.create_host_mesh(
                #     name=_name,
                #     num_hosts=num_hosts,
                # )
                host_id = uuid.uuid1()
                gpu_manager = GpuManager()
                self._host_gpu_map[host_id] = gpu_manager
                # host_mesh._host_id = host_id
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
                # procs = await ProcMesh.from_alloc(
                #     allocation, functools.partial(bootstrap, gpu_ids=gpu_ids)
                # )
                await procs.initialized
                # await procs.sync_workspace(
                #     workspace=monarch.tools.config.workspace.Workspace(
                #         dirs=[workspace_dir for workspace_dir in EDITABLE_WORKSPACES],
                #     ),
                #     conda=True,
                #     auto_reload=False,
                # )
                setup = await procs.spawn(f"setup-{uuid.uuid1()}", _SetupActor)
                # Pick a random host/port, we'll feed this in afterwards
                # Once we have true HostMesh support, we can do this on proc 0 of each host
                # then spin up the proc meshes with the environment afterwards.
                hostname, port = await setup.get_info.choose()
                await setup.mount.choose(mount_dst="/mnt/wsfuse", procs_per_host=8)
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

        return procs

    async def stop_proc_mesh(self, proc_mesh: ProcMesh):
        """Stops a proc mesh."""
        async with self._lock:
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


def _get_provisioner():
    global _provisioner
    if not _provisioner:
        _provisioner = Provisioner()
    return _provisioner


async def get_proc_mesh(config: ProcessConfig) -> ProcMesh:
    return await _get_provisioner().get_proc_mesh(
        num_procs=config.procs,
        with_gpus=config.with_gpus,
        num_hosts=config.hosts,
    )


async def stop_proc_mesh(proc_mesh: ProcMesh):
    return await _get_provisioner().stop_proc_mesh(proc_mesh=proc_mesh)


async def shutdown():
    logger.info("Shutting down provisioner..")
    await _get_provisioner().shutdown()

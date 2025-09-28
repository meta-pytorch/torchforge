# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import functools
import getpass
import logging
import os
import socket
import subprocess
import uuid
from typing import Optional

import torchx.specs as specs

from forge.controller.provisioner import BaseProvisioner, GpuManager, JOB_NAME_KEY
from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints
from monarch._src.actor.meta.allocator import MastAllocator, MastAllocatorConfig

from monarch._src.actor.shape import NDSlice, Shape
from monarch.actor import Actor, endpoint, HostMesh, ProcMesh, this_host
from monarch._src.actor.actor_mesh import Actor, current_rank
from monarch.tools import commands
from monarch.tools.commands import info
from monarch.tools.components.meta import hyperactor
from monarch.tools.config import Config, Workspace
from omegaconf import DictConfig
from torchx.specs import AppState
from torchx.specs.fb.component_helpers import Packages

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SCHEDULER_NAME = "mast_conda"
SKU = "gtt_any"
TIMEOUT_SEC = 1 * 60 * 60  # Kill the job if idle for 1 hour

USER = getpass.getuser()
WORK_DIR = f"/data/users/{USER}"  # on DEVGPU
EDITABLE_WORKSPACES = ["forge"]
REMOTE_WORK_DIR = "/packages/monarch_default_workspace/workspace/"

EDITABLE_WORKSPACE_PATHS = [
    f"{WORK_DIR}/{workspace}" for workspace in EDITABLE_WORKSPACES
]


def _get_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        addr = s.getsockname()
        port = addr[1]
        return str(port)


class MastSetupActor(Actor):
    @endpoint
    def get_info(self) -> [str, str]:
        return socket.gethostname(), _get_port()

    @endpoint
    def mount(self, mount_dst: str):
        point = current_rank()
        # The last dimension is the local proc count.
        last_label = point.extent.labels[-1]
        proc_count = point.size(last_label)
        if current_rank().rank % proc_count != 0:
            # Only use one rank per host to mount the directory
            return
        self.mount_mnt_directory(mount_dst)

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


class MastProvisioner(BaseProvisioner):
    def __init__(self, cfg: DictConfig | None = None):
        self._server_names = []
        self._proc_server_map = {}
        self._lock = asyncio.Lock()
        self._this_host_id = uuid.uuid1()
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
        assert cfg is not None
        self.cfg = cfg
        job_name = cfg.get(JOB_NAME_KEY, None)
        self.job_name = job_name or self.create_job_name()

    async def initialize(self):
        """Call this after creating the instance"""
        await self.launch_mast_job()

    async def get_mast_allocator(
        self,
        job_name: str,
        task_group: str,
    ):
        allocator = MastAllocator(
            MastAllocatorConfig(
                job_name=job_name,
                remote_allocator_port=26600,  # This is the default monarch port
            ),
        )
        alloc_constraints = AllocConstraints(
            {MastAllocator.ALLOC_LABEL_TASK_GROUP: task_group}
        )

        return allocator, alloc_constraints

    async def create_host_mesh(self, name: str, num_hosts: int):
        """Creates a remote server and a HostMesh on it."""
        logger.debug(f"Creating remote server for mesh: {name}")
        server_name = f"{SCHEDULER_NAME}:///{self.job_name}"
        alloc, alloc_constraints = await self.get_mast_allocator(
            task_group=name, job_name=self.job_name
        )
        return (
            HostMesh(
                shape=Shape(["hosts"], NDSlice.new_row_major([num_hosts])),
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
                assert mesh_name is not None
                host_mesh, server_name = await self.create_host_mesh(
                    name=mesh_name,
                    num_hosts=num_hosts,
                )
                host_id = uuid.uuid1()
                gpu_manager = GpuManager()
                self._host_gpu_map[host_id] = gpu_manager
            else:
                host_mesh = this_host()
                gpu_manager = self._host_gpu_map[self._this_host_id]
                host_mesh._host_id = self._this_host_id

            if with_gpus:

                def bootstrap(gpu_ids: list[str]):
                    # This works for single host, needed for vLLM currently.
                    import os

                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
                    os.environ["MASTER_ADDR"] = socket.gethostname()
                    os.environ["MASTER_PORT"] = f"1234{gpu_ids[0]}"
                    os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "600"
                    os.environ["HYPERACTOR_CODE_MAX_FRAME_LENGTH"] = "1073741824"

                gpu_ids = gpu_manager.get_gpus(num_procs)
                procs = host_mesh.spawn_procs(
                    per_host={"gpus": num_procs},
                    bootstrap=functools.partial(bootstrap, gpu_ids=gpu_ids),
                )
                await procs.initialized
                setup = await procs.spawn(f"setup-{uuid.uuid1()}", MastSetupActor)
                hostname, port = await setup.get_info.choose()
                await setup.mount.call(mount_dst="/mnt/wsfuse")
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

    async def launch_mast_job(self):
        handle = self.create_server_handle()
        server_spec = info(handle)
        if server_spec and server_spec.state == AppState.RUNNING:
            print(f"Job {self.job_name} is already running. Skipping launch.")
            return server_spec

        config = Config(
            scheduler="mast_conda",
            scheduler_args={
                # NOTE: TODO: support passing these args from CLI
                "hpcIdentity": "genai_llm_pretraining_data",
                "hpcJobOncall": "monarch",
                "hpcClusterUuid": "MastProdCluster",
                "rmAttribution": "pytorch4all_clients_approved",
                # "hpcClusterUuid": "MastGenAICluster",
                # "rmAttribution": "gen_ai_llama_systems_training",
                # "localityConstraints": ["region", "pci"],
            },
            appdef=self.build_appdef(),
            workspace=Workspace(
                dirs=[workspace_dir for workspace_dir in EDITABLE_WORKSPACE_PATHS],
            ),
        )

        await commands.get_or_create(self.job_name, config)
        return server_spec

    def add_additional_packages(self, packages: Packages) -> Packages:
        packages.add_package("oil.oilfs:stable")
        packages.add_package("manifold.manifoldfs")
        return packages

    def build_appdef(self) -> specs.AppDef:

        # create the app definition for the worker
        REMOTE_END_PYTHONPATH = ":".join(
            [f"{REMOTE_WORK_DIR}{workspace}" for workspace in EDITABLE_WORKSPACE_PATHS]
        )

        default_envs = {
            **hyperactor.DEFAULT_NVRT_ENVS,
            **hyperactor.DEFAULT_NCCL_ENVS,
            **hyperactor.DEFAULT_TORCH_ENVS,
            **{"TORCHX_RUN_PYTHONPATH": f"{REMOTE_END_PYTHONPATH}:{REMOTE_WORK_DIR}"},
            **{
                "HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS": "600",
                "HYPERACTOR_CODE_MAX_FRAME_LENGTH": "1073741824",
            },
        }

        packages = Packages()
        meshes = []
        for mesh_name, config in self.cfg["services"].items():
            num_replicas = config["num_replicas"]
            with_gpus = bool(config["with_gpus"])
            num_hosts = int(config.get("hosts", 0))
            # Create list of mesh names with indices and num_hosts
            if with_gpus and num_hosts > 0:
                mesh_list = [
                    f"{mesh_name}_{i}:{num_hosts}:{SKU}" for i in range(num_replicas)
                ]
                meshes.extend(mesh_list)

        appdef = hyperactor.host_mesh_conda(
            meshes=meshes,
            additional_packages=self.add_additional_packages(packages),
            timeout_sec=TIMEOUT_SEC,
            env=default_envs,
        )

        for role in appdef.roles:
            role.resource.capabilities["server_sub_types"] = [
                # role.resource.capabilities["server_sub_types"][2]  # hardcoded to ROCE
                role.resource.capabilities["server_sub_types"][1]  # GTT
            ]

        return appdef

    def create_job_name(self):
        return f"{USER}-forge-{uuid.uuid4().hex[:6]}"

    def create_server_handle(self) -> str:
        return f"{SCHEDULER_NAME}:///{self.job_name}"

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import os
import subprocess

import tempfile
import uuid
from typing import Any

import monarch

import torchx.specs as specs

from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch.actor import Actor, endpoint, ProcMesh
from monarch.tools import commands
from monarch.tools.commands import info
from monarch.tools.components import hyperactor
from monarch.tools.config import Config, Workspace

from forge.types import Launcher, LauncherConfig

_MAST_AVAILABLE = False

try:
    from monarch._src.actor.actor_mesh import current_rank
    from monarch._src.actor.meta.allocator import MastAllocator, MastAllocatorConfig
    from monarch.tools.components.meta import hyperactor as meta_hyperactor
    from torchx.specs import AppState
    from torchx.specs.fb.component_helpers import Packages

    _MAST_AVAILABLE = True
except ImportError as e:
    # This means there is an error with MAST
    pass

JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"


class MastSetupActor(Actor):
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


class BaseLauncher:
    async def initialize(self) -> None:
        pass

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        pass

    async def remote_setup(self, procs: ProcMesh) -> tuple[str, int]:
        pass


class Slurmlauncher(BaseLauncher):
    async def initialize(self) -> None:
        pass

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        appdef = hyperactor.host_mesh(
            image="test", meshes=[f"{name}:{num_hosts}:gpu.small"]
        )
        for role in appdef.roles:
            # Note - this is hardcoded to SLURM
            # We got this with sinfo
            role.resource.memMB = 2062607
            role.resource.cpu = 128
            role.resource.gpu = 8

        # Note - we cannot add in an empty workspace, so we create a fake temporary one
        temp_workspace = tempfile.mkdtemp(prefix="forge_workspace_")
        server_config = Config(
            scheduler="slurm",
            appdef=appdef,
            workspace=monarch.tools.config.workspace.Workspace(dirs=[temp_workspace]),
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
        return alloc, None, server_name  # (Allocator, AllocConstraints, SeverName)

    async def remote_setup(self, procs: ProcMesh) -> tuple[str, int]:
        return


class Mastlauncher(BaseLauncher):
    def __init__(self, cfg: LauncherConfig | None = None):
        assert cfg is not None
        self.cfg = cfg
        self.default_monarch_port = 26600
        self.scheduler_name = "mast_conda"

        # TODO: enabe taking this from config
        self.sku = "gtt_any"
        self.timeout_sec = 1 * 60 * 60  # Kill the job if idle for 1 hour
        self.user = getpass.getuser()
        self.work_dir = f"/data/users/{self.user}"
        self.edittable_workspaces = ["forge"]
        self.remote_work_dir = "/packages/monarch_default_workspace/workspace/"
        self.editable_workspace_paths = [
            f"{self.work_dir}/{workspace}" for workspace in self.edittable_workspaces
        ]
        self.job_name = self.cfg.job_name or self.create_job_name()

    async def initialize(self) -> None:
        await self.launch_mast_job()

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        allocator = MastAllocator(
            MastAllocatorConfig(
                job_name=self.job_name,
                remote_allocator_port=self.default_monarch_port,
            ),
        )
        alloc_constraints = AllocConstraints(
            {MastAllocator.ALLOC_LABEL_TASK_GROUP: name}
        )

        return allocator, alloc_constraints, self.create_server_handle()

    async def remote_setup(self, procs: ProcMesh) -> tuple[str, int]:
        setup = procs.spawn(f"setup-{uuid.uuid1()}", MastSetupActor)
        await setup.mount.call(mount_dst="/mnt/wsfuse")
        return await setup.get_info.choose()

    async def launch_mast_job(self):
        handle = self.create_server_handle()
        server_spec = info(handle)
        if server_spec and server_spec.state == AppState.RUNNING:
            print(f"Job {self.job_name} is already running. Skipping launch.")
            return server_spec

        config = Config(
            scheduler="mast_conda",
            scheduler_args={
                "hpcIdentity": "hyper_monarch",
                "hpcJobOncall": "monarch",
                "hpcClusterUuid": "MastProdCluster",
                "rmAttribution": "pytorch4all_clients_approved",
            },
            appdef=self.build_appdef(),
            workspace=Workspace(
                dirs=[workspace_dir for workspace_dir in self.editable_workspace_paths],
            ),
        )

        await commands.get_or_create(self.job_name, config)
        return server_spec

    def add_additional_packages(self, packages: "Packages") -> "Packages":
        packages.add_package("oil.oilfs:stable")
        packages.add_package("manifold.manifoldfs")
        return packages

    def build_appdef(self) -> specs.AppDef:

        # create the app definition for the worker
        remote_end_python_path = ":".join(
            [
                f"{self.remote_work_dir}{workspace}"
                for workspace in self.editable_workspace_paths
            ]
        )

        default_envs = {
            **meta_hyperactor.DEFAULT_NVRT_ENVS,
            **meta_hyperactor.DEFAULT_NCCL_ENVS,
            **meta_hyperactor.DEFAULT_TORCH_ENVS,
            **{
                "TORCHX_RUN_PYTHONPATH": f"{remote_end_python_path}:{self.remote_work_dir}"
            },
            **{
                "HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS": "600",
                "HYPERACTOR_CODE_MAX_FRAME_LENGTH": "1073741824",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "TORCH_COMPILE_DISABLE": "1",
                "TORCHDYNAMO_VERBOSE": "1",
                "VLLM_TORCH_COMPILE_LEVEL": "0",
                "VLLM_USE_TRITON_FLASH_ATTN": "0",
            },
        }

        print("DEFAULT ENVS: ", default_envs)

        packages = Packages()
        meshes = []
        # Process both services and actors configurations
        for mesh_name, service in self.cfg.services.items():
            num_replicas = service.num_replicas
            with_gpus = bool(service.with_gpus)
            num_hosts = int(service.hosts or 0)
            # Create list of mesh names with indices and num_hosts
            if with_gpus and num_hosts > 0:
                mesh_list = [
                    f"{mesh_name}_{i}:{num_hosts}:{self.sku}"
                    for i in range(num_replicas)
                ]
                meshes.extend(mesh_list)

        for mesh_name, actor in self.cfg.actors.items():
            num_replicas = 1
            with_gpus = bool(actor.with_gpus)
            num_hosts = int(actor.hosts or 0)
            # single actors with GPUs
            if with_gpus:
                meshes.append(f"{mesh_name}:{num_replicas}:{self.sku}")

        appdef = meta_hyperactor.host_mesh_conda(
            meshes=meshes,
            additional_packages=self.add_additional_packages(packages),
            timeout_sec=self.timeout_sec,
            env=default_envs,
        )

        for role in appdef.roles:
            role.resource.capabilities["server_sub_types"] = [
                # role.resource.capabilities["server_sub_types"][2]  # hardcoded to ROCE
                role.resource.capabilities["server_sub_types"][1]  # GTT
            ]

        return appdef

    def create_job_name(self):
        return f"{self.user}-forge-{uuid.uuid4().hex[:6]}"

    def create_server_handle(self) -> str:
        return f"{self.scheduler_name}:///{self.job_name}"


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher()
    elif cfg.launcher == Launcher.MAST:
        if not _MAST_AVAILABLE:
            raise ValueError(
                "MAST imports did not succeed, cannot launch MAST jobs. Please verify your installation"
            )
        return Mastlauncher(cfg)
    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")

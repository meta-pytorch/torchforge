# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher specific logic (i.e. SLURM, k8s when supported, etc.)"""

import copy
import getpass
import os
import subprocess
import tempfile
import uuid
from typing import Any

import monarch
import torchx.specs as specs

from forge.types import Launcher, LauncherConfig
from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch.actor import Actor, endpoint, ProcMesh
from monarch.tools import commands
from monarch.tools.commands import create, info
from monarch.tools.components import hyperactor
from monarch.tools.config import Config, Workspace

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


def mount_mnt_directory(mount_dst: str) -> None:
    """Mounts the MAST remote directory to the specified destination.

    This function mounts a remote workspace directory that contains huggingface models
    and other shared resources needed for training.

    Args:
        mount_dst: Destination path where the directory should be mounted (e.g., "/mnt/wsfuse")
    """
    # Sanity check of the mounted directory
    sanity_path = os.path.join(mount_dst, "huggingface_models/")
    if os.path.exists(sanity_path):
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
        print(f"Get error during mounting {e}, Stderr: {e.stderr}, Stdout: {e.stdout}")
    finally:
        # Restore original LD_LIBRARY_PATH
        if original_ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = original_ld_library_path
        elif "LD_LIBRARY_PATH" in os.environ:
            del os.environ["LD_LIBRARY_PATH"]

    assert os.path.exists(
        sanity_path
    ), f"Did not find directory {sanity_path}; something wrong with mounting."


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
        mount_mnt_directory(mount_dst)


class BaseLauncher:
    async def initialize(self) -> None:
        pass

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        pass

    async def remote_setup(self, procs: ProcMesh) -> None:
        pass


class Slurmlauncher(BaseLauncher):
    async def initialize(self) -> None:
        # HostMesh currently requires explicit configuration
        # of the underlying transport from client to mesh.
        # This can be removed in the future once this has been removed.
        configure(default_transport=ChannelTransport.TcpWithHostname)

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

    async def remote_setup(self, procs: ProcMesh) -> None:
        return


class MastLauncher(BaseLauncher):
    """Launcher for MAST (Meta's internal cluster scheduler).

    This launcher supports two modes of operation:

    1. Non-detached mode (detached=False):
       - Client runs on your local machine/devserver
       - Only worker roles (GPU hosts) are launched in MAST
       - Client connects to workers remotely via provisioner

    2. Detached mode (detached=True):
       - Client runs entirely inside MAST as a separate role
       - Both client role (CPU-only) and worker roles (GPU) are launched in MAST
       - Client role executes the training script with --mode=remote
       - Everything runs in the cluster, no client needed on local machine

    Args:
        cfg: Launcher configuration including job name, services, and actors
        detached: If True, adds a client role to the MAST job appdef that runs
                  the training script inside MAST. If False, only launches worker
                  roles and expects the client to run on local machine.
        extra_args: Additional CLI arguments to pass through to the client role.

    """

    def __init__(
        self,
        cfg: LauncherConfig | None = None,
        detached: bool = False,
        extra_args: list = None,
    ):
        assert cfg is not None
        self.cfg = cfg
        self.detached = detached
        self.default_monarch_port = 26600
        self.extra_args = extra_args or []
        self.scheduler_name = "mast_conda"

        # TODO: enable taking this from config
        self.sku = "gtt_any"
        self.timeout_sec = 1 * 60 * 60  # Kill the job if idle for 1 hour
        self.user = getpass.getuser()
        self.work_dir = f"/home/{self.user}"
        self.edittable_workspaces = ["torchforge"]
        self.remote_work_dir = "/packages/monarch_default_workspace/workspace/"
        self.editable_workspace_paths = [
            f"{self.work_dir}/{workspace}" for workspace in self.edittable_workspaces
        ]
        self.job_name = self.cfg.job_name or self.create_job_name()

    async def initialize(self) -> None:
        # HostMesh currently requires explicit configuration
        # of the underlying transport from client to mesh.
        # This can be removed in the future once this has been removed.
        configure(default_transport=ChannelTransport.MetaTlsWithHostname)

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

    async def remote_setup(self, procs: ProcMesh) -> None:
        setup = procs.spawn("mast_setup", MastSetupActor)
        await setup.mount.call(mount_dst="/mnt/wsfuse")

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
                "hpcClusterUuid": "MastGenAICluster",
                "rmAttribution": "msl_infra_hw_enab_agentrl",
            },
            appdef=self.build_appdef(),
            workspace=Workspace(
                dirs=[workspace_dir for workspace_dir in self.editable_workspace_paths],
            ),
        )

        job_handle = create(config, name=self.job_name)
        print(
            f"MAST job launched successfully:\n"
            f"\033[92mhttps://www.internalfb.com/mlhub/pipelines/runs/mast/{self.job_name}\033[0m"
        )
        return job_handle

    def add_additional_packages(self, packages: "Packages") -> "Packages":
        packages.add_package("oil.oilfs:stable")
        packages.add_package("manifold.manifoldfs:prod")
        return packages

    def build_appdef(self) -> specs.AppDef:
        # create the app definition for the worker
        additional_python_paths = [
            f"{self.remote_work_dir}{workspace}"
            for workspace in self.editable_workspace_paths
        ]
        additional_python_paths.append(self.remote_work_dir)

        # needed for wandb api key extraction from secret
        additional_python_paths.append("/packages/cif")
        default_envs = {
            **meta_hyperactor.DEFAULT_NVRT_ENVS,
            **meta_hyperactor.DEFAULT_NCCL_ENVS,
            **meta_hyperactor.DEFAULT_TORCH_ENVS,
            **{"TORCHX_RUN_PYTHONPATH": ":".join(additional_python_paths)},
            **{
                "HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS": "600",
                "HYPERACTOR_CODE_MAX_FRAME_LENGTH": "1073741824",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "TORCH_COMPILE_DISABLE": "1",
                "TORCHDYNAMO_VERBOSE": "1",
                "VLLM_TORCH_COMPILE_LEVEL": "0",
                "VLLM_USE_TRITON_FLASH_ATTN": "0",
                "HF_HUB_OFFLINE": "1",
                "MONARCH_HOST_MESH_V1_REMOVE_ME_BEFORE_RELEASE": "1",
                "TORCHSTORE_RDMA_ENABLED": "1",
                "HF_HOME": "/mnt/wsfuse/teamforge/hf",
                "TRANSFORMERS_OFFLINE": "1",
            },
        }

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
        appdef.metadata["mast"] = {
            "HpcJobDefinition": {
                "networkAffinity": {
                    # Ensure colocation
                    "preferredScope": 3,  # DC
                    "fallbackScope": 3,  # REGION
                },
            },
        }

        for role in appdef.roles:
            role.resource.capabilities["server_sub_types"] = [
                # role.resource.capabilities["server_sub_types"][2]  # hardcoded to ROCE
                role.resource.capabilities["server_sub_types"][1]  # GTT
            ]

        # Add client role to run in MAST if in detached mode
        if self.detached:
            client_role = self._create_client_role(appdef)
            appdef.roles.insert(0, client_role)

        return appdef

    def _create_client_role(self, appdef: specs.AppDef) -> specs.Role:
        # Clone an existing worker role to inherit workspace configuration
        if not appdef.roles:
            raise ValueError(
                "Cannot create client role: no worker roles exist to clone from"
            )

        # Clone the first worker role
        client_role = copy.deepcopy(appdef.roles[0])

        # Override with client-specific configuration
        client_role.name = "client"
        # Use the bootstrap script as entrypoint
        client_role.entrypoint = "workspace/torchforge/.meta/mast/client_bootstrap.sh"

        # Build args for the client role (passed to the bootstrap script)
        # These args will be passed to client_bootstrap.sh which forwards them to main.py
        args = [
            "--mode=remote",
            "--job-name",
            self.job_name,
        ]

        # Add any extra args passed from the CLI (includes --config and other args)
        if self.extra_args:
            args.extend(self.extra_args)

        client_role.args = args
        client_role.num_replicas = 1

        return client_role

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
        return MastLauncher(cfg, detached=False)
    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")

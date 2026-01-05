# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher specific logic (i.e. SLURM, k8s when supported, etc.)"""

import tempfile
from typing import Any

import monarch
from forge.controller.base import BaseLauncher
from forge.types import Launcher, LauncherConfig
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch.actor import ProcMesh
from monarch.tools import commands
from monarch.tools.components import hyperactor
from monarch.tools.config import Config


JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"


class Slurmlauncher(BaseLauncher):
    def __init__(
        self,
        cfg: LauncherConfig,
    ):
        self.cfg = cfg

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
            role.resource.memMB = self.cfg.memMB
            role.resource.cpu = self.cfg.cpu
            role.resource.gpu = self.cfg.gpu

        # Note - we cannot add in an empty workspace, so we create a fake temporary one
        temp_workspace = tempfile.mkdtemp(prefix="forge_workspace_")
        server_config = Config(
            scheduler="slurm",
            scheduler_args={
                "account": self.cfg.account,
                "qos": self.cfg.qos,
                "time": "72:00:00",
            },
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


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher(cfg)
    elif cfg.launcher == Launcher.MAST:
        try:
            from forge.fb.mast_launcher import MastLauncher

            return MastLauncher(cfg, detached=False)
        except ImportError as err:
            raise ValueError("MAST is not available, cannot launch MAST jobs.") from err

    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")

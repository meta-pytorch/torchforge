# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Spawning utils for actors and proc_meshes."""

import getpass
import json
import logging

import os
import socket
from functools import partial

from monarch.actor import proc_mesh, ProcMesh
from monarch.tools import commands
from monarch.tools.config import Config
from omegaconf import DictConfig

from forge.controller import ForgeActor

from forge.controller.system_controllers.gpu_manager import get_gpu_ids, release_gpus
from forge.types import ProcessConfig

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAST_SUPPORTED = False
try:
    from monarch._rust_bindings.monarch_hyperactor.alloc import (
        AllocConstraints,
        AllocSpec,
    )
    from monarch._src.actor.meta.allocator import MastAllocator, MastAllocatorConfig
    from monarch.tools.components.meta import hyperactor

    MAST_SUPPORTED = True
except ImportError:
    # Not built with Meta internal
    pass


async def spawn_actors(
    name: str,
    actor_cls: ForgeActor,
    cfg: DictConfig,
    processes: ProcessConfig,
    set_address: bool = False,
):
    """Setup process Mesh and spawn Actors."""
    mesh = await get_proc_mesh(processes)
    actors = await mesh.spawn(name, actor_cls, **cfg)
    actors.mesh = mesh
    return actors


async def get_proc_mesh(process_config: ProcessConfig) -> ProcMesh:
    """Returns a proc mesh with the given process config."""
    # TODO - modify this to work with multi-host
    env = {
        "MASTER_ADDR": str(socket.gethostname()),
        "MASTER_PORT": str(_find_free_port()),
    }
    gpu_ids = None

    def _setup_env(env: dict[str, str]):
        """Sets up the environment on proc mesh creation."""
        for k, v in env.items():
            os.environ[k] = v

    if process_config.scheduler == "local":
        if process_config.num_hosts != 1:
            raise ValueError("Local scheduler only supports 1 host")

        if process_config.with_gpus:
            gpu_ids = await get_gpu_ids(process_config.num_procs)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # TODO - update to use this_host() whenever it supports
        # being run within actors:
        # AttributeError: NYI: attempting to get ProcMesh attribute `slice` on object that's
        # actually a ProcMeshRef
        # return this_host().spawn_procs(
        #     per_host={"procs": process_config.num_procs},
        #     bootstrap=partial(_setup_env, env=env),
        # )
        m = proc_mesh(gpus=process_config.num_procs, env=env)
        m._gpu_ids = gpu_ids
        return m
    elif process_config.scheduler == "mast":
        if not MAST_SUPPORTED:
            raise ValueError("MAST is not supported on this platform")

        if process_config.with_gpus:
            raise ValueError("NYI - need to add HostMesh tracking in GpuManager")

        logging.info("Scheduling on MAST with: ", process_config)
        jobname = f"monarch-{getpass.getuser()}"
        config = Config(
            scheduler="mast_conda",
            scheduler_args={
                "hpcIdentity": process_config.identity,
                "hpcJobOncall": process_config.oncall,
                "hpcClusterUuid": "MastProdCluster",
                "rmAttribution": "pytorch4all_clients_approved",
            },
            appdef=hyperactor.host_mesh_conda(
                image=str(process_config.image),
                meshes=[f"mesh0:{process_config.num_hosts}:gtt_any"],
            ),
            workspace=str(os.getcwd()),
        )
        server_info = await commands.get_or_create(jobname, config)
        logger.info(
            "\n===== Server Info =====\n%s",
            json.dumps(server_info.to_json(), indent=2),
        )

        mesh_dimensions = {
            "host": server_info.get_mesh_spec("mesh0").num_hosts,
            "gpu": server_info.get_mesh_spec("mesh0").gpus,
        }
        # this is redundant but is here for example sake
        mesh_name = server_info.get_mesh_spec("mesh0").name

        allocator = MastAllocator(MastAllocatorConfig(job_name=server_info.name))
        constraints = AllocConstraints(
            {MastAllocator.ALLOC_LABEL_TASK_GROUP: mesh_name}
        )
        alloc = await allocator.allocate(AllocSpec(constraints, **mesh_dimensions))
        if env:
            p = await ProcMesh.from_alloc(alloc, setup=partial(_setup_env, env=env))
        else:
            p = await ProcMesh.from_alloc(alloc)
        await p.logging_option(stream_to_client=True, aggregate_window_sec=3)
        return p
    else:
        raise ValueError("Unsupported scheduler: {}".format(process_config.scheduler))


async def stop_proc_mesh(mesh: ProcMesh) -> None:
    """Stops the given proc mesh."""
    if hasattr(mesh, "_gpu_ids") and mesh._gpu_ids is not None:
        gpu_ids = mesh._gpu_ids
        logger.debug("Releasing GPUs: %s", gpu_ids)
        await release_gpus(gpu_ids)
    await mesh.stop()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        addr = s.getsockname()
        port = addr[1]
        return port

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ProcMesh utils across schedulers."""

import getpass
import json
import logging

import os

from monarch.actor import proc_mesh, ProcMesh
from monarch.tools import commands
from monarch.tools.config import Config
from omegaconf import DictConfig

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
    logger.warning(
        "MAST is not supported on this platform. You can ignore this if you do not work at Meta."
    )


async def get_proc_mesh(scheduler_config: DictConfig) -> ProcMesh:
    if scheduler_config.scheduler == "local":
        if scheduler_config.num_hosts != 1:
            raise ValueError("Local scheduler only supports 1 host")
        return await proc_mesh(gpus=scheduler_config.num_gpus)
    elif scheduler_config.scheduler == "mast":
        if not MAST_SUPPORTED:
            raise ValueError("MAST is not supported on this platform")

        logging.info("Scheduling on MAST with: ", scheduler_config)
        jobname = f"monarch-{getpass.getuser()}"
        config = Config(
            scheduler="mast_conda",
            scheduler_args={
                "hpcIdentity": scheduler_config.identity,
                "hpcJobOncall": scheduler_config.oncall,
                "hpcClusterUuid": "MastProdCluster",
                "rmAttribution": "pytorch4all_clients_approved",
            },
            appdef=hyperactor.host_mesh_conda(
                image=scheduler_config.image,
                # image="monarch_default_workspace:latest",
                meshes=[f"mesh0:{scheduler_config.num_hosts}:gtt_any"],
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
        return await ProcMesh.from_alloc(alloc)
    else:
        raise ValueError("Unsupported scheduler: {}".format(scheduler_config.scheduler))

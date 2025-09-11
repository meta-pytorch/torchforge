# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Factory-based service spawning for the Monarch rollout system."""

import logging
from typing import Type

from monarch.actor import proc_mesh

from forge.controller import ForgeActor
from forge.controller.orchestrator import Orchestrator, OrchestratorActor, ServiceConfig

from forge.controller.orchestrator.interface import (
    OrchestratorInterface,
    OrchestratorInterfaceV2,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def spawn_service(
    service_cfg: ServiceConfig, actor_def: Type[ForgeActor], **actor_kwargs
) -> OrchestratorInterface:
    """Spawns a orchestrator based on the actor class.

    Args:
        service_cfg: Service configuration
        actor_def: Actor class definition
        **actor_kwargs: Keyword arguments to pass to actor constructor

    Returns:
        A OrchestratorInterface that provides access to the Service Actor
    """
    # Assert that actor_def is a subclass of ForgeActor
    if not issubclass(actor_def, ForgeActor):
        raise TypeError(
            f"actor_def must be a subclass of ForgeActor, got {type(actor_def).__name__}"
        )

    # Create a single-node proc_mesh and actor_mesh for the Orchestrator
    logger.info("Spawning Service Actor for %s", actor_def.__name__)
    orchestrator = Orchestrator(service_cfg, actor_def, actor_kwargs)
    await orchestrator.__initialize__()
    # Return the OrchestratorInterface that wraps the proc_mesh, actor_mesh, and actor_def
    return OrchestratorInterface(orchestrator, actor_def)


async def shutdown_service(service: OrchestratorInterface) -> None:
    """Shuts down the service.

    Implemented in this way to avoid actors overriding stop() unintentionally.

    """
    await service._orchestrator.stop()


async def spawn_service_v2(
    service_cfg: ServiceConfig, actor_def: Type[ForgeActor], **actor_kwargs
) -> OrchestratorInterfaceV2:
    """Spawns a service based on the actor class.

    Args:
        service_cfg: Service configuration
        actor_def: Actor class definition
        **actor_kwargs: Keyword arguments to pass to actor constructor

    Returns:
        A OrchestratorInterface that provides access to the Orchestrator Actor
    """
    # Assert that actor_def is a subclass of ForgeActor
    if not issubclass(actor_def, ForgeActor):
        raise TypeError(
            f"actor_def must be a subclass of ForgeActor, got {type(actor_def).__name__}"
        )

    # Create a single-node proc_mesh and actor_mesh for the Orchestrator Actor
    logger.info("Spawning Orchestrator Actor for %s", actor_def.__name__)
    m = await proc_mesh(gpus=1)
    orchestrator_actor = await m.spawn(
        "service", OrchestratorActor, service_cfg, actor_def, actor_kwargs
    )
    await orchestrator_actor.__initialize__.call_one()

    # Return the OrchestratorInterface that wraps the proc_mesh, actor_mesh, and actor_def
    return OrchestratorInterfaceV2(m, orchestrator_actor, actor_def)


async def shutdown_service_v2(orchestrator: OrchestratorInterfaceV2) -> None:
    """Shuts down the orchestrator.

    Implemented in this way to avoid actors overriding stop() unintentionally.

    """
    await orchestrator._orchestrator_actor.stop.call_one()
    await orchestrator._proc_mesh.stop()

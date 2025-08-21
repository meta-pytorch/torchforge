# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Factory-based service spawning for the Monarch rollout system."""

import logging
from typing import Type

from monarch.actor import Actor

from forge.controller import Service, ServiceConfig
from forge.controller.service_v2 import (
    Service as ServiceV2,
    ServiceConfig as ServiceConfigV2,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def spawn_service(
    service_cfg: ServiceConfig, actor_def: Type[Actor], *actor_args, **actor_kwargs
) -> Service:
    """Spawns a service based on the actor class.

    Args:
        service_cfg: Service configuration
        actor_def: Actor class definition
        *actor_args: Arguments to pass to actor constructor
        **actor_kwargs: Keyword arguments to pass to actor constructor

    Returns:
        The appropriate service type based on the actor class
    """
    # Default to base Service
    logger.info("Spawning base Service for %s", actor_def.__name__)
    service = Service(service_cfg, actor_def, *actor_args, **actor_kwargs)
    await service.__initialize__()
    return service


async def spawn_service_v2(
    service_cfg: ServiceConfigV2, actor_def: Type[Actor], *actor_args, **actor_kwargs
) -> ServiceV2:
    """Spawns a service based on the actor class.

    Args:
        service_cfg: Service configuration
        actor_def: Actor class definition
        *actor_args: Arguments to pass to actor constructor
        **actor_kwargs: Keyword arguments to pass to actor constructor

    Returns:
        The appropriate service type based on the actor class
    """
    # Default to base Service
    logger.info("Spawning base Service for %s", actor_def.__name__)
    service = ServiceV2(service_cfg, actor_def, *actor_args, **actor_kwargs)
    await service.__initialize__()
    return service

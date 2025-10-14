# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from monarch.actor import context, current_rank

logger = logging.getLogger(__name__)


def get_actor_name_with_rank(actor_name: Optional[str] = None) -> str:
    """
    Extracts actor information from Monarch context to form a logging name.

    Args:
        actor_name: Actor name to use. Defaults to "UnknownActor" if None.

    Returns:
        str: Format "ActorName_replicaId_rLocalRank" (e.g., "TrainActor_abcd_r0").
             Falls back to just actor name if context unavailable.
    """
    if actor_name is None:
        actor_name = "UnknownActor"

    ctx = context()
    if ctx is None or ctx.actor_instance is None:
        logger.warning("Context unavailable, using fallback actor name for logging.")
        return actor_name

    actor_instance = ctx.actor_instance
    rank = current_rank()
    actor_id_full = str(actor_instance.actor_id)

    # Parse the actor_id
    parts = actor_id_full.split(".")
    if len(parts) < 2:
        return f"{actor_name}_r{rank.rank}"

    world_part = parts[0]  # e.g., "_1rjutFUXQrEJ[0]"

    # Use last 4 characters of world_id as replica identifier
    world_id = world_part.split("[")[0] if "[" in world_part else world_part
    replica_id = world_id[-4:] if len(world_id) >= 4 else world_id

    return f"{actor_name}_{replica_id}_r{rank.rank}"

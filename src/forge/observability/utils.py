# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from monarch.actor import context, current_rank

logger = logging.getLogger(__name__)


def detect_actor_name_from_call_stack() -> str:
    """Detect ForgeActor subclass name from call stack.

    Returns:
        str: Actor name, defaulting to "UnknownActor" if not found.
    """
    try:
        import inspect

        frame = inspect.currentframe()
        frame_count = 0

        while frame:
            frame = frame.f_back
            if not frame:
                break

            frame_count += 1
            if frame_count > 20:  # Prevent infinite loops
                break

            # Check for 'self' (instance method calls)
            if "self" in frame.f_locals:
                obj = frame.f_locals["self"]
                if hasattr(obj, "__class__") and hasattr(obj.__class__, "__mro__"):
                    for base in obj.__class__.__mro__:
                        if base.__name__ == "ForgeActor":
                            return obj.__class__.__name__

            # Check for 'cls' (class method calls)
            if "cls" in frame.f_locals:
                cls = frame.f_locals["cls"]
                if hasattr(cls, "__mro__"):
                    for base in cls.__mro__:
                        if base.__name__ == "ForgeActor":
                            return cls.__name__

    except Exception as e:
        logger.debug(f"Call stack detection failed: {e}")

    return "UnknownActor"


def get_actor_name_with_rank(actor_name: Optional[str] = None) -> str:
    """
    Extracts actor information from Monarch context to form a logging name.

    Args:
        actor_name: Optional actor name to use. If None, will auto-detect from call stack.

    Returns:
        str: Format "ActorName_replicaId_rLocalRank" (e.g., "TrainActor_abcd_r0").
             Falls back to "UnknownActor" if context unavailable.
    """
    ctx = context()
    if ctx is None or ctx.actor_instance is None:
        logger.warning("Context unavailable, using fallback actor name for logging.")
        return "UnknownActor"

    actor_instance = ctx.actor_instance
    rank = current_rank()
    actor_id_full = str(actor_instance.actor_id)

    # Parse the actor_id
    parts = actor_id_full.split(".")
    if len(parts) < 2:
        return "UnknownActor"

    world_part = parts[0]  # e.g., "_1rjutFUXQrEJ[0]"
    actor_part = parts[1]  # e.g., "TestActorConfigured[0]"

    # Use provided actor name or auto-detect from call stack
    if actor_name:
        final_actor_name = actor_name
    else:
        final_actor_name = detect_actor_name_from_call_stack()

    # Use last 4 characters of world_id as replica identifier
    world_id = world_part.split("[")[0] if "[" in world_part else world_part
    replica_id = world_id[-4:] if len(world_id) >= 4 else world_id

    return f"{final_actor_name}_{replica_id}_r{rank.rank}"

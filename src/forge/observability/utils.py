# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from monarch.actor import context, current_rank

logger = logging.getLogger(__name__)


def get_proc_name_with_rank(proc_name: Optional[str] = None) -> str:
    """
    Returns a unique identifier for the current rank from Monarch actor context.

    Multiple ranks from the same ProcMesh will share the same ProcMesh hash suffix,
    but have different rank numbers.

    Format: "{ProcessName}_{ProcMeshHash}_r{rank}" where:
    - ProcessName: The provided proc_name (e.g., "TrainActor") or extracted from actor_name if None.
    - ProcMeshHash: Hash suffix identifying the ProcMesh (e.g., "1abc2def")
    - rank: Local rank within the ProcMesh (0, 1, 2, ...)

    Note: If called from the main process (e.g. main.py), returns "client_r0".

    Args:
        proc_name: Optional override for process name. If None, uses actor_id.actor_name.

    Returns:
        str: Unique identifier per rank (e.g., "TrainActor_1abc2def_r0" or "client_r0").
    """
    ctx = context()
    actor_id = ctx.actor_instance.actor_id
    actor_name = actor_id.actor_name
    rank = current_rank().rank

    # If proc_name provided, extract procmesh hash from actor_name and combine
    if proc_name is not None:
        parts = actor_name.split("_")
        if len(parts) > 1:
            replica_hash = parts[-1]  # (e.g., "MyActor_1abc2def" -> "1abc2def")
            return f"{proc_name}_{replica_hash}_r{rank}"
        else:
            # if a direct process (e.g. called from main), actor_name == "client" -> len(parts) == 1
            return f"{proc_name}_r{rank}"

    # No proc_name override - use full actor_name with rank
    return f"{actor_name}_r{rank}"

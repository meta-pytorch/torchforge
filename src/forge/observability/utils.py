# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from monarch.actor import context

logger = logging.getLogger(__name__)


def get_proc_name_with_rank(proc_name: Optional[str] = None) -> str:
    """
    Returns a unique process identifier from Monarch actor context.

    Format: "ActorName_wxyz_r{rank}" where:
    - ActorName: The actor class name (e.g., "TrainActor")
    - wxyz: Last 4 chars of world_name (unique replica hash)
    - rank: Local rank within the replica (0, 1, 2, ...)

    Note: If called from a direct proccess, defaults to "client_DPROC_r0".

    Args:
        proc_name: Optional override for actor name. If None, uses actor_id.actor_name.

    Returns:
        str: Unique identifier or fallback name if no context available.
    """
    ctx = context()
    actor_id = ctx.actor_instance.actor_id

    # Use actor_name from actor_id if not provided
    if proc_name is None:
        proc_name = actor_id.actor_name

    # Try to get world_name. Each replica has a unique value.
    try:
        world_name = actor_id.world_name
        replica_id = world_name[-4:] if len(world_name) >= 4 else world_name
    except BaseException:  # Catches pyo3_runtime.PanicException from Rust
        # Direct proc (e.g., client) - no world_name available
        replica_id = "DPROC"

    # Get rank within the replica. NOT a global rank.
    try:
        rank = actor_id.rank
    except BaseException:  # Catches pyo3_runtime.PanicException from Rust
        # Direct proc - no rank available
        rank = 0

    return f"{proc_name}_{replica_id}_r{rank}"

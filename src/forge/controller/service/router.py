# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List

from .interface import Router
from .replica import Replica

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RoundRobinRouter(Router):
    """Round-robin router for stateless requests."""

    def __init__(self):
        self._next_idx = 0

    def get_replica(
        self,
        healthy_replicas: List[Replica],
        sess_id: str | None = None,
        session_map: Dict[str, int] | None = None,
    ) -> Replica:
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available for load balancing")

        self._next_idx = (self._next_idx + 1) % len(healthy_replicas)
        replica = healthy_replicas[self._next_idx]

        return replica


class LeastLoadedRouter(Router):
    """Always routes to the replica with the lowest current load."""

    def get_replica(
        self,
        healthy_replicas: List[Replica],
        sess_id: str | None = None,
        session_map: Dict[str, int] | None = None,
    ) -> Replica:
        if not healthy_replicas:
            raise RuntimeError("No healthy replicas available for session assignment")
        return min(healthy_replicas, key=lambda r: r.current_load)


class SessionRouter(Router):
    """Session-based routing: sticky sessions with a fallback router."""

    def __init__(self, fallback_router: Router):
        self.fallback_router = fallback_router

    def get_replica(
        self,
        healthy_replicas: List[Replica],
        sess_id: str | None = None,
        session_map: Dict[str, int] | None = None,
    ) -> Replica:
        if sess_id is None:
            raise ValueError("SessionRouter requires a session ID")

        if session_map is None:
            raise ValueError("Session map must be provided for SessionRouter")

        # Check if session already has a replica
        if sess_id in session_map:
            replica_idx = session_map[sess_id]
            # Find the replica with this index
            for r in healthy_replicas:
                if r.idx == replica_idx:
                    return r
            # If the replica is no longer healthy, remove from session map and reassign
            del session_map[sess_id]

        # Use fallback router to assign a new replica
        replica = self.fallback_router.get_replica(
            healthy_replicas, sess_id, session_map
        )
        session_map[sess_id] = replica.idx
        logger.debug(
            "Assigning session %s to replica %d",
            sess_id,
            replica.idx,
        )
        return replica

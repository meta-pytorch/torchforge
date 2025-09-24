# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, List

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


class BatchRouter(Router):
    """
    Router wrapper that batches routing decisions.
    Uses an inner router to pick the replica for each batch.

    Args:
        inner_router: The underlying Router instance used to make routing decisions
        batch_max_size: Maximum number of requests to collect in a single batch (default: 8)
        batch_max_wait_s: Maximum time to wait before processing a batch in seconds (default: 0.01)

    Example:
        rr_router = RoundRobinRouter()
        batch_router = BatchRouter(rr_router, batch_max_size=16, batch_max_wait_s=0.02)

        replica = await batch_router.get_replica(healthy_replicas, sess_id, session_map)
    """

    def __init__(
        self,
        inner_router: Router,
        batch_max_size: int = 8,
        batch_max_wait_s: float = 0.01,
        get_healthy_replicas: Optional[Callable[[], List["Replica"]]] = None,
        session_map: Optional[Dict[str, int]] = None,
    ):

        self.inner_router = inner_router
        self.batch_max_size = batch_max_size
        self.batch_max_wait_s = batch_max_wait_s
        self.get_healthy_replicas = get_healthy_replicas
        self.session_map = session_map

        # Internal queue for batching routing requests
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = True  # flag to control loop
        # Background task that processes batches continuously
        self._batch_task: asyncio.Task = asyncio.create_task(self._batch_loop())

    async def _batch_loop(self):
        """Background task that continuously processes batches of routing requests.

        This is the core batching logic that runs in a separate asyncio task.
        It collects requests from the queue and processes them in batches based
        on size and time constraints.

        The loop follows these steps:
        1. Wait for the first request to start a new batch
        2. Collect additional requests until batch_max_size or batch_max_wait_s is reached
        3. Make a single routing decision for the entire batch
        4. Fulfill all futures with the selected replica

        This process repeats indefinitely until the task is cancelled.
        """
        while self._running:
            batch = []
            futs = []
            sess_ids = []

            # Wait for first request
            fut, healthy_replicas, sess_id, session_map = await self._queue.get()
            batch.append((healthy_replicas, sess_id, session_map))
            futs.append(fut)
            sess_ids.append(sess_id)
            start_time = time.monotonic()

            while True:
                try:
                    timeout = max(
                        0, self.batch_max_wait_s - (time.monotonic() - start_time)
                    )
                    (
                        fut,
                        healthy_replicas,
                        sess_id,
                        session_map,
                    ) = await asyncio.wait_for(
                        self._queue.get(), timeout
                    )  # wait for timeout or until self._queue.get() finishes
                    batch.append((healthy_replicas, sess_id, session_map))
                    futs.append(fut)
                    sess_ids.append(sess_id)

                    if len(batch) >= self.batch_max_size:
                        break
                except asyncio.TimeoutError:
                    break

            if self.session_map is not None:
                session_map = self.session_map
            else:
                session_map = batch[-1][2]  # use most recent session map
            if self.get_healthy_replicas is not None:
                healthy_replicas = self.get_healthy_replicas()
            else:
                healthy_replicas = batch[-1][0]  # use most recent replica state
                # Check if any replicas have become unhealthy
                healthy_replicas = [r for r in healthy_replicas if r.healthy]

            # One routing decision for the whole batch
            replica = await self.inner_router.get_replica(
                healthy_replicas, None, session_map
            )

            # Fulfill all futures with the chosen replica
            for fut in futs:
                fut.set_result(replica)

    async def get_replica(
        self,
        healthy_replicas: List[Replica],
        sess_id: Optional[str] = None,
        session_map: Optional[Dict[str, int]] = None,
    ) -> Replica:
        """Enqueue request and wait until batch assigns a replica."""
        fut = asyncio.Future()
        # Queue the request for batching - this is non-blocking
        self._queue.put_nowait((fut, healthy_replicas, sess_id, session_map))

        # Wait for the batch processor to resolve our future
        return await fut

    async def shutdown(self):
        """Stop the batch loop gracefully."""
        self._running = False
        self._batch_task.cancel()
        try:
            await self._batch_task
        except asyncio.CancelledError:
            pass

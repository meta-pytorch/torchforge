# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from forge.controller.service.replica import Replica, ServiceRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Router(ABC):
    """Abstract base class for routing logic."""

    @abstractmethod
    def get_replica(
        self,
        healthy_replicas: List[Replica],
        sess_id: str | None = None,
        session_map: Dict[str, int] | None = None,
    ) -> Replica:
        """Select a replica from the list based on routing logic."""
        pass


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


class Batcher:
    """
    Asynchronous batching wrapper around a Router.

    Instead of selecting a replica immediately, incoming requests are enqueued
    and grouped into batches. Once a batch is ready (either reaching the maximum
    size or exceeding the maximum wait time), the batcher makes a single routing
    decision using the inner router. All requests in that batch are then resolved
    with the same replica.

    This reduces router overhead by amortizing multiple requests into one decision.

    Args:
        inner_router: The underlying Router used to pick a replica.
        get_healthy_replicas: Callable that returns the current list of healthy replicas.
        get_session_map: Callable that returns the session-to-replica mapping.
        batch_size: Maximum number of requests to collect in a single batch
                        before routing (default: 8).
        batch_timeout: Maximum time to wait (in seconds) before routing a batch,
                          even if batch_size is not reached (default: 0.01).

    Example:
        rr_router = RoundRobinRouter()
        batcher = Batcher(
            rr_router,
            get_healthy_replicas=service._get_healthy_replicas,
            get_session_map=service._get_session_map,
            batch_size=16,
            batch_timeout=0.01,
        )

        request = ServiceRequest(...)

        # Enqueue a request to be sent to a replica
        await batcher.enqueue(request)
    """

    def __init__(
        self,
        function: str,
        inner_router: Router,
        get_healthy_replicas: Callable[[], List["Replica"]],
        get_session_map: Callable[[], Dict[str, int]],
        batch_size: int = 16,
        batch_timeout: float = 0.01,
    ):
        self.function = function
        self.inner_router = inner_router
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.get_healthy_replicas = get_healthy_replicas
        self.get_session_map = get_session_map

        # Internal queue for batching routing requests
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = True  # flag to control loop
        # Background task that processes batches continuously
        self._batch_task: asyncio.Task = asyncio.create_task(self._batch_loop())
        # Maximum number of routing attempts per batch (1 initial + 1 retry if replica fails)
        self._num_attempts = 2

    async def _batch_loop(self) -> None:
        """Background task that continuously processes batches of routing requests.

        This is the core batching logic that runs in a separate asyncio task.
        Each iteration collects individual (function, args, kwargs, future) entries from
        the internal queue and merges them into a single `ServiceRequest` that is then
        dispatched to one replica.

        The loop follows these steps:
        1. Wait for the first queued call to start a new batch
        2. Continue collecting until batch_size or batch_timeout is reached
        3. Make a single routing decision for the entire batch
        4. Fulfill all futures with the selected replica

        This process repeats indefinitely until the task is cancelled.

        Returns:
            None

        Raises:
            RuntimeError: If no healthy replicas are available
            Exception: Any exception raised by the actor endpoint
        """
        while self._running:

            # Wait for first request
            batch = [await self._queue.get()]
            start_time = time.monotonic()

            # TODO (dxie): consider making timeout adaptive based on replica load.
            while True:
                try:
                    timeout = max(
                        0, self.batch_timeout - (time.monotonic() - start_time)
                    )
                    nxt = await asyncio.wait_for(
                        self._queue.get(), timeout
                    )  # wait for timeout or until self._queue.get() finishes
                    batch.append(nxt)

                    if len(batch) >= self.batch_size:
                        break
                except asyncio.TimeoutError:
                    break

            session_map = self.get_session_map()
            healthy_replicas = self.get_healthy_replicas()

            # One routing decision for the whole batch
            replica = self.inner_router.get_replica(healthy_replicas, None, session_map)

            # Merge args for batched call
            inputs = [b[1][0] for b in batch]

            # One request for the entire batch
            batch_req = ServiceRequest(
                session_id=None,
                function=self.function,
                args=(inputs,),
                kwargs={},
                future=asyncio.Future(),
            )

            for attempt in range(self._num_attempts):
                try:
                    # Send whole batch to replica
                    await replica.enqueue_request(batch_req)
                    results = await batch_req.future
                    # Normalize result shape.
                    # The actor endpoint is expected to return one result per input
                    # If it instead returns a single scalar, replicate that scalar across
                    # all callers so that every waiting future gets a value.
                    if not isinstance(results, list) or len(results) != len(batch):
                        results = [results] * len(batch)

                    for (_, _, _, f), r in zip(batch, results):
                        f.set_result(r)
                    break

                except Exception as e:
                    if attempt < self._num_attempts - 1 and not replica.healthy:
                        logger.debug(
                            f"Replica {replica.idx} failed during request, retrying on healthy replica. Exception: {e}"
                        )
                        healthy_replicas = self.get_healthy_replicas()
                        session_map = self.get_session_map()
                        if not healthy_replicas:
                            raise RuntimeError("No healthy replicas available") from e
                        replica = self.inner_router.get_replica(
                            healthy_replicas, None, session_map
                        )
                        continue
                    else:
                        for _, _, _, f in batch:
                            if not f.done():
                                f.set_exception(e)

    async def route(self, function: str, args: tuple, kwargs: dict) -> Any:
        """Add (args, kwargs) pair to queue, return a Future resolved when batch completes."""
        # Queue the request for batching
        fut = asyncio.Future()
        self._queue.put_nowait((function, args, kwargs, fut))
        # Wait for the batch processor to resolve our future
        return await fut

    async def stop(self):
        """Stop the batch loop gracefully."""
        self._running = False
        self._batch_task.cancel()
        try:
            await self._batch_task
        except asyncio.CancelledError:
            pass

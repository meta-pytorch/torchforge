# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Replica for distributed actor service."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from monarch.actor import Actor, ActorError

from forge.controller import RecoverableProcMesh

logger = logging.getLogger(__name__)


class ReplicaState(Enum):
    HEALTHY = "healthy"
    RECOVERING = "recovering"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    UNINITIALIZED = "uninitialized"


@dataclass
class ServiceRequest:
    session_id: Optional[str]
    function: str
    args: tuple
    kwargs: dict
    future: asyncio.Future


@dataclass
class Replica:
    proc_mesh: RecoverableProcMesh
    actor: Optional[Actor]
    idx: int
    request_queue: asyncio.Queue[ServiceRequest] = field(default_factory=asyncio.Queue)
    active_requests: int = 0
    max_concurrent_requests: int = 10
    _running: bool = False
    metadata: dict = field(default_factory=dict)
    state: ReplicaState = ReplicaState.UNINITIALIZED
    return_first_rank_result: bool = False

    async def enqueue_request(self, request: ServiceRequest):
        """Enqueues a request for processing by this replica."""
        if self.state == ReplicaState.STOPPED:
            raise RuntimeError(f"Replica {self.idx} is stopped")

        # Accept requests in all other states - let the processing loop handle the rest
        await self.request_queue.put(request)

    async def _process_single_request(self, request: ServiceRequest) -> bool:
        """
        Processes a single request and returns success status.

        Returns:
            bool: True if request succeeded, False if it failed
        """
        self.active_requests += 1

        try:
            # Get the actor and endpoint
            actor = self.actor
            endpoint_func = getattr(actor, request.function)

            # Execute the request
            success = True
            try:
                result = await endpoint_func.call(*request.args, **request.kwargs)
                # Unwrap ValueMesh if configured to return first rank result
                if (
                    self.return_first_rank_result
                    and hasattr(result, "_values")
                    and result._values
                ):
                    result = result._values[0]
                request.future.set_result(result)
            except ActorError as e:
                logger.debug("Got failure on replica %d. Error:\n%s", self.idx, e)
                # Mark proc_mesh as failed and transition state
                self.proc_mesh.mark_failed()
                self.state = ReplicaState.RECOVERING
                # Unwrap the ActorError into its raw exception
                request.future.set_exception(e.exception)
                success = False
            except Exception as e:
                logger.debug(
                    "Got unexpected error on replica %d. Error:\n%s", self.idx, e
                )
                # Mark proc_mesh as failed and transition state
                self.proc_mesh.mark_failed()
                self.state = ReplicaState.RECOVERING
                request.future.set_exception(e)
                success = False

            # Mark task as done
            self.request_queue.task_done()
            return success

        finally:
            self.active_requests -= 1

    async def run(self):
        """
        Main processing loop for the replica. This replaces _persistent_processor.

        Continuously processes requests from the queue while the replica is healthy.
        Handles capacity management and graceful degradation on failures.
        """
        self._running = True

        try:
            while self.state in (ReplicaState.HEALTHY, ReplicaState.RECOVERING):
                try:
                    # Wait for a request with timeout to check health periodically
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )

                    # Check if we have capacity
                    if self.active_requests >= self.max_concurrent_requests:
                        # Put the request back and wait
                        await self.request_queue.put(request)
                        await asyncio.sleep(0.1)
                        continue

                    # Update state if proc_mesh recovered
                    if self.state == ReplicaState.RECOVERING and self.proc_mesh.healthy:
                        self.state = ReplicaState.HEALTHY
                        logger.debug("Replica %d recovered to healthy state", self.idx)

                    # If we're still recovering and proc_mesh isn't healthy, reject request
                    if (
                        self.state == ReplicaState.RECOVERING
                        and not self.proc_mesh.healthy
                    ):
                        request.future.set_exception(
                            RuntimeError(f"Replica {self.idx} is still recovering")
                        )
                        self.request_queue.task_done()
                        continue

                    # Process the request
                    asyncio.create_task(self._process_single_request(request))

                except asyncio.TimeoutError:
                    # No requests, check for health state changes
                    if self.state == ReplicaState.RECOVERING and self.proc_mesh.healthy:
                        self.state = ReplicaState.HEALTHY
                        logger.debug("Replica %d recovered to healthy state", self.idx)
                    elif (
                        self.state == ReplicaState.HEALTHY
                        and not self.proc_mesh.healthy
                    ):
                        self.state = ReplicaState.RECOVERING
                        logger.debug("Replica %d entered recovering state", self.idx)
                    continue

                except Exception as e:
                    logger.error(
                        "Error in replica %d processing loop: %s",
                        self.idx,
                        e,
                    )
                    self.state = ReplicaState.UNHEALTHY
                    break

        finally:
            self._running = False
            logger.debug("Replica %d stopped processing", self.idx)

    @property
    def healthy(self) -> bool:
        return self.state == ReplicaState.HEALTHY

    @property
    def load(self) -> int:
        """Get current load (active requests + queue depth)"""
        return self.active_requests + self.request_queue.qsize()

    @property
    def capacity_utilization(self) -> float:
        """Get current capacity utilization (0.0 to 1.0)"""
        if self.max_concurrent_requests <= 0:
            return 0.0
        return self.active_requests / self.max_concurrent_requests

    def can_accept_request(self) -> bool:
        """Check if replica can accept a new request"""
        return (
            self.state == ReplicaState.HEALTHY
            and self.active_requests < self.max_concurrent_requests
        )

    def __repr__(self) -> str:
        return (
            f"Replica(idx={self.idx}, state={self.state.value}, "
            f"active={self.active_requests}/{self.max_concurrent_requests}, "
            f"queue={self.request_queue.qsize()})"
        )

    async def setup(self):
        """
        Sets up the replica and transitions to healthy state.

        This should be called after the proc_mesh has been initialized
        and the actor has been spawned on it.
        """
        if self.state != ReplicaState.UNINITIALIZED:
            logger.warning(
                "Attempting to setup replica %d that's already initialized", self.idx
            )
            return

        if self.actor is None:
            raise RuntimeError(f"Cannot setup replica {self.idx}: actor is None")

        try:
            # Call actor setup if it exists
            if hasattr(self.actor, "setup"):
                await self.actor.setup.call()

            # Transition to healthy state
            self.state = ReplicaState.HEALTHY
            logger.debug("Replica %d setup complete", self.idx)

        except Exception as e:
            logger.error("Failed to setup replica %d: %s", self.idx, e)
            self.state = ReplicaState.UNHEALTHY
            raise

    async def stop(self):
        """
        Stops the replica gracefully.

        Transitions to STOPPED state, stops the processing loop, and cleans up.
        Fails any remaining requests in the queue.
        """
        logger.debug("Stopping replica %d", self.idx)

        # Transition to stopped state to signal the run loop to exit
        self.state = ReplicaState.STOPPED

        # Wait for processor to finish if it's running
        if self._running:
            # Give it a moment to finish current request and exit gracefully
            for _ in range(50):  # Wait up to 5 seconds
                if not self._running:
                    break
                await asyncio.sleep(0.1)

            if self._running:
                logger.warning("Replica %d processor didn't stop gracefully", self.idx)

        # Fail any remaining requests in the queue
        failed_requests = []
        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                failed_requests.append(request)
                self.request_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Fail all the collected requests
        for request in failed_requests:
            if not request.future.done():
                request.future.set_exception(
                    RuntimeError(f"Replica {self.idx} is stopping")
                )

        logger.debug(
            "Replica %d stopped, failed %d remaining requests",
            self.idx,
            len(failed_requests),
        )

        # Stop the proc_mesh
        try:
            await self.proc_mesh.stop()
        except Exception as e:
            logger.warning("Error stopping proc_mesh for replica %d: %s", self.idx, e)

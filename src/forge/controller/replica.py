# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Replica for distributed actor service."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from monarch.actor import Actor, ActorError, ProcMesh

from forge.controller import get_proc_mesh
from forge.types import ProcessConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ReplicaState(Enum):
    HEALTHY = "HEALTHY"
    RECOVERING = "RECOVERING"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"
    UNINITIALIZED = "UNINITIALIZED"


@dataclass
class ReplicaMetrics:
    """Simple metrics tracking for a replica."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    request_times: deque = field(default_factory=lambda: deque(maxlen=100))
    request_latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_request_start(self, timestamp: float):
        """Records when a request starts processing."""
        self.request_times.append(timestamp)
        self.total_requests += 1

    def add_request_completion(self, start_time: float, success: bool):
        """Records when a request completes."""
        latency = time.time() - start_time
        self.request_latencies.append(latency)
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def get_request_rate(self, window_seconds: float = 60.0) -> float:
        """Gets requests per second over the last window_seconds."""
        assert window_seconds > 0, "Window must be positive"
        now = time.time()
        cutoff = now - window_seconds
        recent_requests = [t for t in self.request_times if t >= cutoff]
        return len(recent_requests) / window_seconds

    def get_avg_latency(self, window_requests: int = 50) -> float:
        """Gets average latency over the last N requests."""
        if not self.request_latencies:
            return 0.0
        recent_latencies = list(self.request_latencies)[-window_requests:]
        return sum(recent_latencies) / len(recent_latencies)


@dataclass
class ServiceRequest:
    """Representation of a request to the service.

    A service request will typically be a call to an actor endpoint.
    - The endpoint call is represented by function str/args/kwargs,
    - The session_id is used for stateful routing, and
    - The future is used to return the result of the call.

    """

    session_id: Optional[str]
    function: str
    args: tuple
    kwargs: dict
    future: asyncio.Future


@dataclass
class Replica:
    """
    A distributed replica that serves as the fundamental unit of work within a service.

    Handles process lifecycle, async request queuing and fault recovery.
    Each replica runs independently and can be deployed across multiple hosts via Monarch

    """

    idx: int

    # Configuration for the underlying ProcMesh (scheduler, hosts, GPUs)
    proc_config: ProcessConfig
    actor_def: type[Actor]
    actor_args: tuple
    actor_kwargs: dict

    # The proc_mesh and actor_mesh that this replica is running
    proc_mesh: Optional[ProcMesh] = None
    actor: Optional[Actor] = None

    # Async queue for incoming requests
    request_queue: asyncio.Queue[ServiceRequest] = field(default_factory=asyncio.Queue)
    # Number of currently processing requests
    active_requests: int = 0
    # Maximum number of simultaneous requests
    max_concurrent_requests: int = 10
    # Semaphore to control request capacity
    _capacity_semaphore: asyncio.Semaphore = field(init=False)
    # Whether the processing loop is currently running
    _running: bool = False
    # How often to check for new requests when idle
    _run_poll_rate_s: float = 1.0
    # Current replica health state
    state: ReplicaState = ReplicaState.UNINITIALIZED
    # Whether to auto-unwrap ValueMesh to first rank
    return_first_rank_result: bool = False

    # Recovery-related state
    _recovery_task: Optional[asyncio.Task] = None

    # Run task is the replica's event loop
    _run_task: Optional[asyncio.Task] = None

    # Metrics tracking
    metrics: ReplicaMetrics = field(default_factory=ReplicaMetrics)

    def __post_init__(self):
        # This semaphore is used to enforce max_concurrent_requests
        # Once it is acquired max_concurrent_requests times, future
        # requests are blocked until standing requests complete.
        self._capacity_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    # Initialization related functionalities

    async def initialize(self):
        """
        Initializes the replica completely from proc_mesh creation to ready state.

        This method handles the complete replica initialization process:
        - Creates the proc_mesh
        - Spawns the actor
        - Configures the actor
        - Transitions to healthy state
        - Starts the processing loop
        """
        assert self.proc_mesh is None, "Proc mesh should not be set yet"
        try:
            # Create proc_mesh
            await self.create_proc_mesh()

            # Ensure we have a healthy proc_mesh
            if not self.proc_mesh:
                raise RuntimeError(
                    f"Replica {self.idx}: proc_mesh is None after creation"
                )

            # Spawn the actor
            await self.spawn_actor(
                actor_def=self.actor_def,
                *self.actor_args,
                **self.actor_kwargs,
            )
            # Transition to healthy state and start processing
            self.state = ReplicaState.HEALTHY
            self.start_processing()

            logger.debug("Replica %d initialization complete", self.idx)

        except Exception as e:
            logger.error("Failed to initialize replica %d: %s", self.idx, e)
            self.state = ReplicaState.UNHEALTHY
            raise

    async def recover(self):
        """Recovers the replica by recreating the proc_mesh and respawning actors."""
        if self._recovery_task and not self._recovery_task.done():
            # Recovery already in progress, wait for it
            await self._recovery_task
            return

        async def _do_recovery():
            old_proc_mesh = self.proc_mesh
            self.proc_mesh = None
            self.actor = None

            # Stop old proc_mesh if it exists
            if old_proc_mesh is not None:
                try:
                    await old_proc_mesh.stop()
                    logger.debug("Old proc_mesh stopped for replica %d", self.idx)
                except Exception as e:
                    logger.warning(
                        "Error stopping old proc_mesh for replica %d: %s", self.idx, e
                    )

            try:
                logger.debug("Creating new proc_mesh for replica %d", self.idx)
                await self.initialize()
                self.state = ReplicaState.HEALTHY
                logger.debug("Recovery completed successfully for replica %d", self.idx)
            except Exception as e:
                logger.error("Recovery failed for replica %d: %s", self.idx, e)
                self.state = ReplicaState.UNHEALTHY
                raise

        logger.debug("Starting recovery for replica %d", self.idx)
        self.state = ReplicaState.RECOVERING
        self._recovery_task = asyncio.create_task(_do_recovery())
        await self._recovery_task

    async def create_proc_mesh(self):
        """Creates the proc_mesh using the stored proc_config."""
        # TODO - for policy replica, we would override this method to
        # include multiple proc_meshes
        if self.proc_mesh is not None:
            logger.warning("Proc mesh already initialized for replica %d", self.idx)
            return

        logger.debug("Creating proc_mesh for replica %d", self.idx)
        try:
            self.proc_mesh = await get_proc_mesh(process_config=self.proc_config)
            logger.debug("Proc mesh created successfully for replica %d", self.idx)
        except Exception as e:
            logger.error("Failed to create proc_mesh for replica %d: %s", self.idx, e)
            self.state = ReplicaState.UNHEALTHY
            raise

    async def spawn_actor(self, actor_def, *actor_args, **actor_kwargs):
        """
        Spawn an actor on this replica's proc_mesh.

        This method handles the complete actor spawning process including
        recovery if the proc_mesh has failed.
        """
        if not self.proc_mesh:
            raise RuntimeError(
                f"Replica {self.idx}: proc_mesh is None after recovery attempt"
            )

        try:
            # TODO - expand support so name can stick within kwargs
            actor_name = actor_kwargs.pop("name", actor_def.__name__)

            # Spawn the actor
            self.actor = await self.proc_mesh.spawn(
                actor_name,
                actor_def,
                *actor_args,
                **actor_kwargs,
            )
            # Call actor setup if it exists
            if setup_method := getattr(self.actor, "setup", None):
                await setup_method.call()

            logger.debug("Actor spawned successfully on replica %d", self.idx)

        except Exception as e:
            logger.error("Failed to spawn actor on replica %d: %s", self.idx, e)
            self.mark_failed()
            raise

    # Request handling / processing related functionality

    def start_processing(self):
        """Start the replica's processing loop if not already running."""
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.run())
            logger.debug("Started processing loop for replica %d", self.idx)

    async def enqueue_request(self, request: ServiceRequest):
        """Enqueues a request for processing by this replica."""
        if self.stopped:
            raise RuntimeError(
                f"Replica {self.idx} is stopped and therefore will not accept requests."
            )

        # Accept requests in all other states - let the processing loop handle the rest
        await self.request_queue.put(request)

    async def _process_single_request(self, request: ServiceRequest) -> bool:
        """Processes a single request and returns success status.

        Returns:
            bool: True if request succeeded, False if it failed
        """
        start_time = time.time()
        self.active_requests += 1

        # Record request start for metrics
        self.metrics.add_request_start(start_time)

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
                logger.warning("Got failure on replica %d. Error:\n%s", self.idx, e)
                # The exception came from the actor. It itself is
                # returned to be propagated through the services
                # back to the caller.
                request.future.set_result(e.exception)

                # TODO: we may want to conditionally mark the
                # replica as failed here - i.e. where the actor itself
                # can be healthy but the request failed.
                self.mark_failed()
                success = False
            except Exception as e:
                logger.debug(
                    "Got unexpected error on replica %d. Error:\n%s", self.idx, e
                )
                self.mark_failed()

                # The exception was not from the actor - in this case
                # we will signal back to the service (through set_exception)
                # to retry on another healthy node.
                request.future.set_exception(e)
                success = False

            self.metrics.add_request_completion(start_time, success)
            # Mark task as done
            self.request_queue.task_done()
            return success

        finally:
            self.active_requests -= 1
            # Release the capacity semaphore to allow new requests
            self._capacity_semaphore.release()

    async def run(self):
        """Runs the main processing loop for the replica.

        Continuously processes requests from the queue while the replica is healthy.
        Handles capacity management and graceful degradation on failures.
        """
        self._running = True

        try:
            while self.healthy:
                try:
                    # Wait for a request with timeout to check health periodically
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=self._run_poll_rate_s
                    )

                    # Acquire capacity semaphore - this blocks until capacity is available
                    await self._capacity_semaphore.acquire()

                    # Process the request (semaphore will be released in _process_single_request)
                    asyncio.create_task(self._process_single_request(request))

                except asyncio.TimeoutError:
                    # No requests, just continue checking for new ones
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

    # Replica state management

    @property
    def healthy(self) -> bool:
        return self.state == ReplicaState.HEALTHY

    @property
    def uninitialized(self) -> bool:
        return self.state == ReplicaState.UNINITIALIZED

    @property
    def recovering(self) -> bool:
        return self.state == ReplicaState.RECOVERING

    @property
    def unhealthy(self) -> bool:
        return self.state == ReplicaState.UNHEALTHY

    @property
    def stopped(self) -> bool:
        return self.state == ReplicaState.STOPPED

    @property
    def failed(self) -> bool:
        """Check if the replica has failed and needs recovery."""
        return self.state in (ReplicaState.RECOVERING, ReplicaState.UNHEALTHY)

    def mark_failed(self):
        """Mark the replica as failed, triggering recovery."""
        logger.debug("Marking replica %d as failed", self.idx)
        self.state = ReplicaState.RECOVERING

    async def stop(self):
        """
        Stops the replica gracefully.

        Transitions to STOPPED state, stops the processing loop, and cleans up.
        Fails any remaining requests in the queue.
        """
        logger.debug("Stopping replica %d", self.idx)

        # Transition to stopped state to signal the run loop to exit
        self.state = ReplicaState.STOPPED

        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await asyncio.wait_for(
                    self._run_task, timeout=2 * self._run_poll_rate_s
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                # Expected - task was cancelled or timed out
                pass
            except Exception as e:
                logger.warning("Unexpected error while stopping run task: %s", e)

        # Fail any remaining requests in the queue
        failed_requests = []
        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                failed_requests.append(request)
                self.request_queue.task_done()
            except asyncio.QueueEmpty:
                # catching in case the queue became empty
                # between check and get
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
        if self.proc_mesh:
            try:
                await self.proc_mesh.stop()
            except Exception as e:
                logger.warning(
                    "Error stopping proc_mesh for replica %d: %s", self.idx, e
                )

    # Metric-related getters

    @property
    def current_load(self) -> int:
        """Get current load (active requests + queue depth)"""
        return self.active_requests + self.request_queue.qsize()

    def qsize(self) -> int:
        """Get current queue size"""
        return self.request_queue.qsize()

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

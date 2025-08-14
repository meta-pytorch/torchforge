# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Recoverable Process Mesh

This module provides a fault-tolerant wrapper around ProcMesh that automatically
recovers from crashes and failures. The RecoverableProcMesh class maintains the
same API as ProcMesh while adding automatic recovery capabilities.

Key Features:
- **Automatic Recovery**: Detects mesh failures and automatically respawns processes
- **State Management**: Tracks mesh health and recovery status
- **Graceful Degradation**: Handles failures without losing the entire service
- **Context Management**: Supports async context manager for resource cleanup
- **Actor Respawning**: Automatically respawns actors after mesh recovery

Example:
    Basic usage with automatic recovery:

    >>> mesh = RecoverableProcMesh(num_gpus=2)
    >>>
    >>> async def spawn_actor(proc_mesh):
    ...     actor = await proc_mesh.spawn("MyActor", MyActorClass, *args)
    ...     return actor
    >>>
    >>> await mesh.spawn(spawn_actor)
    >>> # Mesh will automatically recover if it fails

    Context manager usage:

    >>> async with RecoverableProcMesh(num_gpus=1) as mesh:
    ...     await mesh.spawn(spawn_actor)
    ...     # Mesh automatically cleaned up on exit
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, TypeVar

from monarch._rust_bindings.monarch_hyperactor.shape import Shape, Slice
from monarch._src.actor.actor_mesh import Actor
from monarch._src.actor.proc_mesh import proc_mesh, ProcMesh
from monarch._src.actor.shape import MeshTrait

T = TypeVar("T", bound=Actor)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MeshState(Enum):
    """
    Enumeration of possible mesh states for tracking recovery status.

    States:
        HEALTHY: Mesh is operational and ready to handle requests
        RECOVERING: Mesh is in the process of recovering from a failure
        UNHEALTHY: Mesh has failed and needs recovery
        STOPPED: Mesh has been explicitly stopped and cannot be used
    """

    HEALTHY = 0
    RECOVERING = 1
    UNHEALTHY = 2
    STOPPED = 3


class RecoverableProcMesh(MeshTrait):
    """
    A fault-tolerant wrapper around ProcMesh with automatic crash recovery.

    This class provides the same API as ProcMesh while adding robust failure detection
    and automatic recovery capabilities. When the underlying mesh crashes or becomes
    unresponsive, it automatically creates a new mesh and respawns all actors.

    The RecoverableProcMesh maintains state tracking to ensure proper recovery sequencing
    and prevents resource leaks during failure scenarios. It's designed for long-running
    services that need high availability.

    Args:
        num_gpus: Number of GPUs to allocate for the process mesh

    Attributes:
        num_gpus: Number of GPUs allocated to this mesh
        state: Current state of the mesh (HEALTHY, RECOVERING, UNHEALTHY, STOPPED)
        healthy: True if the mesh is operational and ready for requests
        failed: True if the mesh has failed and needs recovery

    Example:
        Basic usage with automatic recovery:

        >>> mesh = RecoverableProcMesh(num_gpus=2)
        >>>
        >>> async def setup_actor(proc_mesh):
        ...     actor = await proc_mesh.spawn("MyActor", MyActorClass)
        ...     await actor.initialize.call()
        >>>
        >>> await mesh.spawn(setup_actor)
        >>> # If mesh fails, it will automatically recover and re-run setup_actor

        Context manager for automatic cleanup:

        >>> async with RecoverableProcMesh(num_gpus=1) as mesh:
        ...     await mesh.spawn(setup_actor)
        ...     # Use mesh for operations
        ...     # Mesh automatically stopped and cleaned up on exit

        Manual state checking:

        >>> if mesh.healthy:
        ...     # Safe to use mesh
        ...     pass
        >>> elif mesh.failed:
        ...     # Mesh needs recovery
        ...     await mesh.spawn(setup_actor)  # Triggers recovery
    """

    def __init__(
        self,
        num_procs: int,
    ) -> None:
        self.num_procs = num_procs
        self._proc_mesh: Optional[ProcMesh] = None
        self._recovery_task: Optional[asyncio.Task[None]] = None
        self.state: MeshState = MeshState.UNHEALTHY

    async def spawn(
        self, hook: Callable[[ProcMesh], Coroutine[Any, Any, None]]
    ) -> None:
        """
        Spawn actors on the mesh with automatic recovery.

        This method ensures the mesh is healthy before spawning actors. If the mesh
        has failed, it automatically triggers recovery and then executes the spawn hook.
        The hook function receives the underlying ProcMesh and should handle actor
        creation and initialization.

        Args:
            hook: Async function that receives a ProcMesh and spawns/initializes actors

        Example:
            >>> async def setup_actors(proc_mesh):
            ...     actor = await proc_mesh.spawn("MyActor", MyActorClass)
            ...     await actor.setup.call()
            >>>
            >>> await mesh.spawn(setup_actors)
        """
        await self._background_spawn(hook)

    def trigger_spawn(
        self, hook: Callable[[ProcMesh], Coroutine[Any, Any, None]]
    ) -> None:
        self._background_spawn(hook)

    def _background_spawn(
        self, hook: Callable[[ProcMesh], Coroutine[Any, Any, None]]
    ) -> asyncio.Task[None]:
        if self.state == MeshState.STOPPED:
            logger.warning("ProcMesh was already stopped when trying to spawn")

        self.state = MeshState.RECOVERING
        self._recovery_task = asyncio.create_task(self._recover(hook))

        return self._recovery_task

    def gpus(self) -> int:
        return self.num_procs

    async def _recover(
        self, hook: Callable[[ProcMesh], Coroutine[Any, Any, None]]
    ) -> None:
        self.state = MeshState.RECOVERING

        old_proc_mesh = self._proc_mesh
        self._proc_mesh = None

        if old_proc_mesh is not None:
            try:
                await old_proc_mesh.stop()
            except Exception as e:
                logger.warning(f"Error stopping old ProcMesh: {e}")

        try:
            self._proc_mesh = await proc_mesh(gpus=self.num_procs)
            if self._proc_mesh is not None:
                await hook(self._proc_mesh)
            self.state = MeshState.HEALTHY

        except Exception as e:
            logger.exception(f"Recovery attempt failed: {e}")
            self.state = MeshState.UNHEALTHY

    @property
    def healthy(self) -> bool:
        return self.state == MeshState.HEALTHY

    @property
    def failed(self) -> bool:
        return self.state == MeshState.UNHEALTHY

    async def stop(self) -> None:
        """
        Stop the mesh and clean up all resources.

        Gracefully shuts down the underlying ProcMesh and marks this recoverable
        mesh as stopped. Once stopped, the mesh cannot be used for further operations.

        This method is idempotent - calling it multiple times is safe.

        Example:
            >>> await mesh.stop()
            >>> # Mesh is now stopped and cannot be used
        """
        logger.info("Stopping RecoverableProcMesh")
        if self.state == MeshState.STOPPED:
            logger.info("RecoverableProcMesh was already stopped")
            return
        try:
            if self._proc_mesh is not None:
                await self._proc_mesh.stop()
        except RuntimeError as e:
            logger.warning("RecoverableProcMesh could not be stopped: %s", e)

        self.state = MeshState.STOPPED

    async def __aenter__(self) -> "RecoverableProcMesh":
        """Enter the async context manager."""
        if self.state == MeshState.STOPPED:
            raise RuntimeError("RecoverableProcMesh has already been stopped")
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        """Exit the async context manager."""
        # In case there are multiple nested "async with" statements, we only
        # want it to close once.
        if self.state != MeshState.STOPPED:
            await self.stop()

    def mark_failed(self):
        """
        Mark the mesh as failed, triggering recovery on next spawn.

        This method is typically called when an operation on the mesh fails
        or when external monitoring detects that the mesh is unresponsive.
        The next call to spawn() will trigger automatic recovery.

        Example:
            >>> try:
            ...     # Some operation that might fail
            ...     await actor.some_method.call()
            >>> except Exception:
            ...     mesh.mark_failed()  # Mark for recovery
        """
        self.state = MeshState.UNHEALTHY

    @property
    def _shape(self) -> Shape:
        if self._proc_mesh is None:
            raise RuntimeError("ProcMesh not initialized")
        return self._proc_mesh._shape

    @property
    def _ndslice(self) -> Slice:
        if self._proc_mesh is None:
            raise RuntimeError("ProcMesh not initialized")
        return self._proc_mesh._ndslice

    @property
    def _labels(self) -> list[str]:
        if self._proc_mesh is None:
            raise RuntimeError("ProcMesh not initialized")
        return self._proc_mesh._labels

    def _new_with_shape(self, shape: Shape) -> "RecoverableProcMesh":
        raise NotImplementedError(
            "RecoverableProcMesh does not support _new_with_shape"
        )

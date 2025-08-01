# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import random
from typing import (
    Any,
    Coroutine,
    Generator,
    Generic,
    Optional,
    overload,
    ParamSpec,
    Tuple,
    Type,
    TypeVar,
)

from monarch._rust_bindings.monarch_hyperactor.shape import Shape
from monarch._src.actor.actor_mesh import Actor, ActorMeshRef, Endpoint, ValueMesh

from monarch._src.actor.future import Future
from monarch._src.actor.shape import MeshTrait, NDSlice

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
P = ParamSpec("P")
R = TypeVar("R")
A = TypeVar("A")


class StackedEndpoint(Generic[P, R]):
    """
    A class that represents a collection of endpoints stacked together.

    This class allows for operations to be performed across multiple endpoints
    as if they were a single entity.

    This provides the same interface as the Endpoint class.

    """

    def __init__(self, endpoints: list[Endpoint]) -> None:
        self.endpoints = endpoints

    def choose(self, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        """Load balanced sends a message to one chosen actor from all stacked actors."""
        endpoint = random.choice(self.endpoints)
        return endpoint.choose(*args, **kwargs)

    def call(self, *args: P.args, **kwargs: P.kwargs) -> "list[Future[ValueMesh[R]]]":
        """Sends a message to all actors in all stacked endpoints and collects results.

        Currently this returns a list of futures rather than a single future, due to
        changes in Monarch.

        """
        return [endpoint.call(*args, **kwargs) for endpoint in self.endpoints]

    def _stream(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> "Generator[Coroutine[Any, Any, R], None, None]":
        """
        Broadcasts to all actors in all stacked endpoints and yields their responses as coroutines.

        This enables processing results from multiple stacked endpoints incrementally as
        they become available. Returns a generator of coroutines that can be awaited.
        """
        for endpoint in self.endpoints:
            # Get the coroutines from each endpoint's _stream method
            for coro in endpoint._stream(*args, **kwargs):
                yield coro

    def stream(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> Generator[Future[R], None, None]:
        """Broadcasts to all actors in all stacked endpoints and yields responses as a stream."""
        for endpoint in self.endpoints:
            # endpoint.stream() returns a Generator[Future[R], None, None]
            for future in endpoint.stream(*args, **kwargs):
                yield future

    def broadcast(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Fire-and-forget broadcast to all actors in all stacked endpoints."""
        for endpoint in self.endpoints:
            endpoint.broadcast(*args, **kwargs)


class StackedActorMeshRef(MeshTrait):
    def __init__(self, *actors: ActorMeshRef, interface=None) -> None:
        self._actors = actors
        self._interface = interface

        # Create endpoints by looking at the interface class for endpoint methods
        if interface is not None and actors:
            # Look for methods decorated with @endpoint in the interface class
            for attr_name in dir(interface):
                if not attr_name.startswith("_"):  # Skip private methods
                    # Check if this method exists as an endpoint on the first actor
                    first_actor = actors[0]
                    if hasattr(first_actor, attr_name):
                        first_endpoint = getattr(first_actor, attr_name)
                        if isinstance(first_endpoint, Endpoint):
                            # Get the corresponding endpoint from each mesh
                            endpoints = []
                            for mesh in self._actors:
                                if hasattr(mesh, attr_name):
                                    endpoint = getattr(mesh, attr_name)
                                    if isinstance(endpoint, Endpoint):
                                        endpoints.append(endpoint)

                            # Create a stacked endpoint with all the collected endpoints
                            if endpoints and len(endpoints) == len(self._actors):
                                setattr(self, attr_name, StackedEndpoint(endpoints))

    def __getattr__(self, name: str) -> StackedEndpoint:
        """
        Fallback for accessing dynamically created endpoint attributes.
        This helps the type checker understand that any attribute access
        on a StackedActorMeshRef should return a StackedEndpoint.
        """
        # This should only be called if the attribute doesn't exist
        # which means it wasn't created in __init__, so we raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @property
    def _ndslice(self) -> NDSlice:
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )

    @property
    def _labels(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )

    def _new_with_shape(self, shape: Shape) -> "StackedActorMeshRef":
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )


def _common_ancestor(*actors: ActorMeshRef) -> Optional[Type]:
    """Finds the common ancestor class of a list of actor mesh references.

    This determines the most specific common base class shared by all
    provided actors.

    Args:
        *actors: Variable number of ActorMeshRef instances to analyze

    Returns:
        Optional[Type]: The most specific common ancestor class, or None if no
                       actors were provided or no common ancestor exists

    Example:
        ```python
        # Find common ancestor of two counter actors
        counter_a = proc.spawn("counter_a", CounterA, 0).get()
        counter_b = proc.spawn("counter_b", CounterB, 0).get()
        common_class = _common_ancestor(counter_a, counter_b)  # Returns Counter
        ```
    """
    if not actors:
        return None
    base_classes = [obj._class for obj in actors]
    all_mros = [inspect.getmro(cls) for cls in base_classes]
    common_bases = set(all_mros[0]).intersection(*all_mros[1:])
    if common_bases:
        return min(
            common_bases, key=lambda cls: min(mro.index(cls) for mro in all_mros)
        )
    return None


@overload
def stack(*actors: Any, interface: Type[T]) -> StackedActorMeshRef:
    pass


@overload
def stack(*actors: Any) -> StackedActorMeshRef:
    pass


def stack(*actors: Any, interface: Optional[Type] = None) -> StackedActorMeshRef:
    """Stacks multiple actor mesh references into a single unified interface.

    This allows you to combine multiple actors that share a common interface
    into a single object that can be used to interact with all of them simultaneously.
    When methods are called on the stacked actor, they are distributed to all
    underlying actors according to the endpoint's behavior (choose, call, stream, etc).

    Args:
        *actors: Variable number of ActorMeshRef instances to stack together
        interface: Optional class that defines the interface to expose. If not provided,
                  the common ancestor class of all actors will be used.

    Returns:
        StackedActorMeshRef: A reference that provides access to all stacked actors
                            through a unified interface.

    Raises:
        TypeError: If any of the provided objects is not an ActorMeshRef, or if
                  no common ancestor can be found and no interface is provided.

    Example:
        ```python
        # Stack two counter actors together
        counter1 = proc1.spawn("counter1", Counter, 0).get()
        counter2 = proc2.spawn("counter2", Counter, 0).get()
        stacked = stack(counter1, counter2)

        # Call methods on all actors at once
        stacked.incr.broadcast()  # Increments both counters
        ```

    """
    for actor in actors:
        if not isinstance(actor, ActorMeshRef):
            raise TypeError(
                "stack be provided with Monarch Actors, got {}".format(type(actor))
            )
    if interface is None:
        interface = _common_ancestor(*actors)

    if interface is None or interface == Actor:
        raise TypeError(
            "No common ancestor found for the given actors. Please provide an interface explicitly."
        )
    return StackedActorMeshRef(*actors, interface=interface)

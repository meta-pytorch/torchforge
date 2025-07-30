# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for monarch_utils.py.

Run this with:
$ pytest ./tests/unit_tests/test_controller.py

"""

import operator

import pytest
from apps.toy_rl.monarch_utils.stack import _common_ancestor, stack, StackedActorMeshRef
from monarch.actor import Accumulator, Actor, endpoint, local_proc_mesh


class Counter(Actor):
    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        self.v += 1

    @endpoint
    async def value(self) -> int:
        return self.v


class CounterA(Counter):
    def __init__(self, v: int, step: int = 1):
        super().__init__(v)
        self.step = step

    @endpoint
    async def decr(self):
        """Decrement the counter by step.
        This is a function that is unique to CounterA.
        """
        self.v -= self.step

    @endpoint
    async def incr(self):
        self.v += self.step


class CounterB(Counter):
    def __init__(self, v: int, multiplier: int = 2):
        super().__init__(v)
        self.multiplier = multiplier

    @endpoint
    async def reset(self):
        """Reset the counter to 0.
        This is a function that is unique to CounterB.
        """
        self.v = 0

    @endpoint
    async def incr(self):
        self.v *= self.multiplier


class CounterC(Actor):
    def __init__(self, v: int, increment: int = 1):
        self.v = v
        self.increment = increment

    @endpoint
    async def incr(self):
        self.v += self.increment

    @endpoint
    async def value(self) -> int:
        return self.v

    @endpoint
    async def double(self):
        self.v *= 2


class CounterD(CounterA):
    def __init__(self, v: int, step: int = 1, factor: int = 3):
        super().__init__(v, step)
        self.factor = factor

    @endpoint
    async def multiply(self):
        """Multiply the counter by factor.
        This is a function that is unique to CounterD.
        """
        self.v *= self.factor

    @endpoint
    async def decr(self):
        """Override decr to decrement by step * factor."""
        self.v -= self.step * self.factor


def test_common_ancestor():
    proc = local_proc_mesh(gpus=1).get()

    # Test with same class
    counter1 = proc.spawn("counter1", Counter, 0).get()
    counter2 = proc.spawn("counter2", Counter, 0).get()
    assert _common_ancestor(counter1, counter2) == Counter

    # Test with parent-child relationship
    counter_a = proc.spawn("counter_a", CounterA, 0).get()
    assert _common_ancestor(counter1, counter_a) == Counter

    # Test with siblings
    counter_b = proc.spawn("counter_b", CounterB, 0).get()
    assert _common_ancestor(counter_a, counter_b) == Counter

    # Test with unrelated classes
    counter_c = proc.spawn("counter_c", CounterC, 0).get()
    assert _common_ancestor(counter_a, counter_c) == Actor

    # Test with empty list
    assert _common_ancestor() is None

    # Test with mixed hierarchy
    assert _common_ancestor(counter1, counter_a, counter_b) == Counter
    assert _common_ancestor(counter_a, counter_b, counter_c) == Actor


def test_identical_actor_stack():
    proc1 = local_proc_mesh(gpus=1).get()
    proc2 = local_proc_mesh(gpus=1).get()

    counter1 = proc1.spawn("counter1", Counter, 0).get()
    counter2 = proc2.spawn("counter2", Counter, 0).get()

    stacked = stack(counter1, counter2)
    assert stacked is not None
    assert isinstance(stacked, StackedActorMeshRef)

    result = stacked.incr.call()
    [r.get() for r in result]
    assert counter1.value.choose().get() == 1
    assert counter2.value.choose().get() == 1


def test_heterogeneous_actor_stack():
    """Test stacking actors of different types that share a common ancestor."""
    proc = local_proc_mesh(gpus=1).get()

    # Create different types of counters
    counter = proc.spawn("counter", Counter, 0).get()
    counter_a = proc.spawn("counter_a", CounterA, 0).get()
    counter_b = proc.spawn("counter_b", CounterB, 0).get()

    # Stack them together - they should use Counter as the common interface
    stacked = stack(counter, counter_a, counter_b)

    # Verify the stacked actor has the common endpoints
    assert hasattr(stacked, "incr")
    assert hasattr(stacked, "value")

    # Verify unique endpoints are not accessible on the stacked actor
    assert not hasattr(stacked, "decr")  # CounterA specific
    assert not hasattr(stacked, "reset")  # CounterB specific

    # Test that the common endpoints work
    res = stacked.incr.call()
    [r.get() for r in res]

    # Verify each actor was affected according to its implementation
    assert counter.value.choose().get() == 1  # Regular counter: +1
    assert counter_a.value.choose().get() == 1  # CounterA: +step (default 1)
    assert counter_b.value.choose().get() == 0  # CounterB: *multiplier (default 2)


def test_stack_with_custom_interface():
    """Test stacking actors with a specified interface."""
    proc = local_proc_mesh(gpus=1).get()

    # Create different types of counters
    counter_a = proc.spawn("counter_a", CounterA, 0).get()
    counter_d = proc.spawn("counter_d", CounterD, 0).get()

    # Without specifying interface, they would use CounterA as common ancestor
    # But we want to use Counter interface instead
    stacked = stack(counter_a, counter_d, interface=Counter)

    # Verify the stacked actor has only Counter endpoints
    assert hasattr(stacked, "incr")
    assert hasattr(stacked, "value")

    # Verify CounterA/CounterD specific endpoints are not accessible
    assert not hasattr(stacked, "decr")  # Should not be available
    assert not hasattr(stacked, "multiply")  # CounterD specific

    # Test that the common endpoints work
    res = stacked.incr.call()
    [r.get() for r in res]

    # Verify each actor was affected according to its implementation
    assert counter_a.value.choose().get() == 1  # CounterA: +step (default 1)
    assert counter_d.value.choose().get() == 1  # CounterD: +step (default 1)


def test_stacked_endpoint_consistency():
    """Tests that the StackedEndpoint shares the same APIs as EndPoint."""

    proc1 = local_proc_mesh(gpus=1).get()
    proc2 = local_proc_mesh(gpus=1).get()
    counter1 = proc1.spawn("counter1", Counter, 0).get()
    counter2 = proc2.spawn("counter2", Counter, 0).get()

    stacked = stack(counter1, counter2)
    regular_endpoint = counter1.incr
    stacked_endpoint = stacked.incr

    # Verify both endpoints implement all methods from EndpointInterface
    for method_name in ["call", "broadcast", "choose", "stream"]:
        assert hasattr(regular_endpoint, method_name), f"Endpoint missing {method_name}"
        assert hasattr(
            stacked_endpoint, method_name
        ), f"StackedEndpoint missing {method_name}"


def test_stacked_endpoint_choose():
    """Tests that the StackedEndpoint.choose method works correctly."""
    proc1 = local_proc_mesh(gpus=1).get()
    proc2 = local_proc_mesh(gpus=1).get()
    counter1 = proc1.spawn("counter1", Counter, 0).get()
    counter2 = proc2.spawn("counter2", Counter, 0).get()

    stacked = stack(counter1, counter2)
    stacked_endpoint = stacked.incr

    # Test choose
    stacked_endpoint.choose().get()
    # At least one counter should be incremented
    assert counter1.value.choose().get() + counter2.value.choose().get() >= 1


def test_stacked_endpoint_call():
    """Tests that the StackedEndpoint.call method works correctly."""
    proc1 = local_proc_mesh(gpus=1).get()
    proc2 = local_proc_mesh(gpus=1).get()
    counter1 = proc1.spawn("counter1", Counter, 0).get()
    counter2 = proc2.spawn("counter2", Counter, 0).get()

    stacked = stack(counter1, counter2)
    stacked_endpoint = stacked.incr

    # Test call
    result = stacked_endpoint.call()
    [r.get() for r in result]
    assert isinstance(result, list)
    assert len(result) == 2

    # Verify both counters were incremented
    assert counter1.value.choose().get() == 1
    assert counter2.value.choose().get() == 1


def test_stacked_endpoint_broadcast():
    """Tests that the StackedEndpoint.broadcast method works correctly."""
    proc1 = local_proc_mesh(gpus=1).get()
    proc2 = local_proc_mesh(gpus=1).get()
    counter1 = proc1.spawn("counter1", Counter, 0).get()
    counter2 = proc2.spawn("counter2", Counter, 0).get()

    stacked = stack(counter1, counter2)
    stacked_endpoint = stacked.incr

    # Test broadcast
    stacked_endpoint.broadcast()
    # Both counters should be incremented
    assert counter1.value.choose().get() == 1
    assert counter2.value.choose().get() == 1


@pytest.mark.asyncio
async def test_stacked_endpoint_stream():
    """Tests that the StackedEndpoint.stream method works correctly."""
    proc1 = await local_proc_mesh(gpus=1)
    proc2 = await local_proc_mesh(gpus=1)
    counter1 = await proc1.spawn("counter1", Counter, 0)
    counter2 = await proc2.spawn("counter2", Counter, 0)

    stacked = stack(counter1, counter2)
    stacked_endpoint = stacked.incr

    # Test stream
    async def test_stream():
        return [await x for x in stacked_endpoint.stream()]

    results = await test_stream()
    assert len(results) == 2

    # Verify both counters were incremented
    assert counter1.value.choose().get() == 1
    assert counter2.value.choose().get() == 1


def test_stacked_actor_with_accumulator():
    """Tests that Accumulator works correctly with StackedActor endpoints."""
    proc1 = local_proc_mesh(gpus=1).get()
    proc2 = local_proc_mesh(gpus=1).get()

    counter1 = proc1.spawn("counter1", Counter, 5).get()
    counter2 = proc2.spawn("counter2", Counter, 10).get()
    stacked = stack(counter1, counter2)

    acc = Accumulator(stacked.value, 0, operator.add)
    result = acc.accumulate().get()
    assert result == 15

    stacked.incr.broadcast()
    result = acc.accumulate().get()
    assert result == 17

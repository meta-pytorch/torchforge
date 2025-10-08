# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for router.py
"""

import asyncio
import logging

import pytest
from forge.controller import ForgeActor
from forge.controller.service import (
    Batcher,
    LeastLoadedRouter,
    Replica,
    ReplicaState,
    RoundRobinRouter,
    service_endpoint,
    SessionRouter,
)

from forge.types import ProcessConfig
from monarch.actor import endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Counter(ForgeActor):
    """Test actor that maintains a counter with various endpoints."""

    def __init__(self, v: int):
        self.v = v
        self._num_calls = 0  # number of calls to endpoint functions

    @endpoint
    async def value(self) -> int:
        """Get the current counter value."""
        return self.v

    @endpoint
    async def fail_me(self):
        """Endpoint that always fails to test error handling."""
        raise RuntimeError("I was asked to fail")

    @endpoint
    async def get_num_calls(self):
        """Get the number of calls to endpoint functions."""
        return self._num_calls

    @endpoint
    async def incr(self):
        """Increment the counter."""
        self._num_calls += 1
        self.v += 1

    @service_endpoint(router=RoundRobinRouter, batch_size=3, batch_timeout=1)
    async def rr_batch_incr_bsize3(self):
        """Increment the round-robin counter with batching (batch size = 3)."""
        self._num_calls += 1
        self.v += 1

    @service_endpoint(router=RoundRobinRouter, batch_size=5, batch_timeout=0.05)
    async def rr_batch_incr_bsize5(self, inputs: list[int]) -> list[int]:
        """Increment the round-robin counter with batching (batch size = 5)."""
        self._num_calls += 1
        self.v += sum(inputs)
        return inputs

    @service_endpoint(router=RoundRobinRouter)
    async def rr_batch_incr_bsize1(self, inputs: list[int]) -> list[int]:
        """Increment the round-robin counter with batching (batch size = 1)."""
        self._num_calls += 1
        self._sum += sum(inputs)
        return inputs


def make_replica(idx: int, healthy: bool = True, load: int = 0) -> Replica:
    """Helper to build a replica with specified state and load."""
    replica = Replica(
        idx=idx,
        proc_config=ProcessConfig(),
        actor_def=Counter,
        actor_args=(),
        actor_kwargs={},
    )
    replica.state = ReplicaState.HEALTHY if healthy else ReplicaState.UNHEALTHY
    replica.active_requests = load
    return replica


# Cnofig tests


@pytest.mark.asyncio
async def test_service_endpoint_router_and_configurations():
    """
    Verify service endpoints are registered with correct router/batching configuration:
    - rr_batch_incr_bsize1: plain RoundRobinRouter, no batching (batch_size=1, timeout=0.01)
    - rr_batch_incr_bsize3: Batcher wrapping RoundRobinRouter (batch_size=3, timeout=1)
    - incr: plain @endpoint, should not appear in service.routers
    """
    service = await Counter.options(procs=1, num_replicas=2).as_service(v=0)

    try:
        # --- rr_batch_incr_bsize1 ---
        router1 = service.routers.get("rr_batch_incr_bsize1")
        assert isinstance(
            router1, RoundRobinRouter
        ), f"Expected RoundRobinRouter, got {type(router1)}"

        prop1 = Counter.rr_batch_incr_bsize1
        assert prop1.batch_size == 1
        assert prop1.batch_timeout == 0.01

        # --- rr_batch_incr_bsize3 ---
        router3 = service.routers.get("rr_batch_incr_bsize3")

        assert isinstance(router3, Batcher), f"Expected Batcher, got {type(router3)}"
        assert router3.batch_size == 3
        assert router3.batch_timeout == 1

        # --- incr ---
        assert (
            "incr" not in service.routers
        ), "Plain @endpoint should not be in service.routers"

    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_service_endpoint_with_invalid_router_noncallable():
    """@service_endpoint with non-callable router should raise ValueError."""

    class BadActor(ForgeActor):
        @service_endpoint(router="roundrobin")  # string, not callable
        async def bad_endpoint(self):
            return 42

    with pytest.raises(ValueError, match="Router must be callable"):
        # Triggers ServiceInterface._set_router during construction
        await BadActor.options(num_replicas=1).as_service()


@pytest.mark.asyncio
async def test_service_endpoint_with_invalid_router_wrong_return_type():
    """@service_endpoint with callable that doesn't return Router should raise ValueError."""

    class NotARouter:
        """Dummy class that is not a Router."""

    class BadActor(ForgeActor):
        @service_endpoint(router=NotARouter)  # returns NotARouter
        async def bad_endpoint(self):
            return 123

    with pytest.raises(ValueError, match="Router must be a Router instance"):
        await BadActor.options(num_replicas=1).as_service()


# Router Tests


@pytest.mark.asyncio
async def test_session_router_fallback_rr_vs_ll():
    """Switch fallback router to round-robin and verify assignment order."""
    # Choose RoundRobinRouter as fallback, r1 and r2 should be assigned to different replicas
    replicas = [make_replica(0, load=0), make_replica(1, load=5)]
    session_map = {}
    fallback = RoundRobinRouter()
    router = SessionRouter(fallback)

    r1 = router.get_replica(replicas, sess_id="sess1", session_map=session_map)
    r2 = router.get_replica(replicas, sess_id="sess2", session_map=session_map)

    assert r1.idx != r2.idx
    assert set(session_map.values()) == {0, 1}

    # If LeastLoadedRouter as fallback, r1 and r2 should be assigned to same replicas
    replicas = [make_replica(0, load=0), make_replica(1, load=5)]
    session_map = {}
    fallback = LeastLoadedRouter()
    router = SessionRouter(fallback)

    r1 = router.get_replica(replicas, sess_id="sess1", session_map=session_map)
    r2 = router.get_replica(replicas, sess_id="sess2", session_map=session_map)

    assert r1.idx == r2.idx == 0


# Router integeration tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_round_robin_router_distribution():
    """Test that the RoundRobinRouter distributes sessionless calls evenly across replicas."""
    service = await Counter.options(procs=1, num_replicas=3).as_service(v=0)

    try:
        # Make multiple sessionless calls using route()
        results = []
        for _ in range(6):
            await service.incr.route()
            values = await service.value.fanout()
            results.append(values)
        # Verify that requests were distributed round-robin
        # Each call increments a single replica, so after 6 calls we expect:
        # 2 increments per replica (since 3 replicas, 6 calls)
        final_values = results[-1]  # last snapshot
        assert sorted(final_values) == [2, 2, 2]

    finally:
        await service.shutdown()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_session_router_assigns_and_updates_session_map_in_service():
    """Integration: Service with SessionRouter preserves sticky sessions."""
    service = await Counter.options(
        procs=1,
        num_replicas=2,
    ).as_service(v=0)

    try:
        # First call with sess_id -> assign a replica
        await service.incr.route(sess_id="sess1")
        values1 = await service.value.fanout()

        # Second call with same sess_id -> must hit same replica
        await service.incr.route(sess_id="sess1")
        values2 = await service.value.fanout()

        # Difference should only be on one replica (sticky session)
        diffs = [v2 - v1 for v1, v2 in zip(values1, values2)]
        assert (
            sum(diffs) == 1
        ), f"Expected exactly one replica to increment, got {diffs}"
        assert max(diffs) == 1 and min(diffs) == 0

        # Session map in service should reflect assigned replica
        assigned_idx = service._session_replica_map["sess1"]
        assert values2[assigned_idx] == values1[assigned_idx] + 1

    finally:
        await service.shutdown()


# Batcher tests


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_service_as_actor_preserves_normal_usage():
    """Ensure that using `as_actor` does not break normal semantics."""
    service = await Counter.as_actor(5)

    try:
        assert await service.value.choose() == 5

        # Test increment
        await service.rr_batch_incr_bsize3.choose()
        assert await service.value.choose() == 6

    finally:
        await Counter.shutdown(service)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_rr_batch_incr_bsize5_behaves_like_normal_incr():
    """Ensure rr_batch_incr_bsize5 (batch_size=5) behaves like a normal incr endpoint for single calls."""
    service = await Counter.options(procs=1, num_replicas=1).as_service(v=5)

    try:
        # Initial value
        assert await service.value.route() == 5

        # Call batched increment once
        await service.rr_batch_incr_bsize5.route(1)

        # Should increment exactly once
        assert await service.value.route() == 6

    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_service_endpoint_batching_preserves_order():
    """Ensure that batching preserves the order of calls."""
    service = await Counter.options(num_replicas=2, procs=1).as_service(0)
    try:
        results = await asyncio.gather(
            *[service.rr_batch_incr_bsize5.route(i) for i in range(5)]
        )
        assert results == [0, 1, 2, 3, 4]
        assert await service.get_num_calls.route() == 1
        assert sorted(await service.value.fanout()) == [0, 10]
    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_service_endpoint_multiple_batches():
    """
    Verify that batching correctly splits requests into two batches —
    one triggered by reaching the batch size limit and another by the batch timeout.
    """
    service = await Counter.options(num_replicas=2, procs=1).as_service(0)
    try:
        # Enqueue 7 calls → expect two batches (5 + 2)
        results = await asyncio.gather(
            *[service.rr_batch_incr_bsize5.route(i) for i in range(7)]
        )
        # Verify all individual results were returned in order
        assert results == [0, 1, 2, 3, 4, 5, 6]
        # Each replica should have executed one batch (round-robin)
        assert await service.get_num_calls.fanout() == [1, 1]

        # Replica values reflect the sum of their respective batch inputs
        # first batch: [0, 1, 2, 3, 4] → 10
        # second batch: [5, 6] → 11
        assert sorted(await service.value.fanout()) == [10, 11]
    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_round_robin_batcher_distribution_no_args():
    """
    Verify that the batching system correctly handles endpoints with **zero arguments**
    and that the RoundRobinRouter distributes such batched calls evenly across replicas.
    """

    # --- Launch service with 3 replicas ---
    service = await Counter.options(procs=1, num_replicas=3).as_service(v=0)

    try:
        # Enqueue 5 no-arg batched calls
        await asyncio.gather(*[service.rr_batch_incr_bsize3.route() for _ in range(5)])

        # Check that two replicas incremented their counters once
        values = await service.value.fanout()
        assert sorted(values) == [0, 1, 1], f"Unexpected replica values: {values}"

        # Ensure exactly 2 actor invocations occurred (2 batches total)
        num_calls = await service.get_num_calls.fanout()
        assert sum(num_calls) == 2, f"Expected 2 batches, got {sum(num_calls)}"

    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_service_endpoint_batching_multi_arg_merge():
    """Ensure that batching merges multiple argument lists correctly."""

    class MultiArgActor(ForgeActor):
        def __init__(self):
            self._num_calls = 0

        @endpoint
        async def get_num_calls(self):
            return self._num_calls

        @service_endpoint(router=RoundRobinRouter, batch_size=5, batch_timeout=0.1)
        async def multi_args_sum(self, v1: list[int], v2: list[str]) -> list[str]:
            """
            Endpoint that accepts multiple argument lists.
            Should be invoked once per batch.
            """
            self._num_calls += 1
            # Combine corresponding elements
            return [f"{x}:{y}" for x, y in zip(v1, v2)]

    service = await MultiArgActor.options(num_replicas=2, procs=1).as_service()

    try:
        # 5 requests will fill one batch of size 5
        results = await asyncio.gather(
            *[service.multi_args_sum.route(i, str(i)) for i in range(5)]
        )

        # Expect exactly one actor invocation
        assert await service.get_num_calls.route() == 1

        # Expect results correspond to all merged pairs
        assert results == ["0:0", "1:1", "2:2", "3:3", "4:4"]

    finally:
        await service.shutdown()

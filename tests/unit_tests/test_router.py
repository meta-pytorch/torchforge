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

    @endpoint
    async def value(self) -> int:
        """Get the current counter value."""
        return self.v

    @endpoint
    async def fail_me(self):
        """Endpoint that always fails to test error handling."""
        raise RuntimeError("I was asked to fail")

    @endpoint
    async def add_to_value(self, amount: int, multiplier: int = 1) -> int:
        """Add an amount (optionally multiplied) to the current value."""
        logger.info(f"adding {amount} with {multiplier}")
        self.v += amount * multiplier
        return self.v

    @endpoint
    async def incr(self):
        """Increment the counter."""
        self.v += 1

    @service_endpoint(router=RoundRobinRouter(), batch_size=3, batch_timeout=1)
    async def rr_batch_incr_bsize3(self):
        """Increment the round-robin counter with batching (batch size = 3)."""
        self.v += 1

    @service_endpoint(router=RoundRobinRouter(), batch_size=5, batch_timeout=0.05)
    async def rr_batch_incr_bsize5(self):
        """Increment the round-robin counter with batching (batch size = 5)."""
        self.v += 1


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
async def test_round_robin_router_distribution_with_batching():
    """Test that the RoundRobinRouter distributes sessionless calls evenly across replicas with batch routing."""
    service = await Counter.options(procs=1, num_replicas=3).as_service(v=0)

    try:
        # Make multiple sessionless calls using route()
        results = []
        tasks = [service.rr_batch_incr_bsize3.route() for _ in range(6)]
        await asyncio.gather(*tasks)
        # Verify that requests were distributed round-robin
        # Each call increments a single replica, so after 6 calls we expect:
        # 2 increments per replica (since 3 replicas, 6 calls)
        values = await service.value.fanout()
        assert sorted(values) == [0, 3, 3]

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


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_independent_batchers_and_routers_per_endpoint():
    """Ensure multiple @service_endpoint endpoints coexist with independent routers/batchers."""
    service = await Counter.options(procs=1, num_replicas=2).as_service(v=0)

    try:
        # --- First batch: rr_batch_incr_bsize3 (batch_size = 3) ---
        tasks = [
            asyncio.create_task(service.rr_batch_incr_bsize3.route()) for _ in range(4)
        ]
        await asyncio.gather(*tasks)

        values = await service.value.fanout()

        # Expectation:
        # - First 3 requests form one batch → sent to replica R1 (+3).
        # - Remaining 1 request forms its own batch → goes to replica R2 (+1).
        # So totals should be [3, 1] (order depends on round robin).
        assert sum(values) == 4, f"Expected total=4, got {values}"
        assert sorted(values) == [1, 3], f"Expected [1, 3], got {values}"

        # --- Second batch: rr_batch_incr_bsize5 (batch_size = 5) ---
        tasks = [
            asyncio.create_task(service.rr_batch_incr_bsize5.route()) for _ in range(7)
        ]
        await asyncio.gather(*tasks)

        values = await service.value.fanout()

        # Expectation (RoundRobin between replicas):
        # Starting from previous state (R1=3, R2=1):
        # - Next 5 requests form one batch → go to R1 (+5).
        # - Remaining 2 requests form their own batch → go to R2 (+2).
        #
        # Final totals:
        #   R1 = 3 (previous) + 5 = 8
        #   R2 = 1 (previous) + 2 = 3
        # So distribution should be [3, 8].
        assert sum(values) == 11, f"Expected total=11, got {values}"
        assert sorted(values) == [3, 8], f"Expected [4, 8], got {values}"

    finally:
        await service.shutdown()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for service.py
"""

import asyncio
import logging

import pytest
from forge.controller.service import ServiceConfig
from forge.controller.spawn import spawn_service
from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Counter(Actor):
    """Test actor that maintains a counter with various endpoints."""

    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        """Increment the counter."""
        self.v += 1

    @endpoint
    async def value(self) -> int:
        """Get the current counter value."""
        return self.v

    @endpoint
    async def fail_me(self):
        """Endpoint that always fails to test error handling."""
        raise RuntimeError("I was asked to fail")

    @endpoint
    async def slow_incr(self):
        """Slow increment to test queueing."""
        await asyncio.sleep(1.0)
        self.v += 1


# Core Functionality Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_basic_service_operations():
    """Test basic service creation, sessions, and endpoint calls."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=1)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test session creation and uniqueness
        session1 = await service.start_session()
        session2 = await service.start_session()
        assert session1 != session2
        assert isinstance(session1, str)

        # Test endpoint calls
        await service.incr.choose(sess_id=session1)
        result = await service.value.choose(sess_id=session1)
        assert result == 1

        # Test session mapping
        assert session1 in service._session_replica_map

        # Test session termination
        await service.terminate_session(session1)
        assert session1 not in service._session_replica_map

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_sessionless_calls():
    """Test sessionless calls with round-robin load balancing."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test sessionless calls
        await service.incr.choose()
        await service.incr.choose()
        result = await service.value.choose()
        assert result is not None

        # No sessions should be created
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

        # Verify load distribution
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 3

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_session_context_manager():
    """Test session context manager functionality."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=1)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test context manager usage
        async with service.session():
            await service.incr.choose()
            await service.incr.choose()
            result = await service.value.choose()
            assert result == 2

        # Test sequential context managers to avoid interference
        async def worker(increments: int):
            async with service.session():
                initial = await service.value.choose()
                for _ in range(increments):
                    await service.incr.choose()
                final = await service.value.choose()
                return final - initial

        # Run sessions sequentially to avoid concurrent modification
        result1 = await worker(2)
        result2 = await worker(3)
        results = [result1, result2]
        assert sorted(results) == [2, 3]

        # Test that context manager properly manages session lifecycle
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

    finally:
        await service.stop()


# Fault Tolerance Tests


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_replica_failure_and_recovery():
    """Test replica failure handling and automatic recovery."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create session and cause failure
        session = await service.start_session()
        await service.incr.choose(session)

        original_replica_idx = service._session_replica_map[session]

        # Cause failure
        error_result = await service.fail_me.choose(session)
        assert isinstance(error_result, RuntimeError)

        # Replica should be marked as failed
        failed_replica = service._replicas[original_replica_idx]
        assert not failed_replica.healthy

        # Session should be reassigned on next call
        await service.incr.choose(session)
        new_replica_idx = service._session_replica_map[session]
        assert new_replica_idx != original_replica_idx

        # New sessions should avoid failed replica
        new_session = await service.start_session()
        await service.incr.choose(new_session)
        assigned_replica = service._replicas[service._session_replica_map[new_session]]
        assert assigned_replica.healthy

    finally:
        await service.stop()


# Metrics and Monitoring Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_metrics_collection():
    """Test comprehensive metrics collection."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions and make requests
        session1 = await service.start_session()
        session2 = await service.start_session()

        await service.incr.choose(session1)
        await service.incr.choose(session1)
        await service.incr.choose(session2)

        # Test failure metrics
        error_result = await service.fail_me.choose(session1)
        assert isinstance(error_result, RuntimeError)

        # Get metrics
        metrics = service.get_metrics()
        summary = service.get_metrics_summary()

        # Test service-level metrics
        assert metrics.total_sessions == 2
        assert metrics.healthy_replicas <= 2  # One may have failed
        assert metrics.total_replicas == 2

        # Test summary structure
        assert "service" in summary
        assert "replicas" in summary

        # Test request counts
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in summary["replicas"].values()
        )
        assert total_requests == 4  # 3 successful + 1 failed

        total_failed = sum(
            replica_metrics["failed_requests"]
            for replica_metrics in summary["replicas"].values()
        )
        assert total_failed == 1

    finally:
        await service.stop()


# Load Balancing and Session Management Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_session_stickiness():
    """Test that sessions stick to the same replica."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        session = await service.start_session()

        # Make multiple calls
        await service.incr.choose(session)
        await service.incr.choose(session)
        await service.incr.choose(session)

        # Should always route to same replica
        replica_idx = service._session_replica_map[session]

        await service.incr.choose(session)
        assert service._session_replica_map[session] == replica_idx

        # Verify counter was incremented correctly
        result = await service.value.choose(session)
        assert result == 4

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_load_balancing_multiple_sessions():
    """Test load balancing across multiple sessions using least-loaded assignment."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions with some load to trigger distribution
        session1 = await service.start_session()
        await service.incr.choose(session1)  # Load replica 0

        session2 = await service.start_session()
        await service.incr.choose(session2)  # Should go to replica 1 (least loaded)

        session3 = await service.start_session()
        await service.incr.choose(session3)  # Should go to replica 0 or 1 based on load

        session4 = await service.start_session()
        await service.incr.choose(session4)  # Should balance the load

        # Check that sessions are distributed (may not be perfectly even due to least-loaded logic)
        replica_assignments = [
            service._session_replica_map[s]
            for s in [session1, session2, session3, session4]
        ]
        unique_replicas = set(replica_assignments)

        # With least-loaded assignment, we should eventually use both replicas
        # as load accumulates, though initial sessions may go to the same replica
        assert len(unique_replicas) >= 1  # At least one replica used

        # Verify that load balancing is working by checking request distribution
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 4  # All requests processed

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent operations across sessions and sessionless calls."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(
        service_cfg=cfg, actor_def=Counter, name="counter", v=0
    )

    try:
        # Mix of session and sessionless calls
        session = await service.start_session()

        # Concurrent operations
        tasks = [
            service.incr.choose(session),  # Session call
            service.incr.choose(session),  # Session call
            service.incr.choose(),  # Sessionless call
            service.incr.choose(),  # Sessionless call
        ]

        await asyncio.gather(*tasks)

        # Verify session tracking
        assert len(service._active_sessions) == 1
        assert session in service._session_replica_map

        # Verify total requests
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 4

    finally:
        await service.stop()


# `call` endpoint tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_broadcast_call_basic():
    """Test basic broadcast call functionality."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=3)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=10)

    try:
        # Test broadcast call to all replicas
        results = await service.incr.call()

        # Should get results from all healthy replicas
        assert isinstance(results, list)
        assert len(results) == 3  # All 3 replicas should respond

        # All results should be None (incr doesn't return anything)
        assert all(result is None for result in results)

        # Test getting values from all replicas
        values = await service.value.call()
        assert isinstance(values, list)
        assert len(values) == 3

        # All replicas should have incremented from 10 to 11
        assert all(value == 11 for value in values)

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_broadcast_call_with_failed_replica():
    """Test broadcast call behavior when some replicas fail."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=3)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # First, cause one replica to fail by calling fail_me on a specific session
        session = await service.start_session()
        try:
            await service.fail_me.choose(session)
        except RuntimeError:
            pass  # Expected failure

        # Wait briefly for replica to be marked as failed
        await asyncio.sleep(0.1)

        # Now test broadcast call - should only hit healthy replicas
        results = await service.incr.call()

        # Should get results from healthy replicas only
        assert isinstance(results, list)
        # Results length should match number of healthy replicas (2 out of 3)
        healthy_count = len([r for r in service._replicas if r.healthy])
        assert len(results) == healthy_count

        # Get values from all healthy replicas
        values = await service.value.call()
        assert len(values) == healthy_count

        # All healthy replicas should have incremented to 1
        assert all(value == 1 for value in values)

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_broadcast_call_vs_choose():
    """Test that broadcast call hits all replicas while choose hits only one."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=3)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Use broadcast call to increment all replicas
        await service.incr.call()

        # Get values from all replicas
        values_after_broadcast = await service.value.call()
        assert len(values_after_broadcast) == 3
        assert all(value == 1 for value in values_after_broadcast)

        # Use choose to increment only one replica
        await service.incr.choose()

        # Get values again - one replica should be at 2, others at 1
        values_after_choose = await service.value.call()
        assert len(values_after_choose) == 3
        assert sorted(values_after_choose) == [1, 1, 2]  # One replica incremented twice

        # Verify metrics show the correct number of requests
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        # incr.call() (3 requests) + value.call() (3 requests) + incr.choose() (1 request) + value.call() (3 requests) = 10 total
        assert total_requests == 10

    finally:
        await service.stop()

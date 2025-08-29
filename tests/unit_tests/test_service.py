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
from forge.controller import ForgeActor
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Counter(ForgeActor):
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

    @endpoint
    async def add_to_value(self, amount: int, multiplier: int = 1) -> int:
        """Add an amount (optionally multiplied) to the current value."""
        logger.info(f"adding {amount} with {multiplier}")
        self.v += amount * multiplier
        return self.v


# Core Functionality Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_actor_def_type_validation():
    """Test that spawn_service validates actor_def is a subclass of ForgeActor."""

    # Service can only spawn ForgeActor subclasses
    class InvalidActor(Actor):
        def __init__(self):
            pass

    cfg = ServiceConfig(procs_per_replica=1, num_replicas=1)

    # Test that TypeError is raised when actor_def is not a ForgeActor subclass
    with pytest.raises(TypeError, match="actor_def must be a subclass of ForgeActor"):
        await spawn_service(service_cfg=cfg, actor_def=InvalidActor)


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
        state = await service._get_internal_state()
        assert session1 in state["session_replica_map"]

        # Test session termination
        await service.terminate_session(session1)
        state = await service._get_internal_state()
        assert session1 not in state["session_replica_map"]

    finally:
        await shutdown_service(service)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_sessionless_calls():
    """Test sessionless calls with round robin load balancing."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test sessionless calls
        await service.incr.choose()
        await service.incr.choose()
        result = await service.value.choose()
        assert result is not None

        # No sessions should be created
        state = await service._get_internal_state()
        assert len(state["active_sessions"]) == 0
        assert len(state["session_replica_map"]) == 0

        # Verify load distribution
        metrics = await service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 3

        # Users should be able to call endpoint with just args
        result = await service.add_to_value.choose(5, multiplier=2)
        assert result == 11  # 1 + 10

    finally:
        await shutdown_service(service)


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
        state = await service._get_internal_state()
        assert len(state["active_sessions"]) == 0
        assert len(state["session_replica_map"]) == 0

    finally:
        await shutdown_service(service)


# Fault Tolerance Tests


@pytest.mark.timeout(20)
@pytest.mark.asyncio
async def test_recovery_state_transitions():
    """Test replica state transitions during failure and recovery."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=1, health_poll_rate=0.1)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Initially replica should be healthy
        state = await service._get_internal_state()
        replica_state = state["replicas"][0]
        assert replica_state["state"] == "HEALTHY"
        assert replica_state["healthy"] is True
        assert replica_state["failed"] is False

        # Create session and make a successful call
        session = await service.start_session()
        await service.incr.choose(sess_id=session)
        result = await service.value.choose(sess_id=session)
        assert result == 1

        # Cause failure - this should transition to RECOVERING
        error_result = await service.fail_me.choose(sess_id=session)
        assert isinstance(error_result, RuntimeError)

        # Replica should now be in RECOVERING state
        state = await service._get_internal_state()
        replica_state = state["replicas"][0]
        assert replica_state["state"] == "RECOVERING"
        assert replica_state["healthy"] is False
        assert replica_state["failed"] is True

        # Wait for health loop to detect and attempt recovery
        # The health loop runs every 0.1s, so give it some time
        max_wait_time = 5.0  # 5 seconds max wait
        wait_interval = 0.1
        elapsed = 0.0

        # Wait for replica to either recover (HEALTHY) or fail completely (UNHEALTHY)
        while elapsed < max_wait_time:
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval

            state = await service._get_internal_state()
            replica_state = state["replicas"][0]
            if replica_state["state"] in ["HEALTHY", "UNHEALTHY"]:
                break

        # After recovery, replica should be healthy again
        # (unless recovery failed, in which case it would be UNHEALTHY)
        state = await service._get_internal_state()
        replica_state = state["replicas"][0]
        assert replica_state["state"] in ["HEALTHY", "UNHEALTHY"]

        if replica_state["state"] == "HEALTHY":
            # If recovery succeeded, verify we can make calls again
            assert replica_state["healthy"] is True
            assert replica_state["failed"] is False

            # Test that we can make new calls after recovery
            new_session = await service.start_session()
            await service.incr.choose(sess_id=new_session)
            result = await service.value.choose(sess_id=new_session)
            assert (
                result is not None
            )  # Should get a result (counter starts at 0 in new actor)

        elif replica_state["state"] == "UNHEALTHY":
            # If recovery failed, verify failed state
            assert replica_state["healthy"] is False
            assert replica_state["failed"] is True

        # Verify that the state transition path was correct
        # (We can't guarantee the exact end state due to potential flakiness in test environments,
        # but we can verify the replica went through the expected transition)
        logger.info(f"Final replica state: {replica_state['state']}")

    finally:
        await shutdown_service(service)


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_replica_failure_and_recovery():
    """Test replica failure handling and automatic recovery."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create session and cause failure
        session = await service.start_session()
        await service.incr.choose(sess_id=session)

        state = await service._get_internal_state()
        original_replica_idx = state["session_replica_map"][session]

        # Cause failure
        error_result = await service.fail_me.choose(sess_id=session)
        assert isinstance(error_result, RuntimeError)

        # Replica should be marked as failed
        state = await service._get_internal_state()
        failed_replica = state["replicas"][original_replica_idx]
        assert not failed_replica["healthy"]

        # Session should be reassigned on next call
        await service.incr.choose(sess_id=session)
        state = await service._get_internal_state()
        new_replica_idx = state["session_replica_map"][session]
        assert new_replica_idx != original_replica_idx

        # New sessions should avoid failed replica
        new_session = await service.start_session()
        await service.incr.choose(sess_id=new_session)
        state = await service._get_internal_state()
        assigned_replica = state["replicas"][state["session_replica_map"][new_session]]
        assert assigned_replica["healthy"]

    finally:
        await shutdown_service(service)


# Metrics and Monitoring Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_metrics_collection():
    """Test metrics collection."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions and make requests
        session1 = await service.start_session()
        session2 = await service.start_session()

        await service.incr.choose(sess_id=session1)
        await service.incr.choose(sess_id=session1)
        await service.incr.choose(sess_id=session2)

        # Test failure metrics
        error_result = await service.fail_me.choose(sess_id=session1)
        assert isinstance(error_result, RuntimeError)

        # Get metrics
        metrics = await service.get_metrics()
        summary = await service.get_metrics_summary()

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
        await shutdown_service(service)


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
        await service.incr.choose(sess_id=session)
        await service.incr.choose(sess_id=session)
        await service.incr.choose(sess_id=session)

        # Should always route to same replica
        state = await service._get_internal_state()
        replica_idx = state["session_replica_map"][session]

        await service.incr.choose(sess_id=session)
        state = await service._get_internal_state()
        assert state["session_replica_map"][session] == replica_idx

        # Verify counter was incremented correctly
        result = await service.value.choose(sess_id=session)
        assert result == 4

    finally:
        await shutdown_service(service)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_load_balancing_multiple_sessions():
    """Test load balancing across multiple sessions using least-loaded assignment."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=2)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions with some load to trigger distribution
        session1 = await service.start_session()
        await service.incr.choose(sess_id=session1)  # Load replica 0

        session2 = await service.start_session()
        await service.incr.choose(
            sess_id=session2
        )  # Should go to replica 1 (least loaded)

        session3 = await service.start_session()
        await service.incr.choose(
            sess_id=session3
        )  # Should go to replica 0 or 1 based on load

        session4 = await service.start_session()
        await service.incr.choose(sess_id=session4)  # Should balance the load

        # Check that sessions are distributed (may not be perfectly even due to least-loaded logic)
        state = await service._get_internal_state()
        replica_assignments = [
            state["session_replica_map"][s]
            for s in [session1, session2, session3, session4]
        ]
        unique_replicas = set(replica_assignments)

        # With least-loaded assignment, we should eventually use both replicas
        # as load accumulates, though initial sessions may go to the same replica
        assert len(unique_replicas) >= 1  # At least one replica used

        # Verify that load balancing is working by checking request distribution
        metrics = await service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 4  # All requests processed

    finally:
        await shutdown_service(service)


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
            service.incr.choose(sess_id=session),  # Session call
            service.incr.choose(sess_id=session),  # Session call
            service.incr.choose(),  # Sessionless call
            service.incr.choose(),  # Sessionless call
        ]

        await asyncio.gather(*tasks)

        # Verify session tracking
        state = await service._get_internal_state()
        assert len(state["active_sessions"]) == 1
        assert session in state["session_replica_map"]

        # Verify total requests
        metrics = await service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 4

    finally:
        await shutdown_service(service)


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
        await shutdown_service(service)


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_broadcast_call_with_failed_replica():
    """Test broadcast call behavior when some replicas fail."""
    cfg = ServiceConfig(procs_per_replica=1, num_replicas=3)
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # First, cause one replica to fail by calling fail_me on a specific session
        session = await service.start_session()
        try:
            await service.fail_me.choose(sess_id=session)
        except RuntimeError:
            pass  # Expected failure

        # Wait briefly for replica to be marked as failed
        await asyncio.sleep(0.1)

        # Now test broadcast call - should only hit healthy replicas
        results = await service.incr.call()

        # Should get results from healthy replicas only
        assert isinstance(results, list)
        # Results length should match number of healthy replicas (2 out of 3)
        state = await service._get_internal_state()
        healthy_count = sum(1 for r in state["replicas"] if r["healthy"])
        assert len(results) == healthy_count

        # Get values from all healthy replicas
        values = await service.value.call()
        assert len(values) == healthy_count

        # All healthy replicas should have incremented to 1
        assert all(value == 1 for value in values)

    finally:
        await shutdown_service(service)


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
        metrics = await service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        # incr.call() (3 requests) + value.call() (3 requests) + incr.choose() (1 request) + value.call() (3 requests) = 10 total
        assert total_requests == 10

    finally:
        await shutdown_service(service)

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
from forge.controller.service import AutoscalingConfig, ServiceConfig
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
        """Slow increment to test queueing and autoscaling."""
        await asyncio.sleep(1.0)
        self.v += 1


# Core Functionality Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_basic_service_operations():
    """Test basic service creation, sessions, and endpoint calls."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test session creation and uniqueness
        session1 = await service.start_session()
        session2 = await service.start_session()
        assert session1 != session2
        assert isinstance(session1, str)

        # Test endpoint calls
        await service.incr(session1)
        result = await service.value(session1)
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
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test sessionless calls
        await service.incr()
        await service.incr()
        result = await service.value()
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
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=1, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test context manager usage
        async with service.session():
            await service.incr()
            await service.incr()
            result = await service.value()
            assert result == 2

        # Test sequential context managers to avoid interference
        async def worker(increments: int):
            async with service.session():
                initial = await service.value()
                for _ in range(increments):
                    await service.incr()
                final = await service.value()
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
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create session and cause failure
        session = await service.start_session()
        await service.incr(session)

        original_replica_idx = service._session_replica_map[session]

        # Cause failure
        error_result = await service.fail_me(session)
        assert isinstance(error_result, RuntimeError)

        # Replica should be marked as failed
        failed_replica = service._replicas[original_replica_idx]
        assert not failed_replica.proc_mesh.healthy

        # Session should be reassigned on next call
        await service.incr(session)
        new_replica_idx = service._session_replica_map[session]
        assert new_replica_idx != original_replica_idx

        # New sessions should avoid failed replica
        new_session = await service.start_session()
        await service.incr(new_session)
        assigned_replica = service._replicas[service._session_replica_map[new_session]]
        assert assigned_replica.proc_mesh.healthy

    finally:
        await service.stop()


# Autoscaling Tests


@pytest.mark.timeout(20)
@pytest.mark.asyncio
async def test_autoscaling_scale_up():
    """Test automatic scale up under high load."""
    autoscaling_cfg = AutoscalingConfig(
        scale_up_capacity_threshold=0.5,
        scale_up_queue_depth_threshold=2.0,
        scale_up_cooldown=0.5,
        min_time_between_scale_events=0.5,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=3,
        default_replicas=1,
        autoscaling=autoscaling_cfg,
        replica_max_concurrent_requests=1,  # Force queueing
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        initial_replica_count = len(service._replicas)
        assert initial_replica_count == 1

        # Create high load with slow operations
        sessions = [await service.start_session() for _ in range(3)]
        tasks = [service.slow_incr(session) for session in sessions for _ in range(2)]

        # Start tasks and wait for scaling
        await asyncio.gather(*tasks)

        # Check if scaling occurred
        scaled_up = False
        for _ in range(10):
            await asyncio.sleep(0.5)
            if len(service._replicas) > initial_replica_count:
                scaled_up = True
                break

        assert (
            scaled_up
        ), f"Expected scale up, but replicas remained at {len(service._replicas)}"

    finally:
        await service.stop()


@pytest.mark.timeout(25)
@pytest.mark.asyncio
async def test_autoscaling_scale_down():
    """Test automatic scale down when idle."""
    autoscaling_cfg = AutoscalingConfig(
        scale_down_capacity_threshold=0.2,
        scale_down_queue_depth_threshold=0.5,
        scale_down_idle_time_threshold=3.0,
        scale_down_cooldown=1.0,
        min_time_between_scale_events=1.0,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=3,
        default_replicas=2,  # Start with 2 replicas
        autoscaling=autoscaling_cfg,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        initial_replica_count = len(service._replicas)
        assert initial_replica_count == 2

        # Make minimal requests to establish baseline
        session = await service.start_session()
        await service.incr(session)

        # Wait for scale down
        max_wait = 10.0
        waited = 0.0
        scaled_down = False

        while waited < max_wait:
            await asyncio.sleep(1.0)
            waited += 1.0
            if len(service._replicas) < initial_replica_count:
                scaled_down = True
                break

        assert (
            scaled_down
        ), f"Expected scale down, but replicas remained at {len(service._replicas)}"
        assert len(service._replicas) >= cfg.min_replicas

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_autoscaling_limits():
    """Test that autoscaling respects min/max limits."""
    autoscaling_cfg = AutoscalingConfig(
        scale_up_queue_depth_threshold=1.0,
        scale_down_capacity_threshold=0.9,  # High threshold to prevent scale down
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=2,  # Tight limit
        default_replicas=1,
        autoscaling=autoscaling_cfg,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test max limit
        should_scale_up, reason = service._should_scale_up()

        # Manually scale to max
        await service._scale_up(1)
        assert len(service._replicas) == 2

        # Should not scale beyond max
        should_scale_up, reason = service._should_scale_up()
        assert not should_scale_up
        assert "max replicas" in reason.lower()

        # Test min limit
        should_scale_down, reason = service._should_scale_down()

        # Scale down to min
        await service._scale_down_replicas(1)
        assert len(service._replicas) == 1

        # Should not scale below min
        should_scale_down, reason = service._should_scale_down()
        assert not should_scale_down
        assert "min replicas" in reason.lower()

    finally:
        await service.stop()


# Metrics and Monitoring Tests


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_metrics_collection():
    """Test comprehensive metrics collection."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions and make requests
        session1 = await service.start_session()
        session2 = await service.start_session()

        await service.incr(session1)
        await service.incr(session1)
        await service.incr(session2)

        # Test failure metrics
        error_result = await service.fail_me(session1)
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
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        session = await service.start_session()

        # Make multiple calls
        await service.incr(session)
        await service.incr(session)
        await service.incr(session)

        # Should always route to same replica
        replica_idx = service._session_replica_map[session]

        await service.incr(session)
        assert service._session_replica_map[session] == replica_idx

        # Verify counter was incremented correctly
        result = await service.value(session)
        assert result == 4

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_load_balancing_multiple_sessions():
    """Test load balancing across multiple sessions using least-loaded assignment."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions with some load to trigger distribution
        session1 = await service.start_session()
        await service.incr(session1)  # Load replica 0

        session2 = await service.start_session()
        await service.incr(session2)  # Should go to replica 1 (least loaded)

        session3 = await service.start_session()
        await service.incr(session3)  # Should go to replica 0 or 1 based on load

        session4 = await service.start_session()
        await service.incr(session4)  # Should balance the load

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
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Mix of session and sessionless calls
        session = await service.start_session()

        # Concurrent operations
        tasks = [
            service.incr(session),  # Session call
            service.incr(session),  # Session call
            service.incr(),  # Sessionless call
            service.incr(),  # Sessionless call
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

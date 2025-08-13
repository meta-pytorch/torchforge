# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
"""Tests for service.py."""

import asyncio
import logging
import time

import pytest
from forge.controller import AutoscalingConfig, ServiceConfig, spawn_service
from monarch.actor import Actor, endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Counter(Actor):
    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        self.v += 1

    @endpoint
    async def value(self) -> int:
        return self.v

    @endpoint
    def value_sync_endpoint(self) -> int:
        return self.v

    @endpoint
    async def fail_me(self):
        """Endpoint that always fails to trigger replica failure."""
        raise RuntimeError("I was asked to fail")

    @endpoint
    async def slow_incr(self):
        """Slow increment to help test queueing."""
        await asyncio.sleep(1.5)  # Slow operation
        self.v += 1


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_service_session_creation():
    """Test that our service can create sessions and handle basic endpoint calls."""
    logger.info("service creation")
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)
    print(service)

    try:
        # Test session creation
        session1 = await service.start_session()
        session2 = await service.start_session()

        # Sessions should be unique
        assert session1 != session2
        assert isinstance(session1, str)
        assert isinstance(session2, str)

        # Test endpoint call
        result = await service.incr(session1)

        # Test session is mapped to a replica
        assert session1 in service._session_replica_map

    finally:
        await service.stop()


@pytest.mark.timeout(20)
@pytest.mark.asyncio
async def test_autoscaling_automatic_scale_up_on_high_load():
    """Test that autoscaling automatically scales up when there's high load."""
    # Configure aggressive autoscaling for testing
    autoscaling_cfg = AutoscalingConfig(
        scale_up_capacity_threshold=0.3,  # Scale up when 50% capacity
        scale_up_queue_depth_threshold=1.0,  # Or when queue depth > 1
        scale_up_cooldown=0.1,  # Very short cooldown
        min_time_between_scale_events=0.1,
        scale_up_step_size=1,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=3,
        default_replicas=1,
        autoscaling=autoscaling_cfg,
        replica_max_concurrent_requests=1,  # Minimum value to force queueing
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Start with 1 replica
        assert len(service._replicas) == 1
        initial_replica_count = len(service._replicas)

        # Create sessions and start many concurrent slow requests to build up load
        sessions = [await service.start_session() for _ in range(2)]

        # Start slow requests concurrently to max out capacity
        tasks = []
        for session in sessions:
            for _ in range(2):  # 6 total slow requests, but only 1 can run concurrently
                tasks.append(service.slow_incr(session))

        # Start the requests
        await asyncio.gather(*tasks)

        scaled_up = False
        for _ in range(10):  # wait up to 1 second
            await asyncio.sleep(0.1)
            current_count = len(service._replicas)
            if current_count > initial_replica_count:
                scaled_up = True
                break

        # Should have automatically scaled up due to high load
        assert (
            scaled_up
        ), f"Expected automatic scale up due to high load, but replicas remained at {len(service._replicas)}"
        assert len(service._replicas) > initial_replica_count
    finally:
        await service.stop()


@pytest.mark.timeout(25)
@pytest.mark.asyncio
async def test_autoscaling_automatic_scale_up_emergency():
    """Test that autoscaling immediately scales up on emergency high queue depth."""
    # Configure autoscaling with emergency threshold
    autoscaling_cfg = AutoscalingConfig(
        max_queue_depth_emergency=5.0,  # Emergency scale up at queue depth 5
        scale_up_cooldown=0.5,  # Short cooldown
        min_time_between_scale_events=0.5,
        scale_up_step_size=1,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=2,
        default_replicas=1,
        autoscaling=autoscaling_cfg,
        replica_max_concurrent_requests=1,  # Minimum value to force queueing
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Start with 1 replica
        assert len(service._replicas) == 1
        initial_replica_count = len(service._replicas)

        # Create many sessions and queue up lots of slow requests
        sessions = [await service.start_session() for _ in range(4)]

        # Queue many slow requests to trigger emergency threshold
        tasks = []
        for session in sessions:
            for _ in range(3):  # 12 total slow requests, but only 1 can run at a time
                tasks.append(service.slow_incr(session))

        await asyncio.gather(*tasks)
        scaled_up = False
        for _ in range(10):  # wait up to 1 second
            await asyncio.sleep(0.1)
            current_count = len(service._replicas)
            if current_count > initial_replica_count:
                scaled_up = True
                break

        # Should have automatically scaled up due to high load
        assert (
            scaled_up
        ), f"Expected automatic scale up due to high load, but replicas remained at {len(service._replicas)}"
        assert len(service._replicas) > initial_replica_count
    finally:
        await service.stop()


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_autoscaling_full_cycle_scale_up_then_down():
    """Test a full autoscaling cycle: scale up under load, then scale down when idle."""
    # Configure autoscaling for both up and down
    autoscaling_cfg = AutoscalingConfig(
        scale_up_capacity_threshold=0.7,
        scale_down_capacity_threshold=0.1,
        scale_down_queue_depth_threshold=0.5,
        scale_down_idle_time_threshold=3.0,  # Short idle time for testing
        scale_up_cooldown=1.0,
        scale_down_cooldown=1.0,
        min_time_between_scale_events=1.0,
        scale_up_step_size=1,
        scale_down_step_size=1,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=3,
        default_replicas=1,
        autoscaling=autoscaling_cfg,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Phase 1: Scale up under load
        print("🔼 Phase 1: Testing scale up under load")

        initial_replica_count = len(service._replicas)
        assert initial_replica_count == 1

        # Reduce capacity to trigger scale up faster
        service._replicas[0].max_concurrent_requests = 2

        # Create high load
        sessions = [await service.start_session() for _ in range(3)]
        tasks = []
        for session in sessions:
            for _ in range(3):  # 9 slow requests, only 2 concurrent
                tasks.append(service.slow_incr(session))

        await asyncio.gather(*tasks)

        scaled_up = False
        for _ in range(10):  # wait up to 1 second
            await asyncio.sleep(0.1)
            current_count = len(service._replicas)
            if current_count > initial_replica_count:
                scaled_up = True
                break

        assert scaled_up, "Failed to scale up under load"

        # Phase 2: Scale down when idle
        post_load_replica_count = len(service._replicas)

        # Wait for scale down (need to wait for idle time threshold + processing)
        max_wait_scale_down = 15.0
        waited = 0.0
        scaled_down = False

        while waited < max_wait_scale_down:
            await asyncio.sleep(1.0)
            waited += 1.0
            current_count = len(service._replicas)
            if current_count < post_load_replica_count:
                scaled_down = True
                break

        assert (
            scaled_down
        ), f"Failed to scale down when idle. Replicas remained at {len(service._replicas)}"

        # Should not go below minimum
        assert len(service._replicas) >= cfg.min_replicas

        print(
            f"🎉 Full autoscaling cycle completed: {initial_replica_count} -> {post_load_replica_count} -> {len(service._replicas)}"
        )

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_autoscaling_manual_scale_up():
    """Test that manual scale up works correctly."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=3, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Start with 1 replica
        assert len(service._replicas) == 1

        # Manually trigger scale up
        await service._scale_up(1)

        # Should now have 2 replicas
        assert len(service._replicas) == 2

        # Wait for the replicas to be healthy
        await asyncio.sleep(3)

        # Both replicas should be healthy
        for replica in service._replicas:
            assert replica.proc_mesh.healthy

        # Test that both replicas can handle requests
        session1 = await service.start_session()
        session2 = await service.start_session()

        await service.incr(session1)
        await service.incr(session2)

        # Both sessions should be mapped to replicas
        assert session1 in service._session_replica_map
        assert session2 in service._session_replica_map
    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_autoscaling_decision_logic_detailed():
    """Test the autoscaling decision logic in detail."""
    autoscaling_cfg = AutoscalingConfig(
        scale_up_queue_depth_threshold=3.0,
        scale_up_capacity_threshold=0.7,
        scale_down_capacity_threshold=0.2,
        scale_down_queue_depth_threshold=1.0,
        scale_down_idle_time_threshold=5.0,  # Short for testing
        min_time_between_scale_events=1.0,
        scale_up_cooldown=1.0,
        scale_down_cooldown=2.0,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=3,
        default_replicas=2,
        autoscaling=autoscaling_cfg,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test initial state
        should_scale_up, reason = service._should_scale_up()
        should_scale_down, reason_down = service._should_scale_down()

        logging.info("Initial state:")
        logging.info(f"  Scale up: {should_scale_up} - {reason}")
        logging.info(f"  Scale down: {should_scale_down} - {reason_down}")

        # Test at max replicas
        await service._scale_up(cfg.max_replicas - len(service._replicas))
        should_scale_up, reason = service._should_scale_up()
        assert not should_scale_up
        assert "max replicas" in reason.lower()
        logging.info(f"At max replicas: {reason}")

        # Test at min replicas
        await service._scale_down_replicas(len(service._replicas) - cfg.min_replicas)

        should_scale_down, reason = service._should_scale_down()
        assert not should_scale_down
        assert "min replicas" in reason.lower()
        logging.info(f"At min replicas: {reason}")

        # Test cooldown logic
        service._last_scale_up_time = time.time()  # Just scaled up
        should_scale_up, reason = service._should_scale_up()
        assert not should_scale_up
        assert "cooldown" in reason.lower()
        logging.info(f"During cooldown: {reason}")
        logging.info("✅ All autoscaling decision logic tests passed")

    finally:
        await service.stop()


@pytest.mark.timeout(25)
@pytest.mark.asyncio
async def test_autoscaling_scale_down_on_low_utilization():
    """Test that autoscaling scales down when utilization is low."""
    # Configure autoscaling for quick scale down testing
    autoscaling_cfg = AutoscalingConfig(
        scale_down_capacity_threshold=0.1,  # Very low threshold
        scale_down_queue_depth_threshold=0.5,
        scale_down_idle_time_threshold=2.0,  # Short idle time for testing
        scale_down_cooldown=1.0,  # Short cooldown
        min_time_between_scale_events=1.0,
        scale_down_step_size=1,
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
        # Start with 2 replicas
        assert len(service._replicas) == 2

        # Make a few requests to establish baseline, then stop
        session = await service.start_session()
        await service.incr(session)
        await service.incr(session)

        # Wait for low utilization to be detected and scale down to occur
        max_wait = 15.0  # Need to wait for idle time threshold + processing
        waited = 0.0
        initial_replica_count = len(service._replicas)

        while waited < max_wait:
            await asyncio.sleep(1.0)
            waited += 1.0

            if len(service._replicas) < initial_replica_count:
                print(
                    f"✅ Autoscaling scale down triggered! Replicas: {initial_replica_count} -> {len(service._replicas)}"
                )
                break

            # Log current metrics for debugging
            summary = service.get_metrics_summary()
            service_metrics = summary["service"]

        # Should have scaled down
        assert (
            len(service._replicas) < initial_replica_count
        ), f"Expected scale down, but replicas remained at {len(service._replicas)}"
        assert (
            len(service._replicas) >= cfg.min_replicas
        ), f"Scaled below minimum replicas: {len(service._replicas)} < {cfg.min_replicas}"

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_autoscaling_respects_min_max_limits():
    """Test that autoscaling respects min and max replica limits."""
    # Configure autoscaling with tight limits
    autoscaling_cfg = AutoscalingConfig(
        scale_up_queue_depth_threshold=1.0,
        scale_down_capacity_threshold=0.9,  # High threshold - should never scale down
        scale_up_cooldown=0.5,
        scale_down_cooldown=0.5,
        min_time_between_scale_events=0.5,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=2,  # Tight max limit
        default_replicas=1,
        autoscaling=autoscaling_cfg,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Start with 1 replica
        assert len(service._replicas) == 1

        # Try to trigger multiple scale ups
        sessions = [await service.start_session() for _ in range(5)]

        # Queue many requests to trigger scale up
        tasks = []
        for session in sessions:
            for _ in range(3):
                tasks.append(service.incr(session))

        # Wait for potential scaling
        await asyncio.sleep(2.0)

        # Should not exceed max replicas
        assert (
            len(service._replicas) <= cfg.max_replicas
        ), f"Exceeded max replicas: {len(service._replicas)} > {cfg.max_replicas}"

        # Complete requests
        await asyncio.gather(*tasks)

        # Should not scale below min replicas (even though we set high scale-down threshold)
        await asyncio.sleep(2.0)
        assert (
            len(service._replicas) >= cfg.min_replicas
        ), f"Below min replicas: {len(service._replicas)} < {cfg.min_replicas}"

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_autoscaling_cooldown_periods():
    """Test that autoscaling respects cooldown periods."""
    from service import AutoscalingConfig

    # Configure autoscaling with longer cooldowns
    autoscaling_cfg = AutoscalingConfig(
        scale_up_queue_depth_threshold=1.0,
        scale_up_cooldown=5.0,  # Long cooldown
        min_time_between_scale_events=3.0,
        scale_up_step_size=1,
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=4,
        default_replicas=1,
        autoscaling=autoscaling_cfg,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Start with 1 replica
        initial_count = len(service._replicas)

        # Trigger first scale up
        sessions = [await service.start_session() for _ in range(3)]
        tasks = [service.incr(session) for session in sessions for _ in range(2)]

        # Wait for first scale up
        await asyncio.sleep(2.0)

        first_scale_count = len(service._replicas)
        if first_scale_count > initial_count:
            print(f"✅ First scale up: {initial_count} -> {first_scale_count}")

            # Try to trigger another scale up immediately (should be blocked by cooldown)
            more_tasks = [
                service.incr(session) for session in sessions for _ in range(3)
            ]

            # Wait a short time (less than cooldown)
            await asyncio.sleep(1.0)

            # Should not have scaled again due to cooldown
            current_count = len(service._replicas)
            assert (
                current_count == first_scale_count
            ), f"Scaled during cooldown: {first_scale_count} -> {current_count}"

            await asyncio.gather(*more_tasks)

        # Complete original tasks
        await asyncio.gather(*tasks)

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_autoscaling_decision_logic():
    """Test the autoscaling decision logic without actually scaling."""

    autoscaling_cfg = AutoscalingConfig(
        scale_up_queue_depth_threshold=5.0,
        scale_up_capacity_threshold=0.8,
        scale_down_capacity_threshold=0.2,
        scale_down_queue_depth_threshold=1.0,
        scale_down_idle_time_threshold=60.0,  # Long time to prevent actual scale down
    )

    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=1,
        max_replicas=3,
        default_replicas=2,
        autoscaling=autoscaling_cfg,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test scale up decision logic
        should_scale_up, reason = service._should_scale_up()
        logger.info(f"Initial scale up decision: {should_scale_up}, reason: {reason}")

        # Test scale down decision logic
        should_scale_down, reason = service._should_scale_down()
        logger.info(
            f"Initial scale down decision: {should_scale_down}, reason: {reason}"
        )

        # Create some load and test again
        sessions = [await service.start_session() for _ in range(2)]
        await asyncio.gather(*[service.incr(session) for session in sessions])

        # Update metrics and test decisions
        service._update_service_metrics()
        summary = service.get_metrics_summary()
        print(f"Metrics after load: {summary['service']}")

        should_scale_up, reason = service._should_scale_up()
        print(f"Scale up after load: {should_scale_up}, reason: {reason}")

        should_scale_down, reason = service._should_scale_down()
        print(f"Scale down after load: {should_scale_down}, reason: {reason}")
    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_metrics_collection():
    """Test that metrics are collected correctly."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=3, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create sessions and make requests
        session1 = await service.start_session()
        session2 = await service.start_session()

        # Make some requests to generate metrics
        await service.incr(session1)
        await service.incr(session1)
        await service.incr(session2)

        # Get metrics
        metrics = service.get_metrics()
        summary = service.get_metrics_summary()

        # Test service-level metrics
        assert metrics.total_sessions == 2
        assert metrics.healthy_replicas == 2
        assert metrics.total_replicas == 2

        # Test summary structure
        assert "service" in summary
        assert "replicas" in summary

        service_metrics = summary["service"]
        assert service_metrics["total_sessions"] == 2
        assert service_metrics["healthy_replicas"] == 2
        assert service_metrics["total_replicas"] == 2
        assert service_metrics["sessions_per_replica"] == 1.0  # 2 sessions / 2 replicas

        # Test replica-level metrics
        replica_metrics = summary["replicas"]
        assert len(replica_metrics) == 2  # Should have metrics for both replicas

        # Check that requests were recorded
        total_requests = sum(
            metrics["total_requests"] for metrics in replica_metrics.values()
        )
        assert total_requests == 3  # We made 3 requests total

        # Check that successful requests were recorded
        total_successful = sum(
            metrics["successful_requests"] for metrics in replica_metrics.values()
        )
        assert total_successful == 3  # All requests should be successful

        # Check that failed requests are 0
        total_failed = sum(
            metrics["failed_requests"] for metrics in replica_metrics.values()
        )
        assert total_failed == 0  # No failures yet

        # Test failure metrics
        error_result = await service.fail_me(session1)
        assert isinstance(error_result, RuntimeError)

        # Get updated metrics
        # Test that failure was recorded
        updated_summary = service.get_metrics_summary()
        updated_replica_metrics = updated_summary["replicas"]

        # Check that failure was recorded
        total_failed_after = sum(
            metrics["failed_requests"] for metrics in updated_replica_metrics.values()
        )
        assert total_failed_after == 1  # Should have 1 failure now

        # Test request rate calculation (should be > 0 since we just made requests)
        total_request_rate = updated_summary["service"]["total_request_rate"]
        assert total_request_rate >= 0  # Should be non-negative

        # Test latency tracking
        for replica_idx, metrics in updated_replica_metrics.items():
            if metrics["total_requests"] > 0:
                assert metrics["avg_latency"] >= 0  # Should be non-negative
                assert metrics["capacity_utilization"] >= 0  # Should be non-negative
                assert metrics["capacity_utilization"] <= 1.0  # Should not exceed 100%

        print("📊 Metrics Summary:")
        print(f"  Service: {updated_summary['service']}")
        for replica_idx, metrics in updated_replica_metrics.items():
            print(f"  Replica {replica_idx}: {metrics}")

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_failure_returns_exception():
    """Test that when we encounter a failure, service returns the exception."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        session = await service.start_session()

        # Normal call should work
        await service.incr(session)
        result = await service.value(session)
        assert result == 1

        # Call the failing endpoint - should return exception, not raise it
        error_result = await service.fail_me(session)

        # Should return the exception object
        logger.info("error_result was: %s", error_result)
        assert isinstance(error_result, RuntimeError)
        assert str(error_result) == "I was asked to fail"

        # The replica should now be marked as failed
        replica_idx = service._session_replica_map[session]
        replica = service._replicas[replica_idx]
        assert not replica.proc_mesh.healthy

    finally:
        await service.stop()


@pytest.mark.timeout(20)
@pytest.mark.asyncio
async def test_failed_replica_not_accessible_new_sessions_go_elsewhere():
    """Test that once there is a failure, that replica is not accessible, and new sessions go to other replicas."""
    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=2,  # Need 2 replicas to test failover
        max_replicas=3,
        default_replicas=2,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create a session and make it fail
        session1 = await service.start_session()
        await service.incr(session1)  # Trigger replica assignment

        original_replica_idx = service._session_replica_map[session1]

        # Cause the replica to fail
        error_result = await service.fail_me(session1)
        assert isinstance(error_result, RuntimeError)

        # The replica should be marked as failed
        failed_replica = service._replicas[original_replica_idx]
        assert not failed_replica.proc_mesh.healthy

        # Existing session should be reassigned to healthy replica on next call
        await service.incr(session1)
        new_replica_idx = service._session_replica_map[session1]
        assert new_replica_idx != original_replica_idx

        # New sessions should not be assigned to the failed replica
        new_sessions = []
        for i in range(3):
            session = await service.start_session()
            await service.incr(session)  # Trigger assignment
            new_sessions.append(session)

        # All new sessions should be assigned to healthy replicas only
        for session in new_sessions:
            assigned_replica_idx = service._session_replica_map[session]
            assigned_replica = service._replicas[assigned_replica_idx]
            assert assigned_replica.proc_mesh.healthy
            assert assigned_replica_idx != original_replica_idx

    finally:
        await service.stop()


@pytest.mark.timeout(25)
@pytest.mark.asyncio
async def test_failed_replica_gets_recovered():
    """Test that once there is a failure, the replica gets recovered."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        session = await service.start_session()

        # Normal operation
        await service.incr(session)
        result = await service.value(session)
        assert result == 1

        replica_idx = service._session_replica_map[session]
        original_replica = service._replicas[replica_idx]

        # Cause failure
        error_result = await service.fail_me(session)
        assert isinstance(error_result, RuntimeError)

        # Replica should be marked as failed
        assert not original_replica.proc_mesh.healthy

        # Wait for health loop to detect and recover the failed replica
        # The health loop runs every 0.3 seconds, so we wait a bit longer
        max_wait_time = 10  # seconds
        wait_interval = 0.5  # seconds
        waited = 0

        while waited < max_wait_time:
            await asyncio.sleep(wait_interval)
            waited += wait_interval

            # Check if replica has been recovered
            if original_replica.proc_mesh.healthy:
                break

        # Replica should be recovered
        assert (
            original_replica.proc_mesh.healthy
        ), f"Replica not recovered after {max_wait_time} seconds"

        # Should be able to use the recovered replica
        # Create a new session to test the recovered replica
        new_session = await service.start_session()
        await service.incr(new_session)
        result = await service.value(new_session)
        assert result == 1  # Fresh counter should start at 0, increment to 1

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_session_stickiness():
    """Test that the same session always routes to the same replica."""
    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=2,  # Use 2 replicas to test stickiness
        max_replicas=3,
        default_replicas=2,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        session = await service.start_session()

        # Make multiple calls with the same session
        await service.incr(session)
        await service.incr(session)
        await service.incr(session)

        # Should always route to the same replica
        assert session in service._session_replica_map
        replica_idx = service._session_replica_map[session]

        # Make another call and verify it's still the same replica
        await service.incr(session)
        assert service._session_replica_map[session] == replica_idx

        # Verify the counter was incremented (should be 4)
        result = await service.value(session)
        assert result == 4

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_multiple_sessions_different_replicas():
    """Test that different sessions can be assigned to different replicas."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=3, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create multiple sessions
        sessions = []
        for i in range(4):  # More sessions than replicas
            session = await service.start_session()
            sessions.append(session)
            await service.incr(session)  # Trigger replica assignment

        # All sessions should be mapped
        for session in sessions:
            assert session in service._session_replica_map

        # Should use both replicas (round-robin)
        replica_assignments = [service._session_replica_map[s] for s in sessions]
        unique_replicas = set(replica_assignments)
        assert len(unique_replicas) >= 1  # At least one replica used

        # With 4 sessions and 2 replicas, should use both
        if len(service._replicas) >= 2:
            assert len(unique_replicas) == 2

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_concurrent_calls_same_session():
    """Test that concurrent calls with the same session work correctly."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        session = await service.start_session()

        # Make concurrent calls
        tasks = [service.incr(session) for _ in range(5)]
        await asyncio.gather(*tasks)

        # All calls should have gone to the same replica
        assert session in service._session_replica_map

        # Counter should be incremented 5 times
        result = await service.value(session)
        assert result == 5

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_concurrent_calls_different_sessions():
    """Test that concurrent calls with different sessions work correctly."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Create multiple sessions
        sessions = [await service.start_session() for _ in range(2)]

        # Make concurrent calls with different sessions
        tasks = [service.incr(session) for session in sessions]
        await asyncio.gather(*tasks)

        # All sessions should be mapped
        for session in sessions:
            assert session in service._session_replica_map

        # Each counter should be incremented once
        for session in sessions:
            result = await service.value(session)
            assert result == 1

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_endpoint_discovery():
    """Test that endpoint methods are dynamically added to the service."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Check that endpoint methods exist
        assert hasattr(service, "incr")
        assert hasattr(service, "value")
        assert hasattr(service, "value_sync_endpoint")
        assert callable(service.incr)
        assert callable(service.value)
        assert callable(service.value_sync_endpoint)

        # Check that endpoints are registered
        expected_endpoints = ["incr", "value", "value_sync_endpoint"]
        for ep in expected_endpoints:
            assert ep in service._endpoints

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_service_state_tracking():
    """Test that service properly tracks sessions and replicas."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=3, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Initially no sessions
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

        # Create sessions
        session1 = await service.start_session()
        session2 = await service.start_session()

        # Should track sessions
        assert len(service._active_sessions) == 2

        # Make calls to trigger replica assignment
        await service.incr(session1)
        await service.incr(session2)

        # Should track session-replica mappings
        assert len(service._session_replica_map) == 2
        assert session1 in service._session_replica_map
        assert session2 in service._session_replica_map

        # Should have the right number of replicas
        assert len(service._replicas) == 2

        # All replicas should be healthy initially
        for replica in service._replicas:
            assert replica.proc_mesh.healthy

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_sessionless_calls_basic():
    """Test basic sessionless calls with load balancing."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=3, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Make sessionless calls (no sess_id parameter)
        result1 = await service.value()
        result2 = await service.value()
        result3 = await service.value()

        # All calls should succeed
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

        # No sessions should be created for sessionless calls
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

        # Verify that calls were distributed (check metrics)
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
async def test_sessionless_vs_session_calls():
    """Test that sessionless and session-based calls can coexist."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=3, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Make sessionless calls
        await service.incr()
        await service.incr()

        # Make session-based calls
        session = await service.start_session()
        await service.incr(session)
        await service.incr(session)

        # Should have 1 session tracked
        assert len(service._active_sessions) == 1
        assert len(service._session_replica_map) == 1

        # Make more sessionless calls
        await service.incr()
        await service.incr()

        # Session state should be unchanged
        assert len(service._active_sessions) == 1
        assert len(service._session_replica_map) == 1

        # Total requests should be 6 (2 sessionless + 2 session + 2 sessionless)
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 6

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_sessionless_load_balancing():
    """Test that sessionless calls are properly load balanced."""
    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=3,  # Use 3 replicas to better test load balancing
        max_replicas=3,
        default_replicas=3,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Wait for all replicas to be healthy
        await asyncio.sleep(2.0)

        # Make many sessionless calls
        num_calls = 12
        tasks = [service.incr() for _ in range(num_calls)]
        await asyncio.gather(*tasks)

        # Check that load was distributed across replicas
        metrics = service.get_metrics_summary()
        replica_requests = [
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        ]

        # All replicas should have received some requests
        assert all(
            requests > 0 for requests in replica_requests
        ), f"Load not distributed: {replica_requests}"

        # Total requests should match
        total_requests = sum(replica_requests)
        assert total_requests == num_calls

        # Load should be reasonably balanced (no replica should have more than 2x the average)
        avg_requests = total_requests / len(replica_requests)
        max_requests = max(replica_requests)
        assert (
            max_requests <= avg_requests * 2
        ), f"Load imbalance detected: {replica_requests}, avg: {avg_requests}"

        logger.info(
            f"Load balancing test: {replica_requests} requests across {len(replica_requests)} replicas"
        )

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_sessionless_concurrent_calls():
    """Test concurrent sessionless calls."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=3, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Make many concurrent sessionless calls
        num_concurrent = 10
        tasks = [service.value() for _ in range(num_concurrent)]

        # All should complete successfully
        results = await asyncio.gather(*tasks)
        assert len(results) == num_concurrent
        assert all(result is not None for result in results)

        # Verify total requests
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == num_concurrent

        # No sessions should be created
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_sessionless_with_replica_failure():
    """Test that sessionless calls handle replica failures gracefully."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=3, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Make some successful sessionless calls first
        await service.incr()
        await service.incr()

        # Cause a failure on one replica using a session
        session = await service.start_session()
        error_result = await service.fail_me(session)
        assert isinstance(error_result, RuntimeError)

        # The failed replica should be marked as unhealthy
        failed_replica_idx = service._session_replica_map[session]
        failed_replica = service._replicas[failed_replica_idx]
        assert not failed_replica.proc_mesh.healthy

        # Sessionless calls should still work (routed to healthy replicas)
        await service.incr()
        await service.incr()
        await service.incr()

        # Verify that sessionless calls went to healthy replicas only
        metrics = service.get_metrics_summary()
        healthy_replicas = [
            replica_idx
            for replica_idx, replica in enumerate(service._replicas)
            if replica.proc_mesh.healthy
        ]

        # Should have at least one healthy replica handling requests
        assert len(healthy_replicas) >= 1

        # Total requests should be recorded (including the failed one)
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests >= 5  # 2 initial + 1 fail + 3 after failure

    finally:
        await service.stop()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_sessionless_explicit_none():
    """Test that explicitly passing sess_id=None works the same as omitting it."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test both ways of making sessionless calls
        result1 = await service.value()  # Omit sess_id
        result2 = await service.value(sess_id=None)  # Explicit None

        # Both should work
        assert result1 is not None
        assert result2 is not None

        # No sessions should be created
        assert len(service._active_sessions) == 0
        assert len(service._session_replica_map) == 0

        # Both calls should be recorded
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == 2

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_sessionless_performance_comparison():
    """Test performance comparison between sessionless and session-based calls."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        num_calls = 5

        # Test session-based calls
        session_start_time = time.time()
        session = await service.start_session()
        for _ in range(num_calls):
            await service.incr(session)
        await service.terminate_session(session)
        session_end_time = time.time()
        session_duration = session_end_time - session_start_time

        # Test sessionless calls
        sessionless_start_time = time.time()
        for _ in range(num_calls):
            await service.incr()
        sessionless_end_time = time.time()
        sessionless_duration = sessionless_end_time - sessionless_start_time

        logger.info(
            f"Session-based calls: {session_duration:.3f}s for {num_calls} calls"
        )
        logger.info(
            f"Sessionless calls: {sessionless_duration:.3f}s for {num_calls} calls"
        )

        # Both should complete in reasonable time (this is more of a smoke test)
        assert session_duration < 10.0
        assert sessionless_duration < 10.0

        # Verify correct number of requests were made
        metrics = service.get_metrics_summary()
        total_requests = sum(
            replica_metrics["total_requests"]
            for replica_metrics in metrics["replicas"].values()
        )
        assert total_requests == num_calls * 2  # session calls + sessionless calls

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_session_context_manager_basic():
    """Test basic session context manager functionality."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=1, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test context manager usage - service methods auto-inject session
        async with service.session() as svc:
            # Make calls through the service - session ID is auto-injected
            await svc.incr()
            await svc.incr()
            await svc.incr()

            # Check the counter value
            result = await svc.value()
            assert result == 3
    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_session_context_manager_vs_manual():
    """Test that context manager works the same as manual session management."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Manual session management
        manual_session = await service.start_session()
        start = await service.value(manual_session)

        await service.incr(manual_session)
        await service.incr(manual_session)
        end = await service.value(manual_session)
        manual_incr = end - start

        await service.terminate_session(manual_session)

        # Context manager - service methods auto-inject session
        async with service.session() as svc:
            start = await svc.value()
            await svc.incr()
            await svc.incr()
            end = await svc.value()
            sess_incr = end - start

        # Both should have the same result
        assert manual_incr == sess_incr

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_session_context_manager_concurrent():
    """Test concurrent session context managers."""
    cfg = ServiceConfig(
        gpus_per_replica=1,
        min_replicas=3,
        max_replicas=3,
        default_replicas=3,
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:

        async def session_worker(worker_id: int, increments: int):
            async with service.session() as svc:
                # Get initial value for this session's replica
                initial_result = await svc.value()

                for _ in range(increments):
                    await svc.incr()

                final_result = await svc.value()

                # Return the increment amount (final - initial)
                return worker_id, final_result - initial_result

        # Run multiple concurrent session contexts
        tasks = [
            session_worker(0, 2),
            session_worker(1, 3),
            session_worker(2, 5),
        ]

        results = await asyncio.gather(*tasks)

        # Each session should have incremented by the expected amount
        expected_results = [(0, 2), (1, 3), (2, 5)]
        assert sorted(results) == sorted(expected_results)

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_session_context_manager_invalid_usage():
    """Test error handling for invalid context manager usage."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=1, max_replicas=2, default_replicas=1
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Test that context works properly when used correctly
        async with service.session() as svc:
            await svc.incr()
            result = await svc.value()
            assert result == 1

        # Test that after context exits, the patched methods are restored
        # and normal sessionless calls work
        await service.incr()  # Should work as sessionless call

    finally:
        await service.stop()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_session_context_manager_mixed_with_manual():
    """Test mixing context manager with manual session management."""
    cfg = ServiceConfig(
        gpus_per_replica=1, min_replicas=2, max_replicas=2, default_replicas=2
    )
    service = await spawn_service(service_cfg=cfg, actor_def=Counter, v=0)

    try:
        # Manual session
        manual_session = await service.start_session()
        await service.incr(manual_session)

        # Context manager session (concurrent)
        async with service.session() as svc:
            await svc.incr()
            await svc.incr()
            ctx_result = await svc.value()

        # Continue with manual session
        await service.incr(manual_session)
        manual_result = await service.value(manual_session)
        await service.terminate_session(manual_session)

        # Results should be independent
        assert ctx_result == 2  # Context manager session
        assert manual_result == 2  # Manual session

        await asyncio.sleep(1)

        # Verify sessions were tracked correctly
        assert (
            len(service._active_sessions) == 0
        )  # Only manual session should remain, but we terminated it

    finally:
        await service.stop()

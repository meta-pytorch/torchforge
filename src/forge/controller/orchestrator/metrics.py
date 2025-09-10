# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Service metrics collection and aggregation.

This module provides comprehensive metrics tracking for distributed services,
including per-replica performance data, service-wide aggregations, and
health status information.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from forge.controller.orchestrator.replica import ReplicaMetrics


# TODO - tie this into metrics logger when it exists.
@dataclass
class ServiceMetrics:
    """
    Aggregated metrics collection for the entire service.

    Provides service-wide visibility into performance, health, and scaling metrics
    by aggregating data from all replica instances.

    Attributes:
        replica_metrics: Per-replica metrics indexed by replica ID
        total_sessions: Number of active sessions across all replicas
        healthy_replicas: Number of currently healthy replicas
        total_replicas: Total number of replicas (healthy + unhealthy)
        last_scale_event: Timestamp of the last scaling operation
    """

    # Replica metrics
    replica_metrics: Dict[int, ReplicaMetrics] = field(default_factory=dict)
    # Service-level metrics
    total_sessions: int = 0
    healthy_replicas: int = 0
    total_replicas: int = 0
    # Time-based metrics
    last_scale_event: float = 0.0

    def get_total_request_rate(self, window_seconds: float = 60.0) -> float:
        """Get total requests per second across all replicas."""
        return sum(
            metrics.get_request_rate(window_seconds)
            for metrics in self.replica_metrics.values()
        )

    def get_avg_queue_depth(self, replicas: List) -> float:
        """Get average queue depth across all healthy replicas."""
        healthy_replicas = [r for r in replicas if r.healthy]
        if not healthy_replicas:
            return 0.0
        total_queue_depth = sum(r.request_queue.qsize() for r in healthy_replicas)
        return total_queue_depth / len(healthy_replicas)

    def get_avg_capacity_utilization(self, replicas: List) -> float:
        """Get average capacity utilization across all healthy replicas."""
        healthy_replicas = [r for r in replicas if r.healthy]
        if not healthy_replicas:
            return 0.0
        total_utilization = sum(r.capacity_utilization for r in healthy_replicas)
        return total_utilization / len(healthy_replicas)

    def get_sessions_per_replica(self) -> float:
        """Get average sessions per replica."""
        if self.total_replicas == 0:
            return 0.0
        return self.total_sessions / self.total_replicas

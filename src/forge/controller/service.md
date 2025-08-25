# Service - Distributed Actor Service Controller

A robust service orchestration system for managing distributed actor-based workloads with fault tolerance and intelligent load balancing.

## Overview

The Service class provides a unified interface for deploying and managing multiple replicas of actor-based services across distributed compute resources. It automatically handles replica lifecycle, request routing, and session management.

## Key Features

### **Fault Tolerance**
- **Health Monitoring**: Continuous health checks with automatic replica recovery
- **Request Migration**: Seamless migration of requests from failed replicas
- **Session Preservation**: Maintains session state during replica failures
- **Graceful Degradation**: Continues operation with reduced capacity

### **Load Balancing**
- **Round-Robin**: Default load distribution across healthy replicas
- **Least-Loaded**: Session assignment to replicas with lowest load
- **Session Affinity**: Sticky sessions for stateful workloads
- **Custom Routing**: Extensible routing logic for specialized use cases

### **Comprehensive Metrics**
- **Request Metrics**: Throughput, latency, success/failure rates
- **Capacity Metrics**: Utilization, queue depth, active requests
- **Service Metrics**: Session counts, replica health, scaling events
- **Real-time Monitoring**: Sliding window metrics for responsive scaling

### **Session Management**
- **Context-Aware Sessions**: Automatic session context propagation
- **Session Lifecycle**: Managed session creation and cleanup
- **Routing Hints**: Custom session routing based on workload characteristics

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client API    │───▶│  Service Layer   │───▶│  Replica Pool   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────┐         ┌─────────────┐
                       │ Autoscaler   │         │ Actor Mesh  │
                       └──────────────┘         └─────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────┐         ┌─────────────┐
                       │   Metrics    │         │   Health    │
                       │  Collector   │         │  Monitor    │
                       └──────────────┘         └─────────────┘
```

## Usage

### Basic Service Setup

```python
from forge.controller.service import Service, ServiceConfig

# Configure service parameters
config = ServiceConfig(
    gpus_per_replica=1,
    min_replicas=2,
    max_replicas=10,
    default_replicas=3,
    replica_max_concurrent_requests=10,
)

# Create service with your actor definition
service = Service(config, MyActorClass, *actor_args, **actor_kwargs)
await service.__initialize__()
```

### Session-Based Calls

```python
# Context manager for session lifecycle
async with service.session() as session:
    result1 = await service.my_endpoint(arg1, arg2)
    result2 = await service.another_endpoint(arg3)
    # Session automatically terminated on exit

# Manual session management
session_id = await service.start_session()
result = await service.my_endpoint(session_id, arg1, arg2)
await service.terminate_session(session_id)
```

### Stateless Calls

```python
# Direct calls without sessions (uses round-robin load balancing)
result = await service.my_endpoint(arg1, arg2)
```

### Custom Routing

```python
# Override _custom_replica_routing for specialized routing logic
class CustomService(Service):
    async def _custom_replica_routing(self, sess_id: str | None, **kwargs) -> Optional[Replica]:
        # Custom routing based on request characteristics
        if kwargs.get('priority') == 'high':
            return self._get_least_loaded_replica()
        return None  # Fall back to default routing

# Use with routing hints
async with service.session(priority='high') as session:
    result = await service.my_endpoint(arg1, arg2)
```

### Monitoring and Metrics

```python
# Get detailed metrics
metrics = service.get_metrics()
print(f"Total request rate: {metrics.get_total_request_rate()}")
print(f"Average queue depth: {metrics.get_avg_queue_depth()}")
print(f"Capacity utilization: {metrics.get_avg_capacity_utilization(service._replicas)}")

# Get summary for monitoring dashboards
summary = service.get_metrics_summary()
print(f"Healthy replicas: {summary['service']['healthy_replicas']}")
print(f"Total sessions: {summary['service']['total_sessions']}")

# Per-replica metrics
for replica_idx, replica_metrics in summary['replicas'].items():
    print(f"Replica {replica_idx}: {replica_metrics['request_rate']:.1f} req/s")
```

### Graceful Shutdown

```python
# Stop the service and all replicas
await service.stop()
```

## Configuration

### ServiceConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpus_per_replica` | int | Number of GPUs allocated per replica |
| `min_replicas` | int | Minimum number of replicas to maintain |
| `max_replicas` | int | Maximum number of replicas allowed |
| `default_replicas` | int | Initial number of replicas to start |
| `replica_max_concurrent_requests` | int | Maximum concurrent requests per replica |
| `health_poll_rate` | float | Health check frequency in seconds |
| `return_first_rank_result` | bool | Auto-unwrap ValueMesh to first rank's result |
| `autoscaling` | AutoscalingConfig | Autoscaling configuration |

### AutoscalingConfig

#### Scale Up Triggers
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_up_queue_depth_threshold` | 5.0 | Average queue depth to trigger scale up |
| `scale_up_capacity_threshold` | 0.8 | Capacity utilization to trigger scale up |
| `scale_up_request_rate_threshold` | 10.0 | Requests/sec to trigger scale up |

#### Scale Down Triggers
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_down_capacity_threshold` | 0.3 | Capacity utilization to trigger scale down |
| `scale_down_queue_depth_threshold` | 1.0 | Average queue depth to trigger scale down |
| `scale_down_idle_time_threshold` | 300.0 | Seconds of low utilization before scale down |

#### Timing Controls
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_time_between_scale_events` | 60.0 | Minimum seconds between scaling events |
| `scale_up_cooldown` | 30.0 | Cooldown after scale up |
| `scale_down_cooldown` | 120.0 | Cooldown after scale down |

#### Scaling Behavior
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_up_step_size` | 1 | How many replicas to add at once |
| `scale_down_step_size` | 1 | How many replicas to remove at once |

#### Safety Limits
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_queue_depth_emergency` | 20.0 | Emergency scale up threshold |
| `min_healthy_replicas_ratio` | 0.5 | Minimum ratio of healthy replicas |

## Metrics

### Service-Level Metrics
- **Total Sessions**: Number of active sessions
- **Healthy Replicas**: Number of operational replicas
- **Total Request Rate**: Requests per second across all replicas
- **Average Queue Depth**: Average pending requests per replica
- **Average Capacity Utilization**: Average resource usage across replicas
- **Sessions Per Replica**: Distribution of sessions across replicas

### Replica-Level Metrics
- **Request Counts**: Total, successful, and failed requests
- **Request Rate**: Requests per second (sliding window)
- **Average Latency**: Response time (sliding window)
- **Active Requests**: Currently processing requests
- **Queue Depth**: Pending requests in queue
- **Assigned Sessions**: Number of sessions assigned to replica
- **Capacity Utilization**: Current load vs maximum capacity

## Use Cases

### ML Model Serving
```python
# High-throughput model inference with automatic scaling
config = ServiceConfig(
    gpus_per_replica=1,
    min_replicas=2,
    max_replicas=20,
    default_replicas=4,
    replica_max_concurrent_requests=8,
    autoscaling=AutoscalingConfig(
        scale_up_capacity_threshold=0.7,
        scale_up_queue_depth_threshold=3.0
    )
)
service = Service(config, ModelInferenceActor, model_path="/path/to/model")
```

### Batch Processing
```python
# Parallel job execution with fault tolerance
config = ServiceConfig(
    gpus_per_replica=2,
    min_replicas=1,
    max_replicas=10,
    default_replicas=3,
    replica_max_concurrent_requests=5,
    autoscaling=AutoscalingConfig(
        scale_up_queue_depth_threshold=10.0,
        scale_down_idle_time_threshold=600.0
    )
)
service = Service(config, BatchProcessorActor, batch_size=100)
```

### Real-time Analytics
```python
# Stream processing with session affinity
config = ServiceConfig(
    gpus_per_replica=1,
    min_replicas=3,
    max_replicas=15,
    default_replicas=5,
    replica_max_concurrent_requests=20,
    autoscaling=AutoscalingConfig(
        scale_up_request_rate_threshold=50.0,
        scale_up_capacity_threshold=0.6
    )
)
service = Service(config, StreamProcessorActor, window_size=1000)
```

## Performance Characteristics

- **Low Latency**: Sub-millisecond request routing overhead
- **High Throughput**: Concurrent request processing across replicas
- **Elastic Scaling**: Responsive to traffic patterns with configurable thresholds
- **Resource Efficient**: Intelligent replica management and load balancing
- **Fault Resilient**: Automatic recovery from replica failures
- **Session Aware**: Maintains state consistency for stateful workloads

## Best Practices

### Configuration
- Set `min_replicas` based on baseline load requirements
- Configure `max_replicas` based on resource constraints
- Tune autoscaling thresholds based on workload characteristics
- Use appropriate cooldown periods to prevent scaling oscillation

### Session Management
- Use sessions for stateful workloads requiring consistency
- Prefer stateless calls for better load distribution
- Implement custom routing for specialized workload requirements

### Monitoring
- Monitor key metrics: request rate, queue depth, capacity utilization
- Set up alerts for unhealthy replicas and scaling events
- Track session distribution for load balancing effectiveness

### Error Handling
- Implement proper error handling in actor endpoints
- Use try-catch blocks around service calls
- Monitor failed request rates for service health

## Dependencies

- `monarch.actor`: Actor framework for distributed computing
- `recoverable_mesh`: Fault-tolerant process mesh management
- `asyncio`: Asynchronous I/O support
- `contextvars`: Context variable support for session management

## Thread Safety

The Service class is designed for use in asyncio environments and is not thread-safe. All operations should be performed within the same event loop.
